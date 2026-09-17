/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.LockFactory;
import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.common.settings.Settings;
import org.opensearch.index.IndexModule;
import org.opensearch.index.IndexSettings;
import org.opensearch.index.shard.ShardPath;
import org.opensearch.index.store.FsDirectoryFactory;
import org.opensearch.knn.common.featureflags.KNNFeatureFlags;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.plugins.IndexStorePlugin;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * A {@link IndexStorePlugin.DirectoryFactory} registered by the k-NN plugin under the
 * {@link #KNN_DIRECT_IO_STORE_TYPE} store type. An index opts in by setting
 * {@code index.store.type: knn_direct_io}, which is a static index setting and therefore takes
 * effect when the index is closed and reopened, with no node restart.
 * <p>
 * The directory it hands back is the very same {@link Directory} the node's default store type
 * would have produced, wrapped in {@link KNNDirectIODirectory} when Direct I/O is enabled. Only
 * {@code .vec} reads are routed through Direct I/O; every other file is served by the stock
 * directory exactly as it is today. When either gate is off the stock directory is returned
 * unwrapped, so the store type is then indistinguishable from the node default on every file.
 * <p>
 * The two gates are the index setting {@code index.knn.direct_io.enabled} and the node setting
 * {@code knn.feature.direct_io.enabled}. Both are read here, at shard open, so both take effect on
 * a close and reopen of the index with no node restart.
 * <p>
 * It reproduces the four steps of {@link FsDirectoryFactory#newDirectory(IndexSettings, ShardPath)}
 * rather than calling it, because that method re-reads {@code index.store.type} out of the
 * {@link IndexSettings} it is handed and rejects any value it does not own. The settings handed to
 * the delegate therefore have {@code index.store.type} removed, which makes the delegate resolve the
 * node's default store type — {@code hybridfs} normally, {@code niofs} when
 * {@code node.store.allow_mmap} is false.
 */
@Log4j2
public class KNNDirectIODirectoryFactory implements IndexStorePlugin.DirectoryFactory {

    /**
     * The {@code index.store.type} value that selects this factory. Must not collide with any
     * built-in store type; a collision fails node start.
     */
    public static final String KNN_DIRECT_IO_STORE_TYPE = "knn_direct_io";

    private final FsDirectoryFactory delegate = new FsDirectoryFactory();

    @Override
    public Directory newDirectory(final IndexSettings indexSettings, final ShardPath shardPath) throws IOException {
        final Path location = shardPath.resolveIndex();
        final LockFactory lockFactory = indexSettings.getValue(FsDirectoryFactory.INDEX_LOCK_FACTOR_SETTING);
        Files.createDirectories(location);
        return newFSDirectory(location, lockFactory, indexSettings);
    }

    @Override
    public Directory newFSDirectory(final Path location, final LockFactory lockFactory, final IndexSettings indexSettings)
        throws IOException {
        final IndexSettings delegateSettings = withDefaultStoreType(indexSettings);
        final Directory directory = delegate.newFSDirectory(location, lockFactory, delegateSettings);
        assert assertMatchesDefaultStoreType(directory, delegateSettings);
        return maybeWrapWithDirectIO(directory, location, indexSettings);
    }

    /**
     * Wraps the stock directory in {@link KNNDirectIODirectory} when both the index setting and the
     * node feature flag allow it, and returns it untouched otherwise.
     * <p>
     * The index setting is read off the {@link IndexSettings} the caller handed us rather than out
     * of cluster state, because that is the copy the shard was opened with. It is read through
     * {@code Setting.get(Settings)} rather than {@link IndexSettings#getValue}, because the latter
     * resolves through the node's registered {@code IndexScopedSettings} and throws for a setting
     * that registry does not know about — which is every plugin setting in a plain unit test, and
     * would turn a missing registration into a shard-open failure rather than a fallback.
     * <p>
     * Nothing this method does can fail a shard open. Every step that can throw — reading the
     * filesystem block size, and the {@link KNNDirectIODirectory} constructor, which reads it a
     * second time inside {@code DirectIODirectory} — is inside one {@code try}, and any failure logs
     * once at WARN and yields the stock directory. That is the whole point of the unit: on a
     * filesystem that cannot report a block size, {@code knn_direct_io} must be a slow index, not an
     * unopenable one.
     */
    private Directory maybeWrapWithDirectIO(final Directory directory, final Path location, final IndexSettings indexSettings) {
        if (KNNSettings.KNN_INDEX_DIRECT_IO_ENABLED_SETTING.get(indexSettings.getSettings()) == false) {
            log.debug(
                "Direct I/O is disabled for index [{}] by {}",
                indexSettings.getIndex().getName(),
                KNNSettings.KNN_INDEX_DIRECT_IO_ENABLED
            );
            return directory;
        }
        if (KNNFeatureFlags.isDirectIOEnabled() == false) {
            log.debug("Direct I/O is disabled on this node by {}", KNNFeatureFlags.KNN_DIRECT_IO_ENABLED_SETTING.getKey());
            return directory;
        }
        if (directory instanceof FSDirectory == false) {
            // DirectIODirectory casts its delegate to FSDirectory and resolves paths through it, so a
            // delegate that is not one cannot be wrapped. The node default store types all are, but
            // this keeps a future default from failing shard open.
            log.warn(
                "Not using Direct I/O for index [{}]: the store directory [{}] is not an FSDirectory",
                indexSettings.getIndex().getName(),
                directory.getClass().getName()
            );
            return directory;
        }
        if (KNNDirectIODirectory.isDirectIOOpenOptionAvailable() == false) {
            // Knowable without touching the filesystem, so it is worth checking before the block size
            // read below: on such a runtime every single openInput would fail and fall back anyway.
            log.warn(
                "Not using Direct I/O for index [{}]: this JDK does not expose "
                    + "com.sun.nio.file.ExtendedOpenOption.DIRECT. Reads will use the store directory [{}].",
                indexSettings.getIndex().getName(),
                directory.getClass().getSimpleName()
            );
            return directory;
        }

        try {
            final int blockSize = blockSize(location);
            // Derived per index from the vector dimensions the mapping declares; see DirectIOBufferSizer
            // for why no flat constant works. The block size is read here rather than there so that the
            // sizer does no filesystem I/O of its own.
            final int readBufferSize = DirectIOBufferSizer.readBufferSize(indexSettings, blockSize);
            final long minBytesDirect = KNNSettings.getDirectIOMinFileSize().getBytes();

            final Directory wrapped = new KNNDirectIODirectory((FSDirectory) directory, readBufferSize, minBytesDirect);
            // Phase 5 reads these three numbers back out of the log to confirm which arm actually ran,
            // so keep the store type, the buffer size and the block size on one line.
            log.info(
                "Using store type [{}] for index [{}]: Direct I/O for {} reads above {} bytes, "
                    + "with a {} byte read buffer over a {} byte filesystem block",
                KNN_DIRECT_IO_STORE_TYPE,
                indexSettings.getIndex().getName(),
                KNNDirectIODirectory.VECTOR_DATA_SUFFIX,
                minBytesDirect,
                readBufferSize,
                blockSize
            );
            return wrapped;
        } catch (IOException | RuntimeException e) {
            // UnsupportedOperationException (the block size attribute is optional and some filesystems
            // do not implement it) and ArithmeticException (a block size that does not fit an int) both
            // arrive as RuntimeException. Error is left to propagate.
            log.warn(
                "Not using Direct I/O for index [{}]: could not set it up over [{}]. Reads will use the store directory [{}].",
                indexSettings.getIndex().getName(),
                location,
                directory.getClass().getSimpleName(),
                e
            );
            return directory;
        }
    }

    /**
     * The filesystem block size at the given location, which is the alignment every Direct I/O read
     * has to respect.
     * <p>
     * Package private and overridable purely so a test can make it throw: this is the one step of
     * shard open that does filesystem I/O of its own, and its fallback is the behaviour unit 13
     * exists to guarantee. There is no other way to inject a failing {@code getFileStore} without a
     * filesystem that has one.
     *
     * @param location the shard's index directory
     * @return the block size in bytes
     * @throws IOException if the file store cannot be resolved
     * @throws UnsupportedOperationException if the file store does not report a block size
     */
    int blockSize(final Path location) throws IOException {
        return Math.toIntExact(Files.getFileStore(location).getBlockSize());
    }

    /**
     * Returns a copy of the given settings with {@code index.store.type} removed, so that
     * {@link FsDirectoryFactory} resolves the node's default store type instead of tripping over
     * {@link #KNN_DIRECT_IO_STORE_TYPE}.
     * <p>
     * The store type lives in the index metadata's settings, and {@link IndexSettings} exposes no
     * setter for it, so the copy is made by rebuilding the {@link IndexMetadata}. Every other key
     * {@link FsDirectoryFactory} reads — {@code index.store.pre_load},
     * {@code index.store.hybrid.nio.extensions} and the node-level {@code node.store.allow_mmap} —
     * is carried through untouched.
     */
    static IndexSettings withDefaultStoreType(final IndexSettings indexSettings) {
        final IndexMetadata indexMetadata = indexSettings.getIndexMetadata();
        final Settings.Builder patched = Settings.builder().put(indexMetadata.getSettings());
        patched.remove(IndexModule.INDEX_STORE_TYPE_SETTING.getKey());
        final IndexMetadata patchedMetadata = IndexMetadata.builder(indexMetadata).settings(patched.build()).build();
        return new IndexSettings(patchedMetadata, indexSettings.getNodeSettings());
    }

    /**
     * Assertion-only sanity check that the delegate really produced the node's default store type.
     * Runs under {@code -ea} only; a false negative must never fail shard open.
     */
    private static boolean assertMatchesDefaultStoreType(final Directory directory, final IndexSettings delegateSettings) {
        final String storeType = delegateSettings.getSettings().get(IndexModule.INDEX_STORE_TYPE_SETTING.getKey());
        assert storeType == null : "index.store.type should have been removed before delegating, but was [" + storeType + "]";

        final boolean allowMmap = IndexModule.NODE_STORE_ALLOW_MMAP.get(delegateSettings.getNodeSettings());
        final IndexModule.Type expected = IndexModule.defaultStoreType(allowMmap);
        assert expected != IndexModule.Type.HYBRIDFS || FsDirectoryFactory.isHybridFs(directory) : "expected a hybridfs directory but got ["
            + directory.getClass().getName()
            + "]";
        return true;
    }
}

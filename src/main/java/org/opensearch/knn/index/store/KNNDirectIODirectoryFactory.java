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
     */
    private static Directory maybeWrapWithDirectIO(final Directory directory, final Path location, final IndexSettings indexSettings)
        throws IOException {
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

        final int blockSize = Math.toIntExact(Files.getFileStore(location).getBlockSize());
        // Derived per index from the vector dimensions the mapping declares; see DirectIOBufferSizer
        // for why no flat constant works. The block size is read here rather than there so that the
        // sizer does no filesystem I/O of its own.
        final int readBufferSize = DirectIOBufferSizer.readBufferSize(indexSettings, blockSize);
        final long minBytesDirect = KNNSettings.getDirectIOMinFileSize().getBytes();

        log.info(
            "Using Direct I/O for .vec reads on index [{}] with a {} byte read buffer (block size {}) above {} bytes",
            indexSettings.getIndex().getName(),
            readBufferSize,
            blockSize,
            minBytesDirect
        );
        return new KNNDirectIODirectory((FSDirectory) directory, readBufferSize, minBytesDirect);
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

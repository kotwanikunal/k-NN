/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.LockFactory;
import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.common.settings.Settings;
import org.opensearch.index.IndexModule;
import org.opensearch.index.IndexSettings;
import org.opensearch.index.shard.ShardPath;
import org.opensearch.index.store.FsDirectoryFactory;
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
 * This class deliberately contains no Direct I/O of its own. Its only job is to hand back the very
 * same {@link Directory} the node's default store type would have produced, so that the store type
 * is behaviour-neutral on every file. Direct I/O is layered on top of this in a later change; when
 * the feature is disabled this factory must remain indistinguishable from the default.
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
        return directory;
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

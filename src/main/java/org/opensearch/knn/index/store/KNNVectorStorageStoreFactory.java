/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.store.Directory;
import org.opensearch.core.index.shard.ShardId;
import org.opensearch.env.ShardLock;
import org.opensearch.index.IndexSettings;
import org.opensearch.index.shard.ShardPath;
import org.opensearch.index.store.Store;
import org.opensearch.plugins.IndexStorePlugin;

import java.io.IOException;

/**
 * The insertion point of the directory design: an {@link IndexStorePlugin.StoreFactory} that puts a
 * {@link KNNVectorStorageDirectory} between the store type's directory and the {@link Store} wrappers
 * above it. An index opts in with {@code index.store.factory: knn_vector_storage}.
 *
 * <h2>Why this key and not {@code index.store.type}</h2>
 * {@code getDirectoryFactories()} is keyed by {@code index.store.type}, so taking it means
 * <em>replacing</em> the operator's store type — which is what the earlier store-type attempt had to do,
 * and its fatal objection: an index cannot be both {@code hybridfs} and the plugin's.
 * {@code getStoreFactories()} is keyed by a separate setting and is handed the directory the chosen
 * store type <em>already built</em>, so wrapping there composes with {@code hybridfs}, {@code niofs},
 * {@code mmapfs} and the remote-store directories instead of displacing them. {@code Store}'s own
 * constructor then puts {@code Store$StoreDirectory(ByteSizeCachingDirectory(ours))} above it, so the
 * wrapper sits at the same depth a store-type directory would have, with none of the exclusivity.
 *
 * <p>The cost of that position is a different exclusivity: one {@code StoreFactory} per index, so this
 * cannot coexist with another plugin's. Recorded rather than solved.
 *
 * <h2>What opting in does and does not turn on</h2>
 * Installing the directory is necessary but not sufficient. A read is only served with {@code O_DIRECT}
 * when the file is a faiss or MOS full-precision flat vector file
 * ({@link KNNVectorStorageDirectory#isFaissVectorData}) and {@code knn.direct_io.rescore.enabled} is on
 * — off by default, because Direct I/O is a win only where the {@code .vec} working set cannot stay in
 * the page cache. So an index that opts in on a node with the setting off reads exactly as it did
 * before, one virtual call per compound-container slice aside.
 */
@Log4j2
public class KNNVectorStorageStoreFactory implements IndexStorePlugin.StoreFactory {

    /**
     * The {@code index.store.factory} value that selects this factory. A value the node has no factory
     * for fails shard open with {@code IllegalArgumentException}, so the name is part of the contract.
     */
    public static final String KNN_VECTOR_STORAGE_STORE_FACTORY = "knn_vector_storage";

    @Override
    public Store newStore(
        final ShardId shardId,
        final IndexSettings indexSettings,
        final Directory directory,
        final ShardLock shardLock,
        final Store.OnClose onClose,
        final ShardPath shardPath
    ) throws IOException {
        return new Store(shardId, indexSettings, wrap(directory, indexSettings), shardLock, onClose, shardPath);
    }

    @Override
    public Store newStore(
        final ShardId shardId,
        final IndexSettings indexSettings,
        final Directory directory,
        final ShardLock shardLock,
        final Store.OnClose onClose,
        final ShardPath shardPath,
        final IndexStorePlugin.DirectoryFactory directoryFactory
    ) throws IOException {
        return new Store(shardId, indexSettings, wrap(directory, indexSettings), shardLock, onClose, shardPath, directoryFactory);
    }

    /**
     * Wraps the store type's directory. Nothing here can fail a shard open: the wrapper has no state to
     * initialise and touches no filesystem, and a failure would still be better reported than swallowed,
     * since the operator asked for this factory by name.
     */
    private static Directory wrap(final Directory directory, final IndexSettings indexSettings) {
        final String indexName = indexSettings.getIndex().getName();
        log.info(
            "k-NN vector storage directory installed for index [{}] over store directory [{}]",
            indexName,
            directory.getClass().getSimpleName()
        );
        return new KNNVectorStorageDirectory(directory, indexName);
    }
}

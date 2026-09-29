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
 * An {@link IndexStorePlugin.StoreFactory} that inserts a {@link KNNVectorIntentProbeDirectory}
 * between the store type's directory and the {@link Store} wrappers above it. An index opts in with
 * {@code index.store.factory: knn_intent_probe}.
 *
 * <p>The insertion point matters more than the probe does. {@code IndexStorePlugin} offers a plugin
 * exactly one way to supply a {@link Directory} by name — {@code getDirectoryFactories()}, keyed by
 * {@code index.store.type} — and taking that key means <em>replacing</em> the user's store type,
 * which is what the earlier store-type attempt had to do. {@code getStoreFactories()} is keyed by a
 * separate setting, {@code index.store.factory}, and is handed the directory the chosen store type
 * already built. Wrapping it there composes with {@code hybridfs}, {@code niofs}, {@code mmapfs} and
 * the remote-store directories instead of displacing them, and {@code Store}'s own constructor then
 * puts {@code Store$StoreDirectory(ByteSizeCachingDirectory(ours))} above it — so the wrapper is at
 * the same depth a store-type directory would have been, with none of the exclusivity.
 *
 * <p>The cost of that position is a different exclusivity: one {@code StoreFactory} per index, so
 * this cannot coexist with another plugin's. It is recorded rather than solved.
 */
@Log4j2
public class KNNVectorIntentProbeStoreFactory implements IndexStorePlugin.StoreFactory {

    /**
     * The {@code index.store.factory} value that selects this factory. A value the node has no
     * factory for fails shard open with {@code IllegalArgumentException}, so the name is part of the
     * contract.
     */
    public static final String KNN_INTENT_PROBE_STORE_FACTORY = "knn_intent_probe";

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
     * Wraps the store type's directory in the probe. Nothing here can fail a shard open: the wrapper
     * has no state to initialise and touches no filesystem, and a failure would still be better
     * reported than swallowed, since the operator asked for this factory by name.
     */
    private static Directory wrap(final Directory directory, final IndexSettings indexSettings) {
        final KNNVectorIntentProbeDirectory probe = new KNNVectorIntentProbeDirectory(directory, indexSettings.getIndex().getName());
        log.info(
            "k-NN intent probe installed for index [{}] over store directory [{}]",
            indexSettings.getIndex().getName(),
            directory.getClass().getSimpleName()
        );
        return probe;
    }
}

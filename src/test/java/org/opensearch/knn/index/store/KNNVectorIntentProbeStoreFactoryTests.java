/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.opensearch.Version;
import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.common.settings.Settings;
import org.opensearch.core.index.shard.ShardId;
import org.opensearch.env.ShardLock;
import org.opensearch.index.IndexSettings;
import org.opensearch.index.store.Store;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.plugin.KNNPlugin;
import org.opensearch.plugins.IndexStorePlugin;

import java.io.IOException;
import java.util.List;
import java.util.Map;

import static org.opensearch.knn.index.store.KNNVectorIntentProbeStoreFactory.KNN_INTENT_PROBE_STORE_FACTORY;

/**
 * The insertion point, not the probe. A plugin gets to supply a {@link Directory} by
 * {@code index.store.type}, which replaces the user's store type, or to wrap the one the store type
 * built via {@code index.store.factory}, which composes with it. These tests pin the second: the
 * factory must put the plugin's directory directly above whatever it was handed and directly below
 * the wrappers {@link Store} adds.
 */
public class KNNVectorIntentProbeStoreFactoryTests extends KNNTestCase {

    private static ShardLock noopShardLock(final ShardId shardId) {
        return new ShardLock(shardId) {
            @Override
            protected void closeInternal() {}
        };
    }

    private static IndexSettings indexSettings() {
        final Settings settings = Settings.builder()
            .put(IndexMetadata.SETTING_VERSION_CREATED, Version.CURRENT)
            .put(IndexMetadata.SETTING_NUMBER_OF_SHARDS, 1)
            .put(IndexMetadata.SETTING_NUMBER_OF_REPLICAS, 0)
            .build();
        return new IndexSettings(IndexMetadata.builder("test-index").settings(settings).build(), Settings.EMPTY);
    }

    public void testFactoryWrapsTheStoreTypeDirectoryRatherThanReplacingIt() throws IOException {
        final ShardId shardId = new ShardId("test-index", "_na_", 0);
        final Directory storeTypeDirectory = new ByteBuffersDirectory();

        try (
            Store store = new KNNVectorIntentProbeStoreFactory().newStore(
                shardId,
                indexSettings(),
                storeTypeDirectory,
                noopShardLock(shardId),
                Store.OnClose.EMPTY,
                null
            )
        ) {
            final KNNVectorIntentProbeDirectory probe = KNNVectorIntentProbeDirectory.find(store.directory());
            assertNotNull("the factory must install the probe", probe);
            assertSame("the store type's directory must be preserved underneath, not replaced", storeTypeDirectory, probe.getDelegate());
            assertEquals("the probe must be attributable to an index", "test-index", probe.indexName());
            assertTrue(KNNVectorIntentProbeDirectory.isInstalledOnNode());
            assertEquals(
                "chain was " + KNNVectorIntentProbeDirectory.wrapperChain(store.directory()),
                List.of("StoreDirectory", "ByteSizeCachingDirectory", "KNNVectorIntentProbeDirectory", "ByteBuffersDirectory"),
                KNNVectorIntentProbeDirectory.wrapperChain(store.directory())
            );
        }
    }

    public void testPluginRegistersTheFactoryUnderItsDocumentedName() {
        final Map<String, IndexStorePlugin.StoreFactory> factories = new KNNPlugin().getStoreFactories();
        assertEquals(1, factories.size());
        assertTrue(factories.containsKey(KNN_INTENT_PROBE_STORE_FACTORY));
        assertTrue(factories.get(KNN_INTENT_PROBE_STORE_FACTORY) instanceof KNNVectorIntentProbeStoreFactory);
    }
}

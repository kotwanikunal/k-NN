/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FilterDirectory;
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
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.opensearch.knn.index.store.KNNVectorStorageStoreFactory.KNN_VECTOR_STORAGE_STORE_FACTORY;

/**
 * The insertion point. A plugin gets to supply a {@link Directory} by {@code index.store.type}, which
 * replaces the user's store type, or to wrap the one the store type built via
 * {@code index.store.factory}, which composes with it. These tests pin the second: the factory must put
 * the plugin's directory directly above whatever it was handed and directly below the wrappers
 * {@link Store} adds.
 */
public class KNNVectorStorageStoreFactoryTests extends KNNTestCase {

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

    /**
     * The chain from {@code directory} down, as simple class names. {@link FilterDirectory#unwrap} answers
     * only what is at the bottom; this test needs what is in between, because any wrapper that rebuilt the
     * {@code IOContext} rather than passing the caller's through would break the intent signal.
     */
    private static List<String> chainOf(final Directory directory) {
        final List<String> chain = new ArrayList<>();
        Directory current = directory;
        for (int depth = 0; current != null && depth < 16; depth++) {
            chain.add(current.getClass().getSimpleName());
            if (current instanceof FilterDirectory filterDirectory) {
                current = filterDirectory.getDelegate();
            } else {
                break;
            }
        }
        return List.copyOf(chain);
    }

    public void testFactoryWrapsTheStoreTypeDirectoryRatherThanReplacingIt() throws IOException {
        final ShardId shardId = new ShardId("test-index", "_na_", 0);
        final Directory storeTypeDirectory = new ByteBuffersDirectory();

        try (
            Store store = new KNNVectorStorageStoreFactory().newStore(
                shardId,
                indexSettings(),
                storeTypeDirectory,
                noopShardLock(shardId),
                Store.OnClose.EMPTY,
                null
            )
        ) {
            final KNNVectorStorageDirectory storage = KNNVectorStorageDirectory.find(store.directory());
            assertNotNull("the factory must install the vector storage directory", storage);
            assertSame("the store type's directory must be preserved underneath, not replaced", storeTypeDirectory, storage.getDelegate());
            assertEquals("the directory must be attributable to an index", "test-index", storage.indexName());
            assertEquals(
                "chain was " + chainOf(store.directory()),
                List.of("StoreDirectory", "ByteSizeCachingDirectory", "KNNVectorStorageDirectory", "ByteBuffersDirectory"),
                chainOf(store.directory())
            );
        }
    }

    /**
     * The factory name is part of the contract — an {@code index.store.factory} the node has no factory
     * for fails shard open — so it is pinned here.
     */
    public void testPluginRegistersTheFactoryUnderItsDocumentedName() {
        final Map<String, IndexStorePlugin.StoreFactory> factories = new KNNPlugin().getStoreFactories();
        assertEquals("the plugin offers exactly the production storage directory", 1, factories.size());
        assertTrue(factories.containsKey(KNN_VECTOR_STORAGE_STORE_FACTORY));
        assertTrue(factories.get(KNN_VECTOR_STORAGE_STORE_FACTORY) instanceof KNNVectorStorageStoreFactory);
    }
}

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.store.NIOFSDirectory;
import org.opensearch.Version;
import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.common.settings.Settings;
import org.opensearch.core.index.shard.ShardId;
import org.opensearch.index.IndexModule;
import org.opensearch.index.IndexSettings;
import org.opensearch.index.shard.ShardPath;
import org.opensearch.index.store.FsDirectoryFactory;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.plugin.KNNPlugin;
import org.opensearch.plugins.IndexStorePlugin;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.opensearch.knn.index.store.KNNDirectIODirectoryFactory.KNN_DIRECT_IO_STORE_TYPE;

/**
 * Unit 10's job is the equivalence proof: selecting {@code index.store.type: knn_direct_io} must hand
 * back exactly the directory the node's default store type would have produced. These tests compare
 * our factory against the stock {@link FsDirectoryFactory} on the same settings and shard path.
 */
public class KNNDirectIODirectoryFactoryTests extends KNNTestCase {

    private static final String TEST_INDEX = "test-index";

    private static IndexSettings indexSettings(final Settings extraIndexSettings, final Settings nodeSettings) {
        final Settings settings = Settings.builder()
            .put(IndexMetadata.SETTING_VERSION_CREATED, Version.CURRENT)
            .put(IndexMetadata.SETTING_NUMBER_OF_SHARDS, 1)
            .put(IndexMetadata.SETTING_NUMBER_OF_REPLICAS, 0)
            .put(extraIndexSettings)
            .build();
        final IndexMetadata indexMetadata = IndexMetadata.builder(TEST_INDEX).settings(settings).build();
        return new IndexSettings(indexMetadata, nodeSettings);
    }

    private static ShardPath shardPath(final Path root, final IndexSettings indexSettings) {
        // ShardPath asserts the data path is <something>/<indexUUID>/<shardId>.
        final Path shardDir = root.resolve(indexSettings.getIndex().getUUID()).resolve("0");
        return new ShardPath(false, shardDir, shardDir, new ShardId(indexSettings.getIndex(), 0));
    }

    /**
     * The store type key must not collide with any built-in one; {@code Node} collide-checks plugin
     * store types against the built-in set and a collision fails node start outright.
     */
    public void testStoreTypeDoesNotCollideWithBuiltInStoreTypes() {
        for (IndexModule.Type type : IndexModule.Type.values()) {
            assertNotEquals("knn_direct_io collides with the built-in store type " + type, type.getSettingsKey(), KNN_DIRECT_IO_STORE_TYPE);
        }
        assertEquals("knn_direct_io", KNN_DIRECT_IO_STORE_TYPE);
    }

    /**
     * The plugin must actually register the store type, unconditionally.
     */
    public void testPluginRegistersTheStoreType() {
        final Map<String, IndexStorePlugin.DirectoryFactory> factories = new KNNPlugin().getDirectoryFactories();
        assertEquals(1, factories.size());
        assertTrue(factories.containsKey(KNN_DIRECT_IO_STORE_TYPE));
        assertTrue(factories.get(KNN_DIRECT_IO_STORE_TYPE) instanceof KNNDirectIODirectoryFactory);
    }

    /**
     * Owed assertion 1: same class as the stock factory, and hybridfs for both.
     */
    public void testReturnsSameDirectoryClassAsStockFactory() throws IOException {
        final Path root = createTempDir();
        final Settings nodeSettings = Settings.EMPTY;

        final IndexSettings ourSettings = indexSettings(
            Settings.builder().put(IndexModule.INDEX_STORE_TYPE_SETTING.getKey(), KNN_DIRECT_IO_STORE_TYPE).build(),
            nodeSettings
        );
        final IndexSettings hybridSettings = indexSettings(
            Settings.builder().put(IndexModule.INDEX_STORE_TYPE_SETTING.getKey(), IndexModule.Type.HYBRIDFS.getSettingsKey()).build(),
            nodeSettings
        );
        final IndexSettings unsetSettings = indexSettings(Settings.EMPTY, nodeSettings);

        try (
            Directory ours = new KNNDirectIODirectoryFactory().newDirectory(ourSettings, shardPath(root, ourSettings));
            Directory stockHybrid = new FsDirectoryFactory().newDirectory(hybridSettings, shardPath(root, hybridSettings));
            Directory stockDefault = new FsDirectoryFactory().newDirectory(unsetSettings, shardPath(root, unsetSettings))
        ) {
            assertEquals(stockHybrid.getClass(), ours.getClass());
            assertEquals(stockDefault.getClass(), ours.getClass());
            assertTrue(FsDirectoryFactory.isHybridFs(ours));
            assertTrue(FsDirectoryFactory.isHybridFs(stockHybrid));
        }
    }

    /**
     * Owed assertion 2a: {@code index.store.hybrid.nio.extensions} survives the doctoring, proven
     * behaviourally — a listed extension is served by {@link NIOFSDirectory} and an unlisted one is
     * not, identically for both factories.
     */
    public void testHybridNioExtensionsSurviveDoctoring() throws IOException {
        final Path root = createTempDir();
        final Settings extensions = Settings.builder()
            .putList(IndexModule.INDEX_STORE_HYBRID_NIO_EXTENSIONS.getKey(), List.of("nio"))
            .build();

        final IndexSettings ourSettings = indexSettings(
            Settings.builder().put(extensions).put(IndexModule.INDEX_STORE_TYPE_SETTING.getKey(), KNN_DIRECT_IO_STORE_TYPE).build(),
            Settings.EMPTY
        );
        final IndexSettings stockSettings = indexSettings(extensions, Settings.EMPTY);

        assertEquals(
            List.of("nio"),
            KNNDirectIODirectoryFactory.withDefaultStoreType(ourSettings).getValue(IndexModule.INDEX_STORE_HYBRID_NIO_EXTENSIONS)
        );

        try (
            Directory ours = new KNNDirectIODirectoryFactory().newDirectory(ourSettings, shardPath(root, ourSettings));
            Directory stock = new FsDirectoryFactory().newDirectory(stockSettings, shardPath(root, stockSettings))
        ) {
            writeFile(ours, "segment.nio");
            writeFile(ours, "segment.mmap");

            assertSameInputClass(ours, stock, "segment.nio");
            assertSameInputClass(ours, stock, "segment.mmap");

            try (IndexInput nio = ours.openInput("segment.nio", IOContext.DEFAULT)) {
                assertTrue(
                    "a listed extension should be served by NIOFSDirectory, got " + nio.getClass().getName(),
                    nio.getClass().getName().contains("NIOFS")
                );
            }
            try (IndexInput mmap = ours.openInput("segment.mmap", IOContext.DEFAULT)) {
                assertFalse(
                    "an unlisted extension should not be served by NIOFSDirectory, got " + mmap.getClass().getName(),
                    mmap.getClass().getName().contains("NIOFS")
                );
            }
        }
    }

    /**
     * Owed assertion 2b: {@code index.store.pre_load} survives the doctoring. It is not observable
     * from the returned directory, so it is asserted on the doctored settings directly.
     */
    public void testPreLoadSurvivesDoctoring() {
        final IndexSettings ourSettings = indexSettings(
            Settings.builder()
                .putList(IndexModule.INDEX_STORE_PRE_LOAD_SETTING.getKey(), List.of("vec", "vex"))
                .put(IndexModule.INDEX_STORE_TYPE_SETTING.getKey(), KNN_DIRECT_IO_STORE_TYPE)
                .build(),
            Settings.EMPTY
        );

        final IndexSettings doctored = KNNDirectIODirectoryFactory.withDefaultStoreType(ourSettings);
        assertEquals(List.of("vec", "vex"), doctored.getValue(IndexModule.INDEX_STORE_PRE_LOAD_SETTING));
        assertNull(doctored.getSettings().get(IndexModule.INDEX_STORE_TYPE_SETTING.getKey()));
        // everything except the store type is untouched
        assertEquals(ourSettings.getIndex(), doctored.getIndex());
        assertEquals(ourSettings.getNumberOfShards(), doctored.getNumberOfShards());
    }

    /**
     * Owed assertion 2c: with {@code node.store.allow_mmap = false} both factories return the same
     * non-hybrid class. This is why the doctoring removes {@code index.store.type} rather than
     * rewriting it to {@code hybridfs} — a hard-coded {@code hybridfs} would mmap anyway and override
     * the operator's choice.
     */
    public void testAllowMmapFalseIsHonoured() throws IOException {
        final Path root = createTempDir();
        final Settings nodeSettings = Settings.builder().put(IndexModule.NODE_STORE_ALLOW_MMAP.getKey(), false).build();

        final IndexSettings ourSettings = indexSettings(
            Settings.builder().put(IndexModule.INDEX_STORE_TYPE_SETTING.getKey(), KNN_DIRECT_IO_STORE_TYPE).build(),
            nodeSettings
        );
        final IndexSettings stockSettings = indexSettings(Settings.EMPTY, nodeSettings);

        try (
            Directory ours = new KNNDirectIODirectoryFactory().newDirectory(ourSettings, shardPath(root, ourSettings));
            Directory stock = new FsDirectoryFactory().newDirectory(stockSettings, shardPath(root, stockSettings))
        ) {
            assertEquals(stock.getClass(), ours.getClass());
            assertEquals(NIOFSDirectory.class, ours.getClass());
            assertFalse(FsDirectoryFactory.isHybridFs(ours));
        }
    }

    /**
     * {@code newDirectory} must create the shard's index directory; omitting the
     * {@code Files.createDirectories} step breaks shard open.
     */
    public void testNewDirectoryCreatesTheIndexDirectory() throws IOException {
        final Path root = createTempDir().resolve("not-yet-created");
        final IndexSettings ourSettings = indexSettings(
            Settings.builder().put(IndexModule.INDEX_STORE_TYPE_SETTING.getKey(), KNN_DIRECT_IO_STORE_TYPE).build(),
            Settings.EMPTY
        );
        final ShardPath path = shardPath(root, ourSettings);
        assertFalse(java.nio.file.Files.exists(path.resolveIndex()));
        try (Directory ours = new KNNDirectIODirectoryFactory().newDirectory(ourSettings, path)) {
            assertNotNull(ours);
            assertTrue(java.nio.file.Files.isDirectory(path.resolveIndex()));
        }
    }

    private static void writeFile(final Directory directory, final String name) throws IOException {
        try (IndexOutput out = directory.createOutput(name, IOContext.DEFAULT)) {
            out.writeBytes(new byte[64], 64);
        }
    }

    private static void assertSameInputClass(final Directory ours, final Directory stock, final String name) throws IOException {
        try (IndexInput a = ours.openInput(name, IOContext.DEFAULT); IndexInput b = stock.openInput(name, IOContext.DEFAULT)) {
            assertEquals("input class differs for " + name, b.getClass(), a.getClass());
        }
    }
}

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import java.nio.charset.StandardCharsets;

import com.sun.management.HotSpotDiagnosticMXBean;
import lombok.SneakyThrows;
import org.opensearch.action.admin.cluster.state.ClusterStateRequest;
import org.opensearch.action.admin.indices.create.CreateIndexRequest;
import org.opensearch.action.admin.indices.settings.put.UpdateSettingsRequest;
import org.opensearch.cluster.ClusterName;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.network.NetworkModule;
import org.opensearch.common.settings.Setting;
import org.opensearch.common.settings.Settings;
import org.opensearch.core.common.unit.ByteSizeValue;
import org.opensearch.env.Environment;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.plugin.KNNPlugin;
import org.opensearch.node.MockNode;
import org.opensearch.node.Node;
import org.opensearch.Version;
import org.opensearch.plugins.Plugin;
import org.opensearch.plugins.PluginInfo;
import org.opensearch.test.InternalTestCluster;
import org.opensearch.test.MockHttpTransport;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.management.ManagementFactory;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Collection;
import java.util.EnumSet;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import static org.opensearch.test.NodeRoles.dataNode;

public class KNNSettingsTests extends KNNTestCase {

    private static final String INDEX_NAME = "myindex";

    @SneakyThrows
    public void testGetSettingValueFromConfig() {
        long expectedKNNCircuitBreakerLimit = 13;
        Node mockNode = createMockNode(
            Map.of(KNNSettings.KNN_MEMORY_CIRCUIT_BREAKER_CLUSTER_LIMIT, "\"" + expectedKNNCircuitBreakerLimit + "kb\"")
        );
        mockNode.start();
        ClusterService clusterService = mockNode.injector().getInstance(ClusterService.class);
        KNNSettings.state().setClusterService(clusterService);
        long actualKNNCircuitBreakerLimit = ((ByteSizeValue) KNNSettings.state()
            .getSettingValue(KNNSettings.KNN_MEMORY_CIRCUIT_BREAKER_CLUSTER_LIMIT)).getKb();
        mockNode.close();
        assertEquals(expectedKNNCircuitBreakerLimit, actualKNNCircuitBreakerLimit);
    }

    @SneakyThrows
    public void testGetSettingValueDefault() {
        Node mockNode = createMockNode(Collections.emptyMap());
        mockNode.start();
        ClusterService clusterService = mockNode.injector().getInstance(ClusterService.class);
        KNNSettings.state().setClusterService(clusterService);
        long actualKNNCircuitBreakerLimit = ((ByteSizeValue) KNNSettings.state()
            .getSettingValue(KNNSettings.KNN_MEMORY_CIRCUIT_BREAKER_CLUSTER_LIMIT)).getKb();
        mockNode.close();
        assertEquals(
            ((ByteSizeValue) KNNSettings.dynamicCacheSettings.get(KNNSettings.KNN_MEMORY_CIRCUIT_BREAKER_CLUSTER_LIMIT)
                .getDefault(Settings.EMPTY)).getKb(),
            actualKNNCircuitBreakerLimit

        );
    }

    @SneakyThrows
    public void testFilteredSearchAdvanceSetting_whenNoValuesProvidedByUsers_thenDefaultSettingsUsed() {
        Node mockNode = createMockNode(Collections.emptyMap());
        mockNode.start();
        ClusterService clusterService = mockNode.injector().getInstance(ClusterService.class);
        mockNode.client().admin().cluster().state(new ClusterStateRequest()).actionGet();
        mockNode.client().admin().indices().create(new CreateIndexRequest(INDEX_NAME)).actionGet();
        KNNSettings.state().setClusterService(clusterService);

        Integer filteredSearchThreshold = KNNSettings.getFilteredExactSearchThreshold(INDEX_NAME);
        mockNode.close();
        assertEquals(KNNSettings.ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD_DEFAULT_VALUE, filteredSearchThreshold);
    }

    @SneakyThrows
    public void testFilteredSearchAdvanceSetting_whenValuesProvidedByUsers_thenValidateSameValues() {
        int userDefinedThreshold = 1000;
        int userDefinedThresholdMinValue = 0;
        Node mockNode = createMockNode(Collections.emptyMap());
        mockNode.start();
        ClusterService clusterService = mockNode.injector().getInstance(ClusterService.class);
        mockNode.client().admin().cluster().state(new ClusterStateRequest()).actionGet();
        mockNode.client().admin().indices().create(new CreateIndexRequest(INDEX_NAME)).actionGet();
        KNNSettings.state().setClusterService(clusterService);

        final Settings filteredSearchAdvanceSettings = Settings.builder()
            .put(KNNSettings.ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD, userDefinedThreshold)
            .build();

        mockNode.client()
            .admin()
            .indices()
            .updateSettings(new UpdateSettingsRequest(filteredSearchAdvanceSettings, INDEX_NAME))
            .actionGet();

        int filteredSearchThreshold = KNNSettings.getFilteredExactSearchThreshold(INDEX_NAME);

        // validate if we are able to set MinValues for the setting
        final Settings filteredSearchAdvanceSettingsWithMinValues = Settings.builder()
            .put(KNNSettings.ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD, userDefinedThresholdMinValue)
            .build();

        mockNode.client()
            .admin()
            .indices()
            .updateSettings(new UpdateSettingsRequest(filteredSearchAdvanceSettingsWithMinValues, INDEX_NAME))
            .actionGet();

        int filteredSearchThresholdMinValue = KNNSettings.getFilteredExactSearchThreshold(INDEX_NAME);

        mockNode.close();
        assertEquals(userDefinedThreshold, filteredSearchThreshold);
        assertEquals(userDefinedThresholdMinValue, filteredSearchThresholdMinValue);
    }

    @SneakyThrows
    public void testGetEfSearch_whenNoValuesProvidedByUsers_thenDefaultSettingsUsed() {
        Node mockNode = createMockNode(Collections.emptyMap());
        mockNode.start();
        ClusterService clusterService = mockNode.injector().getInstance(ClusterService.class);
        mockNode.client().admin().cluster().state(new ClusterStateRequest()).actionGet();
        mockNode.client().admin().indices().create(new CreateIndexRequest(INDEX_NAME)).actionGet();
        KNNSettings.state().setClusterService(clusterService);

        Integer efSearchValue = KNNSettings.getEfSearchParam(INDEX_NAME);
        mockNode.close();
        assertEquals(KNNSettings.INDEX_KNN_DEFAULT_ALGO_PARAM_EF_SEARCH, efSearchValue);
    }

    @SneakyThrows
    public void testGetEfSearch_whenEFSearchValueSetByUser_thenReturnValue() {
        int userProvidedEfSearch = 300;
        Node mockNode = createMockNode(Collections.emptyMap());
        mockNode.start();
        ClusterService clusterService = mockNode.injector().getInstance(ClusterService.class);
        mockNode.client().admin().cluster().state(new ClusterStateRequest()).actionGet();
        final Settings settings = Settings.builder()
            .put(KNNSettings.KNN_ALGO_PARAM_EF_SEARCH, userProvidedEfSearch)
            .put(KNNSettings.KNN_INDEX, true)
            .build();
        mockNode.client().admin().indices().create(new CreateIndexRequest(INDEX_NAME, settings)).actionGet();
        KNNSettings.state().setClusterService(clusterService);

        int efSearchValue = KNNSettings.getEfSearchParam(INDEX_NAME);
        mockNode.close();
        assertEquals(userProvidedEfSearch, efSearchValue);
    }

    @SneakyThrows
    public void testShardLevelRescoringDisabled_whenNoValuesProvidedByUser_thenDefaultSettingsUsed() {
        Node mockNode = createMockNode(Collections.emptyMap());
        mockNode.start();
        ClusterService clusterService = mockNode.injector().getInstance(ClusterService.class);
        mockNode.client().admin().cluster().state(new ClusterStateRequest()).actionGet();
        mockNode.client().admin().indices().create(new CreateIndexRequest(INDEX_NAME)).actionGet();
        KNNSettings.state().setClusterService(clusterService);

        boolean shardLevelRescoringDisabled = KNNSettings.isShardLevelRescoringDisabledForDiskBasedVector(INDEX_NAME);
        mockNode.close();
        assertFalse(shardLevelRescoringDisabled);
    }

    @SneakyThrows
    public void testShardLevelRescoringDisabled_whenValueProvidedByUser_thenSettingApplied() {
        boolean userDefinedRescoringDisabled = true;
        Node mockNode = createMockNode(Collections.emptyMap());
        mockNode.start();
        ClusterService clusterService = mockNode.injector().getInstance(ClusterService.class);
        mockNode.client().admin().cluster().state(new ClusterStateRequest()).actionGet();
        mockNode.client().admin().indices().create(new CreateIndexRequest(INDEX_NAME)).actionGet();
        KNNSettings.state().setClusterService(clusterService);

        final Settings rescoringDisabledSetting = Settings.builder()
            .put(KNNSettings.KNN_DISK_VECTOR_SHARD_LEVEL_RESCORING_DISABLED, userDefinedRescoringDisabled)
            .build();

        mockNode.client().admin().indices().updateSettings(new UpdateSettingsRequest(rescoringDisabledSetting, INDEX_NAME)).actionGet();

        boolean shardLevelRescoringDisabled = KNNSettings.isShardLevelRescoringDisabledForDiskBasedVector(INDEX_NAME);
        mockNode.close();
        assertEquals(userDefinedRescoringDisabled, shardLevelRescoringDisabled);
    }

    @SneakyThrows
    public void testGetFaissAVX2DisabledSettingValueFromConfig_enableSetting_thenValidateAndSucceed() {
        boolean expectedKNNFaissAVX2Disabled = true;
        Node mockNode = createMockNode(Map.of(KNNSettings.KNN_FAISS_AVX2_DISABLED, expectedKNNFaissAVX2Disabled));
        mockNode.start();
        ClusterService clusterService = mockNode.injector().getInstance(ClusterService.class);
        KNNSettings.state().setClusterService(clusterService);
        boolean actualKNNFaissAVX2Disabled = KNNSettings.state().getSettingValue(KNNSettings.KNN_FAISS_AVX2_DISABLED);
        mockNode.close();
        assertEquals(expectedKNNFaissAVX2Disabled, actualKNNFaissAVX2Disabled);
    }

    @SneakyThrows
    public void testGetIndexThreadQty_WithDifferentValues_thenSuccess() {
        Node mockNode = createMockNode(Map.of(KNNSettings.KNN_ALGO_PARAM_INDEX_THREAD_QTY, 3));
        mockNode.start();
        ClusterService clusterService = mockNode.injector().getInstance(ClusterService.class);
        KNNSettings.state().setClusterService(clusterService);
        int threadQty = KNNSettings.getIndexThreadQty();
        mockNode.close();
        assertEquals(3, threadQty);
    }

    private Node createMockNode(Map<String, Object> configSettings) throws IOException {
        Path configDir = createTempDir();
        File configFile = configDir.resolve("opensearch.yml").toFile();
        FileWriter configFileWriter = new FileWriter(configFile, StandardCharsets.UTF_8);

        for (Map.Entry<String, Object> setting : configSettings.entrySet()) {
            configFileWriter.write("\"" + setting.getKey() + "\": " + setting.getValue());
        }
        configFileWriter.close();
        Collection<PluginInfo> plugins = basePlugins().stream()
            .map(
                p -> new PluginInfo(
                    p.getName(),
                    "classpath plugin",
                    "NA",
                    Version.CURRENT,
                    "1.8",
                    p.getName(),
                    null,
                    Collections.emptyList(),
                    false
                )
            )
            .collect(Collectors.toList());
        return new MockNode(baseSettings().build(), plugins, configDir, true);
    }

    private List<Class<? extends Plugin>> basePlugins() {
        List<Class<? extends Plugin>> plugins = new ArrayList<>();
        plugins.add(getTestTransportPlugin());
        plugins.add(MockHttpTransport.TestPlugin.class);
        plugins.add(KNNPlugin.class);
        return plugins;
    }

    private static Settings.Builder baseSettings() {
        final Path tempDir = createTempDir();
        return Settings.builder()
            .put(ClusterName.CLUSTER_NAME_SETTING.getKey(), InternalTestCluster.clusterName("single-node-cluster", randomLong()))
            .put(Environment.PATH_HOME_SETTING.getKey(), tempDir)
            .put(NetworkModule.TRANSPORT_TYPE_KEY, getTestTransportType())
            .put(dataNode());
    }

    @SneakyThrows
    public void testIndexThreadQty_thenUseDefaultValue() {
        // Create a mock node with no user-defined settings
        Node mockNode = createMockNode(Collections.emptyMap());
        mockNode.start();

        ClusterService clusterService = mockNode.injector().getInstance(ClusterService.class);
        KNNSettings.state().setClusterService(clusterService);

        int availableProcessors = Runtime.getRuntime().availableProcessors();
        int expectedThreadQty = (availableProcessors < 32) ? 1 : 4;
        int actualThreadQty = KNNSettings.getHardwareDefaultIndexThreadQty(Settings.EMPTY);

        assertEquals(expectedThreadQty, actualThreadQty);

        mockNode.close();
    }

    @SneakyThrows
    public void testIndexThreadQty_thenUseUserValue() {
        int userDefinedThreadQty = 12;
        Node mockNode = createMockNode(Map.of(KNNSettings.KNN_ALGO_PARAM_INDEX_THREAD_QTY, Integer.toString(userDefinedThreadQty)));
        mockNode.start();
        ClusterService clusterService = mockNode.injector().getInstance(ClusterService.class);
        KNNSettings.state().setClusterService(clusterService);
        int actualThreadQty = KNNSettings.getIndexThreadQty();
        assertEquals(userDefinedThreadQty, actualThreadQty);
        mockNode.close();
    }

    public void testGetHardwareDefaultIndexThreadQty_ProcessorLimits() {
        // Test with settings that limit processors
        Settings settingsWithLimit = Settings.builder().put("processors", 16).build();
        int threadQtyWithLimit = KNNSettings.getHardwareDefaultIndexThreadQty(settingsWithLimit);
        assertEquals(1, threadQtyWithLimit); // 16 < 32, should return 1

        // Test with settings that have high processor count
        Settings settingsWithHighLimit = Settings.builder().put("processors", 64).build();
        int threadQtyWithHighLimit = KNNSettings.getHardwareDefaultIndexThreadQty(settingsWithHighLimit);
        assertEquals(4, threadQtyWithHighLimit); // 64 >= 32, should return 4

        // Test with empty settings (should use system default)
        Settings emptySettings = Settings.EMPTY;
        int threadQtyWithEmpty = KNNSettings.getHardwareDefaultIndexThreadQty(emptySettings);
        int availableProcessors = Runtime.getRuntime().availableProcessors();
        int expected = (availableProcessors >= 32) ? 4 : 1;
        assertEquals(expected, threadQtyWithEmpty);
    }

    public void testDirectIONodeSettings_properties() {
        for (Setting<?> setting : List.of(
            KNNSettings.KNN_DIRECT_IO_RESCORE_ENABLED_SETTING,
            KNNSettings.KNN_DIRECT_IO_RESCORE_PREFETCH_THREADS_SETTING
        )) {
            final EnumSet<Setting.Property> properties = setting.getProperties();
            assertTrue(setting.getKey(), properties.contains(Setting.Property.NodeScope));
            assertTrue(setting.getKey(), properties.contains(Setting.Property.Dynamic));
            assertFalse(setting.getKey(), properties.contains(Setting.Property.IndexScope));
            assertFalse(setting.getKey(), properties.contains(Setting.Property.Final));
        }
    }

    /**
     * Every Direct I/O setting must be returned by getSettings(), or KNNPlugin never registers it
     * and any attempt to set one is rejected as an unknown setting.
     */
    public void testDirectIOSettings_areRegistered() {
        final List<String> registeredKeys = KNNSettings.state().getSettings().stream().map(Setting::getKey).collect(Collectors.toList());
        for (String key : List.of(
            KNNSettings.KNN_DIRECT_IO_RESCORE_ENABLED,
            KNNSettings.KNN_DIRECT_IO_RESCORE_PREFETCH_THREADS,
            KNNSettings.KNN_DIRECT_IO_RESCORE_PREFETCH_STAGED_RANGES,
            KNNSettings.KNN_DIRECT_IO_RESCORE_PREFETCH_MAX_SPAN,
            KNNSettings.KNN_DIRECT_IO_RESCORE_READ_PROFILE,
            KNNSettings.KNN_DIRECT_IO_READAHEAD_WINDOW,
            KNNSettings.KNN_DIRECT_IO_READAHEAD_TRIGGER
        )) {
            assertTrue(key + " is not registered", registeredKeys.contains(key));
        }
    }

    /**
     * P2's read profile: two values, {@code ops} by default, and <b>not</b> dynamic.
     *
     * <p>The static property is the load-bearing one. A {@code _cluster/settings} PUT reaches every node in
     * the cluster, and the profile is a property of a node's <em>device</em>: in a cluster of mixed instance
     * types a dynamic key would flatten one device's profile onto the other's nodes. The control plane writes
     * it to {@code opensearch.yml} per instance type, and an instance type change is a blue/green.
     */
    public void testDirectIOReadProfile_isAStaticNodeSettingWithTwoValues() {
        final Setting<String> setting = KNNSettings.KNN_DIRECT_IO_RESCORE_READ_PROFILE_SETTING;
        final EnumSet<Setting.Property> properties = setting.getProperties();
        assertTrue(properties.contains(Setting.Property.NodeScope));
        assertFalse("the read profile must not be dynamic", properties.contains(Setting.Property.Dynamic));
        assertFalse(properties.contains(Setting.Property.IndexScope));
        assertFalse(properties.contains(Setting.Property.Final));
        assertFalse("and so a cluster settings PUT must not be able to carry it", setting.isDynamic());

        assertEquals("ops", setting.getDefault(Settings.EMPTY));
        assertEquals("ops", setting.get(Settings.EMPTY));
        assertEquals("ops", setting.get(Settings.builder().put(setting.getKey(), "ops").build()));
        assertEquals("bytes", setting.get(Settings.builder().put(setting.getKey(), "bytes").build()));
        // Whitespace an operator's yaml may leave behind is not a different profile.
        assertEquals("bytes", setting.get(Settings.builder().put(setting.getKey(), " bytes ").build()));
    }

    /**
     * Any other value fails, and the message names both profiles. Thrown from the setting's own parser, so
     * {@code SettingsModule}'s validation of registered node settings fails node start rather than letting
     * a typo silently read as one of the two.
     */
    public void testDirectIOReadProfile_rejectsAnythingElse() {
        final Setting<String> setting = KNNSettings.KNN_DIRECT_IO_RESCORE_READ_PROFILE_SETTING;
        for (String bad : List.of("nvme", "ebs", "auto", "OPS", "Bytes", "iops", "")) {
            final IllegalArgumentException thrown = expectThrows(
                IllegalArgumentException.class,
                () -> setting.get(Settings.builder().put(setting.getKey(), bad).build())
            );
            assertTrue("the message must name ops: " + thrown.getMessage(), thrown.getMessage().contains("[ops]"));
            assertTrue("and bytes: " + thrown.getMessage(), thrown.getMessage().contains("[bytes]"));
            assertTrue("and the key: " + thrown.getMessage(), thrown.getMessage().contains(setting.getKey()));
        }
    }

    /**
     * The pool size is the one prefetch tuning default left after phase 12 H5 removed the rescore seam's
     * own window; it is a measurement-backed default, so a change to it should be a change to this
     * assertion too.
     */
    public void testDirectIORescorePrefetchSettings_defaults() {
        assertEquals(
            Integer.valueOf(expectedPrefetchThreadsDefault()),
            KNNSettings.KNN_DIRECT_IO_RESCORE_PREFETCH_THREADS_SETTING.getDefault(Settings.EMPTY)
        );
    }

    /**
     * The pool default is derived from the core count rather than fixed at 64, so that a 4-vCPU node does
     * not get 64 reader threads contending with the search threads that wait on them. Asserted as the
     * clamp rather than as a literal, because the literal differs between the CI host and the node; the
     * properties worth pinning are the two bounds and the 2x slope.
     */
    public void testDirectIORescorePrefetchThreads_defaultIsClampedAroundTheCoreCount() {
        final int derived = KNNSettings.KNN_DIRECT_IO_RESCORE_PREFETCH_THREADS_SETTING.getDefault(Settings.EMPTY);
        assertEquals(expectedPrefetchThreadsDefault(), derived);
        assertTrue(
            "the derived pool size must not drop below " + KNNSettings.KNN_DIRECT_IO_RESCORE_PREFETCH_THREADS_MIN,
            derived >= KNNSettings.KNN_DIRECT_IO_RESCORE_PREFETCH_THREADS_MIN
        );
        assertTrue(
            "the derived pool size must not exceed " + KNNSettings.KNN_DIRECT_IO_RESCORE_PREFETCH_THREADS_MAX,
            derived <= KNNSettings.KNN_DIRECT_IO_RESCORE_PREFETCH_THREADS_MAX
        );
    }

    /** An explicitly set value still wins: only the default moved, so the operator contract is unchanged. */
    public void testDirectIORescorePrefetchThreads_explicitValueStillWins() {
        assertEquals(
            Integer.valueOf(64),
            KNNSettings.KNN_DIRECT_IO_RESCORE_PREFETCH_THREADS_SETTING.get(
                Settings.builder().put(KNNSettings.KNN_DIRECT_IO_RESCORE_PREFETCH_THREADS, 64).build()
            )
        );
    }

    /** {@code clamp(2 x cores, 2, 16)}, recomputed here so the test states the rule rather than a number. */
    private static int expectedPrefetchThreadsDefault() {
        final int cores = Runtime.getRuntime().availableProcessors();
        return Math.min(
            KNNSettings.KNN_DIRECT_IO_RESCORE_PREFETCH_THREADS_MAX,
            Math.max(KNNSettings.KNN_DIRECT_IO_RESCORE_PREFETCH_THREADS_MIN, 2 * cores)
        );
    }

    /** A pool size of zero would mean "no reads at all"; the bound has to reject it. */
    public void testDirectIORescorePrefetchSettings_rejectNonPositiveValues() {
        for (Setting<Integer> setting : List.of(KNNSettings.KNN_DIRECT_IO_RESCORE_PREFETCH_THREADS_SETTING)) {
            expectThrows(IllegalArgumentException.class, () -> setting.get(Settings.builder().put(setting.getKey(), 0).build()));
        }
    }

    public void testDirectIORescorePrefetchAccessors_returnDefaultsWhenUnset() {
        assertEquals(expectedPrefetchThreadsDefault(), KNNSettings.getDirectIORescorePrefetchThreads());
    }

    /**
     * These are read while a query is being scored, from code that must never fail a query, and a harness
     * that never initialized the singleton reaches them too.
     */
    public void testDirectIORescorePrefetchAccessors_whenClusterServiceIsNotSet_thenReturnDefaults() {
        KNNSettings.state().setClusterService(null);
        assertEquals(expectedPrefetchThreadsDefault(), KNNSettings.getDirectIORescorePrefetchThreads());
    }

    public void testDirectIORescorePrefetchSettings_areReadableByKey() {
        assertEquals(
            Integer.valueOf(expectedPrefetchThreadsDefault()),
            KNNSettings.state().getSettingValue(KNNSettings.KNN_DIRECT_IO_RESCORE_PREFETCH_THREADS)
        );
    }

    /** Off by default: with it off the rescore path is byte for byte what it is without the seam. */
    public void testDirectIORescoreEnabledSetting_default() {
        assertFalse(KNNSettings.KNN_DIRECT_IO_RESCORE_ENABLED_SETTING.getDefault(Settings.EMPTY));
        assertFalse(KNNSettings.isDirectIORescoreEnabled());
    }

    /** Read from the query seam, which must never fail a query, so an uninitialized singleton is the default. */
    public void testDirectIORescoreEnabledAccessor_whenClusterServiceIsNotSet_thenReturnsDefault() {
        KNNSettings.state().setClusterService(null);
        assertFalse(KNNSettings.isDirectIORescoreEnabled());
    }

    public void testDirectIORescoreEnabledSetting_isReadableByKey() {
        assertEquals(Boolean.FALSE, KNNSettings.state().getSettingValue(KNNSettings.KNN_DIRECT_IO_RESCORE_ENABLED));
    }

    /**
     * D3b. The direct-memory wall is the operator's explicit {@code -XX:MaxDirectMemorySize} when they
     * pinned one, and {@code heapSize / 2} only when they did not.
     *
     * <p>Why this is a test and not a comment: the bench node runs {@code -XX:MaxDirectMemorySize=2g} with
     * {@code -Xmx24g}, where D3's original default of {@code Runtime.maxMemory() / 4} is 6 GB — a ceiling
     * three times <em>above</em> the wall, which is not a ceiling at all but exactly the
     * {@code OutOfMemoryError: Direct buffer memory} the budget exists to avoid. Pinning the flag is
     * routine, so the wall has to be read.
     *
     * <p>The resolution is a pure function of (flag, heap) so that both directions are testable from one
     * JVM: a JVM cannot change its own {@code MaxDirectMemorySize}, which HotSpot reports as not writeable
     * (MEASURED on Corretto 17.0.20.1: {@code VMOption.isWriteable() == false}).
     */
    public void testTheDirectMemoryWallIsTheExplicitFlagWhenSetAndHalfTheHeapWhenNot() {
        final long heap = 24L * 1024 * 1024 * 1024;

        // The bench node's actual configuration: 2g pinned against a 24g heap.
        assertEquals(
            "an explicit flag is the wall, whatever the heap says",
            2L * 1024 * 1024 * 1024,
            KNNSettings.directMemoryWall("2147483648", heap)
        );
        assertEquals(
            "so the node's default budget is 1 GB, not 6 GB",
            1024L * 1024 * 1024,
            KNNSettings.directMemoryWall("2147483648", heap) / 2
        );

        // Unset. HotSpot reports value "0" with origin DEFAULT (MEASURED on the same JVM), which is the
        // same convention JvmErgonomics.java:74 uses to decide it must add the flag itself.
        assertEquals("\"0\" means the operator did not pin one", heap / 2, KNNSettings.directMemoryWall("0", heap));
        assertEquals("and then the budget is heap/4, which is what D3 shipped", heap / 4, KNNSettings.directMemoryWall("0", heap) / 2);

        // Degenerate readings fall back to the ergonomic wall rather than to zero, because a zero budget
        // means "off" and silently disabling the bound is the wrong direction to fail in.
        assertEquals("a JVM that cannot answer", heap / 2, KNNSettings.directMemoryWall(null, heap));
        assertEquals("a value that is not a number", heap / 2, KNNSettings.directMemoryWall("not-a-number", heap));
        assertEquals("a negative flag", heap / 2, KNNSettings.directMemoryWall("-1", heap));

        // The bean hands back a bare decimal today; trim() is cheap insurance.
        assertEquals("surrounding whitespace", 2147483648L, KNNSettings.directMemoryWall("  2147483648 ", heap));
    }

    /** The wall resolved on this JVM is positive and the shipped budget default is exactly half of it. */
    public void testTheResolvedWallIsUsableAndTheDefaultIsHalfOfIt() {
        assertTrue(
            "the wall must resolve to something positive on any JVM this runs on",
            KNNSettings.KNN_DIRECT_IO_DIRECT_MEMORY_WALL_BYTES > 0
        );
        assertEquals(
            KNNSettings.KNN_DIRECT_IO_DIRECT_MEMORY_WALL_BYTES / 2,
            KNNSettings.KNN_DIRECT_IO_DIRECT_MEMORY_BUDGET_SETTING.getDefault(Settings.EMPTY).getBytes()
        );
        // This JVM is a gradle test worker, so whichever branch it took, the answer must agree with the
        // function applied to what the JVM actually reports.
        assertEquals(
            KNNSettings.directMemoryWall(
                ManagementFactory.getPlatformMXBean(HotSpotDiagnosticMXBean.class).getVMOption("MaxDirectMemorySize").getValue(),
                Runtime.getRuntime().maxMemory()
            ),
            KNNSettings.KNN_DIRECT_IO_DIRECT_MEMORY_WALL_BYTES
        );
    }
}

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import java.nio.charset.StandardCharsets;

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
            KNNSettings.KNN_DIRECT_IO_READAHEAD_WINDOW,
            KNNSettings.KNN_DIRECT_IO_READAHEAD_TRIGGER
        )) {
            assertTrue(key + " is not registered", registeredKeys.contains(key));
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

}

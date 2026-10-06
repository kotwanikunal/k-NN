/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.stats;

import com.google.common.cache.CacheStats;
import com.google.common.collect.ImmutableMap;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.memory.NativeMemoryCacheManager;
import org.opensearch.knn.index.store.DirectIOVectorIndexInput;
import org.opensearch.knn.indices.ModelCache;
import org.opensearch.knn.indices.ModelDao;
import org.opensearch.knn.plugin.stats.suppliers.EventOccurredWithinThresholdSupplier;
import org.opensearch.knn.plugin.stats.suppliers.KNNCircuitBreakerSupplier;
import org.opensearch.knn.plugin.stats.suppliers.KNNCounterSupplier;
import org.opensearch.knn.plugin.stats.suppliers.KNNInnerCacheStatsSupplier;
import org.opensearch.knn.plugin.stats.suppliers.LibraryInitializedSupplier;
import org.opensearch.knn.plugin.stats.suppliers.ModelIndexStatusSupplier;
import org.opensearch.knn.plugin.stats.suppliers.ModelIndexingDegradingSupplier;
import org.opensearch.knn.plugin.stats.suppliers.NativeMemoryCacheManagerSupplier;

import java.time.temporal.ChronoUnit;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Supplier;

/**
 * Class represents all stats the plugin keeps track of
 */
public class KNNStats {

    private final Map<String, KNNStat<?>> knnStats;

    /**
     * Constructor
     */
    public KNNStats() {
        this.knnStats = buildStatsMap();
    }

    /**
     * Get the stats
     *
     * @return all of the stats
     */
    public Map<String, KNNStat<?>> getStats() {
        return knnStats;
    }

    /**
     * Get a map of the stats that are kept at the node level
     *
     * @return Map of stats kept at the node level
     */
    public Map<String, KNNStat<?>> getNodeStats() {
        return getFilteredStats(false);
    }

    /**
     * Get a map of the stats that are kept at the cluster level
     *
     * @return Map of stats kept at the cluster level
     */
    public Map<String, KNNStat<?>> getClusterStats() {
        return getFilteredStats(true);
    }

    private Map<String, KNNStat<?>> getFilteredStats(boolean isClusterLevel) {
        return knnStats.entrySet()
            .stream()
            .filter(entry -> entry.getValue().isClusterLevel() == isClusterLevel)
            .collect(ImmutableMap.toImmutableMap(Map.Entry::getKey, Map.Entry::getValue));
    }

    private Map<String, KNNStat<?>> buildStatsMap() {
        ImmutableMap.Builder<String, KNNStat<?>> builder = ImmutableMap.<String, KNNStat<?>>builder();
        addQueryStats(builder);
        addNativeMemoryStats(builder);
        addEngineStats(builder);
        addScriptStats(builder);
        addModelStats(builder);
        addGraphStats(builder);
        addRemoteIndexBuildStats(builder);
        addDirectIOStats(builder);
        return builder.build();
    }

    /**
     * The Direct I/O read path's node-wide counters, at {@code _plugins/_knn/stats} under
     * {@code direct_io_stats}. This is the k-NN-shaped place for them: {@code KNNStats} is already how every
     * other node-level k-NN counter reaches an operator, and the alternative — the per-segment
     * {@code KNNVectorStorageDirectory} the codec seam creates — is discarded with its reader and so can
     * never be read over REST.
     *
     * <p>Two groups, because they answer two different questions.
     * <ul>
     *   <li><b>dispatch</b> — is the seam engaging, and if not, why? {@code routed_opens} against the five
     *       decline reasons, which want five different operator responses: a compound segment, i.e. every
     *       flushed segment at the shipped defaults and nothing to act on
     *       ({@code declined_compound}); an unsupported substrate ({@code declined_chain}); a bug
     *       ({@code declined_missing}); the size floor working as designed ({@code declined_too_small}); or
     *       a filesystem with no {@code O_DIRECT} ({@code declined_open_failed}).</li>
     *   <li><b>reads</b> — once engaged, what is it doing? The staged/blocking/read-ahead channels and the
     *       byte counters behind the amplification ratio. {@code stage_declined} against
     *       {@code staged_reads} is the queue-capacity decline rate phase 13's audit row 7 measured as the
     *       binding constraint at concurrent load, and the Oct 4-5 node sweep then measured 46-52% declines
     *       and 783k device reads per 5,000 queries at an effective capacity of 64 against 0% and 707k at
     *       1,024. {@code queue_peak} is the occupancy high-water mark the derived default is validated
     *       against, and {@code pool_rebuilds} is how an operator sees that a live capacity change took
     *       effect.</li>
     *   <li><b>memory</b> — how much off-heap is it holding, and is the budget biting? {@code live_bytes}
     *       against {@code budget_bytes} (the setting, echoed so one GET answers both halves),
     *       {@code peak_bytes} for the high-water mark, and {@code declined_budget} for prefetches the
     *       budget refused. The node sat at 1.69 GB of live direct buffers against a 2g
     *       {@code MaxDirectMemorySize} before this section existed, which is the measurement that made it
     *       necessary.</li>
     * </ul>
     *
     * <p>Node level rather than cluster level, and cumulative since node start, because every counter is a
     * property of one node's reads: the device, the filesystem block size and the page cache that decide
     * whether Direct I/O is winning are all per node.
     */
    private void addDirectIOStats(ImmutableMap.Builder<String, KNNStat<?>> builder) {
        builder.put(StatNames.DIRECT_IO_STATS.getName(), createNodeStat(KNNStats::createDirectIOStatsMap));
    }

    private static Map<String, Map<String, Object>> createDirectIOStatsMap() {
        final DirectIOVectorIndexInput.Stats stats = DirectIOVectorIndexInput.STATS;

        final Map<String, Object> dispatch = new HashMap<>();
        dispatch.put("routed_opens", stats.routedOpens());
        dispatch.put("declined_opens", stats.declinedOpens());
        dispatch.put("declined_compound", stats.declinedCompound());
        dispatch.put("declined_chain", stats.declinedChain());
        dispatch.put("declined_missing", stats.declinedMissing());
        dispatch.put("declined_too_small", stats.declinedTooSmall());
        dispatch.put("declined_open_failed", stats.declinedOpenFailed());

        final Map<String, Object> reads = new HashMap<>();
        reads.put("blocking_reads", stats.blockingReads());
        reads.put("staged_reads", stats.stagedReads());
        reads.put("staged_hits", stats.stagedHits());
        reads.put("stage_declined", stats.stageDeclined());
        reads.put("staged_wait_timeouts", stats.stagedWaitTimeouts());
        reads.put("prefetch_calls", stats.prefetchCalls());
        reads.put("in_flight_peak", stats.inFlightPeak());
        reads.put("readahead_armed", stats.readaheadArmed());
        reads.put("readahead_reads", stats.readaheadReads());
        reads.put("readahead_hits", stats.readaheadHits());
        reads.put("readahead_declined", stats.readaheadDeclined());
        reads.put("staged_bytes", stats.stagedBytes());
        reads.put("blocking_bytes", stats.blockingBytes());
        reads.put("readahead_bytes", stats.readaheadBytes());
        reads.put("served_bytes", stats.servedBytes());
        reads.put("declined_span_bytes", stats.declinedSpanBytes());
        reads.put("queue_peak", stats.queuePeak());
        reads.put("pool_rebuilds", stats.poolRebuilds());

        final Map<String, Object> memory = new HashMap<>();
        // Drain first, so the number an operator reads is not stale by however long it has been since the
        // last allocation decided to drain it. See DirectIOVectorIndexInput#reclaim.
        memory.put("live_bytes", DirectIOVectorIndexInput.reclaim());
        memory.put("peak_bytes", stats.directMemoryPeak());
        memory.put("declined_budget", stats.declinedBudget());
        memory.put("budget_bytes", KNNSettings.getDirectIODirectMemoryBudget().getBytes());

        final Map<String, Map<String, Object>> directIOStats = new HashMap<>();
        directIOStats.put("dispatch", dispatch);
        directIOStats.put("reads", reads);
        directIOStats.put("memory", memory);
        return directIOStats;
    }

    private void addQueryStats(ImmutableMap.Builder<String, KNNStat<?>> builder) {
        // KNN Query Stats
        builder.put(StatNames.KNN_QUERY_REQUESTS.getName(), createNodeStat(new KNNCounterSupplier(KNNCounter.KNN_QUERY_REQUESTS)))
            .put(
                StatNames.KNN_QUERY_WITH_FILTER_REQUESTS.getName(),
                createNodeStat(new KNNCounterSupplier(KNNCounter.KNN_QUERY_WITH_FILTER_REQUESTS))
            );

        // Min Score Query Stats
        builder.put(
            StatNames.MIN_SCORE_QUERY_REQUESTS.getName(),
            createNodeStat(new KNNCounterSupplier(KNNCounter.MIN_SCORE_QUERY_REQUESTS))
        )
            .put(
                StatNames.MIN_SCORE_QUERY_WITH_FILTER_REQUESTS.getName(),
                createNodeStat(new KNNCounterSupplier(KNNCounter.MIN_SCORE_QUERY_WITH_FILTER_REQUESTS))
            );

        // Max Distance Query Stats
        builder.put(
            StatNames.MAX_DISTANCE_QUERY_REQUESTS.getName(),
            createNodeStat(new KNNCounterSupplier(KNNCounter.MAX_DISTANCE_QUERY_REQUESTS))
        )
            .put(
                StatNames.MAX_DISTANCE_QUERY_WITH_FILTER_REQUESTS.getName(),
                createNodeStat(new KNNCounterSupplier(KNNCounter.MAX_DISTANCE_QUERY_WITH_FILTER_REQUESTS))
            );
    }

    private void addNativeMemoryStats(ImmutableMap.Builder<String, KNNStat<?>> builder) {
        builder.put(StatNames.HIT_COUNT.getName(), createNodeStat(new KNNInnerCacheStatsSupplier(CacheStats::hitCount)))
            .put(StatNames.MISS_COUNT.getName(), createNodeStat(new KNNInnerCacheStatsSupplier(CacheStats::missCount)))
            .put(StatNames.LOAD_SUCCESS_COUNT.getName(), createNodeStat(new KNNInnerCacheStatsSupplier(CacheStats::loadSuccessCount)))
            .put(StatNames.LOAD_EXCEPTION_COUNT.getName(), createNodeStat(new KNNInnerCacheStatsSupplier(CacheStats::loadExceptionCount)))
            .put(StatNames.TOTAL_LOAD_TIME.getName(), createNodeStat(new KNNInnerCacheStatsSupplier(CacheStats::totalLoadTime)))
            .put(StatNames.EVICTION_COUNT.getName(), createNodeStat(new KNNInnerCacheStatsSupplier(CacheStats::evictionCount)))
            .put(
                StatNames.GRAPH_MEMORY_USAGE.getName(),
                createNodeStat(new NativeMemoryCacheManagerSupplier<>(NativeMemoryCacheManager::getIndicesSizeInKilobytes))
            )
            .put(
                StatNames.GRAPH_MEMORY_USAGE_PERCENTAGE.getName(),
                createNodeStat(new NativeMemoryCacheManagerSupplier<>(NativeMemoryCacheManager::getIndicesSizeAsPercentage))
            )
            .put(
                StatNames.INDICES_IN_CACHE.getName(),
                createNodeStat(new NativeMemoryCacheManagerSupplier<>(NativeMemoryCacheManager::getIndicesCacheStats))
            )
            .put(
                StatNames.CACHE_CAPACITY_REACHED.getName(),
                createNodeStat(new NativeMemoryCacheManagerSupplier<>(NativeMemoryCacheManager::isCacheCapacityReached))
            )
            .put(StatNames.GRAPH_QUERY_ERRORS.getName(), createNodeStat(new KNNCounterSupplier(KNNCounter.GRAPH_QUERY_ERRORS)))
            .put(StatNames.GRAPH_QUERY_REQUESTS.getName(), createNodeStat(new KNNCounterSupplier(KNNCounter.GRAPH_QUERY_REQUESTS)))
            .put(StatNames.GRAPH_INDEX_ERRORS.getName(), createNodeStat(new KNNCounterSupplier(KNNCounter.GRAPH_INDEX_ERRORS)))
            .put(StatNames.GRAPH_INDEX_REQUESTS.getName(), createNodeStat(new KNNCounterSupplier(KNNCounter.GRAPH_INDEX_REQUESTS)))
            .put(StatNames.CIRCUIT_BREAKER_TRIGGERED.getName(), createClusterStat(new KNNCircuitBreakerSupplier()));
    }

    private void addEngineStats(ImmutableMap.Builder<String, KNNStat<?>> builder) {
        builder.put(StatNames.FAISS_LOADED.getName(), createNodeStat(new LibraryInitializedSupplier(KNNEngine.FAISS)))
            .put(StatNames.NMSLIB_LOADED.getName(), createNodeStat(new LibraryInitializedSupplier(KNNEngine.NMSLIB)))
            .put(StatNames.LUCENE_LOADED.getName(), createNodeStat(new LibraryInitializedSupplier(KNNEngine.LUCENE)));
    }

    private void addScriptStats(ImmutableMap.Builder<String, KNNStat<?>> builder) {
        builder.put(StatNames.SCRIPT_COMPILATIONS.getName(), createNodeStat(new KNNCounterSupplier(KNNCounter.SCRIPT_COMPILATIONS)))
            .put(
                StatNames.SCRIPT_COMPILATION_ERRORS.getName(),
                createNodeStat(new KNNCounterSupplier(KNNCounter.SCRIPT_COMPILATION_ERRORS))
            )
            .put(StatNames.SCRIPT_QUERY_REQUESTS.getName(), createNodeStat(new KNNCounterSupplier(KNNCounter.SCRIPT_QUERY_REQUESTS)))
            .put(StatNames.SCRIPT_QUERY_ERRORS.getName(), createNodeStat(new KNNCounterSupplier(KNNCounter.SCRIPT_QUERY_ERRORS)));
    }

    private void addModelStats(ImmutableMap.Builder<String, KNNStat<?>> builder) {
        builder.put(
            StatNames.INDEXING_FROM_MODEL_DEGRADED.getName(),
            createNodeStat(
                new EventOccurredWithinThresholdSupplier(
                    new ModelIndexingDegradingSupplier(ModelCache::getEvictedDueToSizeAt),
                    KNNConstants.MODEL_CACHE_CAPACITY_ATROPHY_THRESHOLD_IN_MINUTES,
                    ChronoUnit.MINUTES
                )
            )
        )
            .put(StatNames.MODEL_INDEX_STATUS.getName(), createClusterStat(new ModelIndexStatusSupplier<>(ModelDao::getHealthStatus)))
            .put(StatNames.TRAINING_REQUESTS.getName(), createNodeStat(new KNNCounterSupplier(KNNCounter.TRAINING_REQUESTS)))
            .put(StatNames.TRAINING_ERRORS.getName(), createNodeStat(new KNNCounterSupplier(KNNCounter.TRAINING_ERRORS)))
            .put(
                StatNames.TRAINING_MEMORY_USAGE.getName(),
                createNodeStat(new NativeMemoryCacheManagerSupplier<>(NativeMemoryCacheManager::getTrainingSizeInKilobytes))
            )
            .put(
                StatNames.TRAINING_MEMORY_USAGE_PERCENTAGE.getName(),
                createNodeStat(new NativeMemoryCacheManagerSupplier<>(NativeMemoryCacheManager::getTrainingSizeAsPercentage))
            );
    }

    private void addGraphStats(ImmutableMap.Builder<String, KNNStat<?>> builder) {
        builder.put(StatNames.GRAPH_STATS.getName(), createNodeStat(new Supplier<Map<String, Map<String, Object>>>() {
            @Override
            public Map<String, Map<String, Object>> get() {
                return createGraphStatsMap();
            }
        }));
    }

    private Map<String, Map<String, Object>> createGraphStatsMap() {
        Map<String, Object> mergeMap = new HashMap<>();
        mergeMap.put(KNNGraphValue.MERGE_CURRENT_OPERATIONS.getName(), KNNGraphValue.MERGE_CURRENT_OPERATIONS.getValue());
        mergeMap.put(KNNGraphValue.MERGE_CURRENT_DOCS.getName(), KNNGraphValue.MERGE_CURRENT_DOCS.getValue());
        mergeMap.put(KNNGraphValue.MERGE_CURRENT_SIZE_IN_BYTES.getName(), KNNGraphValue.MERGE_CURRENT_SIZE_IN_BYTES.getValue());
        mergeMap.put(KNNGraphValue.MERGE_TOTAL_OPERATIONS.getName(), KNNGraphValue.MERGE_TOTAL_OPERATIONS.getValue());
        mergeMap.put(KNNGraphValue.MERGE_TOTAL_TIME_IN_MILLIS.getName(), KNNGraphValue.MERGE_TOTAL_TIME_IN_MILLIS.getValue());
        mergeMap.put(KNNGraphValue.MERGE_TOTAL_DOCS.getName(), KNNGraphValue.MERGE_TOTAL_DOCS.getValue());
        mergeMap.put(KNNGraphValue.MERGE_TOTAL_SIZE_IN_BYTES.getName(), KNNGraphValue.MERGE_TOTAL_SIZE_IN_BYTES.getValue());
        Map<String, Object> refreshMap = new HashMap<>();
        refreshMap.put(KNNGraphValue.REFRESH_TOTAL_OPERATIONS.getName(), KNNGraphValue.REFRESH_TOTAL_OPERATIONS.getValue());
        refreshMap.put(KNNGraphValue.REFRESH_TOTAL_TIME_IN_MILLIS.getName(), KNNGraphValue.REFRESH_TOTAL_TIME_IN_MILLIS.getValue());
        Map<String, Map<String, Object>> graphStatsMap = new HashMap<>();
        graphStatsMap.put(StatNames.MERGE.getName(), mergeMap);
        graphStatsMap.put(StatNames.REFRESH.getName(), refreshMap);
        return graphStatsMap;
    }

    private void addRemoteIndexBuildStats(ImmutableMap.Builder<String, KNNStat<?>> builder) {
        builder.put(StatNames.REMOTE_VECTOR_INDEX_BUILD_STATS.getName(), new KNNStat<>(false, this::createRemoteIndexStatsMap));
    }

    private Map<String, Map<String, Object>> createRemoteIndexStatsMap() {
        Map<String, Object> clientStatsMap = new HashMap<>();
        clientStatsMap.put(
            KNNRemoteIndexBuildValue.BUILD_REQUEST_SUCCESS_COUNT.getName(),
            KNNRemoteIndexBuildValue.BUILD_REQUEST_SUCCESS_COUNT.getValue()
        );
        clientStatsMap.put(
            KNNRemoteIndexBuildValue.BUILD_REQUEST_FAILURE_COUNT.getName(),
            KNNRemoteIndexBuildValue.BUILD_REQUEST_FAILURE_COUNT.getValue()
        );
        clientStatsMap.put(
            KNNRemoteIndexBuildValue.STATUS_REQUEST_SUCCESS_COUNT.getName(),
            KNNRemoteIndexBuildValue.STATUS_REQUEST_SUCCESS_COUNT.getValue()
        );
        clientStatsMap.put(
            KNNRemoteIndexBuildValue.STATUS_REQUEST_FAILURE_COUNT.getName(),
            KNNRemoteIndexBuildValue.STATUS_REQUEST_FAILURE_COUNT.getValue()
        );
        clientStatsMap.put(
            KNNRemoteIndexBuildValue.INDEX_BUILD_SUCCESS_COUNT.getName(),
            KNNRemoteIndexBuildValue.INDEX_BUILD_SUCCESS_COUNT.getValue()
        );
        clientStatsMap.put(
            KNNRemoteIndexBuildValue.INDEX_BUILD_FAILURE_COUNT.getName(),
            KNNRemoteIndexBuildValue.INDEX_BUILD_FAILURE_COUNT.getValue()
        );
        clientStatsMap.put(
            KNNRemoteIndexBuildValue.INDEX_BUILD_MERGE_ABORT_EXCEPTION.getName(),
            KNNRemoteIndexBuildValue.INDEX_BUILD_MERGE_ABORT_EXCEPTION.getValue()
        );
        clientStatsMap.put(
            KNNRemoteIndexBuildValue.INDEX_BUILD_TERMINAL_EXCEPTION.getName(),
            KNNRemoteIndexBuildValue.INDEX_BUILD_TERMINAL_EXCEPTION.getValue()
        );
        clientStatsMap.put(KNNRemoteIndexBuildValue.WAITING_TIME.getName(), KNNRemoteIndexBuildValue.WAITING_TIME.getValue());

        Map<String, Object> repoStatsMap = new HashMap<>();
        repoStatsMap.put(KNNRemoteIndexBuildValue.WRITE_SUCCESS_COUNT.getName(), KNNRemoteIndexBuildValue.WRITE_SUCCESS_COUNT.getValue());
        repoStatsMap.put(KNNRemoteIndexBuildValue.WRITE_FAILURE_COUNT.getName(), KNNRemoteIndexBuildValue.WRITE_FAILURE_COUNT.getValue());
        repoStatsMap.put(KNNRemoteIndexBuildValue.WRITE_TIME.getName(), KNNRemoteIndexBuildValue.WRITE_TIME.getValue());
        repoStatsMap.put(KNNRemoteIndexBuildValue.READ_SUCCESS_COUNT.getName(), KNNRemoteIndexBuildValue.READ_SUCCESS_COUNT.getValue());
        repoStatsMap.put(KNNRemoteIndexBuildValue.READ_FAILURE_COUNT.getName(), KNNRemoteIndexBuildValue.READ_FAILURE_COUNT.getValue());
        repoStatsMap.put(KNNRemoteIndexBuildValue.READ_TIME.getName(), KNNRemoteIndexBuildValue.READ_TIME.getValue());

        Map<String, Object> buildStatsMap = new HashMap<>();
        buildStatsMap.put(
            KNNRemoteIndexBuildValue.REMOTE_INDEX_BUILD_CURRENT_FLUSH_OPERATIONS.getName(),
            KNNRemoteIndexBuildValue.REMOTE_INDEX_BUILD_CURRENT_FLUSH_OPERATIONS.getValue()
        );
        buildStatsMap.put(
            KNNRemoteIndexBuildValue.REMOTE_INDEX_BUILD_CURRENT_MERGE_OPERATIONS.getName(),
            KNNRemoteIndexBuildValue.REMOTE_INDEX_BUILD_CURRENT_MERGE_OPERATIONS.getValue()
        );
        buildStatsMap.put(
            KNNRemoteIndexBuildValue.REMOTE_INDEX_BUILD_CURRENT_FLUSH_SIZE.getName(),
            KNNRemoteIndexBuildValue.REMOTE_INDEX_BUILD_CURRENT_FLUSH_SIZE.getValue()
        );
        buildStatsMap.put(
            KNNRemoteIndexBuildValue.REMOTE_INDEX_BUILD_CURRENT_MERGE_SIZE.getName(),
            KNNRemoteIndexBuildValue.REMOTE_INDEX_BUILD_CURRENT_MERGE_SIZE.getValue()
        );
        buildStatsMap.put(
            KNNRemoteIndexBuildValue.REMOTE_INDEX_BUILD_FLUSH_TIME.getName(),
            KNNRemoteIndexBuildValue.REMOTE_INDEX_BUILD_FLUSH_TIME.getValue()
        );
        buildStatsMap.put(
            KNNRemoteIndexBuildValue.REMOTE_INDEX_BUILD_MERGE_TIME.getName(),
            KNNRemoteIndexBuildValue.REMOTE_INDEX_BUILD_MERGE_TIME.getValue()
        );

        Map<String, Map<String, Object>> remoteIndexBuildStatsMap = new HashMap<>();
        remoteIndexBuildStatsMap.put(StatNames.BUILD_STATS.getName(), buildStatsMap);
        remoteIndexBuildStatsMap.put(StatNames.CLIENT_STATS.getName(), clientStatsMap);
        remoteIndexBuildStatsMap.put(StatNames.REPOSITORY_STATS.getName(), repoStatsMap);
        return remoteIndexBuildStatsMap;
    }

    private static <T> KNNStat<T> createNodeStat(Supplier<T> supplier) {
        return new KNNStat<>(false, supplier);
    }

    private static <T> KNNStat<T> createClusterStat(Supplier<T> supplier) {
        return new KNNStat<>(true, supplier);
    }
}

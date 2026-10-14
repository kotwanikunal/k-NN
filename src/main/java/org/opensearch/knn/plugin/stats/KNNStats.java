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
import org.opensearch.knn.index.store.DirectIOVectorCache;
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
     * {@code KNNDirectIODirectory} the codec seam creates — is discarded with its reader and so can
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
     *       necessary. {@code wall_bytes} is the JVM-wide direct-memory wall the budget was derived from —
     *       the operator's explicit {@code -XX:MaxDirectMemorySize} when they pinned one, else
     *       {@code heapSize / 2} — reported beside {@code budget_bytes} so that "is this ceiling actually
     *       below the wall" is one GET rather than an SSH to read the command line. Neither
     *       {@code live_bytes} nor {@code peak_bytes} is ever reported above {@code wall_bytes}: the node
     *       read a {@code peak_bytes} of 7.46 GB against a 2 GB wall before D3c, which was arithmetic rather
     *       than memory.
     *       <p>{@code arenas_allocated} / {@code arenas_reused} / {@code arenas_retired} are the lease
     *       model's own numbers (D3c). A healthy node allocates one arena per reader thread and reuses it
     *       for every burst after, so {@code arenas_reused} should be orders of magnitude above
     *       {@code arenas_allocated} and {@code arenas_retired} should stay flat — it moves only when a
     *       staged read would not quiesce (watch {@code staged_wait_timeouts} with it) or when the burst
     *       geometry changed. {@code foreign_thread_reads} is ~0 unless some caller prefetches an input on
     *       one thread and reads it on another, which costs that read its read-ahead and nothing else.</li>
     *   <li><b>cache</b> — the node-wide slab store of individual vectors, every counter {@code 0} while
     *       {@code knn.direct_io.rescore.cache.max_bytes} is at its default of {@code 0b}.
     *       {@code hits / (hits + misses)} is the hit rate and is directly comparable with the 40.3%
     *       measured at 16 MB on 10M vectors and with the 16-S node figures, because a vector is counted
     *       <b>once</b>, at the lookup that serves or misses it; the lookup in {@code prefetch} decides
     *       whether the range joins the burst and counts nothing. See
     *       {@link DirectIOVectorCache#lookupsPerVector()}.
     *       <p>{@code bytes} is the slot bytes live vectors occupy and {@code entries} how many there are,
     *       while {@code slabs} and {@code slab_bytes} are what the node is actually holding off heap:
     *       slabs are whole and a partly filled one is charged in full, so {@code slab_bytes} is the figure
     *       to compare with a memory budget and {@code bytes} the one to compare with the working set.
     *       {@code max_bytes} is the target rounded down to whole slabs, which is why a target under one
     *       slab reads {@code 0}. {@code reclaimed_slabs} is how often the FIFO had to take the oldest slab
     *       and {@code readmissions} how many vectors a second chance kept; {@code evictions} is vectors
     *       dropped, by a reclaim or with a released slab. {@code span_fills} is vectors admitted because
     *       they lay inside a span a burst had already staged rather than because a read asked for them,
     *       which is {@code 0} while {@code knn.direct_io.rescore.cache.span_fill} is false and once the
     *       store is at its slab target. {@code slab_alloc_failures} counts the times the backend refused a
     *       slab the store was below its target for; it is one per time growth was given up on, because the
     *       store then stops asking until its target changes or it releases a slab, so non-zero means the
     *       node is holding a smaller cache than {@code max_bytes} asked for. {@code slab_memory} names the
     *       off-heap backend
     *       ({@code ffm} or {@code direct_buffer}), which decides whether releasing memory is immediate and
     *       whether the bytes sit inside {@code MaxDirectMemorySize}.</li>
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
        // O1, D2's rollback: fallback_reads and fallback_bytes rise, while staged_reads, blocking_reads and
        // dispatch.routed_opens stand still, exactly when knn.direct_io.rescore.enabled has been turned off
        // under readers that are already open. switched_to_mmap and switched_to_direct count the route
        // changes themselves, one per input per flip. 16-I's names.
        reads.put("fallback_reads", stats.fallbackReads());
        reads.put("fallback_bytes", stats.fallbackBytes());
        reads.put("switched_to_mmap", stats.switchedToMmap());
        reads.put("switched_to_direct", stats.switchedToDirect());

        final Map<String, Object> memory = new HashMap<>();
        // Drain first, so the number an operator reads is not stale by however long it has been since the
        // last allocation decided to drain it. See DirectIOVectorIndexInput#reclaim.
        memory.put("live_bytes", DirectIOVectorIndexInput.reclaim());
        memory.put("peak_bytes", stats.directMemoryPeak());
        memory.put("declined_budget", stats.declinedBudget());
        memory.put("budget_bytes", KNNSettings.getDirectIODirectMemoryBudget().getBytes());
        memory.put("wall_bytes", KNNSettings.KNN_DIRECT_IO_DIRECT_MEMORY_WALL_BYTES);
        memory.put("arenas_allocated", stats.arenasAllocated());
        memory.put("arenas_reused", stats.arenasReused());
        memory.put("arenas_retired", stats.arenasRetired());
        memory.put("foreign_thread_reads", stats.foreignThreadReads());
        // O2, D3d: the wide-group buffers a lease keeps beside its arena, and their own ceiling.
        // wide_buffer_bytes against wide_buffer_budget_bytes is the headroom; declined_wide_budget rising
        // while declined_budget stands still says the sub-budget and not the overall one is the bound that
        // bit. 16-I's names.
        memory.put("wide_buffer_bytes", stats.wideBufferLive());
        memory.put("wide_buffer_peak_bytes", stats.wideBufferPeak());
        memory.put("wide_buffer_budget_bytes", DirectIOVectorIndexInput.wideBufferBudgetBytes());
        memory.put("declined_wide_budget", stats.declinedWideBudget());

        final Map<String, Object> cache = new HashMap<>();
        cache.put("hits", DirectIOVectorCache.hits());
        cache.put("misses", DirectIOVectorCache.misses());
        cache.put("evictions", DirectIOVectorCache.evictions());
        cache.put("bytes", DirectIOVectorCache.bytes());
        cache.put("entries", DirectIOVectorCache.entries());
        cache.put("max_bytes", DirectIOVectorCache.maxBytes());
        cache.put("slab_memory", DirectIOVectorCache.slabMemory());
        cache.put("slabs", DirectIOVectorCache.slabs());
        cache.put("slab_bytes", DirectIOVectorCache.slabBytes());
        cache.put("reclaimed_slabs", DirectIOVectorCache.reclaimedSlabs());
        cache.put("readmissions", DirectIOVectorCache.readmissions());
        cache.put("span_fills", DirectIOVectorCache.spanFills());
        cache.put("slab_alloc_failures", DirectIOVectorCache.slabAllocFailures());
        // O3: the entries a merged-away or otherwise closed .vec gave back at close, and the files they came
        // from. closed_file_entries rising while evictions stands still is what says the dead segments cost
        // the live ones nothing.
        cache.put("closed_files", DirectIOVectorCache.closedFiles());
        cache.put("closed_file_entries", DirectIOVectorCache.closedFileEntries());

        final Map<String, Map<String, Object>> directIOStats = new HashMap<>();
        directIOStats.put("dispatch", dispatch);
        directIOStats.put("reads", reads);
        directIOStats.put("memory", memory);
        directIOStats.put("cache", cache);
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

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import com.sun.management.HotSpotDiagnosticMXBean;
import lombok.NonNull;
import lombok.Setter;
import lombok.extern.log4j.Log4j2;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.opensearch.OpenSearchParseException;
import org.opensearch.action.admin.cluster.settings.ClusterUpdateSettingsRequest;
import org.opensearch.action.admin.cluster.settings.ClusterUpdateSettingsResponse;
import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.Booleans;
import org.opensearch.common.Nullable;
import org.opensearch.common.settings.SecureSetting;
import org.opensearch.common.settings.Setting;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.unit.TimeValue;
import org.opensearch.core.action.ActionListener;
import org.opensearch.core.common.settings.SecureString;
import org.opensearch.core.common.unit.ByteSizeUnit;
import org.opensearch.core.common.unit.ByteSizeValue;
import org.opensearch.index.IndexModule;
import org.opensearch.index.IndexSettings;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.index.engine.MemoryOptimizedSearchSupportSpec;
import org.opensearch.knn.index.memory.NativeMemoryCacheManager;
import org.opensearch.knn.index.memory.NativeMemoryCacheManagerDto;
import org.opensearch.knn.index.util.IndexHyperParametersUtil;
import org.opensearch.knn.quantization.models.quantizationState.QuantizationStateCacheManager;
import org.opensearch.monitor.jvm.JvmInfo;
import org.opensearch.monitor.os.OsProbe;
import org.opensearch.transport.client.Client;

import java.lang.management.ManagementFactory;
import java.security.InvalidParameterException;
import org.opensearch.common.util.concurrent.OpenSearchExecutors;
import org.opensearch.threadpool.ThreadPool;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static java.util.stream.Collectors.toUnmodifiableMap;
import static org.opensearch.common.Booleans.parseBoolean;
import static org.opensearch.common.settings.Setting.Property.Dynamic;
import static org.opensearch.common.settings.Setting.Property.Final;
import static org.opensearch.common.settings.Setting.Property.IndexScope;
import static org.opensearch.common.settings.Setting.Property.NodeScope;
import static org.opensearch.common.settings.Setting.Property.UnmodifiableOnRestore;
import static org.opensearch.common.unit.MemorySizeValue.parseBytesSizeValueOrHeapRatio;
import static org.opensearch.core.common.unit.ByteSizeValue.parseBytesSizeValue;
import static org.opensearch.knn.common.featureflags.KNNFeatureFlags.getFeatureFlags;

import static org.opensearch.knn.common.KNNConstants.INDEX_THREAD_QUANTITY_THRESHOLD;
import static org.opensearch.knn.common.KNNConstants.INDEX_THREAD_QUANTITY_DEFAULT_LARGE;
import static org.opensearch.knn.common.KNNConstants.INDEX_THREAD_QUANTITY_DEFAULT_SMALL;
import static org.opensearch.knn.common.featureflags.KNNFeatureFlags.getFeatureFlagsWhichRebuildsCache;

/**
 * This class defines
 * 1. KNN settings to hold the <a href="https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md">HNSW algorithm parameters</a>.
 * 2. KNN settings to enable/disable plugin, circuit breaker settings
 * 3. KNN settings to manage graphs loaded in native memory
 */
@Log4j2
public class KNNSettings {

    private static final Logger logger = LogManager.getLogger(KNNSettings.class);
    private static KNNSettings INSTANCE;
    private static final OsProbe osProbe = OsProbe.getInstance();

    private static final int INDEX_THREAD_QTY_MAX = 32;
    private static final QuantizationStateCacheManager quantizationStateCacheManager = QuantizationStateCacheManager.getInstance();

    /**
     * Settings name
     */
    public static final String INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD = "index.knn.advanced.approximate_threshold";
    public static final String KNN_ALGO_PARAM_EF_SEARCH = "index.knn.algo_param.ef_search";
    public static final String KNN_ALGO_PARAM_INDEX_THREAD_QTY = "knn.algo_param.index_thread_qty";
    public static final String KNN_MEMORY_CIRCUIT_BREAKER_ENABLED = "knn.memory.circuit_breaker.enabled";
    public static final String KNN_MEMORY_CIRCUIT_BREAKER_CLUSTER_LIMIT = "knn.memory.circuit_breaker.limit";
    public static final String KNN_MEMORY_CIRCUIT_BREAKER_LIMIT_PREFIX = KNN_MEMORY_CIRCUIT_BREAKER_CLUSTER_LIMIT + ".";
    public static final String KNN_VECTOR_STREAMING_MEMORY_LIMIT_IN_MB = "knn.vector_streaming_memory.limit";
    public static final String KNN_CIRCUIT_BREAKER_TRIGGERED = "knn.circuit_breaker.triggered";
    public static final String KNN_CACHE_ITEM_EXPIRY_ENABLED = "knn.cache.item.expiry.enabled";
    public static final String KNN_CACHE_ITEM_EXPIRY_TIME_MINUTES = "knn.cache.item.expiry.minutes";
    public static final String KNN_CIRCUIT_BREAKER_UNSET_PERCENTAGE = "knn.circuit_breaker.unset.percentage";
    public static final String KNN_INDEX = "index.knn";
    public static final String MODEL_INDEX_NUMBER_OF_SHARDS = "knn.model.index.number_of_shards";
    public static final String MODEL_INDEX_NUMBER_OF_REPLICAS = "knn.model.index.number_of_replicas";
    public static final String MODEL_CACHE_SIZE_LIMIT = "knn.model.cache.size.limit";
    public static final String ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD = "index.knn.advanced.filtered_exact_search_threshold";
    public static final String KNN_FAISS_AVX2_DISABLED = "knn.faiss.avx2.disabled";
    public static final String QUANTIZATION_STATE_CACHE_SIZE_LIMIT = "knn.quantization.cache.size.limit";
    public static final String QUANTIZATION_STATE_CACHE_EXPIRY_TIME_MINUTES = "knn.quantization.cache.expiry.minutes";
    public static final String KNN_FAISS_AVX512_DISABLED = "knn.faiss.avx512.disabled";
    public static final String KNN_FAISS_AVX512_SPR_DISABLED = "knn.faiss.avx512_spr.disabled";
    public static final String KNN_DISK_VECTOR_SHARD_LEVEL_RESCORING_DISABLED = "index.knn.disk.vector.shard_level_rescoring_disabled";
    public static final String KNN_DERIVED_SOURCE_ENABLED = "index.knn.derived_source.enabled";
    // Remote index build index settings
    public static final String KNN_INDEX_REMOTE_VECTOR_BUILD = "index.knn.remote_index_build.enabled";
    public static final String KNN_INDEX_REMOTE_VECTOR_BUILD_SIZE_MIN = "index.knn.remote_index_build.size.min";
    // Remote index build cluster settings
    public static final String KNN_REMOTE_VECTOR_BUILD = "knn.remote_index_build.enabled";
    public static final String KNN_REMOTE_REPOSITORY = "knn.remote_index_build.repository";
    public static final String KNN_REMOTE_VECTOR_BUILD_SIZE_MAX = "knn.remote_index_build.size.max";
    public static final String KNN_REMOTE_BUILD_SERVICE_ENDPOINT = "knn.remote_index_build.service.endpoint";
    public static final String KNN_REMOTE_BUILD_POLL_INTERVAL = "knn.remote_index_build.poll.interval";
    public static final String KNN_REMOTE_BUILD_CLIENT_TIMEOUT = "knn.remote_index_build.client.timeout";
    public static final String KNN_REMOTE_BUILD_SERVICE_USERNAME = "knn.remote_index_build.service.username";
    public static final String KNN_REMOTE_BUILD_SERVICE_PASSWORD = "knn.remote_index_build.service.password";
    public static final String INDEX_KNN_FAISS_EFFICIENT_FILTER_DISABLE_EXACT_SEARCH =
        "index.knn.faiss.efficient_filter.disable_exact_search";
    // Cluster-level opt-in flag for the dynamic knn_vector mapping feature (auto-inference and the
    // match_mapping_type: knn_vector template handler). Off by default because the auto-inference
    // heuristic is shape-based and can misclaim non-vector numeric arrays; operators must opt in.
    public static final String KNN_DYNAMIC_MAPPING_ENABLED = "knn.dynamic_mapping.enabled";

    // Node level gate for routing faiss/MOS full-precision vector data to Direct I/O. Off by default:
    // with it off KNNDirectIODirectory returns the delegate's own input, so every read is byte for
    // byte what it is without this code. Read at every dispatch, so flipping it takes effect on the next
    // reader or values object with no index or node restart.
    public static final String KNN_DIRECT_IO_RESCORE_ENABLED = "knn.direct_io.rescore.enabled";
    public static final boolean KNN_DIRECT_IO_RESCORE_ENABLED_DEFAULT_VALUE = false;

    // The size floor on routing, and the one unambiguous defect phase 13's audit found (FIX 1). A .vec
    // below it is served by the delegate however the rest of the rule reads.
    //
    // The condition the feature actually wants is "this .vec's working set cannot stay in the page cache",
    // and that is a property of the host's memory and the shard's traffic, not of the file. Before this
    // setting the condition was asserted ONCE PER NODE by knn.direct_io.rescore.enabled and never checked
    // per file, and the measured cost of getting it wrong on one small file is three to four orders of
    // magnitude: phase 13 E4, same 16-thread random rescore workload, a 0.98 MiB flushed .vec served at
    // 64.7 ops/s with O_DIRECT against 518,681 ops/s with mmap -- 8,016x -- because mmap reads the file
    // from the device once and then answers from the page cache forever while O_DIRECT goes to the device
    // for every read, forever. At 2.86 GiB the two arms converge (59.5 vs 67.5) because the file no longer
    // fits in whatever cache the workload gets.
    //
    // A size floor does not capture residency. It removes the case where the answer is CERTAINLY wrong --
    // a flush-sized segment -- for one comparison. 10 MB is Lucene's own DEFAULT_MIN_BYTES_DIRECT rather
    // than a number measured here, and is called out as such: E4's two fixtures were 0.98 MiB and 58.6 MiB
    // and both lost on a 123 GiB desktop, so this desktop cannot locate the crossing point. Set 0b to
    // restore pre-phase-14 behaviour and route every size.
    public static final String KNN_DIRECT_IO_MIN_FILE_SIZE = "knn.direct_io.min_file_size";
    public static final ByteSizeValue KNN_DIRECT_IO_MIN_FILE_SIZE_DEFAULT_VALUE = new ByteSizeValue(10, ByteSizeUnit.MB);

    // Direct I/O prefetch settings. These are only consulted once Direct I/O is already engaged, i.e.
    // knn.direct_io.rescore.enabled is on and KNNDirectIODirectory routed a .vec through
    // DirectIOVectorIndexInput, so on a node with that setting off none of them has any effect.
    // Upper bound on in-flight reads across the whole node, not per segment: per-leaf scoring tasks run
    // concurrently, so total in-flight is segments x window, and a per-segment bound oversaturates the
    // device.
    //
    // DERIVED FROM THE CORE COUNT, not fixed at 64. The fixed 64 was sized to the device queue
    // (nr_requests = 63) on a host with cores to spare, and on a 4-vCPU node it is 16x oversubscribed:
    // 64 reader threads contend for 4 cores with the search threads that are waiting on them, and every
    // staged read then queues behind ~13 others (measured: a 16-thread concurrent probe on 4 pinned cores
    // held 654-857 tasks queued with all 64 threads active). clamp(2 x cores, 2, 16) keeps enough threads
    // to overlap reads on a small box without that contention, and still reaches a useful queue depth on a
    // large one. An operator who wants the old value sets the key explicitly; only the DEFAULT moves.
    public static final String KNN_DIRECT_IO_RESCORE_PREFETCH_THREADS = "knn.direct_io.rescore.prefetch_threads";
    public static final int KNN_DIRECT_IO_RESCORE_PREFETCH_THREADS_DEFAULT_VALUE = defaultDirectIORescorePrefetchThreads();

    /** Lower bound on the derived pool size: two threads is the smallest pool that can overlap two reads. */
    static final int KNN_DIRECT_IO_RESCORE_PREFETCH_THREADS_MIN = 2;

    /**
     * Upper bound on the derived pool size. 16 rather than the device queue's 63 because the queue depth
     * that helps is bounded by the device, not by the host: phase 9 measured a 200-wide fan-out at p90 2.2x
     * and p99 2.0x worse than a 64-wide one, and 16 readers x the staged ranges each one holds already
     * offers the device more than it can retire.
     */
    static final int KNN_DIRECT_IO_RESCORE_PREFETCH_THREADS_MAX = 16;

    /**
     * {@code clamp(2 x availableProcessors, 2, 16)} — two readers per core so a read can overlap the
     * scoring that follows it, bounded below so the pool can always overlap something and above so a large
     * host does not offer the device more than it can retire.
     */
    private static int defaultDirectIORescorePrefetchThreads() {
        final int cores = Runtime.getRuntime().availableProcessors();
        return Math.min(KNN_DIRECT_IO_RESCORE_PREFETCH_THREADS_MAX, Math.max(KNN_DIRECT_IO_RESCORE_PREFETCH_THREADS_MIN, 2 * cores));
    }

    // The waste guard on the directory design's read-ahead channel: the largest single prefetch span
    // DirectIOVectorIndexInput will put in flight. A span above it is declined, which is correct by the
    // prefetch contract and leaves the bytes to blocking reads that touch only the blocks a vector is in.
    //
    // It exists because the read-ahead channel is Lucene's IndexInput#prefetch, whose only arguments are
    // an offset and a length, and because the caller on this path -- PrefetchHelper#prefetchExactVectorSize
    // -- asks for the SPAN of a coalesced group of ordinals, gaps included, extending a group while
    // (vector end - group start) <= 128 KB. A group of two ordinals 120 KB apart is therefore one
    // 120 KB request for 6 KB of vectors, and the input cannot see which bytes inside the span will be
    // read: the gap structure is the caller's and is not in the arguments. So a guard here can only be a
    // bound on the span, never a bound on the waste, and 9e-2's R60 -- that the paced p99 tracks byte
    // volume (2.55x against a 2.50x p99 ratio) -- is what says whether bounding the span is worth its
    // cost in queue depth.
    //
    // The default is 128 KB + one 4 KB block, i.e. PrefetchHelper's own grouping budget plus the
    // alignment slack at each end, which is the value 9e-2 measured; at the default this setting changes
    // nothing. Lower it to trade read-ahead for byte economy.
    // How many prefetch ranges ONE Direct I/O vector input keeps in flight for one burst. This is a table
    // that has to HOLD a whole burst, and the burst's size is not ours to choose -- PrefetchHelper is
    // driven by Lucene's 64-ordinal bulk batch, so a table of 48 declines the tail of every burst and
    // those ranges fall back to blocking reads on the calling thread. (The rescore seam's removed
    // prefetch_window was a different quantity: a rolling ring refilled as it drained, swept to 48.)
    //
    // 64 on measurement (task-15, dio-1m, k=100, 200 queries x 3, cold, mincore-verified). At 48 the
    // route made 5.4 blocking reads per query and the paced p99 was 30 ms; at 64 it makes 0.0 and the
    // paced p99 is 11 ms, with byte volume UNCHANGED (2,581 vs 2,591 KiB/query) -- so the tail was
    // serialization, not traffic. Saturated, the same change is p99 31 -> 28 and median 10 -> 9. The
    // value is the batch size rather than a tuned number: raising it further cannot help, because
    // nothing asks for more than 64 ranges in one burst.
    public static final String KNN_DIRECT_IO_RESCORE_PREFETCH_STAGED_RANGES = "knn.direct_io.rescore.prefetch.staged_ranges";
    public static final int KNN_DIRECT_IO_RESCORE_PREFETCH_STAGED_RANGES_DEFAULT_VALUE = 64;

    // How many reads may WAIT behind the shared pool. Phase 13 audit row 7 and phase 14 S3 both measured
    // this as the binding constraint at concurrent-scorer load, and the Oct 4-5 node sweep settled it: at an
    // effective capacity of 64 the node refused 46-52% of staged reads and made 783k device reads per 5,000
    // queries at p50/p90/p99 777/1069/1322 ms; at 1,024 it refused 0% and made 707k at 650/877/954. The
    // declined bursts were falling back to individual blocking reads. Phase 14's claim that declining is the
    // cheaper fallback is WITHDRAWN on that evidence.
    //
    // 0 means "derive", and since phase 15 the derivation is BURST DEMAND rather than pool size:
    // max(256, nextPowerOfTwo((search_max + index_searcher_max) * staged_ranges)) in DirectIOReadPool --
    // 1,024 on a 4-vCPU node, 4,096 at 16. Any positive value is taken literally and still wins.
    //
    // DYNAMIC AND LIVE. ArrayBlockingQueue has no resize, so DirectIOReadPool SWAPS the executor when this
    // changes: a new pool at the new capacity is published and the old one is shutdown() so its queued reads
    // drain. Applied on the next pool acquisition, which is once per prefetch burst. Before phase 15 a change
    // to this key silently did nothing until a node restart.
    public static final String KNN_DIRECT_IO_RESCORE_PREFETCH_QUEUE_CAPACITY = "knn.direct_io.rescore.prefetch.queue_capacity";
    public static final int KNN_DIRECT_IO_RESCORE_PREFETCH_QUEUE_CAPACITY_DEFAULT_VALUE = 0;

    // Node-wide ceiling on the direct (off-heap) buffers the Direct I/O read path holds at once. Phase 13's
    // audit FIX 3, carried through phases 14 and 15: Lucene's OffHeapFloatVectorValues is not Closeable, so
    // close() never runs on a per-query slice and a staging table lives until the GC's Cleaner reclaims it.
    // There is no end-of-life signal to return a buffer at, so the footprint is proportional to live slices
    // with nothing bounding it. MEASURED on the bench node: 1.69 GB of live direct buffers against a
    // MaxDirectMemorySize of 2g.
    //
    // DEFAULT = HALF the JVM's actual direct-memory wall, where the wall is the operator's explicit
    // -XX:MaxDirectMemorySize when they pinned one and heapSize / 2 otherwise -- see directMemoryWall().
    // Half of the wall rather than all of it, because this plugin is not the only user of it: any other
    // plugin's direct allocations come out of the same budget, and a ceiling AT the wall would turn a
    // contained decline into a hard OutOfMemoryError: Direct buffer memory for whoever allocated next.
    //
    // D3b: this default was Runtime.maxMemory() / 4, which silently assumed the ergonomic wall. The bench
    // node PINS the wall -- -XX:MaxDirectMemorySize=2g with -Xmx24g (MEASURED from the live JVM's cmdline) --
    // so that default was 6 GB against a 2 GB wall, a ceiling 3x ABOVE the wall and therefore exactly the
    // OutOfMemoryError this comment warns about rather than a bound against it. Operators pin this flag
    // routinely. Reading the flag makes the default 1 GB there, and leaves it at maxMemory() / 4 when the
    // flag is unset.
    //
    // Past the budget prefetch DECLINES and the bytes are served by blocking reads, which is a path the
    // design already exercises on every queue-capacity decline. 0b disables the budget.
    public static final String KNN_DIRECT_IO_DIRECT_MEMORY_BUDGET = "knn.direct_io.direct_memory_budget";

    /**
     * The JVM-wide direct-memory wall this node actually runs against, resolved once at class load.
     *
     * @see #directMemoryWall(String, long)
     */
    public static final long KNN_DIRECT_IO_DIRECT_MEMORY_WALL_BYTES = directMemoryWall(
        readMaxDirectMemorySizeFlag(),
        Runtime.getRuntime().maxMemory()
    );

    public static final ByteSizeValue KNN_DIRECT_IO_DIRECT_MEMORY_BUDGET_DEFAULT_VALUE = new ByteSizeValue(
        KNN_DIRECT_IO_DIRECT_MEMORY_WALL_BYTES / 2,
        ByteSizeUnit.BYTES
    );

    // Node-wide cache of individual vectors on the Direct I/O random rescore path, in bytes of on-heap
    // payload plus per-entry overhead. 0b = OFF, which is the shipped default in phase 15: the mechanism and
    // its correctness proof ship, and the node arm decides the value.
    //
    // MEASURED 2026-10-05 on dio-10m, exact LRU stack-distance replay of 5,000 real OSB queries against the
    // exact rescore candidate set (1,000,000 refs over 544,060 unique vectors, a 1.70 GB working set):
    // steady hit rate 40.3% at 16mb, 40.9% at 64mb, 41.4% at 128mb, 42.3% at 256mb, 48.1% at 1gb, against a
    // ceiling of 45.6%. 16mb captures 84% of the achievable reuse, and replaying the same queries 3x does not
    // inflate it (128mb: 39.8% every pass), so it is inner-product HNSW hub reuse and not an artefact.
    //
    // The value is ON HEAP, deliberately: it therefore stays out of knn.direct_io.direct_memory_budget and
    // out of MaxDirectMemorySize, and is bounded by Guava's maximumWeight rather than by a GC.
    //
    // Dynamic, applied on the next prefetch burst. A change REBUILDS the cache (Guava fixes maximumWeight at
    // build time), so the entries are dropped and it warms again -- unlike the queue capacity, retuning this
    // is not free.
    public static final String KNN_DIRECT_IO_RESCORE_CACHE_MAX_BYTES = "knn.direct_io.rescore.cache.max_bytes";
    public static final ByteSizeValue KNN_DIRECT_IO_RESCORE_CACHE_MAX_BYTES_DEFAULT_VALUE = new ByteSizeValue(0, ByteSizeUnit.BYTES);

    // F5, span fill. While the vector cache is below its slab target, a batch also admits every whole vector
    // that happens to lie inside the spans it staged, not only the candidates it asked for.
    //
    // The measurement it exists for (ref/PILOT-REPORT.md section 4, MEASURED): on dio-1m with a 4 GB cache,
    // room for every vector in the file, Direct I/O's p50 stayed 9.7x mmap's because 16% of candidates still
    // missed after a 10,000-query warm-up. In that warm-up Direct I/O read 2.01 MB per query in ~40 KB staged
    // spans (20 GB, 6.5x the file) and cached only the requested vectors, while mmap's page cache kept every
    // page it read and ended holding 95% of the file. The cache was paying for the reads and keeping a
    // fraction of what they delivered.
    //
    // A boolean rather than a size, because the policy has no magnitude: at or above the target admission is
    // requested vectors only, exactly as before, so this cannot cause an eviction and cannot take a slab the
    // cache's target does not already allow. Default true, and a node can measure with and without it.
    public static final String KNN_DIRECT_IO_RESCORE_CACHE_SPAN_FILL = "knn.direct_io.rescore.cache.span_fill";
    public static final boolean KNN_DIRECT_IO_RESCORE_CACHE_SPAN_FILL_DEFAULT_VALUE = true;

    public static final String KNN_DIRECT_IO_RESCORE_PREFETCH_MAX_SPAN = "knn.direct_io.rescore.prefetch.max_span_bytes";
    public static final ByteSizeValue KNN_DIRECT_IO_RESCORE_PREFETCH_MAX_SPAN_DEFAULT_VALUE = new ByteSizeValue(
        128 * 1024 + 4096,
        ByteSizeUnit.BYTES
    );

    // D3d, phase 16-I I7. The wide-group buffers a DirectIOStagingLease keeps beside its arena: one per
    // staged_ranges slot, each grown on demand to at most max_span_bytes, kept for the life of the arena and
    // never shrunk. At the defaults that is 64 x 135,168 B = 8.25 MiB PER THREAD, and a lease is per thread
    // with no bound on the number of leases, so 225 search threads reach 1.95 GB (MEASURED as arithmetic in
    // phase 16-I FINDINGS I0.1; the buffers are grown lazily, so it is the worst case rather than the
    // common one).
    //
    // knn.direct_io.direct_memory_budget already counts them, but that is one pot shared with the arenas and
    // the read-ahead rings, and a wide buffer is held for the life of its lease while an arena is reused:
    // wide buffers accumulating to the budget would leave no room for an arena, and then EVERY prefetch
    // declines. A budget that wedges the feature is not a bound on it. So the wide buffers get a sub-budget
    // of their own, checked with the overall one and never instead of it.
    //
    // 0b means DERIVE it: one quarter of knn.direct_io.direct_memory_budget, so an operator who has already
    // sized that one does not have to size this. A quarter and not a half because the arenas are the
    // mechanism's common case (phase 12 S3.1 measured the mean aligned span at 7,310 B, i.e. nearly every
    // group fits an arena slot and needs no wide buffer at all) and they must always have room. With the
    // direct-memory budget itself off the derived value is off too, which keeps the rule the budget has
    // always had: a budget must never be the reason a shard cannot read its vectors.
    //
    // Past the cap bufferFor declines, the range is not staged, and the bytes are served by a blocking read
    // -- the same fallback a queue-capacity or budget decline takes, already exercised on every arm.
    public static final String KNN_DIRECT_IO_RESCORE_PREFETCH_WIDE_BUFFER_BUDGET = "knn.direct_io.rescore.prefetch.wide_buffer_budget";
    public static final ByteSizeValue KNN_DIRECT_IO_RESCORE_PREFETCH_WIDE_BUFFER_BUDGET_DEFAULT_VALUE = new ByteSizeValue(
        0,
        ByteSizeUnit.BYTES
    );

    // Sequential read-ahead for DirectIOVectorIndexInput, a mechanism SEPARATE from the burst table above.
    // The burst table (staged_ranges / max_span_bytes) serves the rescore seam, whose access is random and
    // bursty and declared by prefetch() calls. A sequential bulk consumer -- a merge, the faiss-unquantized
    // exact first pass, unquantized traversal -- issues no prefetch() at all, so every one of its reads is a
    // blocking QD1 pread one buffer at a time, slower than mmap's kernel read-ahead. These two settings add
    // an internal rolling ring that detects a gapless-forward run of blocking reads and puts the next
    // windows in flight, bringing sequential O_DIRECT toward mmap parity WITHOUT changing which bytes are
    // read. The ring stands down whenever a prefetch burst is active, so the rescore path is untouched.
    //
    // window: how many bufferSize-sized windows the ring keeps in flight ahead of the frontier. 0 disables
    // read-ahead entirely -- the input is then byte-for-byte and syscall-for-syscall what it is with this
    // code absent, which is the arm to compare against and the escape hatch if this ever misbehaves.
    //
    // 8 is measured, on a 732 MiB ascending scan of a real 1M x 768 fp32 .vec on EBS-backed nvme: it takes
    // block-layer queue depth from 1.02 to 6.73 and throughput from 13.4 to 49.6 MiB/s, a 3.7x, at an
    // amplification of 1.000 -- the ring replaces blocking reads rather than adding traffic, because every
    // window it fetches is one the scan was going to read anyway. 16 buys depth 14.2 and no throughput
    // (49.8), so 8 is the knee; 2 buys nothing at all (13.0), because the lead never gets far enough ahead
    // of the consumer to overlap. Costs W * bufferSize = 64 KiB of direct buffers per open, allocated
    // lazily on the first arm, so an input that never runs sequentially never pays it.
    public static final String KNN_DIRECT_IO_READAHEAD_WINDOW = "knn.direct_io.readahead.window";
    public static final int KNN_DIRECT_IO_READAHEAD_WINDOW_DEFAULT_VALUE = 8;

    // window_bytes: how LARGE each of those windows is. Phase 13 audit FIX 2, and the largest measured win
    // in that audit. Until phase 14 the ring's request size WAS bufferSize, so one number served two access
    // shapes with opposite needs. Measured (phase 13 e1b.out, 732 MiB ascending scan of a real 1M x 768
    // fp32 .vec, and a 16-thread random rescore arm on the same file):
    //
    // bufferSize | sequential MiB/s at W=8 | random amplification, 16 threads
    // 8 KiB | 49.7 | 2.667 <- shipped
    // 64 KiB | 371.2 | ~21 (INFERRED, see below)
    // 256 KiB | 370.6 | 85.3 <- measured
    //
    // Sequential wants >= 64 KiB: that is 7.5x, and it is the whole of the gap to mmap and past it (371.2
    // against mmap's 239.1). Random wants <= 8 KiB: at 256 KiB amplification is 85.3x and qps collapses
    // from 59.6 to 10.0. The 64 KiB random cell is INFERRED rather than measured -- amplification tracks
    // bufferSize / vectorBytes almost exactly at the two measured points (8 KiB -> 2.667 at a 3,072 B
    // vector; 256 KiB -> 85.3, a ratio of 32.0 against the size ratio of 32) -- and it is why the fix is
    // NOT to raise bufferSize.
    //
    // So the ring gets its own size and bufferSize stays 8 KiB for the random path. 64 KiB is the knee:
    // 256 KiB buys nothing sequentially (370.6 vs 371.2). Costs window * window_bytes of direct buffers per
    // input, allocated lazily on the first ARM -- 512 KiB at the defaults for an input that runs
    // sequentially, and zero for one that does not. The rescore path never arms, so it pays nothing.
    public static final String KNN_DIRECT_IO_READAHEAD_WINDOW_BYTES = "knn.direct_io.readahead.window_bytes";
    public static final ByteSizeValue KNN_DIRECT_IO_READAHEAD_WINDOW_BYTES_DEFAULT_VALUE = new ByteSizeValue(64, ByteSizeUnit.KB);
    // trigger: how many consecutive gapless-forward blocking reads must occur before the ring arms. This is
    // the no-waste guard: a random (rescore) workload must never cross it, so read-ahead issues ZERO
    // speculative reads on that pattern. A knob rather than a constant precisely because that no-waste claim
    // is a claim about this value, and an operator who distrusts it must be able to raise it without a build.
    public static final String KNN_DIRECT_IO_READAHEAD_TRIGGER = "knn.direct_io.readahead.trigger";
    public static final int KNN_DIRECT_IO_READAHEAD_TRIGGER_DEFAULT_VALUE = 2;

    /**
     * For more details on supported engines, refer to {@link MemoryOptimizedSearchSupportSpec}
     */
    public static final String MEMORY_OPTIMIZED_KNN_SEARCH_MODE = "index.knn.memory_optimized_search";
    public static final boolean DEFAULT_MEMORY_OPTIMIZED_KNN_SEARCH_MODE = false;

    /**
     * Default setting values
     *
     */
    public static final boolean KNN_DEFAULT_FAISS_AVX2_DISABLED_VALUE = false;
    public static final boolean KNN_DEFAULT_FAISS_AVX512_DISABLED_VALUE = false;
    public static final boolean KNN_DEFAULT_FAISS_AVX512_SPR_DISABLED_VALUE = false;
    public static final String INDEX_KNN_DEFAULT_SPACE_TYPE = "l2";
    public static final Integer INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD_DEFAULT_VALUE = 0;
    public static final Integer INDEX_KNN_BUILD_VECTOR_DATA_STRUCTURE_THRESHOLD_MIN = -1;
    public static final Integer INDEX_KNN_BUILD_VECTOR_DATA_STRUCTURE_THRESHOLD_MAX = Integer.MAX_VALUE - 2;
    public static final Integer INDEX_KNN_DEFAULT_ALGO_PARAM_M = 16;
    public static final Integer INDEX_KNN_DEFAULT_ALGO_PARAM_EF_SEARCH = 100;
    public static final Integer INDEX_KNN_DEFAULT_ALGO_PARAM_EF_CONSTRUCTION = 100;
    public static final Integer KNN_DEFAULT_CIRCUIT_BREAKER_UNSET_PERCENTAGE = 75;
    public static final Integer KNN_DEFAULT_MODEL_CACHE_SIZE_LIMIT_PERCENTAGE = 10; // By default, set aside 10% of the JVM for the limit
    public static final Integer KNN_MAX_MODEL_CACHE_SIZE_LIMIT_PERCENTAGE = 25; // Model cache limit cannot exceed 25% of the JVM heap
    public static final String KNN_DEFAULT_MEMORY_CIRCUIT_BREAKER_LIMIT = "50%";
    public static final String KNN_DEFAULT_VECTOR_STREAMING_MEMORY_LIMIT_PCT = "1%";

    public static final Integer ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD_DEFAULT_VALUE = -1;
    public static final Integer KNN_DEFAULT_QUANTIZATION_STATE_CACHE_SIZE_LIMIT_PERCENTAGE = 5; // By default, set aside 5% of the JVM for
    // the limit
    public static final Integer KNN_MAX_QUANTIZATION_STATE_CACHE_SIZE_LIMIT_PERCENTAGE = 10; // Quantization state cache limit cannot exceed
    // 10% of the JVM heap
    public static final Integer KNN_DEFAULT_QUANTIZATION_STATE_CACHE_EXPIRY_TIME_MINUTES = 60;
    public static final boolean KNN_DISK_VECTOR_SHARD_LEVEL_RESCORING_DISABLED_VALUE = false;
    public static final ByteSizeValue KNN_REMOTE_VECTOR_BUILD_SIZE_LIMIT_DEFAULT_VALUE = new ByteSizeValue(0, ByteSizeUnit.MB);
    // TODO: Tune this default value based on benchmarking
    public static final ByteSizeValue KNN_INDEX_REMOTE_VECTOR_BUILD_THRESHOLD_DEFAULT_VALUE = new ByteSizeValue(50, ByteSizeUnit.MB);

    // TODO: Tune these default values based on benchmarking
    public static final Integer KNN_DEFAULT_REMOTE_BUILD_CLIENT_TIMEOUT_MINUTES = 60;
    public static final Integer KNN_DEFAULT_REMOTE_BUILD_CLIENT_POLL_INTERVAL_SECONDS = 5;

    /**
     * Settings Definition
     */

    /**
     * This setting controls whether shard-level re-scoring for KNN disk-based vectors is turned off.
     * The setting uses:
     * <ul>
     *     <li><b>KNN_DISK_VECTOR_SHARD_LEVEL_RESCORING_DISABLED:</b> The name of the setting.</li>
     *     <li><b>KNN_DISK_VECTOR_SHARD_LEVEL_RESCORING_DISABLED_VALUE:</b> The default value (true or false).</li>
     *     <li><b>IndexScope:</b> The setting works at the index level.</li>
     *     <li><b>Dynamic:</b> This setting can be changed without restarting the cluster.</li>
     * </ul>
     *
     * @see Setting#boolSetting(String, boolean, Setting.Property...)
     */
    public static final Setting<Boolean> KNN_DISK_VECTOR_SHARD_LEVEL_RESCORING_DISABLED_SETTING = Setting.boolSetting(
        KNN_DISK_VECTOR_SHARD_LEVEL_RESCORING_DISABLED,
        KNN_DISK_VECTOR_SHARD_LEVEL_RESCORING_DISABLED_VALUE,
        IndexScope,
        Dynamic
    );

    // This setting controls how much memory should be used to transfer vectors from Java to JNI Layer. The default
    // 1% of the JVM heap
    public static final Setting<ByteSizeValue> KNN_VECTOR_STREAMING_MEMORY_LIMIT_PCT_SETTING = Setting.memorySizeSetting(
        KNN_VECTOR_STREAMING_MEMORY_LIMIT_IN_MB,
        KNN_DEFAULT_VECTOR_STREAMING_MEMORY_LIMIT_PCT,
        Setting.Property.Dynamic,
        Setting.Property.NodeScope
    );

    /**
     * build_vector_data_structure_threshold - This parameter determines when to build vector data structure for knn fields during indexing
     * and merging. Setting -1 (min) will skip building graph, whereas on any other values, the graph will be built if
     * number of live docs in segment is greater than this threshold. Since max number of documents in a segment can
     * be Integer.MAX_VALUE - 1, this setting will allow threshold to be up to 1 less than max number of documents in a segment
     */
    public static final Setting<Integer> INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD_SETTING = Setting.intSetting(
        INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD,
        INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD_DEFAULT_VALUE,
        INDEX_KNN_BUILD_VECTOR_DATA_STRUCTURE_THRESHOLD_MIN,
        INDEX_KNN_BUILD_VECTOR_DATA_STRUCTURE_THRESHOLD_MAX,
        IndexScope,
        Dynamic
    );

    /**
     *  ef or efSearch - the size of the dynamic list for the nearest neighbors (used during the search).
     *  Higher ef leads to more accurate but slower search. ef cannot be set lower than the number of queried nearest neighbors k.
     *  The value ef can be anything between k and the size of the dataset.
     */
    public static final Setting<Integer> INDEX_KNN_ALGO_PARAM_EF_SEARCH_SETTING = Setting.intSetting(
        KNN_ALGO_PARAM_EF_SEARCH,
        INDEX_KNN_DEFAULT_ALGO_PARAM_EF_SEARCH,
        2,
        IndexScope,
        Dynamic
    );

    public static final Setting<Integer> MODEL_INDEX_NUMBER_OF_SHARDS_SETTING = Setting.intSetting(
        MODEL_INDEX_NUMBER_OF_SHARDS,
        1,
        1,
        Setting.Property.NodeScope,
        Setting.Property.Dynamic
    );

    public static final Setting<Integer> MODEL_INDEX_NUMBER_OF_REPLICAS_SETTING = Setting.intSetting(
        MODEL_INDEX_NUMBER_OF_REPLICAS,
        1,
        0,
        Setting.Property.NodeScope,
        Setting.Property.Dynamic
    );

    public static final Setting<Integer> ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD_SETTING = Setting.intSetting(
        ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD,
        ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD_DEFAULT_VALUE,
        IndexScope,
        Setting.Property.Dynamic
    );

    public static final Setting<ByteSizeValue> MODEL_CACHE_SIZE_LIMIT_SETTING = new Setting<>(
        MODEL_CACHE_SIZE_LIMIT,
        percentageAsString(KNN_DEFAULT_MODEL_CACHE_SIZE_LIMIT_PERCENTAGE),
        (s) -> {
            ByteSizeValue userDefinedLimit = parseBytesSizeValueOrHeapRatio(s, MODEL_CACHE_SIZE_LIMIT);

            // parseBytesSizeValueOrHeapRatio will make sure that the value entered falls between 0 and 100% of the
            // JVM heap. However, we want the maximum percentage of the heap to be much smaller. So, we add
            // some additional validation here before returning
            ByteSizeValue jvmHeapSize = JvmInfo.jvmInfo().getMem().getHeapMax();
            if ((userDefinedLimit.getKbFrac() / jvmHeapSize.getKbFrac()) > percentageAsFraction(
                KNN_MAX_MODEL_CACHE_SIZE_LIMIT_PERCENTAGE
            )) {
                throw new OpenSearchParseException(
                    "{} ({} KB) cannot exceed {}% of the heap ({} KB).",
                    MODEL_CACHE_SIZE_LIMIT,
                    userDefinedLimit.getKb(),
                    KNN_MAX_MODEL_CACHE_SIZE_LIMIT_PERCENTAGE,
                    jvmHeapSize.getKb()
                );
            }

            return userDefinedLimit;
        },
        Setting.Property.NodeScope,
        Setting.Property.Dynamic
    );

    /**
     * This setting identifies KNN index.
     */
    public static final Setting<Boolean> IS_KNN_INDEX_SETTING = Setting.boolSetting(
        KNN_INDEX,
        false,
        IndexScope,
        Final,
        UnmodifiableOnRestore
    );

    public static final Setting<Boolean> KNN_DERIVED_SOURCE_ENABLED_SETTING = new Setting<>(
        KNN_DERIVED_SOURCE_ENABLED,
        (s) -> Boolean.toString(false),
        (b) -> Booleans.parseBooleanStrict(b, false),
        IndexScope,
        Final,
        UnmodifiableOnRestore
    ) {
        @Override
        public Set<SettingDependency> getSettingsDependencies(String key) {
            return Set.of(new SettingDependency() {
                @Override
                public Setting<Boolean> getSetting() {
                    return IS_KNN_INDEX_SETTING;
                }

                @Override
                public void validate(String key, Object value, Object dependency) {
                    if (dependency instanceof Boolean isKnnEnabled && isKnnEnabled == false) {
                        throw new IllegalArgumentException("Index setting \"index.knn\" must be true in order to enabled derived source");
                    }
                }
            });
        }
    };

    public static final Setting<Boolean> MEMORY_OPTIMIZED_KNN_SEARCH_MODE_SETTING = Setting.boolSetting(
        MEMORY_OPTIMIZED_KNN_SEARCH_MODE,
        false,
        IndexScope
    );

    /**
     * Node level switch for reading faiss and memory-optimized-search full-precision vectors with Direct
     * I/O, at the storage layer in {@link org.opensearch.knn.index.store.KNNDirectIODirectory}. Read
     * at every dispatch and needs no index close and reopen, so flipping it takes effect on the next
     * reader or values object.
     * <p>
     * This is the <b>whole</b> opt-in since phase 14: engagement is the codec seam
     * ({@code KNNDirectIODirectory#routeVectorData}) and there is no per-index setting to set. Off by
     * default, and with it off every read is the one every existing measurement was taken against, byte
     * for byte.
     */
    public static final Setting<Boolean> KNN_DIRECT_IO_RESCORE_ENABLED_SETTING = Setting.boolSetting(
        KNN_DIRECT_IO_RESCORE_ENABLED,
        KNN_DIRECT_IO_RESCORE_ENABLED_DEFAULT_VALUE,
        NodeScope,
        Dynamic
    );

    /**
     * The size a faiss/MOS {@code .vec} must reach before it is routed to Direct I/O — condition (iii) of
     * {@code KNNDirectIODirectory}'s guard. Node scoped and dynamic, read at every dispatch, so it
     * takes effect on the next reader.
     * <p>
     * Phase 13's audit called the absence of this term the one unambiguous defect in the design:
     * {@code O_DIRECT} on a flush-sized {@code .vec} measured <b>8,016x</b> slower than the page cache.
     * {@code 0b} disables the floor and restores pre-phase-14 routing.
     */
    public static final Setting<ByteSizeValue> KNN_DIRECT_IO_MIN_FILE_SIZE_SETTING = Setting.byteSizeSetting(
        KNN_DIRECT_IO_MIN_FILE_SIZE,
        KNN_DIRECT_IO_MIN_FILE_SIZE_DEFAULT_VALUE,
        NodeScope,
        Dynamic
    );

    /**
     * Node level size of the shared pool that performs the rescore seam's reads, and so the bound on
     * in-flight Direct I/O reads for the whole node. Threads are created on demand and reaped when idle,
     * so on a node whose rescore traffic never engages the seam the pool costs nothing.
     * <p>
     * The <b>default</b> is {@code clamp(2 x availableProcessors, 2, 16)} rather than a fixed 64, so that
     * the pool scales down to a small node instead of oversubscribing it 16x. The bounds of the setting
     * itself are unchanged, so an operator who has set a value keeps it.
     */
    public static final Setting<Integer> KNN_DIRECT_IO_RESCORE_PREFETCH_THREADS_SETTING = Setting.intSetting(
        KNN_DIRECT_IO_RESCORE_PREFETCH_THREADS,
        KNN_DIRECT_IO_RESCORE_PREFETCH_THREADS_DEFAULT_VALUE,
        1,
        1024,
        NodeScope,
        Dynamic
    );

    /**
     * The waste guard on the read-ahead channel of the directory design: the largest single
     * {@code IndexInput#prefetch} span {@code DirectIOVectorIndexInput} will stage. Read once per
     * {@code .vec} open, so it takes effect on the next open of the index rather than on the next query.
     * <p>
     * Dynamic and node scoped for the same reason the cache budget is: it is the knob an A/B of the
     * read-ahead economy turns, and turning it must not need a rebuild. At its default it is inert.
     */
    /**
     * Ranges one Direct I/O vector input keeps in flight per prefetch burst. Node scoped and dynamic,
     * read once per {@code .vec} open, so it takes effect on the next open of the index.
     * <p>
     * Lower bound 1 rather than 48-ish: a node that wants to measure the serial case should be able to
     * ask for it, and every value is correct — declining a prefetch only costs read-ahead.
     */
    public static final Setting<Integer> KNN_DIRECT_IO_RESCORE_PREFETCH_STAGED_RANGES_SETTING = Setting.intSetting(
        KNN_DIRECT_IO_RESCORE_PREFETCH_STAGED_RANGES,
        KNN_DIRECT_IO_RESCORE_PREFETCH_STAGED_RANGES_DEFAULT_VALUE,
        1,
        NodeScope,
        Dynamic
    );

    /**
     * Reads that may wait behind the shared Direct I/O pool. Node scoped and dynamic; read once, when the
     * pool is created, because {@link java.util.concurrent.ArrayBlockingQueue} cannot be resized.
     * <p>
     * {@code 0} means <b>derive</b>, which is the shipped behaviour and therefore the default: phase 14
     * deliberately does not move it. This exists so the node can sweep the one constraint phase 13's audit
     * row 7 measured as binding — 44% of declared reads refused at 16 concurrent scorers — without a
     * rebuild. A refused submit is the cheaper fallback, not a failure, so every value here is correct.
     */
    public static final Setting<Integer> KNN_DIRECT_IO_RESCORE_PREFETCH_QUEUE_CAPACITY_SETTING = Setting.intSetting(
        KNN_DIRECT_IO_RESCORE_PREFETCH_QUEUE_CAPACITY,
        KNN_DIRECT_IO_RESCORE_PREFETCH_QUEUE_CAPACITY_DEFAULT_VALUE,
        0,
        NodeScope,
        Dynamic
    );

    public static final Setting<ByteSizeValue> KNN_DIRECT_IO_RESCORE_PREFETCH_MAX_SPAN_SETTING = Setting.byteSizeSetting(
        KNN_DIRECT_IO_RESCORE_PREFETCH_MAX_SPAN,
        KNN_DIRECT_IO_RESCORE_PREFETCH_MAX_SPAN_DEFAULT_VALUE,
        NodeScope,
        Dynamic
    );

    /**
     * The node-wide ceiling on the wide-group buffers a {@code DirectIOStagingLease} holds beside its arena,
     * past which the range is not staged and its bytes are served by a blocking read. D3d, phase 16-I I7.
     *
     * <p><b>{@code 0b} derives it</b> as a quarter of {@code knn.direct_io.direct_memory_budget} rather than
     * disabling it, which is the opposite of what {@code 0b} means for that setting. See the comment on the
     * key for why these buffers need a sub-budget at all when the overall one already counts them.
     */
    public static final Setting<ByteSizeValue> KNN_DIRECT_IO_RESCORE_PREFETCH_WIDE_BUFFER_BUDGET_SETTING = Setting.byteSizeSetting(
        KNN_DIRECT_IO_RESCORE_PREFETCH_WIDE_BUFFER_BUDGET,
        KNN_DIRECT_IO_RESCORE_PREFETCH_WIDE_BUFFER_BUDGET_DEFAULT_VALUE,
        NodeScope,
        Dynamic
    );

    /**
     * The node-wide ceiling on direct buffers the Direct I/O read path holds at once, past which a prefetch
     * declines and the bytes are served by blocking reads. Phase 13's audit FIX 3.
     *
     * <p><b>Why half the wall.</b> The default is
     * {@link #KNN_DIRECT_IO_DIRECT_MEMORY_WALL_BYTES} / 2, where the wall is the operator's explicit
     * {@code -XX:MaxDirectMemorySize} when they pinned one and {@code heapSize / 2} otherwise (see
     * {@link #directMemoryWall(String, long)}). Half of the wall rather than all of it, because the wall is
     * shared: every other plugin's direct allocations are charged to the same
     * {@code MaxDirectMemorySize}, and a ceiling set at the wall would convert a contained decline here into
     * an {@code OutOfMemoryError: Direct buffer memory} for whoever allocated next. MEASURED on the bench
     * node before this existed: 1.69 GB live against a 2g wall.
     *
     * <p><b>Why the wall is read rather than assumed (D3b).</b> This default was
     * {@code Runtime.maxMemory() / 4}, which is half the wall only on a node that let the launcher choose
     * it. The bench node pins it: {@code -XX:MaxDirectMemorySize=2g} with {@code -Xmx24g} (MEASURED from
     * the live JVM's command line), where {@code maxMemory() / 4} is 6 GB — a ceiling <em>three times
     * above</em> the 2 GB wall, which is not a bound at all but precisely the {@code OutOfMemoryError} case
     * the paragraph above names. Pinning this flag is routine. Reading it makes the default 1 GB there, and
     * changes nothing on a node that did not pin it.
     *
     * <p>{@code 0b} disables the budget, which is the pre-phase-15 behaviour.
     */
    public static final Setting<ByteSizeValue> KNN_DIRECT_IO_DIRECT_MEMORY_BUDGET_SETTING = Setting.byteSizeSetting(
        KNN_DIRECT_IO_DIRECT_MEMORY_BUDGET,
        KNN_DIRECT_IO_DIRECT_MEMORY_BUDGET_DEFAULT_VALUE,
        NodeScope,
        Dynamic
    );

    /**
     * The node-wide cache of individual vectors on the Direct I/O random rescore path. <b>Default
     * {@code 0b}, i.e. off.</b>
     *
     * <p>The gate for building it at all was measured before it was built: 5,000 real OSB queries replayed
     * against the exact rescore candidate set at 10M vectors give a steady hit rate of <b>40.3% at 16 MB</b>
     * and 41.4% at 128 MB, against a ceiling of 45.6%, and replaying the same queries three times does not
     * inflate it. Every phase 12-14 arm ran with no cache at all, including the store-factory arm they were
     * compared against, so this is new information rather than a restoration.
     *
     * <p>Off by default because the shipped value is the node arm's to decide; this phase ships the
     * mechanism and the correctness proof (the re-proof is 8/8 identical with the cache on and off).
     *
     * <p>The cached bytes are <b>on heap</b>, so they are outside
     * {@code knn.direct_io.direct_memory_budget} and outside {@code MaxDirectMemorySize}, and are bounded by
     * Guava's {@code maximumWeight} rather than by a collection. Applied on the next prefetch burst; a
     * change <b>rebuilds</b> the cache, because Guava fixes {@code maximumWeight} at build time, so the
     * entries are dropped and it warms again.
     */
    public static final Setting<ByteSizeValue> KNN_DIRECT_IO_RESCORE_CACHE_MAX_BYTES_SETTING = Setting.byteSizeSetting(
        KNN_DIRECT_IO_RESCORE_CACHE_MAX_BYTES,
        KNN_DIRECT_IO_RESCORE_CACHE_MAX_BYTES_DEFAULT_VALUE,
        NodeScope,
        Dynamic
    );

    /**
     * Whether a batch also admits the whole vectors that happen to lie inside the spans it staged, while the
     * vector cache is below its slab target. Node scoped and dynamic, read once per scored batch, so it
     * takes effect on the next batch with no reopen. See the key's declaration for the measurement.
     */
    public static final Setting<Boolean> KNN_DIRECT_IO_RESCORE_CACHE_SPAN_FILL_SETTING = Setting.boolSetting(
        KNN_DIRECT_IO_RESCORE_CACHE_SPAN_FILL,
        KNN_DIRECT_IO_RESCORE_CACHE_SPAN_FILL_DEFAULT_VALUE,
        NodeScope,
        Dynamic
    );

    /**
     * How many {@code bufferSize}-sized windows {@code DirectIOVectorIndexInput}'s sequential read-ahead
     * ring keeps in flight ahead of the file-order frontier. Node scoped and dynamic, read once per
     * {@code .vec} open. Lower bound 0, which disables the ring and makes the input exact passthrough.
     * Defaults to 8 on the measurement recorded above.
     */
    public static final Setting<Integer> KNN_DIRECT_IO_READAHEAD_WINDOW_SETTING = Setting.intSetting(
        KNN_DIRECT_IO_READAHEAD_WINDOW,
        KNN_DIRECT_IO_READAHEAD_WINDOW_DEFAULT_VALUE,
        0,
        NodeScope,
        Dynamic
    );

    /**
     * Bytes per read-ahead window — the ring's request size, decoupled from {@code bufferSize} in phase 14
     * (audit FIX 2). Node scoped and dynamic, read once per {@code .vec} open. Rounded up to a filesystem
     * block and never applied below {@code bufferSize}, so it can only widen the ring.
     * <p>
     * Defaults to 64 KB on the measurement recorded above: sequential throughput 49.7 -> 371.2 MiB/s, which
     * is the whole of the gap to mmap and past it, at unchanged random amplification because the random path
     * keeps {@code bufferSize}.
     */
    public static final Setting<ByteSizeValue> KNN_DIRECT_IO_READAHEAD_WINDOW_BYTES_SETTING = Setting.byteSizeSetting(
        KNN_DIRECT_IO_READAHEAD_WINDOW_BYTES,
        KNN_DIRECT_IO_READAHEAD_WINDOW_BYTES_DEFAULT_VALUE,
        NodeScope,
        Dynamic
    );

    /**
     * Consecutive gapless-forward blocking reads needed before the read-ahead ring arms. Node scoped and
     * dynamic, read once per {@code .vec} open. Lower bound 1 so a node can force arm-on-first-read to test
     * the ring in isolation; the no-waste property comes from the default of 2, which random access cannot
     * reach.
     */
    public static final Setting<Integer> KNN_DIRECT_IO_READAHEAD_TRIGGER_SETTING = Setting.intSetting(
        KNN_DIRECT_IO_READAHEAD_TRIGGER,
        KNN_DIRECT_IO_READAHEAD_TRIGGER_DEFAULT_VALUE,
        1,
        NodeScope,
        Dynamic
    );

    /**
     * index_thread_quantity - the parameter specifies how many threads the nms library should use to create the graph.
     * By default, the nms library sets this value to NUM_CORES. However, because ES can spawn NUM_CORES threads for
     * indexing, and each indexing thread calls the NMS library to build the graph, which can also spawn NUM_CORES threads,
     * this could lead to NUM_CORES^2 threads running and could lead to 100% CPU utilization. This setting allows users to
     * configure number of threads for graph construction.
     */
    public static final Setting<Integer> KNN_ALGO_PARAM_INDEX_THREAD_QTY_SETTING = new Setting<>(
        KNN_ALGO_PARAM_INDEX_THREAD_QTY,
        settings -> Integer.toString(getHardwareDefaultIndexThreadQty(settings)),
        s -> {
            int value = Integer.parseInt(s);
            if (value < 1 || value > INDEX_THREAD_QTY_MAX) {
                throw new IllegalArgumentException("Value must be between 1 and " + INDEX_THREAD_QTY_MAX);
            }
            return value;
        },
        NodeScope,
        Dynamic
    );

    public static final Setting<Boolean> KNN_CIRCUIT_BREAKER_TRIGGERED_SETTING = Setting.boolSetting(
        KNN_CIRCUIT_BREAKER_TRIGGERED,
        false,
        NodeScope,
        Dynamic
    );

    public static final Setting<Double> KNN_CIRCUIT_BREAKER_UNSET_PERCENTAGE_SETTING = Setting.doubleSetting(
        KNN_CIRCUIT_BREAKER_UNSET_PERCENTAGE,
        KNN_DEFAULT_CIRCUIT_BREAKER_UNSET_PERCENTAGE,
        0,
        100,
        NodeScope,
        Dynamic
    );

    public static final Setting<Boolean> KNN_FAISS_AVX2_DISABLED_SETTING = Setting.boolSetting(
        KNN_FAISS_AVX2_DISABLED,
        KNN_DEFAULT_FAISS_AVX2_DISABLED_VALUE,
        NodeScope
    );

    /*
     * Quantization state cache settings
     */
    public static final Setting<ByteSizeValue> QUANTIZATION_STATE_CACHE_SIZE_LIMIT_SETTING = new Setting<ByteSizeValue>(
        QUANTIZATION_STATE_CACHE_SIZE_LIMIT,
        percentageAsString(KNN_DEFAULT_QUANTIZATION_STATE_CACHE_SIZE_LIMIT_PERCENTAGE),
        (s) -> {
            ByteSizeValue userDefinedLimit = parseBytesSizeValueOrHeapRatio(s, QUANTIZATION_STATE_CACHE_SIZE_LIMIT);

            // parseBytesSizeValueOrHeapRatio will make sure that the value entered falls between 0 and 100% of the
            // JVM heap. However, we want the maximum percentage of the heap to be much smaller. So, we add
            // some additional validation here before returning
            ByteSizeValue jvmHeapSize = JvmInfo.jvmInfo().getMem().getHeapMax();
            if ((userDefinedLimit.getKbFrac() / jvmHeapSize.getKbFrac()) > percentageAsFraction(
                KNN_MAX_QUANTIZATION_STATE_CACHE_SIZE_LIMIT_PERCENTAGE
            )) {
                throw new OpenSearchParseException(
                    "{} ({} KB) cannot exceed {}% of the heap ({} KB).",
                    QUANTIZATION_STATE_CACHE_SIZE_LIMIT,
                    userDefinedLimit.getKb(),
                    KNN_MAX_QUANTIZATION_STATE_CACHE_SIZE_LIMIT_PERCENTAGE,
                    jvmHeapSize.getKb()
                );
            }

            return userDefinedLimit;
        },
        NodeScope,
        Dynamic
    );

    public static final Setting<TimeValue> QUANTIZATION_STATE_CACHE_EXPIRY_TIME_MINUTES_SETTING = Setting.positiveTimeSetting(
        QUANTIZATION_STATE_CACHE_EXPIRY_TIME_MINUTES,
        TimeValue.timeValueMinutes(KNN_DEFAULT_QUANTIZATION_STATE_CACHE_EXPIRY_TIME_MINUTES),
        NodeScope,
        Dynamic
    );

    public static final Setting<Boolean> KNN_FAISS_AVX512_DISABLED_SETTING = Setting.boolSetting(
        KNN_FAISS_AVX512_DISABLED,
        KNN_DEFAULT_FAISS_AVX512_DISABLED_VALUE,
        NodeScope
    );

    public static final Setting<Boolean> KNN_FAISS_AVX512_SPR_DISABLED_SETTING = Setting.boolSetting(
        KNN_FAISS_AVX512_SPR_DISABLED,
        KNN_DEFAULT_FAISS_AVX512_SPR_DISABLED_VALUE,
        NodeScope
    );

    /**
     * Cluster level setting to control whether remote index build is enabled or not.
     */
    public static final Setting<Boolean> KNN_REMOTE_VECTOR_BUILD_SETTING = Setting.boolSetting(
        KNN_REMOTE_VECTOR_BUILD,
        false,
        NodeScope,
        Dynamic
    );

    /**
     * Index level setting to control whether remote index build is enabled or not.
     */
    public static final Setting<Boolean> KNN_INDEX_REMOTE_VECTOR_BUILD_SETTING = Setting.boolSetting(
        KNN_INDEX_REMOTE_VECTOR_BUILD,
        true,
        Dynamic,
        IndexScope
    );

    /**
     * Cluster level setting which indicates the repository that the remote index build should write to.
     */
    public static final Setting<String> KNN_REMOTE_VECTOR_REPOSITORY_SETTING = Setting.simpleString(
        KNN_REMOTE_REPOSITORY,
        Dynamic,
        NodeScope
    );

    /**
     * Index level setting which indicates the size threshold above which remote vector builds will be enabled.
     */
    public static final Setting<ByteSizeValue> KNN_INDEX_REMOTE_VECTOR_BUILD_SIZE_MIN_SETTING = Setting.byteSizeSetting(
        KNN_INDEX_REMOTE_VECTOR_BUILD_SIZE_MIN,
        KNN_INDEX_REMOTE_VECTOR_BUILD_THRESHOLD_DEFAULT_VALUE,
        Dynamic,
        IndexScope
    );

    /**
     * Cluster level setting which sets an upper bound on the remote vector build segment size.
     * This is the upper bound to {@link KNNSettings#KNN_INDEX_REMOTE_VECTOR_BUILD_SIZE_MIN_SETTING}.
     *
     * Defaults to 0, which means no upper bound, and can be set by users according to their remote vector index build service implementation.
     */
    public static final Setting<ByteSizeValue> KNN_REMOTE_VECTOR_BUILD_SIZE_MAX_SETTING = Setting.byteSizeSetting(
        KNN_REMOTE_VECTOR_BUILD_SIZE_MAX,
        KNN_REMOTE_VECTOR_BUILD_SIZE_LIMIT_DEFAULT_VALUE,
        Dynamic,
        NodeScope
    );

    /**
     * Remote build service endpoint to be used for remote index build.
     */
    public static final Setting<String> KNN_REMOTE_BUILD_SERVICE_ENDPOINT_SETTING = Setting.simpleString(
        KNN_REMOTE_BUILD_SERVICE_ENDPOINT,
        NodeScope,
        Dynamic
    );

    /**
     * Time the remote build service client will wait before falling back to CPU index build.
     */
    public static final Setting<TimeValue> KNN_REMOTE_BUILD_CLIENT_TIMEOUT_SETTING = Setting.timeSetting(
        KNN_REMOTE_BUILD_CLIENT_TIMEOUT,
        TimeValue.timeValueMinutes(KNN_DEFAULT_REMOTE_BUILD_CLIENT_TIMEOUT_MINUTES),
        NodeScope,
        Dynamic
    );

    /**
     * Setting to control how often the remote build service client polls the build service for the status of the job.
     */
    public static final Setting<TimeValue> KNN_REMOTE_BUILD_POLL_INTERVAL_SETTING = Setting.timeSetting(
        KNN_REMOTE_BUILD_POLL_INTERVAL,
        TimeValue.timeValueSeconds(KNN_DEFAULT_REMOTE_BUILD_CLIENT_POLL_INTERVAL_SECONDS),
        NodeScope,
        Dynamic
    );

    /**
     * Cluster-level toggle for the dynamic knn_vector mapping feature. When {@code false} (default),
     * both the auto-inference path ({@link org.opensearch.knn.plugin.KNNDynamicFieldTypeInferencer})
     * and the {@code match_mapping_type: knn_vector} dynamic-template handler
     * ({@link org.opensearch.knn.index.mapper.KNNDynamicTemplateTypeHandler}) short-circuit and defer
     * to core's default mapping behavior. Operators opt in per cluster to trade shape-based inference
     * risk for zero-config vector fields.
     */
    public static final Setting<Boolean> KNN_DYNAMIC_MAPPING_ENABLED_SETTING = Setting.boolSetting(
        KNN_DYNAMIC_MAPPING_ENABLED,
        false,
        NodeScope,
        Dynamic
    );

    public static final Setting<Boolean> INDEX_KNN_FAISS_EFFICIENT_FILTER_DISABLE_EXACT_SEARCH_SETTING = Setting.boolSetting(
        INDEX_KNN_FAISS_EFFICIENT_FILTER_DISABLE_EXACT_SEARCH,
        false,
        IndexScope,
        Dynamic
    );

    /**
     * Keystore settings for build service HTTP authorization
     */
    public static final Setting<SecureString> KNN_REMOTE_BUILD_SERVER_USERNAME_SETTING = SecureSetting.secureString(
        KNN_REMOTE_BUILD_SERVICE_USERNAME,
        null
    );
    public static final Setting<SecureString> KNN_REMOTE_BUILD_SERVER_PASSWORD_SETTING = SecureSetting.secureString(
        KNN_REMOTE_BUILD_SERVICE_PASSWORD,
        null
    );

    /**
     * Dynamic settings
     */
    public static Map<String, Setting<?>> dynamicCacheSettings = new HashMap<String, Setting<?>>() {
        {
            /**
             * Weight circuit breaker settings
             */
            put(KNN_MEMORY_CIRCUIT_BREAKER_ENABLED, Setting.boolSetting(KNN_MEMORY_CIRCUIT_BREAKER_ENABLED, true, NodeScope, Dynamic));

            /**
             * Group setting that manages node-level circuit breaker configurations based on node tiers.
             * Settings under this group define memory limits for different node classifications.
             * Validation of limit occurs before the setting is retrieved.
             */
            put(
                KNN_MEMORY_CIRCUIT_BREAKER_LIMIT_PREFIX,
                Setting.groupSetting(KNNSettings.KNN_MEMORY_CIRCUIT_BREAKER_LIMIT_PREFIX, settings -> {
                    settings.keySet()
                        .forEach(
                            (limit) -> parseknnMemoryCircuitBreakerValue(
                                settings.get(limit),
                                KNNSettings.KNN_MEMORY_CIRCUIT_BREAKER_CLUSTER_LIMIT
                            )
                        );
                }, NodeScope, Dynamic)
            );

            /**
             * Cluster-wide circuit breaker limit that serves as the default configuration.
             * This setting is used when a node either:
             * - Has no knn_cb_tier attribute defined
             * - Has a tier that doesn't match any node-level configuration
             * Default value: {@value KNN_DEFAULT_MEMORY_CIRCUIT_BREAKER_LIMIT}
             */
            put(
                KNN_MEMORY_CIRCUIT_BREAKER_CLUSTER_LIMIT,
                new Setting<>(
                    KNNSettings.KNN_MEMORY_CIRCUIT_BREAKER_CLUSTER_LIMIT,
                    KNNSettings.KNN_DEFAULT_MEMORY_CIRCUIT_BREAKER_LIMIT,
                    (s) -> parseknnMemoryCircuitBreakerValue(s, KNNSettings.KNN_MEMORY_CIRCUIT_BREAKER_CLUSTER_LIMIT),
                    NodeScope,
                    Dynamic
                )
            );
            /**
             * Cache expiry time settings
             */
            put(KNN_CACHE_ITEM_EXPIRY_ENABLED, Setting.boolSetting(KNN_CACHE_ITEM_EXPIRY_ENABLED, false, NodeScope, Dynamic));
            put(
                KNN_CACHE_ITEM_EXPIRY_TIME_MINUTES,
                Setting.positiveTimeSetting(KNN_CACHE_ITEM_EXPIRY_TIME_MINUTES, TimeValue.timeValueHours(3), NodeScope, Dynamic)
            );
        }
    };

    private final static Map<String, Setting<?>> FEATURE_FLAGS_WHICH_REBUILDS_CACHE = getFeatureFlagsWhichRebuildsCache().stream()
        .collect(toUnmodifiableMap(Setting::getKey, Function.identity()));

    private final static Map<String, Setting<?>> FEATURE_FLAGS = getFeatureFlags().stream()
        .collect(toUnmodifiableMap(Setting::getKey, Function.identity()));

    private ClusterService clusterService;
    private Client client;
    /**
     * The node's thread pools, for {@link #getDirectIORescoreBurstThreads()} only. Nullable: a unit test and
     * any call before {@code KNNPlugin#createComponents} has none, and that case falls back to the formulas
     * rather than failing — this must never be the reason a shard cannot open a {@code .vec}.
     */
    @Setter
    private ThreadPool threadPool;
    @Setter
    private Optional<String> nodeCbAttribute;

    private KNNSettings() {}

    public static synchronized KNNSettings state() {
        if (INSTANCE == null) {
            INSTANCE = new KNNSettings();
        }
        return INSTANCE;
    }

    private void setSettingsUpdateConsumers() {
        clusterService.getClusterSettings().addSettingsUpdateConsumer(updatedSettings -> {
            // When any of the dynamic settings are updated, rebuild the cache with the updated values. Use the current
            // cluster settings values as defaults.
            NativeMemoryCacheManagerDto.NativeMemoryCacheManagerDtoBuilder builder = NativeMemoryCacheManagerDto.builder();

            builder.isWeightLimited(
                updatedSettings.getAsBoolean(KNN_MEMORY_CIRCUIT_BREAKER_ENABLED, getSettingValue(KNN_MEMORY_CIRCUIT_BREAKER_ENABLED))
            );

            // Recompute cache weight
            builder.maxWeight(getUpdatedCircuitBreakerLimit(updatedSettings).getKb());

            builder.isExpirationLimited(
                updatedSettings.getAsBoolean(KNN_CACHE_ITEM_EXPIRY_ENABLED, getSettingValue(KNN_CACHE_ITEM_EXPIRY_ENABLED))
            );

            builder.expiryTimeInMin(
                updatedSettings.getAsTime(KNN_CACHE_ITEM_EXPIRY_TIME_MINUTES, getSettingValue(KNN_CACHE_ITEM_EXPIRY_TIME_MINUTES))
                    .getMinutes()
            );

            NativeMemoryCacheManager.getInstance().rebuildCache(builder.build());
        },
            Stream.concat(dynamicCacheSettings.values().stream(), FEATURE_FLAGS_WHICH_REBUILDS_CACHE.values().stream())
                .collect(Collectors.toUnmodifiableList())
        );
        clusterService.getClusterSettings().addSettingsUpdateConsumer(QUANTIZATION_STATE_CACHE_SIZE_LIMIT_SETTING, it -> {
            quantizationStateCacheManager.setMaxCacheSizeInKB(it.getKb());
            quantizationStateCacheManager.rebuildCache();
        });
        clusterService.getClusterSettings().addSettingsUpdateConsumer(QUANTIZATION_STATE_CACHE_EXPIRY_TIME_MINUTES_SETTING, it -> {
            quantizationStateCacheManager.rebuildCache();
        });
    }

    /**
     * Get setting value by key. Return default value if not configured explicitly.
     *
     * @param key   setting key.
     * @param <T> Setting type
     * @return T     setting value or default
     */
    @SuppressWarnings("unchecked")
    public <T> T getSettingValue(String key) {
        return (T) clusterService.getClusterSettings().get(getSetting(key));
    }

    private Setting<?> getSetting(String key) {
        if (dynamicCacheSettings.containsKey(key)) {
            return dynamicCacheSettings.get(key);
        }

        if (FEATURE_FLAGS.containsKey(key)) {
            return FEATURE_FLAGS.get(key);
        }

        if (KNN_CIRCUIT_BREAKER_TRIGGERED.equals(key)) {
            return KNN_CIRCUIT_BREAKER_TRIGGERED_SETTING;
        }

        if (KNN_CIRCUIT_BREAKER_UNSET_PERCENTAGE.equals(key)) {
            return KNN_CIRCUIT_BREAKER_UNSET_PERCENTAGE_SETTING;
        }

        if (KNN_ALGO_PARAM_INDEX_THREAD_QTY.equals(key)) {
            return KNN_ALGO_PARAM_INDEX_THREAD_QTY_SETTING;
        }

        if (ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD.equals(key)) {
            return ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD_SETTING;
        }

        if (KNN_FAISS_AVX2_DISABLED.equals(key)) {
            return KNN_FAISS_AVX2_DISABLED_SETTING;
        }

        if (KNN_FAISS_AVX512_DISABLED.equals(key)) {
            return KNN_FAISS_AVX512_DISABLED_SETTING;
        }

        if (KNN_FAISS_AVX512_SPR_DISABLED.equals(key)) {
            return KNN_FAISS_AVX512_SPR_DISABLED_SETTING;
        }

        if (KNN_VECTOR_STREAMING_MEMORY_LIMIT_IN_MB.equals(key)) {
            return KNN_VECTOR_STREAMING_MEMORY_LIMIT_PCT_SETTING;
        }

        if (QUANTIZATION_STATE_CACHE_SIZE_LIMIT.equals(key)) {
            return QUANTIZATION_STATE_CACHE_SIZE_LIMIT_SETTING;
        }

        if (QUANTIZATION_STATE_CACHE_EXPIRY_TIME_MINUTES.equals(key)) {
            return QUANTIZATION_STATE_CACHE_EXPIRY_TIME_MINUTES_SETTING;
        }

        if (KNN_DISK_VECTOR_SHARD_LEVEL_RESCORING_DISABLED.equals(key)) {
            return KNN_DISK_VECTOR_SHARD_LEVEL_RESCORING_DISABLED_SETTING;
        }
        if (KNN_DERIVED_SOURCE_ENABLED.equals(key)) {
            return KNN_DERIVED_SOURCE_ENABLED_SETTING;
        }

        if (KNN_REMOTE_VECTOR_BUILD.equals(key)) {
            return KNN_REMOTE_VECTOR_BUILD_SETTING;
        }

        if (KNN_INDEX_REMOTE_VECTOR_BUILD.equals(key)) {
            return KNN_INDEX_REMOTE_VECTOR_BUILD_SETTING;
        }

        if (KNN_REMOTE_REPOSITORY.equals(key)) {
            return KNN_REMOTE_VECTOR_REPOSITORY_SETTING;
        }

        if (KNN_INDEX_REMOTE_VECTOR_BUILD_SIZE_MIN.equals(key)) {
            return KNN_INDEX_REMOTE_VECTOR_BUILD_SIZE_MIN_SETTING;
        }

        if (KNN_REMOTE_VECTOR_BUILD_SIZE_MAX.equals(key)) {
            return KNN_REMOTE_VECTOR_BUILD_SIZE_MAX_SETTING;
        }

        if (KNN_REMOTE_BUILD_SERVICE_ENDPOINT.equals(key)) {
            return KNN_REMOTE_BUILD_SERVICE_ENDPOINT_SETTING;
        }

        if (KNN_REMOTE_BUILD_CLIENT_TIMEOUT.equals(key)) {
            return KNN_REMOTE_BUILD_CLIENT_TIMEOUT_SETTING;
        }

        if (KNN_REMOTE_BUILD_POLL_INTERVAL.equals(key)) {
            return KNN_REMOTE_BUILD_POLL_INTERVAL_SETTING;
        }

        if (KNN_REMOTE_BUILD_SERVICE_USERNAME.equals(key)) {
            return KNN_REMOTE_BUILD_SERVER_USERNAME_SETTING;
        }

        if (KNN_REMOTE_BUILD_SERVICE_PASSWORD.equals(key)) {
            return KNN_REMOTE_BUILD_SERVER_PASSWORD_SETTING;
        }

        if (INDEX_KNN_FAISS_EFFICIENT_FILTER_DISABLE_EXACT_SEARCH.equals(key)) {
            return INDEX_KNN_FAISS_EFFICIENT_FILTER_DISABLE_EXACT_SEARCH_SETTING;
        }

        if (KNN_DYNAMIC_MAPPING_ENABLED.equals(key)) {
            return KNN_DYNAMIC_MAPPING_ENABLED_SETTING;
        }

        if (KNN_DIRECT_IO_RESCORE_ENABLED.equals(key)) {
            return KNN_DIRECT_IO_RESCORE_ENABLED_SETTING;
        }

        if (KNN_DIRECT_IO_MIN_FILE_SIZE.equals(key)) {
            return KNN_DIRECT_IO_MIN_FILE_SIZE_SETTING;
        }

        if (KNN_DIRECT_IO_RESCORE_PREFETCH_THREADS.equals(key)) {
            return KNN_DIRECT_IO_RESCORE_PREFETCH_THREADS_SETTING;
        }

        if (KNN_DIRECT_IO_RESCORE_PREFETCH_QUEUE_CAPACITY.equals(key)) {
            return KNN_DIRECT_IO_RESCORE_PREFETCH_QUEUE_CAPACITY_SETTING;
        }

        if (KNN_DIRECT_IO_DIRECT_MEMORY_BUDGET.equals(key)) {
            return KNN_DIRECT_IO_DIRECT_MEMORY_BUDGET_SETTING;
        }

        if (KNN_DIRECT_IO_RESCORE_CACHE_MAX_BYTES.equals(key)) {
            return KNN_DIRECT_IO_RESCORE_CACHE_MAX_BYTES_SETTING;
        }

        if (KNN_DIRECT_IO_RESCORE_CACHE_SPAN_FILL.equals(key)) {
            return KNN_DIRECT_IO_RESCORE_CACHE_SPAN_FILL_SETTING;
        }

        if (KNN_DIRECT_IO_RESCORE_PREFETCH_MAX_SPAN.equals(key)) {
            return KNN_DIRECT_IO_RESCORE_PREFETCH_MAX_SPAN_SETTING;
        }

        if (KNN_DIRECT_IO_RESCORE_PREFETCH_WIDE_BUFFER_BUDGET.equals(key)) {
            return KNN_DIRECT_IO_RESCORE_PREFETCH_WIDE_BUFFER_BUDGET_SETTING;
        }

        if (KNN_DIRECT_IO_RESCORE_PREFETCH_STAGED_RANGES.equals(key)) {
            return KNN_DIRECT_IO_RESCORE_PREFETCH_STAGED_RANGES_SETTING;
        }

        if (KNN_DIRECT_IO_READAHEAD_WINDOW.equals(key)) {
            return KNN_DIRECT_IO_READAHEAD_WINDOW_SETTING;
        }

        if (KNN_DIRECT_IO_READAHEAD_WINDOW_BYTES.equals(key)) {
            return KNN_DIRECT_IO_READAHEAD_WINDOW_BYTES_SETTING;
        }

        if (KNN_DIRECT_IO_READAHEAD_TRIGGER.equals(key)) {
            return KNN_DIRECT_IO_READAHEAD_TRIGGER_SETTING;
        }

        throw new IllegalArgumentException("Cannot find setting by key [" + key + "]");
    }

    public List<Setting<?>> getSettings() {
        List<Setting<?>> settings = Arrays.asList(
            INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD_SETTING,
            INDEX_KNN_ALGO_PARAM_EF_SEARCH_SETTING,
            KNN_ALGO_PARAM_INDEX_THREAD_QTY_SETTING,
            KNN_CIRCUIT_BREAKER_TRIGGERED_SETTING,
            KNN_CIRCUIT_BREAKER_UNSET_PERCENTAGE_SETTING,
            IS_KNN_INDEX_SETTING,
            MODEL_INDEX_NUMBER_OF_SHARDS_SETTING,
            MODEL_INDEX_NUMBER_OF_REPLICAS_SETTING,
            MODEL_CACHE_SIZE_LIMIT_SETTING,
            ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD_SETTING,
            KNN_FAISS_AVX2_DISABLED_SETTING,
            KNN_VECTOR_STREAMING_MEMORY_LIMIT_PCT_SETTING,
            KNN_FAISS_AVX512_DISABLED_SETTING,
            KNN_FAISS_AVX512_SPR_DISABLED_SETTING,
            QUANTIZATION_STATE_CACHE_SIZE_LIMIT_SETTING,
            QUANTIZATION_STATE_CACHE_EXPIRY_TIME_MINUTES_SETTING,
            KNN_DISK_VECTOR_SHARD_LEVEL_RESCORING_DISABLED_SETTING,
            KNN_DERIVED_SOURCE_ENABLED_SETTING,
            MEMORY_OPTIMIZED_KNN_SEARCH_MODE_SETTING,
            // Index level remote vector build settings
            KNN_INDEX_REMOTE_VECTOR_BUILD_SETTING,
            KNN_INDEX_REMOTE_VECTOR_BUILD_SIZE_MIN_SETTING,
            // Cluster level remote vector build settings
            KNN_REMOTE_VECTOR_BUILD_SETTING,
            KNN_REMOTE_VECTOR_REPOSITORY_SETTING,
            KNN_REMOTE_VECTOR_BUILD_SIZE_MAX_SETTING,
            KNN_REMOTE_BUILD_SERVICE_ENDPOINT_SETTING,
            KNN_REMOTE_BUILD_POLL_INTERVAL_SETTING,
            KNN_REMOTE_BUILD_CLIENT_TIMEOUT_SETTING,
            KNN_REMOTE_BUILD_SERVER_USERNAME_SETTING,
            KNN_REMOTE_BUILD_SERVER_PASSWORD_SETTING,
            INDEX_KNN_FAISS_EFFICIENT_FILTER_DISABLE_EXACT_SEARCH_SETTING,
            KNN_DYNAMIC_MAPPING_ENABLED_SETTING,
            // Direct I/O settings
            // Rescore-seam gate
            KNN_DIRECT_IO_RESCORE_ENABLED_SETTING,
            KNN_DIRECT_IO_MIN_FILE_SIZE_SETTING,
            // Rescore-seam prefetch settings
            KNN_DIRECT_IO_RESCORE_PREFETCH_THREADS_SETTING,
            // Rescore-seam vector cache
            // The directory design's read-ahead channel: staging table size and the waste guard
            KNN_DIRECT_IO_RESCORE_PREFETCH_STAGED_RANGES_SETTING,
            KNN_DIRECT_IO_RESCORE_PREFETCH_MAX_SPAN_SETTING,
            KNN_DIRECT_IO_RESCORE_PREFETCH_WIDE_BUFFER_BUDGET_SETTING,
            KNN_DIRECT_IO_RESCORE_PREFETCH_QUEUE_CAPACITY_SETTING,
            KNN_DIRECT_IO_DIRECT_MEMORY_BUDGET_SETTING,
            KNN_DIRECT_IO_RESCORE_CACHE_MAX_BYTES_SETTING,
            KNN_DIRECT_IO_RESCORE_CACHE_SPAN_FILL_SETTING,
            // Sequential read-ahead ring: window and arm trigger
            KNN_DIRECT_IO_READAHEAD_WINDOW_SETTING,
            KNN_DIRECT_IO_READAHEAD_WINDOW_BYTES_SETTING,
            KNN_DIRECT_IO_READAHEAD_TRIGGER_SETTING
        );
        return Stream.concat(settings.stream(), Stream.concat(getFeatureFlags().stream(), dynamicCacheSettings.values().stream()))
            .collect(Collectors.toList());
    }

    /**
     * Reads a node level setting, falling back to the setting's own default when {@link KNNSettings} has
     * not been initialized with a {@link ClusterService} yet. The Direct I/O settings are read while a
     * shard's Directory is being built, which can also happen in harnesses that never initialize the
     * singleton, and Direct I/O must never be the reason a shard fails to open.
     *
     * @param setting the node level setting to read
     * @return the configured value, or the setting's default if it cannot be read
     */
    private static <T> T getNodeSettingValueOrDefault(final Setting<T> setting) {
        try {
            final ClusterService currentClusterService = state().clusterService;
            if (currentClusterService == null) {
                return setting.getDefault(Settings.EMPTY);
            }
            final T value = currentClusterService.getClusterSettings().get(setting);
            // A null is "could not be read" and is treated the same as a missing cluster service. It is not
            // hypothetical: a ClusterSettings that does not have this setting registered answers null, which
            // would otherwise reach the caller as an unboxing NullPointerException at a shard's Directory.
            return value != null ? value : setting.getDefault(Settings.EMPTY);
        } catch (Exception e) {
            logger.warn("Could not read setting [" + setting.getKey() + "]; falling back to its default", e);
            return setting.getDefault(Settings.EMPTY);
        }
    }

    /**
     * Checks the node level switch for Direct I/O reads on the rescore path. Read once per scorer built
     * at the rescore seam, so it takes effect on the next query with no index or node restart. An
     * unreadable setting falls back to the default rather than failing the query.
     *
     * @return true if the rescore path may read full-precision vectors with Direct I/O
     */
    public static boolean isDirectIORescoreEnabled() {
        return getNodeSettingValueOrDefault(KNN_DIRECT_IO_RESCORE_ENABLED_SETTING);
    }

    /**
     * The size floor on routing. Read at every dispatch, for the same reason the enablement switch is:
     * the setting is dynamic and a {@code .vec} is opened once per reader.
     *
     * @return the smallest faiss/MOS {@code .vec} that may be served with Direct I/O
     */
    public static ByteSizeValue getDirectIOMinFileSize() {
        return getNodeSettingValueOrDefault(KNN_DIRECT_IO_MIN_FILE_SIZE_SETTING);
    }

    /**
     * @return the node wide bound on threads performing rescore seam reads
     */
    public static int getDirectIORescorePrefetchThreads() {
        return getNodeSettingValueOrDefault(KNN_DIRECT_IO_RESCORE_PREFETCH_THREADS_SETTING);
    }

    /**
     * @return prefetch ranges one Direct I/O vector input keeps in flight per burst
     */
    public static int getDirectIORescorePrefetchStagedRanges() {
        return getNodeSettingValueOrDefault(KNN_DIRECT_IO_RESCORE_PREFETCH_STAGED_RANGES_SETTING);
    }

    /**
     * Reads that may wait behind the shared Direct I/O pool, or 0 for the derived default.
     *
     * @return the configured queue capacity, or 0 meaning {@code DirectIOReadPool} should derive it
     */
    public static int getDirectIORescorePrefetchQueueCapacity() {
        return getNodeSettingValueOrDefault(KNN_DIRECT_IO_RESCORE_PREFETCH_QUEUE_CAPACITY_SETTING);
    }

    /**
     * How many threads can be inside a rescore at the same time, which is what the Direct I/O queue has to
     * absorb one {@code staged_ranges} burst from each of. The {@code search} pool answers the query and,
     * under concurrent segment search, {@code NativeEngineKnnVectorQuery} dispatches one task per leaf onto
     * the {@code index_searcher} pool through {@code getTaskExecutor().invokeAll}, so both can hold a scorer.
     *
     * <p><b>Read from the live {@link ThreadPool} when there is one</b>, because that honours an operator's
     * explicit {@code thread_pool.search.size} / {@code thread_pool.index_searcher.size} where re-deriving
     * the formula would not. {@code ThreadPool.Info} for a {@code RESIZABLE} pool carries
     * {@code min == max == size} ({@code ResizableExecutorBuilder.java:99-106}) and
     * {@code ThreadPool#info(String)} returns the holder's own {@code Info}
     * ({@code ThreadPool.java:448-454}). It reflects the size the pool was <em>built</em> at: the dynamic
     * {@code cluster.threadpool.*.size} path resizes the executor in place without replacing that
     * {@code Info} ({@code ThreadPool.java:512+}), so a node that resizes its search pool at runtime should
     * set {@code knn.direct_io.rescore.prefetch.queue_capacity} explicitly.
     *
     * <p>With no {@code ThreadPool} — a unit test, or any call before {@code createComponents} — the same two
     * formulas are applied to the configured allocated processors: {@code searchThreadPoolSize(p)} is
     * {@code ((p * 3) / 2) + 1} ({@code ThreadPool.java:787-789}) and {@code index_searcher} is
     * {@code boundedBy(2 * p, 2, MAX_VALUE)} ({@code ThreadPool.java:783-785}, used at {@code :348-357}).
     *
     * @return {@code search_max + index_searcher_max}, never below 2
     */
    public static int getDirectIORescoreBurstThreads() {
        final ThreadPool pool = state().threadPool;
        if (pool != null) {
            final ThreadPool.Info search = pool.info(ThreadPool.Names.SEARCH);
            final ThreadPool.Info indexSearcher = pool.info(ThreadPool.Names.INDEX_SEARCHER);
            if (search != null && indexSearcher != null) {
                return Math.max(2, search.getMax() + indexSearcher.getMax());
            }
        }
        return burstThreadsFor(allocatedProcessors());
    }

    /**
     * {@code search_max + index_searcher_max} for {@code allocatedProcessors}, using the formulas
     * {@code ThreadPool} itself uses. Visible so the derived queue capacity can be pinned at a processor
     * count this host does not have.
     */
    public static int burstThreadsFor(final int allocatedProcessors) {
        final int search = ThreadPool.searchThreadPoolSize(allocatedProcessors);
        // twiceAllocatedProcessors is package private in OpenSearch, so the one line is restated here.
        final int indexSearcher = Math.max(2, 2 * allocatedProcessors);
        return Math.max(2, search + indexSearcher);
    }

    /**
     * Reads HotSpot's final {@code MaxDirectMemorySize} flag, or {@code null} on a JVM that cannot answer.
     *
     * <p><b>Why this bean and not {@code RuntimeMXBean.getInputArguments()}.</b> MEASURED on
     * Corretto 17.0.20.1 under a {@code SecurityManager} with a minimal policy: {@code getVMOption}
     * returns the value, while {@code getInputArguments()} throws
     * {@code AccessControlException: access denied ("java.lang.management.ManagementPermission" "monitor")}.
     * OpenSearch installs a {@code SecurityManager}, so the second mechanism would need a new grant in
     * {@code plugin-security.policy} to read a number the first one hands over for free. It is also the
     * better question to ask: {@code getInputArguments()} returns the argument <em>strings</em>, which
     * would have to be parsed for {@code k/m/g} suffixes and for a later occurrence overriding an earlier
     * one, whereas the flag is the value HotSpot itself resolved.
     *
     * <p>MEASURED on the same JVM: unset gives {@code value="0"}, {@code origin=DEFAULT};
     * {@code -XX:MaxDirectMemorySize=2g} gives {@code value="2147483648"}, {@code origin=VM_CREATION}. The
     * value alone is sufficient, so the origin is not consulted.
     *
     * @return the flag as a decimal string, or {@code null} if it could not be read
     */
    private static String readMaxDirectMemorySizeFlag() {
        try {
            final HotSpotDiagnosticMXBean bean = ManagementFactory.getPlatformMXBean(HotSpotDiagnosticMXBean.class);
            return bean == null ? null : bean.getVMOption("MaxDirectMemorySize").getValue();
        } catch (RuntimeException | LinkageError e) {
            // A non-HotSpot JVM, or jdk.management absent, or the flag renamed. Fall back to the ergonomic
            // wall rather than failing class initialisation of the plugin's whole settings class.
            return null;
        }
    }

    /**
     * The JVM-wide direct-memory wall: the operator's explicit {@code -XX:MaxDirectMemorySize} when they
     * pinned one, else {@code heapMaxBytes / 2}.
     *
     * <p>{@code heapMaxBytes / 2} is not a guess. OpenSearch's launcher adds
     * {@code -XX:MaxDirectMemorySize=heapSize / 2} to the final JVM options when, and only when, the
     * operator has not set it themselves
     * ({@code distribution/tools/launchers/src/main/java/org/opensearch/tools/launchers/JvmErgonomics.java:74-75}).
     * That same code reads the resolved flag and treats {@code 0} as "the operator did not set it"
     * ({@code :72-76}, with {@code extractMaxDirectMemorySize} at {@code :142-144}), which is the
     * convention this method follows.
     *
     * <p>A consequence worth stating: on a node started through the launcher the flag is <em>always</em>
     * set by the time this runs, either by the operator or by the launcher's own ergonomic choice, so the
     * two branches agree there. The {@code heapMaxBytes / 2} branch is what an embedded JVM, a test JVM, or
     * a node started without the launcher lands on.
     *
     * @param maxDirectMemorySizeFlag HotSpot's final flag value as a decimal string; {@code "0"} or
     *                                {@code null} means the operator did not pin one
     * @param heapMaxBytes            {@code Runtime.maxMemory()}
     * @return the wall in bytes, always positive on any JVM with a positive heap
     */
    static long directMemoryWall(final String maxDirectMemorySizeFlag, final long heapMaxBytes) {
        if (maxDirectMemorySizeFlag != null) {
            try {
                final long pinned = Long.parseLong(maxDirectMemorySizeFlag.trim());
                if (pinned > 0) {
                    return pinned;
                }
            } catch (NumberFormatException e) {
                // Not a number we understand; the ergonomic wall is the safer answer.
            }
        }
        return heapMaxBytes / 2;
    }

    /** The configured allocated processors, which is what every {@code ThreadPool} builder is sized from. */
    private static int allocatedProcessors() {
        final ClusterService currentClusterService = state().clusterService;
        final Settings settings = currentClusterService == null ? Settings.EMPTY : currentClusterService.getSettings();
        return OpenSearchExecutors.allocatedProcessors(settings);
    }

    /**
     * The node-wide ceiling on direct buffers the Direct I/O read path may hold at once; {@code 0b} off.
     *
     * @return the direct-memory budget
     */
    public static ByteSizeValue getDirectIODirectMemoryBudget() {
        return getNodeSettingValueOrDefault(KNN_DIRECT_IO_DIRECT_MEMORY_BUDGET_SETTING);
    }

    /**
     * The node-wide vector cache's budget on the Direct I/O rescore path; {@code 0b} off.
     *
     * @return the cache's maximum weight in bytes
     */
    public static ByteSizeValue getDirectIORescoreCacheMaxBytes() {
        return getNodeSettingValueOrDefault(KNN_DIRECT_IO_RESCORE_CACHE_MAX_BYTES_SETTING);
    }

    /**
     * Whether span fill is on: see {@link #KNN_DIRECT_IO_RESCORE_CACHE_SPAN_FILL} for what it admits and
     * the measurement it exists for. Read once per scored batch.
     *
     * @return true if a batch may also admit the whole vectors inside the spans it staged
     */
    public static boolean isDirectIORescoreCacheSpanFillEnabled() {
        return getNodeSettingValueOrDefault(KNN_DIRECT_IO_RESCORE_CACHE_SPAN_FILL_SETTING);
    }

    /**
     * @return the largest single prefetch span the Direct I/O vector input will stage, in bytes
     */
    public static ByteSizeValue getDirectIORescorePrefetchMaxSpan() {
        return getNodeSettingValueOrDefault(KNN_DIRECT_IO_RESCORE_PREFETCH_MAX_SPAN_SETTING);
    }

    /**
     * The node-wide ceiling on wide-group staging buffers; {@code 0b} means derive it from
     * {@code knn.direct_io.direct_memory_budget}. D3d.
     *
     * @return the configured wide-buffer budget, {@code 0b} for the derived one
     */
    public static ByteSizeValue getDirectIORescorePrefetchWideBufferBudget() {
        return getNodeSettingValueOrDefault(KNN_DIRECT_IO_RESCORE_PREFETCH_WIDE_BUFFER_BUDGET_SETTING);
    }

    /**
     * @return how many windows the Direct I/O sequential read-ahead ring keeps in flight; 0 disables it
     */
    public static int getDirectIOReadaheadWindow() {
        return getNodeSettingValueOrDefault(KNN_DIRECT_IO_READAHEAD_WINDOW_SETTING);
    }

    /**
     * @return bytes per read-ahead window — the sequential ring's request size, independent of {@code bufferSize}
     */
    public static ByteSizeValue getDirectIOReadaheadWindowBytes() {
        return getNodeSettingValueOrDefault(KNN_DIRECT_IO_READAHEAD_WINDOW_BYTES_SETTING);
    }

    /**
     * @return consecutive gapless-forward blocking reads needed before the read-ahead ring arms
     */
    public static int getDirectIOReadaheadTrigger() {
        return getNodeSettingValueOrDefault(KNN_DIRECT_IO_READAHEAD_TRIGGER_SETTING);
    }

    public static boolean isCircuitBreakerTriggered() {
        return KNNSettings.state().getSettingValue(KNNSettings.KNN_CIRCUIT_BREAKER_TRIGGERED);
    }

    /**
     * Retrieves the node-specific circuit breaker limit based on the existing settings.
     *
     * @return String representation of the node-specific circuit breaker limit,
     *         or null if no node-specific limit is set or found
     */
    private String getNodeCbLimit() {
        if (nodeCbAttribute.isPresent()) {
            Settings configuredNodeCbLimits = KNNSettings.state().getSettingValue(KNNSettings.KNN_MEMORY_CIRCUIT_BREAKER_LIMIT_PREFIX);
            return configuredNodeCbLimits.get(nodeCbAttribute.get());
        }
        return null;
    }

    /**
     * Gets node-specific circuit breaker limit from updated settings.
     *
     * @param updatedSettings Settings object containing pending updates
     * @return String representation of new limit if exists for this node's tier, null otherwise
     */
    private String getNodeCbLimit(Settings updatedSettings) {
        if (nodeCbAttribute.isPresent()) {
            return updatedSettings.getByPrefix(KNN_MEMORY_CIRCUIT_BREAKER_LIMIT_PREFIX).get(nodeCbAttribute.get());
        }
        return null;
    }

    /**
     * Returns the cluster-level circuit breaker limit. Needed for initialization
     * during startup when node attributes are not yet available through ClusterService.
     * This limit serves two purposes:
     * 1. As a temporary value during node startup before node-specific limits can be determined
     * 2. As a fallback value for nodes that don't have a knn_cb_tier attribute or
     *    whose tier doesn't match any configured node-level limit
     *
     * @return ByteSizeValue representing the cluster-wide circuit breaker limit
     */
    public static ByteSizeValue getClusterCbLimit() {
        return KNNSettings.state().getSettingValue(KNNSettings.KNN_MEMORY_CIRCUIT_BREAKER_CLUSTER_LIMIT);
    }

    /**
     * Returns the circuit breaker limit for this node using existing configuration. The limit is determined by:
     * 1. Node-specific limit based on the node's circuit breaker tier attribute, if configured
     * 2. Cluster-level default limit if no node-specific configuration exists
     *
     * @return ByteSizeValue representing the circuit breaker limit, either as a percentage
     *         of available memory or as an absolute value
     */
    public ByteSizeValue getCircuitBreakerLimit() {

        return parseknnMemoryCircuitBreakerValue(getNodeCbLimit(), getClusterCbLimit(), KNN_MEMORY_CIRCUIT_BREAKER_CLUSTER_LIMIT);

    }

    /**
     * Determines if and how the circuit breaker limit should be updated for this node.
     * Evaluates both node-specific and cluster-level updates in the updated settings,
     * maintaining proper precedence:
     * 1. Node-tier specific limit from updates (if available)
     * 2. Appropriate fallback value based on node's current configuration
     *
     * @param updatedCbLimits Settings object containing pending circuit breaker updates
     * @return ByteSizeValue representing the new circuit breaker limit to apply,
     *         or null if no applicable updates found
     */
    private ByteSizeValue getUpdatedCircuitBreakerLimit(Settings updatedCbLimits) {
        // Parse any updates, using appropriate fallback if no node-specific limit update exists
        return parseknnMemoryCircuitBreakerValue(
            getNodeCbLimit(updatedCbLimits),
            getFallbackCbLimitValue(updatedCbLimits),
            KNN_MEMORY_CIRCUIT_BREAKER_CLUSTER_LIMIT
        );
    }

    /**
     * Determines the appropriate fallback circuit breaker limit value.
     * The fallback logic follows this hierarchy:
     * 1. If node currently uses cluster-level limit:
     *    - Use updated cluster-level limit if available
     *    - Otherwise maintain current limit
     * 2. If node uses tier-specific limit:
     *    - Maintain current limit (ignore cluster-level updates)
     *
     * This ensures nodes maintain their configuration hierarchy and don't
     * inadvertently fall back to cluster-level limits when they should use
     * tier-specific values.
     *
     * @param updatedCbLimits Settings object containing pending updates
     * @return ByteSizeValue representing the appropriate fallback limit
     */
    private ByteSizeValue getFallbackCbLimitValue(Settings updatedCbLimits) {
        // Update cluster level limit if used
        if (getNodeCbLimit() == null && updatedCbLimits.hasValue(KNN_MEMORY_CIRCUIT_BREAKER_CLUSTER_LIMIT)) {
            return (ByteSizeValue) getSetting(KNN_MEMORY_CIRCUIT_BREAKER_CLUSTER_LIMIT).get(updatedCbLimits);

        }

        // Otherwise maintain current limit (either tier-specific or cluster-level)
        return getCircuitBreakerLimit();
    }

    public static double getCircuitBreakerUnsetPercentage() {
        return KNNSettings.state().getSettingValue(KNNSettings.KNN_CIRCUIT_BREAKER_UNSET_PERCENTAGE);
    }

    public static boolean isKnnIndexFaissEfficientFilterExactSearchDisabled(String indexName) {
        return KNNSettings.state().clusterService.state()
            .getMetadata()
            .index(indexName)
            .getSettings()
            .getAsBoolean(INDEX_KNN_FAISS_EFFICIENT_FILTER_DISABLE_EXACT_SEARCH, false);
    }

    /**
     * @return true if remote vector index build cluster is enabled
     */
    public static boolean isKNNRemoteVectorBuildEnabled() {
        return Booleans.parseBooleanStrict(KNNSettings.state().getSettingValue(KNN_REMOTE_VECTOR_BUILD).toString(), false);
    }

    /**
     * @return true if the dynamic knn_vector mapping feature is enabled cluster-wide.
     */
    public static boolean isDynamicMappingEnabled() {
        return Booleans.parseBooleanStrict(KNNSettings.state().getSettingValue(KNN_DYNAMIC_MAPPING_ENABLED).toString(), false);
    }

    /**
     * Gets the remote build service endpoint.
     * @return String representation of the remote build service endpoint URL
     */
    public static String getRemoteBuildServiceEndpoint() {
        return KNNSettings.state().getSettingValue(KNNSettings.KNN_REMOTE_BUILD_SERVICE_ENDPOINT);
    }

    /**
     * Gets the amount of time the client will wait before abandoning a remote build.
     */
    public static TimeValue getRemoteBuildClientTimeout() {
        return KNNSettings.state().getSettingValue(KNNSettings.KNN_REMOTE_BUILD_CLIENT_TIMEOUT);
    }

    /**
     * Gets the interval at which a RemoteIndexPoller will poll for remote build status.
     */
    public static TimeValue getRemoteBuildClientPollInterval() {
        return KNNSettings.state().getSettingValue(KNNSettings.KNN_REMOTE_BUILD_POLL_INTERVAL);
    }

    public static boolean isFaissAVX2Disabled() {
        try {
            return KNNSettings.state().getSettingValue(KNNSettings.KNN_FAISS_AVX2_DISABLED);
        } catch (Exception e) {
            // In some UTs we identified that cluster setting is not set properly an leads to NPE. This check will avoid
            // those cases and will still return the default value.
            log.warn(
                "Unable to get setting value {} from cluster settings. Using default value as {}",
                KNN_FAISS_AVX2_DISABLED,
                KNN_DEFAULT_FAISS_AVX2_DISABLED_VALUE,
                e
            );
            return KNN_DEFAULT_FAISS_AVX2_DISABLED_VALUE;
        }
    }

    /**
     * check this index enabled/disabled derived source
     * @param settings Settings
     */
    public static boolean isKNNDerivedSourceEnabled(Settings settings) {
        return KNN_DERIVED_SOURCE_ENABLED_SETTING.get(settings);
    }

    public static boolean isFaissAVX512Disabled() {
        return parseBoolean(
            Objects.requireNonNullElse(
                KNNSettings.state().getSettingValue(KNNSettings.KNN_FAISS_AVX512_DISABLED),
                KNN_DEFAULT_FAISS_AVX512_DISABLED_VALUE
            ).toString()
        );
    }

    public static boolean isFaissAVX512SPRDisabled() {
        return parseBoolean(
            Objects.requireNonNullElse(
                KNNSettings.state().getSettingValue(KNNSettings.KNN_FAISS_AVX512_SPR_DISABLED),
                KNN_DEFAULT_FAISS_AVX512_SPR_DISABLED_VALUE
            ).toString()
        );
    }

    public static Integer getFilteredExactSearchThreshold(final String indexName) {
        return getIndexSettings(indexName).getAsInt(
            ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD,
            ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD_DEFAULT_VALUE
        );
    }

    public static boolean isShardLevelRescoringDisabledForDiskBasedVector(final String indexName) {
        return getIndexSettings(indexName).getAsBoolean(KNN_DISK_VECTOR_SHARD_LEVEL_RESCORING_DISABLED, false);
    }

    public void initialize(Client client, ClusterService clusterService) {
        this.client = client;
        this.clusterService = clusterService;
        this.nodeCbAttribute = Optional.empty();
        setSettingsUpdateConsumers();
    }

    public static ByteSizeValue parseknnMemoryCircuitBreakerValue(String sValue, String settingName) {
        return parseknnMemoryCircuitBreakerValue(sValue, null, settingName);
    }

    public static ByteSizeValue parseknnMemoryCircuitBreakerValue(String sValue, ByteSizeValue defaultValue, String settingName) {
        settingName = Objects.requireNonNull(settingName);
        if (sValue != null && sValue.endsWith("%")) {
            final String percentAsString = sValue.substring(0, sValue.length() - 1);
            try {
                final double percent = Double.parseDouble(percentAsString);
                if (percent < 0 || percent > 100) {
                    throw new OpenSearchParseException("percentage should be in [0-100], got [{}]", percentAsString);
                }
                long physicalMemoryInBytes = osProbe.getTotalPhysicalMemorySize();
                if (physicalMemoryInBytes <= 0) {
                    throw new IllegalStateException("Physical memory size could not be determined");
                }
                long esJvmSizeInBytes = JvmInfo.jvmInfo().getMem().getHeapMax().getBytes();
                long eligibleMemoryInBytes = physicalMemoryInBytes - esJvmSizeInBytes;
                return new ByteSizeValue((long) ((percent / 100) * eligibleMemoryInBytes), ByteSizeUnit.BYTES);
            } catch (NumberFormatException e) {
                throw new OpenSearchParseException("failed to parse [{}] as a double", e, percentAsString);
            }
        } else {
            return parseBytesSizeValue(sValue, defaultValue, settingName);
        }
    }

    /**
     * Updates knn.circuit_breaker.triggered setting to true/false
     * @param flag true/false
     */
    public synchronized void updateCircuitBreakerSettings(boolean flag) {
        ClusterUpdateSettingsRequest clusterUpdateSettingsRequest = new ClusterUpdateSettingsRequest();
        Settings circuitBreakerSettings = Settings.builder().put(KNNSettings.KNN_CIRCUIT_BREAKER_TRIGGERED, flag).build();
        clusterUpdateSettingsRequest.persistentSettings(circuitBreakerSettings);
        client.admin().cluster().updateSettings(clusterUpdateSettingsRequest, new ActionListener<ClusterUpdateSettingsResponse>() {
            @Override
            public void onResponse(ClusterUpdateSettingsResponse clusterUpdateSettingsResponse) {
                logger.debug(
                    "Cluster setting {}, acknowledged: {} ",
                    clusterUpdateSettingsRequest.persistentSettings(),
                    clusterUpdateSettingsResponse.isAcknowledged()
                );
            }

            @Override
            public void onFailure(Exception e) {
                logger.info(
                    "Exception while updating circuit breaker setting {} to {}",
                    clusterUpdateSettingsRequest.persistentSettings(),
                    e.getMessage()
                );
            }
        });
    }

    public static ByteSizeValue getVectorStreamingMemoryLimit() {
        return KNNSettings.state().getSettingValue(KNN_VECTOR_STREAMING_MEMORY_LIMIT_IN_MB);
    }

    /**
     *
     * @param index Name of the index
     * @return efSearch value
     */
    public static int getEfSearchParam(String index) {
        final IndexMetadata indexMetadata = KNNSettings.state().clusterService.state().getMetadata().index(index);
        return indexMetadata.getSettings()
            .getAsInt(
                KNNSettings.KNN_ALGO_PARAM_EF_SEARCH,
                IndexHyperParametersUtil.getHNSWEFSearchValue(indexMetadata.getCreationVersion())
            );
    }

    /**
     * Return whether memory optimized search enabled for the given index.
     *
     * @param indexName The name of target index to test whether if it is on.
     * @return True if memory optimized search is enabled, otherwise False.
     */
    public static boolean isMemoryOptimizedKnnSearchModeEnabled(@NonNull final String indexName) {
        return getIndexSettings(indexName).getAsBoolean(MEMORY_OPTIMIZED_KNN_SEARCH_MODE, DEFAULT_MEMORY_OPTIMIZED_KNN_SEARCH_MODE);
    }

    public static Settings getIndexSettings(@NonNull final String indexName) {
        return KNNSettings.state().clusterService.state().getMetadata().index(indexName).getSettings();
    }

    public void setClusterService(ClusterService clusterService) {
        this.clusterService = clusterService;
    }

    static class SpaceTypeValidator implements Setting.Validator<String> {
        @Override
        public void validate(String value) {
            try {
                SpaceType.getSpace(value);
            } catch (IllegalArgumentException ex) {
                throw new InvalidParameterException(ex.getMessage());
            }
        }
    }

    public void onIndexModule(IndexModule module) {
        module.addSettingsUpdateConsumer(INDEX_KNN_ALGO_PARAM_EF_SEARCH_SETTING, newVal -> {
            logger.debug("The value of [KNN] setting [{}] changed to [{}]", KNN_ALGO_PARAM_EF_SEARCH, newVal);
            // TODO: replace cache-rebuild with index reload into the cache
            NativeMemoryCacheManager.getInstance().rebuildCache();
        });
    }

    /**
     * Finds the suggested number of indexing threads based on the number of available processors
     *
     * @return suggested number of indexing threads
     */
    static int getHardwareDefaultIndexThreadQty(final Settings settings) {
        try {
            int availableProcessors = OpenSearchExecutors.allocatedProcessors(settings);
            if (availableProcessors >= INDEX_THREAD_QUANTITY_THRESHOLD) {
                return INDEX_THREAD_QUANTITY_DEFAULT_LARGE;
            } else {
                return INDEX_THREAD_QUANTITY_DEFAULT_SMALL;
            }
        } catch (Exception e) {
            logger.info("[KNN] Failed to determine available processors. Defaulting to 1. [{}]", e.getMessage(), e);
            return 1;
        }
    }

    /**
     * Get the index thread quantity setting value from cluster setting.
     * @return int
     */
    public static int getIndexThreadQty() {
        return KNNSettings.state().getSettingValue(KNN_ALGO_PARAM_INDEX_THREAD_QTY);
    }

    /**
     * Retrieves the {@code index.knn.advanced.approximate_threshold} value from the given
     * {@link org.opensearch.index.mapper.MapperService}'s index settings, or returns the default
     * value when {@code mapperService} is null or the setting is not explicitly configured.
     *
     * @param mapperService the mapper service (nullable)
     * @return the configured threshold or default
     */
    public static int getApproximateThresholdValue(@Nullable MapperService mapperService) {
        if (mapperService == null) {
            return INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD_DEFAULT_VALUE;
        }
        final IndexSettings indexSettings = mapperService.getIndexSettings();
        final Integer approximateThresholdValue = indexSettings.getValue(INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD_SETTING);
        return approximateThresholdValue != null ? approximateThresholdValue : INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD_DEFAULT_VALUE;
    }

    private static String percentageAsString(Integer percentage) {
        return percentage + "%";
    }

    private static Double percentageAsFraction(Integer percentage) {
        return percentage / 100.0;
    }
}

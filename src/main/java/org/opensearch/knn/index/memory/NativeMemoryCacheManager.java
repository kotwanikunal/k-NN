/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

package org.opensearch.knn.index.memory;

import com.google.common.cache.Cache;
import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheStats;
import com.google.common.cache.RemovalCause;
import com.google.common.cache.RemovalNotification;
import lombok.Getter;
import lombok.Setter;
import org.apache.commons.lang3.Validate;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.opensearch.common.unit.TimeValue;
import org.opensearch.knn.common.exception.OutOfNativeMemoryException;
import org.opensearch.knn.common.featureflags.KNNFeatureFlags;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.plugin.stats.StatNames;
import org.opensearch.threadpool.ThreadPool;
import org.opensearch.threadpool.Scheduler.Cancellable;

import java.io.Closeable;
import java.util.Deque;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Callable;
import java.util.concurrent.ConcurrentLinkedDeque;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Manages native memory allocations made by JNI.
 */
public class NativeMemoryCacheManager implements Closeable {

    public static String GRAPH_COUNT = "graph_count";

    private static final Logger logger = LogManager.getLogger(NativeMemoryCacheManager.class);
    private static NativeMemoryCacheManager INSTANCE;
    @Setter
    private static ThreadPool threadPool;

    private Cache<String, NativeMemoryAllocation> cache;
    private Deque<String> accessRecencyQueue;
    private final ConcurrentHashMap<String, ReentrantLock> indexLocks = new ConcurrentHashMap<>();
    private final ExecutorService executor;
    private AtomicBoolean cacheCapacityReached;
    private long maxWeight;
    @Getter
    private Cancellable maintenanceTask;

    /**
     * The cache's size in KB, maintained as entries are loaded and removed rather than re-summed.
     *
     * <p>It replaces an {@code O(entries)} stream sum that {@link #get} called on every miss, and twice more
     * per iteration of the force-evict loop. Equal to the sum by construction and by test: the only two ways
     * an entry enters are the two {@code cache.get} calls in {@link #get}, both of which add the loaded
     * allocation's size, and the only way one leaves is {@link #onRemoval}, which subtracts it.
     * {@code getCacheSizeInKilobytesByScan} is the old sum, kept so a test can assert they agree.
     */
    private final AtomicLong cacheSizeInKb = new AtomicLong();

    /** Bytes of vector cache slab granted, in KB. Not part of {@link #cacheSizeInKb}; see the javadoc. */
    private final AtomicLong slabSizeInKb = new AtomicLong();

    /** The slabs this manager has granted, by address, so that it can free them when it is closed. */
    private final ConcurrentHashMap<Long, NativeMemoryAllocation.VectorCacheSlabAllocation> grantedSlabs = new ConcurrentHashMap<>();

    /**
     * The node's full-precision vector cache. Its memory is this manager's to account, so the cache is this
     * manager's to own: every slab it holds is a grant taken under the same budget as a graph.
     */
    @Getter
    private final SlabVectorCache vectorCache;

    NativeMemoryCacheManager() {
        this.executor = Executors.newSingleThreadExecutor();
        this.cacheCapacityReached = new AtomicBoolean(false);
        this.maxWeight = Long.MAX_VALUE;
        this.vectorCache = new SlabVectorCache(new ManagedSlabSource(), VectorCacheSlab.SLAB_BYTES);
        initialize();
    }

    /**
     * Make sure we just have one instance of cache.
     *
     * @return NativeMemoryCacheManager instance
     */
    public static synchronized NativeMemoryCacheManager getInstance() {
        if (INSTANCE == null) {
            INSTANCE = new NativeMemoryCacheManager();
        }
        return INSTANCE;
    }

    /**
     * Initialize NativeMemoryCacheManager with configurations.
     * Note: maxWeight is initially set to the cluster-level circuit breaker limit
     * because node attributes are not yet available during startup. Once the
     * ClusterService is fully bootstrapped, the circuit breaker will update this
     * value to use any node-specific limits based on the node's circuit_breaker_tier
     * attribute if configured.
     */
    private void initialize() {
        initialize(
            NativeMemoryCacheManagerDto.builder()
                .isWeightLimited(KNNSettings.state().getSettingValue(KNNSettings.KNN_MEMORY_CIRCUIT_BREAKER_ENABLED))
                // Initially use cluster-level limit; will be updated later during cache refresh if node-specific limit exists
                .maxWeight(KNNSettings.getClusterCbLimit().getKb())
                .isExpirationLimited(KNNSettings.state().getSettingValue(KNNSettings.KNN_CACHE_ITEM_EXPIRY_ENABLED))
                .expiryTimeInMin(
                    ((TimeValue) KNNSettings.state().getSettingValue(KNNSettings.KNN_CACHE_ITEM_EXPIRY_TIME_MINUTES)).getMinutes()
                )
                .build()
        );
    }

    private void initialize(NativeMemoryCacheManagerDto nativeMemoryCacheDTO) {
        CacheBuilder<String, NativeMemoryAllocation> cacheBuilder = CacheBuilder.newBuilder()
            .recordStats()
            .concurrencyLevel(1)
            .removalListener(this::onRemoval);

        if (nativeMemoryCacheDTO.isWeightLimited()) {
            this.maxWeight = nativeMemoryCacheDTO.getMaxWeight();
            cacheBuilder.maximumWeight(this.maxWeight).weigher((k, v) -> v.getSizeInKB());
        }

        if (nativeMemoryCacheDTO.isExpirationLimited()) {
            cacheBuilder.expireAfterAccess(nativeMemoryCacheDTO.getExpiryTimeInMin(), TimeUnit.MINUTES);
        }

        cacheCapacityReached = new AtomicBoolean(false);
        accessRecencyQueue = new ConcurrentLinkedDeque<>();
        cache = cacheBuilder.build();
        cacheSizeInKb.set(0);

        if (threadPool != null) {
            startMaintenance(cache);
        } else {
            logger.warn("ThreadPool is null during NativeMemoryCacheManager initialization. Maintenance will not start.");
        }

        // A limit change changes what is left over for vector cache slabs. The vector cache resizes by
        // adding or releasing slabs rather than being rebuilt, so unlike the graph cache above it keeps
        // whatever still fits.
        updateVectorCacheTarget();
    }

    /**
     * Evicts all entries from the cache and rebuilds.
     */
    public synchronized void rebuildCache() {
        rebuildCache(
            NativeMemoryCacheManagerDto.builder()
                .isWeightLimited(KNNSettings.state().getSettingValue(KNNSettings.KNN_MEMORY_CIRCUIT_BREAKER_ENABLED))
                .maxWeight(KNNSettings.state().getCircuitBreakerLimit().getKb())
                .isExpirationLimited(KNNSettings.state().getSettingValue(KNNSettings.KNN_CACHE_ITEM_EXPIRY_ENABLED))
                .expiryTimeInMin(
                    ((TimeValue) KNNSettings.state().getSettingValue(KNNSettings.KNN_CACHE_ITEM_EXPIRY_TIME_MINUTES)).getMinutes()
                )
                .build()
        );
    }

    /**
     * Evict all entries from the cache and rebuilds
     *
     * @param nativeMemoryCacheDTO DTO for cache configuration
     */
    public synchronized void rebuildCache(NativeMemoryCacheManagerDto nativeMemoryCacheDTO) {
        logger.info("KNN Cache rebuilding.");

        // TODO: Does this really need to be executed with an executor?
        executor.execute(() -> {
            // Explicitly invalidate all so that we do not have to wait for garbage collection to be invoked to
            // free up native memory
            cache.invalidateAll();
            initialize(nativeMemoryCacheDTO);
        });
    }

    @Override
    public void close() {
        // Through the vector cache and not through grantedSlabs directly: a slab must not be freed while
        // the cache's index still names it, and only the cache can drop those entries first.
        vectorCache.configure(0);
        executor.shutdown();
        if (maintenanceTask != null) {
            maintenanceTask.cancel();
        }
    }

    /**
     * Getter for current cache size in Kilobytes. Vector cache slabs are not counted here; see
     * {@link #getVectorCacheSlabSizeInKilobytes()} for why, and {@link #getTotalNativeMemoryInKilobytes()}
     * for the two together.
     *
     * @return current size of the cache
     */
    public long getCacheSizeInKilobytes() {
        return cacheSizeInKb.get();
    }

    /**
     * The same figure by the summation this used to do on every miss, so that a test can show the running
     * counter agrees with it.
     *
     * @return current size of the cache, by scanning every entry
     */
    long getCacheSizeInKilobytesByScan() {
        return cache.asMap().values().stream().mapToLong(NativeMemoryAllocation::getSizeInKB).sum();
    }

    /**
     * Returns how full the cache is as a percentage of the total cache capacity.
     *
     * @return Percentage of the cache full
     */
    public Float getCacheSizeAsPercentage() {
        return getSizeAsPercentage(getCacheSizeInKilobytes());
    }

    /**
     * Getter for current size of all indices in Kilobytes.
     *
     * @return current size of the cache
     */
    public long getIndicesSizeInKilobytes() {
        return cache.asMap()
            .values()
            .stream()
            .filter(nativeMemoryAllocation -> nativeMemoryAllocation instanceof NativeMemoryAllocation.IndexAllocation)
            .mapToLong(NativeMemoryAllocation::getSizeInKB)
            .sum();
    }

    /**
     * Returns how full the cache is as a percentage of the total cache capacity.
     *
     * @return Percentage of the cache full
     */
    public Float getIndicesSizeAsPercentage() {
        return getSizeAsPercentage(getIndicesSizeInKilobytes());
    }

    /**
     * Returns the current size of an index in the cache in KiloBytes.
     *
     * @param indexName Name if index to get the weight for
     * @return Size of the index in the cache in kilobytes
     */
    public Long getIndexSizeInKilobytes(final String indexName) {
        Validate.notNull(indexName, "Index name cannot be null");
        return cache.asMap()
            .values()
            .stream()
            .filter(nativeMemoryAllocation -> nativeMemoryAllocation instanceof NativeMemoryAllocation.IndexAllocation)
            .filter(
                indexAllocation -> indexName.equals(((NativeMemoryAllocation.IndexAllocation) indexAllocation).getOpenSearchIndexName())
            )
            .mapToLong(NativeMemoryAllocation::getSizeInKB)
            .sum();
    }

    /**
     * Returns the how much space an index is taking up in the cache as a percentage of the total cache capacity.
     *
     * @param indexName name of the index
     * @return Percentage of the cache full
     */
    public Float getIndexSizeAsPercentage(final String indexName) {
        Validate.notNull(indexName, "Index name cannot be null");
        return getSizeAsPercentage(getIndexSizeInKilobytes(indexName));
    }

    /**
     * Getter for current size of all training jobs in Kilobytes.
     *
     * @return current size of the cache
     */
    public long getTrainingSizeInKilobytes() {
        // Currently, all allocations that are not index allocations will be for training.
        return cache.asMap()
            .values()
            .stream()
            .filter(
                nativeMemoryAllocation -> nativeMemoryAllocation instanceof NativeMemoryAllocation.TrainingDataAllocation
                    || nativeMemoryAllocation instanceof NativeMemoryAllocation.AnonymousAllocation
            )
            .mapToLong(NativeMemoryAllocation::getSizeInKB)
            .sum();
    }

    /**
     * Returns how full the cache is as a percentage of the total cache capacity.
     *
     * @return Percentage of the cache full
     */
    public Float getTrainingSizeAsPercentage() {
        return getSizeAsPercentage(getTrainingSizeInKilobytes());
    }

    /**
     * Getter for maximum weight of the cache.
     *
     * @return maximum cache weight
     */
    public long getMaxCacheSizeInKilobytes() {
        return maxWeight;
    }

    // -------------------------------------------------------------------------------------------------
    // Vector cache slabs (D1 proof b)
    // -------------------------------------------------------------------------------------------------

    /**
     * Bytes of vector cache slab granted right now, in KB.
     *
     * <p>Deliberately not part of {@link #getCacheSizeInKilobytes()}, and the reason is the circuit breaker
     * rather than bookkeeping. {@code KNNCircuitBreaker} clears a tripped breaker only when the cache is at
     * or below the unset percentage of the limit, so if a full vector cache counted towards that figure a
     * breaker tripped by a graph eviction could never be cleared and k-NN indexing would stay rejected
     * cluster-wide for as long as the cache stayed full. The budget is still respected, because the cache's
     * target is the limit minus the non-slab allocations: see {@link #updateVectorCacheTarget()}.
     *
     * @return size of the granted vector cache slabs in KB
     */
    public long getVectorCacheSlabSizeInKilobytes() {
        return slabSizeInKb.get();
    }

    /**
     * Everything k-NN holds in native memory under this budget: cache entries plus vector cache slabs.
     *
     * @return total native memory held in KB
     */
    public long getTotalNativeMemoryInKilobytes() {
        return cacheSizeInKb.get() + slabSizeInKb.get();
    }

    /** Vector cache slabs granted right now. */
    public int getVectorCacheSlabCount() {
        return grantedSlabs.size();
    }

    /**
     * Sets the vector cache's target to {@code min(knn.direct_io.rescore.cache.max_bytes, the native budget
     * minus everything that is not a slab)} and resizes it to match.
     *
     * <p>The second term is what makes the cache yield: a graph load raises the non-slab total, which lowers
     * the target, which releases slabs. The cache resizes by adding or releasing slabs rather than being
     * rebuilt, so a change keeps every vector that still fits. The breaker limit is a ceiling here and not a
     * target, which is the design note's point: on a memory-optimized-search node those bytes would
     * otherwise be page cache for the graph.
     *
     * <p>A setting that cannot be read leaves the target alone rather than guessing at zero, which is the
     * same choice {@code DirectIOVectorCache#applyConfiguredSize} makes for the same reason.
     */
    public void updateVectorCacheTarget() {
        applyVectorCacheTarget(cacheSizeInKb.get());
    }

    /**
     * Shrinks the vector cache so that {@code neededKb} of non-slab allocation will fit beside what is
     * already there, and returns whether it had to.
     *
     * <p>This is rule two of the four the design note asks for: when a graph needs room, vector cache slabs
     * go first and graphs are evicted only if that is not enough. It runs before every path in {@link #get}
     * that could evict or refuse, so by the time Guava weighs the cache or the force-evict loop walks the
     * recency queue, the slabs have already been handed back.
     */
    private boolean makeRoomForNonSlab(final long neededKb) {
        final long nonSlab = cacheSizeInKb.get() + neededKb;
        if (slabSizeInKb.get() == 0 || nonSlab + slabSizeInKb.get() <= maxWeight) {
            return false;
        }
        applyVectorCacheTarget(nonSlab);
        return true;
    }

    private void applyVectorCacheTarget(final long nonSlabKb) {
        final long configuredBytes;
        try {
            if (KNNSettings.isDirectIORescoreEnabled() == false) {
                // D2: the cache is the Direct I/O read path's cache, so turning that path off empties it and
                // returns every slab. There is no automatic switching in either direction; this is the
                // operator's lever and it is the same one.
                vectorCache.configure(0L);
                return;
            }
            final org.opensearch.core.common.unit.ByteSizeValue configured = KNNSettings.getDirectIORescoreCacheMaxBytes();
            configuredBytes = configured == null ? 0L : configured.getBytes();
        } catch (RuntimeException e) {
            logger.debug("Could not read knn.direct_io.rescore.cache.max_bytes; leaving the vector cache target alone", e);
            return;
        }
        final long budgetKb = Math.max(0L, maxWeight - nonSlabKb);
        final long budgetBytes = budgetKb > Long.MAX_VALUE / 1024 ? Long.MAX_VALUE : budgetKb * 1024;
        vectorCache.configure(Math.min(configuredBytes, budgetBytes));
    }

    /**
     * The vector cache's slab supply: every slab is a grant taken under this manager's budget, and giving
     * one back frees it and returns its bytes.
     *
     * <p>Neither method takes any lock this manager holds while it is calling into the vector cache, which
     * is what keeps the two from deadlocking: the cache calls {@code acquire} with its own write lock held,
     * and {@code acquire} does nothing but a compare-and-set on a counter and a {@code malloc}.
     */
    private final class ManagedSlabSource implements SlabVectorCache.SlabSource {

        @Override
        public VectorCacheSlab acquire(final int bytes) {
            final long kb = bytes / 1024L;
            while (true) {
                final long granted = slabSizeInKb.get();
                if (cacheSizeInKb.get() + granted + kb > maxWeight) {
                    // The budget moved under the cache's target, which a concurrent graph load can do.
                    // Declining is the whole protocol: the cache keeps what it has and fills no further.
                    return null;
                }
                if (slabSizeInKb.compareAndSet(granted, granted + kb)) {
                    break;
                }
            }
            final VectorCacheSlab slab;
            try {
                slab = VectorCacheSlab.allocate(bytes);
            } catch (Throwable t) {
                slabSizeInKb.addAndGet(-kb);
                logger.warn("Could not allocate a {} byte vector cache slab", bytes, t);
                return null;
            }
            grantedSlabs.put(slab.address(), new NativeMemoryAllocation.VectorCacheSlabAllocation(slab, () -> slabSizeInKb.addAndGet(-kb)));
            return slab;
        }

        @Override
        public void release(final VectorCacheSlab slab) {
            final NativeMemoryAllocation.VectorCacheSlabAllocation allocation = grantedSlabs.remove(slab.address());
            if (allocation == null) {
                // Not one of ours, which only a test that mixed sources can produce. Free it anyway rather
                // than leaking it, and do not touch the accounting for a grant that was never made.
                slab.close();
                return;
            }
            // Frees the slab and returns its bytes. Never reaches onRemoval, so it cannot trip the breaker
            // or set capacity-reached: rule one of the four.
            allocation.close();
        }
    }

    /**
     * Get graph count for a particular index
     *
     * @param indexName name of OpenSearch index
     * @return number of graphs for a particular OpenSearch index
     */
    public int getIndexGraphCount(String indexName) {
        Validate.notNull(indexName, "Index name cannot be null");
        return Long.valueOf(
            cache.asMap()
                .values()
                .stream()
                .filter(nativeMemoryAllocation -> nativeMemoryAllocation instanceof NativeMemoryAllocation.IndexAllocation)
                .filter(
                    indexAllocation -> indexName.equals(((NativeMemoryAllocation.IndexAllocation) indexAllocation).getOpenSearchIndexName())
                )
                .count()
        ).intValue();
    }

    /**
     * Getter for cache stats.
     *
     * @return cache stats
     */
    public CacheStats getCacheStats() {
        return cache.stats();
    }

    /**
     * Opens a vector index with proper locking mechanism to ensure thread safety.
     * The method uses a ReentrantLock to synchronize access to the index file and
     * cleans up the lock when no other threads are waiting.
     *
     * @param key the unique identifier for the index
     * @param nativeMemoryEntryContext the context containing vector index information
     */
    private void open(String key, NativeMemoryEntryContext nativeMemoryEntryContext) {
        ReentrantLock indexFileLock = indexLocks.computeIfAbsent(key, k -> new ReentrantLock());
        try {
            indexFileLock.lock();
            nativeMemoryEntryContext.open();
        } finally {
            indexFileLock.unlock();
            if (!indexFileLock.hasQueuedThreads()) {
                indexLocks.remove(key, indexFileLock);
            }
        }
    }

    /**
     * Retrieves an entry from the cache and updates its access recency if found.
     * This method combines cache access with recency queue management to maintain
     * the least recently used (LRU) order of cached entries.
     *
     * @param key the unique identifier for the cached entry
     * @return the cached NativeMemoryAllocation if present, null otherwise
     */
    private NativeMemoryAllocation getFromCacheAndUpdateRecency(String key) {
        NativeMemoryAllocation result = cache.getIfPresent(key);
        if (result != null) {
            updateAccessRecency(key);
        }
        return result;
    }

    /**
     * Updates the access recency of a cached entry by moving it to the end of the queue.
     * This method maintains the least recently used (LRU) order by removing the entry
     * from its current position and adding it to the end of the queue.
     *
     * @param key the unique identifier for the cached entry whose recency needs to be updated
     */
    private void updateAccessRecency(String key) {
        accessRecencyQueue.remove(key);
        accessRecencyQueue.addLast(key);
    }

    /**
     * Retrieves NativeMemoryAllocation associated with the nativeMemoryEntryContext.
     *
     * @param nativeMemoryEntryContext Context from which to get NativeMemoryAllocation
     * @param isAbleToTriggerEviction Determines if getting this allocation can evict other entries
     * @return NativeMemoryAllocation associated with nativeMemoryEntryContext
     * @throws ExecutionException if there is an exception when loading from the cache
     */
    public NativeMemoryAllocation get(NativeMemoryEntryContext<?> nativeMemoryEntryContext, boolean isAbleToTriggerEviction)
        throws ExecutionException {
        // Vector cache slabs go back before anything else is weighed, evicted or refused, so that a graph
        // never loses its place to a cache of full-precision vectors. Rule two of the design note's four.
        // Nothing below has to account for the slabs that remain: after this returns they fit in whatever
        // the limit has left once this entry is in, which is what the target it sets means.
        makeRoomForNonSlab(nativeMemoryEntryContext.calculateSizeInKB());

        if (!isAbleToTriggerEviction
            && (maxWeight - getCacheSizeInKilobytes() - nativeMemoryEntryContext.calculateSizeInKB()) <= 0
            && !cache.asMap().containsKey(nativeMemoryEntryContext.getKey())) {
            throw new OutOfNativeMemoryException(
                "Entry cannot be loaded into cache because it would not fit. "
                    + "Entry size: "
                    + nativeMemoryEntryContext.calculateSizeInKB()
                    + " KB "
                    + "Current Cache Size: "
                    + getCacheSizeInKilobytes()
                    + " KB "
                    + "Max Cache Size: "
                    + maxWeight
            );
        }

        if (KNNFeatureFlags.isForceEvictCacheEnabled()) {
            // Utilizes a force eviction mechanism to free up memory before the entry can be added to the cache
            // In case of a cache hit, the operation just updates the locally maintained recency list
            // In case of a cache miss, least recently accessed entries are evicted in a blocking manner
            // before the new entry can be added to the cache.
            String key = nativeMemoryEntryContext.getKey();

            // Cache Hit
            // In case of a cache hit, moving the item to the end of the recency queue adds
            // some overhead to the get operation. This can be optimized further to make this operation
            // as lightweight as possible. Multiple approaches and their outcomes were documented
            // before moving forward with the current solution.
            // The details are outlined here: https://github.com/opensearch-project/k-NN/pull/2015#issuecomment-2327064680
            NativeMemoryAllocation result = getFromCacheAndUpdateRecency(key);
            if (result != null) {
                return result;
            }

            // Cache Miss
            // Evict before put
            // open the graph file before proceeding to load the graph into memory
            open(key, nativeMemoryEntryContext);
            synchronized (this) {
                // recheck if another thread already loaded this entry into the cache
                result = getFromCacheAndUpdateRecency(key);
                if (result != null) {
                    return result;
                }
                if (getCacheSizeInKilobytes() + nativeMemoryEntryContext.calculateSizeInKB() >= maxWeight) {
                    Iterator<String> lruIterator = accessRecencyQueue.iterator();
                    while (lruIterator.hasNext()
                        && (getCacheSizeInKilobytes() + nativeMemoryEntryContext.calculateSizeInKB() >= maxWeight)) {

                        String keyToRemove = lruIterator.next();
                        NativeMemoryAllocation allocationToRemove = cache.getIfPresent(keyToRemove);
                        if (allocationToRemove != null) {
                            allocationToRemove.close();
                            cache.invalidate(keyToRemove);
                        }
                        lruIterator.remove();
                    }
                }

                result = loadIntoCache(key, nativeMemoryEntryContext::load);
                accessRecencyQueue.addLast(key);

                return result;
            }
        } else {
            try (nativeMemoryEntryContext) {
                String key = nativeMemoryEntryContext.getKey();
                // if we already have the allocation we should not open file again as this will cause slowdown in
                // heavy throughput cases, since open() function do locking while opening and mapping the graph file to
                // memory.
                return loadIntoCache(key, () -> {
                    // open graphFile before load
                    open(key, nativeMemoryEntryContext);
                    return nativeMemoryEntryContext.load();
                });
            }
        }
    }

    /**
     * {@code cache.get(key, loader)}, plus the running size counter.
     *
     * <p>The counter is adjusted only when the loader actually ran, so a hit on an entry another thread
     * loaded does not count twice. {@link #onRemoval} is the other side of it.
     */
    private NativeMemoryAllocation loadIntoCache(final String key, final Callable<NativeMemoryAllocation> loader)
        throws ExecutionException {
        final boolean[] loaded = new boolean[1];
        final NativeMemoryAllocation allocation = cache.get(key, () -> {
            final NativeMemoryAllocation value = loader.call();
            loaded[0] = true;
            return value;
        });
        if (loaded[0]) {
            cacheSizeInKb.addAndGet(allocation.getSizeInKB());
        }
        return allocation;
    }

    /**
     * Returns the NativeMemoryAllocation associated with given index
     * @param indexName name of OpenSearch index
     * @return NativeMemoryAllocation associated with given index
     */
    public Optional<NativeMemoryAllocation> getIndexMemoryAllocation(String indexName) {
        Validate.notNull(indexName, "Index name cannot be null");
        return cache.asMap()
            .values()
            .stream()
            .filter(nativeMemoryAllocation -> nativeMemoryAllocation instanceof NativeMemoryAllocation.IndexAllocation)
            .filter(
                indexAllocation -> indexName.equals(((NativeMemoryAllocation.IndexAllocation) indexAllocation).getOpenSearchIndexName())
            )
            .findFirst();
    }

    /**
     * Invalidate entry from the cache.
     *
     * @param key Identifier of entry to invalidate
     */
    public void invalidate(String key) {
        cache.invalidate(key);
    }

    /**
     * Invalidate all entries in the cache.
     */
    public void invalidateAll() {
        cache.invalidateAll();
    }

    /**
     * Returns whether or not the capacity of the cache has been reached
     *
     * @return Boolean of whether cache limit has been reached
     */
    public Boolean isCacheCapacityReached() {
        return cacheCapacityReached.get();
    }

    /**
     * Sets cache capacity reached
     *
     * @param value Boolean value to set cache Capacity Reached to
     */
    public void setCacheCapacityReached(Boolean value) {
        cacheCapacityReached.set(value);
    }

    /**
     * Get the stats of all of the OpenSearch indices currently loaded into the cache
     *
     * @return Map containing all of the OpenSearch indices in the cache and their stats
     */
    public Map<String, Map<String, Object>> getIndicesCacheStats() {
        Map<String, Map<String, Object>> statValues = new HashMap<>();
        NativeMemoryAllocation.IndexAllocation indexAllocation;

        for (Map.Entry<String, NativeMemoryAllocation> entry : cache.asMap().entrySet()) {

            if (entry.getValue() instanceof NativeMemoryAllocation.IndexAllocation) {
                indexAllocation = (NativeMemoryAllocation.IndexAllocation) entry.getValue();
                String indexName = indexAllocation.getOpenSearchIndexName();

                Map<String, Object> indexMap = statValues.computeIfAbsent(indexName, name -> new HashMap<>());
                indexMap.computeIfAbsent(GRAPH_COUNT, key -> getIndexGraphCount(indexName));
                indexMap.computeIfAbsent(StatNames.GRAPH_MEMORY_USAGE.getName(), key -> getIndexSizeInKilobytes(indexName));
                indexMap.computeIfAbsent(StatNames.GRAPH_MEMORY_USAGE_PERCENTAGE.getName(), key -> getIndexSizeAsPercentage(indexName));
            }
        }

        return statValues;
    }

    private void onRemoval(RemovalNotification<String, NativeMemoryAllocation> removalNotification) {
        NativeMemoryAllocation nativeMemoryAllocation = removalNotification.getValue();
        cacheSizeInKb.addAndGet(-nativeMemoryAllocation.getSizeInKB());
        nativeMemoryAllocation.close();

        // A vector cache slab cannot arrive here, because a slab is never a cache entry: see
        // NativeMemoryAllocation.VectorCacheSlabAllocation for the two reasons. That is what makes rule one
        // of the design note's four true by construction rather than by a test of the cause below, and it is
        // why releasing a slab, for any cause, neither trips the breaker nor sets capacity-reached.
        if (RemovalCause.SIZE == removalNotification.getCause()) {
            KNNSettings.state().updateCircuitBreakerSettings(true);
            setCacheCapacityReached(true);
        }

        logger.debug("[KNN] Cache evicted. Key {}, Reason: {}", removalNotification.getKey(), removalNotification.getCause());
    }

    private Float getSizeAsPercentage(long size) {
        long cbLimit = KNNSettings.state().getCircuitBreakerLimit().getKb();
        if (cbLimit == 0) {
            return 0.0F;
        }
        return 100 * size / (float) cbLimit;
    }

    /**
     * Starts the scheduled maintenance for the cache. Without this thread calling cleanUp(), the Guava cache only
     * performs maintenance operations (such as evicting expired entries) when the cache is accessed. This
     * ensures that the cache is also cleaned up based on the configured expiry time.
     * @see <a href="https://github.com/google/guava/wiki/cachesexplained#timed-eviction"> Guava Cache Guide</a>
     * @param cacheInstance cache on which to call cleanUp()
     */
    private void startMaintenance(Cache<String, NativeMemoryAllocation> cacheInstance) {
        if (maintenanceTask != null) {
            maintenanceTask.cancel();
        }

        Runnable cleanUp = () -> {
            try {
                cacheInstance.cleanUp();
            } catch (Exception e) {
                logger.error("Error cleaning up cache", e);
            }
        };

        TimeValue interval = KNNSettings.state().getSettingValue(KNNSettings.KNN_CACHE_ITEM_EXPIRY_TIME_MINUTES);

        maintenanceTask = threadPool.scheduleWithFixedDelay(cleanUp, interval, ThreadPool.Names.MANAGEMENT);
    }
}

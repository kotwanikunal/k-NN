/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.memory;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.IndexReader;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Carries the vector cache's key, {@code (segment core identity, field number)}, from the one place that
 * knows it to the one place that needs it, and owns the release of a segment's entries when its core closes.
 *
 * <h2>Why a handoff is needed at all (CODE)</h2>
 * D1 keys the cache by the segment that owns the vector, and {@link SlabVectorCache} is built to take that
 * key. The scorer cannot derive it. Lucene hands a {@code FlatVectorsScorer} a similarity function, a
 * {@code KnnVectorValues} and the query vector and nothing else
 * ({@code PrefetchableFlatVectorScorer#getRandomVectorScorer}), {@code KnnVectorValues} exposes no identity
 * of any kind, and the scorer instance itself is a process-wide static
 * ({@code FlatVectorsScorerProvider#PREFETCHABLE_LUCENE99_SCORER}), so it cannot carry a segment either.
 * The call that builds the scorer runs inside Lucene, between the k-NN code that has the segment
 * ({@code ExactSearcher#createVectorScorer}, which holds the {@code SegmentReader} and the
 * {@code FieldInfo}) and the k-NN code that builds the scorer, and there is no parameter on it to add.
 *
 * <p>So the caller installs the key for the duration of that one call and the scorer takes a copy of it at
 * construction. The scope is the narrowest thing that works: it is set and cleared in a {@code finally} by
 * one method, every frame between the two is on the same thread, and the scorer stores the value rather than
 * reading the scope later, so nothing outside that window can see it. A scorer built anywhere else finds no
 * key and simply does not use the cache, which is a miss and never a wrong answer, exactly as
 * {@code PrefetchableVectorValuesHelper#doPrefetch} declines for values it cannot unwrap.
 *
 * <h2>Release</h2>
 * The key's lifetime is the segment core's, so {@link #registerRelease} hooks Lucene's core-closed listener,
 * which fires when the last reader of that core closes and the cached bytes become unreachable. That is the
 * gap phase 15's file-keyed cache documented and could not close: its entries for a merged-away segment
 * stayed until LRU pushed them out.
 */
@Log4j2
public final class VectorCacheScope {

    private VectorCacheScope() {}

    /** One segment field: what {@link SlabVectorCache} takes as the first two parts of its key. */
    public record Key(Object core, int field) {
    }

    private static final ThreadLocal<Key> CURRENT = new ThreadLocal<>();

    /** Cores whose release is already hooked, so a listener is added once per core and not once per query. */
    private static final Map<Object, Boolean> REGISTERED = new ConcurrentHashMap<>();

    /** Set by a test that has no node, so that there is a cache to hit; see {@link #nodeCache()}. */
    private static volatile SlabVectorCache testCache;

    /**
     * Installs the key for the current thread. Callers must clear it in a {@code finally}.
     *
     * @param core the segment core identity, from {@code LeafReader#getCoreCacheHelper().getKey()}
     * @param field the field number
     */
    public static void open(final Object core, final int field) {
        CURRENT.set(new Key(core, field));
    }

    /** Clears whatever {@link #open} installed. */
    public static void clear() {
        CURRENT.remove();
    }

    /** The key installed for this thread, or {@code null} outside a scope. */
    public static Key current() {
        return CURRENT.get();
    }

    /**
     * Arranges for {@code core}'s cached vectors to be released when the core closes. Idempotent per core.
     *
     * @param cacheHelper the reader's core cache helper, or {@code null} when the reader has none, in which
     *                    case nothing is registered and the entries live until their slab is reclaimed
     */
    public static void registerRelease(final IndexReader.CacheHelper cacheHelper) {
        if (cacheHelper == null) {
            return;
        }
        final Object core = cacheHelper.getKey();
        if (core == null || REGISTERED.putIfAbsent(core, Boolean.TRUE) != null) {
            return;
        }
        try {
            cacheHelper.addClosedListener(key -> {
                REGISTERED.remove(key);
                final SlabVectorCache cache = nodeCache();
                if (cache != null) {
                    cache.releaseSegment(key);
                }
            });
        } catch (RuntimeException e) {
            // A reader that is already closed throws rather than calling back. Nothing to release then.
            REGISTERED.remove(core);
            log.debug("Could not register a vector cache release listener", e);
        }
    }

    /**
     * The node's vector cache, or {@code null} when there is none to ask. Never throws: a scorer that cannot
     * reach the cache must score the query, not fail it.
     */
    public static SlabVectorCache nodeCache() {
        final SlabVectorCache injected = testCache;
        if (injected != null) {
            return injected;
        }
        try {
            return NativeMemoryCacheManager.getInstance().getVectorCache();
        } catch (RuntimeException | LinkageError e) {
            log.debug("The native memory cache manager is not available, so the vector cache is off", e);
            return null;
        }
    }

    /**
     * Asks the manager to recompute the vector cache's target, which is where a change to
     * {@code knn.direct_io.rescore.cache.max_bytes}, to the native budget, or to
     * {@code knn.direct_io.rescore.enabled} reaches the cache. Called once per prefetch burst, which is the
     * cadence {@code DirectIOVectorCache#applyConfiguredSize} established and the reason
     * {@code SlabVectorCache#configure} returns on a volatile read when nothing has changed.
     *
     * <p>Never throws: a setting that cannot be read must not fail a query.
     */
    public static void refreshNodeCacheTarget() {
        if (testCache != null) {
            return;
        }
        try {
            NativeMemoryCacheManager.getInstance().updateVectorCacheTarget();
        } catch (RuntimeException | LinkageError e) {
            log.debug("Could not refresh the vector cache target", e);
        }
    }

    /**
     * Test seam: the cache {@link #nodeCache()} returns. A unit test has no cluster settings, so the manager
     * singleton it would otherwise build cannot be constructed.
     *
     * @param cache the cache to serve, or {@code null} to go back to the manager's
     */
    public static void setNodeCacheForTesting(final SlabVectorCache cache) {
        testCache = cache;
    }

    /** Test seam: forgets which cores have a release listener. */
    static void forgetRegisteredCoresForTesting() {
        REGISTERED.clear();
    }
}

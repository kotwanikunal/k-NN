/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import org.opensearch.common.settings.ClusterSettings;
import org.opensearch.common.settings.Setting;
import org.opensearch.common.settings.Settings;
import org.opensearch.core.common.unit.ByteSizeUnit;
import org.opensearch.core.common.unit.ByteSizeValue;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.KNNSettings;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.Collectors;

import static org.mockito.Mockito.when;

/**
 * The cache's promises, tested as promises rather than as an implementation: the budget is never
 * exceeded, the eviction order is least-recently-used, a lookup never invents or corrupts a vector under
 * concurrency, and a budget of zero produces no cache at all.
 */
public class LruVectorCacheTests extends KNNTestCase {

    private static final int DIM = 8;
    /** Bytes one entry costs at {@link #DIM}: the payload plus the accounted per-entry overhead. */
    private static final long ENTRY = LruVectorCache.ENTRY_OVERHEAD_BYTES + DIM * (long) Float.BYTES;

    /** A vector whose every element encodes its ordinal, so a wrong hit is visible in the values. */
    private static float[] vectorFor(final int ord) {
        final float[] vector = new float[DIM];
        for (int i = 0; i < DIM; i++) {
            vector[i] = ord * 100f + i;
        }
        return vector;
    }

    private static LruVectorCache cacheHolding(final int entries) {
        final LruVectorCache cache = LruVectorCache.forSource("test", DIM, entries * ENTRY);
        assertNotNull(cache);
        assertEquals(entries, cache.capacity());
        return cache;
    }

    /** A hit must return the vector that was put, element for element, into the caller's own array. */
    public void testPutThenLoadReturnsTheSameVector() {
        final LruVectorCache cache = cacheHolding(4);
        cache.put(7, vectorFor(7));

        final float[] dst = new float[DIM];
        assertTrue(cache.load(7, dst));
        assertArrayEquals(vectorFor(7), dst, 0.0f);
    }

    /** A miss must leave the caller's array alone rather than half filling it. */
    public void testLoadOnMissLeavesTheDestinationUntouched() {
        final LruVectorCache cache = cacheHolding(4);
        cache.put(1, vectorFor(1));

        final float[] dst = new float[DIM];
        Arrays.fill(dst, -1f);
        assertFalse(cache.load(2, dst));
        for (float element : dst) {
            assertEquals(-1f, element, 0.0f);
        }
    }

    /**
     * The cache copies on the way in. A caller that reuses one scratch array for every decode - which is
     * exactly what the loader does - must not find every entry holding the last vector it decoded.
     */
    public void testPutCopiesSoTheCallerMayReuseItsArray() {
        final LruVectorCache cache = cacheHolding(4);
        final float[] scratch = new float[DIM];

        System.arraycopy(vectorFor(1), 0, scratch, 0, DIM);
        cache.put(1, scratch);
        System.arraycopy(vectorFor(2), 0, scratch, 0, DIM);
        cache.put(2, scratch);

        final float[] dst = new float[DIM];
        assertTrue(cache.load(1, dst));
        assertArrayEquals(vectorFor(1), dst, 0.0f);
        assertTrue(cache.load(2, dst));
        assertArrayEquals(vectorFor(2), dst, 0.0f);
    }

    /** And on the way out: mutating what a hit produced must not change what the cache holds. */
    public void testLoadCopiesSoTheCallerCannotCorruptTheEntry() {
        final LruVectorCache cache = cacheHolding(4);
        cache.put(3, vectorFor(3));

        final float[] dst = new float[DIM];
        assertTrue(cache.load(3, dst));
        Arrays.fill(dst, Float.NaN);

        final float[] second = new float[DIM];
        assertTrue(cache.load(3, second));
        assertArrayEquals(vectorFor(3), second, 0.0f);
    }

    /**
     * The headline bound. Whatever the insertion pattern, held bytes must never exceed the budget - not
     * after the fact, and not transiently either, which is what checking after every single put covers.
     */
    public void testHeldBytesNeverExceedTheBudget() {
        final long budget = 10 * ENTRY;
        final LruVectorCache cache = LruVectorCache.forSource("test", DIM, budget);
        assertNotNull(cache);

        for (int ord = 0; ord < 500; ord++) {
            cache.put(ord, vectorFor(ord));
            assertTrue("held " + cache.stats().bytes() + " > budget " + budget, cache.stats().bytes() <= budget);
        }
        assertEquals(10, cache.stats().entries());
        assertEquals(10 * ENTRY, cache.stats().bytes());
    }

    /**
     * A budget that is not a whole number of entries must be floored, not rounded up: the cache holds
     * what fits and leaves the remainder unspent.
     */
    public void testBudgetIsFlooredToWholeEntries() {
        final LruVectorCache cache = LruVectorCache.forSource("test", DIM, 3 * ENTRY + ENTRY / 2);
        assertNotNull(cache);
        assertEquals(3, cache.capacity());

        for (int ord = 0; ord < 10; ord++) {
            cache.put(ord, vectorFor(ord));
        }
        assertEquals(3, cache.stats().entries());
        assertEquals(3 * ENTRY, cache.stats().bytes());
    }

    /** Eviction is least-recently-used, and a load is a use. */
    public void testEvictionFollowsLeastRecentlyUsedOrder() {
        final LruVectorCache cache = cacheHolding(3);
        cache.put(1, vectorFor(1));
        cache.put(2, vectorFor(2));
        cache.put(3, vectorFor(3));

        // Touch 1, so 2 is now the least recently used and 1 must survive the next insert.
        final float[] dst = new float[DIM];
        assertTrue(cache.load(1, dst));

        cache.put(4, vectorFor(4));

        assertTrue("1 was touched and must have survived", cache.load(1, dst));
        assertFalse("2 was least recently used and must have been evicted", cache.load(2, dst));
        assertTrue(cache.load(3, dst));
        assertTrue(cache.load(4, dst));
    }

    /** Re-putting an ordinal already held is a use of it too, and must not double-charge the budget. */
    public void testPutOfAHeldOrdinalRefreshesItWithoutGrowing() {
        final LruVectorCache cache = cacheHolding(3);
        cache.put(1, vectorFor(1));
        cache.put(2, vectorFor(2));
        cache.put(3, vectorFor(3));
        assertEquals(3 * ENTRY, cache.stats().bytes());

        cache.put(1, vectorFor(1));
        assertEquals("a repeat put must not add an entry", 3, cache.stats().entries());
        assertEquals(3 * ENTRY, cache.stats().bytes());

        cache.put(4, vectorFor(4));
        final float[] dst = new float[DIM];
        assertTrue("the repeat put must have refreshed 1", cache.load(1, dst));
        assertFalse(cache.load(2, dst));
    }

    /** Eviction accounting: one entry out per entry in, once the cache is full, and the count is exact. */
    public void testEvictionCountIsExact() {
        final LruVectorCache cache = cacheHolding(5);
        for (int ord = 0; ord < 5; ord++) {
            cache.put(ord, vectorFor(ord));
        }
        assertEquals("nothing evicted while there was room", 0, cache.stats().evictions());

        for (int ord = 5; ord < 20; ord++) {
            cache.put(ord, vectorFor(ord));
        }
        assertEquals(15, cache.stats().evictions());
        assertEquals(5, cache.stats().entries());
    }

    /**
     * The counting contract: one lookup per vector the loader delivers, counted at the point it resolves.
     * A hit is counted by {@link LruVectorCache#load}, a miss by {@link LruVectorCache#put} - because a put
     * happens exactly once per device read - and a failed {@code load} counts nothing, because the caller is
     * about to call {@code put} and would otherwise be charged twice for one vector.
     */
    public void testOneLookupIsCountedPerDeliveredVector() {
        final LruVectorCache cache = cacheHolding(4);
        assertEquals(0.0d, cache.stats().hitRate(), 0.0d);

        final float[] dst = new float[DIM];
        // Miss, then the device read that follows it. One vector delivered, one lookup, one miss.
        assertFalse(cache.load(1, dst));
        cache.put(1, vectorFor(1));
        assertEquals(0, cache.stats().hits());
        assertEquals(1, cache.stats().misses());

        // Three hits on the entry that put left behind.
        assertTrue(cache.load(1, dst));
        assertTrue(cache.load(1, dst));
        assertTrue(cache.load(1, dst));

        assertEquals(3, cache.stats().hits());
        assertEquals(1, cache.stats().misses());
        assertEquals("four vectors delivered, four lookups counted", 0.75d, cache.stats().hitRate(), 1e-9);
    }

    /**
     * The stage-time half of a staged lookup counts nothing at all, or every hit would be counted twice -
     * once here and once when the consumer reaches that position and calls {@code load}.
     */
    public void testTouchIfResidentReportsResidencyWithoutCounting() {
        final LruVectorCache cache = cacheHolding(4);
        cache.put(1, vectorFor(1));
        final long missesAfterThePut = cache.stats().misses();

        assertTrue(cache.touchIfResident(1));
        assertFalse(cache.touchIfResident(2));
        assertFalse(cache.touchIfResident(-1));

        assertEquals(0, cache.stats().hits());
        assertEquals(missesAfterThePut, cache.stats().misses());
    }

    /**
     * Touching at stage time rather than at consume time is what stops a query evicting its own working
     * set: every ordinal the query is about to want becomes most-recently-used before any of that query's
     * misses start making room. Here ordinal 1 is the least recently used until it is touched, after which
     * the insert has to take 2 instead.
     */
    public void testTouchIfResidentMakesAnEntryMostRecentlyUsed() {
        final LruVectorCache cache = cacheHolding(3);
        cache.put(1, vectorFor(1));
        cache.put(2, vectorFor(2));
        cache.put(3, vectorFor(3));

        assertTrue(cache.touchIfResident(1));
        cache.put(4, vectorFor(4));

        final float[] dst = new float[DIM];
        assertTrue("1 was touched and must have survived", cache.load(1, dst));
        assertFalse("2 became least recently used and must have been evicted", cache.load(2, dst));
    }

    /** A budget of zero is not a cache that always misses; it is no cache object at all. */
    public void testZeroBudgetYieldsNoCache() {
        assertNull(LruVectorCache.forSource("test", DIM, 0));
        assertNull(LruVectorCache.forSource("test", DIM, -1));
    }

    /** A budget too small for a single vector is the same thing: no cache rather than a useless one. */
    public void testBudgetBelowOneEntryYieldsNoCache() {
        assertNull(LruVectorCache.forSource("test", DIM, ENTRY - 1));
        assertNotNull(LruVectorCache.forSource("test", DIM, ENTRY));
    }

    public void testNonPositiveDimensionYieldsNoCache() {
        assertNull(LruVectorCache.forSource("test", 0, 1 << 20));
        assertNull(LruVectorCache.forSource("test", -8, 1 << 20));
    }

    /** A vector of the wrong length is a bug upstream; retaining it would turn it into a wrong score. */
    public void testPutRejectsMalformedInput() {
        final LruVectorCache cache = cacheHolding(4);
        cache.put(1, new float[DIM - 1]);
        cache.put(2, new float[DIM + 1]);
        cache.put(3, null);
        cache.put(-1, vectorFor(1));
        assertEquals(0, cache.stats().entries());
        assertEquals(0, cache.stats().bytes());
    }

    /** The settings-driven factory: the 8 MB default, and zero meaning no cache. */
    public void testForSourceReadsTheNodeSetting() {
        final LruVectorCache atDefault = LruVectorCache.forSource("test", 768);
        assertNotNull(atDefault);
        assertEquals(new ByteSizeValue(8, ByteSizeUnit.MB).getBytes(), atDefault.budgetBytes());
        // 8 MB / (3072 + 80) bytes per entry. Worth pinning: it is the number to read the measured
        // 44.2% steady-state hit rate against a 200-candidate query set with.
        assertEquals(2661, atDefault.capacity());

        setNodeSettings(Settings.builder().put(KNNSettings.KNN_DIRECT_IO_RESCORE_CACHE_BYTES_PER_SOURCE, "0b").build());
        assertNull(LruVectorCache.forSource("test", 768));

        setNodeSettings(Settings.builder().put(KNNSettings.KNN_DIRECT_IO_RESCORE_CACHE_BYTES_PER_SOURCE, "4mb").build());
        final LruVectorCache atFourMegabytes = LruVectorCache.forSource("test", 768);
        assertNotNull(atFourMegabytes);
        assertEquals(new ByteSizeValue(4, ByteSizeUnit.MB).getBytes(), atFourMegabytes.budgetBytes());
    }

    /** A source opened in a harness that never gave KNNSettings a ClusterService still gets the default. */
    public void testForSourceWithoutAClusterServiceUsesTheDefault() {
        KNNSettings.state().setClusterService(null);
        final LruVectorCache cache = LruVectorCache.forSource("test", 768);
        assertNotNull(cache);
        assertEquals(new ByteSizeValue(8, ByteSizeUnit.MB).getBytes(), cache.budgetBytes());
    }

    /** Re-stubs the mock ClusterService the base class installs, with {@code settings} applied. */
    private void setNodeSettings(final Settings settings) {
        final Set<Setting<?>> registered = new HashSet<>(ClusterSettings.BUILT_IN_CLUSTER_SETTINGS);
        registered.addAll(
            KNNSettings.state()
                .getSettings()
                .stream()
                .filter(s -> s.getProperties().contains(Setting.Property.NodeScope))
                .collect(Collectors.toList())
        );
        when(clusterService.getClusterSettings()).thenReturn(new ClusterSettings(settings, registered));
        KNNSettings.state().setClusterService(clusterService);
    }

    /**
     * Concurrent lookup and populate over one cache, which is the real shape: Lucene runs per-leaf
     * scorers in parallel and every one of them holds a loader over the same source. The assertion is not
     * about which entries survive - the LRU order under a race is not defined - but that no lookup ever
     * produces a vector belonging to a different ordinal, and that the budget holds throughout.
     */
    public void testConcurrentLoadAndPutStayCorrectAndInBudget() throws Exception {
        final int threads = 8;
        final int ordinals = 64;
        final long budget = 16 * ENTRY;
        final LruVectorCache cache = LruVectorCache.forSource("test", DIM, budget);
        assertNotNull(cache);

        final CountDownLatch start = new CountDownLatch(1);
        final AtomicReference<AssertionError> failure = new AtomicReference<>();
        final List<Thread> workers = new ArrayList<>();
        for (int t = 0; t < threads; t++) {
            final int seed = t;
            final Thread worker = new Thread(() -> {
                final float[] dst = new float[DIM];
                try {
                    start.await();
                    for (int i = 0; i < 20_000; i++) {
                        final int ord = (seed * 7 + i * 13) % ordinals;
                        if (cache.load(ord, dst)) {
                            // The whole correctness question for a cache: did it hand back this ordinal's
                            // vector, or some other ordinal's?
                            if (Arrays.equals(vectorFor(ord), dst) == false) {
                                throw new AssertionError("ordinal " + ord + " loaded as " + Arrays.toString(dst));
                            }
                        } else {
                            cache.put(ord, vectorFor(ord));
                        }
                        if (cache.stats().bytes() > budget) {
                            throw new AssertionError("held " + cache.stats().bytes() + " > budget " + budget);
                        }
                    }
                } catch (AssertionError e) {
                    failure.compareAndSet(null, e);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
            }, "lru-vector-cache-" + t);
            workers.add(worker);
            worker.start();
        }
        start.countDown();
        for (Thread worker : workers) {
            worker.join(TimeUnit.MINUTES.toMillis(1));
            assertFalse("worker did not finish", worker.isAlive());
        }
        if (failure.get() != null) {
            throw failure.get();
        }

        assertTrue(cache.stats().bytes() <= budget);
        assertEquals(cache.stats().entries() * ENTRY, cache.stats().bytes());
        assertTrue("the workload must have produced hits", cache.stats().hits() > 0);
        assertTrue("and evictions, since 64 ordinals do not fit in 16 slots", cache.stats().evictions() > 0);

        // Every surviving entry must still be the vector for its own ordinal.
        final Set<Integer> present = new HashSet<>();
        final float[] dst = new float[DIM];
        for (int ord = 0; ord < ordinals; ord++) {
            if (cache.load(ord, dst)) {
                present.add(ord);
                assertArrayEquals(vectorFor(ord), dst, 0.0f);
            }
        }
        assertFalse(present.isEmpty());
    }

    /**
     * The stats-line cadence. Pinned because it is the only way the counters leave the JVM - the benchmark
     * and the operator both read the hit rate off these lines - and because both ways it can break are
     * silent: "every lookup" floods a live node's log, "never" leaves a run with no numbers, and no other
     * assertion in this suite touches either.
     */
    public void testStatsLinesAreDueOncePerInterval() {
        final LruVectorCache cache = LruVectorCache.forSource("test", DIM, 4 * ENTRY);
        assertNotNull(cache);

        for (int i = 1; i < LruVectorCache.STATS_LOG_INTERVAL; i++) {
            assertFalse("lookup " + i + " is not the interval", cache.statsLineDue());
        }
        assertTrue("lookup " + LruVectorCache.STATS_LOG_INTERVAL + " is", cache.statsLineDue());

        // And the counter resets rather than latching, so the second interval behaves like the first.
        for (int i = 1; i < LruVectorCache.STATS_LOG_INTERVAL; i++) {
            assertFalse(cache.statsLineDue());
        }
        assertTrue(cache.statsLineDue());
    }

    /**
     * Lookups are what drive the lines, so a cache nobody asks about never writes one. The reason to assert
     * it: the interval counter is advanced from inside the lookup methods, and moving it to {@code stats()}
     * - a plausible refactor, since that is what builds the line - would make an operator's own polling the
     * thing that triggers the logging.
     */
    public void testStatsAreNotADueLookup() {
        final LruVectorCache cache = cacheHolding(4);
        for (int i = 0; i < LruVectorCache.STATS_LOG_INTERVAL * 2; i++) {
            cache.stats();
            cache.budgetBytes();
            cache.capacity();
        }
        assertFalse("reading the stats is not a lookup", cache.statsLineDue());
    }
}

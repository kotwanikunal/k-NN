/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import com.carrotsearch.randomizedtesting.ThreadFilter;
import com.carrotsearch.randomizedtesting.annotations.ThreadLeakFilters;
import lombok.SneakyThrows;
import org.opensearch.knn.KNNTestCase;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

/**
 * The node-wide read pool. What matters about it is the properties a wrong pool would break silently: that
 * it is shared rather than per caller (a per-segment pool cannot bound in-flight I/O across segments), that
 * its threads are daemons (it is never shut down), and that submitted reads actually run.
 */
@ThreadLeakFilters(defaultFilters = true, filters = { DirectIOReadPoolTests.ReadPoolThreadFilter.class })
public class DirectIOReadPoolTests extends KNNTestCase {

    /**
     * The pool outlives any one query, so on a node it is created once and never shut down and its threads
     * are daemons that idle out. A test suite that used it therefore ends with those threads still parked,
     * which is the intended lifecycle rather than a leak.
     */
    public static final class ReadPoolThreadFilter implements ThreadFilter {
        @Override
        public boolean reject(final Thread thread) {
            return thread.getName().startsWith(DirectIOReadPool.THREAD_NAME_PREFIX);
        }
    }

    @Override
    public void tearDown() throws Exception {
        DirectIOReadPool.resetForTesting();
        super.tearDown();
    }

    public void testTheSamePoolIsHandedToEveryCaller() {
        final ExecutorService first = DirectIOReadPool.executor();
        assertNotNull(first);
        assertSame("a per-caller pool cannot bound in-flight reads across segments", first, DirectIOReadPool.executor());
    }

    @SneakyThrows
    public void testSubmittedWorkRuns() {
        final ExecutorService pool = DirectIOReadPool.executor();
        final List<Future<Integer>> results = new ArrayList<>();
        for (int i = 0; i < 32; i++) {
            final int value = i;
            results.add(pool.submit(() -> value * 2));
        }
        for (int i = 0; i < results.size(); i++) {
            assertEquals(Integer.valueOf(i * 2), results.get(i).get(10, TimeUnit.SECONDS));
        }
    }

    /**
     * The pool is created lazily and never shut down, so a non-daemon thread in it would hold up JVM exit
     * for its whole keep-alive after the last read.
     */
    @SneakyThrows
    public void testReaderThreadsAreDaemonsAndNamed() {
        final ExecutorService pool = DirectIOReadPool.executor();
        final CountDownLatch seen = new CountDownLatch(1);
        final Thread[] worker = new Thread[1];
        pool.submit(() -> {
            worker[0] = Thread.currentThread();
            seen.countDown();
        });
        assertTrue(seen.await(10, TimeUnit.SECONDS));
        assertTrue("reader threads must be daemons", worker[0].isDaemon());
        assertTrue("got thread name " + worker[0].getName(), worker[0].getName().startsWith(DirectIOReadPool.THREAD_NAME_PREFIX));
    }

    /**
     * The bound is a dynamic setting, so it has to be applied to the live pool rather than only at creation.
     * With no cluster service in a unit test the setting reads its default, so what is pinned here is that
     * the pool is sized from the setting and that re-reading it does not resize or replace the pool.
     */
    public void testThePoolIsSizedFromTheSetting() {
        final ThreadPoolExecutor pool = (ThreadPoolExecutor) DirectIOReadPool.executor();
        assertNotNull(pool);
        final int expected = org.opensearch.knn.index.KNNSettings.getDirectIORescorePrefetchThreads();
        assertEquals(expected, pool.getCorePoolSize());
        assertEquals(expected, pool.getMaximumPoolSize());
        assertTrue("idle reader threads must be reclaimable", pool.allowsCoreThreadTimeOut());
        assertEquals(DirectIOReadPool.KEEP_ALIVE_SECONDS, pool.getKeepAliveTime(TimeUnit.SECONDS));

        assertSame(pool, DirectIOReadPool.executor());
        assertEquals(expected, pool.getCorePoolSize());
    }

    /**
     * The queue is bounded, which is the whole of the hardening: with an unbounded one a search thread can
     * end up waiting on a read queued behind hundreds of others with no bound at all. Measured on the real
     * stack: 654-857 tasks queued, sustained, at 16 concurrent readers on 4 cores.
     */
    public void testTheQueueIsBounded() {
        final ThreadPoolExecutor pool = (ThreadPoolExecutor) DirectIOReadPool.executor();
        final int expected = DirectIOReadPool.queueCapacity();
        assertEquals(
            "an unbounded queue is the thing this bound exists to prevent",
            expected,
            pool.getQueue().remainingCapacity() + pool.getQueue().size()
        );
    }

    /**
     * The floor matters for a reason that is not about safety: one prefetch burst is Lucene's 64-ordinal
     * bulk batch, and task-15 measured that a staging path which declines the tail of every burst makes
     * 5.4 blocking reads per query and a paced p99 of 30 ms against 0.0 and 11 ms for one that holds it.
     * So the queue must hold several whole bursts however small the host is.
     */
    public void testTheQueueAlwaysHoldsAWholeBurst() {
        assertEquals(256, DirectIOReadPool.MIN_QUEUE_CAPACITY);
        assertEquals(DirectIOReadPool.MIN_QUEUE_CAPACITY, DirectIOReadPool.derivedQueueCapacity(1, 64));
        assertEquals(DirectIOReadPool.MIN_QUEUE_CAPACITY, DirectIOReadPool.derivedQueueCapacity(2, 64));
        assertTrue(
            "the floor must hold at least one burst at the shipped staged_ranges",
            DirectIOReadPool.MIN_QUEUE_CAPACITY >= DirectIOVectorIndexInput.DEFAULT_MAX_STAGED_RANGES
        );
    }

    /**
     * D1(b): the capacity is a setting, and its <b>default is derived from burst demand</b> —
     * {@code max(256, nextPowerOfTwo((search_max + index_searcher_max) × staged_ranges))}.
     *
     * <p>The old derivation was {@code max(64, 8 × pool_threads)}, which used the wrong variable: pool
     * threads set the drain rate, while the queue has to absorb the peak burst. The Oct 4-5 node sweep
     * measured the cost — at the effective capacity of 64 that formula gave the 4-vCPU node, 46-52% of
     * staged reads were refused and the node made 783k device reads per 5,000 queries at
     * p50/p90/p99 777/1069/1322 ms; at 1,024 it refused 0% and made 707k at 650/877/954.
     *
     * <p>Pinned at the two processor counts the decision was taken for. The formulas are OpenSearch's own:
     * {@code searchThreadPoolSize(p) = ((p × 3) / 2) + 1} ({@code ThreadPool.java:787-789}) and
     * {@code index_searcher = boundedBy(2p, 2, MAX)} ({@code ThreadPool.java:783-785}).
     */
    public void testTheDefaultQueueCapacityIsDerivedFromBurstDemand() {
        // The 4-vCPU bench node: search 7, index_searcher 8, 15 x 64 = 960 -> 1024, which is the capacity
        // the sweep measured as best.
        assertEquals("search(4) + index_searcher(4)", 15, org.opensearch.knn.index.KNNSettings.burstThreadsFor(4));
        assertEquals(1024, DirectIOReadPool.derivedQueueCapacity(15, 64));

        // 16 allocated processors: search 25, index_searcher 32, 57 x 64 = 3648 -> 4096.
        assertEquals("search(16) + index_searcher(16)", 57, org.opensearch.knn.index.KNNSettings.burstThreadsFor(16));
        assertEquals(4096, DirectIOReadPool.derivedQueueCapacity(57, 64));

        // 64 processors, the shape the formula has to stay sane at: 97 + 128 = 225, x 64 = 14400 -> 16384.
        assertEquals(225, org.opensearch.knn.index.KNNSettings.burstThreadsFor(64));
        assertEquals(16384, DirectIOReadPool.derivedQueueCapacity(225, 64));

        // Exactly a power of two stays put rather than doubling.
        assertEquals(1024, DirectIOReadPool.derivedQueueCapacity(16, 64));
    }

    /**
     * D1(a): a changed capacity is applied to the live pool without a restart, which is the bug the node
     * found — {@code queue_capacity} was registered dynamic and read once, so a change silently did nothing
     * until a JVM restart.
     *
     * <p>Driven through {@link DirectIOReadPool#queueCapacity()}'s own input rather than through a cluster
     * service, because a unit test has none: the capacity is taken from the setting, so the test asserts on
     * the three things a rebuild must do — the new pool is a different object, {@code pool_rebuilds}
     * increments, and the old pool terminates after its queue drains rather than being interrupted.
     */
    @SneakyThrows
    public void testACapacityChangeRebuildsTheLivePoolWithoutLosingQueuedReads() {
        DirectIOVectorIndexInput.STATS.reset();
        final ThreadPoolExecutor first = (ThreadPoolExecutor) DirectIOReadPool.executor();
        assertNotNull(first);
        assertEquals("the first creation is not a rebuild", 0L, DirectIOVectorIndexInput.STATS.poolRebuilds());
        final int initialCapacity = DirectIOReadPool.liveCapacityForTesting();
        assertEquals(DirectIOReadPool.queueCapacity(), initialCapacity);

        // A read already queued on the old pool, held until after the swap so that the swap has something to
        // drain. One task per thread occupies the pool, the next one sits in the queue.
        final CountDownLatch release = new CountDownLatch(1);
        final CountDownLatch running = new CountDownLatch(first.getMaximumPoolSize());
        final List<Future<?>> inFlight = new ArrayList<>();
        for (int i = 0; i < first.getMaximumPoolSize(); i++) {
            inFlight.add(first.submit(() -> {
                running.countDown();
                release.await();
                return null;
            }));
        }
        assertTrue(running.await(10, TimeUnit.SECONDS));
        final Future<?> queued = first.submit(() -> 7);

        // The swap, driven by a capacity the current pool was not built at.
        final ThreadPoolExecutor second = DirectIOReadPool.rebuildForTesting(first.getMaximumPoolSize(), initialCapacity * 2);
        assertNotSame("a capacity change must publish a new pool, not resize the old queue", first, second);
        assertEquals(1L, DirectIOVectorIndexInput.STATS.poolRebuilds());
        assertEquals(initialCapacity * 2, DirectIOReadPool.liveCapacityForTesting());
        assertSame(
            "and an acquisition at the same capacity reuses it rather than rebuilding again",
            second,
            DirectIOReadPool.rebuildForTesting(first.getMaximumPoolSize(), initialCapacity * 2)
        );
        assertEquals("one capacity change, one rebuild", 1L, DirectIOVectorIndexInput.STATS.poolRebuilds());
        assertEquals(
            "the new queue really is deeper",
            initialCapacity * 2,
            second.getQueue().remainingCapacity() + second.getQueue().size()
        );

        // shutdown(), not shutdownNow(): the old pool must still run what it had, including the queued task.
        assertTrue("the old pool is retired", first.isShutdown());
        release.countDown();
        for (final Future<?> future : inFlight) {
            future.get(10, TimeUnit.SECONDS);
        }
        assertEquals("a read queued before the swap still completes", 7, queued.get(10, TimeUnit.SECONDS));
        assertTrue("and the old pool terminates once drained", first.awaitTermination(10, TimeUnit.SECONDS));
    }

    /** Two concurrent acquisitions across a capacity change create exactly one pool between them. */
    @SneakyThrows
    public void testConcurrentAcquisitionsDuringASwapCreateExactlyOnePool() {
        DirectIOVectorIndexInput.STATS.reset();
        final ThreadPoolExecutor first = (ThreadPoolExecutor) DirectIOReadPool.executor();
        final int threads = first.getMaximumPoolSize();
        final int capacity = DirectIOReadPool.liveCapacityForTesting() * 2;

        final int callers = 8;
        final CountDownLatch start = new CountDownLatch(1);
        final List<Thread> racers = new ArrayList<>();
        final ThreadPoolExecutor[] seen = new ThreadPoolExecutor[callers];
        for (int i = 0; i < callers; i++) {
            final int slot = i;
            final Thread thread = new Thread(() -> {
                try {
                    start.await();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    return;
                }
                seen[slot] = DirectIOReadPool.rebuildForTesting(threads, capacity);
            }, "swap-racer-" + i);
            racers.add(thread);
            thread.start();
        }
        start.countDown();
        for (final Thread thread : racers) {
            thread.join(10_000);
        }
        for (int i = 0; i < callers; i++) {
            assertNotNull("caller " + i + " got no pool", seen[i]);
            assertSame("every caller must see the same pool", seen[0], seen[i]);
            assertFalse("and a usable one", seen[i].isShutdown());
        }
        assertEquals("exactly one rebuild", 1L, DirectIOVectorIndexInput.STATS.poolRebuilds());
    }

    /**
     * {@code queue_peak} is the occupancy high-water mark the derived default is validated against on the
     * node, which is why it exists: the sweep saw 0 declines at both 1,024 and 4,096 and could not tell
     * whether the queue had come near filling at either.
     */
    @SneakyThrows
    public void testQueuePeakRecordsOccupancyAtSubmit() {
        DirectIOVectorIndexInput.STATS.reset();
        assertEquals(0L, DirectIOVectorIndexInput.STATS.queuePeak());
        final ThreadPoolExecutor pool = (ThreadPoolExecutor) DirectIOReadPool.executor();
        final CountDownLatch release = new CountDownLatch(1);
        final CountDownLatch running = new CountDownLatch(pool.getMaximumPoolSize());
        try {
            for (int i = 0; i < pool.getMaximumPoolSize(); i++) {
                pool.submit(() -> {
                    running.countDown();
                    release.await();
                    return null;
                });
            }
            assertTrue(running.await(10, TimeUnit.SECONDS));
            // Every thread is busy, so these three go to the queue and the peak must reach three.
            for (int i = 0; i < 3; i++) {
                pool.submit(() -> null);
                DirectIOReadPool.sampleQueueDepth(pool);
            }
            assertEquals(3L, DirectIOVectorIndexInput.STATS.queuePeak());
        } finally {
            release.countDown();
        }
        // A high-water mark never falls back.
        DirectIOReadPool.sampleQueueDepth(pool);
        assertEquals(3L, DirectIOVectorIndexInput.STATS.queuePeak());
    }

    /** A setting that cannot be read must keep a usable capacity rather than fail a read. */
    public void testAnUnreadableSettingStillYieldsAUsableCapacity() {
        assertTrue(DirectIOReadPool.queueCapacity() >= DirectIOReadPool.MIN_QUEUE_CAPACITY);
        assertTrue(DirectIOReadPool.derivedQueueCapacity(0, 0) >= DirectIOReadPool.MIN_QUEUE_CAPACITY);
        assertTrue(DirectIOReadPool.derivedQueueCapacity(-5, -5) >= DirectIOReadPool.MIN_QUEUE_CAPACITY);
        assertEquals(
            "and an absurd demand is clamped rather than overflowed",
            Integer.MAX_VALUE / 2,
            DirectIOReadPool.derivedQueueCapacity(Integer.MAX_VALUE, 64)
        );
    }

    /**
     * Past the bound a submit is refused rather than queued, and refused is a condition both submit sites
     * already handle: {@code prefetch} counts {@code stageDeclined} and lets a blocking read serve the
     * bytes, {@code refillReadahead} counts {@code readaheadDeclined}. This test pins that the pool really
     * does refuse, because if it silently queued instead, those fallbacks would stay dead code.
     */
    @SneakyThrows
    public void testSubmitIsRefusedPastTheBoundRatherThanQueuedForever() {
        final ThreadPoolExecutor pool = (ThreadPoolExecutor) DirectIOReadPool.executor();
        final CountDownLatch release = new CountDownLatch(1);
        final int toSubmit = pool.getMaximumPoolSize() + pool.getQueue().remainingCapacity() + 1;
        int accepted = 0;
        boolean refused = false;
        try {
            for (int i = 0; i < toSubmit; i++) {
                try {
                    pool.submit(() -> {
                        release.await();
                        return null;
                    });
                    accepted++;
                } catch (RejectedExecutionException e) {
                    refused = true;
                    break;
                }
            }
        } finally {
            release.countDown();
        }
        assertTrue("the pool must refuse past its bound, not queue without limit; accepted " + accepted, refused);
    }
}

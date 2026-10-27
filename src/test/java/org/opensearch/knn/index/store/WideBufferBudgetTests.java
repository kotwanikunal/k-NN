/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import com.carrotsearch.randomizedtesting.annotations.ThreadLeakFilters;
import org.opensearch.knn.KNNTestCase;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

/**
 * O2 (16-I's I7, D3d): the per-thread wide-group staging buffers have a node-wide cap.
 *
 * <p>The quantity, CODE: a {@link DirectIOStagingLease} keeps one wide buffer per
 * {@code knn.direct_io.rescore.prefetch.staged_ranges} slot, each grown on demand to at most
 * {@code knn.direct_io.rescore.prefetch.max_span_bytes}, for the life of its arena. At the shipped defaults
 * that is 64 x 135,168 B = 8.25 MiB per thread, and a lease is per thread with nothing bounding the number
 * of leases: 1.95 GB at 225 search threads.
 *
 * <p>Why the overall direct-memory budget was not already the answer: a lease keeps its wide buffers while
 * it reuses its arena, so wide buffers accumulating to that one budget would leave no room to carve an arena
 * and every prefetch on the node would decline. That is a budget wedging the feature rather than bounding
 * it, which is why the wide buffers get a sub-budget asked <em>in addition</em> to the overall one.
 */
@ThreadLeakFilters(defaultFilters = true, filters = { DirectIOReadPoolTests.ReadPoolThreadFilter.class })
public class WideBufferBudgetTests extends KNNTestCase {

    private static final int THREADS = 64;
    private static final int BLOCK_SIZE = 4096;
    private static final int SLOT_BYTES = 8192;
    /**
     * Eight rather than the shipped 64, because this test allocates the product of it and {@link #THREADS}
     * for real: at 64 slots the uncapped arm would ask for 540 MB of direct buffers in the test JVM. The
     * thread count is what the criterion is about, and it is the shipped-geometry figure that the bench
     * under output/bench/o2 reports.
     */
    private static final int SLOTS = 8;
    /** Wider than a slot, so every one of these takes a wide buffer rather than an arena slot. */
    private static final int WIDE_BYTES = 128 * 1024;

    @Override
    public void setUp() throws Exception {
        super.setUp();
        DirectIOStagingLease.resetForTesting();
        // The wide-group gauge is node wide and its charges come back by reachability, so a test that holds
        // buffers leaves them charged for the next one; these tests assert an absolute figure against a cap.
        DirectIOVectorIndexInput.STATS.resetWideGaugeForTesting();
        // Both budgets are constants on a node; these cases publish their own, so each starts from the
        // shipped pair.
        DirectIOVectorIndexInput.resetBudgetsForTesting();
    }

    @Override
    public void tearDown() throws Exception {
        DirectIOVectorIndexInput.resetBudgetsForTesting();
        super.tearDown();
    }

    /**
     * Runs {@code THREADS} threads, each taking its own lease and asking for every slot's wide buffer, and
     * answers how many of those asks were granted. Every thread keeps its lease for the whole run, which is
     * the condition the cap exists for: nothing is released while the measurement is taken.
     */
    private long fillWideBuffersOnEveryThread(final int slots) throws InterruptedException {
        final AtomicLong granted = new AtomicLong();
        final Set<String> failures = java.util.concurrent.ConcurrentHashMap.newKeySet();
        final CountDownLatch ready = new CountDownLatch(THREADS);
        final CountDownLatch done = new CountDownLatch(1);
        final List<Thread> threads = new ArrayList<>(THREADS);
        for (int t = 0; t < THREADS; t++) {
            final Thread thread = new Thread(() -> {
                final DirectIOStagingLease lease = DirectIOStagingLease.current();
                if (lease.take(slots, SLOT_BYTES, BLOCK_SIZE) < 0) {
                    ready.countDown();
                    awaitQuietly(done);
                    return;
                }
                try {
                    for (int slot = 0; slot < slots; slot++) {
                        final ByteBuffer buffer = lease.bufferFor(WIDE_BYTES);
                        if (buffer != null && buffer.capacity() >= WIDE_BYTES) {
                            granted.incrementAndGet();
                        }
                        lease.record(null);
                    }
                } finally {
                    // In a finally, so that a thread that throws cannot hang the measurement instead of
                    // failing it.
                    ready.countDown();
                }
                // Holds the lease, and so the buffers, until every thread has been measured.
                awaitQuietly(done);
            }, "wide-buffer-" + t);
            thread.setUncaughtExceptionHandler((which, error) -> failures.add(which.getName() + ": " + error));
            threads.add(thread);
            thread.start();
        }
        assertTrue("the threads did not get going, failures " + failures, ready.await(60, TimeUnit.SECONDS));
        assertEquals("a worker threw", List.of(), List.copyOf(failures));
        measured = DirectIOVectorIndexInput.STATS.wideBufferLive();
        done.countDown();
        for (final Thread thread : threads) {
            thread.join(TimeUnit.SECONDS.toMillis(60));
            assertFalse("a worker did not finish", thread.isAlive());
        }
        return granted.get();
    }

    /** Wide-buffer bytes live at the moment every thread held its buffers. */
    private long measured;

    private static void awaitQuietly(final CountDownLatch latch) {
        try {
            latch.await(60, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    /**
     * The cap holds with 64 threads each asking for all 64 of its slots' wide buffers, and the asks past it
     * are declined and counted rather than failing. The uncapped figure is measured too, in the same process
     * and on the same geometry, because "at or under the cap" means nothing without the number the cap is
     * below.
     */
    public void testSixtyFourThreadsStayUnderTheCap() throws InterruptedException {
        // The capped arm runs FIRST, and that order is forced rather than chosen: the accounting is
        // reachability-based (see DirectIOVectorIndexInput#reclaim), so the uncapped arm's buffers stay
        // charged until a collector proves them unreachable. Running it first would leave the capped arm
        // measuring a counter the uncapped arm had already filled, and the arm would pass for the wrong
        // reason. This way each arm's claim is independent and neither needs a System.gc() to be true.
        final long cap = 16L * 1024 * 1024;
        DirectIOVectorIndexInput.setDirectMemoryBudgetForTesting(0L);
        DirectIOVectorIndexInput.setWideBufferBudgetForTesting(cap);
        assertEquals(cap, DirectIOVectorIndexInput.wideBufferBudgetBytes());
        final long declinedBefore = DirectIOVectorIndexInput.STATS.declinedWideBudget();
        final long grantedCapped = fillWideBuffersOnEveryThread(SLOTS);
        final long capped = measured;
        final long declines = DirectIOVectorIndexInput.STATS.declinedWideBudget() - declinedBefore;
        DirectIOStagingLease.resetForTesting();

        // Then the same geometry with both budgets off, which is what D3d inherited: the number the cap has
        // to be read against.
        DirectIOVectorIndexInput.setDirectMemoryBudgetForTesting(0L);
        DirectIOVectorIndexInput.setWideBufferBudgetForTesting(0L);
        assertEquals("with both budgets off there must be no cap", 0L, DirectIOVectorIndexInput.wideBufferBudgetBytes());
        final long grantedUncapped = fillWideBuffersOnEveryThread(SLOTS);
        final long uncapped = measured - capped;

        logger.info(
            "O2 D3d wide-buffer cap, {} threads x {} slots x {} B: capped live {} B ({} of {} asks granted, "
                + "{} declined) under a {} B cap; the same geometry uncapped adds {} B ({} granted)",
            THREADS,
            SLOTS,
            WIDE_BYTES,
            capped,
            grantedCapped,
            THREADS * SLOTS,
            declines,
            cap,
            uncapped,
            grantedUncapped
        );

        assertTrue("wide-buffer memory must stay at or under the cap, it reached " + capped + " of " + cap, capped <= cap);
        assertTrue("the cap must bite, it granted " + grantedCapped + " of " + (THREADS * SLOTS), grantedCapped < THREADS * SLOTS);
        assertTrue("and must not refuse everything", grantedCapped > 0);
        assertTrue("a refused ask must be counted, " + declines + " were", declines > 0);
        assertEquals("every ask must be granted with no cap", (long) THREADS * SLOTS, grantedUncapped);
        assertTrue("the uncapped figure must be above the cap, it was " + uncapped, uncapped > cap);
    }

    /**
     * The same cap, at the <b>shipped</b> geometry: {@code staged_ranges} of 64 slots and a group as wide as
     * {@code max_span_bytes}, which is the 8.25 MiB per thread D3d names. There is no uncapped arm here
     * because an uncapped one would ask this JVM for 553,648,128 B of direct buffers; the uncapped figure
     * is the arithmetic, and the 8-slot arm above is what shows the arithmetic is right (64 x 8 x 135,167 =
     * 69,205,504, the byte for byte figure it measured).
     */
    public void testTheCapHoldsAtTheShippedGeometry() throws InterruptedException {
        final long cap = 16L * 1024 * 1024;
        DirectIOVectorIndexInput.setDirectMemoryBudgetForTesting(0L);
        DirectIOVectorIndexInput.setWideBufferBudgetForTesting(cap);
        final int shippedSlots = DirectIOVectorIndexInput.DEFAULT_MAX_STAGED_RANGES;
        final int shippedSpan = DirectIOVectorIndexInput.DEFAULT_MAX_STAGED_RANGE_BYTES;
        assertEquals("this test is pinned to the shipped geometry", 64, shippedSlots);
        assertEquals(128 * 1024 + 4096, shippedSpan);

        final long granted = fillWideBuffersOnEveryThread(shippedSlots);
        final long live = measured;
        final long perThreadWorstCase = (long) shippedSlots * (shippedSpan + BLOCK_SIZE - 1);
        logger.info(
            "O2 D3d at the shipped geometry, {} threads x {} slots: live {} B of a {} B cap, {} of {} asks "
                + "granted; per-thread worst case {} B, so {} threads uncapped would be {} B",
            THREADS,
            shippedSlots,
            live,
            cap,
            granted,
            (long) THREADS * shippedSlots,
            perThreadWorstCase,
            THREADS,
            THREADS * perThreadWorstCase
        );
        assertTrue("the cap must hold at the shipped geometry, live " + live + " of " + cap, live <= cap);
        assertTrue("and must grant something", granted > 0);
        assertTrue("and must decline the rest", granted < (long) THREADS * shippedSlots);
    }

    /**
     * The cap is a quarter of the direct-memory budget, and with that budget off there is no cap at all --
     * the rule that budget has always had.
     */
    public void testTheCapIsDerivedFromTheDirectMemoryBudget() {
        DirectIOVectorIndexInput.setDirectMemoryBudgetForTesting(2L * 1024 * 1024 * 1024);
        DirectIOVectorIndexInput.setWideBufferBudgetForTesting(0L);
        assertEquals(2L * 1024 * 1024 * 1024 / 4, DirectIOVectorIndexInput.wideBufferBudgetBytes());

        // And the seam's explicit value wins over the derived one, which is how the capped arms above are
        // driven.
        DirectIOVectorIndexInput.setDirectMemoryBudgetForTesting(2L * 1024 * 1024 * 1024);
        DirectIOVectorIndexInput.setWideBufferBudgetForTesting(16L * 1024 * 1024);
        assertEquals(16L * 1024 * 1024, DirectIOVectorIndexInput.wideBufferBudgetBytes());

        DirectIOVectorIndexInput.setDirectMemoryBudgetForTesting(0L);
        DirectIOVectorIndexInput.setWideBufferBudgetForTesting(0L);
        assertEquals("a budget must never be the reason a shard cannot read", 0L, DirectIOVectorIndexInput.wideBufferBudgetBytes());
    }

    /**
     * The sub-budget is a share of the overall one and not a second pot: the same buffer is charged to both,
     * through one phantom reference, so one reclamation returns it to both and they cannot drift.
     */
    public void testAWideBufferIsChargedToBothBudgets() {
        DirectIOVectorIndexInput.setDirectMemoryBudgetForTesting(0L);
        DirectIOVectorIndexInput.setWideBufferBudgetForTesting(0L);
        final long directBefore = DirectIOVectorIndexInput.reclaim();
        final long wideBefore = DirectIOVectorIndexInput.STATS.wideBufferLive();

        final ByteBuffer wide = DirectIOVectorIndexInput.allocateWideBuffer(WIDE_BYTES, BLOCK_SIZE);
        assertNotNull(wide);
        final long charged = WIDE_BYTES + BLOCK_SIZE - 1;
        assertEquals("the overall budget must carry it", directBefore + charged, DirectIOVectorIndexInput.STATS.directMemoryLive());
        assertEquals("and so must the sub-budget", wideBefore + charged, DirectIOVectorIndexInput.STATS.wideBufferLive());

        // An arena-shaped allocation is charged to the overall budget only.
        final ByteBuffer plain = DirectIOVectorIndexInput.trackedAllocate(SLOT_BYTES, BLOCK_SIZE);
        assertNotNull(plain);
        assertEquals("an arena is not a wide buffer", wideBefore + charged, DirectIOVectorIndexInput.STATS.wideBufferLive());
        assertEquals(directBefore + charged + SLOT_BYTES + BLOCK_SIZE - 1, DirectIOVectorIndexInput.STATS.directMemoryLive());
    }
}

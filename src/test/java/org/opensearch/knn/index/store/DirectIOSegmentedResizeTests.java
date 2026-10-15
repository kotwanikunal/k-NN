/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import org.opensearch.knn.KNNTestCase;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Random;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Q's C5: {@code configure} across segments.
 *
 * <p>Three things have to keep holding and one is new. Growing within one segment count keeps every cached
 * vector; shrinking within one keeps every vector that still fits, with the slabs dropped being the oldest
 * of the fullest segments rather than the oldest globally, which is the one documented tie-break change;
 * {@code 0b} releases every slab and every index, and "off is off" stays one volatile read. The new one is a
 * target that crosses a segment-count boundary: it releases, refills, counts and logs, because a key's
 * segment is a function of the count and after a change every cached vector would be in the wrong one.
 *
 * <p>And the property that is load bearing rather than merely true: a {@code configure} with an
 * <b>unchanged</b> target takes no lock at all. It is called once per prefetch burst from the read path, so
 * taking a lock to learn there is nothing to do would put every burst through the lock a fill uses.
 */
public class DirectIOSegmentedResizeTests extends KNNTestCase {

    private static final int TEST_SLAB_BYTES = 64 * 1024;
    private static final int DIMENSION = 32;
    private static final int VECTOR_BYTES = DIMENSION * Float.BYTES;
    private static final int STRIPE_BYTES = 256 * 1024;

    @Override
    public void tearDown() throws Exception {
        DirectIOVectorCache.setAcquisitionCountingForTesting(false);
        DirectIOVectorCache.resetForTesting();
        super.tearDown();
    }

    private void cacheOf(final int segments, final int slabs) {
        DirectIOVectorCache.resetForTesting();
        DirectIOVectorCache.setSlabBytesForTesting(TEST_SLAB_BYTES);
        DirectIOVectorCache.setSegmentsForTesting(segments);
        DirectIOVectorCache.configure((long) slabs * TEST_SLAB_BYTES);
        assertEquals(segments, DirectIOVectorCache.segmentCount());
    }

    // ---------------------------------------------------------------------------------------------------
    // Within one segment count
    // ---------------------------------------------------------------------------------------------------

    /** Growing keeps every cached vector and lets the next miss allocate. */
    public void testGrowingWithinOneCountKeepsEveryVector() {
        cacheOf(16, 16);
        final String file = "/index/grow.vec#1";
        final int vectors = 2_000;
        fill(file, 0, vectors);
        final long entriesBefore = DirectIOVectorCache.entries();
        final long slabsBefore = DirectIOVectorCache.slabs();
        assertEquals("the fixture must fit", vectors, entriesBefore);

        DirectIOVectorCache.configure(64L * TEST_SLAB_BYTES);

        assertEquals("the count did not change", 16, DirectIOVectorCache.segmentCount());
        assertEquals("no re-warm", 0L, DirectIOVectorCache.segmentCountChanges());
        assertEquals("growing must not drop a vector", entriesBefore, DirectIOVectorCache.entries());
        assertEquals("nor release a slab", slabsBefore, DirectIOVectorCache.slabs());
        assertEquals(64L * TEST_SLAB_BYTES, DirectIOVectorCache.maxBytes());
        final float[] read = new float[DIMENSION];
        for (int i = 0; i < vectors; i++) {
            final long offset = offsetOf(i);
            assertTrue("vector " + i + " survived the growth", DirectIOVectorCache.getFloats(file, offset, VECTOR_BYTES, read, 0));
            assertArrayEquals(vector(file, offset), read, 0.0f);
        }
        // And the next misses allocate rather than evict, which is what growing is for: 14,000 vectors
        // would not have fitted in the 16 slabs the store had (8,192 slots) and do fit in the 64 it now has.
        fill(file, vectors, 14_000);
        assertTrue(
            "the next fills took slabs, " + slabsBefore + " to " + DirectIOVectorCache.slabs(),
            DirectIOVectorCache.slabs() > slabsBefore
        );
        assertEquals("and nothing was evicted to hold them", 0L, DirectIOVectorCache.reclaimedSlabs());
        assertEquals(14_000, DirectIOVectorCache.entries());
        DirectIOVectorCache.checkInvariantsForTesting();
    }

    /**
     * Shrinking keeps every vector that still fits, and the slabs dropped are the oldest of the fullest
     * segments. That is the same "keeps what fits" guarantee the unsegmented store gave with a different
     * tie-break, which is what each segment owning its slabs implies.
     */
    public void testShrinkingWithinOneCountKeepsWhatFits() {
        cacheOf(16, 64);
        final String file = "/index/shrink.vec#1";
        final int vectors = 12_000;
        fill(file, 0, vectors);
        final long slabsBefore = DirectIOVectorCache.slabs();
        assertTrue("the fixture must take several slabs, saw " + slabsBefore, slabsBefore >= 8);
        assertEquals(vectors, DirectIOVectorCache.entries());
        final long perSlab = DirectIOVectorCache.slotsPerSlab(VECTOR_BYTES);

        final long target = slabsBefore / 2;
        DirectIOVectorCache.configure(target * TEST_SLAB_BYTES);

        assertEquals("the count did not change", 16, DirectIOVectorCache.segmentCount());
        assertEquals(target, DirectIOVectorCache.slabs());
        assertTrue("something was kept", DirectIOVectorCache.entries() > 0);
        assertTrue(
            "and no more than the new target can hold: " + DirectIOVectorCache.entries() + " in " + target + " slabs",
            DirectIOVectorCache.entries() <= target * perSlab
        );
        // Every vector the store still claims serves its own bytes, which is the only thing a shrink must
        // never get wrong.
        final float[] read = new float[DIMENSION];
        int served = 0;
        for (int i = 0; i < vectors; i++) {
            final long offset = offsetOf(i);
            if (DirectIOVectorCache.getFloats(file, offset, VECTOR_BYTES, read, 0)) {
                assertArrayEquals("vector " + i + " after a shrink", vector(file, offset), read, 0.0f);
                served++;
            }
        }
        assertTrue("a shrink to half keeps a useful share, served " + served + " of " + vectors, served > vectors / 8);
        DirectIOVectorCache.checkInvariantsForTesting();
        logger.info("16 segments shrunk from {} slabs to {}: {} of {} vectors kept", slabsBefore, target, served, vectors);
    }

    /** Off is off: every slab released, every index gone, every counter at zero, and one volatile read. */
    public void testConfigureZeroReleasesEverything() {
        DirectIOSlab.resetCountersForTesting();
        cacheOf(16, 16);
        final String file = "/index/off.vec#1";
        fill(file, 0, 3_000);
        assertTrue(DirectIOVectorCache.slabs() > 0);
        assertTrue(DirectIOVectorCache.entries() > 0);
        assertTrue(DirectIOVectorCache.isOn());
        final long allocated = DirectIOSlab.allocatedCount();
        assertTrue("the fixture must have allocated slabs", allocated > 0);

        DirectIOVectorCache.configure(0);
        DirectIOVectorCache.awaitReleasesForTesting();

        assertEquals("every slab released", 0L, DirectIOVectorCache.slabs());
        assertEquals("every entry gone", 0L, DirectIOVectorCache.entries());
        assertEquals(0L, DirectIOVectorCache.bytes());
        assertEquals(0L, DirectIOVectorCache.maxBytes());
        assertEquals("no index table anywhere", 0, DirectIOVectorCache.indexedFiles());
        assertFalse(DirectIOVectorCache.hasIndexFor(file));
        assertFalse("off is off", DirectIOVectorCache.isOn());
        assertFalse("and span fill with it", DirectIOVectorCache.isSpanFillOn());
        assertFalse("and read-around", DirectIOVectorCache.readAroundBytes(1_024L) > 0);
        assertEquals("no slab leaked", DirectIOSlab.allocatedCount(), DirectIOSlab.freedCount());
        assertEquals("the live population is zero", 0L, DirectIOSlab.allocatedCount() - DirectIOSlab.freedCount());
        // And the entry points short-circuit: no lock is taken at all with the store off.
        DirectIOVectorCache.setAcquisitionCountingForTesting(true);
        final float[] read = new float[DIMENSION];
        assertFalse(DirectIOVectorCache.getFloats(file, offsetOf(0), VECTOR_BYTES, read, 0));
        assertFalse(DirectIOVectorCache.contains(file, offsetOf(0), VECTOR_BYTES));
        assertEquals(-1, DirectIOVectorCache.selectMissing(file, 0L, VECTOR_BYTES, new int[] { 0, 1 }, 2, new int[2]));
        put(file, offsetOf(0));
        assertEquals("not one lock acquisition with the store off", 0L, DirectIOVectorCache.readAcquisitionsForTesting());
        assertEquals(0L, DirectIOVectorCache.writeAcquisitionsForTesting());
        assertEquals(0L, DirectIOVectorCache.brokerLockAcquisitionsForTesting());
        DirectIOVectorCache.setAcquisitionCountingForTesting(false);
        DirectIOVectorCache.checkInvariantsForTesting();
    }

    // ---------------------------------------------------------------------------------------------------
    // Across a segment-count boundary
    // ---------------------------------------------------------------------------------------------------

    /**
     * A target that crosses a segment-count boundary changes the count, releases every slab, refills, and
     * counts the change. The count is the derived one here, not a pinned one, because an operator's
     * {@code max_bytes} is what moves it on a node.
     */
    public void testATargetCrossingACountBoundaryResegments() {
        DirectIOVectorCache.resetForTesting();
        DirectIOVectorCache.setSlabBytesForTesting(TEST_SLAB_BYTES);
        DirectIOVectorCache.configure(128L * TEST_SLAB_BYTES);
        final int before = DirectIOVectorCache.segmentCount();
        assertTrue("128 test slabs derives more than one segment on any core count, saw " + before, before > 1);
        final String file = "/index/resegment.vec#1";
        fill(file, 0, 4_000);
        assertEquals(4_000, DirectIOVectorCache.entries());
        assertTrue(DirectIOVectorCache.slabs() > 0);
        assertEquals(0L, DirectIOVectorCache.segmentCountChanges());

        DirectIOVectorCache.configure(32L * TEST_SLAB_BYTES);

        assertEquals("32 test slabs is four segments at any core count", 4, DirectIOVectorCache.segmentCount());
        assertEquals("the change is counted", 1L, DirectIOVectorCache.segmentCountChanges());
        assertEquals("and it cost a re-warm: every slab released", 0L, DirectIOVectorCache.slabs());
        assertEquals("and every entry", 0L, DirectIOVectorCache.entries());
        assertEquals("and every index table", 0, DirectIOVectorCache.indexedFiles());
        assertEquals("the new target is in force", 32L * TEST_SLAB_BYTES, DirectIOVectorCache.maxBytes());
        assertTrue("the store is on", DirectIOVectorCache.isOn());

        // It refills, into the new segments, and serves its own bytes.
        fill(file, 0, 4_000);
        assertEquals(4_000, DirectIOVectorCache.entries());
        final float[] read = new float[DIMENSION];
        for (int i = 0; i < 4_000; i++) {
            final long offset = offsetOf(i);
            assertTrue("vector " + i + " after the re-warm", DirectIOVectorCache.getFloats(file, offset, VECTOR_BYTES, read, 0));
            assertArrayEquals(vector(file, offset), read, 0.0f);
        }
        DirectIOVectorCache.checkInvariantsForTesting();

        // Back up across the boundary: the count changes again and is counted again.
        DirectIOVectorCache.configure(128L * TEST_SLAB_BYTES);
        assertEquals(before, DirectIOVectorCache.segmentCount());
        assertEquals(2L, DirectIOVectorCache.segmentCountChanges());
        assertEquals(0L, DirectIOVectorCache.entries());
        DirectIOVectorCache.awaitReleasesForTesting();
        DirectIOVectorCache.checkInvariantsForTesting();
    }

    /** A resize that does not cross a boundary does not change the count and does not count a change. */
    public void testAResizeWithinABoundaryDoesNotResegment() {
        DirectIOVectorCache.resetForTesting();
        DirectIOVectorCache.setSlabBytesForTesting(TEST_SLAB_BYTES);
        DirectIOVectorCache.configure(32L * TEST_SLAB_BYTES);
        assertEquals(4, DirectIOVectorCache.segmentCount());
        final String file = "/index/within.vec#1";
        fill(file, 0, 2_000);
        final long entries = DirectIOVectorCache.entries();

        // 32 to 63 slabs are all four segments: previousPowerOfTwo(63 / 8) is 4.
        DirectIOVectorCache.configure(63L * TEST_SLAB_BYTES);

        assertEquals(4, DirectIOVectorCache.segmentCount());
        assertEquals("no change to count", 0L, DirectIOVectorCache.segmentCountChanges());
        assertEquals("and no re-warm", entries, DirectIOVectorCache.entries());
        DirectIOVectorCache.checkInvariantsForTesting();
    }

    // ---------------------------------------------------------------------------------------------------
    // The steady-state path
    // ---------------------------------------------------------------------------------------------------

    /**
     * <b>A {@code configure} with an unchanged target takes no lock at all</b>, which is the once-per-burst
     * path and is load bearing: {@code applyConfiguredSize} is called from the read path at every prefetch
     * burst boundary.
     */
    public void testAnUnchangedTargetTakesNoLock() {
        cacheOf(16, 16);
        fill("/index/noop.vec#1", 0, 500);
        DirectIOVectorCache.setAcquisitionCountingForTesting(true);
        for (int i = 0; i < 10_000; i++) {
            DirectIOVectorCache.configure(16L * TEST_SLAB_BYTES);
        }
        final long read = DirectIOVectorCache.readAcquisitionsForTesting();
        final long write = DirectIOVectorCache.writeAcquisitionsForTesting();
        final long broker = DirectIOVectorCache.brokerLockAcquisitionsForTesting();
        DirectIOVectorCache.setAcquisitionCountingForTesting(false);
        assertEquals("10,000 no-op resizes took no segment read lock", 0L, read);
        assertEquals("nor a segment write lock", 0L, write);
        assertEquals("nor the broker's resize lock", 0L, broker);
        assertEquals("and changed nothing", 16L * TEST_SLAB_BYTES, DirectIOVectorCache.maxBytes());
    }

    /** And a target that HAS changed takes the broker's lock, so the counter above is measuring something. */
    public void testAChangedTargetTakesTheBrokersLock() {
        cacheOf(16, 16);
        DirectIOVectorCache.setAcquisitionCountingForTesting(true);
        DirectIOVectorCache.configure(32L * TEST_SLAB_BYTES);
        final long broker = DirectIOVectorCache.brokerLockAcquisitionsForTesting();
        DirectIOVectorCache.setAcquisitionCountingForTesting(false);
        assertTrue("a real resize takes the resize lock, saw " + broker, broker >= 1);
    }

    // ---------------------------------------------------------------------------------------------------
    // Under load
    // ---------------------------------------------------------------------------------------------------

    /** A resize racing lookups and fills at 32 threads leaves the invariants clean and no vector crossed. */
    public void testResizeRacingLookupsAndFills() throws Exception {
        cacheOf(16, 32);
        final String file = "/index/race.vec#1";
        final int threads = 32;
        final int keys = 20_000;
        final AtomicBoolean stop = new AtomicBoolean();
        final AtomicReference<Throwable> failure = new AtomicReference<>();
        final CountDownLatch start = new CountDownLatch(1);
        final CountDownLatch done = new CountDownLatch(threads + 1);

        for (int t = 0; t < threads; t++) {
            final int seed = t;
            final Thread worker = new Thread(() -> {
                try {
                    final Random random = new Random(900L + seed);
                    final float[] read = new float[DIMENSION];
                    start.await();
                    while (stop.get() == false) {
                        final int ordinal = random.nextInt(keys);
                        final long offset = offsetOf(ordinal);
                        if (DirectIOVectorCache.getFloats(file, offset, VECTOR_BYTES, read, 0)) {
                            final float[] wanted = vector(file, offset);
                            for (int i = 0; i < DIMENSION; i++) {
                                if (read[i] != wanted[i]) {
                                    throw new AssertionError("a crossed vector at ordinal " + ordinal);
                                }
                            }
                        } else {
                            put(file, offset);
                        }
                    }
                } catch (Throwable e) {
                    failure.compareAndSet(null, e);
                } finally {
                    done.countDown();
                }
            }, "c5-worker-" + t);
            worker.setDaemon(true);
            worker.start();
        }

        final Thread resizer = new Thread(() -> {
            try {
                start.await();
                // Up and down, both within one segment count (32 to 63 slabs is four segments at any core
                // count, and the pinned 16 holds throughout) and across a pinned-count boundary is covered
                // by the resegment test; here the point is the race against live traffic.
                final long[] targets = { 32, 48, 16, 64, 24, 32 };
                for (int round = 0; round < 8; round++) {
                    for (final long slabs : targets) {
                        DirectIOVectorCache.configure(slabs * TEST_SLAB_BYTES);
                        Thread.sleep(5);
                    }
                }
            } catch (Throwable e) {
                failure.compareAndSet(null, e);
            } finally {
                done.countDown();
            }
        }, "c5-resizer");
        resizer.setDaemon(true);
        resizer.start();

        start.countDown();
        resizer.join(TimeUnit.SECONDS.toMillis(120));
        stop.set(true);
        assertTrue(done.await(2, TimeUnit.MINUTES));
        if (failure.get() != null) {
            throw new AssertionError(failure.get());
        }
        DirectIOVectorCache.checkInvariantsForTesting();
        DirectIOVectorCache.configure(0);
        DirectIOVectorCache.awaitReleasesForTesting();
        assertEquals("no slab leaked across the race", 0L, DirectIOVectorCache.slabs());
        DirectIOVectorCache.checkInvariantsForTesting();
    }

    // ---------------------------------------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------------------------------------

    /** Vectors spread one per stripe at first and then packed, so every segment is used. */
    private static void fill(final String file, final int from, final int to) {
        for (int i = from; i < to; i++) {
            put(file, offsetOf(i));
        }
    }

    private static long offsetOf(final int ordinal) {
        // 64 vectors to a stripe, so 2,000 vectors cover 32 stripes and 12,000 cover 188.
        final int stripe = ordinal / 64;
        return (long) stripe * STRIPE_BYTES + (long) (ordinal % 64) * VECTOR_BYTES;
    }

    private static void put(final String fileId, final long offset) {
        final ByteBuffer staging = ByteBuffer.allocate(VECTOR_BYTES).order(ByteOrder.LITTLE_ENDIAN);
        final float[] vector = vector(fileId, offset);
        for (int i = 0; i < vector.length; i++) {
            staging.putFloat(i * Float.BYTES, vector[i]);
        }
        DirectIOVectorCache.put(fileId, offset, staging, 0, VECTOR_BYTES);
    }

    private static float[] vector(final String fileId, final long offset) {
        final long seed = fileId.hashCode() * 31L + offset;
        final float[] vector = new float[DIMENSION];
        for (int i = 0; i < DIMENSION; i++) {
            vector[i] = seed + i * 0.5f;
        }
        return vector;
    }
}

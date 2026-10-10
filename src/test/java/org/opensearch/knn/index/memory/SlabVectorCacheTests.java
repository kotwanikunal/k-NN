/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.memory;

import org.opensearch.knn.KNNTestCase;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

/**
 * I2's acceptance tests for {@link SlabVectorCache}: lookup, insert and eviction under concurrency, the
 * second chance, release on segment close, and vector sizes from 128 to 4,096 dimensions.
 *
 * <p>The slab size is a test parameter rather than the shipped 32 MiB, so that a handful of inserts fills a
 * slab and eviction is reached in a unit test instead of after a gigabyte of vectors. Everything else is the
 * shipped path: real slabs, real native memory, the real index.
 */
public class SlabVectorCacheTests extends KNNTestCase {

    /** 64 KiB: 128 slots for a 128-dim vector, 4 for a 4,096-dim one. */
    private static final int TEST_SLAB_BYTES = 64 * 1024;

    private SlabVectorCache cache;

    @Override
    public void tearDown() throws Exception {
        if (cache != null) {
            cache.resetForTesting();
            cache = null;
        }
        super.tearDown();
    }

    private SlabVectorCache cacheOf(final int slabs) {
        cache = new SlabVectorCache(SlabVectorCache.DIRECT_SOURCE, TEST_SLAB_BYTES);
        cache.configure((long) slabs * TEST_SLAB_BYTES);
        return cache;
    }

    // ---------------------------------------------------------------------------------------------------
    // Round trip, at every vector size the plugin supports
    // ---------------------------------------------------------------------------------------------------

    public void testRoundTripAtEveryVectorSize() {
        final SlabVectorCache cache = cacheOf(4);
        final Object core = new Object();
        for (final int dimension : new int[] { 128, 256, 768, 1024, 2048, 4096 }) {
            final int bytes = dimension * Float.BYTES;
            final int slots = cache.slotsPerSlab(bytes);
            assertTrue("a " + dimension + "-dim vector should fit a test slab", slots >= 1);
            for (int ordinal = 0; ordinal < Math.min(slots, 8); ordinal++) {
                put(cache, core, dimension, ordinal, dimension);
            }
            final float[] read = new float[dimension];
            for (int ordinal = 0; ordinal < Math.min(slots, 8); ordinal++) {
                assertTrue("dimension " + dimension + " ordinal " + ordinal, cache.get(core, dimension, ordinal, bytes, read));
                assertArrayEquals(vector(core, dimension, ordinal, dimension), read, 0.0f);
            }
            cache.checkInvariantsForTesting();
        }
    }

    public void testMissesAreMissesAndNotWrongVectors() {
        final SlabVectorCache cache = cacheOf(1);
        final Object core = new Object();
        final Object otherCore = new Object();
        put(cache, core, 0, 5, 128);
        final float[] read = new float[128];

        assertFalse("another segment core", cache.get(otherCore, 0, 5, 512, read));
        assertFalse("another field", cache.get(core, 1, 5, 512, read));
        assertFalse("another ordinal", cache.get(core, 0, 6, 512, read));
        assertFalse("another vector length", cache.get(core, 0, 5, 1024, read));
        assertTrue(cache.get(core, 0, 5, 512, read));
    }

    public void testOffHoldsNothingAndCostsNothing() {
        final SlabVectorCache cache = cacheOf(0);
        assertFalse(cache.isOn());
        final Object core = new Object();
        put(cache, core, 0, 1, 128);
        assertEquals(0, cache.entries());
        assertEquals(0, cache.slabCount());
        assertFalse(cache.get(core, 0, 1, 512, new float[128]));
        // With the cache off a lookup must not even count as a miss: it never asked.
        assertEquals(0, cache.hits());
        assertEquals(0, cache.misses());
    }

    public void testRecachingAnOrdinalReplacesItRatherThanAddingIt() {
        final SlabVectorCache cache = cacheOf(1);
        final Object core = new Object();
        put(cache, core, 0, 3, 128);
        put(cache, core, 0, 3, 128);
        put(cache, core, 0, 3, 128);
        assertEquals(1, cache.entries());
        assertEquals(512, cache.bytes());
        final float[] read = new float[128];
        assertTrue(cache.get(core, 0, 3, 512, read));
        assertArrayEquals(vector(core, 0, 3, 128), read, 0.0f);
        cache.checkInvariantsForTesting();
    }

    // ---------------------------------------------------------------------------------------------------
    // Eviction and the second chance
    // ---------------------------------------------------------------------------------------------------

    public void testFifoEvictionDropsTheColdOnes() {
        final SlabVectorCache cache = cacheOf(1);
        final Object core = new Object();
        final int slots = cache.slotsPerSlab(512);
        for (int ordinal = 0; ordinal < slots; ordinal++) {
            put(cache, core, 0, ordinal, 128);
        }
        assertEquals(slots, cache.entries());
        assertEquals(0, cache.reclaimedSlabs());

        // One more vector than the slab holds, and nothing has been read, so nothing earns a second chance.
        put(cache, core, 0, slots, 128);
        assertEquals(1, cache.reclaimedSlabs());
        assertEquals(0, cache.readmissions());
        assertEquals(1, cache.entries());
        final float[] read = new float[128];
        assertTrue("the vector that caused the eviction", cache.get(core, 0, slots, 512, read));
        for (int ordinal = 0; ordinal < slots; ordinal++) {
            assertFalse("ordinal " + ordinal + " should have gone with its slab", cache.get(core, 0, ordinal, 512, read));
        }
        cache.checkInvariantsForTesting();
    }

    public void testASecondChanceKeepsTheVectorsThatWereRead() {
        final SlabVectorCache cache = cacheOf(1);
        final Object core = new Object();
        final int slots = cache.slotsPerSlab(512);
        for (int ordinal = 0; ordinal < slots; ordinal++) {
            put(cache, core, 0, ordinal, 128);
        }
        final float[] read = new float[128];
        final int touched = 10;
        for (int ordinal = 0; ordinal < touched; ordinal++) {
            assertTrue(cache.get(core, 0, ordinal, 512, read));
        }

        put(cache, core, 0, slots, 128);

        assertEquals(1, cache.reclaimedSlabs());
        assertEquals(touched, cache.readmissions());
        assertEquals(touched + 1, cache.entries());
        for (int ordinal = 0; ordinal < touched; ordinal++) {
            assertTrue("ordinal " + ordinal + " was read, so it should have survived", cache.get(core, 0, ordinal, 512, read));
            assertArrayEquals(vector(core, 0, ordinal, 128), read, 0.0f);
        }
        for (int ordinal = touched; ordinal < slots; ordinal++) {
            assertFalse("ordinal " + ordinal + " was never read", cache.get(core, 0, ordinal, 512, read));
        }
        cache.checkInvariantsForTesting();
    }

    /** The cap is what stops a wholly hot slab from re-admitting itself and leaving no room. */
    public void testTheSecondChanceIsCappedAtHalfASlab() {
        final SlabVectorCache cache = cacheOf(1);
        final Object core = new Object();
        final int slots = cache.slotsPerSlab(512);
        for (int ordinal = 0; ordinal < slots; ordinal++) {
            put(cache, core, 0, ordinal, 128);
        }
        final float[] read = new float[128];
        for (int ordinal = 0; ordinal < slots; ordinal++) {
            assertTrue(cache.get(core, 0, ordinal, 512, read));
        }

        put(cache, core, 0, slots, 128);

        assertEquals(slots / 2, cache.readmissions());
        assertEquals(slots / 2 + 1, cache.entries());
        for (int ordinal = 0; ordinal < slots / 2; ordinal++) {
            assertTrue("the oldest half survives first", cache.get(core, 0, ordinal, 512, read));
        }
        cache.checkInvariantsForTesting();
    }

    /** A survivor is read back from the slot it was moved to, which is a different slot than it went in at. */
    public void testASurvivorsBytesMoveWithIt() {
        final SlabVectorCache cache = cacheOf(1);
        final Object core = new Object();
        final int slots = cache.slotsPerSlab(512);
        for (int ordinal = 0; ordinal < slots; ordinal++) {
            put(cache, core, 0, ordinal, 128);
        }
        final float[] read = new float[128];
        // Read the last two, so compaction has to move them from the end of the slab to the front.
        assertTrue(cache.get(core, 0, slots - 2, 512, read));
        assertTrue(cache.get(core, 0, slots - 1, 512, read));

        put(cache, core, 0, slots, 128);

        assertEquals(2, cache.readmissions());
        assertTrue(cache.get(core, 0, slots - 2, 512, read));
        assertArrayEquals(vector(core, 0, slots - 2, 128), read, 0.0f);
        assertTrue(cache.get(core, 0, slots - 1, 512, read));
        assertArrayEquals(vector(core, 0, slots - 1, 128), read, 0.0f);
        cache.checkInvariantsForTesting();
    }

    public void testEvictionAcrossSeveralSlabsTakesTheOldestFirst() {
        final SlabVectorCache cache = cacheOf(3);
        final Object core = new Object();
        final int slots = cache.slotsPerSlab(512);
        for (int ordinal = 0; ordinal < slots * 3; ordinal++) {
            put(cache, core, 0, ordinal, 128);
        }
        assertEquals(3, cache.slabCount());
        assertEquals(slots * 3, cache.entries());

        put(cache, core, 0, slots * 3, 128);

        assertEquals(1, cache.reclaimedSlabs());
        assertEquals(3, cache.slabCount());
        final float[] read = new float[128];
        // The first slab's worth is gone; the second and third are not.
        assertFalse(cache.get(core, 0, 0, 512, read));
        assertTrue(cache.get(core, 0, slots, 512, read));
        assertTrue(cache.get(core, 0, slots * 2, 512, read));
        assertTrue(cache.get(core, 0, slots * 3, 512, read));
        cache.checkInvariantsForTesting();
    }

    /** Two vector lengths in one cache: a slab reopened for the other length drops what it held. */
    public void testTwoVectorLengthsShareTheSlabBudget() {
        final SlabVectorCache cache = cacheOf(2);
        final Object core = new Object();
        final int smallSlots = cache.slotsPerSlab(512);
        final int largeSlots = cache.slotsPerSlab(4096);
        for (int ordinal = 0; ordinal < smallSlots; ordinal++) {
            put(cache, core, 0, ordinal, 128);
        }
        for (int ordinal = 0; ordinal < largeSlots; ordinal++) {
            put(cache, core, 1, ordinal, 1024);
        }
        assertEquals(2, cache.slabCount());
        assertEquals(smallSlots + largeSlots, cache.entries());
        assertEquals(smallSlots * 512L + largeSlots * 4096L, cache.bytes());

        // A third slab's worth of small vectors has to come out of the two slabs in service.
        for (int ordinal = smallSlots; ordinal < smallSlots + 4; ordinal++) {
            put(cache, core, 0, ordinal, 128);
        }
        assertEquals(2, cache.slabCount());
        cache.checkInvariantsForTesting();
        final float[] read = new float[128];
        assertTrue(cache.get(core, 0, smallSlots + 3, 512, read));
    }

    // ---------------------------------------------------------------------------------------------------
    // Segment close
    // ---------------------------------------------------------------------------------------------------

    public void testSegmentCloseReleasesOnlyThatSegmentsEntries() {
        final SlabVectorCache cache = cacheOf(2);
        final Object closing = new Object();
        final Object staying = new Object();
        for (int ordinal = 0; ordinal < 20; ordinal++) {
            put(cache, closing, 0, ordinal, 128);
            put(cache, closing, 1, ordinal, 128);
            put(cache, staying, 0, ordinal, 128);
        }
        assertEquals(60, cache.entries());

        cache.releaseSegment(closing);

        assertEquals(20, cache.entries());
        assertEquals(20 * 512L, cache.bytes());
        cache.checkInvariantsForTesting();
        final float[] read = new float[128];
        for (int ordinal = 0; ordinal < 20; ordinal++) {
            assertFalse(cache.get(closing, 0, ordinal, 512, read));
            assertFalse(cache.get(closing, 1, ordinal, 512, read));
            assertTrue(cache.get(staying, 0, ordinal, 512, read));
            assertArrayEquals(vector(staying, 0, ordinal, 128), read, 0.0f);
        }
    }

    /** A released segment's slots must not be re-admitted by a second chance they earned before the close. */
    public void testAReleasedSegmentIsNotBroughtBackByASecondChance() {
        final SlabVectorCache cache = cacheOf(1);
        final Object closing = new Object();
        final int slots = cache.slotsPerSlab(512);
        for (int ordinal = 0; ordinal < slots; ordinal++) {
            put(cache, closing, 0, ordinal, 128);
        }
        final float[] read = new float[128];
        for (int ordinal = 0; ordinal < 4; ordinal++) {
            assertTrue(cache.get(closing, 0, ordinal, 512, read));
        }
        cache.releaseSegment(closing);
        assertEquals(0, cache.entries());

        final Object next = new Object();
        put(cache, next, 0, 0, 128);

        assertEquals(0, cache.readmissions());
        assertEquals(1, cache.entries());
        cache.checkInvariantsForTesting();
    }

    public void testReleasingAnUnknownSegmentDoesNothing() {
        final SlabVectorCache cache = cacheOf(1);
        final Object core = new Object();
        put(cache, core, 0, 0, 128);
        cache.releaseSegment(new Object());
        assertEquals(1, cache.entries());
        cache.checkInvariantsForTesting();
    }

    // ---------------------------------------------------------------------------------------------------
    // Resize
    // ---------------------------------------------------------------------------------------------------

    public void testResizingUpKeepsTheVectorsAndResizingDownReturnsTheSlabs() {
        final SlabVectorCache cache = cacheOf(1);
        final Object core = new Object();
        final int slots = cache.slotsPerSlab(512);
        for (int ordinal = 0; ordinal < slots; ordinal++) {
            put(cache, core, 0, ordinal, 128);
        }
        assertEquals(1, cache.slabCount());
        final float[] read = new float[128];

        cache.configure(4L * TEST_SLAB_BYTES);
        assertEquals(4L * TEST_SLAB_BYTES, cache.maxBytes());
        assertEquals("growing must not drop a vector", slots, cache.entries());
        assertTrue(cache.get(core, 0, 0, 512, read));
        for (int ordinal = slots; ordinal < slots * 4; ordinal++) {
            put(cache, core, 0, ordinal, 128);
        }
        assertEquals(4, cache.slabCount());
        assertEquals(0, cache.reclaimedSlabs());

        cache.configure(2L * TEST_SLAB_BYTES);
        assertEquals(2, cache.slabCount());
        assertEquals(2L * TEST_SLAB_BYTES, cache.slabBytes());
        assertEquals("the two oldest slabs' vectors go", slots * 2, cache.entries());
        cache.checkInvariantsForTesting();

        cache.configure(0);
        assertEquals(0, cache.slabCount());
        assertEquals(0, cache.entries());
        assertEquals(0, cache.bytes());
        assertFalse(cache.isOn());
    }

    public void testATargetUnderOneSlabHoldsNothing() {
        final SlabVectorCache cache = cacheOf(0);
        cache.configure(TEST_SLAB_BYTES - 1);
        assertEquals(0, cache.maxBytes());
        assertFalse(cache.isOn());
    }

    // ---------------------------------------------------------------------------------------------------
    // Concurrency
    // ---------------------------------------------------------------------------------------------------

    /**
     * Eight threads looking up, inserting and evicting at once against a cache far smaller than their key
     * space, so eviction and compaction run throughout. Every vector's content is derived from its key, so
     * a hit that returns the wrong bytes is caught at the point of the hit rather than by a checksum at the
     * end, and a torn read of a slot being compacted would show up as a mismatch.
     */
    public void testConcurrentLookupInsertAndEvict() throws Exception {
        final SlabVectorCache cache = cacheOf(4);
        final int dimension = 128;
        final int bytes = dimension * Float.BYTES;
        final Object[] cores = new Object[] { new Object(), new Object(), new Object() };
        final int ordinals = 20_000;
        final int threads = 8;
        final int operationsPerThread = 60_000;

        final CountDownLatch start = new CountDownLatch(1);
        final AtomicReference<Throwable> failure = new AtomicReference<>();
        final List<Thread> workers = new ArrayList<>();
        for (int t = 0; t < threads; t++) {
            final int seed = t;
            final Thread worker = new Thread(() -> {
                final Random random = new Random(seed);
                final float[] read = new float[dimension];
                final ByteBuffer staging = ByteBuffer.allocate(bytes).order(ByteOrder.LITTLE_ENDIAN);
                try {
                    start.await();
                    for (int i = 0; i < operationsPerThread; i++) {
                        final Object core = cores[random.nextInt(cores.length)];
                        final int field = random.nextInt(2);
                        final int ordinal = random.nextInt(ordinals);
                        if (cache.get(core, field, ordinal, bytes, read)) {
                            final float[] expected = vector(core, field, ordinal, dimension);
                            for (int d = 0; d < dimension; d++) {
                                if (read[d] != expected[d]) {
                                    throw new AssertionError(
                                        "a hit for ordinal " + ordinal + " returned the wrong bytes at " + d + ": " + read[d]
                                    );
                                }
                            }
                        } else {
                            fill(staging, vector(core, field, ordinal, dimension));
                            cache.put(core, field, ordinal, staging, 0, bytes);
                        }
                    }
                } catch (Throwable e) {
                    failure.compareAndSet(null, e);
                }
            }, "slab-cache-" + t);
            worker.setDaemon(true);
            workers.add(worker);
            worker.start();
        }
        start.countDown();
        for (final Thread worker : workers) {
            worker.join(TimeUnit.MINUTES.toMillis(2));
            assertFalse("worker " + worker.getName() + " did not finish", worker.isAlive());
        }
        if (failure.get() != null) {
            throw new AssertionError("a worker failed", failure.get());
        }

        cache.checkInvariantsForTesting();
        assertEquals(4, cache.slabCount());
        assertTrue("the run should have evicted", cache.reclaimedSlabs() > 0);
        assertTrue("the run should have hit", cache.hits() > 0);
        assertEquals((long) threads * operationsPerThread, cache.hits() + cache.misses());
        assertTrue("entries cannot exceed the slabs' slots", cache.entries() <= 4L * cache.slotsPerSlab(bytes));
    }

    /** A segment closing while other threads are reading and filling it must not produce a wrong vector. */
    public void testConcurrentReleaseWhileReading() throws Exception {
        final SlabVectorCache cache = cacheOf(2);
        final int dimension = 128;
        final int bytes = dimension * Float.BYTES;
        final Object[] cores = new Object[] { new Object(), new Object(), new Object(), new Object() };
        final CountDownLatch start = new CountDownLatch(1);
        final AtomicReference<Throwable> failure = new AtomicReference<>();
        final List<Thread> workers = new ArrayList<>();
        for (int t = 0; t < 6; t++) {
            final int seed = t;
            final Thread worker = new Thread(() -> {
                final Random random = new Random(seed);
                final float[] read = new float[dimension];
                final ByteBuffer staging = ByteBuffer.allocate(bytes).order(ByteOrder.LITTLE_ENDIAN);
                try {
                    start.await();
                    for (int i = 0; i < 20_000; i++) {
                        final Object core = cores[random.nextInt(cores.length)];
                        final int ordinal = random.nextInt(2_000);
                        if (seed == 0 && i % 500 == 0) {
                            cache.releaseSegment(cores[random.nextInt(cores.length)]);
                            continue;
                        }
                        if (cache.get(core, 0, ordinal, bytes, read)) {
                            assertArrayEquals(vector(core, 0, ordinal, dimension), read, 0.0f);
                        } else {
                            fill(staging, vector(core, 0, ordinal, dimension));
                            cache.put(core, 0, ordinal, staging, 0, bytes);
                        }
                    }
                } catch (Throwable e) {
                    failure.compareAndSet(null, e);
                }
            }, "slab-cache-release-" + t);
            worker.setDaemon(true);
            workers.add(worker);
            worker.start();
        }
        start.countDown();
        for (final Thread worker : workers) {
            worker.join(TimeUnit.MINUTES.toMillis(2));
            assertFalse(worker.isAlive());
        }
        if (failure.get() != null) {
            throw new AssertionError("a worker failed", failure.get());
        }
        cache.checkInvariantsForTesting();
    }

    /** Resizing under load, which is what I6's setting flip does on a live node. */
    public void testConcurrentResizeWhileReading() throws Exception {
        final SlabVectorCache cache = cacheOf(2);
        final int dimension = 128;
        final int bytes = dimension * Float.BYTES;
        final Object core = new Object();
        final CountDownLatch start = new CountDownLatch(1);
        final AtomicReference<Throwable> failure = new AtomicReference<>();
        final List<Thread> workers = new ArrayList<>();
        for (int t = 0; t < 6; t++) {
            final int seed = t;
            final Thread worker = new Thread(() -> {
                final Random random = new Random(seed);
                final float[] read = new float[dimension];
                final ByteBuffer staging = ByteBuffer.allocate(bytes).order(ByteOrder.LITTLE_ENDIAN);
                try {
                    start.await();
                    for (int i = 0; i < 10_000; i++) {
                        if (seed == 0 && i % 200 == 0) {
                            cache.configure((long) (1 + random.nextInt(4)) * TEST_SLAB_BYTES);
                            continue;
                        }
                        final int ordinal = random.nextInt(5_000);
                        if (cache.get(core, 0, ordinal, bytes, read)) {
                            assertArrayEquals(vector(core, 0, ordinal, dimension), read, 0.0f);
                        } else {
                            fill(staging, vector(core, 0, ordinal, dimension));
                            cache.put(core, 0, ordinal, staging, 0, bytes);
                        }
                    }
                } catch (Throwable e) {
                    failure.compareAndSet(null, e);
                }
            }, "slab-cache-resize-" + t);
            worker.setDaemon(true);
            workers.add(worker);
            worker.start();
        }
        start.countDown();
        for (final Thread worker : workers) {
            worker.join(TimeUnit.MINUTES.toMillis(2));
            assertFalse(worker.isAlive());
        }
        if (failure.get() != null) {
            throw new AssertionError("a worker failed", failure.get());
        }
        cache.checkInvariantsForTesting();
    }

    // ---------------------------------------------------------------------------------------------------
    // Index cost
    // ---------------------------------------------------------------------------------------------------

    /** The figure I2 reports, asserted only as an upper bound so that it cannot drift unnoticed. */
    public void testIndexBytesPerEntry() {
        final SlabVectorCache cache = cacheOf(4);
        final Object core = new Object();
        final int slots = cache.slotsPerSlab(512) * 4;
        for (int ordinal = 0; ordinal < slots; ordinal++) {
            put(cache, core, 0, ordinal, 128);
        }
        final long perEntry = cache.indexBytes() / cache.entries();
        logger.info("the vector cache index costs {} bytes per entry at {} entries", perEntry, cache.entries());
        assertTrue("index cost per entry was " + perEntry + " B", perEntry <= 48);
    }

    // ---------------------------------------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------------------------------------

    private static void put(final SlabVectorCache cache, final Object core, final int field, final int ordinal, final int dimension) {
        final ByteBuffer staging = ByteBuffer.allocate(dimension * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        fill(staging, vector(core, field, ordinal, dimension));
        cache.put(core, field, ordinal, staging, 0, dimension * Float.BYTES);
    }

    private static void fill(final ByteBuffer staging, final float[] vector) {
        for (int i = 0; i < vector.length; i++) {
            staging.putFloat(i * Float.BYTES, vector[i]);
        }
    }

    /** A vector whose every float is derived from its key, so a wrong vector cannot look right. */
    private static float[] vector(final Object core, final int field, final int ordinal, final int dimension) {
        final int seed = System.identityHashCode(core) * 31 + field * 7919 + ordinal;
        final float[] vector = new float[dimension];
        for (int i = 0; i < dimension; i++) {
            vector[i] = seed + i * 0.5f;
        }
        return vector;
    }
}

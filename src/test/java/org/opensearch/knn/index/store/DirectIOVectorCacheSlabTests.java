/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import org.opensearch.knn.KNNTestCase;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

/**
 * M2's acceptance tests for the slab store inside {@link DirectIOVectorCache}, adapted from 16-I's
 * {@code SlabVectorCacheTests} for this phase's key: the file and the byte offset, with no segment core, no
 * field and no ordinal.
 *
 * <p>These exercise the store directly. {@code DirectIOVectorCacheTests} drives the same store through the
 * real {@link DirectIOVectorIndexInput}, which is where the two call sites and the fill rule are proved; what
 * is proved here is the policy: eviction, the second chance and its cap, resize, the closed-file rule, and
 * that none of them can ever serve one key's bytes for another's.
 *
 * <p>The slab size is a test parameter rather than the shipped 32 MiB, so that a handful of fills reaches
 * eviction instead of a gigabyte of vectors. Everything else is the shipped path: real off-heap slabs, the
 * real index, the real locking.
 */
public class DirectIOVectorCacheSlabTests extends KNNTestCase {

    /** 64 KiB: 128 slots for a 128-dim vector, 4 for a 4,096-dim one. */
    private static final int TEST_SLAB_BYTES = 64 * 1024;

    /** Where a Lucene {@code .vec}'s vector data starts, so offsets look like the ones the input passes. */
    private static final long HEADER = 44L;

    @Override
    public void tearDown() throws Exception {
        DirectIOVectorCache.resetForTesting();
        super.tearDown();
    }

    /** The store on at {@code slabs} test slabs, from empty. */
    private void cacheOf(final int slabs) {
        DirectIOVectorCache.resetForTesting();
        DirectIOVectorCache.setSlabBytesForTesting(TEST_SLAB_BYTES);
        DirectIOVectorCache.configure((long) slabs * TEST_SLAB_BYTES);
    }

    // ---------------------------------------------------------------------------------------------------
    // Round trip, at every vector size the plugin supports
    // ---------------------------------------------------------------------------------------------------

    public void testRoundTripAtEveryVectorSize() {
        cacheOf(4);
        final String file = "round.vec#1";
        for (final int dimension : new int[] { 128, 256, 768, 1024, 2048, 4096 }) {
            final int bytes = dimension * Float.BYTES;
            final int slots = DirectIOVectorCache.slotsPerSlab(bytes);
            assertTrue("a " + dimension + "-dim vector should fit a test slab", slots >= 1);
            final int count = Math.min(slots, 8);
            for (int i = 0; i < count; i++) {
                put(file, offsetOf(dimension, i), dimension);
            }
            final float[] read = new float[dimension];
            for (int i = 0; i < count; i++) {
                final long offset = offsetOf(dimension, i);
                assertTrue("dimension " + dimension + " vector " + i, DirectIOVectorCache.getFloats(file, offset, bytes, read, 0));
                assertArrayEquals(vector(file, offset, dimension), read, 0.0f);
            }
            DirectIOVectorCache.checkInvariantsForTesting();
        }
    }

    public void testMissesAreMissesAndNotWrongVectors() {
        cacheOf(1);
        final String file = "a.vec#1";
        final long offset = offsetOf(128, 5);
        put(file, offset, 128);
        final float[] read = new float[128];

        assertFalse("another file at the same offset", DirectIOVectorCache.getFloats("b.vec#1", offset, 512, read, 0));
        assertFalse("the same path at another length", DirectIOVectorCache.getFloats("a.vec#2", offset, 512, read, 0));
        assertFalse("another offset", DirectIOVectorCache.getFloats(file, offset + 512, 512, read, 0));
        assertFalse("one byte off the offset", DirectIOVectorCache.getFloats(file, offset + 4, 512, read, 0));
        assertFalse("another vector length", DirectIOVectorCache.getFloats(file, offset, 1024, read, 0));
        assertTrue(DirectIOVectorCache.getFloats(file, offset, 512, read, 0));
        assertArrayEquals(vector(file, offset, 128), read, 0.0f);
    }

    /** A destination offset, because the input hands one through from {@code readFloats}. */
    public void testAHitHonoursTheDestinationOffset() {
        cacheOf(1);
        final String file = "dest.vec#1";
        final long offset = offsetOf(128, 2);
        put(file, offset, 128);
        final float[] read = new float[128 + 7];
        read[0] = -1.0f;
        assertTrue(DirectIOVectorCache.getFloats(file, offset, 512, read, 7));
        assertEquals("nothing before the offset may be touched", -1.0f, read[0], 0.0f);
        final float[] expected = vector(file, offset, 128);
        for (int i = 0; i < 128; i++) {
            assertEquals("component " + i, expected[i], read[7 + i], 0.0f);
        }
    }

    public void testOffHoldsNothingAndCostsNothing() {
        cacheOf(0);
        assertFalse(DirectIOVectorCache.isOn());
        final String file = "off.vec#1";
        put(file, HEADER, 128);
        assertEquals(0L, DirectIOVectorCache.entries());
        assertEquals(0L, DirectIOVectorCache.slabs());
        assertFalse(DirectIOVectorCache.getFloats(file, HEADER, 512, new float[128], 0));
        assertFalse(DirectIOVectorCache.contains(file, HEADER, 512));
        // With the store off a lookup must not even count as a miss: it never asked.
        assertEquals(0L, DirectIOVectorCache.hits());
        assertEquals(0L, DirectIOVectorCache.misses());
    }

    /** {@code contains} is the prefetch-side lookup: the same answer as a hit, and no counter movement. */
    public void testContainsAnswersWithoutCounting() {
        cacheOf(1);
        final String file = "contains.vec#1";
        final long offset = offsetOf(128, 3);
        put(file, offset, 128);

        assertTrue(DirectIOVectorCache.contains(file, offset, 512));
        assertFalse(DirectIOVectorCache.contains(file, offset + 512, 512));
        assertFalse("a length mismatch is not a hit here either", DirectIOVectorCache.contains(file, offset, 1024));
        assertEquals("the prefetch-side lookup counts nothing", 0L, DirectIOVectorCache.hits());
        assertEquals(0L, DirectIOVectorCache.misses());

        assertTrue(DirectIOVectorCache.getFloats(file, offset, 512, new float[128], 0));
        assertEquals("only the lookup that serves counts", 1L, DirectIOVectorCache.hits());
        assertEquals(0L, DirectIOVectorCache.misses());
    }

    /**
     * The prefetch-side lookup is asked about a <b>coalesced group</b>, not a vector: it must answer yes only
     * when every vector of the run is cached, and it must walk the run by the slots' own lengths rather than
     * needing the caller to know the field's vector length. M4 MEASURED what the one-vector form costs
     * instead: 27.9x the file in device reads against 1.21x.
     */
    public void testContainsAnswersForAWholeRun() {
        cacheOf(2);
        final String file = "run-contains.vec#1";
        final int dimension = 128;
        final int bytes = dimension * Float.BYTES;
        final long first = offsetOf(dimension, 10);

        // Four consecutive vectors, with the third one left out.
        put(file, first, dimension);
        put(file, first + bytes, dimension);
        put(file, first + 3L * bytes, dimension);

        assertTrue("one vector", DirectIOVectorCache.contains(file, first, bytes));
        assertTrue("two consecutive vectors", DirectIOVectorCache.contains(file, first, 2 * bytes));
        assertFalse("the third is missing, so the run is not covered", DirectIOVectorCache.contains(file, first, 3 * bytes));
        assertFalse(DirectIOVectorCache.contains(file, first, 4 * bytes));

        put(file, first + 2L * bytes, dimension);
        assertTrue("and now it is", DirectIOVectorCache.contains(file, first, 4 * bytes));
        assertFalse("a run starting one byte off the grid is not covered", DirectIOVectorCache.contains(file, first + 4, 2 * bytes));
        assertFalse(
            "and neither is one that would need a vector past the last cached one",
            DirectIOVectorCache.contains(file, first, 5 * bytes)
        );
        assertFalse(
            "a run whose last vector overshoots the range asked about is not an answer",
            DirectIOVectorCache.contains(file, first, 2 * bytes + 4)
        );
        assertEquals("and none of this counted as a hit or a miss", 0L, DirectIOVectorCache.hits());
        assertEquals(0L, DirectIOVectorCache.misses());
    }

    /** A run of mixed vector lengths is still walked correctly, because the stride comes from each slot. */
    public void testContainsWalksARunOfTwoLengths() {
        cacheOf(2);
        final String file = "mixed-run.vec#1";
        final long first = 44L;
        // A 128-dim vector (512 B) followed by a 256-dim one (1,024 B), which is not a layout Lucene writes
        // but is the case that proves the stride is the slot's and not the caller's.
        put(file, first, 128);
        put(file, first + 512, 256);
        assertTrue(DirectIOVectorCache.contains(file, first, 512));
        assertTrue(DirectIOVectorCache.contains(file, first, 512 + 1024));
        assertFalse(DirectIOVectorCache.contains(file, first, 512 + 1024 + 4));
    }

    public void testRecachingAVectorReplacesItRatherThanAddingIt() {
        cacheOf(1);
        final String file = "recache.vec#1";
        final long offset = offsetOf(128, 3);
        put(file, offset, 128);
        put(file, offset, 128);
        put(file, offset, 128);
        assertEquals(1L, DirectIOVectorCache.entries());
        assertEquals(512L, DirectIOVectorCache.bytes());
        final float[] read = new float[128];
        assertTrue(DirectIOVectorCache.getFloats(file, offset, 512, read, 0));
        assertArrayEquals(vector(file, offset, 128), read, 0.0f);
        DirectIOVectorCache.checkInvariantsForTesting();
    }

    // ---------------------------------------------------------------------------------------------------
    // Eviction and the second chance
    // ---------------------------------------------------------------------------------------------------

    public void testFifoEvictionDropsTheColdOnes() {
        cacheOf(1);
        final String file = "fifo.vec#1";
        final int slots = DirectIOVectorCache.slotsPerSlab(512);
        for (int i = 0; i < slots; i++) {
            put(file, offsetOf(128, i), 128);
        }
        assertEquals(slots, DirectIOVectorCache.entries());
        assertEquals(0L, DirectIOVectorCache.reclaimedSlabs());

        // One more vector than the slab holds, and nothing has been read, so nothing earns a second chance.
        put(file, offsetOf(128, slots), 128);
        assertEquals(1L, DirectIOVectorCache.reclaimedSlabs());
        assertEquals(0L, DirectIOVectorCache.readmissions());
        assertEquals(1L, DirectIOVectorCache.entries());
        final float[] read = new float[128];
        assertTrue("the vector that caused the eviction", DirectIOVectorCache.getFloats(file, offsetOf(128, slots), 512, read, 0));
        for (int i = 0; i < slots; i++) {
            assertFalse(
                "vector " + i + " should have gone with its slab",
                DirectIOVectorCache.getFloats(file, offsetOf(128, i), 512, read, 0)
            );
        }
        DirectIOVectorCache.checkInvariantsForTesting();
    }

    public void testASecondChanceKeepsTheVectorsThatWereRead() {
        cacheOf(1);
        final String file = "chance.vec#1";
        final int slots = DirectIOVectorCache.slotsPerSlab(512);
        for (int i = 0; i < slots; i++) {
            put(file, offsetOf(128, i), 128);
        }
        final float[] read = new float[128];
        final int touched = 10;
        for (int i = 0; i < touched; i++) {
            assertTrue(DirectIOVectorCache.getFloats(file, offsetOf(128, i), 512, read, 0));
        }

        put(file, offsetOf(128, slots), 128);

        assertEquals(1L, DirectIOVectorCache.reclaimedSlabs());
        assertEquals(touched, DirectIOVectorCache.readmissions());
        assertEquals(touched + 1, DirectIOVectorCache.entries());
        for (int i = 0; i < touched; i++) {
            final long offset = offsetOf(128, i);
            assertTrue("vector " + i + " was read, so it should have survived", DirectIOVectorCache.getFloats(file, offset, 512, read, 0));
            assertArrayEquals(vector(file, offset, 128), read, 0.0f);
        }
        for (int i = touched; i < slots; i++) {
            assertFalse("vector " + i + " was never read", DirectIOVectorCache.getFloats(file, offsetOf(128, i), 512, read, 0));
        }
        DirectIOVectorCache.checkInvariantsForTesting();
    }

    /** The cap is what stops a wholly hot slab from re-admitting itself and leaving no room. */
    public void testTheSecondChanceIsCappedAtHalfASlab() {
        cacheOf(1);
        final String file = "cap.vec#1";
        final int slots = DirectIOVectorCache.slotsPerSlab(512);
        for (int i = 0; i < slots; i++) {
            put(file, offsetOf(128, i), 128);
        }
        final float[] read = new float[128];
        for (int i = 0; i < slots; i++) {
            assertTrue(DirectIOVectorCache.getFloats(file, offsetOf(128, i), 512, read, 0));
        }

        put(file, offsetOf(128, slots), 128);

        assertEquals(slots / 2, DirectIOVectorCache.readmissions());
        assertEquals(slots / 2 + 1, DirectIOVectorCache.entries());
        for (int i = 0; i < slots / 2; i++) {
            assertTrue("the oldest half survives first", DirectIOVectorCache.getFloats(file, offsetOf(128, i), 512, read, 0));
        }
        DirectIOVectorCache.checkInvariantsForTesting();
    }

    /**
     * A survivor keeps its slot, so nothing is copied and its index entry is untouched; the slots the reclaim
     * freed are then usable one at a time without a second reclaim.
     */
    public void testAReclaimKeepsSurvivorsWhereTheyAreAndFreesTheRest() {
        cacheOf(1);
        final String file = "reclaim.vec#1";
        final int slots = DirectIOVectorCache.slotsPerSlab(512);
        for (int i = 0; i < slots; i++) {
            put(file, offsetOf(128, i), 128);
        }
        final float[] read = new float[128];
        for (int i = 0; i < slots; i++) {
            assertTrue(DirectIOVectorCache.getFloats(file, offsetOf(128, i), 512, read, 0));
        }

        put(file, offsetOf(128, slots), 128);

        assertEquals(1L, DirectIOVectorCache.reclaimedSlabs());
        assertEquals("the cap is half the slots", slots / 2, DirectIOVectorCache.readmissions());
        int survivors = 0;
        for (int i = 0; i < slots; i++) {
            final long offset = offsetOf(128, i);
            if (DirectIOVectorCache.getFloats(file, offset, 512, read, 0)) {
                assertArrayEquals("a survivor's bytes must be its own", vector(file, offset, 128), read, 0.0f);
                survivors++;
            }
        }
        assertEquals(slots / 2, survivors);
        for (int i = slots + 1; i < slots + slots / 2; i++) {
            final long offset = offsetOf(128, i);
            put(file, offset, 128);
            assertTrue(DirectIOVectorCache.getFloats(file, offset, 512, read, 0));
        }
        assertEquals("no second reclaim was needed to place them", 1L, DirectIOVectorCache.reclaimedSlabs());
        DirectIOVectorCache.checkInvariantsForTesting();
        logger.info("one reclaim of a {}-slot slab held the write lock for {} ns", slots, DirectIOVectorCache.reclaimNanos());
    }

    public void testEvictionAcrossSeveralSlabsTakesTheOldestFirst() {
        cacheOf(3);
        final String file = "many.vec#1";
        final int slots = DirectIOVectorCache.slotsPerSlab(512);
        for (int i = 0; i < slots * 3; i++) {
            put(file, offsetOf(128, i), 128);
        }
        assertEquals(3L, DirectIOVectorCache.slabs());
        assertEquals(slots * 3, DirectIOVectorCache.entries());

        put(file, offsetOf(128, slots * 3), 128);

        assertEquals(1L, DirectIOVectorCache.reclaimedSlabs());
        assertEquals(3L, DirectIOVectorCache.slabs());
        final float[] read = new float[128];
        // The first slab's worth is gone; the second and third are not.
        assertFalse(DirectIOVectorCache.getFloats(file, offsetOf(128, 0), 512, read, 0));
        assertTrue(DirectIOVectorCache.getFloats(file, offsetOf(128, slots), 512, read, 0));
        assertTrue(DirectIOVectorCache.getFloats(file, offsetOf(128, slots * 2), 512, read, 0));
        assertTrue(DirectIOVectorCache.getFloats(file, offsetOf(128, slots * 3), 512, read, 0));
        DirectIOVectorCache.checkInvariantsForTesting();
    }

    /** Two vector lengths in one store: a slab reopened for the other length drops what it held. */
    public void testTwoVectorLengthsShareTheSlabBudget() {
        cacheOf(2);
        final String small = "small.vec#1";
        final String large = "large.vec#1";
        final int smallSlots = DirectIOVectorCache.slotsPerSlab(512);
        final int largeSlots = DirectIOVectorCache.slotsPerSlab(4096);
        for (int i = 0; i < smallSlots; i++) {
            put(small, offsetOf(128, i), 128);
        }
        for (int i = 0; i < largeSlots; i++) {
            put(large, offsetOf(1024, i), 1024);
        }
        assertEquals(2L, DirectIOVectorCache.slabs());
        assertEquals(smallSlots + largeSlots, DirectIOVectorCache.entries());
        assertEquals(smallSlots * 512L + largeSlots * 4096L, DirectIOVectorCache.bytes());

        // A third slab's worth of small vectors has to come out of the two slabs in service.
        for (int i = smallSlots; i < smallSlots + 4; i++) {
            put(small, offsetOf(128, i), 128);
        }
        assertEquals(2L, DirectIOVectorCache.slabs());
        DirectIOVectorCache.checkInvariantsForTesting();
        final float[] read = new float[128];
        assertTrue(DirectIOVectorCache.getFloats(small, offsetOf(128, smallSlots + 3), 512, read, 0));
    }

    // ---------------------------------------------------------------------------------------------------
    // The closed-file rule
    // ---------------------------------------------------------------------------------------------------

    /**
     * A merged-away file's entries are never wrong, only cold: nothing reads them, so they earn no second
     * chance and the FIFO frees them. The half this phase adds is that the file's index table goes with its
     * last entry, so a dead file costs bounded eviction pressure and then nothing at all.
     */
    public void testADeadFilesEntriesAgeOutAndItsIndexCellGoesWithThem() {
        cacheOf(1);
        final String dead = "merged-away.vec#1";
        final String live = "current.vec#1";
        final int slots = DirectIOVectorCache.slotsPerSlab(512);
        for (int i = 0; i < slots; i++) {
            put(dead, offsetOf(128, i), 128);
        }
        assertTrue(DirectIOVectorCache.hasIndexFor(dead));
        assertEquals(1, DirectIOVectorCache.indexedFiles());

        // The new file's first fill reclaims the dead file's slab. Nothing of the dead file was ever read,
        // so nothing survives and its table empties.
        put(live, offsetOf(128, 0), 128);

        assertEquals(1L, DirectIOVectorCache.reclaimedSlabs());
        assertEquals(0L, DirectIOVectorCache.readmissions());
        assertFalse("the dead file's index table must be gone", DirectIOVectorCache.hasIndexFor(dead));
        assertTrue(DirectIOVectorCache.hasIndexFor(live));
        assertEquals("one file indexed, not two", 1, DirectIOVectorCache.indexedFiles());
        assertEquals(1L, DirectIOVectorCache.entries());
        final float[] read = new float[128];
        for (int i = 0; i < slots; i++) {
            assertFalse(DirectIOVectorCache.getFloats(dead, offsetOf(128, i), 512, read, 0));
        }
        DirectIOVectorCache.checkInvariantsForTesting();
    }

    /** A file that still has one live vector keeps its table, which is the other side of the same rule. */
    public void testAFileWithOneSurvivorKeepsItsIndexCell() {
        cacheOf(1);
        final String file = "survivor.vec#1";
        final String other = "other.vec#1";
        final int slots = DirectIOVectorCache.slotsPerSlab(512);
        for (int i = 0; i < slots; i++) {
            put(file, offsetOf(128, i), 128);
        }
        final float[] read = new float[128];
        assertTrue("read one, so exactly one earns a second chance", DirectIOVectorCache.getFloats(file, offsetOf(128, 4), 512, read, 0));

        put(other, offsetOf(128, 0), 128);

        assertEquals(1L, DirectIOVectorCache.readmissions());
        assertTrue("the file still holds a vector, so it keeps its table", DirectIOVectorCache.hasIndexFor(file));
        assertEquals(2, DirectIOVectorCache.indexedFiles());
        assertTrue(DirectIOVectorCache.getFloats(file, offsetOf(128, 4), 512, read, 0));
        assertArrayEquals(vector(file, offsetOf(128, 4), 128), read, 0.0f);
        DirectIOVectorCache.checkInvariantsForTesting();
    }

    /** Releasing the slabs releases the tables too: {@code 0b} must leave nothing behind. */
    public void testTurningTheStoreOffDropsEveryIndexCell() {
        cacheOf(2);
        for (int file = 0; file < 5; file++) {
            for (int i = 0; i < 4; i++) {
                put("f" + file + ".vec#1", offsetOf(128, i), 128);
            }
        }
        assertEquals(5, DirectIOVectorCache.indexedFiles());

        DirectIOVectorCache.configure(0);

        assertEquals(0, DirectIOVectorCache.indexedFiles());
        assertEquals(0L, DirectIOVectorCache.entries());
        assertEquals(0L, DirectIOVectorCache.slabs());
        assertEquals(0L, DirectIOVectorCache.slabBytes());
        assertFalse(DirectIOVectorCache.isOn());
    }

    // ---------------------------------------------------------------------------------------------------
    // Resize
    // ---------------------------------------------------------------------------------------------------

    public void testResizingUpKeepsTheVectorsAndResizingDownReturnsTheSlabs() {
        cacheOf(1);
        final String file = "resize.vec#1";
        final int slots = DirectIOVectorCache.slotsPerSlab(512);
        for (int i = 0; i < slots; i++) {
            put(file, offsetOf(128, i), 128);
        }
        assertEquals(1L, DirectIOVectorCache.slabs());
        final float[] read = new float[128];

        DirectIOVectorCache.configure(4L * TEST_SLAB_BYTES);
        assertEquals(4L * TEST_SLAB_BYTES, DirectIOVectorCache.maxBytes());
        assertEquals("growing must not drop a vector", slots, DirectIOVectorCache.entries());
        assertTrue(DirectIOVectorCache.getFloats(file, offsetOf(128, 0), 512, read, 0));
        for (int i = slots; i < slots * 4; i++) {
            put(file, offsetOf(128, i), 128);
        }
        assertEquals(4L, DirectIOVectorCache.slabs());
        assertEquals(0L, DirectIOVectorCache.reclaimedSlabs());

        DirectIOVectorCache.configure(2L * TEST_SLAB_BYTES);
        assertEquals(2L, DirectIOVectorCache.slabs());
        assertEquals(2L * TEST_SLAB_BYTES, DirectIOVectorCache.slabBytes());
        assertEquals("the two oldest slabs' vectors go", slots * 2, DirectIOVectorCache.entries());
        assertTrue("and the newest are the ones kept", DirectIOVectorCache.getFloats(file, offsetOf(128, slots * 4 - 1), 512, read, 0));
        DirectIOVectorCache.checkInvariantsForTesting();

        DirectIOVectorCache.configure(0);
        assertEquals(0L, DirectIOVectorCache.slabs());
        assertEquals(0L, DirectIOVectorCache.entries());
        assertEquals(0L, DirectIOVectorCache.bytes());
        assertFalse(DirectIOVectorCache.isOn());
    }

    public void testATargetUnderOneSlabHoldsNothing() {
        cacheOf(0);
        DirectIOVectorCache.configure(TEST_SLAB_BYTES - 1);
        assertEquals(0L, DirectIOVectorCache.maxBytes());
        assertFalse(DirectIOVectorCache.isOn());
    }

    /** Every slab the store ever took is released, which is what the backend's counters are there to show. */
    public void testEverySlabIsReleased() {
        DirectIOSlab.resetCountersForTesting();
        cacheOf(3);
        final String file = "leak.vec#1";
        for (int i = 0; i < DirectIOVectorCache.slotsPerSlab(512) * 3; i++) {
            put(file, offsetOf(128, i), 128);
        }
        assertEquals(3L, DirectIOSlab.allocatedCount());
        assertEquals(0L, DirectIOSlab.freedCount());

        DirectIOVectorCache.configure(0);
        // The closes run on a release thread of their own since N3, so wait for them.
        DirectIOVectorCache.awaitReleasesForTesting();

        assertEquals("every slab back", DirectIOSlab.allocatedCount(), DirectIOSlab.freedCount());
    }

    // ---------------------------------------------------------------------------------------------------
    // Concurrency
    // ---------------------------------------------------------------------------------------------------

    /**
     * Eight threads looking up, inserting and evicting at once against a store far smaller than their key
     * space, so eviction and compaction run throughout. Every vector's content is derived from its key, so a
     * hit that returns the wrong bytes is caught at the point of the hit rather than by a checksum at the
     * end, and a torn read of a slot being refilled would show up as a mismatch.
     */
    public void testConcurrentLookupInsertAndEvict() throws Exception {
        cacheOf(4);
        final int dimension = 128;
        final int bytes = dimension * Float.BYTES;
        final String[] files = new String[] { "c0.vec#1", "c1.vec#1", "c2.vec#1" };
        final int vectors = 20_000;
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
                        final String file = files[random.nextInt(files.length)];
                        final long offset = offsetOf(dimension, random.nextInt(vectors));
                        if (DirectIOVectorCache.getFloats(file, offset, bytes, read, 0)) {
                            final float[] expected = vector(file, offset, dimension);
                            for (int d = 0; d < dimension; d++) {
                                if (read[d] != expected[d]) {
                                    throw new AssertionError(
                                        "a hit for offset " + offset + " of " + file + " returned the wrong bytes at " + d + ": " + read[d]
                                    );
                                }
                            }
                        } else {
                            fill(staging, vector(file, offset, dimension));
                            DirectIOVectorCache.put(file, offset, staging, 0, bytes);
                        }
                    }
                } catch (Throwable e) {
                    failure.compareAndSet(null, e);
                }
            }, "slab-store-" + t);
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

        DirectIOVectorCache.checkInvariantsForTesting();
        assertEquals(4L, DirectIOVectorCache.slabs());
        assertTrue("the run should have evicted", DirectIOVectorCache.reclaimedSlabs() > 0);
        assertTrue("the run should have hit", DirectIOVectorCache.hits() > 0);
        assertEquals((long) threads * operationsPerThread, DirectIOVectorCache.hits() + DirectIOVectorCache.misses());
        assertTrue("entries cannot exceed the slabs' slots", DirectIOVectorCache.entries() <= 4L * DirectIOVectorCache.slotsPerSlab(bytes));
    }

    /** Resizing under load, which is what an operator's live change to {@code max_bytes} does. */
    public void testConcurrentResizeWhileReading() throws Exception {
        cacheOf(2);
        final int dimension = 128;
        final int bytes = dimension * Float.BYTES;
        final String file = "resize-under-load.vec#1";
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
                            DirectIOVectorCache.configure((long) (1 + random.nextInt(4)) * TEST_SLAB_BYTES);
                            continue;
                        }
                        final long offset = offsetOf(dimension, random.nextInt(5_000));
                        if (DirectIOVectorCache.getFloats(file, offset, bytes, read, 0)) {
                            assertArrayEquals(vector(file, offset, dimension), read, 0.0f);
                        } else {
                            fill(staging, vector(file, offset, dimension));
                            DirectIOVectorCache.put(file, offset, staging, 0, bytes);
                        }
                    }
                } catch (Throwable e) {
                    failure.compareAndSet(null, e);
                }
            }, "slab-store-resize-" + t);
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
        DirectIOVectorCache.checkInvariantsForTesting();
    }

    /** Many files at once, so the index map itself is mutated while readers walk it. */
    public void testConcurrentFillAcrossManyFiles() throws Exception {
        cacheOf(2);
        final int dimension = 128;
        final int bytes = dimension * Float.BYTES;
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
                        final String file = "many-" + random.nextInt(200) + ".vec#1";
                        final long offset = offsetOf(dimension, random.nextInt(50));
                        if (DirectIOVectorCache.getFloats(file, offset, bytes, read, 0)) {
                            assertArrayEquals(vector(file, offset, dimension), read, 0.0f);
                        } else {
                            fill(staging, vector(file, offset, dimension));
                            DirectIOVectorCache.put(file, offset, staging, 0, bytes);
                        }
                    }
                } catch (Throwable e) {
                    failure.compareAndSet(null, e);
                }
            }, "slab-store-files-" + t);
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
        DirectIOVectorCache.checkInvariantsForTesting();
        assertTrue("files with nothing left must not accumulate", DirectIOVectorCache.indexedFiles() <= 200);
    }

    // ---------------------------------------------------------------------------------------------------
    // Index cost
    // ---------------------------------------------------------------------------------------------------

    /** The figure M2 reports, asserted as an upper bound so that it cannot drift unnoticed. */
    public void testIndexBytesPerEntry() {
        cacheOf(4);
        final String file = "cost.vec#1";
        final int slots = DirectIOVectorCache.slotsPerSlab(512) * 4;
        for (int i = 0; i < slots; i++) {
            put(file, offsetOf(128, i), 128);
        }
        final long perEntry = DirectIOVectorCache.indexBytes() / DirectIOVectorCache.entries();
        logger.info("the vector cache index costs {} bytes per entry at {} entries", perEntry, DirectIOVectorCache.entries());
        assertTrue("index cost per entry was " + perEntry + " B", perEntry <= 64);
    }

    // ---------------------------------------------------------------------------------------------------
    // The span-fill admission API (M4 drives it from the input)
    // ---------------------------------------------------------------------------------------------------

    /** A run of consecutive vectors goes in under one call, and every one of them is readable afterwards. */
    public void testPutRunAdmitsEveryVectorOfTheRun() {
        cacheOf(2);
        final String file = "run.vec#1";
        final int dimension = 128;
        final int bytes = dimension * Float.BYTES;
        final long first = offsetOf(dimension, 10);
        final ByteBuffer span = runOf(file, first, 6, dimension);

        assertEquals(6, DirectIOVectorCache.putRun(file, first, 6, span, 0, bytes));
        assertEquals("the counter must name them as span fills", 6L, DirectIOVectorCache.spanFills());

        final float[] read = new float[dimension];
        for (int i = 0; i < 6; i++) {
            final long offset = first + (long) i * bytes;
            assertTrue("vector at " + offset, DirectIOVectorCache.getFloats(file, offset, bytes, read, 0));
            assertArrayEquals(vector(file, offset, dimension), read, 0.0f);
        }
        assertFalse("and nothing outside the run", DirectIOVectorCache.getFloats(file, first + 6L * bytes, bytes, read, 0));
        assertFalse(DirectIOVectorCache.getFloats(file, first - bytes, bytes, read, 0));
        DirectIOVectorCache.checkInvariantsForTesting();
    }

    /** A vector the store already holds is skipped rather than given a second slot. */
    public void testPutRunSkipsWhatIsAlreadyCached() {
        cacheOf(2);
        final String file = "skip.vec#1";
        final int dimension = 128;
        final int bytes = dimension * Float.BYTES;
        final long first = offsetOf(dimension, 10);
        put(file, first + bytes, dimension);
        final long entriesBefore = DirectIOVectorCache.entries();

        assertEquals(
            "one of the four was already in",
            3,
            DirectIOVectorCache.putRun(file, first, 4, runOf(file, first, 4, dimension), 0, bytes)
        );
        assertEquals(entriesBefore + 3, DirectIOVectorCache.entries());

        final float[] read = new float[dimension];
        for (int i = 0; i < 4; i++) {
            final long offset = first + (long) i * bytes;
            assertTrue(DirectIOVectorCache.getFloats(file, offset, bytes, read, 0));
            assertArrayEquals(vector(file, offset, dimension), read, 0.0f);
        }
        DirectIOVectorCache.checkInvariantsForTesting();
    }

    /**
     * The rule span fill is bounded by: at or above the slab target it admits nothing and evicts nothing. A
     * requested fill still evicts, which is unchanged; this is the admission that must not.
     */
    public void testPutRunAdmitsNothingAtTheTargetAndEvictsNothing() {
        cacheOf(1);
        final String file = "at-target.vec#1";
        final int dimension = 128;
        final int bytes = dimension * Float.BYTES;
        final int slots = DirectIOVectorCache.slotsPerSlab(bytes);

        for (int i = 0; i < slots; i++) {
            put(file, offsetOf(dimension, i), dimension);
        }
        assertFalse("the store is at its target now", DirectIOVectorCache.belowSlabTarget());
        final long entriesBefore = DirectIOVectorCache.entries();
        final long evictedBefore = DirectIOVectorCache.evictions();
        final long reclaimedBefore = DirectIOVectorCache.reclaimedSlabs();

        final long first = offsetOf(dimension, slots);
        assertEquals(0, DirectIOVectorCache.putRun(file, first, 8, runOf(file, first, 8, dimension), 0, bytes));
        assertEquals("nothing may be admitted", entriesBefore, DirectIOVectorCache.entries());
        assertEquals("and nothing evicted", evictedBefore, DirectIOVectorCache.evictions());
        assertEquals("and no slab reclaimed", reclaimedBefore, DirectIOVectorCache.reclaimedSlabs());
        assertEquals(0L, DirectIOVectorCache.spanFills());
        assertEquals(1L, DirectIOVectorCache.slabs());

        final float[] read = new float[dimension];
        assertFalse(DirectIOVectorCache.getFloats(file, first, bytes, read, 0));
        assertTrue(
            "and the vectors that were there are still there",
            DirectIOVectorCache.getFloats(file, offsetOf(dimension, 0), bytes, read, 0)
        );
        DirectIOVectorCache.checkInvariantsForTesting();
    }

    /**
     * The target is the bound, not the free slots of the open slab: a run stops where the next vector would
     * need a slab the target does not allow, and the slab it was filling keeps what it took.
     */
    public void testPutRunStopsWhenTheTargetIsReachedMidRun() {
        cacheOf(1);
        final String file = "mid-run.vec#1";
        final int dimension = 1024;
        final int bytes = dimension * Float.BYTES;
        final int slots = DirectIOVectorCache.slotsPerSlab(bytes);
        assertTrue("this test needs several slots in a slab, it has " + slots, slots >= 4);
        final long first = offsetOf(dimension, 0);

        final int offered = slots + 4;
        final int admitted = DirectIOVectorCache.putRun(file, first, offered, runOf(file, first, offered, dimension), 0, bytes);
        assertEquals("exactly the slab's worth", slots, admitted);
        assertEquals(1L, DirectIOVectorCache.slabs());
        assertEquals(0L, DirectIOVectorCache.reclaimedSlabs());

        final float[] read = new float[dimension];
        assertTrue(DirectIOVectorCache.getFloats(file, first, bytes, read, 0));
        assertArrayEquals(vector(file, first, dimension), read, 0.0f);
        assertTrue(DirectIOVectorCache.getFloats(file, first + (long) (slots - 1) * bytes, bytes, read, 0));
        assertFalse(
            "and the rest of the run was refused rather than evicting anything",
            DirectIOVectorCache.getFloats(file, first + (long) slots * bytes, bytes, read, 0)
        );
        DirectIOVectorCache.checkInvariantsForTesting();
    }

    /** With the store off there is no admission and no counter movement. */
    public void testPutRunHoldsNothingWhenOff() {
        cacheOf(0);
        final String file = "off-run.vec#1";
        assertEquals(0, DirectIOVectorCache.putRun(file, HEADER, 8, runOf(file, HEADER, 8, 128), 0, 512));
        assertEquals(0L, DirectIOVectorCache.entries());
        assertEquals(0L, DirectIOVectorCache.spanFills());
        assertFalse(DirectIOVectorCache.belowSlabTarget());
    }

    /** Span-filled entries follow the closed-file rule like any other, because nothing distinguishes them. */
    public void testSpanFilledEntriesFollowTheClosedFileRule() {
        cacheOf(1);
        final String dead = "span-dead.vec#1";
        final String live = "span-live.vec#1";
        final int dimension = 128;
        final int bytes = dimension * Float.BYTES;
        final int slots = DirectIOVectorCache.slotsPerSlab(bytes);
        final long first = offsetOf(dimension, 0);
        assertEquals(slots, DirectIOVectorCache.putRun(dead, first, slots, runOf(dead, first, slots, dimension), 0, bytes));
        assertTrue(DirectIOVectorCache.hasIndexFor(dead));

        // At the target now, so the next requested fill reclaims the slab the span filled.
        put(live, first, dimension);

        assertFalse("the dead file's table goes with its last span-filled vector", DirectIOVectorCache.hasIndexFor(dead));
        assertEquals(1, DirectIOVectorCache.indexedFiles());
        DirectIOVectorCache.checkInvariantsForTesting();
    }

    // ---------------------------------------------------------------------------------------------------
    // A fill that cannot be honoured
    // ---------------------------------------------------------------------------------------------------

    /** Nonsense a caller could pass has to be refused rather than cached as something else. */
    public void testAFillThatCannotBeHonouredIsRefused() {
        cacheOf(1);
        final String file = "refuse.vec#1";
        final ByteBuffer staging = ByteBuffer.allocate(1024).order(ByteOrder.LITTLE_ENDIAN);

        DirectIOVectorCache.put(file, HEADER, staging, 0, 0);
        DirectIOVectorCache.put(file, HEADER, staging, 0, -4);
        DirectIOVectorCache.put(file, HEADER, staging, 0, 513);
        DirectIOVectorCache.put(file, -1L, staging, 0, 512);
        DirectIOVectorCache.put(file, HEADER, staging, 0, TEST_SLAB_BYTES + 4);

        assertEquals("none of those may be held", 0L, DirectIOVectorCache.entries());
        assertEquals(0, DirectIOVectorCache.indexedFiles());
        DirectIOVectorCache.checkInvariantsForTesting();
    }

    /** The slab size cannot be changed under a live store, because the slots would stop meaning anything. */
    public void testTheSlabSizeCannotChangeWhileTheStoreHoldsSlabs() {
        cacheOf(1);
        expectThrows(IllegalStateException.class, () -> DirectIOVectorCache.setSlabBytesForTesting(4096));
    }

    // ---------------------------------------------------------------------------------------------------
    // N2: a failed slab allocation is not retried
    // ---------------------------------------------------------------------------------------------------

    /**
     * Many fills below the target make exactly one allocation attempt, and count one failure. MEASURED by
     * the orchestrator: a refused reservation costs the caller 515 to 525 ms and one full GC, under this
     * store's write lock, and every retry costs the same for the same answer.
     */
    public void testAFailedAllocationIsAttemptedOnceAndCounted() {
        final AtomicInteger attempts = new AtomicInteger();
        cacheOf(0);
        DirectIOVectorCache.setSlabBytesForTesting(TEST_SLAB_BYTES);
        DirectIOVectorCache.setSlabMemoryForTesting(refusing(attempts));
        DirectIOVectorCache.configure(4L * TEST_SLAB_BYTES);

        final String file = "no-slab.vec#1";
        for (int i = 0; i < 200; i++) {
            put(file, offsetOf(128, i), 128);
        }
        assertEquals("exactly one attempt, however many fills asked", 1, attempts.get());
        assertEquals(1L, DirectIOVectorCache.slabAllocFailures());
        assertEquals("and nothing was cached", 0L, DirectIOVectorCache.entries());
        assertEquals(0L, DirectIOVectorCache.slabs());
        // The file does get an index table and no slot in it, which is this store's documented behaviour for
        // a fill that cannot be placed: one empty table until the next fill for that file uses it.
        assertTrue(DirectIOVectorCache.hasIndexFor(file));
        DirectIOVectorCache.checkInvariantsForTesting();
    }

    /** A new target is a new question, so it re-arms growth: one more attempt, one more failure. */
    public void testATargetChangeReArmsGrowth() {
        final AtomicInteger attempts = new AtomicInteger();
        cacheOf(0);
        DirectIOVectorCache.setSlabBytesForTesting(TEST_SLAB_BYTES);
        DirectIOVectorCache.setSlabMemoryForTesting(refusing(attempts));
        final String file = "rearm.vec#1";

        DirectIOVectorCache.configure(2L * TEST_SLAB_BYTES);
        for (int i = 0; i < 20; i++) {
            put(file, offsetOf(128, i), 128);
        }
        assertEquals(1, attempts.get());

        DirectIOVectorCache.configure(4L * TEST_SLAB_BYTES);
        for (int i = 0; i < 20; i++) {
            put(file, offsetOf(128, i), 128);
        }
        assertEquals(2, attempts.get());
        assertEquals(2L, DirectIOVectorCache.slabAllocFailures());
    }

    /**
     * The store keeps serving and keeps filling the slab it already holds: a fill that cannot grow reclaims
     * instead, which is what {@code obtainSlab} did before N2 as well. The only thing that changed is that
     * it no longer pays for an allocation attempt first.
     */
    public void testTheStoreKeepsServingTheSlabsItHoldsAfterAFailure() {
        final AtomicInteger attempts = new AtomicInteger();
        cacheOf(1);
        final String file = "keep.vec#1";
        final int dimension = 128;
        final int slots = DirectIOVectorCache.slotsPerSlab(dimension * Float.BYTES);
        put(file, offsetOf(dimension, 0), dimension);
        assertEquals(1L, DirectIOVectorCache.slabs());

        // The target goes up but the backend will not give the store a second slab.
        DirectIOVectorCache.setSlabMemoryForTesting(refusing(attempts));
        DirectIOVectorCache.configure(4L * TEST_SLAB_BYTES);
        assertTrue(DirectIOVectorCache.belowSlabTarget());

        final float[] read = new float[dimension];
        assertTrue(
            "the vector it already held is still served",
            DirectIOVectorCache.getFloats(file, offsetOf(dimension, 0), dimension * Float.BYTES, read, 0)
        );
        assertArrayEquals(vector(file, offsetOf(dimension, 0), dimension), read, 0.0f);

        // Past the one slab's slot count, so every further fill has to reclaim it.
        for (int i = 1; i < 2 * slots; i++) {
            put(file, offsetOf(dimension, i), dimension);
        }
        assertEquals("one attempt in all of that", 1, attempts.get());
        assertEquals(1L, DirectIOVectorCache.slabAllocFailures());
        assertEquals("and the slab it holds is still in service", 1L, DirectIOVectorCache.slabs());
        assertTrue(DirectIOVectorCache.reclaimedSlabs() > 0);
        final long last = offsetOf(dimension, 2 * slots - 1);
        assertTrue(DirectIOVectorCache.getFloats(file, last, dimension * Float.BYTES, read, 0));
        assertArrayEquals(vector(file, last, dimension), read, 0.0f);
        DirectIOVectorCache.checkInvariantsForTesting();
    }

    /**
     * Releasing the store's slabs gives the memory back, so an earlier failure stops standing. This is the
     * operator sequence: a cap the JVM would not honour, then {@code 0b}, then a cap again. The re-arm sits in
     * the release itself as well as in the target change, so that it stays true of any other release path.
     */
    public void testAReleaseReArmsGrowth() {
        final AtomicInteger attempts = new AtomicInteger();
        cacheOf(2);
        final String file = "release.vec#1";
        put(file, offsetOf(128, 0), 128);
        assertEquals(1L, DirectIOVectorCache.slabs());

        // The target goes up and the backend refuses, so growth latches. The 4,096-dim fill needs a slab of
        // its own slot layout, which is what asks for one.
        DirectIOVectorCache.setSlabMemoryForTesting(refusing(attempts));
        DirectIOVectorCache.configure(4L * TEST_SLAB_BYTES);
        put(file, offsetOf(4096, 100), 4096);
        put(file, offsetOf(4096, 101), 4096);
        assertEquals("one attempt, and the second fill used the slab the first reclaimed", 1, attempts.get());

        // Every slab released, and then a cap again.
        DirectIOVectorCache.configure(0);
        assertEquals(0L, DirectIOVectorCache.slabs());
        DirectIOVectorCache.configure(4L * TEST_SLAB_BYTES);
        put(file, offsetOf(128, 0), 128);
        assertEquals("the release re-armed it", 2, attempts.get());
        assertEquals(2L, DirectIOVectorCache.slabAllocFailures());
        DirectIOVectorCache.checkInvariantsForTesting();
    }

    // ---------------------------------------------------------------------------------------------------
    // N3: no slab allocate or close while holding the store's lock
    // ---------------------------------------------------------------------------------------------------

    /**
     * Neither an allocation nor a release ever runs while the calling thread holds this store's lock, in
     * either mode. MEASURED (M1, output/bench/n2): an allocation is 7 to 10 ms and a release about 2.2 ms per
     * 32 MiB slab, so before N3 a fill held the write lock for the first and a {@code configure(0)} at 4 GB
     * held it for about 0.26 s of the second, with every lookup waiting behind them.
     */
    public void testNeitherAllocateNorReleaseRunsUnderTheStoresLock() {
        final List<String> violations = Collections.synchronizedList(new ArrayList<>());
        final AtomicInteger allocations = new AtomicInteger();
        final AtomicInteger releases = new AtomicInteger();
        cacheOf(0);
        DirectIOVectorCache.setSlabBytesForTesting(TEST_SLAB_BYTES);
        DirectIOVectorCache.setSlabMemoryForTesting(new DirectIOVectorCache.SlabMemory() {
            @Override
            public DirectIOSlab allocate(final int bytes) {
                allocations.incrementAndGet();
                if (DirectIOVectorCache.lockHeldByCurrentThreadForTesting()) {
                    violations.add("allocate held the store's lock");
                }
                return DirectIOSlab.allocate(bytes);
            }

            @Override
            public void release(final DirectIOSlab slab) {
                releases.incrementAndGet();
                if (DirectIOVectorCache.lockHeldByCurrentThreadForTesting()) {
                    violations.add("release held the store's lock");
                }
                slab.close();
            }
        });
        DirectIOVectorCache.configure(4L * TEST_SLAB_BYTES);

        final String file = "lockfree.vec#1";
        final int dimension = 128;
        final int slots = DirectIOVectorCache.slotsPerSlab(dimension * Float.BYTES);
        // Four slabs' worth of requested fills, so growth runs four times.
        for (int i = 0; i < 4 * slots; i++) {
            put(file, offsetOf(dimension, i), dimension);
        }
        assertEquals(4L, DirectIOVectorCache.slabs());
        // And a span fill, which grows through the other half of place().
        DirectIOVectorCache.configure(0);
        DirectIOVectorCache.awaitReleasesForTesting();
        DirectIOVectorCache.configure(4L * TEST_SLAB_BYTES);
        for (int run = 0; run < 4; run++) {
            final long at = offsetOf(dimension, run * slots);
            DirectIOVectorCache.putRun(file, at, slots, runOf(file, at, slots, dimension), 0, dimension * Float.BYTES);
        }
        assertTrue("the span fills took slabs too", DirectIOVectorCache.slabs() > 1);
        assertTrue(DirectIOVectorCache.spanFills() > 0);

        // Down, which releases some, and off, which releases the rest.
        DirectIOVectorCache.configure(TEST_SLAB_BYTES);
        DirectIOVectorCache.configure(0);
        DirectIOVectorCache.awaitReleasesForTesting();

        assertEquals("every allocate and release must run outside the lock", List.of(), violations);
        assertTrue("the test has to have exercised allocation, saw " + allocations.get(), allocations.get() >= 4);
        assertTrue("and release, saw " + releases.get(), releases.get() >= 4);
    }

    /** Eight threads filling from empty never take more slabs than the target allows. */
    public void testConcurrentFillsFromEmptyNeverExceedTheTarget() throws Exception {
        final int target = 3;
        final AtomicInteger allocations = new AtomicInteger();
        final AtomicInteger overshoot = new AtomicInteger();
        cacheOf(0);
        DirectIOVectorCache.setSlabBytesForTesting(TEST_SLAB_BYTES);
        DirectIOVectorCache.setSlabMemoryForTesting(new DirectIOVectorCache.SlabMemory() {
            @Override
            public DirectIOSlab allocate(final int bytes) {
                allocations.incrementAndGet();
                // Slow on purpose: this is the window in which two fills could both decide to grow.
                try {
                    Thread.sleep(5);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
                return DirectIOSlab.allocate(bytes);
            }

            @Override
            public void release(final DirectIOSlab slab) {
                slab.close();
            }
        });
        DirectIOVectorCache.configure((long) target * TEST_SLAB_BYTES);

        final int threads = 8;
        final CountDownLatch start = new CountDownLatch(1);
        final CountDownLatch done = new CountDownLatch(threads);
        final AtomicReference<Throwable> failure = new AtomicReference<>();
        for (int t = 0; t < threads; t++) {
            final int seed = t;
            final Thread thread = new Thread(() -> {
                try {
                    final Random random = new Random(400L + seed);
                    start.await();
                    for (int i = 0; i < 400; i++) {
                        put("race.vec#1", offsetOf(128, random.nextInt(4096)), 128);
                        if (DirectIOVectorCache.slabs() > target) {
                            overshoot.incrementAndGet();
                        }
                    }
                } catch (Throwable e) {
                    failure.compareAndSet(null, e);
                } finally {
                    done.countDown();
                }
            }, "n3-fill-" + t);
            thread.setDaemon(true);
            thread.start();
        }
        start.countDown();
        assertTrue(done.await(2, TimeUnit.MINUTES));
        if (failure.get() != null) {
            throw new AssertionError(failure.get());
        }
        assertEquals("the target was never exceeded", 0, overshoot.get());
        assertEquals((long) target, DirectIOVectorCache.slabs());
        assertTrue("at most one allocation per slab the target allows, saw " + allocations.get(), allocations.get() <= target);
        DirectIOVectorCache.checkInvariantsForTesting();
    }

    // ---------------------------------------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------------------------------------

    /** A backend that counts what it is asked for and refuses every slab. */
    private static DirectIOVectorCache.SlabMemory refusing(final AtomicInteger attempts) {
        return new DirectIOVectorCache.SlabMemory() {
            @Override
            public DirectIOSlab allocate(final int bytes) {
                attempts.incrementAndGet();
                return null;
            }

            @Override
            public void release(final DirectIOSlab slab) {
                slab.close();
            }
        };
    }

    /** The byte offset of vector {@code i} of a {@code dimension}-dim field, past a Lucene header. */
    private static long offsetOf(final int dimension, final int i) {
        return HEADER + (long) i * dimension * Float.BYTES;
    }

    private static void put(final String fileId, final long offset, final int dimension) {
        final ByteBuffer staging = ByteBuffer.allocate(dimension * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        fill(staging, vector(fileId, offset, dimension));
        DirectIOVectorCache.put(fileId, offset, staging, 0, dimension * Float.BYTES);
    }

    /** The bytes of {@code count} consecutive vectors, laid out as a staged span holds them. */
    private static ByteBuffer runOf(final String fileId, final long firstOffset, final int count, final int dimension) {
        final int bytes = dimension * Float.BYTES;
        final ByteBuffer span = ByteBuffer.allocate(count * bytes).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < count; i++) {
            final float[] vector = vector(fileId, firstOffset + (long) i * bytes, dimension);
            for (int component = 0; component < dimension; component++) {
                span.putFloat(i * bytes + component * Float.BYTES, vector[component]);
            }
        }
        return span;
    }

    private static void fill(final ByteBuffer staging, final float[] vector) {
        for (int i = 0; i < vector.length; i++) {
            staging.putFloat(i * Float.BYTES, vector[i]);
        }
    }

    /** A vector whose every float is derived from its key, so a wrong vector cannot look right. */
    private static float[] vector(final String fileId, final long offset, final int dimension) {
        final long seed = fileId.hashCode() * 31L + offset;
        final float[] vector = new float[dimension];
        for (int i = 0; i < dimension; i++) {
            vector[i] = seed + i * 0.5f;
        }
        return vector;
    }
}

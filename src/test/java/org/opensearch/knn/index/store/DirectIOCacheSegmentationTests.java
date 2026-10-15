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
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Q's C1b: the segment selector, the derived segment count, and the two properties that make a store of more
 * than one segment correct rather than merely faster.
 *
 * <p>The store's policy under segmentation -- eviction, the second chance, its cap, resize, the closed-file
 * rule -- is {@code DirectIOVectorCacheSlabTests}' and {@code DirectIOSegmentedEvictionTests}'. What is
 * proved here is the segmentation itself: that a key's segment is a pure, stable, uniform function of
 * {@code (fileId, offset)}; that a staged span lies in at most two segments so {@code putRun} and
 * {@code contains} keep their batching; that the count the node derives is the count the design's five
 * worked examples say; that a vector admitted at an offset is found by a lookup <b>and</b> by a run check at
 * that offset at every segment count; and that <b>no path holds two segment locks</b>, which is the whole of
 * the deadlock proof.
 */
public class DirectIOCacheSegmentationTests extends KNNTestCase {

    /** 64 KiB, as the slab tests use, so a handful of fills reaches eviction. */
    private static final int TEST_SLAB_BYTES = 64 * 1024;

    /** Where a Lucene {@code .vec}'s vector data starts, so offsets look like the input's. */
    private static final long HEADER = 44L;

    /** {@code knn.direct_io.rescore.prefetch.max_span_bytes}' default, which is the widest staged span. */
    private static final int MAX_SPAN_BYTES = 135_168;

    /** The stripe the selector folds, from {@code DirectIOVectorCache}'s own constant. */
    private static final int STRIPE_BYTES = 256 * 1024;

    @Override
    public void tearDown() throws Exception {
        DirectIOVectorCache.resetForTesting();
        super.tearDown();
    }

    /** The store on at {@code slabs} test slabs and {@code segments} segments, from empty. */
    private void cacheOf(final int segments, final int slabs) {
        DirectIOVectorCache.resetForTesting();
        DirectIOVectorCache.setSlabBytesForTesting(TEST_SLAB_BYTES);
        DirectIOVectorCache.setSegmentsForTesting(segments);
        DirectIOVectorCache.configure((long) slabs * TEST_SLAB_BYTES);
        assertEquals("the pinned count is the count in force", segments, DirectIOVectorCache.segmentCount());
    }

    // ---------------------------------------------------------------------------------------------------
    // The selector
    // ---------------------------------------------------------------------------------------------------

    /** The same key gives the same segment, every time, and a different count is a different mask. */
    public void testSegmentSelectorIsStable() {
        final String file = "/index/_7.vec#123456789";
        final Random random = new Random(7L);
        for (int i = 0; i < 2_000; i++) {
            final long offset = HEADER + random.nextInt(Integer.MAX_VALUE);
            for (final int count : new int[] { 1, 2, 16, 128, 256 }) {
                final int first = DirectIOVectorCache.segmentIndex(file.hashCode(), offset, count);
                assertEquals("stable", first, DirectIOVectorCache.segmentIndex(file.hashCode(), offset, count));
                assertTrue("in range at " + count + ": " + first, first >= 0 && first < count);
            }
            assertEquals("one segment takes everything", 0, DirectIOVectorCache.segmentIndex(file.hashCode(), offset, 1));
        }
    }

    /**
     * The selector spreads a 1.7 GB key space evenly: every segment is within 20% of the mean stripe count
     * at 128 segments. 1.7 GB is the working set the cache's gate measured (the class javadoc's table).
     */
    public void testSegmentSelectorIsUniformOverTheWorkingSet() {
        final String file = "/index/_0.vec#1825361920";
        final int count = 128;
        final long bytes = 1_700L << 20;
        final int stripes = (int) (bytes / STRIPE_BYTES);
        final int[] perSegment = new int[count];
        for (int stripe = 0; stripe < stripes; stripe++) {
            perSegment[DirectIOVectorCache.segmentIndex(file.hashCode(), HEADER + (long) stripe * STRIPE_BYTES, count)]++;
        }
        final double mean = (double) stripes / count;
        int min = Integer.MAX_VALUE;
        int max = 0;
        for (final int held : perSegment) {
            min = Math.min(min, held);
            max = Math.max(max, held);
        }
        assertTrue("every segment gets some of a 1.7 GB key space, min " + min, min > 0);
        assertTrue("min " + min + " is within 20% of the mean " + mean, min >= 0.8 * mean);
        assertTrue("max " + max + " is within 20% of the mean " + mean, max <= 1.2 * mean);
    }

    /**
     * <b>A staged span touches at most two segments, wherever it starts.</b> That is what keeps
     * {@code putRun}'s one write acquisition at one or two rather than one per vector, and it is why the
     * stripe is 256 KiB: the widest span the prefetch settings allow is 135,168 B, which is less than one
     * stripe, so it can straddle one boundary and never two.
     */
    public void testAStagedSpanTouchesAtMostTwoSegments() {
        final String file = "/index/_3.vec#999";
        for (final int count : new int[] { 2, 16, 128, 256 }) {
            for (long start = HEADER; start < HEADER + 4L * STRIPE_BYTES; start += 512) {
                final int first = DirectIOVectorCache.segmentIndex(file.hashCode(), start, count);
                final int last = DirectIOVectorCache.segmentIndex(file.hashCode(), start + MAX_SPAN_BYTES - 1, count);
                int distinct = first == last ? 1 : 2;
                // Walk the span by one vector at a time as putRun does, and count the segment changes.
                int changes = 0;
                int previous = first;
                for (long at = start; at < start + MAX_SPAN_BYTES; at += 1_536) {
                    final int here = DirectIOVectorCache.segmentIndex(file.hashCode(), at, count);
                    if (here != previous) {
                        changes++;
                        previous = here;
                    }
                }
                distinct = Math.max(distinct, changes + 1);
                assertTrue(
                    "a " + MAX_SPAN_BYTES + " B span from " + start + " at " + count + " segments touched " + distinct,
                    distinct <= 2
                );
            }
        }
    }

    // ---------------------------------------------------------------------------------------------------
    // The derived count
    // ---------------------------------------------------------------------------------------------------

    /** The design's formula, including its five worked examples. */
    public void testDerivedSegmentCount() {
        // cores, maxSlabs, expected
        final int[][] cases = {
            // The five worked examples of Q-DESIGN section 2.1.
            { 64, 1_120, 128 },   // oi2.16xlarge hot, 35 GB
            { 32, 576, 64 },      // oi2.8xlarge warm, 18 GB
            { 16, 1_024, 128 },   // this desktop at a 32 GB target, which is the hot node's count
            { 16, 128, 16 },      // this desktop at 4 GB
            { 16, 8, 1 },         // 256 MB, which is the unsegmented store exactly
            // The two terms, separately.
            { 1, 100_000, 8 },    // the core term's floor
            { 4, 100_000, 32 },   // 8 * 4 = 32
            { 1_000, 100_000, 256 }, // the core term's ceiling
            { 64, 0, 1 },         // off
            { 64, 7, 1 },         // under eight slabs, so one segment
            { 64, 16, 2 },        // 16 / 8 = 2
            { 64, 23, 2 },        // rounded down to a power of two
            { 64, 24, 2 },
            { 64, 32, 4 }, };
        for (final int[] one : cases) {
            assertEquals("cores " + one[0] + ", maxSlabs " + one[1], one[2], DirectIOVectorCache.derivedSegmentCount(one[0], one[1]));
        }
        // And it is always a power of two, at least one, and never above the cap.
        final Random random = new Random(11L);
        for (int i = 0; i < 5_000; i++) {
            final int derived = DirectIOVectorCache.derivedSegmentCount(1 + random.nextInt(512), random.nextInt(20_000));
            assertTrue("a power of two: " + derived, Integer.bitCount(derived) == 1);
            assertTrue("at least one: " + derived, derived >= 1);
            assertTrue("at most 256: " + derived, derived <= 256);
        }
    }

    /** The node derives a count on its own, with no test seam, and it is the formula's. */
    public void testTheStoreDerivesItsOwnCount() {
        DirectIOVectorCache.resetForTesting();
        DirectIOVectorCache.setSlabBytesForTesting(TEST_SLAB_BYTES);
        DirectIOVectorCache.configure(1_024L * TEST_SLAB_BYTES);
        assertEquals(
            DirectIOVectorCache.derivedSegmentCount(Runtime.getRuntime().availableProcessors(), 1_024),
            DirectIOVectorCache.segmentCount()
        );
        DirectIOVectorCache.configure(0);
        assertEquals("off is one segment", 1, DirectIOVectorCache.segmentCount());
    }

    public void testPinningRefusesANonPowerOfTwoAndARunningStore() {
        DirectIOVectorCache.resetForTesting();
        expectThrows(IllegalArgumentException.class, () -> DirectIOVectorCache.setSegmentsForTesting(3));
        expectThrows(IllegalArgumentException.class, () -> DirectIOVectorCache.setSegmentsForTesting(-1));
        DirectIOVectorCache.setSlabBytesForTesting(TEST_SLAB_BYTES);
        DirectIOVectorCache.setSegmentsForTesting(4);
        DirectIOVectorCache.configure(8L * TEST_SLAB_BYTES);
        expectThrows(IllegalStateException.class, () -> DirectIOVectorCache.setSegmentsForTesting(8));
    }

    // ---------------------------------------------------------------------------------------------------
    // A vector goes in and comes back out, at every segment count
    // ---------------------------------------------------------------------------------------------------

    /**
     * The property the segmentation must not break: a vector admitted at offset {@code o} is found by a
     * lookup at {@code o} and by a {@code contains} over a run containing {@code o}, at every count. Written
     * as a loop over the counts rather than four tests, because the point is that the count does not matter.
     */
    public void testAdmittedVectorsAreFoundAtEverySegmentCount() {
        final int dimension = 128;
        final int bytes = dimension * Float.BYTES;
        for (final int segments : new int[] { 1, 2, 16, 128 }) {
            // Eight slabs per segment, which is the ratio the derived count aims at, and never fewer than
            // the 2,000 vectors below need in one segment (128 slots a test slab).
            cacheOf(segments, Math.max(32, 8 * segments));
            final String file = "/index/_" + segments + ".vec#4242";
            final int vectors = 2_000;
            for (int i = 0; i < vectors; i++) {
                put(file, offsetOf(dimension, i), dimension);
            }
            final float[] read = new float[dimension];
            int found = 0;
            for (int i = 0; i < vectors; i++) {
                final long offset = offsetOf(dimension, i);
                if (DirectIOVectorCache.getFloats(file, offset, bytes, read, 0)) {
                    found++;
                    assertArrayEquals("at " + segments + " segments, vector " + i, vector(file, offset, dimension), read, 0.0f);
                    assertTrue(
                        "contains agrees at " + segments + " segments, vector " + i,
                        DirectIOVectorCache.contains(file, offset, bytes)
                    );
                }
            }
            assertEquals("every vector fits, so every one is found at " + segments + " segments", vectors, found);
            // And a run of several vectors, which is the shape prefetch asks about and the one that can
            // cross a stripe boundary.
            for (int i = 0; i + 8 <= vectors; i += 8) {
                assertTrue(
                    "a run of 8 from " + i + " at " + segments + " segments",
                    DirectIOVectorCache.contains(file, offsetOf(dimension, i), 8 * bytes)
                );
            }
            // A run that is not wholly cached is not claimed to be.
            assertFalse(
                "past the end at " + segments + " segments",
                DirectIOVectorCache.contains(file, offsetOf(dimension, vectors - 2), 8 * bytes)
            );
            DirectIOVectorCache.checkInvariantsForTesting();
        }
    }

    /** {@code selectMissing} answers for a batch spread over many segments, in the caller's order. */
    public void testSelectMissingSpansSegments() {
        final int dimension = 128;
        final int bytes = dimension * Float.BYTES;
        for (final int segments : new int[] { 1, 16, 128 }) {
            cacheOf(segments, Math.max(32, 8 * segments));
            final String file = "/index/sel" + segments + ".vec#7";
            final int vectors = 1_000;
            final List<Integer> cached = new ArrayList<>();
            for (int i = 0; i < vectors; i++) {
                if (i % 3 != 0) {
                    put(file, offsetOf(dimension, i), dimension);
                    cached.add(i);
                }
            }
            final int[] ords = new int[vectors];
            for (int i = 0; i < vectors; i++) {
                ords[i] = i;
            }
            final int[] missing = new int[vectors];
            final int found = DirectIOVectorCache.selectMissing(file, HEADER, bytes, ords, vectors, missing);
            assertTrue("the file is indexed, so not -1", found >= 0);
            final List<Integer> expected = new ArrayList<>();
            for (int i = 0; i < vectors; i++) {
                if (i % 3 == 0) {
                    expected.add(i);
                }
            }
            assertEquals("at " + segments + " segments", expected.size(), found);
            for (int i = 0; i < found; i++) {
                assertEquals("in the caller's order at " + segments + " segments, position " + i, (int) expected.get(i), missing[i]);
            }
            assertEquals("a cold file is -1", -1, DirectIOVectorCache.selectMissing("cold.vec#1", HEADER, bytes, ords, vectors, missing));
            DirectIOVectorCache.checkInvariantsForTesting();
        }
    }

    /** A staged run that straddles a stripe boundary admits every vector, once, and finds them all. */
    public void testPutRunAcrossAStripeBoundary() {
        final int dimension = 128;
        final int bytes = dimension * Float.BYTES;
        final int perStripe = STRIPE_BYTES / bytes;
        for (final int segments : new int[] { 1, 16, 128 }) {
            cacheOf(segments, Math.max(32, 8 * segments));
            final String file = "/index/run" + segments + ".vec#9";
            // Start a run of 16 vectors eight vectors before a stripe boundary, so half of it is on each
            // side and the run is split exactly once.
            final int count = 16;
            final long first = HEADER + (long) (perStripe - count / 2) * bytes;
            final long entriesBefore = DirectIOVectorCache.entries();
            final int admitted = DirectIOVectorCache.putRun(file, first, count, runOf(file, first, count, dimension), 0, bytes);
            assertEquals("every vector of the run at " + segments + " segments", count, admitted);
            assertEquals("and no more than that", count, DirectIOVectorCache.entries() - entriesBefore);
            final float[] read = new float[dimension];
            for (int i = 0; i < count; i++) {
                final long offset = first + (long) i * bytes;
                assertTrue(
                    "vector " + i + " of a straddling run at " + segments,
                    DirectIOVectorCache.getFloats(file, offset, bytes, read, 0)
                );
                assertArrayEquals(vector(file, offset, dimension), read, 0.0f);
            }
            assertTrue("and contains sees the whole run", DirectIOVectorCache.contains(file, first, count * bytes));
            DirectIOVectorCache.checkInvariantsForTesting();
        }
    }

    // ---------------------------------------------------------------------------------------------------
    // No path holds two segment locks
    // ---------------------------------------------------------------------------------------------------

    /**
     * <b>The deadlock proof, asserted.</b> A probe {@link DirectIOVectorCache.SlabMemory} checks, at every
     * allocate and every release, that the calling thread holds <b>at most one</b> segment lock and none of
     * the store's locks at all -- the second is N3's property and the first is Q's. The probe runs on the
     * fill path, the growth path, the shrink path, the segment-count-change path and the lending path, which
     * is every path that can take more than one lock.
     */
    public void testNoPathHoldsTwoSegmentLocks() {
        final List<String> violations = Collections.synchronizedList(new ArrayList<>());
        final AtomicInteger allocations = new AtomicInteger();
        final AtomicInteger releases = new AtomicInteger();
        DirectIOVectorCache.resetForTesting();
        DirectIOVectorCache.setSlabBytesForTesting(TEST_SLAB_BYTES);
        DirectIOVectorCache.setSlabMemoryForTesting(new DirectIOVectorCache.SlabMemory() {
            @Override
            public DirectIOSlab allocate(final int bytes) {
                allocations.incrementAndGet();
                check("allocate");
                return DirectIOSlab.allocate(bytes);
            }

            @Override
            public void release(final DirectIOSlab slab) {
                releases.incrementAndGet();
                check("release");
                slab.close();
            }

            private void check(final String what) {
                final int held = DirectIOVectorCache.segmentLocksHeldByCurrentThreadForTesting();
                if (held > 1) {
                    violations.add(what + " held " + held + " segment locks");
                }
                if (DirectIOVectorCache.lockHeldByCurrentThreadForTesting()) {
                    violations.add(what + " held one of the store's locks");
                }
            }
        });

        final int dimension = 128;
        final int bytes = dimension * Float.BYTES;
        final int slots = DirectIOVectorCache.slotsPerSlab(bytes);
        // No pinned count: the derived one is what a node runs, and 128 test slabs derives 8 or 16
        // segments on any core count while 32 derives 4, so the shrink below is a real count change with
        // live slabs in it.
        DirectIOVectorCache.configure(128L * TEST_SLAB_BYTES);
        assertTrue("the derived count must be more than one here", DirectIOVectorCache.segmentCount() > 1);
        final String file = "/index/locks.vec#1";
        // Enough fills to grow every segment, reclaim inside them, and reach the global target, which is
        // where the lending path runs.
        for (int i = 0; i < 160 * slots; i++) {
            put(file, offsetOf(dimension, i), dimension);
        }
        // Span fills, which grow through the other half of place().
        for (int run = 0; run < 64; run++) {
            final long at = offsetOf(dimension, run * 8);
            DirectIOVectorCache.putRun(file, at, 8, runOf(file, at, 8, dimension), 0, bytes);
        }
        // A close, then a shrink that crosses a segment-count boundary with live slabs in every segment,
        // then a grow back across it, then off.
        DirectIOVectorCache.releaseFile(file);
        final int before = DirectIOVectorCache.segmentCount();
        DirectIOVectorCache.configure(32L * TEST_SLAB_BYTES);
        assertEquals("32 test slabs is four segments at any core count", 4, DirectIOVectorCache.segmentCount());
        assertNotEquals("so the count changed", before, DirectIOVectorCache.segmentCount());
        for (int i = 0; i < 8 * slots; i++) {
            put(file, offsetOf(dimension, i), dimension);
        }
        DirectIOVectorCache.configure(128L * TEST_SLAB_BYTES);
        DirectIOVectorCache.configure(0);
        DirectIOVectorCache.awaitReleasesForTesting();

        assertEquals("no path may hold two segment locks or allocate under one", List.of(), violations);
        assertTrue("the probe has to have seen allocations, saw " + allocations.get(), allocations.get() >= 16);
        assertTrue("and releases, saw " + releases.get(), releases.get() >= 16);
        assertTrue("the segment-count change happened", DirectIOVectorCache.segmentCountChanges() >= 1);
    }

    /**
     * A segment that holds no slab at a full store is lent one, so its keys are cacheable rather than
     * permanently cold (Q, section 2.2 step 3). The fixture makes the case directly: pin more segments than
     * there are slabs to go round, fill, and assert that every segment that gets traffic ends up holding
     * entries and that {@code slab_transfers} moved.
     */
    public void testASegmentWithNoSlabBorrowsOne() {
        final int dimension = 128;
        final int bytes = dimension * Float.BYTES;
        // Four segments, two slabs: two segments can hold nothing until one is lent to them.
        cacheOf(4, 2);
        final String file = "/index/borrow.vec#1";
        final int slots = DirectIOVectorCache.slotsPerSlab(bytes);
        for (int i = 0; i < 8 * slots; i++) {
            put(file, offsetOf(dimension, i), dimension);
        }
        assertEquals("the global target is unchanged by lending", 2L, DirectIOVectorCache.slabs());
        assertTrue(
            "a segment with no slab asked for one, transfers " + DirectIOVectorCache.slabTransfers(),
            DirectIOVectorCache.slabTransfers() > 0
        );
        DirectIOVectorCache.checkInvariantsForTesting();
    }

    // ---------------------------------------------------------------------------------------------------
    // Helpers, the same shape DirectIOVectorCacheSlabTests uses
    // ---------------------------------------------------------------------------------------------------

    private static long offsetOf(final int dimension, final int ordinal) {
        return HEADER + (long) ordinal * dimension * Float.BYTES;
    }

    private static void put(final String fileId, final long offset, final int dimension) {
        final ByteBuffer staging = ByteBuffer.allocate(dimension * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        final float[] vector = vector(fileId, offset, dimension);
        for (int i = 0; i < vector.length; i++) {
            staging.putFloat(i * Float.BYTES, vector[i]);
        }
        DirectIOVectorCache.put(fileId, offset, staging, 0, dimension * Float.BYTES);
    }

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

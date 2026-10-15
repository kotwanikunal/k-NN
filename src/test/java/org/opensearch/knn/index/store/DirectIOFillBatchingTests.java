/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import org.opensearch.knn.KNNTestCase;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Q's C2: the batched sites stay batched.
 *
 * <p>Three call sites were one lock acquisition before the store was segmented, and all three would have
 * become one acquisition per vector if the segmentation had been naive: {@code selectMissing} for a whole
 * batch, {@code contains} for a run, and {@code putRun} for a staged span. What is proved here is that they
 * are still batched -- at most one acquisition per segment the call touches -- <b>and</b> that the batching
 * did not change a single answer, in particular that {@code selectMissing} still writes its misses out in the
 * caller's ordinal order, which {@code DirectIOVectorIndexInput#recordFilteredAsCached} relies on when it
 * finds the complement in one pass over both arrays.
 */
public class DirectIOFillBatchingTests extends KNNTestCase {

    private static final int TEST_SLAB_BYTES = 64 * 1024;
    private static final long HEADER = 44L;
    private static final int DIMENSION = 128;
    private static final int VECTOR_BYTES = DIMENSION * Float.BYTES;

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
    // selectMissing answers exactly what the unsegmented store answered
    // ---------------------------------------------------------------------------------------------------

    /**
     * <b>Exactly the same ints in the same order as a segment-blind reference</b>, for 1,000 random batches
     * at each of three segment counts, including an all-cached batch, an all-uncached batch and the cold-file
     * {@code -1}. The reference is computed from {@code getFloats} on a copy of the store state, which is the
     * definition {@code selectMissing} documents: "the members of {@code ords} whose vector this store does
     * not hold whole".
     */
    public void testSelectMissingIsExactlyTheReferenceAnswer() {
        final Random random = new Random(31L);
        for (final int segments : new int[] { 1, 16, 128 }) {
            cacheOf(segments, Math.max(32, 8 * segments));
            final String file = "/index/batch" + segments + ".vec#5";
            final int vectors = 600;
            final boolean[] cached = new boolean[vectors];
            for (int i = 0; i < vectors; i++) {
                if (random.nextInt(3) != 0) {
                    put(file, offsetOf(i));
                    cached[i] = true;
                }
            }
            // What the store actually holds, read one vector at a time through the serving lookup. That is
            // the reference: selectMissing must agree with getFloats on every ordinal.
            final float[] read = new float[DIMENSION];
            for (int i = 0; i < vectors; i++) {
                cached[i] = DirectIOVectorCache.getFloats(file, offsetOf(i), VECTOR_BYTES, read, 0);
            }

            for (int batch = 0; batch < 1_000; batch++) {
                final int numOrds = 1 + random.nextInt(64);
                final int[] ords = new int[numOrds];
                for (int i = 0; i < numOrds; i++) {
                    ords[i] = random.nextInt(vectors);
                }
                assertSelectMissingMatches(file, ords, numOrds, cached, segments);
            }
            // The three shapes a random batch may not produce.
            final int[] allCached = new int[32];
            final int[] allUncached = new int[32];
            int cachedAt = 0;
            int uncachedAt = 0;
            for (int i = 0; i < vectors && (cachedAt < 32 || uncachedAt < 32); i++) {
                if (cached[i] && cachedAt < 32) {
                    allCached[cachedAt++] = i;
                } else if (cached[i] == false && uncachedAt < 32) {
                    allUncached[uncachedAt++] = i;
                }
            }
            assertEquals("the fixture must hold 32 cached vectors at " + segments + " segments", 32, cachedAt);
            assertEquals("and 32 uncached", 32, uncachedAt);
            assertSelectMissingMatches(file, allCached, 32, cached, segments);
            assertSelectMissingMatches(file, allUncached, 32, cached, segments);
            // A file the store holds nothing of is -1 and not a copy of every ordinal.
            final int[] missing = new int[32];
            assertEquals(
                "a cold file at " + segments + " segments",
                -1,
                DirectIOVectorCache.selectMissing("/index/cold.vec#1", HEADER, VECTOR_BYTES, allCached, 32, missing)
            );
            // And a repeated ordinal is answered once per occurrence, in place.
            final int[] repeated = { allUncached[0], allUncached[0], allCached[0], allUncached[1], allUncached[0] };
            assertSelectMissingMatches(file, repeated, repeated.length, cached, segments);
            DirectIOVectorCache.checkInvariantsForTesting();
        }
    }

    private void assertSelectMissingMatches(
        final String file,
        final int[] ords,
        final int numOrds,
        final boolean[] cached,
        final int segments
    ) {
        final List<Integer> expected = new ArrayList<>();
        for (int i = 0; i < numOrds; i++) {
            if (cached[ords[i]] == false) {
                expected.add(ords[i]);
            }
        }
        final int[] missing = new int[Math.max(1, numOrds)];
        final int found = DirectIOVectorCache.selectMissing(file, HEADER, VECTOR_BYTES, ords, numOrds, missing);
        assertEquals("count at " + segments + " segments for " + Arrays.toString(Arrays.copyOf(ords, numOrds)), expected.size(), found);
        for (int i = 0; i < found; i++) {
            assertEquals("position " + i + " at " + segments + " segments", (int) expected.get(i), missing[i]);
        }
        // The ordering requirement, stated as the consumer states it: the misses are a subsequence of the
        // batch, in order. DirectIOVectorIndexInput#recordFilteredAsCached walks both arrays once and would
        // record the wrong complement otherwise.
        int at = 0;
        for (int i = 0; i < numOrds && at < found; i++) {
            if (ords[i] == missing[at]) {
                at++;
            }
        }
        assertEquals("the misses are a subsequence of the batch, in order", found, at);
    }

    // ---------------------------------------------------------------------------------------------------
    // The acquisition counts
    // ---------------------------------------------------------------------------------------------------

    /**
     * <b>At most one read acquisition per segment the batch touches</b>, which is the BACKLOG's criterion for
     * {@code selectMissing}, and the 1T shape's number recorded whether or not it flatters the design: a
     * 190-candidate batch over 128 segments.
     */
    public void testSelectMissingTakesOneAcquisitionPerTouchedSegment() {
        final Random random = new Random(41L);
        for (final int segments : new int[] { 1, 16, 128 }) {
            cacheOf(segments, Math.max(32, 8 * segments));
            final String file = "/index/acq" + segments + ".vec#5";
            // A few hundred vectors cached, but the ordinals asked about are drawn from the 1T shape's own
            // key space: 12.5 million vectors in a shard (readiness section 2), which is 73 thousand 256 KiB
            // stripes, so 190 random candidates land in 190 different stripes and the segment count is what
            // decides how many segments that is. A fixture small enough to cache whole would put every
            // candidate in a handful of stripes and would measure the fixture rather than the design.
            final int perShard = 12_500_000;
            for (int i = 0; i < 400; i++) {
                put(file, offsetOf(random.nextInt(perShard)));
            }
            final int numOrds = 190;
            final int[] ords = new int[numOrds];
            for (int i = 0; i < numOrds; i++) {
                ords[i] = random.nextInt(perShard);
            }
            final int touched = distinctSegments(file, ords, numOrds, segments);
            final int[] missing = new int[numOrds];
            DirectIOVectorCache.setAcquisitionCountingForTesting(true);
            DirectIOVectorCache.selectMissing(file, HEADER, VECTOR_BYTES, ords, numOrds, missing);
            final long acquisitions = DirectIOVectorCache.readAcquisitionsForTesting();
            DirectIOVectorCache.setAcquisitionCountingForTesting(false);
            assertEquals(
                "one read acquisition per touched segment at " + segments + " segments (touched " + touched + ")",
                touched,
                acquisitions
            );
            assertTrue("and never more than the batch", acquisitions <= numOrds);
            assertTrue("nor more than the segment count", acquisitions <= segments);
            logger.info(
                "selectMissing at {} segments: {} acquisitions for {} candidates over a {}-vector key space",
                segments,
                acquisitions,
                numOrds,
                perShard
            );
        }
    }

    /**
     * <b>Write acquisitions per burst no more than the segments the burst touched.</b> A burst here is one
     * staged span admitted through {@code putRun}, which is the write-locking site span fill uses, plus the
     * one requested fill behind each miss.
     */
    public void testWriteAcquisitionsPerBurstAreBoundedBySegmentsTouched() {
        for (final int segments : new int[] { 16, 128 }) {
            cacheOf(segments, Math.max(32, 8 * segments));
            final String file = "/index/burst" + segments + ".vec#5";
            // A staged span of 43 vectors, which is max_span_bytes / 3,072 at the defaults and so the widest
            // run putRun is ever handed. It starts near a stripe boundary, so it is the two-segment case.
            final int count = 43;
            final int firstOrdinal = 1_000;
            final long first = offsetOf(firstOrdinal);
            final int touched = distinctSegmentsOfRun(file, first, count, segments);
            assertTrue("a span of 43 vectors touches at most two segments, saw " + touched, touched <= 2);
            // Warm the stripes the span lies in, so each of its segments has an open slab with room and the
            // run needs no slab of its own. That is the steady state, which is where a node lives: at the
            // slab target span fill admits nothing at all, and below it a run that fills the open slab needs
            // one more acquisition to fetch a slab outside the lock, which is N3's rule and predates Q.
            for (int i = firstOrdinal - 50; i < firstOrdinal + count + 50; i++) {
                if (i < firstOrdinal || i >= firstOrdinal + count) {
                    put(file, offsetOf(i));
                }
            }
            DirectIOVectorCache.setAcquisitionCountingForTesting(true);
            final int admitted = DirectIOVectorCache.putRun(file, first, count, runOf(file, first, count), 0, VECTOR_BYTES);
            final long acquisitions = DirectIOVectorCache.writeAcquisitionsForTesting();
            DirectIOVectorCache.setAcquisitionCountingForTesting(false);
            assertEquals("every vector of the span", count, admitted);
            assertEquals(
                "one write acquisition per touched segment at " + segments + " segments (touched " + touched + ")",
                touched,
                acquisitions
            );
            logger.info("putRun at {} segments: {} write acquisitions for a {}-vector span", segments, acquisitions, count);
            DirectIOVectorCache.checkInvariantsForTesting();
        }
    }

    /**
     * And the one case that needs a second acquisition per segment, so that the bound above is not read as
     * stricter than it is: a run that fills the open slab leaves the lock to fetch a slab and comes back,
     * which is N3's rule and predates the segmentation. At most two acquisitions per touched segment.
     */
    public void testASpanThatNeedsASlabTakesAtMostTwoAcquisitionsPerSegment() {
        for (final int segments : new int[] { 16, 128 }) {
            cacheOf(segments, Math.max(32, 8 * segments));
            final String file = "/index/grow" + segments + ".vec#5";
            final int count = 43;
            final long first = offsetOf(1_000);
            final int touched = distinctSegmentsOfRun(file, first, count, segments);
            DirectIOVectorCache.setAcquisitionCountingForTesting(true);
            final int admitted = DirectIOVectorCache.putRun(file, first, count, runOf(file, first, count), 0, VECTOR_BYTES);
            final long acquisitions = DirectIOVectorCache.writeAcquisitionsForTesting();
            DirectIOVectorCache.setAcquisitionCountingForTesting(false);
            assertEquals("every vector of the span", count, admitted);
            assertTrue(
                "at most two per touched segment at " + segments + " segments (touched " + touched + ", took " + acquisitions + ")",
                acquisitions <= 2L * touched
            );
            DirectIOVectorCache.checkInvariantsForTesting();
        }
    }

    /**
     * {@code putRun} over a run that straddles a stripe boundary admits every vector <b>exactly once</b> and
     * admits the same set as the unsegmented store does.
     */
    public void testPutRunAdmitsEachVectorOnceAcrossABoundary() {
        final int perStripe = (256 * 1024) / VECTOR_BYTES;
        final int count = 24;
        // The same run at one segment and at 128, and the sets must be equal.
        final List<Long> atOne = admittedOffsets(1, perStripe, count);
        final List<Long> atMany = admittedOffsets(128, perStripe, count);
        assertEquals("the same vectors are admitted", atOne, atMany);
        assertEquals("and every one of them, once", count, atMany.size());
    }

    private List<Long> admittedOffsets(final int segments, final int perStripe, final int count) {
        cacheOf(segments, Math.max(32, 8 * segments));
        final String file = "/index/straddle.vec#5";
        final long first = HEADER + (long) (perStripe - count / 2) * VECTOR_BYTES;
        final int admitted = DirectIOVectorCache.putRun(file, first, count, runOf(file, first, count), 0, VECTOR_BYTES);
        assertEquals("every vector of the straddling run at " + segments + " segments", count, admitted);
        final List<Long> offsets = new ArrayList<>();
        final float[] read = new float[DIMENSION];
        for (int i = 0; i < count; i++) {
            final long offset = first + (long) i * VECTOR_BYTES;
            if (DirectIOVectorCache.getFloats(file, offset, VECTOR_BYTES, read, 0)) {
                offsets.add(offset);
                assertArrayEquals(vector(file, offset), read, 0.0f);
            }
        }
        assertEquals("no vector admitted twice", count, DirectIOVectorCache.entries());
        DirectIOVectorCache.checkInvariantsForTesting();
        return offsets;
    }

    // ---------------------------------------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------------------------------------

    private static int distinctSegments(final String file, final int[] ords, final int numOrds, final int segments) {
        final boolean[] seen = new boolean[segments];
        int distinct = 0;
        for (int i = 0; i < numOrds; i++) {
            final int segment = DirectIOVectorCache.segmentIndex(file.hashCode(), offsetOf(ords[i]), segments);
            if (seen[segment] == false) {
                seen[segment] = true;
                distinct++;
            }
        }
        return distinct;
    }

    private static int distinctSegmentsOfRun(final String file, final long first, final int count, final int segments) {
        final boolean[] seen = new boolean[segments];
        int distinct = 0;
        for (int i = 0; i < count; i++) {
            final int segment = DirectIOVectorCache.segmentIndex(file.hashCode(), first + (long) i * VECTOR_BYTES, segments);
            if (seen[segment] == false) {
                seen[segment] = true;
                distinct++;
            }
        }
        return distinct;
    }

    private static long offsetOf(final int ordinal) {
        return HEADER + (long) ordinal * VECTOR_BYTES;
    }

    private static void put(final String fileId, final long offset) {
        final ByteBuffer staging = ByteBuffer.allocate(VECTOR_BYTES).order(ByteOrder.LITTLE_ENDIAN);
        final float[] vector = vector(fileId, offset);
        for (int i = 0; i < vector.length; i++) {
            staging.putFloat(i * Float.BYTES, vector[i]);
        }
        DirectIOVectorCache.put(fileId, offset, staging, 0, VECTOR_BYTES);
    }

    private static ByteBuffer runOf(final String fileId, final long firstOffset, final int count) {
        final ByteBuffer span = ByteBuffer.allocate(count * VECTOR_BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < count; i++) {
            final float[] vector = vector(fileId, firstOffset + (long) i * VECTOR_BYTES);
            for (int component = 0; component < DIMENSION; component++) {
                span.putFloat(i * VECTOR_BYTES + component * Float.BYTES, vector[component]);
            }
        }
        return span;
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

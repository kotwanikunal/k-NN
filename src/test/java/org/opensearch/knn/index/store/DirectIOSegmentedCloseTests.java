/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import org.opensearch.knn.KNNTestCase;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Q's C4: a close that nothing waits behind.
 *
 * <p>{@code releaseFile} took the store's one write lock and walked a whole file's index, which the readiness
 * doc prices at 37 ms per 500 thousand entries, about 16 ms per shard when a tenant leaves hot. Segmented it
 * takes one segment's write lock at a time and walks that segment's cells only, so a lookup in any of the
 * other {@code N - 1} segments waits for nothing and a lookup in the one being drained waits for a
 * {@code 1/N} share.
 *
 * <p>What is proved here is the bound, measured against the <b>unsegmented hold on this same host</b> rather
 * than against the readiness doc's figure, and that the accounting did not move:
 * {@code DirectIOClosedFileReleaseTests} is the accounting's own suite and passes unedited.
 */
public class DirectIOSegmentedCloseTests extends KNNTestCase {

    private static final int TEST_SLAB_BYTES = 64 * 1024;
    /** 32 dimensions, so a test slab holds 512 of them and a big fixture is still cheap. */
    private static final int DIMENSION = 32;
    private static final int VECTOR_BYTES = DIMENSION * Float.BYTES;
    private static final int STRIPE_BYTES = 256 * 1024;

    /** 1,024 stripes of 128 vectors: 131,072 entries, spread so that all 128 segments are used. */
    private static final int STRIPES = 1_024;
    private static final int PER_STRIPE = 128;
    private static final int ENTRIES = STRIPES * PER_STRIPE;
    /** 512 slabs of 512 slots is 262,144 slots, so the fixture sits at half load and never evicts. */
    private static final int SLABS = 512;

    @Override
    public void tearDown() throws Exception {
        DirectIOVectorCache.resetForTesting();
        super.tearDown();
    }

    private void cacheOf(final int segments) {
        DirectIOVectorCache.resetForTesting();
        DirectIOVectorCache.setSlabBytesForTesting(TEST_SLAB_BYTES);
        if (segments > 0) {
            DirectIOVectorCache.setSegmentsForTesting(segments);
        }
        DirectIOVectorCache.configure((long) SLABS * TEST_SLAB_BYTES);
    }

    /**
     * <b>The bound, measured both ways on this host.</b> The same fixture is closed at one segment and at
     * 128, and the longest single hold at 128 segments must be a small fraction of the one at one segment.
     * The bar is a {@code 1/N} share with headroom; this asserts a thirty-second, which is four times the
     * slack, and logs what it actually measured.
     *
     * <p>Three repetitions at each count and the <b>median</b> of the maxima, because the statistic is a
     * maximum of a wall-clock interval and a collection pause inside one hold lands in it whole: one
     * observation at 128 segments came out at 1,256 us against a median of about 130, which is a pause and
     * not a hold.
     */
    public void testTheLongestCloseHoldFallsWithTheSegmentCount() {
        final long[] atOne = new long[3];
        final long[] atMany = new long[3];
        for (int rep = 0; rep < atOne.length; rep++) {
            atOne[rep] = closeAndMeasure(1);
        }
        for (int rep = 0; rep < atMany.length; rep++) {
            atMany[rep] = closeAndMeasure(128);
        }
        final long[] oneSorted = atOne.clone();
        final long[] manySorted = atMany.clone();
        Arrays.sort(oneSorted);
        Arrays.sort(manySorted);
        final long holdAtOne = oneSorted[1];
        final long holdAtMany = manySorted[1];
        logger.info(
            "closing {} entries: longest hold {} us at one segment (of {}), {} us at 128 (of {}), a factor of {}",
            ENTRIES,
            holdAtOne / 1_000,
            Arrays.toString(atOne),
            holdAtMany / 1_000,
            Arrays.toString(atMany),
            holdAtOne / Math.max(1, holdAtMany)
        );
        assertTrue("the unsegmented hold must be measurable to compare against, saw " + holdAtOne + " ns", holdAtOne > 1_000_000);
        assertTrue(
            "the longest hold at 128 segments ("
                + holdAtMany
                + " ns) must be a small fraction of the one at one segment ("
                + holdAtOne
                + " ns)",
            holdAtMany * 32 <= holdAtOne
        );
        for (final long one : atMany) {
            assertTrue("every hold at 128 segments is under 5 ms, saw " + one / 1_000 + " us", one < 5_000_000L);
        }
    }

    /** One fill and one close at {@code segments} segments; returns the longest hold it took. */
    private long closeAndMeasure(final int segments) {
        final String file = "/index/close.vec#1";
        cacheOf(segments);
        assertEquals(segments, DirectIOVectorCache.segmentCount());
        fill(file);
        assertEquals("the fixture must fit whole at " + segments + " segments", ENTRIES, DirectIOVectorCache.entries());
        final long evictionsBefore = DirectIOVectorCache.evictions();
        final long freed = DirectIOVectorCache.releaseFile(file);
        assertEquals("every entry freed at " + segments + " segments", ENTRIES, freed);
        assertEquals("nothing was evicted to do it", evictionsBefore, DirectIOVectorCache.evictions());
        assertEquals("and the store is empty", 0L, DirectIOVectorCache.entries());
        assertEquals(1L, DirectIOVectorCache.closedFiles());
        assertEquals(ENTRIES, DirectIOVectorCache.closedFileEntries());
        assertFalse("no index table is left anywhere", DirectIOVectorCache.hasIndexFor(file));
        DirectIOVectorCache.checkInvariantsForTesting();
        return DirectIOVectorCache.longestCloseHoldNanos();
    }

    /**
     * <b>No lookup waits behind a close</b>, measured as the A/B it is: the same reader, the same fixture,
     * the same close, at one segment and at 128. What a lookup waits behind is one segment's hold, not the
     * whole file's, so its tail across the close falls with the segment count. The reader's offsets are
     * spread one per stripe, so its lookups land in every segment exactly as a real batch's do.
     */
    public void testALookupsTailAcrossACloseFallsWithTheSegmentCount() throws Exception {
        final long[] atOne = readerTailAcrossAClose(1);
        final long[] atMany = readerTailAcrossAClose(128);
        logger.info(
            "a lookup across a {}-entry close: quiet p99 {} ns / during p99 {} ns / during max {} ns / longest hold {} us "
                + "at ONE segment; {} / {} / {} / {} at 128",
            ENTRIES,
            atOne[0],
            atOne[1],
            atOne[2],
            atOne[3] / 1_000,
            atMany[0],
            atMany[1],
            atMany[2],
            atMany[3] / 1_000
        );
        logger.info("and it served {} lookups across the close at one segment against {} at 128", atOne[4], atMany[4]);
        // The statistic is the WORST lookup and the number served, not the p99. At one segment the reader
        // is blocked for the whole close and so takes only a couple of hundred samples, of which exactly one
        // is the blocked one, so its p99 says nothing; the max says everything.
        assertTrue("the unsegmented close must block a lookup for a millisecond or more, saw " + atOne[2] + " ns", atOne[2] > 1_000_000);
        assertTrue(
            "the worst lookup across the close must fall with the segment count: "
                + atOne[2]
                + " ns at one segment, "
                + atMany[2]
                + " ns at 128",
            atMany[2] * 10 <= atOne[2]
        );
        assertTrue(
            "and its worst single lookup is bounded by ONE segment's hold, not the file's: max "
                + atMany[2]
                + " ns against a "
                + "longest hold of "
                + atMany[3]
                + " ns",
            atMany[2] <= 4 * atMany[3]
        );
        assertTrue(
            "and it served far more lookups across the close at 128 segments (" + atMany[4] + ") than at one (" + atOne[4] + ")",
            atMany[4] >= 5 * atOne[4]
        );
        assertTrue("the longest hold at 128 segments is under 5 ms, saw " + atMany[3] / 1_000 + " us", atMany[3] < 5_000_000L);
        assertTrue("and it fell with the count: " + atOne[3] + " ns against " + atMany[3] + " ns", atMany[3] * 8 <= atOne[3]);
    }

    /**
     * Runs a reader across a close at {@code segments} segments and returns
     * {@code {quiet p99, during p99, during max, longest close hold}} in nanoseconds.
     */
    private long[] readerTailAcrossAClose(final int segments) throws Exception {
        cacheOf(segments);
        final String closing = "/index/draining.vec#1";
        final String reading = "/index/reading.vec#1";
        fill(closing);
        final long[] readerOffsets = new long[STRIPES];
        for (int stripe = 0; stripe < STRIPES; stripe++) {
            readerOffsets[stripe] = offsetOf(stripe, 0);
            put(reading, readerOffsets[stripe]);
        }

        final AtomicBoolean stop = new AtomicBoolean();
        final AtomicReference<Throwable> failure = new AtomicReference<>();
        final CountDownLatch warm = new CountDownLatch(1);
        final long[][] samples = new long[2][];
        final int sampleCount = 400_000;

        final Thread reader = new Thread(() -> {
            try {
                final float[] read = new float[DIMENSION];
                final long[] quiet = new long[sampleCount];
                final long[] during = new long[sampleCount];
                int at = 0;
                long[] into = quiet;
                boolean switched = false;
                int index = 0;
                final long quietUntil = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(400);
                while (stop.get() == false || switched == false) {
                    if (switched == false && System.nanoTime() > quietUntil) {
                        samples[0] = Arrays.copyOf(quiet, at);
                        at = 0;
                        into = during;
                        switched = true;
                        warm.countDown();
                    }
                    final long started = System.nanoTime();
                    DirectIOVectorCache.getFloats(reading, readerOffsets[index], VECTOR_BYTES, read, 0);
                    final long took = System.nanoTime() - started;
                    if (at < into.length) {
                        into[at++] = took;
                    }
                    index = index + 1 == readerOffsets.length ? 0 : index + 1;
                }
                samples[1] = Arrays.copyOf(during, at);
            } catch (Throwable e) {
                failure.compareAndSet(null, e);
                warm.countDown();
            }
        }, "c4-reader-" + segments);
        reader.setDaemon(true);
        reader.start();

        assertTrue("the reader must get its quiet period", warm.await(30, TimeUnit.SECONDS));
        final long freed = DirectIOVectorCache.releaseFile(closing);
        stop.set(true);
        reader.join(TimeUnit.SECONDS.toMillis(30));
        if (failure.get() != null) {
            throw new AssertionError(failure.get());
        }
        assertEquals("the close did its work at " + segments + " segments", ENTRIES, freed);
        assertNotNull("the quiet samples", samples[0]);
        assertNotNull("the samples taken across the close", samples[1]);
        // Deliberately low: at ONE segment the reader is blocked for the whole close, which is the thing
        // being measured, so this guard only says the sampling worked.
        assertTrue("the reader took some samples across the close, saw " + samples[1].length, samples[1].length > 20);
        DirectIOVectorCache.checkInvariantsForTesting();
        return new long[] {
            percentile(samples[0], 0.99),
            percentile(samples[1], 0.99),
            percentile(samples[1], 1.0),
            DirectIOVectorCache.longestCloseHoldNanos(),
            samples[1].length };
    }

    /** Closing one of two files frees exactly its share at 128 segments, and the other keeps serving. */
    public void testClosingOneOfTwoFilesFreesExactlyItsShareAcrossSegments() {
        cacheOf(128);
        final String going = "/index/going.vec#1";
        final String staying = "/index/staying.vec#1";
        for (int stripe = 0; stripe < 256; stripe++) {
            for (int i = 0; i < 16; i++) {
                put(going, offsetOf(stripe, i));
                put(staying, offsetOf(stripe, i));
            }
        }
        final long total = DirectIOVectorCache.entries();
        assertEquals(2L * 256 * 16, total);
        final long evictionsBefore = DirectIOVectorCache.evictions();

        final long freed = DirectIOVectorCache.releaseFile(going);

        assertEquals("its share and no more", total / 2, freed);
        assertEquals(total / 2, DirectIOVectorCache.entries());
        assertEquals("evictions does not move for a close", evictionsBefore, DirectIOVectorCache.evictions());
        assertFalse("the closed file has no index table in any segment", DirectIOVectorCache.hasIndexFor(going));
        assertTrue("the other file still has one", DirectIOVectorCache.hasIndexFor(staying));
        final float[] read = new float[DIMENSION];
        for (int stripe = 0; stripe < 256; stripe++) {
            for (int i = 0; i < 16; i++) {
                final long offset = offsetOf(stripe, i);
                assertFalse("the closed file's vectors are gone", DirectIOVectorCache.getFloats(going, offset, VECTOR_BYTES, read, 0));
                assertTrue("the other file's are not", DirectIOVectorCache.getFloats(staying, offset, VECTOR_BYTES, read, 0));
                assertArrayEquals(vector(staying, offset), read, 0.0f);
            }
        }
        DirectIOVectorCache.checkInvariantsForTesting();
    }

    // ---------------------------------------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------------------------------------

    private static void fill(final String file) {
        for (int stripe = 0; stripe < STRIPES; stripe++) {
            for (int i = 0; i < PER_STRIPE; i++) {
                put(file, offsetOf(stripe, i));
            }
        }
    }

    private static long offsetOf(final int stripe, final int index) {
        return (long) stripe * STRIPE_BYTES + (long) index * VECTOR_BYTES;
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

    private static long percentile(final long[] samples, final double at) {
        if (samples.length == 0) {
            return 0;
        }
        final long[] sorted = samples.clone();
        Arrays.sort(sorted);
        return sorted[Math.min(sorted.length - 1, (int) (sorted.length * at))];
    }
}

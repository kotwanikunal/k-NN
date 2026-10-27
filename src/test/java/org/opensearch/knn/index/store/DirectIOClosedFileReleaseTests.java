/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import org.opensearch.knn.KNNTestCase;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

/**
 * O3 (the merge note's fix 1): a closed file's cached vectors are freed at the close.
 *
 * <p><b>What this is for.</b> A merged-away segment's entries are never <em>wrong</em> -- the key is the
 * file and the offset, and that file's bytes are immutable -- only cold, so M2 left them to age out through
 * the FIFO. The merge analysis (ref/MERGE-NOTE.md section 4) found what that costs: until the FIFO cycles
 * past them the store sits at its slab target, span fill runs only below the target, so it does not re-arm
 * for the merged segment and that segment refills at one vector per miss. Freeing them at close gives the
 * store room back that <b>nothing was pushed out to get</b>, which is why span fill may use it and why
 * {@code evictions} must not move.
 *
 * <p>These tests drive the store directly. The ownership rule -- only the input {@code open} returned frees
 * anything, never a clone or a slice -- is proved through the real {@link DirectIOVectorIndexInput} in
 * {@code DirectIOVectorCacheTests}; here it is the policy and the accounting.
 */
public class DirectIOClosedFileReleaseTests extends KNNTestCase {

    /** 64 KiB: 128 slots for a 128-dim vector. */
    private static final int TEST_SLAB_BYTES = 64 * 1024;
    /** Where a Lucene {@code .vec}'s vector data starts, so offsets look like the input's. */
    private static final long HEADER = 44L;
    private static final int DIMENSION = 128;
    private static final int VECTOR_BYTES = DIMENSION * Float.BYTES;

    @Override
    public void tearDown() throws Exception {
        DirectIOVectorCache.resetForTesting();
        super.tearDown();
    }

    /** The store on at {@code slabs} test slabs, from empty. */
    private void cacheOf(final int slabs) {
        DirectIOVectorCache.resetForTesting();
        DirectIOVectorCache.setSlabBytesForTesting(TEST_SLAB_BYTES);
        DirectIOVectorCache.publishMaxBytes((long) slabs * TEST_SLAB_BYTES);
    }

    // ---------------------------------------------------------------------------------------------------
    // The accounting: what the close takes out, and what it does not
    // ---------------------------------------------------------------------------------------------------

    /**
     * Two files fill the store to its target; closing one drops exactly that file's entries and bytes, and
     * drops its index table, inside the call. {@code evictions} does not move, because nothing was
     * displaced.
     */
    public void testClosingOneOfTwoFilesFreesExactlyItsShare() {
        cacheOf(2);
        final String kept = "kept.vec#1";
        final String closed = "closed.vec#2";
        final int perFile = fillToTarget(kept, closed);
        assertTrue("both files must have been cached, " + perFile + " each", perFile > 0);

        final long entriesBefore = DirectIOVectorCache.entries();
        final long bytesBefore = DirectIOVectorCache.bytes();
        final long evictionsBefore = DirectIOVectorCache.evictions();
        final long closedEntries = countOf(closed, perFile);
        assertTrue("the closed file must hold something", closedEntries > 0);
        assertEquals("two files, two index tables", 2, DirectIOVectorCache.indexedFiles());

        final long freed = DirectIOVectorCache.releaseFile(closed);

        assertEquals("the close must free exactly the entries that file held", closedEntries, freed);
        assertEquals("entries must drop by that much, inside the call", entriesBefore - closedEntries, DirectIOVectorCache.entries());
        assertEquals("and bytes with them", bytesBefore - closedEntries * VECTOR_BYTES, DirectIOVectorCache.bytes());
        assertEquals("the file's index table must be gone", 1, DirectIOVectorCache.indexedFiles());
        assertEquals("nothing was displaced, so evictions must not move", evictionsBefore, DirectIOVectorCache.evictions());
        assertEquals("one file was released", 1L, DirectIOVectorCache.closedFiles());
        assertEquals(closedEntries, DirectIOVectorCache.closedFileEntries());

        // And the file that stayed is untouched, byte for byte.
        for (int i = 0; i < perFile; i++) {
            final long offset = offsetOf(i);
            if (contains(kept, offset)) {
                assertVector(kept, offset);
            }
        }
        for (int i = 0; i < perFile; i++) {
            assertFalse("the closed file must hold nothing", contains(closed, offsetOf(i)));
        }
    }

    /**
     * The freed slots are handed out before anything is evicted, and span fill -- which never evicts -- can
     * use them, which is the whole point: after a merge the store must be able to warm the new segment.
     */
    public void testTheFreedSlotsAreReusedBeforeAnythingIsEvictedAndSpanFillResumes() {
        cacheOf(2);
        final String kept = "kept.vec#1";
        final String closed = "closed.vec#2";
        final int perFile = fillToTarget(kept, closed);

        assertFalse("the store is at its slab target", DirectIOVectorCache.belowSlabTarget());
        // At the target, span fill admits nothing: that is M4's policy and O3 does not change it.
        final String fresh = "merged.vec#3";
        assertEquals("span fill must be refused at the target", 0, spanFill(fresh, 0, 8));

        final long freed = DirectIOVectorCache.releaseFile(closed);
        assertTrue(freed > 0);
        final long evictionsBefore = DirectIOVectorCache.evictions();

        // Now it may run again, into the slots the closed file vacated, and still with no eviction.
        int admitted = 0;
        int run = 0;
        boolean stopped = false;
        while (run < 4_000) {
            final int took = spanFill(fresh, run, 8);
            if (took == 0) {
                stopped = true;
                break;
            }
            admitted += took;
            run += 8;
        }
        assertTrue("span fill must resume after the close, it admitted " + admitted, admitted > 0);
        assertEquals("and must still not evict anything", evictionsBefore, DirectIOVectorCache.evictions());
        // And it must stop again once the room is gone, or it would be span fill running at the target,
        // which is the scan-resistance question this phase does not decide.
        assertTrue("span fill must stop again once the room is used, it admitted " + admitted, stopped);
        assertEquals("it never evicted on the way", evictionsBefore, DirectIOVectorCache.evictions());
        logger.info(
            "O3: a close freed {} slots and span fill then admitted {} vectors of the merged file with 0 evictions",
            freed,
            admitted
        );

        // The vectors it admitted are the merged file's own bytes.
        for (int i = 0; i < run; i++) {
            final long offset = offsetOf(i);
            if (contains(fresh, offset)) {
                assertVector(fresh, offset);
            }
        }
        // And the file that stayed lost nothing to it.
        assertEquals(evictionsBefore, DirectIOVectorCache.evictions());
    }

    /** A file the store never held costs the close nothing and moves no counter. */
    public void testReleasingAFileTheStoreNeverHeldDoesNothing() {
        cacheOf(1);
        put("real.vec#1", offsetOf(0));
        final long entries = DirectIOVectorCache.entries();

        assertEquals(0L, DirectIOVectorCache.releaseFile("never-cached.vec#9"));
        assertEquals(entries, DirectIOVectorCache.entries());
        assertEquals("no file was released", 0L, DirectIOVectorCache.closedFiles());
        assertEquals(0L, DirectIOVectorCache.closedFileEntries());
    }

    /** With the store off a release is a no-op, and must not take the lock or build a key. */
    public void testReleasingWithTheStoreOffIsANoOp() {
        DirectIOVectorCache.resetForTesting();
        assertFalse(DirectIOVectorCache.isOn());
        assertEquals(0L, DirectIOVectorCache.releaseFile("anything.vec#1"));
        assertEquals(0L, DirectIOVectorCache.closedFiles());
    }

    /** A reopened file fills again and hits again: the release invalidates, it does not poison. */
    public void testAReopenedFileFillsAndHitsAgain() {
        cacheOf(1);
        final String file = "reopened.vec#1";
        for (int i = 0; i < 16; i++) {
            put(file, offsetOf(i));
        }
        assertTrue(contains(file, offsetOf(3)));

        DirectIOVectorCache.releaseFile(file);
        assertFalse("the release must have emptied it", contains(file, offsetOf(3)));
        assertEquals(0, DirectIOVectorCache.indexedFiles());

        // The same file id, because a reopen of the same .vec gives the same id, and its bytes are the same.
        for (int i = 0; i < 16; i++) {
            put(file, offsetOf(i));
        }
        assertTrue("it must fill again", contains(file, offsetOf(3)));
        assertVector(file, offsetOf(3));
        assertEquals(16, DirectIOVectorCache.entries());
    }

    /**
     * A release twice over is the second one finding nothing: a double close must not free a slot the store
     * has since handed to somebody else.
     */
    public void testASecondReleaseOfTheSameFileFreesNothing() {
        cacheOf(1);
        final String closed = "closed.vec#1";
        final String other = "other.vec#2";
        for (int i = 0; i < 16; i++) {
            put(closed, offsetOf(i));
        }
        assertEquals(16L, DirectIOVectorCache.releaseFile(closed));

        // Somebody else takes the freed slots.
        for (int i = 0; i < 16; i++) {
            put(other, offsetOf(i));
        }
        final long entries = DirectIOVectorCache.entries();

        assertEquals("the second release must find nothing", 0L, DirectIOVectorCache.releaseFile(closed));
        assertEquals("and must not have freed anybody else's slots", entries, DirectIOVectorCache.entries());
        for (int i = 0; i < 16; i++) {
            assertTrue(contains(other, offsetOf(i)));
            assertVector(other, offsetOf(i));
        }
    }

    // ---------------------------------------------------------------------------------------------------
    // Concurrency: a close against lookups and fills, with every read byte-exact
    // ---------------------------------------------------------------------------------------------------

    /**
     * Eight threads looking up and filling while files are released under them. Every vector that comes back
     * must be the one its key says, because the store copies under the read lock
     * ({@code DirectIOVectorCache#getFloats}, the copy inside the lock it takes) and a release holds the
     * write lock: a reader either finishes before the release starts or starts after it ends. A wrong vector
     * here would mean a slot was freed while a reader held it.
     */
    public void testConcurrentReleaseAgainstLookupsAndFills() throws Exception {
        cacheOf(4);
        final int threads = 8;
        final int rounds = 400;
        final int files = 6;
        final CountDownLatch start = new CountDownLatch(1);
        final CountDownLatch done = new CountDownLatch(threads);
        final AtomicReference<String> wrong = new AtomicReference<>();
        final AtomicInteger hits = new AtomicInteger();
        final AtomicInteger releases = new AtomicInteger();
        final List<Thread> workers = new ArrayList<>();

        for (int t = 0; t < threads; t++) {
            final int seed = t;
            final Thread worker = new Thread(() -> {
                final float[] into = new float[DIMENSION];
                try {
                    start.await();
                    for (int round = 0; round < rounds; round++) {
                        final String file = "concurrent-" + ((seed + round) % files) + ".vec#1";
                        final long offset = offsetOf((seed * 7 + round * 3) % 64);
                        if ((round & 7) == 0 && seed == 0) {
                            // One thread also closes files, which is what a merge does on a node.
                            DirectIOVectorCache.releaseFile("concurrent-" + (round % files) + ".vec#1");
                            releases.incrementAndGet();
                            continue;
                        }
                        if ((round & 1) == 0) {
                            put(file, offset);
                            continue;
                        }
                        if (DirectIOVectorCache.getFloats(file, offset, VECTOR_BYTES, into, 0)) {
                            hits.incrementAndGet();
                            final float[] expected = vector(file, offset);
                            for (int i = 0; i < DIMENSION; i++) {
                                if (into[i] != expected[i]) {
                                    wrong.compareAndSet(
                                        null,
                                        "hit for " + file + "@" + offset + " component " + i + ": " + into[i] + " not " + expected[i]
                                    );
                                    return;
                                }
                            }
                        }
                    }
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                } finally {
                    done.countDown();
                }
            }, "o3-concurrent-" + t);
            workers.add(worker);
            worker.start();
        }
        start.countDown();
        assertTrue("the workers did not finish", done.await(2, TimeUnit.MINUTES));
        for (final Thread worker : workers) {
            worker.join(TimeUnit.SECONDS.toMillis(30));
        }
        assertNull("a hit served the wrong bytes: " + wrong.get(), wrong.get());
        assertTrue("the run must have released something, it released " + releases.get(), releases.get() > 0);
        assertTrue("and must have served some hits, it served " + hits.get(), hits.get() > 0);
        // The accounting must still add up after all of it.
        assertEquals(DirectIOVectorCache.entries() * VECTOR_BYTES, DirectIOVectorCache.bytes());
        assertTrue("entries must not be negative", DirectIOVectorCache.entries() >= 0);
    }

    // ---------------------------------------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------------------------------------

    /**
     * Fills {@code files} round robin until the store stops being below its slab target, and answers how
     * many vectors of each file were offered. Requested fills, so eviction is allowed and the store ends at
     * its target, which is the state a merge finds.
     */
    private int fillToTarget(final String... files) {
        int i = 0;
        while (DirectIOVectorCache.belowSlabTarget() && i < 10_000) {
            for (final String file : files) {
                put(file, offsetOf(i));
            }
            i++;
        }
        // One more round, so the store is at the target rather than one fill short of it.
        for (final String file : files) {
            put(file, offsetOf(i));
        }
        return i + 1;
    }

    private static long countOf(final String fileId, final int offered) {
        long held = 0;
        for (int i = 0; i < offered; i++) {
            if (contains(fileId, offsetOf(i))) {
                held++;
            }
        }
        return held;
    }

    /** One span fill of {@code count} consecutive vectors starting at ordinal {@code first}. */
    private static int spanFill(final String fileId, final int first, final int count) {
        final ByteBuffer span = ByteBuffer.allocate(count * VECTOR_BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < count; i++) {
            final float[] vector = vector(fileId, offsetOf(first + i));
            for (int component = 0; component < DIMENSION; component++) {
                span.putFloat(i * VECTOR_BYTES + component * Float.BYTES, vector[component]);
            }
        }
        return DirectIOVectorCache.putRun(fileId, offsetOf(first), count, span, 0, VECTOR_BYTES);
    }

    private static boolean contains(final String fileId, final long offset) {
        return DirectIOVectorCache.contains(fileId, offset, VECTOR_BYTES);
    }

    private static void assertVector(final String fileId, final long offset) {
        final float[] into = new float[DIMENSION];
        assertTrue(
            "expected " + fileId + "@" + offset + " to be cached",
            DirectIOVectorCache.getFloats(fileId, offset, VECTOR_BYTES, into, 0)
        );
        assertArrayEquals("the bytes of " + fileId + "@" + offset, vector(fileId, offset), into, 0.0f);
    }

    private static long offsetOf(final int ordinal) {
        return HEADER + (long) ordinal * VECTOR_BYTES;
    }

    private static void put(final String fileId, final long offset) {
        final ByteBuffer staging = ByteBuffer.allocate(VECTOR_BYTES).order(ByteOrder.LITTLE_ENDIAN);
        final float[] vector = vector(fileId, offset);
        for (int i = 0; i < DIMENSION; i++) {
            staging.putFloat(i * Float.BYTES, vector[i]);
        }
        DirectIOVectorCache.put(fileId, offset, staging, 0, VECTOR_BYTES);
    }

    /** A vector whose every float is derived from its key, so a wrong vector cannot look right. */
    private static float[] vector(final String fileId, final long offset) {
        final long seed = fileId.hashCode() * 31L + offset;
        final float[] vector = new float[DIMENSION];
        for (int i = 0; i < DIMENSION; i++) {
            vector[i] = seed + i * 0.5f;
        }
        return vector;
    }
}

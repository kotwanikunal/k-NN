/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import org.opensearch.common.settings.Setting;
import org.opensearch.core.common.unit.ByteSizeUnit;
import org.opensearch.core.common.unit.ByteSizeValue;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.KNNSettings;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Map;
import java.util.Random;

/**
 * Q's C6: the stats the node harness reads keep their names and their meanings, the added keys move in the
 * direction they say they do, the instrumentation costs what it claims, and the segmentation did not cost
 * the index heap.
 *
 * <p>The key <b>set</b> is pinned by {@code KNNStatsTests}, which asserts it literally because it is the
 * API. What is pinned here is the <b>meaning</b> of the six the node harness contract names, which is a
 * different statement and the one that would be broken silently.
 */
public class DirectIOSegmentedStatsTests extends KNNTestCase {

    private static final int TEST_SLAB_BYTES = 64 * 1024;
    private static final int DIMENSION = 32;
    private static final int VECTOR_BYTES = DIMENSION * Float.BYTES;
    private static final int STRIPE_BYTES = 256 * 1024;

    @Override
    public void tearDown() throws Exception {
        DirectIOVectorCache.setLockTimingForTesting(true);
        DirectIOVectorCache.setAcquisitionCountingForTesting(false);
        DirectIOVectorCache.resetForTesting();
        super.tearDown();
    }

    private void cacheOf(final int segments, final int slabs) {
        DirectIOVectorCache.resetForTesting();
        DirectIOVectorCache.setSlabBytesForTesting(TEST_SLAB_BYTES);
        if (segments > 0) {
            DirectIOVectorCache.setSegmentsForTesting(segments);
        }
        DirectIOVectorCache.publishMaxBytes((long) slabs * TEST_SLAB_BYTES);
    }

    // ---------------------------------------------------------------------------------------------------
    // The six the node harness contract names
    // ---------------------------------------------------------------------------------------------------

    /**
     * {@code direct_io_stats.cache.{hits, misses, entries, bytes, evictions, max_bytes}} keep their names
     * <b>and their meanings</b>: one count per vector at the lookup that serves it, the live entry count and
     * its payload bytes, the vectors the store had to displace, and {@code max_bytes} as the <b>explicit
     * cap</b> with {@code 0b} off. A node bench reads these six and a segmented store summing them wrongly
     * would be invisible otherwise.
     */
    public void testTheSixHarnessKeysKeepTheirMeanings() {
        cacheOf(16, 32);
        final String file = "/index/harness.vec#1";
        final int vectors = 2_000;
        for (int i = 0; i < vectors; i++) {
            put(file, offsetOf(i));
        }

        // entries and bytes: the live count, and the slot bytes behind it.
        assertEquals("entries is the live vector count", vectors, DirectIOVectorCache.entries());
        assertEquals("bytes is their slot bytes", (long) vectors * VECTOR_BYTES, DirectIOVectorCache.bytes());
        assertEquals("and they sum the segments", DirectIOVectorCache.entries(), sumOfSegmentEntries());

        // max_bytes: the explicit cap, rounded down to whole slabs, and 0b is off.
        assertEquals("max_bytes is the explicit cap", 32L * TEST_SLAB_BYTES, DirectIOVectorCache.maxBytes());
        assertTrue(DirectIOVectorCache.isOn());

        // hits and misses: one per vector, at the lookup that serves or misses it.
        final long hitsBefore = DirectIOVectorCache.hits();
        final long missesBefore = DirectIOVectorCache.misses();
        final float[] read = new float[DIMENSION];
        for (int i = 0; i < 100; i++) {
            assertTrue(DirectIOVectorCache.getFloats(file, offsetOf(i), VECTOR_BYTES, read, 0));
        }
        assertEquals("one hit per vector served", hitsBefore + 100, DirectIOVectorCache.hits());
        assertEquals("and no miss", missesBefore, DirectIOVectorCache.misses());
        for (int i = 0; i < 50; i++) {
            assertFalse(DirectIOVectorCache.getFloats("/index/cold.vec#1", offsetOf(i), VECTOR_BYTES, read, 0));
        }
        assertEquals("one miss per vector not served", missesBefore + 50, DirectIOVectorCache.misses());
        assertEquals("and no extra hit", hitsBefore + 100, DirectIOVectorCache.hits());
        // contains counts nothing, which is what makes hits / (hits + misses) a hit rate.
        final long hits = DirectIOVectorCache.hits();
        final long misses = DirectIOVectorCache.misses();
        DirectIOVectorCache.contains(file, offsetOf(0), 8 * VECTOR_BYTES);
        DirectIOVectorCache.selectMissing(file, 0L, VECTOR_BYTES, new int[] { 0, 1, 2 }, 3, new int[3]);
        assertEquals("contains counts nothing", hits, DirectIOVectorCache.hits());
        assertEquals("nor does selectMissing", misses, DirectIOVectorCache.misses());

        // evictions: vectors the store had to displace, and nothing else. A close is not an eviction.
        final long evictionsBefore = DirectIOVectorCache.evictions();
        DirectIOVectorCache.releaseFile(file);
        assertEquals("a close does not move evictions", evictionsBefore, DirectIOVectorCache.evictions());
        assertEquals("it moves closed_file_entries instead", vectors, DirectIOVectorCache.closedFileEntries());
        for (int i = 0; i < 32 * DirectIOVectorCache.slotsPerSlab(VECTOR_BYTES) * 2; i++) {
            put(file, offsetOf(i));
        }
        assertTrue("and an eviction does move it", DirectIOVectorCache.evictions() > evictionsBefore);

        // 0b is off, and every one of the six goes with it.
        DirectIOVectorCache.publishMaxBytes(0);
        DirectIOVectorCache.awaitReleasesForTesting();
        assertEquals("max_bytes 0 with the cache off", 0L, DirectIOVectorCache.maxBytes());
        assertEquals(0L, DirectIOVectorCache.entries());
        assertEquals(0L, DirectIOVectorCache.bytes());
        assertFalse(DirectIOVectorCache.isOn());
    }

    /** The added keys read as the segmented store's state, not as something else's. */
    @SuppressWarnings("unchecked")
    public void testTheAddedKeysReadTheSegmentedStore() {
        cacheOf(16, 32);
        final String file = "/index/added.vec#1";
        for (int stripe = 0; stripe < 512; stripe++) {
            for (int i = 0; i < 8; i++) {
                put(file, (long) stripe * STRIPE_BYTES + (long) i * VECTOR_BYTES);
            }
        }
        final Map<String, Object> cache = (Map<String, Object>) ((Map<
            String,
            Map<String, Object>>) new org.opensearch.knn.plugin.stats.KNNStats().getStats().get("direct_io_stats").getValue()).get("cache");

        assertEquals("segments is the count in force", 16, (int) (Integer) cache.get("segments"));
        assertEquals(DirectIOVectorCache.entries(), (long) (Long) cache.get("entries"));
        assertEquals(DirectIOVectorCache.maxSegmentEntries(), (long) (Long) cache.get("max_segment_entries"));
        assertEquals(DirectIOVectorCache.minSegmentEntries(), (long) (Long) cache.get("min_segment_entries"));
        assertTrue("the spread is readable", (long) (Long) cache.get("max_segment_entries") > 0);
        assertTrue(
            "and min is at most max",
            (long) (Long) cache.get("min_segment_entries") <= (long) (Long) cache.get("max_segment_entries")
        );
        assertEquals("the segments sum to entries", DirectIOVectorCache.entries(), sumOfSegmentEntries());
        assertEquals("no lending was needed", 0L, (long) (Long) cache.get("slab_transfers"));
        assertEquals("nothing was closed", 0L, (long) (Long) cache.get("longest_close_hold_nanos"));
        DirectIOVectorCache.releaseFile(file);
        assertTrue("and a close populates the hold", DirectIOVectorCache.longestCloseHoldNanos() > 0);
        assertTrue(DirectIOVectorCache.longestCloseHoldEntries() > 0);
    }

    // ---------------------------------------------------------------------------------------------------
    // What the instrumentation costs
    // ---------------------------------------------------------------------------------------------------

    /**
     * <b>The lock instrumentation costs nothing on an uncontended acquisition</b>, which is what makes it
     * shippable. The design run proposed sampling one acquisition in 1,024 because its instrumented build
     * timed every one and cost 26%; this times only the contended ones, so the comparison the BACKLOG asks
     * for -- the build with the timing on against the same build with it off -- is a comparison of a path
     * that takes one {@code hasQueuedThreads()} volatile read either way.
     *
     * <p>One thread, so no acquisition is contended at all, which is the case the bar is about.
     */
    public void testTheLockTimingCostsNothingUncontended() {
        cacheOf(16, 64);
        final String file = "/index/cost.vec#1";
        final int keys = 20_000;
        for (int i = 0; i < keys; i++) {
            put(file, offsetOf(i));
        }
        final float[] read = new float[DIMENSION];
        // Warm, then alternate so that drift cannot favour either arm.
        for (int rep = 0; rep < 2; rep++) {
            lookupBurst(file, keys, read, 200_000);
        }
        // Seven alternating repetitions and the MINIMUM of each arm, not the sum. The minimum is the
        // pause-free run by construction, which is the statistic for "what does this code path cost"; a sum
        // or a mean prices whatever else the JVM was doing, and MEASURED that is the difference between
        // 0.12% and 5.11% depending on whether this test ran alone or with the rest of the store suite.
        long withTiming = Long.MAX_VALUE;
        long withoutTiming = Long.MAX_VALUE;
        for (int rep = 0; rep < 7; rep++) {
            DirectIOVectorCache.setLockTimingForTesting(true);
            withTiming = Math.min(withTiming, lookupBurst(file, keys, read, 200_000));
            DirectIOVectorCache.setLockTimingForTesting(false);
            withoutTiming = Math.min(withoutTiming, lookupBurst(file, keys, read, 200_000));
        }
        DirectIOVectorCache.setLockTimingForTesting(true);
        assertEquals("no acquisition was contended, so nothing was timed", 0L, DirectIOVectorCache.readLockWaitNanos());
        assertEquals(0L, DirectIOVectorCache.contendedReadAcquisitions());
        final double ratio = (double) withTiming / withoutTiming;
        logger.info(
            "200,000 uncontended lookups, best of seven: {} us with the timing on, {} us with it off, a factor of {}",
            withTiming / 1_000,
            withoutTiming / 1_000,
            String.format("%.4f", ratio)
        );
        assertTrue(
            "the timing must cost under 2% of an uncontended lookup burst, saw " + String.format("%.2f%%", 100 * (ratio - 1)),
            ratio < 1.02
        );
    }

    private long lookupBurst(final String file, final int keys, final float[] read, final int lookups) {
        final Random random = new Random(5L);
        final long started = System.nanoTime();
        for (int i = 0; i < lookups; i++) {
            DirectIOVectorCache.getFloats(file, offsetOf(random.nextInt(keys)), VECTOR_BYTES, read, 0);
        }
        return System.nanoTime() - started;
    }

    // ---------------------------------------------------------------------------------------------------
    // What the segmentation costs the index heap
    // ---------------------------------------------------------------------------------------------------

    /**
     * <b>Index heap per entry, within 15% at 128 segments against one.</b> The risk is
     * {@code DirectIOSlotMap#capacityFor}'s rounding to a power of two: {@code N} maps of {@code E/N}
     * entries can cost up to twice one map of {@code E} entries in the worst placement, though the expected
     * cost is identical because the load factor is the same. M2's unsegmented figure is 53.0 to 67.8 B per
     * entry.
     */
    public void testIndexBytesPerEntryIsUnchangedBySegmenting() {
        final double atOne = indexBytesPerEntry(1);
        final double atMany = indexBytesPerEntry(128);
        logger.info(
            "index heap per entry: {} B at one segment, {} B at 128, a factor of {}",
            String.format("%.1f", atOne),
            String.format("%.1f", atMany),
            String.format("%.3f", atMany / atOne)
        );
        assertTrue("the unsegmented figure must be in M2's range, saw " + atOne, atOne > 20 && atOne < 200);
        assertTrue(
            "128 segments must cost within 15% per entry: "
                + String.format("%.1f", atOne)
                + " B against "
                + String.format("%.1f", atMany)
                + " B",
            atMany <= 1.15 * atOne
        );
    }

    private double indexBytesPerEntry(final int segments) {
        cacheOf(segments, 1_024);
        final String file = "/index/heap.vec#1";
        // 64 vectors in each of 4,096 stripes: 262,144 entries over every segment, at half the slot
        // capacity so nothing is evicted and the maps are at their natural load.
        for (int stripe = 0; stripe < 4_096; stripe++) {
            for (int i = 0; i < 64; i++) {
                put(file, (long) stripe * STRIPE_BYTES + (long) i * VECTOR_BYTES);
            }
        }
        final long entries = DirectIOVectorCache.entries();
        assertEquals("the fixture must fit whole at " + segments + " segments", 4_096L * 64, entries);
        return (double) DirectIOVectorCache.indexBytes() / entries;
    }

    // ---------------------------------------------------------------------------------------------------
    // Nothing removed, nothing changed
    // ---------------------------------------------------------------------------------------------------

    /**
     * The {@code knn.direct_io.*} surface is exactly the six keys an operator has, each still with the
     * default it had: the nine F-03 removed are constants at the same values, which
     * {@code KNNDirectIOConstantsTests} pins against the baseline dump, and nothing may change a default.
     */
    public void testTheSettingSurfaceIsTheSixKeptKeysAtTheirDefaults() {
        final Setting<?>[] kept = {
            KNNSettings.KNN_DIRECT_IO_RESCORE_ENABLED_SETTING,
            KNNSettings.KNN_DIRECT_IO_MIN_FILE_SIZE_SETTING,
            KNNSettings.KNN_DIRECT_IO_RESCORE_PREFETCH_THREADS_SETTING,
            KNNSettings.KNN_DIRECT_IO_RESCORE_CACHE_MAX_BYTES_SETTING,
            KNNSettings.KNN_DIRECT_IO_RESCORE_READ_PROFILE_SETTING,
            KNNSettings.KNN_DIRECT_IO_READAHEAD_WINDOW_SETTING };
        for (final Setting<?> setting : kept) {
            assertNotNull("a kept setting must still exist", setting);
            assertNotNull(
                setting.getKey() + " must still have a default",
                setting.getDefaultRaw(org.opensearch.common.settings.Settings.EMPTY)
            );
        }
        assertEquals(
            "and the surface is those six and nothing else",
            6L,
            KNNSettings.state().getSettings().stream().filter(s -> s.getKey().startsWith("knn.direct_io.")).count()
        );
        // The three the cache itself owns, by value, because those are the ones Q touches the code of.
        assertEquals(
            "knn.direct_io.rescore.cache.max_bytes still defaults to 0b, so the cache is still off by default",
            new ByteSizeValue(0, ByteSizeUnit.BYTES),
            KNNSettings.KNN_DIRECT_IO_RESCORE_CACHE_MAX_BYTES_DEFAULT_VALUE
        );
        assertTrue("span fill is still on", DirectIOVectorCache.SPAN_FILL);
        assertEquals("read-around is still 64kb", 64L * 1024, DirectIOVectorCache.READ_AROUND_BYTES);
        // And the store is off at its default, which is what "off is off" rests on.
        DirectIOVectorCache.resetForTesting();
        assertFalse("the store is off before any configure", DirectIOVectorCache.isOn());
        assertEquals("and reports one segment", 1, DirectIOVectorCache.segmentCount());
    }

    // ---------------------------------------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------------------------------------

    private static long sumOfSegmentEntries() {
        long total = 0;
        for (final DirectIOCacheSegment segment : DirectIOVectorCache.segmentsSnapshot()) {
            total += segment.liveEntries();
        }
        return total;
    }

    private static long offsetOf(final int ordinal) {
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

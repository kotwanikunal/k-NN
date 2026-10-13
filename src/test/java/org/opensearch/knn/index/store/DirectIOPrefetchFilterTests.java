/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import com.carrotsearch.randomizedtesting.ThreadFilter;
import com.carrotsearch.randomizedtesting.annotations.ThreadLeakFilters;
import lombok.SneakyThrows;
import org.opensearch.common.settings.ClusterSettings;
import org.opensearch.common.settings.Setting;
import org.opensearch.common.settings.Settings;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.codec.scorer.PrefetchHelper;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

import static org.mockito.Mockito.when;

/**
 * N1: the prefetch helper drops the candidates the store already holds before it groups them.
 *
 * <p>Why there is anything to test. {@code PrefetchHelper} coalesces a batch's ordinals into spans of up to
 * 128 KB, gaps included, and the input can only accept or decline a whole span, so one missing vector in a
 * span makes the input stage the whole of it. MEASURED on the EBS node (ref/NODE-16M.md): 73.4 staged reads
 * per query for 77.1 missing vectors at 22.9 KB a read, against 62.7 for 76.9 at 12.6 KB for phase 16-I,
 * whose cache sat in the scorer and grouped only its misses. Asking the store first is what makes the spans
 * the same spans 16-I built.
 *
 * <p>What these tests hold the change to:
 * <ul>
 *   <li>the filter drops <b>exactly</b> the ordinals the store holds whole, at the length being read, and
 *       counts no hit and no miss while doing it;</li>
 *   <li>the spans built from what is left are the spans today's code builds for that set, which is checked
 *       by running the same batch through the same production path with the store off;</li>
 *   <li>the caller's array comes back the way it comes back today, because the delegate scores it;</li>
 *   <li>a single remaining miss is still prefetched, so it is still admitted (the store admits only staged
 *       reads, M3), and nothing goes blocking;</li>
 *   <li>an evicted vector is read correctly, which is the whole safety argument: the filter serves no bytes.</li>
 * </ul>
 */
@ThreadLeakFilters(defaultFilters = true, filters = { DirectIOPrefetchFilterTests.ReadPoolThreadFilter.class })
public class DirectIOPrefetchFilterTests extends KNNTestCase {

    /** The read pool is never shut down by design; its daemon threads outlive any one suite. */
    public static final class ReadPoolThreadFilter implements ThreadFilter {
        @Override
        public boolean reject(final Thread thread) {
            return thread.getName().startsWith(DirectIOReadPool.THREAD_NAME_PREFIX);
        }
    }

    private static final int DIMENSION = 128;
    private static final int VECTOR_BYTES = DIMENSION * Float.BYTES;
    /** Not a multiple of the vector length and not a multiple of a block, which is the awkward real case. */
    private static final int HEADER_LENGTH = 44;
    private static final int FOOTER_LENGTH = 16;
    private static final int VECTOR_COUNT = 2_048;
    private static final int TEST_SLAB_BYTES = 256 * 1024;

    private float[][] vectors;
    private Path path;

    @Override
    public void tearDown() throws Exception {
        DirectIOVectorCache.resetForTesting();
        DirectIOReadPool.resetForTesting();
        DirectIOVectorIndexInput.STATS.reset();
        super.tearDown();
    }

    // -----------------------------------------------------------------------------------------------
    // What the filter answers
    // -----------------------------------------------------------------------------------------------

    /** Exactly the cached ordinals are dropped, in order, and the lookup counts nothing. */
    @SneakyThrows
    public void testTheFilterDropsExactlyTheCachedOrdinals() {
        assumeDirectIOWorksHere();
        writeFixture(1L);
        configure(8, false);

        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(input);
            final int[] cached = { 7, 8, 40, 41, 42, 900 };
            warm(input, cached);

            final int[] batch = { 7, 8, 9, 40, 41, 42, 43, 900, 901 };
            final int[] expected = { 9, 43, 901 };

            final long hitsBefore = DirectIOVectorCache.hits();
            final long missesBefore = DirectIOVectorCache.misses();
            final int[] uncached = new int[batch.length];
            final int remaining = input.selectUncachedOrdinals(HEADER_LENGTH, VECTOR_BYTES, batch, batch.length, uncached);

            assertEquals(expected.length, remaining);
            assertArrayEquals(expected, Arrays.copyOf(uncached, remaining));
            assertEquals("the filter counts no hit", hitsBefore, DirectIOVectorCache.hits());
            assertEquals("and no miss", missesBefore, DirectIOVectorCache.misses());
        }
    }

    /** A vector the store does not hold <em>whole</em>, at the length being asked about, counts as missing. */
    @SneakyThrows
    public void testAVectorHeldAtAnotherLengthCountsAsMissing() {
        assumeDirectIOWorksHere();
        writeFixture(2L);
        configure(8, false);

        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(input);
            // Half a vector, admitted by the same fill rule at half the length and at the same offset.
            final int half = DIMENSION / 2;
            input.prefetch(offsetOf(300), (long) half * Float.BYTES);
            input.seek(offsetOf(300));
            input.readFloats(new float[half], 0, half);
            assertEquals(1L, DirectIOVectorCache.entries());
            assertTrue(DirectIOVectorCache.contains(fileId(), offsetOf(300), half * Float.BYTES));

            final int[] batch = { 300, 301 };
            final int[] uncached = new int[batch.length];
            final int remaining = input.selectUncachedOrdinals(HEADER_LENGTH, VECTOR_BYTES, batch, batch.length, uncached);
            assertEquals("neither is held as a whole 512 byte vector", 2, remaining);
            assertArrayEquals(new int[] { 300, 301 }, Arrays.copyOf(uncached, remaining));
        }
    }

    /** Nothing of the file cached, the store off, and a closed input all decline to filter at all. */
    @SneakyThrows
    public void testTheFilterDeclinesWhenItCanSayNothingUseful() {
        assumeDirectIOWorksHere();
        writeFixture(3L);
        configure(8, false);
        final int[] batch = { 5, 6, 7 };
        final int[] uncached = new int[batch.length];

        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(input);
            assertEquals(
                "a file the store holds nothing of takes the unfiltered path",
                -1,
                input.selectUncachedOrdinals(HEADER_LENGTH, VECTOR_BYTES, batch, batch.length, uncached)
            );
            warm(input, new int[] { 5 });
            assertEquals(2, input.selectUncachedOrdinals(HEADER_LENGTH, VECTOR_BYTES, batch, batch.length, uncached));

            configure(0, false);
            assertEquals(
                "the store at 0b takes the unfiltered path",
                -1,
                input.selectUncachedOrdinals(HEADER_LENGTH, VECTOR_BYTES, batch, batch.length, uncached)
            );
        }
        configure(8, false);
        final DirectIOVectorIndexInput closed = DirectIOVectorIndexInput.open(path);
        assertNotNull(closed);
        closed.close();
        assertEquals(-1, closed.selectUncachedOrdinals(HEADER_LENGTH, VECTOR_BYTES, batch, batch.length, uncached));
    }

    /** The slice Lucene hands the scorer starts at the field's offset, and the filter has to add it. */
    @SneakyThrows
    public void testTheFilterWorksThroughASlice() {
        assumeDirectIOWorksHere();
        writeFixture(4L);
        configure(8, false);

        try (DirectIOVectorIndexInput root = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(root);
            try (DirectIOVectorIndexInput slice = root.slice("field", HEADER_LENGTH, (long) VECTOR_COUNT * VECTOR_BYTES)) {
                final float[] scratch = new float[DIMENSION];
                for (final int ordinal : new int[] { 11, 12 }) {
                    slice.prefetch((long) ordinal * VECTOR_BYTES, VECTOR_BYTES);
                    slice.seek((long) ordinal * VECTOR_BYTES);
                    slice.readFloats(scratch, 0, DIMENSION);
                    assertArrayEquals(vectors[ordinal], scratch, 0.0f);
                }
                final int[] batch = { 11, 12, 13 };
                final int[] uncached = new int[batch.length];
                // Base offset 0, which is what PrefetchableVectorValuesHelper passes for a slice.
                final int remaining = slice.selectUncachedOrdinals(0, VECTOR_BYTES, batch, batch.length, uncached);
                assertEquals(1, remaining);
                assertEquals(13, uncached[0]);
            }
        }
    }

    // -----------------------------------------------------------------------------------------------
    // What the helper then prefetches
    // -----------------------------------------------------------------------------------------------

    /**
     * The spans built from what is left are today's spans for that set. Both sides are production code: the
     * filtered run is the full batch through a store holding most of it, the reference run is the miss list
     * through the same path with the store off, which is what this code did before N1 when handed that set.
     */
    @SneakyThrows
    public void testTheSpansAreTodaysGroupingOfTheRemainder() {
        assumeDirectIOWorksHere();
        writeFixture(5L);

        final int[] batch = batchOf(20260505L, 64);
        final int[] cached = Arrays.copyOfRange(batch, 0, 48);
        final int[] misses = new int[batch.length];

        // The filtered run.
        configure(16, false);
        final Burst filtered;
        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(input);
            warm(input, cached);
            final int remaining = input.selectUncachedOrdinals(HEADER_LENGTH, VECTOR_BYTES, batch, batch.length, misses);
            assertEquals(batch.length - cached.length, remaining);
            DirectIOVectorIndexInput.STATS.reset();
            PrefetchHelper.prefetch(input, HEADER_LENGTH, VECTOR_BYTES, batch.clone(), batch.length);
            readAll(input, batch);
            filtered = burst();
        }

        // The reference run: the same misses, the store off, so nothing is filtered.
        final int[] reference = Arrays.copyOf(misses, batch.length - cached.length);
        configure(0, false);
        final Burst unfiltered;
        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(input);
            DirectIOVectorIndexInput.STATS.reset();
            PrefetchHelper.prefetch(input, HEADER_LENGTH, VECTOR_BYTES, reference.clone(), reference.length);
            readAll(input, reference);
            unfiltered = burst();
        }

        assertTrue("the batch coalesced into more than one span, or this proves nothing", filtered.prefetchCalls() > 1);
        assertEquals("same number of spans", unfiltered.prefetchCalls(), filtered.prefetchCalls());
        assertEquals("same staged reads", unfiltered.stagedReads(), filtered.stagedReads());
        assertEquals("same device bytes", unfiltered.stagedBytes(), filtered.stagedBytes());
        assertEquals("and nothing went blocking", 0L, filtered.blockingReads());
    }

    /**
     * The store holding everything costs the device nothing at all: no span is asked for, so there is no
     * group for the input to decline and no read to wait for.
     */
    @SneakyThrows
    public void testAFullyCachedBatchPrefetchesNothing() {
        assumeDirectIOWorksHere();
        writeFixture(6L);
        configure(16, false);

        final int[] batch = batchOf(20260506L, 32);
        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(input);
            warm(input, batch);
            DirectIOVectorIndexInput.STATS.reset();
            PrefetchHelper.prefetch(input, HEADER_LENGTH, VECTOR_BYTES, batch.clone(), batch.length);
            assertEquals("no span was asked for", 0L, DirectIOVectorIndexInput.STATS.prefetchCalls());
            readAll(input, batch);
            assertEquals(0L, burst().stagedReads());
            assertEquals(0L, burst().blockingReads());
        }
    }

    /**
     * One remaining miss is still staged, and so still admitted. This is why the {@code numOrds <= 1} early
     * return stays on the batch as the caller gives it: the store admits only vectors a staged read covered
     * (M3), so turning the common single-miss batch into a blocking read would leave it a miss forever.
     */
    @SneakyThrows
    public void testASingleRemainingMissIsStagedAndAdmitted() {
        assumeDirectIOWorksHere();
        writeFixture(7L);
        configure(16, false);

        final int[] batch = batchOf(20260507L, 24);
        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(input);
            warm(input, Arrays.copyOf(batch, batch.length - 1));
            final int only = batch[batch.length - 1];
            assertFalse(DirectIOVectorCache.contains(fileId(), offsetOf(only), VECTOR_BYTES));

            DirectIOVectorIndexInput.STATS.reset();
            PrefetchHelper.prefetch(input, HEADER_LENGTH, VECTOR_BYTES, batch.clone(), batch.length);
            assertEquals("exactly one span, for the one miss", 1L, DirectIOVectorIndexInput.STATS.prefetchCalls());
            readAll(input, batch);

            assertEquals("the miss was served by a staged read", 1L, burst().stagedReads());
            assertEquals("not a blocking one", 0L, burst().blockingReads());
            assertTrue("and so it was admitted", DirectIOVectorCache.contains(fileId(), offsetOf(only), VECTOR_BYTES));
        }
    }

    /** With the store off, the Direct I/O input gets the spans the batch asks for, exactly as before N1. */
    @SneakyThrows
    public void testTheStoreOffGetsTodaysPrefetchCalls() {
        assumeDirectIOWorksHere();
        writeFixture(8L);

        final int[] batch = batchOf(20260508L, 64);
        configure(0, false);
        final long withStoreOff;
        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(input);
            DirectIOVectorIndexInput.STATS.reset();
            PrefetchHelper.prefetch(input, HEADER_LENGTH, VECTOR_BYTES, batch.clone(), batch.length);
            withStoreOff = DirectIOVectorIndexInput.STATS.prefetchCalls();
        }

        // The same batch through an input that is not a Direct I/O input at all: the grouping is the only
        // thing both have in common, so the call count is the grouping's own answer for this batch.
        final CountingInput counting = new CountingInput();
        PrefetchHelper.prefetch(counting, HEADER_LENGTH, VECTOR_BYTES, batch.clone(), batch.length);
        assertTrue(counting.ranges.size() > 1);
        assertEquals(counting.ranges.size(), (int) withStoreOff);
    }

    /** The caller's array comes back sorted ascending with the same members, whatever the store holds. */
    @SneakyThrows
    public void testTheCallersArrayIsLeftAsItIsToday() {
        assumeDirectIOWorksHere();
        writeFixture(9L);
        configure(16, false);

        final int[] batch = batchOf(20260509L, 40);
        final int[] shuffled = batch.clone();
        shuffle(shuffled, 7L);
        final int[] expected = shuffled.clone();
        Arrays.sort(expected);

        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(input);
            warm(input, Arrays.copyOf(batch, 30));
            final int[] given = shuffled.clone();
            PrefetchHelper.prefetch(input, HEADER_LENGTH, VECTOR_BYTES, given, given.length);
            assertArrayEquals("sorted in place, nothing dropped and nothing duplicated", expected, given);
        }
    }

    // -----------------------------------------------------------------------------------------------
    // Safety: the filter serves no bytes
    // -----------------------------------------------------------------------------------------------

    /**
     * A vector evicted between the filter and the read is read correctly. The filter decides what to
     * prefetch and nothing else, so losing an entry after it has answered costs one blocking read of the
     * same bytes and cannot change a score.
     */
    @SneakyThrows
    public void testAVectorEvictedAfterTheFilterIsStillReadCorrectly() {
        assumeDirectIOWorksHere();
        writeFixture(10L);
        // One slab, so filling it past its slot count reclaims it while the batch is in flight.
        configure(1, false);
        final int slotsPerSlab = TEST_SLAB_BYTES / VECTOR_BYTES;

        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(input);
            final int[] batch = new int[16];
            for (int i = 0; i < batch.length; i++) {
                batch[i] = i;
            }
            warm(input, batch);
            final int[] uncached = new int[batch.length];
            assertEquals(
                "every candidate is cached, so the filter would drop them all",
                0,
                input.selectUncachedOrdinals(HEADER_LENGTH, VECTOR_BYTES, batch, batch.length, uncached)
            );

            // And now they are gone, between the filter and the read.
            for (int ordinal = 100; ordinal < 100 + 3 * slotsPerSlab; ordinal++) {
                readVector(input, offsetOf(ordinal), new float[DIMENSION]);
            }
            assertTrue("the slab was reclaimed", DirectIOVectorCache.reclaimedSlabs() > 0);
            assertFalse(DirectIOVectorCache.contains(fileId(), offsetOf(batch[0]), VECTOR_BYTES));

            final float[] scratch = new float[DIMENSION];
            for (final int ordinal : batch) {
                input.seek(offsetOf(ordinal));
                input.readFloats(scratch, 0, DIMENSION);
                assertArrayEquals("vector " + ordinal + " after eviction", vectors[ordinal], scratch, 0.0f);
            }
            assertTrue("read without a prefetch, so blocking", DirectIOVectorIndexInput.STATS.blockingReads() > 0);
            DirectIOVectorCache.checkInvariantsForTesting();
        }
    }

    /** Eight threads filtering, filling and evicting the same store: no wrong vector, no broken invariant. */
    @SneakyThrows
    public void testConcurrentFilterFillAndEvict() {
        assumeDirectIOWorksHere();
        writeFixture(11L);
        // Two slabs against 2,048 vectors of 512 bytes, so eviction runs throughout.
        configure(2, true);

        final int threads = 8;
        final CountDownLatch start = new CountDownLatch(1);
        final CountDownLatch done = new CountDownLatch(threads);
        final AtomicReference<Throwable> failure = new AtomicReference<>();
        try (DirectIOVectorIndexInput root = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(root);
            final List<DirectIOVectorIndexInput> slices = new ArrayList<>();
            for (int t = 0; t < threads; t++) {
                final DirectIOVectorIndexInput slice = root.slice("field-" + t, HEADER_LENGTH, (long) VECTOR_COUNT * VECTOR_BYTES);
                slices.add(slice);
                final int seed = t;
                final Thread thread = new Thread(() -> {
                    try {
                        final Random random = new Random(900L + seed);
                        final float[] scratch = new float[DIMENSION];
                        final int[] batch = new int[32];
                        final int[] uncached = new int[batch.length];
                        start.await();
                        for (int round = 0; round < 200; round++) {
                            for (int i = 0; i < batch.length; i++) {
                                batch[i] = random.nextInt(VECTOR_COUNT);
                            }
                            PrefetchHelper.prefetch(slice, 0, VECTOR_BYTES, batch, batch.length);
                            for (final int ordinal : batch) {
                                slice.seek((long) ordinal * VECTOR_BYTES);
                                slice.readFloats(scratch, 0, DIMENSION);
                                assertArrayEquals("vector " + ordinal, vectors[ordinal], scratch, 0.0f);
                            }
                            slice.selectUncachedOrdinals(0, VECTOR_BYTES, batch, batch.length, uncached);
                        }
                    } catch (Throwable e) {
                        failure.compareAndSet(null, e);
                    } finally {
                        done.countDown();
                    }
                }, "n1-filter-" + t);
                thread.setDaemon(true);
                thread.start();
            }
            start.countDown();
            assertTrue("the threads finished", done.await(3, TimeUnit.MINUTES));
            for (final DirectIOVectorIndexInput slice : slices) {
                slice.close();
            }
        }
        if (failure.get() != null) {
            throw new AssertionError(failure.get());
        }
        assertTrue("the store was evicting", DirectIOVectorCache.evictions() > 0 || DirectIOVectorCache.reclaimedSlabs() > 0);
        DirectIOVectorCache.checkInvariantsForTesting();
    }

    /**
     * A batch of one is still not prefetched, store or no store: that bound is on the batch the caller gives,
     * because a single-vector read is the caller's own and the grouping has nothing to coalesce.
     */
    @SneakyThrows
    public void testABatchOfOneIsStillNotPrefetched() {
        assumeDirectIOWorksHere();
        writeFixture(12L);
        configure(8, false);

        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(input);
            DirectIOVectorIndexInput.STATS.reset();
            PrefetchHelper.prefetch(input, HEADER_LENGTH, VECTOR_BYTES, new int[] { 44 }, 1);
            assertEquals(0L, DirectIOVectorIndexInput.STATS.prefetchCalls());
        }
    }

    /**
     * A batch whose every candidate is cached makes no prefetch call, so the burst boundary the settings
     * refresh hangs off had to move to the filter: without that, an operator's {@code max_bytes} change
     * would never be read on a node whose cache serves everything.
     */
    @SneakyThrows
    public void testASettingChangeIsPickedUpThroughTheFilterAlone() {
        assumeDirectIOWorksHere();
        writeFixture(13L);
        configure(8, false);

        final int[] batch = batchOf(20260513L, 16);
        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(input);
            warm(input, batch);
            assertTrue(DirectIOVectorCache.slabs() > 0);

            // The operator turns the cache off. Nothing but a fully cached batch happens afterwards.
            withNodeSettings(
                Settings.builder()
                    .put(KNNSettings.KNN_DIRECT_IO_RESCORE_CACHE_MAX_BYTES, "0b")
                    .put(KNNSettings.KNN_DIRECT_IO_READAHEAD_WINDOW, 0)
                    .build()
            );
            PrefetchHelper.prefetch(input, HEADER_LENGTH, VECTOR_BYTES, batch.clone(), batch.length);
            assertEquals("the store released its slabs at the next burst", 0L, DirectIOVectorCache.slabs());
            assertEquals(0L, DirectIOVectorCache.entries());
        }
    }

    // -----------------------------------------------------------------------------------------------
    // Fixture and helpers
    // -----------------------------------------------------------------------------------------------

    /** A snapshot of the counters a burst moves, so two runs can be compared field by field. */
    private record Burst(long prefetchCalls, long stagedReads, long stagedBytes, long blockingReads) {
    }

    private static Burst burst() {
        final DirectIOVectorIndexInput.Stats stats = DirectIOVectorIndexInput.STATS;
        return new Burst(stats.prefetchCalls(), stats.stagedReads(), stats.stagedBytes(), stats.blockingReads());
    }

    /** An input that is not a Direct I/O input, recording the ranges the grouping asks for. */
    private static final class CountingInput extends org.apache.lucene.store.IndexInput {
        private final List<long[]> ranges = new ArrayList<>();

        private CountingInput() {
            super("counting");
        }

        @Override
        public void prefetch(final long offset, final long length) {
            ranges.add(new long[] { offset, length });
        }

        @Override
        public void close() {}

        @Override
        public long getFilePointer() {
            return 0;
        }

        @Override
        public void seek(final long pos) {}

        @Override
        public long length() {
            return Long.MAX_VALUE;
        }

        @Override
        public org.apache.lucene.store.IndexInput slice(final String sliceDescription, final long offset, final long length) {
            return this;
        }

        @Override
        public byte readByte() {
            return 0;
        }

        @Override
        public void readBytes(final byte[] b, final int offset, final int len) {}
    }

    /** Distinct ordinals, which is what a candidate batch is. */
    private static int[] batchOf(final long seed, final int size) {
        final Random random = new Random(seed);
        final Set<Integer> chosen = new HashSet<>();
        while (chosen.size() < size) {
            chosen.add(random.nextInt(VECTOR_COUNT));
        }
        final int[] batch = chosen.stream().mapToInt(Integer::intValue).toArray();
        Arrays.sort(batch);
        return batch;
    }

    private static void shuffle(final int[] values, final long seed) {
        final Random random = new Random(seed);
        for (int i = values.length - 1; i > 0; i--) {
            final int j = random.nextInt(i + 1);
            final int swap = values[i];
            values[i] = values[j];
            values[j] = swap;
        }
    }

    /** Puts these ordinals in the store the way the read path does: declare one, consume one. */
    @SneakyThrows
    private void warm(final DirectIOVectorIndexInput input, final int[] ordinals) {
        final float[] scratch = new float[DIMENSION];
        for (final int ordinal : ordinals) {
            readVector(input, offsetOf(ordinal), scratch);
            assertArrayEquals(vectors[ordinal], scratch, 0.0f);
        }
    }

    /** Consumes a batch in file order, which is the order the scorer reads a sorted batch in. */
    @SneakyThrows
    private void readAll(final DirectIOVectorIndexInput input, final int[] ordinals) {
        final int[] sorted = ordinals.clone();
        Arrays.sort(sorted);
        final float[] scratch = new float[DIMENSION];
        for (final int ordinal : sorted) {
            input.seek(offsetOf(ordinal));
            input.readFloats(scratch, 0, DIMENSION);
            assertArrayEquals(vectors[ordinal], scratch, 0.0f);
        }
    }

    private static long offsetOf(final int ordinal) {
        return HEADER_LENGTH + (long) ordinal * VECTOR_BYTES;
    }

    @SneakyThrows
    private String fileId() {
        return DirectIOVectorCache.fileId(path, Files.size(path));
    }

    private void assumeDirectIOWorksHere() throws Exception {
        assumeTrue("this JDK has no ExtendedOpenOption.DIRECT", DirectIOVectorIndexInput.directOpenOption() != null);
        final Path probe = createTempDir().resolve("direct-io-probe");
        Files.write(probe, new byte[8192]);
        try (FileChannel channel = FileChannel.open(probe, StandardOpenOption.READ, DirectIOVectorIndexInput.directOpenOption())) {
            channel.read(ByteBuffer.allocateDirect(8192).alignedSlice(4096));
        } catch (Exception e) {
            assumeNoException("this filesystem refuses O_DIRECT", e);
        }
    }

    private void withNodeSettings(final Settings settings) {
        final Set<Setting<?>> registered = new HashSet<>(ClusterSettings.BUILT_IN_CLUSTER_SETTINGS);
        registered.addAll(
            KNNSettings.state()
                .getSettings()
                .stream()
                .filter(setting -> setting.getProperties().contains(Setting.Property.NodeScope))
                .toList()
        );
        when(clusterService.getClusterSettings()).thenReturn(new ClusterSettings(settings, registered));
        KNNSettings.state().setClusterService(clusterService);
    }

    private void configure(final int slabs, final boolean spanFill) {
        final long maxBytes = (long) slabs * TEST_SLAB_BYTES;
        withNodeSettings(
            Settings.builder()
                .put(KNNSettings.KNN_DIRECT_IO_RESCORE_CACHE_MAX_BYTES, maxBytes + "b")
                .put(KNNSettings.KNN_DIRECT_IO_RESCORE_CACHE_SPAN_FILL, spanFill)
                .put(KNNSettings.KNN_DIRECT_IO_READAHEAD_WINDOW, 0)
                .build()
        );
        DirectIOVectorCache.resetForTesting();
        DirectIOVectorCache.setSlabBytesForTesting(TEST_SLAB_BYTES);
        DirectIOVectorCache.configure(maxBytes);
        DirectIOVectorIndexInput.STATS.reset();
    }

    @SneakyThrows
    private void writeFixture(final long seed) {
        final Random random = new Random(seed);
        vectors = new float[VECTOR_COUNT][DIMENSION];
        for (final float[] vector : vectors) {
            for (int i = 0; i < DIMENSION; i++) {
                vector[i] = (random.nextFloat() - 0.5f) * 8.0f;
            }
        }
        path = createTempDir().resolve("filter-" + seed + ".vec");
        final ByteBuffer buffer = ByteBuffer.allocate(HEADER_LENGTH + VECTOR_COUNT * VECTOR_BYTES + FOOTER_LENGTH)
            .order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < HEADER_LENGTH; i++) {
            buffer.put((byte) (0xA0 + (i & 0x0F)));
        }
        for (final float[] vector : vectors) {
            for (final float value : vector) {
                buffer.putFloat(value);
            }
        }
        for (int i = 0; i < FOOTER_LENGTH; i++) {
            buffer.put((byte) (0xF0 + (i & 0x0F)));
        }
        Files.write(path, buffer.array());
    }

    /** One vector through the real input: declare it, consume it. */
    @SneakyThrows
    private void readVector(final DirectIOVectorIndexInput input, final long at, final float[] into) {
        input.prefetch(at, VECTOR_BYTES);
        input.seek(at);
        input.readFloats(into, 0, DIMENSION);
    }
}

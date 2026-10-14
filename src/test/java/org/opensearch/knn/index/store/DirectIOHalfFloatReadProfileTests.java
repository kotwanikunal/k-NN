/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import com.carrotsearch.randomizedtesting.ThreadFilter;
import com.carrotsearch.randomizedtesting.annotations.ThreadLeakFilters;
import lombok.SneakyThrows;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.MMapDirectory;
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
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

import static org.mockito.Mockito.when;

/**
 * P4: P2's read profile on the FP16 path, where a vector is {@code 2 x dimension} bytes and is served through
 * {@code readBytes} rather than {@code readFloats}.
 *
 * <p><b>Why the FP16 path needs its own cases and the fp32 ones do not cover it.</b> Two of P2's three moving
 * parts are precision-blind: the grouping is arithmetic over a byte length the helper is given
 * ({@code getVectorByteLength()}, which the FP16 values answer {@code 2 x dimension}), and the store keys on
 * {@code (fileId, offset)} with the slot carrying its own byte length. The third is not: the serve and
 * admission sites for FP16 are inside {@link DirectIOVectorIndexInput#readBytes} (O8), gated on the field's
 * vector grid, and a `bytes` read is a <em>different range</em> from an `ops` read of the same ordinals. So
 * what has to be shown here is that a read the `bytes` grouping shaped still lands on the grid, still admits,
 * and still serves the same bytes the delegate would — not that the grouping arithmetic works, which the fp32
 * cases already pin.
 *
 * <p>And one thing only this length can show: at 1,536 B a vector (768 half floats) two misses fit in one
 * 4 KiB block, which at 3,072 B of fp32 never happens. That is the `bytes` profile's best case and the fp32
 * suite cannot reach it.
 */
@ThreadLeakFilters(defaultFilters = true, filters = { DirectIOHalfFloatReadProfileTests.ReadPoolThreadFilter.class })
public class DirectIOHalfFloatReadProfileTests extends KNNTestCase {

    /** The read pool is never shut down by design; its daemon threads outlive any one suite. */
    public static final class ReadPoolThreadFilter implements ThreadFilter {
        @Override
        public boolean reject(final Thread thread) {
            return thread.getName().startsWith(DirectIOReadPool.THREAD_NAME_PREFIX);
        }
    }

    /** 768 half floats = 1,536 B: narrower than a 4 KiB block, so two of these can share one. */
    private static final int EVEN_DIMENSION = 768;
    /** 65 half floats = 130 B: a multiple of 2 and not of 4, the length O8's store rule had to allow. */
    private static final int ODD_DIMENSION = 65;
    /** Not a multiple of any vector length and not a multiple of a block, which is the awkward real case. */
    private static final int HEADER_LENGTH = 44;
    private static final int FOOTER_LENGTH = 16;
    private static final int VECTOR_COUNT = 1_024;
    private static final int TEST_SLAB_BYTES = 1024 * 1024;
    private static final long BYTES_128 = 128 * 1024;
    private static final String LOW_CAP = "16kb";

    private Path path;
    private int vectorBytes;
    /** Each vector's raw FP16 bytes, which is what the FP16 read path reads and what a hit must equal. */
    private byte[][] vectors;

    @Override
    public void tearDown() throws Exception {
        DirectIOVectorCache.resetForTesting();
        DirectIOReadPool.resetForTesting();
        DirectIOVectorIndexInput.STATS.reset();
        DirectIOVectorIndexInput.resetReadProfileForTesting();
        super.tearDown();
    }

    // -----------------------------------------------------------------------------------------------
    // Grouping at an FP16 vector length
    // -----------------------------------------------------------------------------------------------

    /**
     * The {@code bytes} rule at 1,536 B, including the case only an FP16 length reaches: two misses inside
     * one block, one read.
     */
    @SneakyThrows
    public void testBytesGroupingAtAnFp16VectorLength() {
        assumeDirectIOWorksHere();
        writeFixture(1L, EVEN_DIMENSION);
        configure("bytes", LOW_CAP, 0, false);

        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(input);
            assumeTrue("this filesystem's block is not 4 KiB", input.blockSize() == 4096);
            assertEquals("bytes", DirectIOVectorIndexInput.readProfile());
            assertEquals("the fixture must be 2 bytes a dimension", 2 * EVEN_DIMENSION, vectorBytes);

            // ords 0 and 1 are [44,1580) and [1580,3116): both inside block 0. One read, which is the case
            // an fp32 length can never produce.
            assertEquals(1, groupsFor(input, new int[] { 0, 1 }));
            // ord 3 is [4652,6188), block 1; the group ends in block 0. Adjacent, so still one read.
            assertEquals(1, groupsFor(input, new int[] { 0, 3 }));
            // ord 6 is [9260,10796), block 2; block 1 would carry nothing. Two reads.
            assertEquals(2, groupsFor(input, new int[] { 0, 6 }));
            // And a gapless run is one read up to the cap, then split.
            assertEquals(1, groupsFor(input, consecutive(0, 10)));
            assertTrue("a long gapless run must split", groupsFor(input, consecutive(0, 40)) > 1);
        }
    }

    /** The odd length too: 130 B a vector, a multiple of 2 and not of 4. */
    @SneakyThrows
    public void testBytesGroupingAtAnOddFp16VectorLength() {
        assumeDirectIOWorksHere();
        writeFixture(2L, ODD_DIMENSION);
        configure("bytes", LOW_CAP, 0, false);

        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(input);
            assumeTrue("this filesystem's block is not 4 KiB", input.blockSize() == 4096);
            assertEquals(2 * ODD_DIMENSION, vectorBytes);

            // At 130 B a vector, 31 of them fit in a block, so a lot of pairs share one read.
            assertEquals(1, groupsFor(input, new int[] { 0, 1, 2, 3 }));
            // ord 63 is at 44 + 8,190 = 8,234, block 2; the group ends in block 0, so block 1 is a gap.
            assertEquals(2, groupsFor(input, new int[] { 0, 63 }));
        }
    }

    /** Under {@code ops} the FP16 path keeps P1's grouping exactly, as the fp32 one does. */
    @SneakyThrows
    public void testOpsIsP1sGroupingAtAnFp16VectorLength() {
        assumeDirectIOWorksHere();
        writeFixture(3L, EVEN_DIMENSION);
        configure("ops", LOW_CAP, 0, false);

        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(input);
            assertEquals("ops", DirectIOVectorIndexInput.readProfile());
            for (final int[] batch : List.of(
                new int[] { 0, 1 },
                new int[] { 0, 3 },
                new int[] { 0, 6 },
                consecutive(0, 10),
                consecutive(0, 40),
                new int[] { 0, 40, 41, 300, 900, 901 }
            )) {
                assertEquals(
                    "ops must group " + Arrays.toString(batch) + " exactly as P1 does",
                    p1Groups(input, batch),
                    groupsFor(input, batch)
                );
            }
        }
    }

    // -----------------------------------------------------------------------------------------------
    // A bytes read of an FP16 .vec is admitted and byte-exact against the delegate
    // -----------------------------------------------------------------------------------------------

    /**
     * The one that matters: a batch shaped by the {@code bytes} grouping is staged, every vector of it is
     * served through {@code readBytes} with the bytes an mmap read of the same file gives, every one is
     * admitted, and the second pass is all hits.
     */
    @SneakyThrows
    public void testABytesReadOfAnFp16VecIsAdmittedAndByteExactAgainstTheDelegate() {
        assumeDirectIOWorksHere();
        writeFixture(4L, EVEN_DIMENSION);
        configure("bytes", LOW_CAP, 8, false);

        // Pairs: two adjacent ordinals (which share a block or two, so one read) spaced far apart (so each
        // pair is its own group). This is the shape the bytes profile is for.
        final int[] batch = { 10, 11, 110, 111, 210, 211, 310, 311 };
        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(input);
            final int[] copy = batch.clone();
            PrefetchHelper.prefetch(input, HEADER_LENGTH, vectorBytes, copy, copy.length);
            assertEquals("one group a pair", 4L, DirectIOVectorIndexInput.STATS.prefetchCalls());
            assertEquals("nothing declined", 0L, DirectIOVectorIndexInput.STATS.stageDeclined());

            final byte[] scratch = new byte[vectorBytes];
            try (MMapDirectory delegate = new MMapDirectory(path.getParent())) {
                try (IndexInput mmap = delegate.openInput(path.getFileName().toString(), org.apache.lucene.store.IOContext.DEFAULT)) {
                    final byte[] expected = new byte[vectorBytes];
                    for (final int ordinal : batch) {
                        input.seek(offsetOf(ordinal));
                        input.readBytes(scratch, 0, vectorBytes);
                        mmap.seek(offsetOf(ordinal));
                        mmap.readBytes(expected, 0, vectorBytes);
                        assertArrayEquals("ordinal " + ordinal + " must equal the delegate's bytes", expected, scratch);
                        assertArrayEquals("and the fixture's own bytes", vectors[ordinal], scratch);
                    }
                }
            }
            assertEquals("no read went blocking", 0L, DirectIOVectorIndexInput.STATS.blockingReads());
            for (final int ordinal : batch) {
                assertTrue(
                    "ordinal " + ordinal + " must be admitted from the bytes read",
                    DirectIOVectorCache.contains(fileId(), offsetOf(ordinal), vectorBytes)
                );
            }

            final long hitsBefore = DirectIOVectorCache.hits();
            for (final int ordinal : batch) {
                input.seek(offsetOf(ordinal));
                input.readBytes(scratch, 0, vectorBytes);
                assertArrayEquals(vectors[ordinal], scratch);
            }
            assertEquals("the second pass is all hits", hitsBefore + batch.length, DirectIOVectorCache.hits());
        }
    }

    /** Span fill still harvests the FP16 neighbours inside a {@code bytes} read, and they are byte-exact. */
    @SneakyThrows
    public void testSpanFillAdmitsFp16NeighboursInsideABytesRead() {
        assumeDirectIOWorksHere();
        writeFixture(5L, EVEN_DIMENSION);
        configure("bytes", LOW_CAP, 8, true);

        final int[] batch = { 10, 110, 210, 310, 410, 510 };
        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(input);
            assumeTrue("this filesystem's block is not 4 KiB", input.blockSize() == 4096);
            final int[] copy = batch.clone();
            PrefetchHelper.prefetch(input, HEADER_LENGTH, vectorBytes, copy, copy.length);

            final byte[] scratch = new byte[vectorBytes];
            for (final int ordinal : batch) {
                input.seek(offsetOf(ordinal));
                input.readBytes(scratch, 0, vectorBytes);
                assertArrayEquals(vectors[ordinal], scratch);
            }
            assertTrue("span fill ran inside the bytes reads", DirectIOVectorCache.spanFills() > batch.length);

            int neighbour = -1;
            for (int candidate = 0; candidate < VECTOR_COUNT && neighbour < 0; candidate++) {
                if (Arrays.binarySearch(copy, 0, copy.length, candidate) < 0
                    && DirectIOVectorCache.contains(fileId(), offsetOf(candidate), vectorBytes)) {
                    neighbour = candidate;
                }
            }
            assertTrue("span fill must have admitted an FP16 vector nobody asked for", neighbour >= 0);
            final long hitsBefore = DirectIOVectorCache.hits();
            input.seek(offsetOf(neighbour));
            input.readBytes(scratch, 0, vectorBytes);
            assertArrayEquals("a span-filled FP16 neighbour must be byte-exact", vectors[neighbour], scratch);
            assertEquals("and served from the store", hitsBefore + 1, DirectIOVectorCache.hits());
        }
    }

    // -----------------------------------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------------------------------

    /**
     * How many prefetch calls the production helper makes for {@code ords}. The helper's own filter call
     * declares the vector grid, which is what O8 gates {@code readBytes} on, so this is the production entry
     * point and not a shortcut.
     */
    @SneakyThrows
    private int groupsFor(final DirectIOVectorIndexInput input, final int[] ords) {
        final long before = DirectIOVectorIndexInput.STATS.prefetchCalls();
        final int[] copy = ords.clone();
        PrefetchHelper.prefetch(input, HEADER_LENGTH, vectorBytes, copy, copy.length);
        return (int) (DirectIOVectorIndexInput.STATS.prefetchCalls() - before);
    }

    /** P1's grouping: 128 KB and the input's span cap, and nothing about blocks. */
    private int p1Groups(final DirectIOVectorIndexInput input, final int[] ords) {
        final int[] sorted = ords.clone();
        Arrays.sort(sorted);
        final long cap = input.maxStagedRangeBytes();
        int groups = 1;
        long groupStart = offsetOf(sorted[0]);
        for (int i = 1; i < sorted.length; i++) {
            final long end = offsetOf(sorted[i]) + vectorBytes;
            if (end - groupStart > BYTES_128 || input.alignedSpanBytes(groupStart, end - groupStart) > cap) {
                groups++;
                groupStart = offsetOf(sorted[i]);
            }
        }
        return groups;
    }

    private static int[] consecutive(final int first, final int count) {
        final int[] ords = new int[count];
        for (int i = 0; i < count; i++) {
            ords[i] = first + i;
        }
        return ords;
    }

    private long offsetOf(final int ordinal) {
        return HEADER_LENGTH + (long) ordinal * vectorBytes;
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
        when(clusterService.getSettings()).thenReturn(settings);
        KNNSettings.state().setClusterService(clusterService);
    }

    private void configure(final String profile, final String cap, final int slabs, final boolean spanFill) {
        final long maxBytes = (long) slabs * TEST_SLAB_BYTES;
        withNodeSettings(
            Settings.builder()
                .put(KNNSettings.KNN_DIRECT_IO_RESCORE_READ_PROFILE, profile)
                .put(KNNSettings.KNN_DIRECT_IO_RESCORE_PREFETCH_MAX_SPAN, cap)
                .put(KNNSettings.KNN_DIRECT_IO_RESCORE_CACHE_MAX_BYTES, maxBytes + "b")
                .put(KNNSettings.KNN_DIRECT_IO_RESCORE_CACHE_SPAN_FILL, spanFill)
                .put(KNNSettings.KNN_DIRECT_IO_RESCORE_CACHE_READ_AROUND_BYTES, "0b")
                .put(KNNSettings.KNN_DIRECT_IO_READAHEAD_WINDOW, 0)
                .build()
        );
        DirectIOVectorCache.resetForTesting();
        DirectIOVectorCache.setSlabBytesForTesting(TEST_SLAB_BYTES);
        DirectIOVectorCache.configure(maxBytes);
        DirectIOVectorIndexInput.STATS.reset();
        DirectIOVectorIndexInput.resetReadProfileForTesting();
    }

    /**
     * A file shaped like a half-float {@code .vec}: a header, {@code VECTOR_COUNT} vectors of
     * {@code 2 x dimension} raw bytes, a footer. The bytes are arbitrary rather than encoded half floats,
     * because every claim here is about which bytes are read and none is about what they decode to.
     */
    @SneakyThrows
    private void writeFixture(final long seed, final int dimension) {
        vectorBytes = 2 * dimension;
        final Random random = new Random(seed);
        vectors = new byte[VECTOR_COUNT][vectorBytes];
        for (final byte[] vector : vectors) {
            random.nextBytes(vector);
        }
        path = createTempDir().resolve("p4-hf-" + seed + ".vec");
        final ByteBuffer buffer = ByteBuffer.allocate(HEADER_LENGTH + VECTOR_COUNT * vectorBytes + FOOTER_LENGTH)
            .order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < HEADER_LENGTH; i++) {
            buffer.put((byte) (0xA0 + (i & 0x0F)));
        }
        for (final byte[] vector : vectors) {
            buffer.put(vector);
        }
        for (int i = 0; i < FOOTER_LENGTH; i++) {
            buffer.put((byte) (0xF0 + (i & 0x0F)));
        }
        Files.write(path, buffer.array());
    }
}

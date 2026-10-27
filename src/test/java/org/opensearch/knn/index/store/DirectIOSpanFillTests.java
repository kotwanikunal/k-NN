/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import com.carrotsearch.randomizedtesting.ThreadFilter;
import com.carrotsearch.randomizedtesting.annotations.ThreadLeakFilters;
import lombok.SneakyThrows;
import org.apache.lucene.store.IndexInput;
import org.opensearch.common.settings.ClusterSettings;
import org.opensearch.common.settings.Setting;
import org.opensearch.common.settings.Settings;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.KNNSettings;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

import static org.mockito.Mockito.when;

/**
 * M4: span fill at the input level, ported from phase 16-I's F5.
 *
 * <p>While the store is below its slab target, a staged span also admits every whole vector of the reading
 * slice inside it. The tests here are about the <b>geometry</b>, because that is where this can go wrong in a
 * way nothing else would catch: a span is block aligned in the whole file, so its ends routinely sit inside a
 * vector and routinely reach outside the slice being read, into another field's vectors or into the file's
 * header or footer. Admitting a partial vector, or bytes from outside the slice, would put a key in the store
 * whose bytes are not the vector that key names — and the next hit on it would return the wrong answer
 * silently.
 *
 * <p>So each test computes the expected admitted set from the arithmetic the input has to be doing (the
 * block-aligned span, clipped to the slice, rounded in to whole vectors, ended at the bytes the read
 * returned) and asserts the store holds exactly that: nothing less, and in particular nothing more.
 */
@ThreadLeakFilters(defaultFilters = true, filters = { DirectIOSpanFillTests.ReadPoolThreadFilter.class })
public class DirectIOSpanFillTests extends KNNTestCase {

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
    private static final int VECTOR_COUNT = 512;
    private static final int TEST_SLAB_BYTES = 64 * 1024;

    private float[][] vectors;
    private Path path;

    @Override
    public void tearDown() throws Exception {
        DirectIOVectorCache.resetForTesting();
        DirectIOReadPool.resetForTesting();
        DirectIOVectorIndexInput.STATS.reset();
        super.tearDown();
    }

    /**
     * One span, in the middle of the file: exactly the whole vectors inside the block-aligned span are
     * admitted, and the vectors the span only partly covers are not.
     */
    @SneakyThrows
    public void testASpanAdmitsOnlyTheWholeVectorsInsideIt() {
        assumeDirectIOWorksHere();
        writeFixture(1L);
        spanFillOn(4);

        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(input);
            final int ordinal = 200;
            final long at = offsetOf(ordinal);
            final Span span = spanOf(input.blockSize(), at, VECTOR_BYTES, 0, Files.size(path));
            readVector(input, at, new float[DIMENSION]);

            final List<Integer> expected = wholeVectorsIn(span, at, 0, Files.size(path));
            assertTrue("the span must hold more than the requested vector, it held " + expected.size(), expected.size() > 1);
            assertEquals("the requested vector is in the run", true, expected.contains(ordinal));
            assertCachedExactly(expected);
            assertEquals(
                "every admitted vector but the requested one counts as a span fill",
                expected.size() - 1L,
                DirectIOVectorCache.spanFills()
            );

            // And the two the span only partly covers are the ones that must not be there.
            final int before = expected.get(0) - 1;
            final int after = expected.get(expected.size() - 1) + 1;
            if (before >= 0) {
                assertFalse(
                    "vector " + before + " starts before the span and must not be admitted",
                    DirectIOVectorCache.contains(fileId(), offsetOf(before), VECTOR_BYTES)
                );
            }
            if (after < VECTOR_COUNT) {
                assertFalse(
                    "vector " + after + " ends after the span and must not be admitted",
                    DirectIOVectorCache.contains(fileId(), offsetOf(after), VECTOR_BYTES)
                );
            }
            DirectIOVectorCache.checkInvariantsForTesting();
        }
    }

    /**
     * The same read through a slice that ends before the span does. Nothing outside
     * {@code [sliceOffset, sliceOffset + sliceLength)} may be admitted, because outside it the bytes belong
     * to another field, to the header or to the footer, and a vector boundary computed across that line would
     * be a boundary of nothing.
     */
    @SneakyThrows
    public void testASpanMayNotReachOutsideTheReadingSlice() {
        assumeDirectIOWorksHere();
        writeFixture(2L);
        spanFillOn(4);

        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(input);
            // A slice holding four vectors only, so a 4 KiB block reaches well past both of its ends.
            final int firstOrdinal = 100;
            final int sliceVectors = 4;
            final long sliceOffset = offsetOf(firstOrdinal);
            final long sliceLength = (long) sliceVectors * VECTOR_BYTES;
            try (IndexInput slice = input.slice("four-vectors", sliceOffset, sliceLength)) {
                // Read the slice's second vector, so the span reaches outside the slice at both ends.
                final long inSlice = (long) VECTOR_BYTES;
                slice.prefetch(inSlice, VECTOR_BYTES);
                slice.seek(inSlice);
                slice.readFloats(new float[DIMENSION], 0, DIMENSION);
            }

            final Span span = spanOf(input.blockSize(), sliceOffset + VECTOR_BYTES, VECTOR_BYTES, sliceOffset, sliceOffset + sliceLength);
            final List<Integer> expected = wholeVectorsIn(span, sliceOffset + VECTOR_BYTES, sliceOffset, sliceOffset + sliceLength);
            // Not the whole slice: at a 4 KiB block and a 44 B header the span's end lands inside the
            // slice's last vector, so three of the four go in. That is the clipping being asserted, so the
            // count is derived rather than asserted as a constant, with bounds to show it is a real clip at
            // both ends.
            assertTrue("the span must offer more than the requested vector, it offered " + expected.size(), expected.size() > 1);
            assertTrue("and fewer than the whole slice, it offered " + expected.size(), expected.size() < sliceVectors);
            assertCachedExactly(expected);
            for (final int outside : new int[] {
                firstOrdinal - 2,
                firstOrdinal - 1,
                firstOrdinal + sliceVectors,
                firstOrdinal + sliceVectors + 1 }) {
                assertFalse(
                    "vector " + outside + " is outside the reading slice and must not be admitted",
                    DirectIOVectorCache.contains(fileId(), offsetOf(outside), VECTOR_BYTES)
                );
            }
            DirectIOVectorCache.checkInvariantsForTesting();
        }
    }

    /** A span at the very end of the file ends at the bytes the read returned, not at the aligned length. */
    @SneakyThrows
    public void testASpanAtTheEndOfTheFileOffersNothingBeyondIt() {
        assumeDirectIOWorksHere();
        writeFixture(3L);
        spanFillOn(4);

        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(input);
            final int ordinal = VECTOR_COUNT - 1;
            final long at = offsetOf(ordinal);
            readVector(input, at, new float[DIMENSION]);

            final Span span = spanOf(input.blockSize(), at, VECTOR_BYTES, 0, Files.size(path));
            final List<Integer> expected = wholeVectorsIn(span, at, 0, Files.size(path));
            assertTrue("the last vector itself is admitted", expected.contains(ordinal));
            assertCachedExactly(expected);
            // Nothing past the last vector exists to be admitted, and the footer must not have been read as
            // one: every cached key is a real vector offset, which assertCachedExactly already proves, so
            // what is checked here is that the count did not quietly include a phantom.
            assertEquals(expected.size(), (int) DirectIOVectorCache.entries());
            DirectIOVectorCache.checkInvariantsForTesting();
        }
    }

    /** The point of it all: a vector nobody asked for is a hit afterwards, and costs no device read. */
    @SneakyThrows
    public void testSpanFilledVectorsHitAfterwardsWithNoDeviceRead() {
        assumeDirectIOWorksHere();
        writeFixture(4L);
        spanFillOn(4);

        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(input);
            final int ordinal = 64;
            final long at = offsetOf(ordinal);
            readVector(input, at, new float[DIMENSION]);
            final List<Integer> admitted = wholeVectorsIn(
                spanOf(input.blockSize(), at, VECTOR_BYTES, 0, Files.size(path)),
                at,
                0,
                Files.size(path)
            );
            assertTrue(admitted.size() > 1);

            final long stagedBefore = DirectIOVectorIndexInput.STATS.stagedReads();
            final long blockingBefore = DirectIOVectorIndexInput.STATS.blockingReads();
            final long hitsBefore = DirectIOVectorCache.hits();
            final float[] scratch = new float[DIMENSION];
            for (final int neighbour : admitted) {
                readVector(input, offsetOf(neighbour), scratch);
                assertArrayEquals("a span-filled vector must be its own bytes", vectors[neighbour], scratch, 0.0f);
            }
            assertEquals("every one of them hit", hitsBefore + admitted.size(), DirectIOVectorCache.hits());
            assertEquals("and none of them reached the device", stagedBefore, DirectIOVectorIndexInput.STATS.stagedReads());
            assertEquals(blockingBefore, DirectIOVectorIndexInput.STATS.blockingReads());
        }
    }

    /**
     * At the slab target span fill admits nothing and evicts nothing. The requested fill still evicts, which
     * is unchanged; what must not happen is a span pushing the working set out to make room for vectors
     * nobody asked for.
     */
    @SneakyThrows
    public void testNoAdmissionAtTheTargetAndNoEvictionFromSpanFill() {
        assumeDirectIOWorksHere();
        writeFixture(5L);
        // Two 64 KiB slabs, 128 slots each for this fixture's 512 B vectors.
        //
        // Two and not one, and the reason is the policy rather than the test: "below the target" means the
        // store may still TAKE a slab, so a target of one slab is at its target the moment its only slab is
        // allocated and span fill never runs at all. That is 16-I's F5 rule as written, ported unchanged,
        // and it is worth knowing: a one-slab cache span fills nothing. With two, the first slab's life is
        // the window.
        spanFillOn(2);
        final int slots = DirectIOVectorCache.slotsPerSlab(VECTOR_BYTES);
        final float[] scratch = new float[DIMENSION];

        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(input);
            // Read until both slabs the target allows have been taken. Span fill gets there faster than the
            // requested reads alone would, which is the mechanism working.
            int ordinal = 0;
            while (DirectIOVectorCache.belowSlabTarget() && ordinal < VECTOR_COUNT) {
                readVector(input, offsetOf(ordinal++), scratch);
            }
            assertFalse("the store is at its target now", DirectIOVectorCache.belowSlabTarget());
            assertEquals("and it holds exactly the two slabs the target allows", 2L, DirectIOVectorCache.slabs());
            final long spanFillsAtTarget = DirectIOVectorCache.spanFills();
            assertTrue("span fill did the filling", spanFillsAtTarget > 0);
            assertTrue(
                "it took fewer reads than vectors admitted, which is the whole point: " + ordinal + " reads",
                ordinal < DirectIOVectorCache.entries()
            );

            // Now read vectors from the far end of the file, whose spans offer plenty. None of it may be
            // admitted beyond the requested vector itself.
            final long entriesBefore = DirectIOVectorCache.entries();
            for (int far = VECTOR_COUNT - 1; far > VECTOR_COUNT - 20; far--) {
                readVector(input, offsetOf(far), scratch);
                assertArrayEquals(vectors[far], scratch, 0.0f);
            }
            assertEquals("no span fill at the target", spanFillsAtTarget, DirectIOVectorCache.spanFills());
            assertEquals("and the store is still its two slabs", 2L, DirectIOVectorCache.slabs());
            assertTrue(
                "entries cannot exceed the slabs' slots, they were " + DirectIOVectorCache.entries(),
                DirectIOVectorCache.entries() <= 2L * slots
            );
            assertTrue("the requested vectors still went in", DirectIOVectorCache.entries() > 0);
            assertTrue("and the store did not shrink by span fill's doing", DirectIOVectorCache.entries() > entriesBefore / 2);
            DirectIOVectorCache.checkInvariantsForTesting();
        }
    }

    /** Span fill off restores M3's admission: the requested vector and nothing else. */
    @SneakyThrows
    public void testSpanFillOffRestoresRequestedOnlyAdmission() {
        assumeDirectIOWorksHere();
        writeFixture(6L);
        spanFillOff(4);

        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(input);
            readVector(input, offsetOf(77), new float[DIMENSION]);
            assertEquals("one vector asked for, one vector held", 1L, DirectIOVectorCache.entries());
            assertEquals(0L, DirectIOVectorCache.spanFills());
            assertTrue(DirectIOVectorCache.contains(fileId(), offsetOf(77), VECTOR_BYTES));
            assertFalse(DirectIOVectorCache.contains(fileId(), offsetOf(78), VECTOR_BYTES));
            assertFalse(DirectIOVectorCache.contains(fileId(), offsetOf(76), VECTOR_BYTES));
            DirectIOVectorCache.checkInvariantsForTesting();
        }
    }

    /**
     * Span fill is answered per read and not captured at open, so turning it on under an open input takes
     * effect on that input's next burst.
     */
    @SneakyThrows
    public void testSpanFillTakesEffectWithoutReopeningTheInput() {
        assumeDirectIOWorksHere();
        writeFixture(7L);
        spanFillOff(4);

        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(input);
            readVector(input, offsetOf(10), new float[DIMENSION]);
            assertEquals(0L, DirectIOVectorCache.spanFills());

            DirectIOVectorCache.setSpanFillForTesting(true);
            readVector(input, offsetOf(300), new float[DIMENSION]);
            assertTrue("the next burst picked span fill up", DirectIOVectorCache.spanFills() > 0);
        }
    }

    /**
     * The prefetch-side drop at the granularity the scorer actually asks for: a coalesced group of several
     * vectors. All of them cached means no staged read at all; one of them missing means the group is staged,
     * because the input cannot stage half a range.
     */
    @SneakyThrows
    public void testAFullyCachedGroupIsDroppedFromTheBurst() {
        assumeDirectIOWorksHere();
        writeFixture(11L);
        spanFillOff(4);

        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(input);
            final float[] scratch = new float[DIMENSION];
            // Four consecutive vectors, one at a time, so each is admitted as a requested fill.
            for (int ordinal = 50; ordinal < 54; ordinal++) {
                readVector(input, offsetOf(ordinal), scratch);
            }
            assertEquals(4L, DirectIOVectorCache.entries());

            // Now the group the scorer would ask for: the span of all four in one prefetch.
            final long stagedBefore = DirectIOVectorIndexInput.STATS.stagedReads();
            input.prefetch(offsetOf(50), 4L * VECTOR_BYTES);
            for (int ordinal = 50; ordinal < 54; ordinal++) {
                input.seek(offsetOf(ordinal));
                input.readFloats(scratch, 0, DIMENSION);
                assertArrayEquals(vectors[ordinal], scratch, 0.0f);
            }
            assertEquals("a fully cached group costs no device read", stagedBefore, DirectIOVectorIndexInput.STATS.stagedReads());

            // And a group with one vector missing is staged, because a range is all or nothing.
            input.prefetch(offsetOf(50), 6L * VECTOR_BYTES);
            input.seek(offsetOf(55));
            input.readFloats(scratch, 0, DIMENSION);
            assertArrayEquals(vectors[55], scratch, 0.0f);
            assertTrue("the group with a miss in it had to be staged", DirectIOVectorIndexInput.STATS.stagedReads() > stagedBefore);
        }
    }

    /** A blocking read is not a staged span, so it fills nothing at all, span fill on or off. */
    @SneakyThrows
    public void testABlockingReadSpanFillsNothing() {
        assumeDirectIOWorksHere();
        writeFixture(8L);
        spanFillOn(4);

        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(input);
            final float[] scratch = new float[DIMENSION];
            // No prefetch and backwards through the file, so neither the burst table nor the ring serves it.
            for (int ordinal = 400; ordinal > 380; ordinal--) {
                input.seek(offsetOf(ordinal));
                input.readFloats(scratch, 0, DIMENSION);
                assertArrayEquals(vectors[ordinal], scratch, 0.0f);
            }
            assertTrue("these were blocking reads", DirectIOVectorIndexInput.STATS.blockingReads() > 0);
            assertEquals(0L, DirectIOVectorCache.spanFills());
            assertEquals(0L, DirectIOVectorCache.entries());
        }
    }

    /** The key: a span-filled vector of one file is never served for another file at the same offset. */
    @SneakyThrows
    public void testSpanFilledEntriesAreKeyedByFile() {
        assumeDirectIOWorksHere();
        writeFixture(9L);
        final Path other = path;
        final float[][] otherVectors = vectors;
        writeFixture(10L);
        assertEquals("the two fixtures are the same size, which is the hard case", Files.size(other), Files.size(path));
        spanFillOn(4);

        final float[] scratch = new float[DIMENSION];
        final float[] fromFirst = new float[DIMENSION];
        // Both open: since O3 the first input's close would free its file's span-filled entries, and the
        // claim here is that the two files' entries cannot be confused while both are cached.
        try (
            DirectIOVectorIndexInput first = DirectIOVectorIndexInput.open(other);
            DirectIOVectorIndexInput second = DirectIOVectorIndexInput.open(path)
        ) {
            assertNotNull(first);
            assertNotNull(second);
            readVector(first, offsetOf(33), new float[DIMENSION]);

            readVector(second, offsetOf(34), scratch);
            assertArrayEquals("the second file's own bytes, not the first file's span fill", vectors[34], scratch, 0.0f);

            assertTrue(
                DirectIOVectorCache.getFloats(
                    DirectIOVectorCache.fileId(other, Files.size(other)),
                    offsetOf(34),
                    VECTOR_BYTES,
                    fromFirst,
                    0
                )
            );
            assertArrayEquals("and the first file's span fill is still the first file's bytes", otherVectors[34], fromFirst, 0.0f);
        }
    }

    // -----------------------------------------------------------------------------------------------
    // The geometry the input has to be doing, recomputed here rather than trusted
    // -----------------------------------------------------------------------------------------------

    /** A block-aligned staged span: where it starts and how much of it the read could have delivered. */
    private record Span(long start, long end) {
    }

    /**
     * The span {@code prefetch(at, needed)} stages, clipped to the reading slice and to the file: aligned
     * down to a block, aligned up by a block, and ended at what the file actually holds.
     */
    private static Span spanOf(final int blockSize, final long at, final int needed, final long sliceFrom, final long sliceTo) {
        final long alignedStart = at - Math.floorMod(at, (long) blockSize);
        final long alignedLength = ((at + needed - alignedStart) + blockSize - 1) / blockSize * blockSize;
        return new Span(Math.max(alignedStart, sliceFrom), Math.min(alignedStart + alignedLength, sliceTo));
    }

    /**
     * The vectors of the grid anchored at {@code runStart} that lie wholly inside {@code span}: the first
     * boundary at or after its start, the last vector ending at or before its end.
     */
    private List<Integer> wholeVectorsIn(final Span span, final long runStart, final long sliceFrom, final long sliceTo) {
        final List<Integer> inside = new ArrayList<>();
        for (int ordinal = 0; ordinal < VECTOR_COUNT; ordinal++) {
            final long start = offsetOf(ordinal);
            if (start < sliceFrom || start + VECTOR_BYTES > sliceTo) {
                continue;
            }
            if ((start - runStart) % VECTOR_BYTES != 0) {
                continue;
            }
            if (start >= span.start() && start + VECTOR_BYTES <= span.end()) {
                inside.add(ordinal);
            }
        }
        return inside;
    }

    /** The store holds exactly these vectors of this file, with their own bytes, and nothing else. */
    private void assertCachedExactly(final List<Integer> expected) {
        final Set<Integer> wanted = new HashSet<>(expected);
        final float[] read = new float[DIMENSION];
        for (int ordinal = 0; ordinal < VECTOR_COUNT; ordinal++) {
            final boolean held = DirectIOVectorCache.contains(fileId(), offsetOf(ordinal), VECTOR_BYTES);
            assertEquals("vector " + ordinal + " cached?", wanted.contains(ordinal), held);
            if (held) {
                assertTrue(DirectIOVectorCache.getFloats(fileId(), offsetOf(ordinal), VECTOR_BYTES, read, 0));
                assertArrayEquals("vector " + ordinal + "'s bytes", vectors[ordinal], read, 0.0f);
            }
        }
        assertEquals("and nothing else at all", expected.size(), (int) DirectIOVectorCache.entries());
    }

    // -----------------------------------------------------------------------------------------------

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
        // The route is on unless the caller says otherwise. These cases read through O_DIRECT with no
        // delegate to fall back to, which is the one shape the route switch does not reach, so the node
        // they model is a node that is routing: publishing the route off would give the store back under
        // them. A case whose subject is the switch puts its own value and that wins.
        final Settings routed = Settings.builder().put(KNNSettings.KNN_DIRECT_IO_RESCORE_ENABLED, true).put(settings).build();
        when(clusterService.getClusterSettings()).thenReturn(new ClusterSettings(routed, registered));
        KNNSettings.state().setClusterService(clusterService);
        // The seeding a node does at startup: the dynamic Direct I/O settings live in volatile fields of
        // the components that own them, so publishing is how a setting reaches the read path.
        KNNSettings.state().publishDirectIOSettings();
    }

    private void spanFillOn(final int slabs) {
        configure(slabs, true);
    }

    private void spanFillOff(final int slabs) {
        configure(slabs, false);
    }

    private void configure(final int slabs, final boolean spanFill) {
        final long maxBytes = (long) slabs * TEST_SLAB_BYTES;
        withNodeSettings(
            Settings.builder()
                .put(KNNSettings.KNN_DIRECT_IO_RESCORE_CACHE_MAX_BYTES, maxBytes + "b")
                .put(KNNSettings.KNN_DIRECT_IO_READAHEAD_WINDOW, 0)
                .build()
        );
        DirectIOVectorCache.resetForTesting();
        DirectIOVectorCache.setSpanFillForTesting(spanFill);
        DirectIOVectorCache.setSlabBytesForTesting(TEST_SLAB_BYTES);
        DirectIOVectorCache.publishMaxBytes(maxBytes);
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
        path = createTempDir().resolve("span-" + seed + ".vec");
        final ByteBuffer buffer = ByteBuffer.allocate(HEADER_LENGTH + VECTOR_COUNT * VECTOR_BYTES + FOOTER_LENGTH)
            .order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < HEADER_LENGTH; i++) {
            // A header of recognisable bytes: if any of it were ever admitted as a vector, the bytes a hit
            // returned would not match the fixture and assertCachedExactly would say so.
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

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import com.carrotsearch.randomizedtesting.ThreadFilter;
import com.carrotsearch.randomizedtesting.annotations.ThreadLeakFilters;
import lombok.SneakyThrows;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexOutput;
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
import java.util.Set;

import static org.mockito.Mockito.when;

/**
 * P2: {@code knn.direct_io.rescore.read_profile}, the shape of a staged read.
 *
 * <p><b>Why there is a setting at all (MEASURED, ref/RESULTS-NVME.md).</b> The build that wins on an EBS node
 * (gp3, 3,000 IOPS: throughput +35 / +43 / +48% against mmap) loses or ties on an instance-store node
 * (nvme1n1, about 275-295 MB/s measured byte cap: -25 / -6 / -8.5%), for one reason: a staged read carries
 * the gaps between the misses it coalesces, 772 KB a query in the r3 regime at 1.22 misses a 12.2 KB read,
 * which is 2.44x mmap's bytes on a device limited by bytes rather than by reads. On gp3 the reads are the
 * scarce resource and that same coalescing is the win. So the read shape has to follow the device, and which
 * device a node has is the deployer's knowledge: the AOS control plane sets this per instance type, k-NN
 * senses nothing.
 *
 * <p>What these tests hold the change to:
 * <ul>
 *   <li>{@code bytes} closes a group whenever a whole block between the group and the next vector would hold
 *       no byte of any missing vector, and keeps it open for two misses in one block or in adjacent blocks;</li>
 *   <li>{@code ops} is P1's grouping, call for call;</li>
 *   <li>an input that is not a Direct I/O input is unchanged under {@code bytes}, so the mmap baseline every
 *       measurement is taken against cannot move;</li>
 *   <li>a {@code bytes} read is staged, admitted and served byte-exact, and span fill still admits the
 *       whole vectors inside it;</li>
 *   <li>both a 768-dimension fp32 length (3,072 B, wider than a block) and a 2-bytes-per-dimension one
 *       (1,536 B, narrower) are covered, because the first can never put two misses in one block and the
 *       second can.</li>
 * </ul>
 * The setting's own properties are in {@code KNNSettingsTests}.
 */
@ThreadLeakFilters(defaultFilters = true, filters = { DirectIOReadProfileTests.ReadPoolThreadFilter.class })
public class DirectIOReadProfileTests extends KNNTestCase {

    /** The read pool is never shut down by design; its daemon threads outlive any one suite. */
    public static final class ReadPoolThreadFilter implements ThreadFilter {
        @Override
        public boolean reject(final Thread thread) {
            return thread.getName().startsWith(DirectIOReadPool.THREAD_NAME_PREFIX);
        }
    }

    /** 768 fp32 dimensions, 3,072 B: wider than a 4 KiB block, so two of these are never in one block. */
    private static final int FP32_DIMENSION = 768;
    private static final int FP32_VECTOR_BYTES = FP32_DIMENSION * Float.BYTES;
    /**
     * 1,536 B, which is what a 768-dimension half-float vector occupies: narrower than a 4 KiB block, so two
     * of these can sit in one block where two fp32 ones never can. The fixture stores fp32 floats, so at this
     * length a "vector" is 384 of them; for the grouping only the byte length matters, and where bytes are
     * read back the test reads 384 floats.
     */
    private static final int NARROW_VECTOR_BYTES = FP32_DIMENSION * 2;

    private static final int VECTOR_COUNT = 2_048;
    private static final int TEST_SLAB_BYTES = 1024 * 1024;
    /** The span cap the current arm sweeps, applied by {@link #openAtCap(Path)}. */
    private int capBytes = DirectIOVectorIndexInput.DEFAULT_MAX_STAGED_RANGE_BYTES;
    private static final long BYTES_128 = 128 * 1024;
    /** A cap low enough that it, and not the 128 KB group budget, is what closes a gapless run. */
    private static final int LOW_CAP = 16 * 1024;

    private float[][] vectors;
    private Path path;
    private int headerLength;

    @Override
    public void tearDown() throws Exception {
        DirectIOVectorCache.resetForTesting();
        DirectIOReadPool.resetForTesting();
        DirectIOVectorIndexInput.STATS.reset();
        DirectIOVectorIndexInput.resetReadProfileForTesting();
        super.tearDown();
    }

    // -----------------------------------------------------------------------------------------------
    // What the profile resolves to
    // -----------------------------------------------------------------------------------------------

    /** The profile is read from the node's own settings, defaults to {@code ops}, and is then cached. */
    @SneakyThrows
    public void testTheProfileIsReadFromTheNodeSettingsAndCached() {
        DirectIOVectorIndexInput.resetReadProfileForTesting();
        withNodeSettings(Settings.builder().build());
        assertEquals("ops", DirectIOVectorIndexInput.readProfile());

        DirectIOVectorIndexInput.resetReadProfileForTesting();
        withNodeSettings(Settings.builder().put(KNNSettings.KNN_DIRECT_IO_RESCORE_READ_PROFILE, "bytes").build());
        assertEquals("bytes", DirectIOVectorIndexInput.readProfile());

        // Cached, because the setting is static: a later settings change is deliberately not picked up
        // without a restart, and the test asserts that rather than leaving it to chance.
        withNodeSettings(Settings.builder().put(KNNSettings.KNN_DIRECT_IO_RESCORE_READ_PROFILE, "ops").build());
        assertEquals("the resolved profile must not change under a running node", "bytes", DirectIOVectorIndexInput.readProfile());
    }

    // -----------------------------------------------------------------------------------------------
    // The bytes grouping, on hand-built ordinal sets
    // -----------------------------------------------------------------------------------------------

    /**
     * The four cases BACKLOG names, at a 1,536 B vector length, which is the only one of the two that can put
     * two misses in one block. Expectations are written as block arithmetic over a 4 KiB block and the test
     * assumes that block size rather than deriving the expectation from the code under test.
     */
    @SneakyThrows
    public void testBytesGroupingOnAHalfFloatShapedLength() {
        assumeDirectIOWorksHere();
        writeFixture(1L, NARROW_VECTOR_BYTES, 0);
        configure("bytes", LOW_CAP);

        try (DirectIOVectorIndexInput input = openAtCap(path)) {
            assertNotNull(input);
            assumeTrue("this filesystem's block is not 4 KiB", input.blockSize() == 4096);
            assertEquals("bytes", DirectIOVectorIndexInput.readProfile());

            // ords 0 and 1 are [0,1536) and [1536,3072): both in block 0. One read.
            assertEquals(1, groupsFor(input, NARROW_VECTOR_BYTES, new int[] { 0, 1 }));
            // ord 3 is [4608,6144), block 1; the group ends in block 0. Adjacent, so still one read.
            assertEquals(1, groupsFor(input, NARROW_VECTOR_BYTES, new int[] { 0, 3 }));
            // ord 6 is [9216,10752), block 2; the group ends in block 0, so block 1 would carry nothing.
            // Two reads.
            assertEquals(2, groupsFor(input, NARROW_VECTOR_BYTES, new int[] { 0, 6 }));
            // A gapless run: one read until the cap. 10 x 1,536 = 15,360 aligns to 16,384, which is the cap;
            // an eleventh would align to 20,480. Twenty consecutive vectors are THREE reads and not two,
            // because only the first group starts on a block: the second starts at 15,360, 3,072 B into a
            // block, so its aligned window has room for 8 vectors rather than 10, and the last two are a
            // third group. The cap bounds the aligned length, which includes the leading partial block.
            assertEquals(1, groupsFor(input, NARROW_VECTOR_BYTES, consecutive(0, 10)));
            assertEquals(3, groupsFor(input, NARROW_VECTOR_BYTES, consecutive(0, 20)));
        }
    }

    /**
     * The same rules at 3,072 B, where a vector is wider than a block, so "two misses in one block" cannot
     * happen and the interesting boundary moves.
     */
    @SneakyThrows
    public void testBytesGroupingOnAnFp32Length() {
        assumeDirectIOWorksHere();
        writeFixture(2L, FP32_VECTOR_BYTES, 0);
        configure("bytes", LOW_CAP);

        try (DirectIOVectorIndexInput input = openAtCap(path)) {
            assertNotNull(input);
            assumeTrue("this filesystem's block is not 4 KiB", input.blockSize() == 4096);

            // ord 1 is [3072,6144), which starts in block 0 where the group ends. One read.
            assertEquals(1, groupsFor(input, FP32_VECTOR_BYTES, new int[] { 0, 1 }));
            // ord 2 is [6144,9216), block 1; the group ends in block 0. Adjacent, one read.
            assertEquals(1, groupsFor(input, FP32_VECTOR_BYTES, new int[] { 0, 2 }));
            // ord 3 is [9216,12288), block 2; block 1 would carry nothing. Two reads.
            assertEquals(2, groupsFor(input, FP32_VECTOR_BYTES, new int[] { 0, 3 }));
            // A gapless run: 5 x 3,072 = 15,360 aligns to 16,384, the cap. Ten are three reads, not two, for
            // the reason the half-float test spells out: the second group starts 3,072 B into a block and so
            // holds four vectors, leaving the tenth to a third.
            assertEquals(1, groupsFor(input, FP32_VECTOR_BYTES, consecutive(0, 5)));
            assertEquals(3, groupsFor(input, FP32_VECTOR_BYTES, consecutive(0, 10)));
        }
    }

    /**
     * The gap rule is taken over the <em>absolute</em> file offset, so a field whose data starts partway into
     * a block, and a slice whose base does, both shift which pairs share a read.
     *
     * <p>Two different shifts are at work and the test separates them. The field's base offset (a 44 byte
     * header here) is an argument the helper is given and simply moves every vector. The slice's own base is
     * not visible to the helper at all, which is why {@link DirectIOVectorIndexInput#blockIndexOf} is on the
     * input: a slice answers block indexes in the file's frame, not in its own.
     */
    @SneakyThrows
    public void testTheGapRuleFollowsTheFieldsAbsoluteOffset() {
        assumeDirectIOWorksHere();
        writeFixture(3L, NARROW_VECTOR_BYTES, 44);
        configure("bytes", LOW_CAP);

        try (DirectIOVectorIndexInput input = openAtCap(path)) {
            assertNotNull(input);
            assumeTrue("this filesystem's block is not 4 KiB", input.blockSize() == 4096);

            // With a 44 byte header, ord 5 is [7724,9260), block 1, and the group ending in block 0 keeps it.
            assertEquals(1, groupsFor(input, NARROW_VECTOR_BYTES, new int[] { 0, 5 }));
            // ord 6 is [9260,10796), block 2, so the gap block closes the group.
            assertEquals(2, groupsFor(input, NARROW_VECTOR_BYTES, new int[] { 0, 6 }));
            // The accessor itself, on the root input whose base is the file's own zero.
            assertEquals(0L, input.blockIndexOf(0));
            assertEquals(0L, input.blockIndexOf(4095));
            assertEquals(1L, input.blockIndexOf(4096));
            // And on a slice, whose base the helper never sees: the same byte of the file, the same block.
            try (DirectIOVectorIndexInput slice = (DirectIOVectorIndexInput) input.slice("field", 44, 64L * 4096)) {
                assertEquals("a slice answers in the file's frame", input.blockIndexOf(4096), slice.blockIndexOf(4096 - 44));
                assertEquals(0L, slice.blockIndexOf(0));
            }
        }
    }

    // -----------------------------------------------------------------------------------------------
    // ops is P1's grouping, and an mmap input is nobody's business
    // -----------------------------------------------------------------------------------------------

    /**
     * Under {@code ops} every call is P1's: the 128 KB budget and the span cap, and no block rule. Checked
     * over the same ordinal sets the {@code bytes} tests use plus two scattered ones, against the expectation
     * computed here.
     */
    @SneakyThrows
    public void testOpsIsP1sGroupingCallForCall() {
        assumeDirectIOWorksHere();
        writeFixture(4L, FP32_VECTOR_BYTES, 44);
        configure("ops", LOW_CAP);

        try (DirectIOVectorIndexInput input = openAtCap(path)) {
            assertNotNull(input);
            assertEquals("ops", DirectIOVectorIndexInput.readProfile());
            for (final int[] batch : List.of(
                new int[] { 0, 1 },
                new int[] { 0, 2 },
                new int[] { 0, 3 },
                consecutive(0, 5),
                consecutive(0, 10),
                consecutive(100, 60),
                new int[] { 0, 40, 41, 300, 900, 901 }
            )) {
                assertEquals(
                    "ops must group " + Arrays.toString(batch) + " exactly as P1 does",
                    p1Groups(input, FP32_VECTOR_BYTES, 44, batch),
                    groupsFor(input, FP32_VECTOR_BYTES, batch)
                );
            }
        }
    }

    /** An input that is not a Direct I/O input gets the 128 KB grouping whatever the profile says. */
    @SneakyThrows
    public void testAnMmapInputIsUnchangedUnderBytes() {
        configure("bytes", LOW_CAP);
        assertEquals("bytes", DirectIOVectorIndexInput.readProfile());

        for (final int[] batch : List.of(
            new int[] { 0, 1 },
            new int[] { 0, 3 },
            new int[] { 0, 6 },
            consecutive(0, 20),
            new int[] { 0, 40, 41, 300, 900, 901 }
        )) {
            final TrackingIndexInput tracking = trackingInput(VECTOR_COUNT * FP32_VECTOR_BYTES);
            final int[] copy = batch.clone();
            PrefetchHelper.prefetch(tracking, 0L, FP32_VECTOR_BYTES, copy, copy.length);
            final List<long[]> expected = unsplitGrouping(0L, FP32_VECTOR_BYTES, batch);
            assertEquals("group count for " + Arrays.toString(batch), expected.size(), tracking.prefetchCalls.size());
            for (int i = 0; i < expected.size(); i++) {
                assertEquals(expected.get(i)[0], tracking.prefetchCalls.get(i).offset());
                assertEquals(expected.get(i)[1], tracking.prefetchCalls.get(i).length());
            }
            tracking.close();
        }
    }

    // -----------------------------------------------------------------------------------------------
    // A bytes read still serves and still admits
    // -----------------------------------------------------------------------------------------------

    /**
     * Every miss of a {@code bytes} batch is staged, not turned into a blocking read; its bytes decode
     * exactly; and it is admitted. At the fp32 length, where a vector is wider than a block.
     */
    @SneakyThrows
    public void testABytesReadIsStagedAdmittedAndServedExactly() {
        assumeDirectIOWorksHere();
        writeFixture(5L, FP32_VECTOR_BYTES, 44);
        configure("bytes", LOW_CAP, 8, true);

        // Scattered far enough apart that every one of them is its own bytes group.
        final int[] batch = { 10, 100, 200, 300, 400, 500, 600, 700 };
        try (DirectIOVectorIndexInput input = openAtCap(path)) {
            assertNotNull(input);
            final int[] copy = batch.clone();
            PrefetchHelper.prefetch(input, 44, FP32_VECTOR_BYTES, copy, copy.length);
            assertEquals("one group a miss", batch.length, (int) DirectIOVectorIndexInput.STATS.prefetchCalls());
            assertEquals("nothing declined", 0L, DirectIOVectorIndexInput.STATS.stageDeclined());

            final float[] scratch = new float[FP32_DIMENSION];
            for (final int ordinal : batch) {
                readVector(input, offsetOf(ordinal, FP32_VECTOR_BYTES, 44), scratch, FP32_DIMENSION);
                assertArrayEquals("a bytes read must decode exactly", vectors[ordinal], scratch, 0.0f);
            }
            assertEquals("every miss was staged", 0L, DirectIOVectorIndexInput.STATS.blockingReads());
            assertTrue("every requested vector is admitted", DirectIOVectorCache.entries() >= batch.length);

            // Read them again: every one is now served from the store, byte-exact.
            final long hitsBefore = DirectIOVectorCache.hits();
            for (final int ordinal : batch) {
                readVector(input, offsetOf(ordinal, FP32_VECTOR_BYTES, 44), scratch, FP32_DIMENSION);
                assertArrayEquals(vectors[ordinal], scratch, 0.0f);
            }
            assertEquals("and served from the store", hitsBefore + batch.length, DirectIOVectorCache.hits());
        }
    }

    /**
     * Span fill still admits the whole vectors inside a {@code bytes} read, so a neighbour nobody asked for
     * hits afterwards and hits byte-exact.
     *
     * <p>At the 1,536 B length, because that is where a one-vector {@code bytes} read has room for
     * neighbours: a 1,536 B vector at a 3,072 B offset into a block aligns to an 8,192 B window holding five
     * whole vectors, four of which nobody asked for. At 3,072 B the same window usually holds the requested
     * vector and nothing else whole, which is the profile working as designed and leaves nothing for this
     * test to observe.
     */
    @SneakyThrows
    public void testSpanFillStillAdmitsTheNeighboursInsideABytesRead() {
        assumeDirectIOWorksHere();
        writeFixture(6L, NARROW_VECTOR_BYTES, 0);
        configure("bytes", LOW_CAP, 8, true);
        final int dimension = NARROW_VECTOR_BYTES / Float.BYTES;

        final int[] batch = { 10, 110, 210, 310, 410, 510, 610, 710 };
        try (DirectIOVectorIndexInput input = openAtCap(path)) {
            assertNotNull(input);
            assumeTrue("this filesystem's block is not 4 KiB", input.blockSize() == 4096);
            final int[] copy = batch.clone();
            PrefetchHelper.prefetch(input, 0, NARROW_VECTOR_BYTES, copy, copy.length);
            assertEquals("one group a miss", batch.length, (int) DirectIOVectorIndexInput.STATS.prefetchCalls());

            final float[] scratch = new float[dimension];
            for (final int ordinal : batch) {
                readVector(input, offsetOf(ordinal, NARROW_VECTOR_BYTES, 0), scratch, dimension);
                assertArrayEquals(vectors[ordinal], scratch, 0.0f);
            }
            assertEquals("every miss was staged", 0L, DirectIOVectorIndexInput.STATS.blockingReads());
            assertTrue("span fill ran inside the bytes reads", DirectIOVectorCache.spanFills() > batch.length);

            int neighbour = -1;
            for (int candidate = 0; candidate < VECTOR_COUNT && neighbour < 0; candidate++) {
                if (Arrays.binarySearch(copy, 0, copy.length, candidate) < 0
                    && DirectIOVectorCache.contains(fileId(), offsetOf(candidate, NARROW_VECTOR_BYTES, 0), NARROW_VECTOR_BYTES)) {
                    neighbour = candidate;
                }
            }
            assertTrue("span fill must have admitted a vector nobody asked for", neighbour >= 0);
            final long hitsBefore = DirectIOVectorCache.hits();
            readVector(input, offsetOf(neighbour, NARROW_VECTOR_BYTES, 0), scratch, dimension);
            assertArrayEquals("a span-filled neighbour must be byte-exact", vectors[neighbour], scratch, 0.0f);
            assertEquals("and it is served from the store", hitsBefore + 1, DirectIOVectorCache.hits());
        }
    }

    // -----------------------------------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------------------------------

    /** How many prefetch calls the production helper makes for {@code ords}. */
    @SneakyThrows
    private int groupsFor(final DirectIOVectorIndexInput input, final int vectorBytes, final int[] ords) {
        final long before = DirectIOVectorIndexInput.STATS.prefetchCalls();
        final int[] copy = ords.clone();
        PrefetchHelper.prefetch(input, headerLength, vectorBytes, copy, copy.length);
        return (int) (DirectIOVectorIndexInput.STATS.prefetchCalls() - before);
    }

    /** P1's grouping: 128 KB and the input's span cap, and nothing about blocks. */
    private static int p1Groups(final DirectIOVectorIndexInput input, final int vectorBytes, final long baseOffset, final int[] ords) {
        final int[] sorted = ords.clone();
        Arrays.sort(sorted);
        final long cap = input.maxStagedRangeBytes();
        int groups = 1;
        long groupStart = baseOffset + (long) sorted[0] * vectorBytes;
        for (int i = 1; i < sorted.length; i++) {
            final long end = baseOffset + (long) sorted[i] * vectorBytes + vectorBytes;
            if (end - groupStart > BYTES_128 || input.alignedSpanBytes(groupStart, end - groupStart) > cap) {
                groups++;
                groupStart = baseOffset + (long) sorted[i] * vectorBytes;
            }
        }
        return groups;
    }

    /** The 128 KB grouping every input that is not a Direct I/O input gets. {@code {offset, length}} a group. */
    private static List<long[]> unsplitGrouping(final long baseOffset, final long vectorBytes, final int[] ords) {
        final int[] sorted = ords.clone();
        Arrays.sort(sorted);
        final List<long[]> groups = new ArrayList<>();
        long groupStart = baseOffset + (long) sorted[0] * vectorBytes;
        long groupEnd = groupStart + vectorBytes;
        for (int i = 1; i < sorted.length; i++) {
            final long current = baseOffset + (long) sorted[i] * vectorBytes;
            if ((current + vectorBytes) - groupStart > BYTES_128) {
                groups.add(new long[] { groupStart, groupEnd - groupStart });
                groupStart = current;
            }
            groupEnd = current + vectorBytes;
        }
        groups.add(new long[] { groupStart, groupEnd - groupStart });
        return groups;
    }

    private static int[] consecutive(final int first, final int count) {
        final int[] ords = new int[count];
        for (int i = 0; i < count; i++) {
            ords[i] = first + i;
        }
        return ords;
    }

    private static long offsetOf(final int ordinal, final int vectorBytes, final int header) {
        return header + (long) ordinal * vectorBytes;
    }

    @SneakyThrows
    private String fileId() {
        return DirectIOVectorCache.fileId(path, Files.size(path));
    }

    @SneakyThrows
    private void readVector(final DirectIOVectorIndexInput input, final long at, final float[] into, final int dimension) {
        input.seek(at);
        input.readFloats(into, 0, dimension);
    }

    @SneakyThrows
    private TrackingIndexInput trackingInput(final int sizeInBytes) {
        final ByteBuffersDirectory dir = new ByteBuffersDirectory();
        try (IndexOutput out = dir.createOutput("tracking.vec", IOContext.DEFAULT)) {
            out.writeBytes(new byte[sizeInBytes], sizeInBytes);
        }
        return new TrackingIndexInput(dir.openInput("tracking.vec", IOContext.DEFAULT));
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
        when(clusterService.getSettings()).thenReturn(routed);
        KNNSettings.state().setClusterService(clusterService);
        // The seeding a node does at startup: the dynamic Direct I/O settings live in volatile fields of
        // the components that own them, so publishing is how a setting reaches the read path.
        KNNSettings.state().publishDirectIOSettings();
    }

    /** The store off: the geometry under test is then the grouping's and nothing else. */
    private void configure(final String profile, final int cap) {
        configure(profile, cap, 0, false);
    }

    private void configure(final String profile, final int cap, final int slabs, final boolean spanFill) {
        final long maxBytes = (long) slabs * TEST_SLAB_BYTES;
        capBytes = cap;
        withNodeSettings(
            Settings.builder()
                .put(KNNSettings.KNN_DIRECT_IO_RESCORE_READ_PROFILE, profile)
                .put(KNNSettings.KNN_DIRECT_IO_RESCORE_CACHE_MAX_BYTES, maxBytes + "b")
                .put(KNNSettings.KNN_DIRECT_IO_READAHEAD_WINDOW, 0)
                .build()
        );
        DirectIOVectorCache.resetForTesting();
        DirectIOVectorCache.setSpanFillForTesting(spanFill);
        DirectIOVectorCache.setReadAroundBytesForTesting(0L);
        DirectIOVectorCache.setSlabBytesForTesting(TEST_SLAB_BYTES);
        DirectIOVectorCache.publishMaxBytes(maxBytes);
        DirectIOVectorIndexInput.STATS.reset();
        DirectIOVectorIndexInput.resetReadProfileForTesting();
    }

    /**
     * An input opened at the span cap the arm is sweeping. The cap is a constant on a node, so a sweep of it
     * goes through the sizing overload, which is what that overload exists for.
     */
    private DirectIOVectorIndexInput openAtCap(final Path file) {
        return DirectIOVectorIndexInput.open(file, 0, DirectIOVectorIndexInput.DEFAULT_MAX_STAGED_RANGES, capBytes, 0, 0, 2);
    }

    @SneakyThrows
    private void writeFixture(final long seed, final int vectorBytes, final int header) {
        final int dimension = vectorBytes / Float.BYTES;
        this.headerLength = header;
        final java.util.Random random = new java.util.Random(seed);
        vectors = new float[VECTOR_COUNT][dimension];
        for (final float[] vector : vectors) {
            for (int i = 0; i < dimension; i++) {
                vector[i] = (random.nextFloat() - 0.5f) * 8.0f;
            }
        }
        path = createTempDir().resolve("p2-" + seed + ".vec");
        final ByteBuffer buffer = ByteBuffer.allocate(header + VECTOR_COUNT * vectorBytes + 16).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < header; i++) {
            buffer.put((byte) (0xA0 + (i & 0x0F)));
        }
        for (final float[] vector : vectors) {
            for (final float value : vector) {
                buffer.putFloat(value);
            }
        }
        for (int i = 0; i < 16; i++) {
            buffer.put((byte) (0xF0 + (i & 0x0F)));
        }
        Files.write(path, buffer.array());
    }
}

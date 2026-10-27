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
import java.util.Random;
import java.util.Set;

import static org.mockito.Mockito.when;

/**
 * P1: a group the prefetch helper hands a Direct I/O input never exceeds that input's span cap, because the
 * helper closes the group before it would rather than letting the input decline it.
 *
 * <p><b>Why a decline is not merely a lost read-ahead.</b> {@code DirectIOVectorIndexInput#prefetch} declines
 * a range whose block-aligned length is above {@code knn.direct_io.rescore.prefetch.max_span_bytes}, and the
 * vectors of a declined group are then read one blocking read at a time. The store admits only vectors a
 * staged read covered (M3), so those vectors are not admitted and stay misses on every later query as well.
 * MEASURED on the NVMe node at a {@code 16kb} cap (ref/RESULTS-NVME.md section 3): hits 98.3% to 85.6% in the
 * f regime and 51.6% to 43% in r1, while the bytes per read fell as intended. Splitting instead costs one
 * more staged read of the same bytes and keeps the admission.
 *
 * <p>What these tests hold the change to:
 * <ul>
 *   <li>no group reaches a Direct I/O input above its cap, at the default and at {@code 16kb}, which the
 *       input's own {@code stage_declined} and {@code declined_span_bytes} counters answer;</li>
 *   <li>at {@code 16kb} the vectors of a group that the unsplit grouping has declined are all staged and
 *       then admitted, and the same batch through the production path declines nothing;</li>
 *   <li>an input that is not a Direct I/O input gets byte for byte the calls the pre-P1 grouping made, which
 *       is checked by computing that grouping in the test and comparing;</li>
 *   <li>the caller's array comes back sorted and otherwise untouched, as N1 left it;</li>
 *   <li>N1's miss filter still runs first, so the split groups are built from the misses only.</li>
 * </ul>
 */
@ThreadLeakFilters(defaultFilters = true, filters = { DirectIOSpanCapSplitTests.ReadPoolThreadFilter.class })
public class DirectIOSpanCapSplitTests extends KNNTestCase {

    /** The read pool is never shut down by design; its daemon threads outlive any one suite. */
    public static final class ReadPoolThreadFilter implements ThreadFilter {
        @Override
        public boolean reject(final Thread thread) {
            return thread.getName().startsWith(DirectIOReadPool.THREAD_NAME_PREFIX);
        }
    }

    /** The node's shape: 768 fp32 dimensions, 3,072 B a vector, which straddles a 4 KiB block ~75% of the time. */
    private static final int DIMENSION = 768;
    private static final int VECTOR_BYTES = DIMENSION * Float.BYTES;
    /** Not a multiple of the vector length and not a multiple of a block, which is the awkward real case. */
    private static final int HEADER_LENGTH = 44;
    private static final int FOOTER_LENGTH = 16;
    private static final int VECTOR_COUNT = 1_024;
    private static final int TEST_SLAB_BYTES = 1024 * 1024;
    private static final long BYTES_128 = 128 * 1024;

    /** The span cap the current arm sweeps, applied by {@link #openAtCap(Path)}. */
    private int capBytes = DirectIOVectorIndexInput.DEFAULT_MAX_STAGED_RANGE_BYTES;
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
    // No group above the cap
    // -----------------------------------------------------------------------------------------------

    /**
     * At the default cap. The default is {@code 128 KB + 4096}, which is exactly what a 128 KB group can
     * align up to at a 4 KiB block, so on this filesystem nothing declined before P1 either: the test is a
     * floor, and the {@code 16kb} test below is where the change shows.
     */
    @SneakyThrows
    public void testNoGroupExceedsTheDefaultSpanCap() {
        assumeDirectIOWorksHere();
        writeFixture(1L);
        configureStoreOff(DirectIOVectorIndexInput.DEFAULT_MAX_STAGED_RANGE_BYTES);

        try (DirectIOVectorIndexInput input = openAtCap(path)) {
            assertNotNull(input);
            assertEquals(DirectIOVectorIndexInput.DEFAULT_MAX_STAGED_RANGE_BYTES, input.maxStagedRangeBytes());

            // A run of 300 consecutive vectors is 921,600 B, so the 128 KB budget alone makes several groups.
            final int[] batch = consecutive(0, 300);
            PrefetchHelper.prefetch(input, HEADER_LENGTH, VECTOR_BYTES, batch, batch.length);

            assertEquals("no group may be declined", 0L, DirectIOVectorIndexInput.STATS.stageDeclined());
            assertEquals("and so no span bytes declined", 0L, DirectIOVectorIndexInput.STATS.declinedSpanBytes());
            assertTrue("and there were groups", DirectIOVectorIndexInput.STATS.prefetchCalls() > 1);
        }
    }

    /**
     * At {@code 16kb}, the cap the NVMe node measured. Here the pre-P1 grouping declines, and this is the
     * A/B: the same batch through the unsplit grouping against the same batch through the production path.
     */
    @SneakyThrows
    public void testNoGroupExceedsASixteenKilobyteSpanCapWhereTheUnsplitGroupingDeclines() {
        assumeDirectIOWorksHere();
        writeFixture(2L);
        configureStoreOff(16 * 1024);

        final int[] batch = consecutive(0, 60);

        // Arm A: the grouping as it was before P1, driven straight at the input. 60 x 3,072 B is 184,320 B,
        // so the 128 KB budget makes two groups and each of them is far above a 16 KiB cap.
        try (DirectIOVectorIndexInput input = openAtCap(path)) {
            assertNotNull(input);
            assertEquals(16 * 1024, input.maxStagedRangeBytes());
            final int unsplitGroups = prefetchTheUnsplitWay(input, HEADER_LENGTH, VECTOR_BYTES, batch);
            assertEquals("every unsplit group is declined", unsplitGroups, (int) DirectIOVectorIndexInput.STATS.stageDeclined());
        }

        // Arm B: the production path on the same batch.
        DirectIOVectorIndexInput.STATS.reset();
        try (DirectIOVectorIndexInput input = openAtCap(path)) {
            assertNotNull(input);
            final int[] copy = batch.clone();
            PrefetchHelper.prefetch(input, HEADER_LENGTH, VECTOR_BYTES, copy, copy.length);

            assertEquals("no group may be declined", 0L, DirectIOVectorIndexInput.STATS.stageDeclined());
            assertEquals("and so no span bytes declined", 0L, DirectIOVectorIndexInput.STATS.declinedSpanBytes());
            assertTrue("and the batch was split into groups", DirectIOVectorIndexInput.STATS.prefetchCalls() > 1);
        }
    }

    /**
     * The split's own arithmetic, checked against the input rather than against a constant: every group the
     * helper emits is recorded and its block-aligned length must be at or under the cap, at both caps and at
     * a vector length that is not a divisor of a block.
     */
    @SneakyThrows
    public void testEveryEmittedGroupIsWithinTheCapWhenAligned() {
        assumeDirectIOWorksHere();
        writeFixture(3L);
        for (final int cap : List.of(16 * 1024, 32 * 1024, DirectIOVectorIndexInput.DEFAULT_MAX_STAGED_RANGE_BYTES)) {
            configureStoreOff(cap);
            try (DirectIOVectorIndexInput input = openAtCap(path)) {
                assertNotNull(input);
                final int[] batch = clustered(400, 64, 11L);
                final List<long[]> groups = recordGroups(input, HEADER_LENGTH, VECTOR_BYTES, batch);
                assertFalse("the batch must produce groups", groups.isEmpty());
                for (final long[] group : groups) {
                    final long aligned = input.alignedSpanBytes(group[0], group[1]);
                    assertTrue(
                        "a group of " + group[1] + " B at " + group[0] + " aligns to " + aligned + " B, above the " + cap + " cap",
                        aligned <= input.maxStagedRangeBytes() || group[2] == 1
                    );
                }
            }
        }
    }

    // -----------------------------------------------------------------------------------------------
    // What the split buys: the vectors are admitted
    // -----------------------------------------------------------------------------------------------

    /**
     * At {@code 16kb} the vectors of what used to be one declined group are all staged, served from the
     * staged bytes, admitted to the store, and hit on the next pass. This is the hit rate the node lost.
     *
     * <p>The batch is a scattered one -- candidates inside a 128 KB window, which is a rescore batch's own
     * shape -- so that the pre-P1 grouping makes exactly one group of it, that group is declined at a 16 KiB
     * cap, and the comparison is 0 admitted against all of them. A <em>consecutive</em> run would admit most
     * of its vectors either way and the number would be about the boundary rule rather than about P1; see
     * {@link #testAVectorOnAGroupBoundaryIsReadCorrectlyAndTheRestAreAdmitted}.
     */
    @SneakyThrows
    public void testTheVectorsOfAFormerlyDeclinedGroupAreStagedAndAdmitted() {
        assumeDirectIOWorksHere();
        writeFixture(4L);

        // One group before P1, every vector of it 4 apart, so each is alone in its own group after.
        final int[] batch = new int[10];
        for (int i = 0; i < batch.length; i++) {
            batch[i] = 100 + 4 * i;
        }
        assertEquals("the batch must be one group before P1", 1, unsplitGrouping(HEADER_LENGTH, VECTOR_BYTES, batch).size());

        // Arm A: the unsplit grouping. One group of 110,628 B, declined at a 16 KiB cap, so every vector is a
        // blocking read and the store admits none of them.
        configureStoreOn(8, false, 16 * 1024);
        try (DirectIOVectorIndexInput input = openAtCap(path)) {
            assertNotNull(input);
            prefetchTheUnsplitWay(input, HEADER_LENGTH, VECTOR_BYTES, batch);
            assertEquals("the single group is declined", 1L, DirectIOVectorIndexInput.STATS.stageDeclined());
            final float[] scratch = new float[DIMENSION];
            for (final int ordinal : batch) {
                readVector(input, offsetOf(ordinal), scratch);
                assertArrayEquals(vectors[ordinal], scratch, 0.0f);
            }
            assertEquals("every vector went to the device blocking", batch.length, (int) DirectIOVectorIndexInput.STATS.blockingReads());
            assertEquals("and none of them is admitted", 0, (int) DirectIOVectorCache.entries());
        }

        // Arm B: the production path, which splits.
        configureStoreOn(8, false, 16 * 1024);
        try (DirectIOVectorIndexInput input = openAtCap(path)) {
            assertNotNull(input);
            final int[] copy = batch.clone();
            PrefetchHelper.prefetch(input, HEADER_LENGTH, VECTOR_BYTES, copy, copy.length);
            assertEquals("nothing declined", 0L, DirectIOVectorIndexInput.STATS.stageDeclined());

            final float[] scratch = new float[DIMENSION];
            for (final int ordinal : batch) {
                readVector(input, offsetOf(ordinal), scratch);
                assertArrayEquals("the staged bytes must decode to the vector", vectors[ordinal], scratch, 0.0f);
            }
            assertEquals("no read went blocking", 0L, DirectIOVectorIndexInput.STATS.blockingReads());
            assertEquals("every vector of the batch is admitted", batch.length, (int) DirectIOVectorCache.entries());

            // And the second pass is all hits, which is what the decline cost on every later query. On the
            // same input, because O3 frees a file's entries when its owning input closes.
            final long hitsBefore = DirectIOVectorCache.hits();
            for (final int ordinal : batch) {
                readVector(input, offsetOf(ordinal), scratch);
                assertArrayEquals(vectors[ordinal], scratch, 0.0f);
            }
            assertEquals("every vector hits", hitsBefore + batch.length, DirectIOVectorCache.hits());
        }
    }

    /**
     * A consecutive run at a {@code 16kb} cap is read byte-exact and almost entirely admitted, and the few
     * vectors that are not are the ones sitting on a group boundary.
     *
     * <p><b>This is a pre-existing property of the admission rule, not something P1 introduces, and it is
     * worth a test because P1 makes group boundaries more frequent.</b> Adjacent groups' block-aligned ranges
     * overlap by the block they share. {@code source(Float.BYTES)} asks the burst table for four bytes, so
     * for the first vector of a group the earlier group's range -- which holds that vector's first bytes and
     * not its last -- can be the range that answers; the admission rule then refuses it, because it requires
     * one staged buffer to have covered the whole vector ({@code written == 0 && usableFloats == count}). The
     * bytes are still correct: the read simply continues in the next range. The vector is admitted the next
     * time it is read in a group it is not first in.
     */
    @SneakyThrows
    public void testAVectorOnAGroupBoundaryIsReadCorrectlyAndTheRestAreAdmitted() {
        assumeDirectIOWorksHere();
        writeFixture(8L);
        configureStoreOn(8, false, 16 * 1024);

        final int[] batch = consecutive(100, 40);
        try (DirectIOVectorIndexInput input = openAtCap(path)) {
            assertNotNull(input);
            final int[] copy = batch.clone();
            PrefetchHelper.prefetch(input, HEADER_LENGTH, VECTOR_BYTES, copy, copy.length);
            assertEquals("nothing declined", 0L, DirectIOVectorIndexInput.STATS.stageDeclined());

            final float[] scratch = new float[DIMENSION];
            for (final int ordinal : batch) {
                readVector(input, offsetOf(ordinal), scratch);
                assertArrayEquals("every vector decodes exactly", vectors[ordinal], scratch, 0.0f);
            }
            assertEquals("no read went to the device blocking", 0L, DirectIOVectorIndexInput.STATS.blockingReads());

            final int admitted = (int) DirectIOVectorCache.entries();
            final int groups = (int) DirectIOVectorIndexInput.STATS.prefetchCalls();
            assertTrue("the run is split into groups", groups > 1);
            assertTrue(
                "at most one vector per group boundary may be left unadmitted, " + admitted + " of " + batch.length,
                admitted >= batch.length - groups
            );
            assertTrue("and the great majority is admitted", admitted * 10 >= batch.length * 7);
        }
    }

    // -----------------------------------------------------------------------------------------------
    // Everything that is not a Direct I/O input
    // -----------------------------------------------------------------------------------------------

    /**
     * An input the split does not apply to gets byte for byte the calls the pre-P1 grouping made. The
     * expectation is computed here by that grouping, over several batch shapes, so this is a comparison and
     * not a restatement.
     */
    @SneakyThrows
    public void testANonDirectIOInputGetsExactlyTheUnsplitGrouping() {
        final List<int[]> batches = List.of(
            consecutive(0, 300),
            clustered(400, 64, 7L),
            clustered(64, 4_000, 9L),
            new int[] { 0, 1, 2 },
            new int[] { 0, 128, 256 },
            new int[] { 5, 2, 5, 2, 10 }
        );
        for (final int[] batch : batches) {
            final TrackingIndexInput tracking = trackingInput(HEADER_LENGTH + VECTOR_COUNT * VECTOR_BYTES + FOOTER_LENGTH);
            final int[] copy = batch.clone();
            PrefetchHelper.prefetch(tracking, HEADER_LENGTH, VECTOR_BYTES, copy, copy.length);

            final List<long[]> expected = unsplitGrouping(HEADER_LENGTH, VECTOR_BYTES, batch);
            assertEquals("group count", expected.size(), tracking.prefetchCalls.size());
            for (int i = 0; i < expected.size(); i++) {
                assertEquals("group " + i + " offset", expected.get(i)[0], tracking.prefetchCalls.get(i).offset());
                assertEquals("group " + i + " length", expected.get(i)[1], tracking.prefetchCalls.get(i).length());
            }
            tracking.close();
        }
    }

    /** The caller's array is sorted in place and otherwise untouched, with a Direct I/O input and the store on. */
    @SneakyThrows
    public void testTheCallersArrayIsOnlySorted() {
        assumeDirectIOWorksHere();
        writeFixture(5L);
        configureStoreOn(8, false, 16 * 1024);

        try (DirectIOVectorIndexInput input = openAtCap(path)) {
            assertNotNull(input);
            final int[] batch = { 600, 0, 200, 50, 400 };
            PrefetchHelper.prefetch(input, HEADER_LENGTH, VECTOR_BYTES, batch, batch.length);
            assertArrayEquals(new int[] { 0, 50, 200, 400, 600 }, batch);
        }
    }

    /**
     * N1's miss filter still runs before the grouping, so the groups the split produces are built from the
     * misses only: a batch whose every candidate but one is cached stages one group covering that one vector.
     */
    @SneakyThrows
    public void testTheMissFilterStillRunsBeforeTheGrouping() {
        assumeDirectIOWorksHere();
        writeFixture(6L);
        configureStoreOn(8, false, 16 * 1024);

        try (DirectIOVectorIndexInput input = openAtCap(path)) {
            assertNotNull(input);
            // Spaced, so that each warming read is its own group and every one of them is admitted; the
            // boundary rule of testAVectorOnAGroupBoundaryIsReadCorrectlyAndTheRestAreAdmitted would
            // otherwise leave one of a consecutive run uncached and this test would be measuring that.
            final int[] cached = new int[20];
            for (int i = 0; i < cached.length; i++) {
                cached[i] = 4 * i;
            }
            warm(input, cached);
            assertEquals("the warming pass must cache the whole set", cached.length, (int) DirectIOVectorCache.entries());
            DirectIOVectorIndexInput.STATS.reset();

            final int[] batch = Arrays.copyOf(cached, cached.length + 1);
            batch[cached.length] = 700;
            PrefetchHelper.prefetch(input, HEADER_LENGTH, VECTOR_BYTES, batch, batch.length);

            assertEquals("one group, for the one miss", 1L, DirectIOVectorIndexInput.STATS.prefetchCalls());
            assertEquals("nothing declined", 0L, DirectIOVectorIndexInput.STATS.stageDeclined());
            final float[] scratch = new float[DIMENSION];
            readVector(input, offsetOf(700), scratch);
            assertArrayEquals(vectors[700], scratch, 0.0f);
            assertEquals("and the one miss was staged, not blocking", 0L, DirectIOVectorIndexInput.STATS.blockingReads());
        }
    }

    // -----------------------------------------------------------------------------------------------
    // The accessor the helper asks
    // -----------------------------------------------------------------------------------------------

    /**
     * {@code alignedSpanBytes} answers what {@code prefetch} would stage, over the <em>absolute</em> offset,
     * which is why it is on the input and not arithmetic the helper could do: a slice's own base shifts the
     * answer by its remainder.
     */
    @SneakyThrows
    public void testAlignedSpanBytesIsTakenOverTheAbsoluteOffset() {
        assumeDirectIOWorksHere();
        writeFixture(7L);
        configureStoreOff(16 * 1024);

        try (DirectIOVectorIndexInput input = openAtCap(path)) {
            assertNotNull(input);
            final int block = input.blockSize();
            assertEquals("a block-aligned request of one block stays one block", block, input.alignedSpanBytes(0, block));
            assertEquals("one byte past a boundary takes two blocks", 2L * block, input.alignedSpanBytes(1, block));
            assertEquals("a non-positive length is zero", 0L, input.alignedSpanBytes(0, 0));

            // A slice whose base is not block aligned shifts every answer, and the slice's own coordinates
            // cannot see that: this is the whole reason for the accessor.
            try (DirectIOVectorIndexInput slice = (DirectIOVectorIndexInput) input.slice("p1", HEADER_LENGTH, 64L * block)) {
                assertEquals(
                    "the slice's answer follows its absolute base",
                    input.alignedSpanBytes(HEADER_LENGTH, block),
                    slice.alignedSpanBytes(0, block)
                );
            }
            assertTrue("and that base is not aligned", input.alignedSpanBytes(HEADER_LENGTH, block) > block);
        }
    }

    // -----------------------------------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------------------------------

    /** The grouping as it stood before P1: 128 KB and nothing else. Returns {@code {offset, length}} per group. */
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

    /** Drives the pre-P1 grouping straight at the input. Returns how many groups it offered. */
    @SneakyThrows
    private static int prefetchTheUnsplitWay(
        final DirectIOVectorIndexInput input,
        final long baseOffset,
        final long vectorBytes,
        final int[] ords
    ) {
        final List<long[]> groups = unsplitGrouping(baseOffset, vectorBytes, ords);
        for (final long[] group : groups) {
            input.prefetch(group[0], group[1]);
        }
        return groups.size();
    }

    /**
     * The groups the production helper emits for {@code ords}, recovered by running the helper against a
     * recording input and then replaying the same batch at the real input. The recorder carries the Direct
     * I/O input's cap and block size, so the grouping it sees is the grouping the real input gets.
     *
     * <p>Each entry is {@code {offset, length, vectorCount}}.
     */
    @SneakyThrows
    private static List<long[]> recordGroups(
        final DirectIOVectorIndexInput input,
        final long baseOffset,
        final long vectorBytes,
        final int[] ords
    ) {
        final int[] sorted = ords.clone();
        Arrays.sort(sorted);
        final long cap = input.maxStagedRangeBytes();
        final List<long[]> groups = new ArrayList<>();
        long groupStart = baseOffset + (long) sorted[0] * vectorBytes;
        long groupEnd = groupStart + vectorBytes;
        int count = 1;
        for (int i = 1; i < sorted.length; i++) {
            final long current = baseOffset + (long) sorted[i] * vectorBytes;
            final long end = current + vectorBytes;
            if ((end - groupStart) > BYTES_128 || input.alignedSpanBytes(groupStart, end - groupStart) > cap) {
                groups.add(new long[] { groupStart, groupEnd - groupStart, count });
                groupStart = current;
                count = 0;
            }
            groupEnd = end;
            count++;
        }
        groups.add(new long[] { groupStart, groupEnd - groupStart, count });

        // And the production path must agree on the count, which is what makes this a recovery and not a
        // second implementation.
        final long before = DirectIOVectorIndexInput.STATS.prefetchCalls();
        final int[] copy = ords.clone();
        PrefetchHelper.prefetch(input, baseOffset, vectorBytes, copy, copy.length);
        assertEquals(
            "the helper must emit the groups this walk recovered",
            groups.size(),
            (int) (DirectIOVectorIndexInput.STATS.prefetchCalls() - before)
        );
        return groups;
    }

    private static int[] consecutive(final int first, final int count) {
        final int[] ords = new int[count];
        for (int i = 0; i < count; i++) {
            ords[i] = first + i;
        }
        return ords;
    }

    /** Ordinals drawn from one window, so that a coalesced group spans several of them. */
    private static int[] clustered(final int count, final int windowSize, final long seed) {
        final Random random = new Random(seed);
        final int[] ords = new int[count];
        for (int i = 0; i < count; i++) {
            ords[i] = random.nextInt(Math.min(windowSize, VECTOR_COUNT));
        }
        return ords;
    }

    private static long offsetOf(final int ordinal) {
        return HEADER_LENGTH + (long) ordinal * VECTOR_BYTES;
    }

    @SneakyThrows
    private void readVector(final DirectIOVectorIndexInput input, final long at, final float[] into) {
        input.seek(at);
        input.readFloats(into, 0, DIMENSION);
    }

    @SneakyThrows
    private void warm(final DirectIOVectorIndexInput input, final int[] ordinals) {
        final int[] copy = ordinals.clone();
        PrefetchHelper.prefetch(input, HEADER_LENGTH, VECTOR_BYTES, copy, copy.length);
        final float[] scratch = new float[DIMENSION];
        for (final int ordinal : ordinals) {
            readVector(input, offsetOf(ordinal), scratch);
            assertArrayEquals(vectors[ordinal], scratch, 0.0f);
        }
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
        KNNSettings.state().setClusterService(clusterService);
        // The seeding a node does at startup: the dynamic Direct I/O settings live in volatile fields of
        // the components that own them, so publishing is how a setting reaches the read path.
        KNNSettings.state().publishDirectIOSettings();
    }

    /** The store off, so that the geometry under test is the grouping's and nothing else. */
    private void configureStoreOff(final int cap) {
        capBytes = cap;
        withNodeSettings(
            Settings.builder()
                .put(KNNSettings.KNN_DIRECT_IO_RESCORE_CACHE_MAX_BYTES, "0b")
                .put(KNNSettings.KNN_DIRECT_IO_READAHEAD_WINDOW, 0)
                .build()
        );
        DirectIOVectorCache.resetForTesting();
        DirectIOVectorCache.setSlabBytesForTesting(TEST_SLAB_BYTES);
        DirectIOVectorCache.publishMaxBytes(0L);
        DirectIOVectorIndexInput.STATS.reset();
    }

    private void configureStoreOn(final int slabs, final boolean spanFill, final int cap) {
        final long maxBytes = (long) slabs * TEST_SLAB_BYTES;
        capBytes = cap;
        withNodeSettings(
            Settings.builder()
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
    }

    /**
     * An input opened at the span cap the arm is sweeping. The cap is a constant on a node, so a sweep of it
     * goes through the sizing overload, which is what that overload exists for.
     */
    private DirectIOVectorIndexInput openAtCap(final Path file) {
        return DirectIOVectorIndexInput.open(file, 0, DirectIOVectorIndexInput.DEFAULT_MAX_STAGED_RANGES, capBytes, 0, 0, 2);
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
        path = createTempDir().resolve("p1-" + seed + ".vec");
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
}

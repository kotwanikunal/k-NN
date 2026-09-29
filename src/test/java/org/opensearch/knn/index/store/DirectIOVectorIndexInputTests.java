/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import com.carrotsearch.randomizedtesting.annotations.ThreadLeakFilters;
import lombok.SneakyThrows;
import org.apache.lucene.codecs.hnsw.FlatVectorScorerUtil;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.codecs.lucene95.HasIndexSlice;
import org.apache.lucene.codecs.lucene95.OffHeapFloatVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FilterIndexInput;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.opensearch.common.settings.ClusterSettings;
import org.opensearch.common.settings.Setting;
import org.opensearch.common.settings.Settings;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.KNNSettings;

import java.io.EOFException;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.OpenOption;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;

import static org.mockito.Mockito.when;

/**
 * Phase 9b gate 2: whether an {@code O_DIRECT} {@link IndexInput} can carry read ahead by itself.
 *
 * <p>The seam reaches its queue depth through a query-layer scorer wrapper and a staging ring. A
 * {@code Directory}-based design has no such wrapper, so it can only match the seam if
 * {@link IndexInput#prefetch} — a no-op by default, and absent entirely from Lucene's
 * {@code DirectIOIndexInput} — can put reads in flight. These tests assert the four things that has to be
 * true, in increasing order of how much they decide:
 *
 * <ol>
 *   <li><b>Correctness first.</b> Every byte read through {@link DirectIOVectorIndexInput} is the byte an
 *       {@link MMapDirectory} input returns for the same position, with and without prefetch, in order and
 *       out of order.</li>
 *   <li><b>Direct I/O is inherited, not re-decided.</b> An {@code IOContext} does not survive
 *       {@code slice()}, so a slice and a clone must be {@code O_DIRECT} by construction.</li>
 *   <li><b>Prefetch really queues.</b> A burst of prefetches puts more than one device read in flight at
 *       once; without prefetch the depth is exactly one. This is the gate.</li>
 *   <li><b>It binds where it has to.</b> Wrapped in Lucene's own {@code OffHeapFloatVectorValues}, this
 *       input keeps {@code HasIndexSlice} (so the shipped prefetch path reaches it) while declining the
 *       memory-segment SIMD scorer (so {@code vectorValue(ord)} is actually called), and the scores it
 *       produces match the mmap path exactly.</li>
 * </ol>
 *
 * <p>Tests needing an open handle are skipped, not failed, where the filesystem refuses {@code O_DIRECT},
 * for the reason {@link DirectIOVectorSourceTests} gives. Staged reads run on {@link DirectIOReadPool},
 * whose daemon threads outlive a suite by design, hence the same thread filter.
 */
@ThreadLeakFilters(defaultFilters = true, filters = { DirectIOReadPoolTests.ReadPoolThreadFilter.class })
public class DirectIOVectorIndexInputTests extends KNNTestCase {

    private static final int DIMENSION = 768;
    private static final int VECTOR_BYTES = DIMENSION * Float.BYTES;
    /** Deliberately not a multiple of the block size or of {@code gcd(vectorBytes, blockSize)}. */
    private static final int HEADER_LENGTH = 44;

    @SneakyThrows
    private void assumeDirectIOWorksHere() {
        final OpenOption direct = DirectIOVectorSource.directOpenOption();
        assumeTrue("this JDK has no ExtendedOpenOption.DIRECT", direct != null);
        final Path probe = createTempDir().resolve("probe");
        Files.write(probe, new byte[8192]);
        try (FileChannel channel = FileChannel.open(probe, StandardOpenOption.READ, direct)) {
            assertNotNull(channel);
        } catch (IOException | UnsupportedOperationException e) {
            assumeNoException("this filesystem refuses O_DIRECT", e);
        }
    }

    /** A file shaped like Lucene's flat vector file: header, one vector region, a 16 byte footer. */
    @SneakyThrows
    private Path writeVectorFile(final Path dir, final float[][] vectors) {
        final Path path = dir.resolve("_0_Test_0.vec");
        final ByteBuffer buffer = ByteBuffer.allocate(HEADER_LENGTH + vectors.length * VECTOR_BYTES + 16).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < HEADER_LENGTH; i++) {
            buffer.put((byte) (0xC0 + i));
        }
        for (final float[] vector : vectors) {
            for (final float value : vector) {
                buffer.putFloat(value);
            }
        }
        buffer.put(new byte[16]);
        Files.write(path, buffer.array());
        return path;
    }

    private static float[][] randomVectors(final int count) {
        final Random random = new Random(20250929L);
        final float[][] vectors = new float[count][DIMENSION];
        for (int i = 0; i < count; i++) {
            for (int d = 0; d < DIMENSION; d++) {
                vectors[i][d] = random.nextFloat() * 200 - 100;
            }
        }
        return vectors;
    }

    /** The ordinals a rescore batch asks for: scattered, unsorted, and 64 of them like Lucene's batch. */
    private static int[] scatteredOrds(final int count, final int size) {
        final Random random = new Random(7L);
        final int[] ords = new int[count];
        for (int i = 0; i < count; i++) {
            ords[i] = random.nextInt(size);
        }
        return ords;
    }

    // -----------------------------------------------------------------------------------------------
    // 1. Correctness against mmap
    // -----------------------------------------------------------------------------------------------

    @SneakyThrows
    public void testEveryVectorReadsBackWhatMmapReturns() {
        assumeDirectIOWorksHere();
        final Path dir = createTempDir();
        final float[][] vectors = randomVectors(500);
        final Path path = writeVectorFile(dir, vectors);

        try (
            Directory mmap = new MMapDirectory(dir);
            IndexInput reference = mmap.openInput(path.getFileName().toString(), IOContext.DEFAULT);
            DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path)
        ) {
            assertNotNull("open should have succeeded on a regular file", input);
            assertEquals(Files.size(path), input.length());

            final float[] expected = new float[DIMENSION];
            final float[] actual = new float[DIMENSION];
            for (int ord = 0; ord < vectors.length; ord++) {
                final long at = HEADER_LENGTH + (long) ord * VECTOR_BYTES;
                reference.seek(at);
                reference.readFloats(expected, 0, DIMENSION);
                input.seek(at);
                input.readFloats(actual, 0, DIMENSION);
                assertArrayEquals("ordinal " + ord, vectors[ord], actual, 0.0f);
                assertArrayEquals("ordinal " + ord + " against mmap", expected, actual, 0.0f);
            }
        }
    }

    @SneakyThrows
    public void testReadsAreCorrectWithAndWithoutPrefetchAndInAnyOrder() {
        assumeDirectIOWorksHere();
        final Path dir = createTempDir();
        final float[][] vectors = randomVectors(4000);
        final Path path = writeVectorFile(dir, vectors);
        final int[] ords = scatteredOrds(64, vectors.length);

        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(input);
            final float[] actual = new float[DIMENSION];

            // A burst of prefetches in ascending file order, exactly as PrefetchHelper issues them, then
            // reads in the batch's own (unsorted) order. That mismatch is the whole reason the staging table
            // is keyed by file range rather than being a consume-once ring.
            final int[] sorted = ords.clone();
            java.util.Arrays.sort(sorted);
            for (final int ord : sorted) {
                input.prefetch(HEADER_LENGTH + (long) ord * VECTOR_BYTES, VECTOR_BYTES);
            }
            for (final int ord : ords) {
                input.seek(HEADER_LENGTH + (long) ord * VECTOR_BYTES);
                input.readFloats(actual, 0, DIMENSION);
                assertArrayEquals("prefetched ordinal " + ord, vectors[ord], actual, 0.0f);
            }

            // Then read ordinals nobody predicted. Staging is advisory in both directions, so these have to
            // be correct too.
            for (final int ord : new int[] { 0, vectors.length - 1, 1234, 77 }) {
                input.seek(HEADER_LENGTH + (long) ord * VECTOR_BYTES);
                input.readFloats(actual, 0, DIMENSION);
                assertArrayEquals("unpredicted ordinal " + ord, vectors[ord], actual, 0.0f);
            }
        }
    }

    @SneakyThrows
    public void testReadBytesAndReadByteCrossBufferBoundaries() {
        assumeDirectIOWorksHere();
        final Path dir = createTempDir();
        final float[][] vectors = randomVectors(40);
        final Path path = writeVectorFile(dir, vectors);
        final byte[] whole = Files.readAllBytes(path);

        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(input);
            // One bulk read spanning many buffers, which is the loop in readBytes.
            final byte[] bulk = new byte[whole.length - 16];
            input.seek(0);
            input.readBytes(bulk, 0, bulk.length);
            for (int i = 0; i < bulk.length; i++) {
                assertEquals("byte " + i, whole[i], bulk[i]);
            }
            // And a byte at a time across a block boundary.
            input.seek(4090);
            for (int i = 0; i < 12; i++) {
                assertEquals("byte " + (4090 + i), whole[4090 + i], input.readByte());
            }
        }
    }

    @SneakyThrows
    public void testReadPastEndOfInputThrows() {
        assumeDirectIOWorksHere();
        final Path dir = createTempDir();
        final Path path = writeVectorFile(dir, randomVectors(4));
        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(input);
            input.seek(input.length());
            expectThrows(EOFException.class, input::readByte);
            expectThrows(EOFException.class, () -> input.seek(input.length() + 1));
        }
    }

    // -----------------------------------------------------------------------------------------------
    // 2. Direct I/O is inherited by slices and clones
    // -----------------------------------------------------------------------------------------------

    @SneakyThrows
    public void testSliceAndCloneStayDirectIOAndReadIndependently() {
        assumeDirectIOWorksHere();
        final Path dir = createTempDir();
        final float[][] vectors = randomVectors(300);
        final Path path = writeVectorFile(dir, vectors);

        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(input);
            // The vector region, sliced the way Lucene's OffHeapFloatVectorValues.load slices it.
            final IndexInput region = input.slice("vector-data", HEADER_LENGTH, (long) vectors.length * VECTOR_BYTES);

            // This is the gate-2 sub-claim: there is no IOContext on this overload, so Direct I/O cannot be
            // re-decided per slice and must be a property of the object.
            assertTrue("a slice of a Direct I/O input must itself be one", region instanceof DirectIOVectorIndexInput);
            assertEquals((long) vectors.length * VECTOR_BYTES, region.length());

            final IndexInput copy = region.clone();
            assertTrue("a clone of a Direct I/O input must itself be one", copy instanceof DirectIOVectorIndexInput);

            // Independent cursors: the clone reading does not move the original, which is the contract
            // OffHeapFloatVectorValues.copy() relies on when it hands each scoring task its own view.
            final float[] fromRegion = new float[DIMENSION];
            final float[] fromCopy = new float[DIMENSION];
            region.seek(0);
            copy.seek((long) 299 * VECTOR_BYTES);
            region.readFloats(fromRegion, 0, DIMENSION);
            copy.readFloats(fromCopy, 0, DIMENSION);
            assertArrayEquals(vectors[0], fromRegion, 0.0f);
            assertArrayEquals(vectors[299], fromCopy, 0.0f);

            // A slice's prefetch offsets are its own, not the file's.
            copy.prefetch(0, VECTOR_BYTES);
            copy.seek(0);
            copy.readFloats(fromCopy, 0, DIMENSION);
            assertArrayEquals(vectors[0], fromCopy, 0.0f);

            // Closing a slice must not close the shared handle; the root still reads.
            region.close();
            copy.close();
            input.seek(HEADER_LENGTH);
            input.readFloats(fromRegion, 0, DIMENSION);
            assertArrayEquals(vectors[0], fromRegion, 0.0f);
        }
    }

    @SneakyThrows
    public void testIsNeitherAFilterIndexInputNorRandomAccess() {
        assumeDirectIOWorksHere();
        final Path dir = createTempDir();
        final Path path = writeVectorFile(dir, randomVectors(4));
        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(input);
            // Load bearing, and asserted rather than commented because a later refactor that "tidied" this
            // into a FilterIndexInput would silently rebind the SIMD scorer to an mmap delegate through
            // FilterIndexInput.unwrapOnlyTest and quietly stop calling this class. Written through the
            // IndexInput supertype because javac rejects the direct instanceof outright — this class is
            // final and unrelated to FilterIndexInput, which is the strongest form the check can take.
            final IndexInput asIndexInput = input;
            assertFalse("must not be a FilterIndexInput", asIndexInput instanceof FilterIndexInput);
            assertSame("must not unwrap to anything else", input, FilterIndexInput.unwrapOnlyTest(asIndexInput));
        }
    }

    // -----------------------------------------------------------------------------------------------
    // 3. The gate: prefetch actually queues reads
    // -----------------------------------------------------------------------------------------------

    /**
     * <b>Gate 2.</b> A burst of prefetches must put several device reads in flight at once; the same reads
     * without prefetch must be strictly serial.
     *
     * <p>Queue depth is measured where it is unambiguous — the high-water mark of reads concurrently inside
     * {@code FileChannel#read} on {@link DirectIOReadPool}, counted by
     * {@link DirectIOVectorIndexInput.Stats#inFlightPeak()}. Latency is deliberately not the assertion: on
     * a small temp file the device is a cache hit either way, and the question gate 2 asks is whether the
     * reads <em>overlap</em>, not how fast this filesystem is.
     */
    @SneakyThrows
    public void testPrefetchPutsSeveralReadsInFlightAndNoPrefetchPutsOne() {
        assumeDirectIOWorksHere();
        final Path dir = createTempDir();
        final float[][] vectors = randomVectors(20000);
        final Path path = writeVectorFile(dir, vectors);
        final int[] ords = scatteredOrds(64, vectors.length);
        final float[] scratch = new float[DIMENSION];

        // Arm A: no prefetch at all. Every read is issued on the calling thread, so the depth is one.
        // The pool is dropped first so that this reading does not depend on whichever test ran before it —
        // the executor is static, and the suite's order is randomized.
        DirectIOReadPool.resetForTesting();
        DirectIOVectorIndexInput.STATS.reset();
        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(input);
            for (final int ord : ords) {
                input.seek(HEADER_LENGTH + (long) ord * VECTOR_BYTES);
                input.readFloats(scratch, 0, DIMENSION);
            }
        }
        final long blockingOnly = DirectIOVectorIndexInput.STATS.blockingReads();
        assertEquals("no prefetch must stage nothing", 0L, DirectIOVectorIndexInput.STATS.stagedReads());
        assertEquals("no prefetch must offer no queue depth", 0L, DirectIOVectorIndexInput.STATS.inFlightPeak());
        assertTrue("every read should have gone to the device", blockingOnly >= 60);

        // Arm B: the same reads, preceded by the prefetch burst PrefetchHelper issues.
        DirectIOVectorIndexInput.STATS.reset();
        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(input);
            final int[] sorted = ords.clone();
            java.util.Arrays.sort(sorted);
            for (final int ord : sorted) {
                input.prefetch(HEADER_LENGTH + (long) ord * VECTOR_BYTES, VECTOR_BYTES);
            }
            for (final int ord : ords) {
                input.seek(HEADER_LENGTH + (long) ord * VECTOR_BYTES);
                input.readFloats(scratch, 0, DIMENSION);
                assertArrayEquals(vectors[ord], scratch, 0.0f);
            }
        }
        final long staged = DirectIOVectorIndexInput.STATS.stagedReads();
        final long hits = DirectIOVectorIndexInput.STATS.stagedHits();
        final long peak = DirectIOVectorIndexInput.STATS.inFlightPeak();
        final long blockingWithPrefetch = DirectIOVectorIndexInput.STATS.blockingReads();

        logger.info("gate2 arm A blockingReads={} | arm B {}", blockingOnly, DirectIOVectorIndexInput.STATS);

        assertTrue("the burst should have staged most of the batch, staged=" + staged, staged >= 32);
        assertTrue("reads should have been served from staged ranges, hits=" + hits, hits >= 32);
        // The claim gate 2 turns on. One is what a plain IndexInput over O_DIRECT gets, and what approach A
        // got when it lost the madvise-driven depth; anything well above one means the mechanism works.
        assertTrue("prefetch must offer real queue depth, peak=" + peak, peak > 1);
        assertTrue(
            "prefetch should have removed most blocking reads, blocking=" + blockingWithPrefetch,
            blockingWithPrefetch < blockingOnly
        );
    }

    /**
     * The same measurement over many bursts, because the first burst understates the depth badly.
     *
     * <p>{@link DirectIOReadPool} creates its threads on demand, so during the very first burst on a node
     * the submission loop is racing thread creation: a thread start and a warm {@code O_DIRECT} read of one
     * block are the same order of magnitude, and reads retire about as fast as workers appear. That is a real
     * property worth knowing about — the first rescore query after a pool goes idle for
     * {@link DirectIOReadPool#KEEP_ALIVE_SECONDS} pays it — but it is not the depth the mechanism offers.
     *
     * <p>The assertion stays at "deeper than serial" rather than at a number, because the number is a
     * property of this host's cores and device; the number itself is logged and recorded in the baton.
     */
    @SneakyThrows
    public void testQueueDepthIsDeeperOnceTheReadPoolIsWarm() {
        assumeDirectIOWorksHere();
        final Path dir = createTempDir();
        final float[][] vectors = randomVectors(20000);
        final Path path = writeVectorFile(dir, vectors);
        final float[] scratch = new float[DIMENSION];

        DirectIOReadPool.resetForTesting();
        DirectIOVectorIndexInput.STATS.reset();
        long firstBurstPeak = 0;
        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(input);
            for (int burst = 0; burst < 20; burst++) {
                final int[] ords = scatteredOrds(64, vectors.length);
                final int[] sorted = ords.clone();
                java.util.Arrays.sort(sorted);
                for (final int ord : sorted) {
                    input.prefetch(HEADER_LENGTH + (long) ord * VECTOR_BYTES, VECTOR_BYTES);
                }
                for (final int ord : ords) {
                    input.seek(HEADER_LENGTH + (long) ord * VECTOR_BYTES);
                    input.readFloats(scratch, 0, DIMENSION);
                    assertArrayEquals("burst " + burst + " ordinal " + ord, vectors[ord], scratch, 0.0f);
                }
                if (burst == 0) {
                    firstBurstPeak = DirectIOVectorIndexInput.STATS.inFlightPeak();
                }
            }
        }
        final long steadyPeak = DirectIOVectorIndexInput.STATS.inFlightPeak();
        logger.info(
            "gate2 steady-state over 20 bursts: firstBurstPeak={} peak={} | {}",
            firstBurstPeak,
            steadyPeak,
            DirectIOVectorIndexInput.STATS
        );
        assertTrue("20 bursts must offer queue depth above serial, peak=" + steadyPeak, steadyPeak > 1);
        assertTrue("a warm pool must not be shallower than the first burst", steadyPeak >= firstBurstPeak);
    }

    /**
     * A prefetch burst larger than the table declines the overflow rather than evicting a read that is still
     * in flight, and a range larger than the cap is declined outright. Both must still read correctly.
     */
    @SneakyThrows
    public void testOversizedBurstsAndRangesDeclineRatherThanMisread() {
        assumeDirectIOWorksHere();
        final Path dir = createTempDir();
        final float[][] vectors = randomVectors(2000);
        final Path path = writeVectorFile(dir, vectors);
        final float[] scratch = new float[DIMENSION];

        DirectIOVectorIndexInput.STATS.reset();
        // A table of four, offered sixteen ranges, and a cap smaller than one whole prefetch request.
        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path, 8192, 4, 8192)) {
            assertNotNull(input);
            for (int ord = 0; ord < 16; ord++) {
                input.prefetch(HEADER_LENGTH + (long) ord * VECTOR_BYTES, VECTOR_BYTES);
            }
            // One range far larger than the cap.
            input.prefetch(HEADER_LENGTH, 1024L * 1024L);
            for (int ord = 0; ord < 16; ord++) {
                input.seek(HEADER_LENGTH + (long) ord * VECTOR_BYTES);
                input.readFloats(scratch, 0, DIMENSION);
                assertArrayEquals("ordinal " + ord, vectors[ord], scratch, 0.0f);
            }
        }
        assertTrue("the overflow and the oversized range should have been declined", DirectIOVectorIndexInput.STATS.stageDeclined() > 0);
    }

    // -----------------------------------------------------------------------------------------------
    // 4. It binds correctly inside Lucene's own values
    // -----------------------------------------------------------------------------------------------

    /**
     * Wrapped in Lucene's {@code DenseOffHeapVectorValues}, this input must produce the same scores as the
     * mmap path while keeping {@code HasIndexSlice} and declining the memory-segment SIMD scorer.
     *
     * <p>That combination is the structural reason a directory design is better placed than the seam. The
     * seam had to hide {@code HasIndexSlice} from the values to keep the SIMD scorer from binding straight
     * to a mapping — and lost the shipped prefetch path with it, which is why it built its own. Here the
     * narrower {@code MemorySegmentAccessInput} is what the SIMD scorer actually tests
     * ({@code Lucene99MemorySegmentFloatVectorScorer.create} returns an empty {@code Optional} otherwise),
     * so the slice can stay visible for prefetch and still fall back to the scalar scorer that calls
     * {@code vectorValue(ord)}.
     */
    @SneakyThrows
    public void testScoresMatchMmapThroughLuceneOffHeapValues() {
        assumeDirectIOWorksHere();
        final Path dir = createTempDir();
        final float[][] vectors = randomVectors(600);
        final Path path = writeVectorFile(dir, vectors);
        final long regionBytes = (long) vectors.length * VECTOR_BYTES;
        final FlatVectorsScorer scorer = FlatVectorScorerUtil.getLucene99FlatVectorsScorer();
        final float[] query = vectors[13].clone();

        try (
            Directory mmap = new MMapDirectory(dir);
            IndexInput mmapFile = mmap.openInput(path.getFileName().toString(), IOContext.DEFAULT);
            DirectIOVectorIndexInput dioFile = DirectIOVectorIndexInput.open(path)
        ) {
            assertNotNull(dioFile);
            final OffHeapFloatVectorValues.DenseOffHeapVectorValues mmapValues = new OffHeapFloatVectorValues.DenseOffHeapVectorValues(
                DIMENSION,
                vectors.length,
                mmapFile.slice("vector-data", HEADER_LENGTH, regionBytes),
                VECTOR_BYTES,
                scorer,
                VectorSimilarityFunction.EUCLIDEAN
            );
            final OffHeapFloatVectorValues.DenseOffHeapVectorValues dioValues = new OffHeapFloatVectorValues.DenseOffHeapVectorValues(
                DIMENSION,
                vectors.length,
                dioFile.slice("vector-data", HEADER_LENGTH, regionBytes),
                VECTOR_BYTES,
                scorer,
                VectorSimilarityFunction.EUCLIDEAN
            );

            // The slice stays visible, which is how PrefetchableVectorValuesHelper finds it.
            assertTrue(dioValues instanceof HasIndexSlice);
            assertTrue(((HasIndexSlice) dioValues).getSlice() instanceof DirectIOVectorIndexInput);

            final RandomVectorScorer mmapScorer = scorer.getRandomVectorScorer(VectorSimilarityFunction.EUCLIDEAN, mmapValues, query);
            final RandomVectorScorer dioScorer = scorer.getRandomVectorScorer(VectorSimilarityFunction.EUCLIDEAN, dioValues, query);

            // Every ordinal, scored both ways. Bit-identical because both decode the same fp32 bytes into
            // the same scalar dot product: the SIMD scorer declines the Direct I/O input, so the arithmetic
            // on both sides is DefaultFlatVectorScorer's.
            for (int ord = 0; ord < vectors.length; ord++) {
                assertEquals("ordinal " + ord, mmapScorer.score(ord), dioScorer.score(ord), 0.0f);
            }

            // And through Lucene's own batch prefetch entry point, which is the shape the shipped
            // PrefetchableFlatVectorScorer drives.
            final int[] ords = scatteredOrds(64, vectors.length);
            dioValues.prefetch(ords, ords.length);
            for (final int ord : ords) {
                assertEquals("prefetched ordinal " + ord, mmapScorer.score(ord), dioScorer.score(ord), 0.0f);
            }
        }
    }

    // -----------------------------------------------------------------------------------------------
    // 5. Phase 9d-5: the read-ahead economy — the byte counters and the span bound
    // -----------------------------------------------------------------------------------------------

    /**
     * Rewrites the cluster settings this test case's mock {@code ClusterService} answers with, so that a
     * test can open an input under a non-default bound. {@code KNNTestCase#setUp} installs a
     * {@code ClusterSettings} built from {@code Settings.EMPTY}, i.e. every setting at its default, and
     * re-installs it between tests, so nothing here leaks into another test.
     */
    private void withNodeSettings(final Settings settings) {
        final Set<Setting<?>> registered = new HashSet<>(ClusterSettings.BUILT_IN_CLUSTER_SETTINGS);
        registered.addAll(
            KNNSettings.state()
                .getSettings()
                .stream()
                .filter(setting -> setting.getProperties().contains(Setting.Property.NodeScope))
                .collect(java.util.stream.Collectors.toList())
        );
        when(clusterService.getClusterSettings()).thenReturn(new ClusterSettings(settings, registered));
        KNNSettings.state().setClusterService(clusterService);
    }

    /**
     * {@code PrefetchHelper#prefetchExactVectorSize}'s grouping, driven against this input.
     *
     * <p>A mirror of {@code PrefetchHelper:92-108} rather than a call to it, for one reason: that method is
     * gated behind {@code KNNFeatureFlags#isPrefetchEnabled()}, which reads the settings singleton through a
     * different accessor than the one this test case initializes, so a call would test the harness rather
     * than the grouping. The node runs the real one — this asserts what the shape of its requests does to
     * the input. Kept literal, including the 128 KB budget being a parameter here so a test can show what
     * the group span controls.
     */
    @SneakyThrows
    private static int prefetchLikeHelper(
        final DirectIOVectorIndexInput input,
        final long baseOffset,
        final int vectorBytes,
        final int[] ords,
        final long groupBudget
    ) {
        final int[] sorted = ords.clone();
        java.util.Arrays.sort(sorted);
        int groups = 0;
        long groupStart = baseOffset + (long) sorted[0] * vectorBytes;
        for (int i = 1; i < sorted.length; i++) {
            final long current = baseOffset + (long) sorted[i] * vectorBytes;
            if ((current + vectorBytes) - groupStart > groupBudget) {
                final long previous = baseOffset + (long) sorted[i - 1] * vectorBytes;
                input.prefetch(groupStart, (previous + vectorBytes) - groupStart);
                groups++;
                groupStart = current;
            }
        }
        final long last = baseOffset + (long) sorted[sorted.length - 1] * vectorBytes;
        input.prefetch(groupStart, (last + vectorBytes) - groupStart);
        return groups + 1;
    }

    /** Ordinals drawn from one narrow window, so that a coalesced group spans several of them. */
    private static int[] clusteredOrds(final int count, final int windowSize, final int firstOrd) {
        final Random random = new Random(11L);
        final int[] ords = new int[count];
        for (int i = 0; i < count; i++) {
            ords[i] = firstOrd + random.nextInt(windowSize);
        }
        return ords;
    }

    /**
     * Both bounds come from settings, and both defaults are the values 9e-2 measured — so installing this
     * change on a node with no settings written changes nothing.
     */
    @SneakyThrows
    public void testBothStagingBoundsAreReadFromSettingsAndDefaultToTheMeasuredValues() {
        assumeDirectIOWorksHere();
        final Path dir = createTempDir();
        final Path path = writeVectorFile(dir, randomVectors(8));

        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(input);
            assertEquals(
                "the span bound must default to PrefetchHelper's group budget plus a block",
                DirectIOVectorIndexInput.DEFAULT_MAX_STAGED_RANGE_BYTES,
                input.maxStagedRangeBytes()
            );
            assertEquals(
                "the table size must default to the rescore prefetch window",
                KNNSettings.KNN_DIRECT_IO_RESCORE_PREFETCH_WINDOW_DEFAULT_VALUE,
                input.maxStagedRanges()
            );
        }

        withNodeSettings(
            Settings.builder()
                .put(KNNSettings.KNN_DIRECT_IO_RESCORE_PREFETCH_MAX_SPAN, "16kb")
                .put(KNNSettings.KNN_DIRECT_IO_RESCORE_PREFETCH_WINDOW, 64)
                .build()
        );
        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(input);
            assertEquals("the span bound must follow the setting", 16 * 1024, input.maxStagedRangeBytes());
            assertEquals("the table size must follow the setting", 64, input.maxStagedRanges());
        }
    }

    /**
     * The byte counters must account for every read and for exactly the bytes handed out, because the whole
     * point of them is that read amplification stops being an inference from {@code /proc/diskstats}.
     *
     * <p>9e-2's R60 divided a block's device {@code sectors_read} by its candidate count and attributed the
     * result to this class. That numerator is the whole node's traffic — the faiss graph and the quantized
     * codes are read from the same device in the same block — so the ratio it produced was an upper bound
     * on this path's amplification, not a measurement of it. These counters are taken inside the object
     * that issues the reads, so they can neither borrow another reader's bytes nor miss any of their own.
     */
    @SneakyThrows
    public void testByteCountersMeasureAmplificationExactly() {
        assumeDirectIOWorksHere();
        final Path dir = createTempDir();
        final float[][] vectors = randomVectors(2000);
        final Path path = writeVectorFile(dir, vectors);
        final int[] ords = scatteredOrds(40, vectors.length);
        final float[] scratch = new float[DIMENSION];

        // Arm A: no prefetch, so every byte read is a blocking read and the served total is exact.
        DirectIOVectorIndexInput.STATS.reset();
        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(input);
            for (final int ord : ords) {
                input.seek(HEADER_LENGTH + (long) ord * VECTOR_BYTES);
                input.readFloats(scratch, 0, DIMENSION);
                assertArrayEquals(vectors[ord], scratch, 0.0f);
            }
            final int blockSize = input.blockSize();
            assertEquals(
                "servedBytes must be exactly the vectors handed out",
                (long) ords.length * VECTOR_BYTES,
                DirectIOVectorIndexInput.STATS.servedBytes()
            );
            assertEquals("nothing may be staged without a prefetch", 0L, DirectIOVectorIndexInput.STATS.stagedBytes());
            final long blocking = DirectIOVectorIndexInput.STATS.blockingBytes();
            assertEquals("every blocking read is block aligned", 0L, blocking % blockSize);
            assertTrue(
                "a 3072 byte vector cannot be read in fewer than 3072 bytes, got " + blocking,
                blocking >= (long) ords.length * VECTOR_BYTES
            );
            logger.info("9d-5 amplification, no prefetch: {}", DirectIOVectorIndexInput.STATS);
        }

        // Arm B: the same reads behind a prefetch burst. The served total is unchanged -- the same vectors
        // were handed out -- and the device traffic has moved from the blocking counter to the staged one.
        DirectIOVectorIndexInput.STATS.reset();
        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(input);
            prefetchLikeHelper(input, HEADER_LENGTH, VECTOR_BYTES, ords, 128 * 1024);
            for (final int ord : ords) {
                input.seek(HEADER_LENGTH + (long) ord * VECTOR_BYTES);
                input.readFloats(scratch, 0, DIMENSION);
                assertArrayEquals(vectors[ord], scratch, 0.0f);
            }
        }
        assertEquals(
            "servedBytes must not depend on how the bytes were fetched",
            (long) ords.length * VECTOR_BYTES,
            DirectIOVectorIndexInput.STATS.servedBytes()
        );
        assertTrue("the burst must have read through the staging channel", DirectIOVectorIndexInput.STATS.stagedBytes() > 0);
        assertTrue(
            "bytes read must cover bytes served",
            DirectIOVectorIndexInput.STATS.stagedBytes() + DirectIOVectorIndexInput.STATS.blockingBytes() >= (long) ords.length
                * VECTOR_BYTES
        );
        logger.info("9d-5 amplification, prefetched: {}", DirectIOVectorIndexInput.STATS);
    }

    /**
     * <b>The waste guard, and the two opposite things it does — which is the whole finding.</b>
     *
     * <p>Two batches of the same size over the same file, both through {@code PrefetchHelper}'s grouping:
     * one <em>scattered</em> over 20,000 vectors, one <em>clustered</em> inside a 40-vector window. The span
     * bound is then lowered to one block pair and both are read again.
     *
     * <p>The measurement (numbers from this test's own log, 9d-5):
     * <table>
     *   <caption>bytes the device delivered, against bytes handed to the caller</caption>
     *   <tr><th>batch</th><th>groups</th><th>bound</th><th>read</th><th>served</th><th>amplification</th></tr>
     *   <tr><td>scattered</td><td>53</td><td>132 KB</td><td>1,064,960</td><td>196,608</td><td>5.42x</td></tr>
     *   <tr><td>scattered</td><td>53</td><td>8 KB</td><td>450,560</td><td>196,608</td><td><b>2.29x</b></td></tr>
     *   <tr><td>clustered</td><td>1</td><td>132 KB</td><td>126,976</td><td>196,608</td><td><b>0.65x</b></td></tr>
     *   <tr><td>clustered</td><td>1</td><td>8 KB</td><td>499,712</td><td>196,608</td><td>2.54x</td></tr>
     * </table>
     *
     * <p>So the bound's sign depends on the batch, and the span it is given cannot tell it which it has.
     * A <em>sparse</em> group — 53 groups covering 64 scattered ordinals, so some groups hold two or three
     * vectors 100 KB apart — is mostly gap, and declining it saves 58% of the traffic. A <em>dense</em>
     * group — 64 ordinals inside 40 vectors, coalescing into one 124 KB span that every read then hits —
     * is the case the grouping was built for, and declining it costs 3.9x, because the span it replaces
     * with 64 blocking reads was being read once and reused. Span length is the same 124 KB in both.
     *
     * <p>That is why the bound is a setting whose default leaves 9e-2's behaviour exactly as measured,
     * rather than a new constant: it is a trade to be made per substrate, and the shape-aware version of
     * it (a bound driven by the amplification the <em>previous</em> burst actually achieved, which is
     * observable here where the gaps are not) is a further step this does not take. Correctness is
     * asserted on every ordinal in all four combinations, because a guard that changed a byte would be
     * worse than no guard at all.
     */
    @SneakyThrows
    public void testTheSpanBoundCutsSparseGroupsAndCostsDenseOnes() {
        assumeDirectIOWorksHere();
        final Path dir = createTempDir();
        final float[][] vectors = randomVectors(20000);
        final Path path = writeVectorFile(dir, vectors);
        final float[] scratch = new float[DIMENSION];
        final int[] scattered = scatteredOrds(64, vectors.length);
        final int[] clustered = clusteredOrds(64, 40, 5000);

        final long[][] measured = new long[4][];
        final String[] labels = { "scattered/default-bound", "clustered/default-bound", "scattered/8kb-bound", "clustered/8kb-bound" };
        int slot = 0;
        for (final int bound : new int[] { DirectIOVectorIndexInput.DEFAULT_MAX_STAGED_RANGE_BYTES, 8192 }) {
            for (final int[] ords : new int[][] { scattered, clustered }) {
                DirectIOVectorIndexInput.STATS.reset();
                try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path, 0, 64, bound)) {
                    assertNotNull(input);
                    final int groups = prefetchLikeHelper(input, HEADER_LENGTH, VECTOR_BYTES, ords, 128 * 1024);
                    for (final int ord : ords) {
                        input.seek(HEADER_LENGTH + (long) ord * VECTOR_BYTES);
                        input.readFloats(scratch, 0, DIMENSION);
                        // The guard must be invisible in the bytes, in every combination.
                        assertArrayEquals(labels[slot] + " ordinal " + ord, vectors[ord], scratch, 0.0f);
                    }
                    measured[slot] = new long[] {
                        groups,
                        DirectIOVectorIndexInput.STATS.stagedBytes() + DirectIOVectorIndexInput.STATS.blockingBytes(),
                        DirectIOVectorIndexInput.STATS.servedBytes(),
                        DirectIOVectorIndexInput.STATS.stageDeclined(),
                        DirectIOVectorIndexInput.STATS.declinedSpanBytes() };
                }
                logger.info(
                    "9d-5 span bound {}: {} groups={} read={} served={} declined={} declinedSpan={}",
                    bound,
                    labels[slot],
                    measured[slot][0],
                    measured[slot][1],
                    measured[slot][2],
                    measured[slot][3],
                    measured[slot][4]
                );
                slot++;
            }
        }

        // The grouping itself, because if it ever changes the rest of this test measures something else:
        // scattered ordinals coalesce only incidentally (53 groups for 64 ordinals), clustered ones fully.
        assertTrue("a scattered batch must stay mostly ungrouped, groups=" + measured[0][0], measured[0][0] > 32);
        assertTrue("a scattered batch must coalesce somewhat, groups=" + measured[0][0], measured[0][0] < 64);
        assertEquals("64 ordinals inside 40 vectors must coalesce into one group", 1L, measured[1][0]);

        // Neither batch is declined at the default bound: it is PrefetchHelper's own budget plus a block,
        // so by construction no group can exceed it. Installing the setting changes nothing by itself.
        assertEquals("nothing may be declined at the default bound, scattered", 0L, measured[0][3]);
        assertEquals("nothing may be declined at the default bound, clustered", 0L, measured[1][3]);

        // The finding, in both directions. Sparse groups are mostly gap, so declining them saves traffic.
        assertTrue("an 8 KB bound must decline sparse groups", measured[2][3] > 0);
        assertTrue(
            "the bound must reduce bytes read on a scattered batch, " + measured[0][1] + " -> " + measured[2][1],
            measured[2][1] < measured[0][1]
        );
        // Dense groups are the case the grouping exists for, so declining them costs traffic. This is the
        // assertion that stops the bound being described as a free win anywhere.
        assertTrue("an 8 KB bound must decline the dense group too", measured[3][3] > 0);
        assertTrue(
            "the bound must COST bytes on a clustered batch, " + measured[1][1] + " -> " + measured[3][1],
            measured[3][1] > measured[1][1]
        );

        // And in every combination the caller got the same bytes, which is the only invariant the guard owes.
        assertEquals("served bytes cannot depend on the bound, scattered", measured[0][2], measured[2][2]);
        assertEquals("served bytes cannot depend on the bound, clustered", measured[1][2], measured[3][2]);
    }
}

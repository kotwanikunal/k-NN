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
import java.util.concurrent.Callable;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ThreadPoolExecutor;

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
 * <p>Tests needing an open handle are skipped, not failed, where the filesystem refuses {@code O_DIRECT}:
 * tmpfs and overlayfs answer {@code EINVAL} on an {@code O_DIRECT} open, so a skip there is the harness
 * reporting its own substrate rather than this code failing. Staged reads run on {@link DirectIOReadPool},
 * whose daemon threads outlive a suite by design, hence the same thread filter.
 */
@ThreadLeakFilters(defaultFilters = true, filters = { DirectIOReadPoolTests.ReadPoolThreadFilter.class })
public class DirectIOVectorIndexInputTests extends KNNTestCase {

    private static final int DIMENSION = 768;
    private static final int VECTOR_BYTES = DIMENSION * Float.BYTES;
    /** Deliberately not a multiple of the block size or of {@code gcd(vectorBytes, blockSize)}. */
    private static final int HEADER_LENGTH = 44;

    /**
     * The staging arena belongs to the <b>thread</b> since D3c, and a test method runs on the same thread as
     * the one before it, so without this one test would inherit the previous test's arena — and the
     * accounting tests, which assert what an allocation charges, would measure a reuse instead.
     */
    @Override
    public void setUp() throws Exception {
        super.setUp();
        DirectIOStagingLease.resetForTesting();
        // The direct-memory budget is a constant on a node; the cases below publish their own, so each
        // starts from the shipped one and leaves it there.
        DirectIOVectorIndexInput.resetBudgetsForTesting();
    }

    @Override
    public void tearDown() throws Exception {
        DirectIOVectorIndexInput.resetBudgetsForTesting();
        super.tearDown();
    }

    @SneakyThrows
    private void assumeDirectIOWorksHere() {
        final OpenOption direct = DirectIOVectorIndexInput.directOpenOption();
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
            // 64, the size of Lucene's bulk batch. A smaller table declines the tail of every burst,
            // which task-15 measured as a paced p99 of 30 ms against 11 ms at identical byte volume.
            assertEquals(
                "the table size must default to the burst size",
                DirectIOVectorIndexInput.DEFAULT_MAX_STAGED_RANGES,
                input.maxStagedRanges()
            );
            assertEquals("and that default must be Lucene's bulk batch size", 64, input.maxStagedRanges());
        }

        // Both are constants on a node, so a measurement that wants another geometry asks for it at open.
        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path, 0, 12, 16 * 1024)) {
            assertNotNull(input);
            assertEquals("the span bound must follow the overload", 16 * 1024, input.maxStagedRangeBytes());
            assertEquals("the table size must follow the overload", 12, input.maxStagedRanges());
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

    // -----------------------------------------------------------------------------------------------
    // 5. Sequential read-ahead ring
    //
    // The ring exists to close the gap prefetch never covers: a bulk sequential consumer (a merge, the
    // faiss-unquantized exact first pass, traversal) issues no prefetch() and so reads one blocking QD1
    // window at a time. These tests assert the ring arms on a forward run and not otherwise, serves
    // correct bytes, stands down under a prefetch burst, and is exact passthrough when disabled.
    // -----------------------------------------------------------------------------------------------

    /** Reads every vector in ascending order and checks it against the written data. @return blockingReads. */
    @SneakyThrows
    private long sequentialScan(final DirectIOVectorIndexInput input, final float[][] vectors) {
        final float[] scratch = new float[DIMENSION];
        for (int ord = 0; ord < vectors.length; ord++) {
            input.seek(HEADER_LENGTH + (long) ord * VECTOR_BYTES);
            input.readFloats(scratch, 0, DIMENSION);
            assertArrayEquals("ordinal " + ord, vectors[ord], scratch, 0.0f);
        }
        return DirectIOVectorIndexInput.STATS.blockingReads();
    }

    /**
     * With the window at zero the ring is off: no field is written, no read is issued, and the input is
     * byte-for-byte and syscall-for-syscall what it is with this code absent. This is the arm every other
     * Direct I/O number was taken against, so it has to be a true passthrough rather than a quiet ring.
     */
    @SneakyThrows
    public void testReadaheadWindowZeroIsExactPassthrough() {
        assumeDirectIOWorksHere();
        final Path dir = createTempDir();
        final float[][] vectors = randomVectors(1500);
        final Path path = writeVectorFile(dir, vectors);

        DirectIOReadPool.resetForTesting();
        DirectIOVectorIndexInput.STATS.reset();
        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path, 0, 64, 128 * 1024 + 4096, 0, 2)) {
            assertNotNull(input);
            assertEquals("window zero must mean the ring is off", 0, input.readaheadWindows());
            final long blocking = sequentialScan(input, vectors);
            assertTrue("a passthrough scan still issues its blocking reads, blocking=" + blocking, blocking > 100);
        }
        assertEquals("the ring must never arm when off", 0L, DirectIOVectorIndexInput.STATS.readaheadArmed());
        assertEquals("the ring must issue no reads when off", 0L, DirectIOVectorIndexInput.STATS.readaheadReads());
        assertEquals("the ring must serve nothing when off", 0L, DirectIOVectorIndexInput.STATS.readaheadHits());
        assertEquals("no ring, no ring bytes", 0L, DirectIOVectorIndexInput.STATS.readaheadBytes());
    }

    /**
     * Phase 13 audit FIX 2: the ring's window size is its own, independent of {@code bufferSize}. Three
     * things have to hold, and the third is the one the audit cares about.
     *
     * <ol>
     *   <li>{@code bufferSize} does not move. The random rescore path reads through
     *       {@code blockingSource} and the burst table, and at 256 KiB its measured amplification is 85.3x
     *       against 2.667x at 8 KiB — so a fix that raised {@code bufferSize} would pay for sequential
     *       throughput with 32x the device traffic on the path the feature exists for.</li>
     *   <li>The ring submits <em>that</em> size, which is what turns 49.7 MiB/s into 371.2.</li>
     *   <li><b>The bytes are identical.</b> A wider window reads the same bytes in fewer, larger requests;
     *       it must not read different ones. Asserted as {@code servedBytes} equality against an 8 KiB ring
     *       over the same scan.</li>
     * </ol>
     */
    @SneakyThrows
    public void testTheReadaheadWindowSizeIsDecoupledFromBufferSize() {
        assumeDirectIOWorksHere();
        final Path dir = createTempDir();
        final float[][] vectors = randomVectors(1500);
        final Path path = writeVectorFile(dir, vectors);

        // The coupled shape, for the byte-level control: a ring window of one bufferSize.
        DirectIOReadPool.resetForTesting();
        DirectIOVectorIndexInput.STATS.reset();
        final long coupledServed;
        final long coupledRingReads;
        final int bufferSize;
        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path, 0, 64, 128 * 1024 + 4096, 8, 0, 2)) {
            bufferSize = input.bufferSize();
            assertEquals("window bytes of 0 must mean bufferSize, i.e. the pre-phase-14 ring", bufferSize, input.readaheadWindowBytes());
            sequentialScan(input, vectors);
            coupledServed = DirectIOVectorIndexInput.STATS.servedBytes();
            coupledRingReads = DirectIOVectorIndexInput.STATS.readaheadReads();
        }

        // Decoupled: a 64 KiB ring window over the same 8 KiB blocking reads.
        final int wideWindow = 64 * 1024;
        DirectIOReadPool.resetForTesting();
        DirectIOVectorIndexInput.STATS.reset();
        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path, 0, 64, 128 * 1024 + 4096, 8, wideWindow, 2)) {
            assertEquals("bufferSize must not move: the random path reads through it", bufferSize, input.bufferSize());
            assertEquals("the ring must take the wider window", wideWindow, input.readaheadWindowBytes());
            assertEquals("a slice inherits both sizes", wideWindow, input.slice("region", 0, input.length()).readaheadWindowBytes());
            assertEquals("and so does a clone", wideWindow, input.clone().readaheadWindowBytes());
            sequentialScan(input, vectors);
            final long wideRingReads = DirectIOVectorIndexInput.STATS.readaheadReads();
            logger.info(
                "readahead decoupled: bufferSize={} coupledRingReads={} wideRingReads={} | {}",
                bufferSize,
                coupledRingReads,
                wideRingReads,
                DirectIOVectorIndexInput.STATS
            );
            assertTrue("the ring must still arm", DirectIOVectorIndexInput.STATS.readaheadArmed() > 0);
            assertTrue(
                "a window "
                    + wideWindow / bufferSize
                    + "x wider must take fewer requests for the same scan, "
                    + coupledRingReads
                    + " -> "
                    + wideRingReads,
                wideRingReads < coupledRingReads
            );
            assertEquals(
                "a wider window must read the SAME bytes, in fewer requests",
                coupledServed,
                DirectIOVectorIndexInput.STATS.servedBytes()
            );
        }
    }

    /** The shipped window size, pinned: 64 KB, the knee the phase 13 audit measured. */
    public void testTheShippedReadaheadWindowBytesIsSixtyFourKilobytes() {
        assertEquals(64L * 1024, DirectIOVectorIndexInput.DEFAULT_READAHEAD_WINDOW_BYTES);
    }

    /**
     * A forward scan arms the ring and most of its reads come from it, so the blocking-read count collapses
     * far below the passthrough count at identical served bytes. This is the whole point: sequential
     * O_DIRECT stops being one QD1 pread per window.
     */
    @SneakyThrows
    public void testSequentialScanArmsAndServesFromTheRing() {
        assumeDirectIOWorksHere();
        final Path dir = createTempDir();
        final float[][] vectors = randomVectors(1500);
        final Path path = writeVectorFile(dir, vectors);

        // Baseline: passthrough.
        DirectIOReadPool.resetForTesting();
        DirectIOVectorIndexInput.STATS.reset();
        final long passthroughBlocking;
        final long passthroughServed;
        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path, 0, 64, 128 * 1024 + 4096, 0, 2)) {
            passthroughBlocking = sequentialScan(input, vectors);
            passthroughServed = DirectIOVectorIndexInput.STATS.servedBytes();
        }

        // Armed: the same scan with an eight-window ring.
        DirectIOReadPool.resetForTesting();
        DirectIOVectorIndexInput.STATS.reset();
        final long armedBlocking;
        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path, 0, 64, 128 * 1024 + 4096, 8, 2)) {
            assertEquals(8, input.readaheadWindows());
            armedBlocking = sequentialScan(input, vectors);
        }
        logger.info(
            "readahead sequential: passthroughBlocking={} armedBlocking={} | {}",
            passthroughBlocking,
            armedBlocking,
            DirectIOVectorIndexInput.STATS
        );
        assertTrue("the ring must arm on a forward scan", DirectIOVectorIndexInput.STATS.readaheadArmed() > 0);
        assertTrue(
            "most reads must come from the ring, hits=" + DirectIOVectorIndexInput.STATS.readaheadHits(),
            DirectIOVectorIndexInput.STATS.readaheadHits() > 100
        );
        assertTrue("the ring must issue reads", DirectIOVectorIndexInput.STATS.readaheadReads() > 0);
        assertTrue(
            "arming must collapse the blocking-read count, " + passthroughBlocking + " -> " + armedBlocking,
            armedBlocking < passthroughBlocking / 2
        );
        assertEquals("served bytes cannot change with the ring on", passthroughServed, DirectIOVectorIndexInput.STATS.servedBytes());
    }

    /**
     * The no-waste proof. A random (rescore-shaped) access pattern must never cross the trigger, so the
     * ring issues ZERO speculative reads — O_DIRECT's no-amplification behaviour on the rescore pattern is
     * preserved exactly even with read-ahead configured on.
     */
    @SneakyThrows
    public void testRandomAccessNeverArmsTheRing() {
        assumeDirectIOWorksHere();
        final Path dir = createTempDir();
        final float[][] vectors = randomVectors(20000);
        final Path path = writeVectorFile(dir, vectors);
        final int[] ords = scatteredOrds(400, vectors.length);
        final float[] scratch = new float[DIMENSION];

        DirectIOReadPool.resetForTesting();
        DirectIOVectorIndexInput.STATS.reset();
        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path, 0, 64, 128 * 1024 + 4096, 8, 2)) {
            assertNotNull(input);
            for (final int ord : ords) {
                input.seek(HEADER_LENGTH + (long) ord * VECTOR_BYTES);
                input.readFloats(scratch, 0, DIMENSION);
                assertArrayEquals(vectors[ord], scratch, 0.0f);
            }
        }
        logger.info("readahead no-waste: {}", DirectIOVectorIndexInput.STATS);
        assertEquals("random access must never arm the ring", 0L, DirectIOVectorIndexInput.STATS.readaheadArmed());
        assertEquals("random access must issue zero speculative reads", 0L, DirectIOVectorIndexInput.STATS.readaheadReads());
        assertEquals("and serve nothing from it", 0L, DirectIOVectorIndexInput.STATS.readaheadHits());
    }

    /**
     * While a prefetch burst is active the ring stands down — the rescore scorer owns the input, its access
     * is random and bursty, and the burst table serves it. The ring issues zero reads; the burst's hits are
     * unchanged. The two mechanisms coexist without resetting each other.
     */
    @SneakyThrows
    public void testActiveBurstMakesTheRingStandDown() {
        assumeDirectIOWorksHere();
        final Path dir = createTempDir();
        final float[][] vectors = randomVectors(1500);
        final Path path = writeVectorFile(dir, vectors);
        final float[] scratch = new float[DIMENSION];

        DirectIOReadPool.resetForTesting();
        DirectIOVectorIndexInput.STATS.reset();
        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path, 0, 64, 128 * 1024 + 4096, 8, 2)) {
            assertNotNull(input);
            // One burst up front, left active for the whole scan (no second prefetch, so no reset). Then
            // read ascending: reads the burst covers are served from it, the rest go blocking and try to
            // arm the ring, which must stand down because stagedCount > 0.
            for (int ord = 0; ord < vectors.length; ord++) {
                input.prefetch(HEADER_LENGTH + (long) ord * VECTOR_BYTES, VECTOR_BYTES);
            }
            for (int ord = 0; ord < vectors.length; ord++) {
                input.seek(HEADER_LENGTH + (long) ord * VECTOR_BYTES);
                input.readFloats(scratch, 0, DIMENSION);
                assertArrayEquals("ordinal " + ord, vectors[ord], scratch, 0.0f);
            }
        }
        logger.info("readahead stand-down: {}", DirectIOVectorIndexInput.STATS);
        assertTrue("the burst must have served reads", DirectIOVectorIndexInput.STATS.stagedHits() > 0);
        assertEquals("an active burst must suppress all ring reads", 0L, DirectIOVectorIndexInput.STATS.readaheadReads());
        assertEquals("and all ring hits", 0L, DirectIOVectorIndexInput.STATS.readaheadHits());
    }

    /**
     * A backward jump mid-scan is a discontinuity: the ring quiesces the now-wrong windows, the bytes stay
     * correct, and a renewed forward run re-arms it. Proves the detection ramp both tears down and rebuilds.
     */
    @SneakyThrows
    public void testBackwardSeekQuiescesThenReArms() {
        assumeDirectIOWorksHere();
        final Path dir = createTempDir();
        final float[][] vectors = randomVectors(1500);
        final Path path = writeVectorFile(dir, vectors);
        final float[] scratch = new float[DIMENSION];

        DirectIOReadPool.resetForTesting();
        DirectIOVectorIndexInput.STATS.reset();
        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path, 0, 64, 128 * 1024 + 4096, 8, 2)) {
            assertNotNull(input);
            // Forward through the first half: arms.
            for (int ord = 0; ord < 700; ord++) {
                input.seek(HEADER_LENGTH + (long) ord * VECTOR_BYTES);
                input.readFloats(scratch, 0, DIMENSION);
                assertArrayEquals("fwd " + ord, vectors[ord], scratch, 0.0f);
            }
            final long armedFirst = DirectIOVectorIndexInput.STATS.readaheadArmed();
            assertTrue("the first run must arm", armedFirst > 0);
            // Jump back to the start and scan forward again: the ring must quiesce and re-arm.
            for (int ord = 0; ord < 700; ord++) {
                input.seek(HEADER_LENGTH + (long) ord * VECTOR_BYTES);
                input.readFloats(scratch, 0, DIMENSION);
                assertArrayEquals("back " + ord, vectors[ord], scratch, 0.0f);
            }
            assertTrue("the renewed forward run must re-arm", DirectIOVectorIndexInput.STATS.readaheadArmed() > armedFirst);
        }
    }

    /**
     * A slice carries its own ring, as it carries its own buffers and burst table today, so a slice of this
     * input reads correctly and arms on its own forward run.
     */
    @SneakyThrows
    public void testSliceCarriesItsOwnRing() {
        assumeDirectIOWorksHere();
        final Path dir = createTempDir();
        final float[][] vectors = randomVectors(1500);
        final Path path = writeVectorFile(dir, vectors);
        final float[] scratch = new float[DIMENSION];

        DirectIOReadPool.resetForTesting();
        DirectIOVectorIndexInput.STATS.reset();
        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path, 0, 64, 128 * 1024 + 4096, 8, 2)) {
            assertNotNull(input);
            final DirectIOVectorIndexInput region = input.slice("vectors", HEADER_LENGTH, (long) vectors.length * VECTOR_BYTES);
            assertEquals("a slice inherits the window", 8, region.readaheadWindows());
            for (int ord = 0; ord < vectors.length; ord++) {
                region.seek((long) ord * VECTOR_BYTES);
                region.readFloats(scratch, 0, DIMENSION);
                assertArrayEquals("slice ordinal " + ord, vectors[ord], scratch, 0.0f);
            }
        }
        assertTrue("the slice's ring must arm and serve", DirectIOVectorIndexInput.STATS.readaheadHits() > 100);
    }

    /**
     * Both read-ahead settings are read at open and default to the measured values, mirroring the
     * staging-bound test above. The window default of 8 is what the block-layer sweep earned; 0 is still
     * reachable by setting, and is the exact-passthrough escape hatch.
     */
    @SneakyThrows
    public void testReadaheadSettingsAreReadFromSettingsAndDefaultToTheMeasuredValues() {
        assumeDirectIOWorksHere();
        final Path dir = createTempDir();
        final Path path = writeVectorFile(dir, randomVectors(8));

        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(input);
            assertEquals("the window must default to the measured value", 8, input.readaheadWindows());
            assertEquals("the trigger must default to the measured value", 2, input.readaheadTrigger());
        }

        // The window count is the one that stays a setting; the trigger is a constant, so a sweep of it
        // goes through the sizing overload.
        withNodeSettings(Settings.builder().put(KNNSettings.KNN_DIRECT_IO_READAHEAD_WINDOW, 8).build());
        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(input);
            assertEquals("the window must follow the setting", 8, input.readaheadWindows());
        }
        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path, 0, 64, 128 * 1024 + 4096, 8, 3)) {
            assertNotNull(input);
            assertEquals("the trigger must follow the overload", 3, input.readaheadTrigger());
        }
    }

    // -----------------------------------------------------------------------------------------------
    // 5. No search thread waits forever on a pool future
    // -----------------------------------------------------------------------------------------------

    /**
     * The hardening property: a staged read that cannot be retired must not hold the reading thread.
     *
     * <p>The pool is saturated deliberately — every thread busy and every queue slot but one taken by a
     * latched task — so the range this prefetch stages is queued behind work that will not finish until the
     * test releases it. The read that follows therefore cannot be served from the staged range, and what is
     * asserted is that it returns anyway, with the right bytes, having counted the event.
     *
     * <p>Before the bound this read waited on an unbounded {@code Future#get}: it would have blocked until
     * the latch released, i.e. forever on a node where the pool never catches up. That is the shape of the
     * Oct-2 incident's search tasks, which logged elapsed times of 2.1 hours.
     */
    @SneakyThrows
    public void testAReadGivesUpOnASaturatedPoolAndServesItselfInstead() {
        assumeDirectIOWorksHere();
        final Path dir = createTempDir();
        final float[][] vectors = randomVectors(64);
        final Path path = writeVectorFile(dir, vectors);
        final float[] scratch = new float[DIMENSION];

        DirectIOReadPool.resetForTesting();
        final CountDownLatch release = new CountDownLatch(1);
        saturatePoolLeavingOneSlot(release);

        DirectIOVectorIndexInput.STATS.reset();
        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path, 0, 64, 128 * 1024 + 4096, 0, 2)) {
            assertNotNull(input);
            final long offset = HEADER_LENGTH + (long) 7 * VECTOR_BYTES;
            input.prefetch(offset, VECTOR_BYTES);
            // If the staging table could not take the range at all there is nothing to time out, so this
            // test would be vacuous; assert it did stage before relying on the wait.
            assumeTrue("the pool refused the prefetch, so there is no wait to bound", DirectIOVectorIndexInput.STATS.stageDeclined() == 0);

            final long start = System.nanoTime();
            input.seek(offset);
            input.readFloats(scratch, 0, DIMENSION);
            final long elapsedMillis = (System.nanoTime() - start) / 1_000_000L;

            assertArrayEquals("the fallback must deliver the same bytes", vectors[7], scratch, 0.0f);
            assertEquals(
                "the give-up must be counted, so a saturated pool is visible rather than silent",
                1L,
                DirectIOVectorIndexInput.STATS.stagedWaitTimeouts()
            );
            assertTrue(
                "the fallback must be a blocking read on this thread, blockingReads=" + DirectIOVectorIndexInput.STATS.blockingReads(),
                DirectIOVectorIndexInput.STATS.blockingReads() > 0
            );
            assertTrue(
                "the read must not have waited much past the bound, waited " + elapsedMillis + " ms",
                elapsedMillis < 10 * DirectIOVectorIndexInput.STAGED_READ_WAIT_MILLIS
            );
        } finally {
            release.countDown();
            DirectIOReadPool.resetForTesting();
        }
    }

    /**
     * An abandoned range serves nothing afterwards, which is what keeps a buffer a read may still be
     * writing into from being decoded as if it held the bytes. Without it, the second read of the same
     * vector would find the range still in the burst table, wait again, and — worse — a buffer later reused
     * for another range could be read as this one.
     */
    @SneakyThrows
    public void testAnAbandonedRangeIsNeverServedFromAgain() {
        assumeDirectIOWorksHere();
        final Path dir = createTempDir();
        final float[][] vectors = randomVectors(64);
        final Path path = writeVectorFile(dir, vectors);
        final float[] scratch = new float[DIMENSION];

        DirectIOReadPool.resetForTesting();
        final CountDownLatch release = new CountDownLatch(1);
        saturatePoolLeavingOneSlot(release);

        DirectIOVectorIndexInput.STATS.reset();
        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path, 0, 64, 128 * 1024 + 4096, 0, 2)) {
            final long offset = HEADER_LENGTH + (long) 3 * VECTOR_BYTES;
            input.prefetch(offset, VECTOR_BYTES);
            assumeTrue("the pool refused the prefetch", DirectIOVectorIndexInput.STATS.stageDeclined() == 0);

            input.seek(offset);
            input.readFloats(scratch, 0, DIMENSION);
            assertEquals(1L, DirectIOVectorIndexInput.STATS.stagedWaitTimeouts());
            assertArrayEquals(vectors[3], scratch, 0.0f);

            // Read it again. The range is still in the burst table, but abandoned, so it must be skipped
            // outright rather than waited on a second time.
            final long start = System.nanoTime();
            input.seek(offset);
            input.readFloats(scratch, 0, DIMENSION);
            final long elapsedMillis = (System.nanoTime() - start) / 1_000_000L;
            assertArrayEquals(vectors[3], scratch, 0.0f);
            assertEquals(
                "the second read must not wait on the abandoned range again",
                1L,
                DirectIOVectorIndexInput.STATS.stagedWaitTimeouts()
            );
            assertTrue(
                "and so must be fast, took " + elapsedMillis + " ms",
                elapsedMillis < DirectIOVectorIndexInput.STAGED_READ_WAIT_MILLIS
            );
        } finally {
            release.countDown();
            DirectIOReadPool.resetForTesting();
        }
    }

    /**
     * Closing an input whose staged reads cannot be retired must still return. {@code close} quiesces the
     * burst table, and that quiesce used an unbounded {@code Future#get} too — so a shard close, or the
     * {@code resetStaging} at the head of the next {@link DirectIOVectorIndexInput#prefetch}, could hang on
     * exactly the same condition as a read.
     */
    @SneakyThrows
    public void testCloseReturnsEvenWhenAStagedReadCannotBeRetired() {
        assumeDirectIOWorksHere();
        final Path dir = createTempDir();
        final float[][] vectors = randomVectors(64);
        final Path path = writeVectorFile(dir, vectors);

        DirectIOReadPool.resetForTesting();
        final CountDownLatch release = new CountDownLatch(1);
        saturatePoolLeavingOneSlot(release);

        DirectIOVectorIndexInput.STATS.reset();
        try {
            final DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path, 0, 64, 128 * 1024 + 4096, 0, 2);
            assertNotNull(input);
            input.prefetch(HEADER_LENGTH, VECTOR_BYTES);
            assumeTrue("the pool refused the prefetch", DirectIOVectorIndexInput.STATS.stageDeclined() == 0);
            final long start = System.nanoTime();
            input.close();
            final long elapsedMillis = (System.nanoTime() - start) / 1_000_000L;
            assertTrue(
                "close must be bounded, took " + elapsedMillis + " ms",
                elapsedMillis < 10 * DirectIOVectorIndexInput.STAGED_READ_WAIT_MILLIS
            );
        } finally {
            release.countDown();
            DirectIOReadPool.resetForTesting();
        }
    }

    /** On a healthy pool nothing ever gives up, so the counter stays at zero on the ordinary rescore shape. */
    @SneakyThrows
    public void testAHealthyPoolNeverGivesUp() {
        assumeDirectIOWorksHere();
        final Path dir = createTempDir();
        final float[][] vectors = randomVectors(2000);
        final Path path = writeVectorFile(dir, vectors);
        final int[] ords = scatteredOrds(64, vectors.length);
        final float[] scratch = new float[DIMENSION];

        DirectIOReadPool.resetForTesting();
        DirectIOVectorIndexInput.STATS.reset();
        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path, 0, 64, 128 * 1024 + 4096, 8, 2)) {
            for (int repeat = 0; repeat < 10; repeat++) {
                for (final int ord : ords) {
                    input.prefetch(HEADER_LENGTH + (long) ord * VECTOR_BYTES, VECTOR_BYTES);
                }
                for (final int ord : ords) {
                    input.seek(HEADER_LENGTH + (long) ord * VECTOR_BYTES);
                    input.readFloats(scratch, 0, DIMENSION);
                    assertArrayEquals(vectors[ord], scratch, 0.0f);
                }
            }
        }
        assertEquals(
            "a healthy pool must never trip the bound, " + DirectIOVectorIndexInput.STATS,
            0L,
            DirectIOVectorIndexInput.STATS.stagedWaitTimeouts()
        );
    }

    // -----------------------------------------------------------------------------------------------
    // 7. One allocation per input, not one per slot
    //
    // Phase 12 S1.3 measured direct-buffer capacity cycling to 4,365 MB under 16-way load, against a node
    // now configured with -XX:MaxDirectMemorySize=2g. The cause is structural: a fresh input per query per
    // segment, each allocating its own sync window plus up to staged_ranges staged slots plus a ring, and
    // no end-of-life signal on the production path to free any of them (Lucene's OffHeapFloatVectorValues
    // is not Closeable, so close() never runs and the Cleaner waits for a GC). These tests assert the fix:
    // the per-slot allocations become one arena per table, the alignment slack per slot goes away, and
    // nothing about what the reads return changes.
    // -----------------------------------------------------------------------------------------------

    /**
     * The burst table is one allocation whose slots carry no alignment slack, it is allocated once for the
     * input's life however many bursts run through it, and every slot is block aligned — which is what
     * {@code O_DIRECT} requires of the buffer address.
     */
    @SneakyThrows
    public void testTheBurstTableIsOneAllocationAndNotOnePerSlot() {
        assumeDirectIOWorksHere();
        final Path dir = createTempDir();
        final float[][] vectors = randomVectors(4000);
        final Path path = writeVectorFile(dir, vectors);
        final int[] ords = scatteredOrds(64, vectors.length);
        final float[] scratch = new float[DIMENSION];

        DirectIOReadPool.resetForTesting();
        DirectIOVectorIndexInput.STATS.reset();
        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path, 0, 64, 128 * 1024 + 4096, 0, 2)) {
            assertNotNull(input);
            assertNull("nothing is allocated before the first prefetch", input.stagingArena());

            for (final int ord : ords) {
                input.prefetch(HEADER_LENGTH + (long) ord * VECTOR_BYTES, VECTOR_BYTES);
            }
            final ByteBuffer arena = input.stagingArena();
            assertNotNull("the first prefetch must carve the table's one allocation", arena);

            final int slotBytes = input.arenaSlotBytes();
            assertEquals("a slot is one bufferSize window", input.bufferSize(), slotBytes);
            assertTrue(
                "the arena must hold every slot, capacity=" + arena.capacity() + " need=" + 64 * slotBytes,
                arena.capacity() >= 64 * slotBytes
            );
            // The saving this change exists for, stated as the arithmetic rather than as a measurement:
            // per-slot allocation needed blockSize-1 of slack in each of 64 allocations so that an aligned
            // slice existed inside it. One arena needs that slack once.
            assertTrue(
                "the arena must cost less than 64 slack-padded slots, "
                    + arena.capacity()
                    + " vs "
                    + 64 * (slotBytes + input.blockSize() - 1),
                arena.capacity() < 64L * (slotBytes + input.blockSize() - 1)
            );

            // Several further bursts, each ended by a read, must reuse the same allocation.
            for (int repeat = 0; repeat < 5; repeat++) {
                for (final int ord : ords) {
                    input.seek(HEADER_LENGTH + (long) ord * VECTOR_BYTES);
                    input.readFloats(scratch, 0, DIMENSION);
                    assertArrayEquals("burst " + repeat + " ordinal " + ord, vectors[ord], scratch, 0.0f);
                }
                for (final int ord : ords) {
                    input.prefetch(HEADER_LENGTH + (long) ord * VECTOR_BYTES, VECTOR_BYTES);
                }
                assertSame("the table must not be re-carved per burst", arena, input.stagingArena());
            }
            assertTrue(
                "the arena-backed slots must actually have served the reads, " + DirectIOVectorIndexInput.STATS,
                DirectIOVectorIndexInput.STATS.stagedHits() > 0
            );
        }
    }

    /**
     * <b>Re-pinned by D3c, and the inversion is the deliverable.</b> This test used to assert that two
     * slices each carved their own arena ("two slices must not share one arena"), which was true and was
     * precisely the defect: a per-query {@code OffHeapFloatVectorValues} slice carved ~516 KiB on its first
     * prefetch and held it until a collection found the input, so arenas churned at
     * {@code QPS x segments x 516 KiB/s} — 9-13 MiB/s on the node — against about 8 MiB genuinely in use.
     * The arena now belongs to the thread, so two slices interleaving bursts on one thread take it from each
     * other.
     *
     * <p>What may not change is the bytes. The revoked slice's staged table is void, so its reads fall back
     * to the blocking path and return exactly the same vectors — which is the property the sharing is only
     * acceptable because of.
     */
    @SneakyThrows
    public void testTwoSlicesOnOneThreadShareTheThreadsArenaAndTheRevokedOneStillReadsCorrectly() {
        assumeDirectIOWorksHere();
        final Path dir = createTempDir();
        final float[][] vectors = randomVectors(2000);
        final Path path = writeVectorFile(dir, vectors);
        final long region = (long) vectors.length * VECTOR_BYTES;
        final float[] scratch = new float[DIMENSION];

        DirectIOReadPool.resetForTesting();
        DirectIOVectorIndexInput.STATS.reset();
        try (
            DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path, 0, 64, 128 * 1024 + 4096, 0, 2);
            DirectIOVectorIndexInput first = input.slice("first", HEADER_LENGTH, region);
            DirectIOVectorIndexInput second = input.slice("second", HEADER_LENGTH, region)
        ) {
            for (int ord = 0; ord < 32; ord++) {
                first.prefetch((long) ord * VECTOR_BYTES, VECTOR_BYTES);
            }
            final ByteBuffer arena = first.stagingArena();
            assertNotNull("the first burst on this thread carves the thread's arena", arena);

            // The second slice's first prefetch is a new burst on the same thread, so it takes the arena.
            for (int ord = 0; ord < 32; ord++) {
                second.prefetch((long) ord * VECTOR_BYTES, VECTOR_BYTES);
            }
            assertSame("the second slice must take the same arena, not carve a second one", arena, second.stagingArena());
            assertNull("and the first slice's table is void once revoked", first.stagingArena());
            assertNull("the parent never prefetched, so it holds nothing", input.stagingArena());
            assertEquals("one arena for this thread, however many slices used it", 1L, DirectIOVectorIndexInput.STATS.arenasAllocated());
            assertEquals("the second burst reused it", 1L, DirectIOVectorIndexInput.STATS.arenasReused());
            assertEquals("and nothing had to be retired", 0L, DirectIOVectorIndexInput.STATS.arenasRetired());

            for (int ord = 0; ord < 32; ord++) {
                first.seek((long) ord * VECTOR_BYTES);
                first.readFloats(scratch, 0, DIMENSION);
                assertArrayEquals("revoked slice ordinal " + ord, vectors[ord], scratch, 0.0f);
                second.seek((long) ord * VECTOR_BYTES);
                second.readFloats(scratch, 0, DIMENSION);
                assertArrayEquals("holding slice ordinal " + ord, vectors[ord], scratch, 0.0f);
            }
            assertEquals("reads on the owner thread are never foreign", 0L, DirectIOVectorIndexInput.STATS.foreignThreadReads());
        }
    }

    /**
     * A coalesced group wider than an arena slot keeps its own allocation and still reads correctly. The
     * arena is sized for the measured common case — S3.1 put the mean aligned span at 7,310 B for a 3,072 B
     * vector — and this is the case it is deliberately not sized for.
     */
    @SneakyThrows
    public void testAGroupWiderThanASlotKeepsItsOwnBufferAndReadsCorrectly() {
        assumeDirectIOWorksHere();
        final Path dir = createTempDir();
        final float[][] vectors = randomVectors(2000);
        final Path path = writeVectorFile(dir, vectors);
        // Clustered in one narrow window, so prefetchLikeHelper coalesces them into groups far wider than
        // the two blocks a slot holds.
        final int[] ords = clusteredOrds(64, 64, 500);
        final float[] scratch = new float[DIMENSION];

        DirectIOReadPool.resetForTesting();
        DirectIOVectorIndexInput.STATS.reset();
        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path, 0, 64, 128 * 1024 + 4096, 0, 2)) {
            assertNotNull(input);
            final int groups = prefetchLikeHelper(input, HEADER_LENGTH, VECTOR_BYTES, ords, 128 * 1024);
            assertTrue("the window must coalesce into few wide groups, groups=" + groups, groups < ords.length / 4);
            assertNotNull("a table is still carved for the groups that do fit", input.stagingArena());
            for (final int ord : ords) {
                input.seek(HEADER_LENGTH + (long) ord * VECTOR_BYTES);
                input.readFloats(scratch, 0, DIMENSION);
                assertArrayEquals("clustered ordinal " + ord, vectors[ord], scratch, 0.0f);
            }
            assertTrue(
                "the wide groups must have been staged and served, " + DirectIOVectorIndexInput.STATS,
                DirectIOVectorIndexInput.STATS.stagedHits() > 0
            );
        }
    }

    /**
     * The read-ahead ring is one allocation too, and needs no per-slot fall-back at all: every window it
     * submits is exactly one {@code bufferSize}, which is exactly what a slot holds.
     */
    @SneakyThrows
    public void testTheReadaheadRingIsOneAllocation() {
        assumeDirectIOWorksHere();
        final Path dir = createTempDir();
        final float[][] vectors = randomVectors(1500);
        final Path path = writeVectorFile(dir, vectors);

        DirectIOReadPool.resetForTesting();
        DirectIOVectorIndexInput.STATS.reset();
        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path, 0, 64, 128 * 1024 + 4096, 8, 2)) {
            assertNotNull(input);
            assertNull("the ring is not allocated until it arms", input.readaheadArena());
            sequentialScan(input, vectors);
            final ByteBuffer arena = input.readaheadArena();
            assertNotNull("a forward scan must arm the ring and carve it", arena);
            assertTrue(
                "the arena must hold all eight windows, capacity=" + arena.capacity(),
                arena.capacity() >= 8 * input.arenaSlotBytes()
            );
            assertTrue(
                "the ring must have served the scan, " + DirectIOVectorIndexInput.STATS,
                DirectIOVectorIndexInput.STATS.readaheadHits() > 100
            );
        }
    }

    /**
     * Fills the read pool so that exactly one submit can still be accepted and will not run until
     * {@code release} is counted down. Deterministic rather than arithmetic: submit until the pool refuses,
     * which means every thread is busy and the queue is full, then drop one queued task to free exactly one
     * slot. Computing the count from {@code maximumPoolSize + remainingCapacity} races against workers
     * picking tasks up.
     *
     * @return the pool, saturated but for one slot
     */
    private ThreadPoolExecutor saturatePoolLeavingOneSlot(final CountDownLatch release) {
        final ThreadPoolExecutor pool = (ThreadPoolExecutor) DirectIOReadPool.executor();
        assertNotNull(pool);
        pool.prestartAllCoreThreads();
        final Callable<Void> blocker = () -> {
            release.await();
            return null;
        };
        boolean refused = false;
        // Bounded so a pool that somehow never refuses fails the test rather than hanging it.
        for (int i = 0; i < 100_000; i++) {
            try {
                pool.submit(blocker);
            } catch (RejectedExecutionException e) {
                refused = true;
                break;
            }
        }
        assertTrue("the pool must be boundable for this test to mean anything", refused);
        assertNotNull("dropping one queued task must free exactly one slot", pool.getQueue().poll());
        return pool;
    }

    // -----------------------------------------------------------------------------------------------
    // 8. Phase 15 D3: the direct-memory budget (phase 13 audit FIX 3)
    // -----------------------------------------------------------------------------------------------

    /**
     * Past {@code knn.direct_io.direct_memory_budget} a prefetch declines and the bytes are served by
     * blocking reads — <b>and the reads are still right</b>, which is the only property that may not be
     * traded for a bound. Phase 13's audit called this the one unambiguous structural defect in the design:
     * {@code OffHeapFloatVectorValues} is not {@code Closeable}, so a staging table lives until the GC
     * reclaims it, and the node sat at 1.69 GB of live direct buffers against a 2g
     * {@code MaxDirectMemorySize} with nothing bounding it.
     *
     * <p>Driven with a budget of {@code 1b}, which no allocation can fit, so the first table is refused —
     * "declines after N tables" with N = 0, because N is a function of the budget and the interesting value
     * of N is the smallest one. The budget is then raised live and the same input is asked again, which is
     * the half that matters operationally: a node that has hit the budget must recover when given more,
     * without a restart.
     */
    @SneakyThrows
    public void testTheDirectMemoryBudgetDeclinesPrefetchAndTheReadsAreStillCorrect() {
        assumeDirectIOWorksHere();
        final Path dir = createTempDir();
        final float[][] vectors = randomVectors(2000);
        final Path path = writeVectorFile(dir, vectors);
        final int[] ords = scatteredOrds(24, vectors.length);
        final float[] scratch = new float[DIMENSION];

        // A budget no allocation can fit. The ring is off so that this measures the burst table alone.
        withNodeSettings(Settings.builder().put(KNNSettings.KNN_DIRECT_IO_READAHEAD_WINDOW, 0).build());
        DirectIOVectorIndexInput.setDirectMemoryBudgetForTesting(1L);
        DirectIOVectorIndexInput.STATS.reset();
        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(input);
            for (final int ord : ords) {
                input.prefetch(HEADER_LENGTH + (long) ord * VECTOR_BYTES, VECTOR_BYTES);
            }
            assertNull("no staging table may be allocated under a budget of one byte", input.stagingArena());
            assertEquals("every prefetch in the burst must be declined", ords.length, DirectIOVectorIndexInput.STATS.stageDeclined());
            assertEquals("and the reason must be the budget", ords.length, DirectIOVectorIndexInput.STATS.declinedBudget());
            assertEquals("so nothing was staged", 0L, DirectIOVectorIndexInput.STATS.stagedReads());

            // The property the bound may not cost: the bytes.
            for (final int ord : ords) {
                input.seek(HEADER_LENGTH + (long) ord * VECTOR_BYTES);
                input.readFloats(scratch, 0, DIMENSION);
                assertArrayEquals("a declined prefetch must not change a single byte", vectors[ord], scratch, 0.0f);
            }
            assertTrue("the bytes came from blocking reads instead", DirectIOVectorIndexInput.STATS.blockingReads() > 0);
            assertEquals(
                "servedBytes is still exactly the vectors handed out",
                (long) ords.length * VECTOR_BYTES,
                DirectIOVectorIndexInput.STATS.servedBytes()
            );
            // The blocking path's own buffer is one bufferSize per input and is tracked but never declined --
            // it is what every other decline falls back to, so declining it would leave nothing to read with.
            assertTrue("the blocking window is accounted for", DirectIOVectorIndexInput.STATS.directMemoryLive() > 0);
        }

        // Raised. The same input class, a fresh input, no restart: prefetch must resume.
        DirectIOVectorIndexInput.setDirectMemoryBudgetForTesting(64L * 1024 * 1024);
        DirectIOVectorIndexInput.STATS.reset();
        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(input);
            for (final int ord : ords) {
                input.prefetch(HEADER_LENGTH + (long) ord * VECTOR_BYTES, VECTOR_BYTES);
            }
            assertNotNull("with budget to spare the table must be allocated", input.stagingArena());
            assertEquals("and nothing may be declined on the budget", 0L, DirectIOVectorIndexInput.STATS.declinedBudget());
            assertEquals("nothing is declined at all", 0L, DirectIOVectorIndexInput.STATS.stageDeclined());
            for (final int ord : ords) {
                input.seek(HEADER_LENGTH + (long) ord * VECTOR_BYTES);
                input.readFloats(scratch, 0, DIMENSION);
                assertArrayEquals(vectors[ord], scratch, 0.0f);
            }
            // stagedHits is the one assertion that forces the staged reads to have actually run: a hit only
            // happens after await() retires the future. Asserting stagedReads before any read would race the
            // pool.
            assertTrue("and served out of the table", DirectIOVectorIndexInput.STATS.stagedHits() > 0);
            assertTrue("so reads really were staged", DirectIOVectorIndexInput.STATS.stagedReads() > 0);
        }
    }

    /**
     * The accounting itself: a staging table charges exactly what it allocates, and the peak follows the
     * live gauge up. Without this the budget is a number compared against a number nobody checked.
     */
    @SneakyThrows
    public void testDirectMemoryAccountingChargesWhatItAllocates() {
        assumeDirectIOWorksHere();
        final Path dir = createTempDir();
        final float[][] vectors = randomVectors(500);
        final Path path = writeVectorFile(dir, vectors);

        withNodeSettings(Settings.builder().put(KNNSettings.KNN_DIRECT_IO_READAHEAD_WINDOW, 0).build());
        DirectIOVectorIndexInput.STATS.reset();
        // Earlier tests in this JVM left charged buffers that are unreachable but not yet drained, and a
        // drain during this test would lower the baseline under it. So: collect and drain until two
        // consecutive drains agree, then measure. Enqueueing a phantom reference is asynchronous with the
        // collection that made it unreachable, so one System.gc() plus one reclaim() is not enough and a
        // single pass was observed to miss 1.5 MB. This is the one test where forcing a collection earns its
        // keep, because the thing under test is a reachability-driven counter.
        long settled = -1;
        for (int attempt = 0; attempt < 20 && settled != DirectIOVectorIndexInput.reclaim(); attempt++) {
            settled = DirectIOVectorIndexInput.STATS.directMemoryLive();
            System.gc();
            Thread.sleep(20);
        }
        final long before = DirectIOVectorIndexInput.reclaim();
        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(input);
            input.prefetch(HEADER_LENGTH, VECTOR_BYTES);
            final ByteBuffer arena = input.stagingArena();
            assertNotNull(arena);

            // The arena is over-allocated by blockSize - 1 so that a block-aligned slice exists inside it,
            // and the charge is the allocation rather than the aligned slice, because the allocation is what
            // the JVM counts against MaxDirectMemorySize.
            final long expectedArena = (long) input.maxStagedRanges() * input.arenaSlotBytes() + input.blockSize() - 1;
            assertEquals(
                "the staging table must charge its whole allocation",
                before + expectedArena,
                DirectIOVectorIndexInput.STATS.directMemoryLive()
            );
            assertTrue(
                "and the peak must have followed it up",
                DirectIOVectorIndexInput.STATS.directMemoryPeak() >= before + expectedArena
            );
        }
    }

    /**
     * A budget of {@code 0b} is off, which is the pre-phase-15 behaviour and must stay reachable: an
     * operator whose workload the budget mis-serves has to be able to turn it off without downgrading.
     */
    @SneakyThrows
    public void testAZeroBudgetIsOff() {
        assumeDirectIOWorksHere();
        final Path dir = createTempDir();
        final Path path = writeVectorFile(dir, randomVectors(500));

        DirectIOVectorIndexInput.setDirectMemoryBudgetForTesting(0L);
        DirectIOVectorIndexInput.STATS.reset();
        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(input);
            input.prefetch(HEADER_LENGTH, VECTOR_BYTES);
            assertNotNull("a budget of zero must not bound anything", input.stagingArena());
            assertEquals(0L, DirectIOVectorIndexInput.STATS.declinedBudget());
        }
    }

    /**
     * The shipped budget is half the JVM's direct-memory wall, and the wall is read rather than assumed.
     * Pinned because it is the one number in D3 that is a judgement rather than a measurement.
     */
    public void testTheBudgetDefaultIsHalfTheDirectMemoryWall() {
        assertEquals(
            "the budget is exactly half of the resolved wall",
            KNNSettings.KNN_DIRECT_IO_DIRECT_MEMORY_WALL_BYTES / 2,
            DirectIOVectorIndexInput.DIRECT_MEMORY_BUDGET_BYTES
        );
        assertTrue("and it is a positive number on this JVM", DirectIOVectorIndexInput.DIRECT_MEMORY_BUDGET_BYTES > 0);
        assertEquals(
            "and it is what a node bounds itself by",
            DirectIOVectorIndexInput.DIRECT_MEMORY_BUDGET_BYTES,
            DirectIOVectorIndexInput.directMemoryBudgetBytes()
        );
    }

    // -----------------------------------------------------------------------------------------------
    // 9. Phase 15b D3c: the staging arena belongs to the thread, so live direct memory is bounded by
    // concurrent bursts rather than by when the collector next runs.
    //
    // The node measured the defect twice on 2026-10-06 with jar 4a8cf9e9 (16 clients, -Xmx24g,
    // -XX:MaxDirectMemorySize=2g): at the 1 GB default budget live_bytes reached 1,068-1,087 MB within two
    // minutes, staged_reads froze at 159,137, declined_budget ran to 614k and twenty seconds later
    // live_bytes fell to 0 in one step on a young collection -- the budget wedging the feature between
    // collections rather than bounding it; and with the budget off, peak_bytes ratcheted to 7.46 GB against
    // a 2 GB wall because the phantom queue drained only on stats reads. The cause was that every slice()
    // and clone() carved its own ~516 KiB arena and held it for the input's life, which on a per-query
    // OffHeapFloatVectorValues slice ends only when a collection finds it. These tests assert the fix.
    // -----------------------------------------------------------------------------------------------

    /** Drains the phantom queue until two consecutive drains agree, so a baseline is not moving under a test. */
    @SneakyThrows
    private long settledDirectMemory() {
        long settled = -1;
        for (int attempt = 0; attempt < 20 && settled != DirectIOVectorIndexInput.reclaim(); attempt++) {
            settled = DirectIOVectorIndexInput.STATS.directMemoryLive();
            System.gc();
            Thread.sleep(20);
        }
        return DirectIOVectorIndexInput.reclaim();
    }

    /**
     * <b>The deliverable, as one assertion.</b> Twenty thousand short-lived inputs that each run exactly one
     * burst and are never closed — the production shape, because Lucene's {@code OffHeapFloatVectorValues}
     * is not {@code Closeable} — spread over a fixed set of four threads, must allocate at most
     * {@code 4 + arenas_retired} arenas. Before D3c this would have been 20,000 of them, which at
     * ~516 KiB each is the 9-13 MiB/s of direct-buffer churn the node measured and the budget then declined
     * on.
     *
     * <p>The budget is set to four arenas plus slack <em>above the settled baseline</em>, so that
     * {@code declined_budget} staying at 0 means something: a budget that bounds only four arenas cannot be
     * satisfied by 20,000 of them, and the only way to pass is to reuse.
     */
    @SneakyThrows
    public void testTwentyThousandSingleBurstInputsOnFourThreadsAllocateFourArenas() {
        assumeDirectIOWorksHere();
        final Path dir = createTempDir();
        final float[][] vectors = randomVectors(2000);
        final Path path = writeVectorFile(dir, vectors);
        final long region = (long) vectors.length * VECTOR_BYTES;
        final int threads = 4;
        final int inputsPerThread = 5000;

        DirectIOReadPool.resetForTesting();
        DirectIOStagingLease.resetForTesting();
        DirectIOVectorIndexInput.STATS.reset();

        // One arena is staged_ranges x arenaSlotBytes plus a block of alignment slack; a thread also holds
        // one blocking window. Four of each, plus a megabyte of slack for the wide-group buffers a coalesced
        // group may need, is the whole footprint this test may use.
        try (DirectIOVectorIndexInput probe = DirectIOVectorIndexInput.open(path, 0, 64, 128 * 1024 + 4096, 0, 2)) {
            assertNotNull(probe);
            final long perThread = (long) probe.maxStagedRanges() * probe.arenaSlotBytes() + probe.blockSize() - 1 + probe.bufferSize()
                + probe.blockSize() - 1;
            final long budget = settledDirectMemory() + threads * perThread + (1L << 20);
            withNodeSettings(Settings.builder().put(KNNSettings.KNN_DIRECT_IO_READAHEAD_WINDOW, 0).build());
            DirectIOVectorIndexInput.setDirectMemoryBudgetForTesting(budget);
            DirectIOVectorIndexInput.STATS.reset();

            final CountDownLatch start = new CountDownLatch(1);
            final Thread[] workers = new Thread[threads];
            final Throwable[] failures = new Throwable[threads];
            for (int t = 0; t < threads; t++) {
                final int worker = t;
                workers[t] = new Thread(() -> {
                    try {
                        start.await();
                        // One channel per thread, then one slice per "query": slice() opens no file
                        // descriptor, which is exactly why the production path creates them per query per
                        // segment and exactly why their arenas were the problem.
                        final DirectIOVectorIndexInput owner = DirectIOVectorIndexInput.open(path, 0, 64, 128 * 1024 + 4096, 0, 2);
                        assertNotNull(owner);
                        final float[] scratch = new float[DIMENSION];
                        for (int i = 0; i < inputsPerThread; i++) {
                            // Deliberately NOT closed, and deliberately dropped on the floor after one
                            // burst. This is the lifecycle the fix is for.
                            final DirectIOVectorIndexInput perQuery = owner.slice("q" + worker + "-" + i, HEADER_LENGTH, region);
                            final int ord = (worker * inputsPerThread + i) % vectors.length;
                            perQuery.prefetch((long) ord * VECTOR_BYTES, VECTOR_BYTES);
                            perQuery.seek((long) ord * VECTOR_BYTES);
                            perQuery.readFloats(scratch, 0, DIMENSION);
                            assertArrayEquals("worker " + worker + " input " + i, vectors[ord], scratch, 0.0f);
                        }
                        owner.close();
                    } catch (Throwable e) {
                        failures[worker] = e;
                    }
                }, "d3c-worker-" + t);
                workers[t].start();
            }
            start.countDown();
            for (final Thread w : workers) {
                w.join();
            }
            for (int t = 0; t < threads; t++) {
                if (failures[t] != null) {
                    throw new AssertionError("worker " + t + " failed", failures[t]);
                }
            }

            final long allocated = DirectIOVectorIndexInput.STATS.arenasAllocated();
            final long retired = DirectIOVectorIndexInput.STATS.arenasRetired();
            final long reused = DirectIOVectorIndexInput.STATS.arenasReused();
            assertTrue(
                "an arena per thread, not per input: allocated="
                    + allocated
                    + " retired="
                    + retired
                    + ", "
                    + DirectIOVectorIndexInput.STATS,
                allocated <= threads + retired
            );
            assertEquals(
                "every other burst must have taken its thread's arena back, " + DirectIOVectorIndexInput.STATS,
                (long) threads * inputsPerThread - allocated,
                reused
            );
            assertEquals(
                "and a budget of four arenas must never bind, " + DirectIOVectorIndexInput.STATS,
                0L,
                DirectIOVectorIndexInput.STATS.declinedBudget()
            );
            assertEquals("each burst read on the thread that staged it", 0L, DirectIOVectorIndexInput.STATS.foreignThreadReads());
            assertTrue("the reads were really staged, " + DirectIOVectorIndexInput.STATS, DirectIOVectorIndexInput.STATS.stagedHits() > 0);
        }
    }

    /**
     * Takeback correctness. Thread T bursts on input A, then bursts on input B, then reads a range A
     * staged: A's read must take the blocking path and return A's bytes from the file — never whatever B's
     * burst has since put in that buffer.
     *
     * <p>A and B are given <b>the same slot</b> deliberately: both stage their first range into arena slot
     * 0, so if the generation check were missing A's read would decode B's vector and the assertion would
     * fail with a vector that exists rather than with garbage.
     */
    @SneakyThrows
    public void testAReadAfterItsLeaseWasTakenBackGetsItsOwnBytesAndNotTheNextBursts() {
        assumeDirectIOWorksHere();
        final Path dir = createTempDir();
        final float[][] vectors = randomVectors(2000);
        final Path path = writeVectorFile(dir, vectors);
        final long region = (long) vectors.length * VECTOR_BYTES;
        final float[] scratch = new float[DIMENSION];
        final int ordA = 11;
        final int ordB = 1511;

        DirectIOReadPool.resetForTesting();
        DirectIOStagingLease.resetForTesting();
        DirectIOVectorIndexInput.STATS.reset();
        try (
            DirectIOVectorIndexInput owner = DirectIOVectorIndexInput.open(path, 0, 64, 128 * 1024 + 4096, 0, 2);
            DirectIOVectorIndexInput a = owner.slice("a", HEADER_LENGTH, region);
            DirectIOVectorIndexInput b = owner.slice("b", HEADER_LENGTH, region)
        ) {
            a.prefetch((long) ordA * VECTOR_BYTES, VECTOR_BYTES);
            assertNotNull("A holds the lease", a.stagingArena());

            b.prefetch((long) ordB * VECTOR_BYTES, VECTOR_BYTES);
            assertNull("B's burst revoked A's table", a.stagingArena());
            assertNotNull("and B now holds the same arena", b.stagingArena());

            final long blockingBefore = DirectIOVectorIndexInput.STATS.blockingReads();
            a.seek((long) ordA * VECTOR_BYTES);
            a.readFloats(scratch, 0, DIMENSION);
            assertArrayEquals("the revoked read must return A's own vector", vectors[ordA], scratch, 0.0f);
            assertFalse("and must not be B's", java.util.Arrays.equals(vectors[ordB], scratch));
            assertTrue(
                "it must have come from a blocking read, not from the arena",
                DirectIOVectorIndexInput.STATS.blockingReads() > blockingBefore
            );

            b.seek((long) ordB * VECTOR_BYTES);
            b.readFloats(scratch, 0, DIMENSION);
            assertArrayEquals("and the holder still reads out of the arena correctly", vectors[ordB], scratch, 0.0f);
        }
    }

    /**
     * A range that would not quiesce retires the whole arena, because a read that may still be writing into
     * one slot is a read that may still be writing into the allocation every slot is carved from. The next
     * burst must get a fresh arena, the late write must land only in the retired one, and every read must be
     * right throughout.
     *
     * <p>The stuck read is produced by saturating the pool rather than by a stubbed {@code FileChannel}:
     * {@link DirectIOVectorIndexInput.Handle} is built inside {@code open()}, and a channel-injection seam
     * for one test is more surface than the test is worth when the saturated pool is the <em>measured</em>
     * production route to this state ({@code staged_wait_timeouts}, phase 13 audit). The latch is released at
     * the end, which is when the queued read actually runs and writes into the retired buffer.
     */
    @SneakyThrows
    public void testARangeThatWillNotQuiesceRetiresTheArenaAndTheLateWriteHarmsNothing() {
        assumeDirectIOWorksHere();
        final Path dir = createTempDir();
        final float[][] vectors = randomVectors(2000);
        final Path path = writeVectorFile(dir, vectors);
        final float[] scratch = new float[DIMENSION];

        DirectIOReadPool.resetForTesting();
        DirectIOStagingLease.resetForTesting();
        final CountDownLatch release = new CountDownLatch(1);
        saturatePoolLeavingOneSlot(release);

        DirectIOVectorIndexInput.STATS.reset();
        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path, 0, 64, 128 * 1024 + 4096, 0, 2)) {
            assertNotNull(input);
            final long offsetA = (long) 5 * VECTOR_BYTES;
            input.prefetch(HEADER_LENGTH + offsetA, VECTOR_BYTES);
            assumeTrue("the pool refused the prefetch", DirectIOVectorIndexInput.STATS.stageDeclined() == 0);
            final ByteBuffer first = input.stagingArena();
            assertNotNull(first);
            assertEquals(1L, DirectIOVectorIndexInput.STATS.arenasAllocated());

            // The bounded wait gives up, which abandons the range: from here its buffer may still be written.
            input.seek(HEADER_LENGTH + offsetA);
            input.readFloats(scratch, 0, DIMENSION);
            assertArrayEquals(vectors[5], scratch, 0.0f);
            assertEquals("the wait must have been given up on", 1L, DirectIOVectorIndexInput.STATS.stagedWaitTimeouts());

            // The next burst on this thread cannot reuse that arena.
            input.prefetch(HEADER_LENGTH + (long) 9 * VECTOR_BYTES, VECTOR_BYTES);
            assertEquals("the abandoned range must retire the arena", 1L, DirectIOVectorIndexInput.STATS.arenasRetired());
            assertEquals("so the next burst carves a fresh one", 2L, DirectIOVectorIndexInput.STATS.arenasAllocated());
            assertEquals("and nothing was reused", 0L, DirectIOVectorIndexInput.STATS.arenasReused());
            assertNotSame("the fresh arena must not be the retired one", first, input.stagingArena());

            // Let the stuck read run. It writes into the retired buffer, which nothing reads from again.
            release.countDown();
            Thread.sleep(200);
            for (int ord = 0; ord < 40; ord++) {
                input.seek(HEADER_LENGTH + (long) ord * VECTOR_BYTES);
                input.readFloats(scratch, 0, DIMENSION);
                assertArrayEquals("after the late write, ordinal " + ord, vectors[ord], scratch, 0.0f);
            }
        } finally {
            release.countDown();
            DirectIOReadPool.resetForTesting();
        }
    }

    /**
     * Owner confinement: a read arriving on a thread other than the one that staged the burst does not look
     * at the table at all. It takes the blocking path, returns the right bytes, and is counted — because it
     * is the premise of the lease model being violated rather than the lease model working, and an operator
     * should be able to see that.
     *
     * <p>This is also the rule that makes the generation check race free without a seqlock: the one thread
     * that can observe a stale generation is the one thread that cannot concurrently change it.
     */
    @SneakyThrows
    public void testAReadFromAForeignThreadTakesTheBlockingPathAndIsCounted() {
        assumeDirectIOWorksHere();
        final Path dir = createTempDir();
        final float[][] vectors = randomVectors(2000);
        final Path path = writeVectorFile(dir, vectors);
        final float[] scratch = new float[DIMENSION];
        final int ord = 777;

        DirectIOReadPool.resetForTesting();
        DirectIOStagingLease.resetForTesting();
        DirectIOVectorIndexInput.STATS.reset();
        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path, 0, 64, 128 * 1024 + 4096, 0, 2)) {
            assertNotNull(input);
            final Throwable[] failure = new Throwable[1];
            final Thread stager = new Thread(() -> {
                try {
                    input.prefetch(HEADER_LENGTH + (long) ord * VECTOR_BYTES, VECTOR_BYTES);
                } catch (Throwable e) {
                    failure[0] = e;
                }
            }, "d3c-stager");
            stager.start();
            stager.join();
            if (failure[0] != null) {
                throw new AssertionError("the staging thread failed", failure[0]);
            }
            assumeTrue(
                "the prefetch was declined, so there is no table to be foreign to",
                DirectIOVectorIndexInput.STATS.stageDeclined() == 0
            );

            // This thread is not the lease owner, so the table is not consulted.
            final long blockingBefore = DirectIOVectorIndexInput.STATS.blockingReads();
            input.seek(HEADER_LENGTH + (long) ord * VECTOR_BYTES);
            input.readFloats(scratch, 0, DIMENSION);
            assertArrayEquals("a foreign read must still be exactly right", vectors[ord], scratch, 0.0f);
            assertTrue("it must be counted, " + DirectIOVectorIndexInput.STATS, DirectIOVectorIndexInput.STATS.foreignThreadReads() > 0);
            assertTrue("and served by a blocking read", DirectIOVectorIndexInput.STATS.blockingReads() > blockingBefore);
            assertEquals("nothing may have been served out of the foreign table", 0L, DirectIOVectorIndexInput.STATS.stagedHits());
        }
    }

    /**
     * The blocking window is one buffer per thread since D3c, so two inputs on one thread share it — and the
     * cached-range check therefore cannot hang on the lease generation alone. <b>It hangs on which input
     * last filled the window</b>, because two inputs doing nothing but blocking reads never move the
     * generation: the merge path, the faiss first pass and an exact search with prefetch off are all that
     * shape. Without the owner token, A's second read would be served B's bytes.
     */
    @SneakyThrows
    public void testTwoInputsSharingAThreadsBlockingWindowNeverServeEachOthersBytes() {
        assumeDirectIOWorksHere();
        final Path dir = createTempDir();
        final float[][] vectors = randomVectors(2000);
        final Path path = writeVectorFile(dir, vectors);
        final long region = (long) vectors.length * VECTOR_BYTES;
        final float[] scratch = new float[DIMENSION];
        // Far apart, so each sits in a different block-aligned window and the second read evicts the first.
        final int ordA = 3;
        final int ordB = 1900;

        DirectIOReadPool.resetForTesting();
        DirectIOStagingLease.resetForTesting();
        DirectIOVectorIndexInput.STATS.reset();
        // The ring off and no prefetch at all: every read here is a blocking read, which is the shape that
        // never moves the lease generation.
        withNodeSettings(Settings.builder().put(KNNSettings.KNN_DIRECT_IO_READAHEAD_WINDOW, 0).build());
        try (
            DirectIOVectorIndexInput owner = DirectIOVectorIndexInput.open(path);
            DirectIOVectorIndexInput a = owner.slice("a", HEADER_LENGTH, region);
            DirectIOVectorIndexInput b = owner.slice("b", HEADER_LENGTH, region)
        ) {
            for (int repeat = 0; repeat < 20; repeat++) {
                a.seek((long) ordA * VECTOR_BYTES);
                a.readFloats(scratch, 0, DIMENSION);
                assertArrayEquals("A on repeat " + repeat, vectors[ordA], scratch, 0.0f);

                b.seek((long) ordB * VECTOR_BYTES);
                b.readFloats(scratch, 0, DIMENSION);
                assertArrayEquals("B on repeat " + repeat, vectors[ordB], scratch, 0.0f);

                // A again, from the same position: the window now holds B's bytes, so A must re-read rather
                // than be served the range it cached.
                a.seek((long) ordA * VECTOR_BYTES);
                a.readFloats(scratch, 0, DIMENSION);
                assertArrayEquals("A re-read on repeat " + repeat, vectors[ordA], scratch, 0.0f);
            }
            assertEquals("the ring must not have armed on this access shape", 0L, DirectIOVectorIndexInput.STATS.readaheadReads());
        }
    }

    /**
     * Two defects the node found with {@code direct_memory_budget=0b}, as one test.
     *
     * <ol>
     *   <li>{@code reclaim()} ran only inside {@code fitsBudget}, which short-circuits when the budget is
     *       off, so the phantom queue drained on stats reads alone and {@code peak_bytes} ratcheted to
     *       <b>7.46 GB, 9.0 GB and 28 GB</b> against a 2 GB wall. It now drains on every allocation: a
     *       charge made unreachable and collected is returned by the next {@code trackedAllocate} with no
     *       stats read in between.</li>
     *   <li>Neither gauge may be reported above the JVM's wall, whatever the raw counter says.</li>
     * </ol>
     */
    @SneakyThrows
    public void testWithTheBudgetOffTheGaugeStillDrainsOnAllocationAndIsNeverAboveTheWall() {
        assumeDirectIOWorksHere();
        final Path dir = createTempDir();
        final Path path = writeVectorFile(dir, randomVectors(500));

        withNodeSettings(Settings.builder().put(KNNSettings.KNN_DIRECT_IO_READAHEAD_WINDOW, 0).build());
        DirectIOVectorIndexInput.setDirectMemoryBudgetForTesting(0L);
        DirectIOStagingLease.resetForTesting();
        DirectIOVectorIndexInput.STATS.reset();

        // Charge something and then make it unreachable, so there is a charge waiting to be returned.
        final long baseline = settledDirectMemory();
        try (DirectIOVectorIndexInput doomed = DirectIOVectorIndexInput.open(path, 0, 64, 128 * 1024 + 4096, 0, 2)) {
            doomed.prefetch(HEADER_LENGTH, VECTOR_BYTES);
            assertNotNull("the budget is off, so nothing may be declined", doomed.stagingArena());
        }
        DirectIOStagingLease.resetForTesting();
        final long charged = DirectIOVectorIndexInput.STATS.directMemoryLive();
        assertTrue("the arena must have been charged, " + charged + " vs baseline " + baseline, charged > baseline);
        for (int attempt = 0; attempt < 20 && DirectIOVectorIndexInput.STATS.directMemoryLive() >= charged; attempt++) {
            System.gc();
            Thread.sleep(20);
        }
        // Still charged as far as the gauge is concerned: nothing has drained the queue yet, and the point is
        // that no stats read is allowed to be what drains it.
        assumeTrue(
            "the collector has not enqueued the charge yet, so there is nothing for an allocation to drain",
            DirectIOVectorIndexInput.STATS.directMemoryLive() >= charged
        );

        try (DirectIOVectorIndexInput next = DirectIOVectorIndexInput.open(path, 0, 64, 128 * 1024 + 4096, 0, 2)) {
            // One allocation, no stats read, no reclaim() call: the drain has to happen inside it.
            next.prefetch(HEADER_LENGTH, VECTOR_BYTES);
            assertNotNull(next.stagingArena());
            assertTrue(
                "an allocation must drain the queue even with the budget off, live=" + DirectIOVectorIndexInput.STATS.directMemoryLive(),
                DirectIOVectorIndexInput.STATS.directMemoryLive() < charged + (long) next.maxStagedRanges() * next.arenaSlotBytes()
            );
        }

        assertTrue(
            "live_bytes may never be reported above the wall",
            DirectIOVectorIndexInput.STATS.directMemoryLive() <= KNNSettings.KNN_DIRECT_IO_DIRECT_MEMORY_WALL_BYTES
        );
        assertTrue(
            "nor may peak_bytes -- the node read 7.46 GB against a 2 GB wall",
            DirectIOVectorIndexInput.STATS.directMemoryPeak() <= KNNSettings.KNN_DIRECT_IO_DIRECT_MEMORY_WALL_BYTES
        );
        assertEquals("and 0b must still decline nothing", 0L, DirectIOVectorIndexInput.STATS.declinedBudget());
    }

}

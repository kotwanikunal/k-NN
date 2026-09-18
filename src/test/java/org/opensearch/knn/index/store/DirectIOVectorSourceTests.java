/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import com.carrotsearch.randomizedtesting.annotations.ThreadLeakFilters;
import lombok.SneakyThrows;
import org.apache.lucene.index.FloatVectorValues;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.query.scorers.VectorScorerMode;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.OpenOption;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * The Direct I/O byte source, tested against real files on the test filesystem.
 * <p>
 * Every test builds a file shaped like Lucene's flat vector file — a codec header, one vector region, a
 * 16 byte footer — with a <b>header length that is not a multiple of the block size or of
 * {@code gcd(vectorBytes, blockSize)}</b>, because that is the case the buffer-sizing rule has to survive
 * and the case a real {@code .vec} presents.
 * <p>
 * The tests that need an open file handle are skipped, not failed, on a filesystem that refuses
 * {@code O_DIRECT} (tmpfs and several network filesystems answer {@code EINVAL}), since that says nothing
 * about this code. {@link #testOpenDeclinesWhenTheRegionIsNotWhereItWasDerivedToBe} and the argument
 * checks run everywhere.
 * <p>
 * Staging runs on {@link DirectIOReadPool}, whose daemon threads outlive a suite by design, so this suite
 * carries the same thread filter {@link DirectIOReadPoolTests} does.
 */
@ThreadLeakFilters(defaultFilters = true, filters = { DirectIOReadPoolTests.ReadPoolThreadFilter.class })
public class DirectIOVectorSourceTests extends KNNTestCase {

    private static final int DIMENSION = 8;
    private static final int VECTOR_BYTES = DIMENSION * Float.BYTES;
    /** Deliberately odd, so the vector region starts unaligned to any block or gcd boundary. */
    private static final int HEADER_LENGTH = 41;
    /**
     * Block- and gcd-unaligned like {@link #HEADER_LENGTH}, but a multiple of four, which is the case
     * every real {@code .vec} presents and the only one that takes the bulk decode. 41 is not, so a test
     * that wants the bulk path has to ask for this explicitly.
     */
    private static final int ALIGNED_HEADER_LENGTH = 44;

    /**
     * Skips the calling test when this filesystem will not open a file with {@code O_DIRECT}, which is a
     * property of the environment rather than of the code under test.
     */
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

    /** Writes header, {@code vectors}, footer — the layout {@link DirectIOVectorSource} derives against. */
    private Path writeVectorFile(final List<float[]> vectors, final int trailingPadding) {
        return writeVectorFile(vectors, trailingPadding, HEADER_LENGTH);
    }

    /** As {@link #writeVectorFile(List, int)}, with the header length chosen by the caller. */
    @SneakyThrows
    private Path writeVectorFile(final List<float[]> vectors, final int trailingPadding, final int headerLength) {
        final Path path = createTempDir().resolve("_0_Test_0.vec");
        final ByteBuffer buffer = ByteBuffer.allocate(headerLength + vectors.size() * VECTOR_BYTES + trailingPadding + 16)
            .order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < headerLength; i++) {
            buffer.put((byte) (0xC0 + i));
        }
        for (final float[] vector : vectors) {
            for (final float value : vector) {
                buffer.putFloat(value);
            }
        }
        buffer.put(new byte[trailingPadding]);
        buffer.put(new byte[16]);
        Files.write(path, buffer.array());
        return path;
    }

    private static List<float[]> randomVectors(final int count) {
        final Random random = new Random(42);
        final List<float[]> vectors = new ArrayList<>(count);
        for (int i = 0; i < count; i++) {
            final float[] vector = new float[DIMENSION];
            for (int d = 0; d < DIMENSION; d++) {
                vector[d] = random.nextFloat() * 100 - 50;
            }
            vectors.add(vector);
        }
        return vectors;
    }

    private static FloatVectorValues reference(final List<float[]> vectors) {
        return FloatVectorValues.fromFloats(vectors, DIMENSION);
    }

    @SneakyThrows
    public void testEveryOrdinalReadsBackTheBytesThatWereWritten() {
        assumeDirectIOWorksHere();
        // 600 vectors of 32 bytes is 19200 bytes, so the region spans several blocks and the last ordinal
        // sits in a partially filled one - the case a short read would show up in.
        final List<float[]> vectors = randomVectors(600);
        final Path path = writeVectorFile(vectors, 0);

        try (DirectIOVectorSource source = DirectIOVectorSource.open(path, reference(vectors))) {
            assertNotNull("open should have succeeded on a well-formed file", source);
            assertEquals(HEADER_LENGTH, source.baseOffset());
            assertEquals(vectors.size(), source.size());
            assertEquals(DIMENSION, source.dimension());
            assertEquals(VECTOR_BYTES, source.vectorByteLength());

            final DirectIOVectorSource.Reader reader = source.newLoader(VectorScorerMode.RESCORE);
            for (int ord = 0; ord < vectors.size(); ord++) {
                assertArrayEquals("ordinal " + ord, vectors.get(ord), reader.read(ord), 0.0f);
            }
            // Out of order, since a rescore reads a sparse candidate set rather than a scan.
            assertArrayEquals(vectors.get(599), reader.read(599), 0.0f);
            assertArrayEquals(vectors.get(3), reader.read(3), 0.0f);
            assertArrayEquals(vectors.get(412), reader.read(412), 0.0f);
        }
    }

    /**
     * The same read-back over a region that starts on a four-byte boundary, which is what every real
     * {@code .vec} does and what every other test in this suite deliberately does not: with a header of 41
     * the decode falls back to reading a float at a time, and only a header like this one takes the bulk
     * copy. Both paths have to produce the same floats, so this is the guard on the one that ships.
     */
    @SneakyThrows
    public void testEveryOrdinalReadsBackTheBytesThatWereWrittenWhenTheRegionIsFourAligned() {
        assumeDirectIOWorksHere();
        final List<float[]> vectors = randomVectors(600);
        final Path path = writeVectorFile(vectors, 0, ALIGNED_HEADER_LENGTH);

        try (DirectIOVectorSource source = DirectIOVectorSource.open(path, reference(vectors))) {
            assertNotNull("open should have succeeded on a well-formed file", source);
            assertEquals(ALIGNED_HEADER_LENGTH, source.baseOffset());

            final DirectIOVectorSource.Reader reader = source.newLoader(VectorScorerMode.RESCORE);
            for (int ord = 0; ord < vectors.size(); ord++) {
                assertArrayEquals("ordinal " + ord, vectors.get(ord), reader.read(ord), 0.0f);
            }
            assertArrayEquals(vectors.get(599), reader.read(599), 0.0f);
            assertArrayEquals(vectors.get(3), reader.read(3), 0.0f);
        }
    }

    /**
     * Staged reads over a four-aligned region: the bulk decode runs against a ring slot's own
     * {@code float} view rather than the blocking buffer's, and a view built over the wrong slot would
     * return another ordinal's vector.
     */
    @SneakyThrows
    public void testAStagedBatchReadsBackEveryVectorWhenTheRegionIsFourAligned() {
        assumeDirectIOWorksHere();
        final List<float[]> vectors = randomVectors(600);
        final Path path = writeVectorFile(vectors, 0, ALIGNED_HEADER_LENGTH);

        try (DirectIOVectorSource source = DirectIOVectorSource.open(path, reference(vectors))) {
            assertNotNull(source);
            final DirectIOVectorSource.Reader reader = source.newLoader(VectorScorerMode.RESCORE);
            final int[] ords = sparseOrdinals(200, 7, vectors.size());
            reader.stage(ords, ords.length);
            assertReadsInOrder(reader, ords, vectors);
        }
    }

    @SneakyThrows
    public void testBufferIsSizedForTheRegionStartNotForOffsetZero() {
        assumeDirectIOWorksHere();
        final List<float[]> vectors = randomVectors(64);
        final Path path = writeVectorFile(vectors, 0);

        try (DirectIOVectorSource source = DirectIOVectorSource.open(path, reference(vectors))) {
            assertNotNull(source);
            assertEquals(
                DirectIOBufferSizer.requiredBufferSize(VECTOR_BYTES, source.blockSize(), source.baseOffset()),
                source.bufferSize()
            );
            // Whatever the block size is, one buffer has to cover the worst straddle from this base offset.
            assertTrue(source.bufferSize() % source.blockSize() == 0);
            assertTrue(source.bufferSize() >= VECTOR_BYTES);
        }
    }

    @SneakyThrows
    public void testEachReaderIsIndependent() {
        assumeDirectIOWorksHere();
        final List<float[]> vectors = randomVectors(64);
        final Path path = writeVectorFile(vectors, 0);

        try (DirectIOVectorSource source = DirectIOVectorSource.open(path, reference(vectors))) {
            assertNotNull(source);
            final DirectIOVectorSource.Reader first = source.newLoader(VectorScorerMode.RESCORE);
            final DirectIOVectorSource.Reader second = source.newLoader(VectorScorerMode.RESCORE);
            final float[] fromFirst = first.read(7);
            final float[] fromSecond = second.read(31);
            assertNotSame(fromFirst, fromSecond);
            assertArrayEquals(vectors.get(7), fromFirst, 0.0f);
            assertArrayEquals(vectors.get(31), fromSecond, 0.0f);
        }
    }

    /**
     * This is the one implementation that fills both seams at once, and that is a property of the
     * implementation rather than of the seams: the loader seam says how bytes for an ordinal arrive and the
     * staging seam says which ordinals are coming, and a cache introduced at the loader seam would implement
     * only the first. Asserted structurally so that splitting {@code Reader} into two objects later - or
     * fusing the two interfaces into one, which is what the design forbids - is a test failure rather than a
     * silent change of shape.
     */
    public void testTheReaderFillsBothSeamsWhileTheSeamsThemselvesStaySeparate() {
        assertTrue(VectorLoaderSource.Loader.class.isAssignableFrom(DirectIOVectorSource.Reader.class));
        assertTrue(VectorStagingArea.class.isAssignableFrom(DirectIOVectorSource.Reader.class));
        assertTrue(VectorLoaderSource.class.isAssignableFrom(DirectIOVectorSource.class));
        // Neither seam may require the other, or a loader without read-ahead could not exist.
        assertFalse(VectorStagingArea.class.isAssignableFrom(VectorLoaderSource.Loader.class));
        assertFalse(VectorLoaderSource.Loader.class.isAssignableFrom(VectorStagingArea.class));
    }

    /**
     * The reuse hint has to survive the trip to the loader, where it is easy to drop as an unused
     * constructor argument, because it is what decides whether that loader caches. The hint changes when a
     * read costs a device round trip and never which bytes it produces, which is why both modes are asserted
     * against the written contents here rather than against each other.
     */
    @SneakyThrows
    public void testEachLoaderReportsTheReuseHintItWasBuiltWith() {
        assumeDirectIOWorksHere();
        final List<float[]> vectors = randomVectors(8);
        final Path path = writeVectorFile(vectors, 0);

        try (DirectIOVectorSource source = DirectIOVectorSource.open(path, reference(vectors))) {
            assertNotNull(source);
            final DirectIOVectorSource.Reader rescore = source.newLoader(VectorScorerMode.RESCORE);
            final DirectIOVectorSource.Reader score = source.newLoader(VectorScorerMode.SCORE);

            assertSame(VectorScorerMode.RESCORE, rescore.reuseHint());
            assertSame(VectorScorerMode.SCORE, score.reuseHint());
            assertArrayEquals(vectors.get(3), rescore.read(3), 0.0f);
            assertArrayEquals(vectors.get(3), score.read(3), 0.0f);
        }
    }

    /**
     * The derivation {@code baseOffset = fileLength - footer - size * vectorBytes} is an inference about a
     * file layout this code does not own, so it is verified before use. Padding between the vectors and the
     * footer is what a second vector region in the same file looks like from the outside: the derived
     * offset lands past the real one, the bytes do not match, and the source declines instead of scoring
     * garbage.
     */
    @SneakyThrows
    public void testOpenDeclinesWhenTheRegionIsNotWhereItWasDerivedToBe() {
        final List<float[]> vectors = randomVectors(64);
        final Path path = writeVectorFile(vectors, 512);
        assertNull(DirectIOVectorSource.open(path, reference(vectors)));
    }

    @SneakyThrows
    public void testOpenDeclinesWhenTheFileIsTooShortForTheVectors() {
        final List<float[]> vectors = randomVectors(64);
        final Path path = writeVectorFile(vectors.subList(0, 8), 0);
        assertNull(DirectIOVectorSource.open(path, reference(vectors)));
    }

    public void testOpenDeclinesWhenThereIsNoFile() {
        final List<float[]> vectors = randomVectors(4);
        assertNull(DirectIOVectorSource.open(createTempDir().resolve("absent.vec"), reference(vectors)));
    }

    public void testOpenDeclinesOnNullArguments() {
        assertNull(DirectIOVectorSource.open(null, reference(randomVectors(4))));
        assertNull(DirectIOVectorSource.open(createTempDir().resolve("x.vec"), null));
    }

    @SneakyThrows
    public void testOpenDeclinesForAnEmptySegment() {
        final Path path = writeVectorFile(List.of(), 0);
        assertNull(DirectIOVectorSource.open(path, reference(List.of())));
    }

    @SneakyThrows
    public void testSingleVectorSegmentVerifiesAgainstItsOnlyOrdinal() {
        assumeDirectIOWorksHere();
        final List<float[]> vectors = randomVectors(1);
        final Path path = writeVectorFile(vectors, 0);
        try (DirectIOVectorSource source = DirectIOVectorSource.open(path, reference(vectors))) {
            assertNotNull(source);
            assertArrayEquals(vectors.get(0), source.newLoader(VectorScorerMode.RESCORE).read(0), 0.0f);
        }
    }

    @SneakyThrows
    public void testReadRejectsOrdinalsOutsideTheRegion() {
        assumeDirectIOWorksHere();
        final List<float[]> vectors = randomVectors(16);
        final Path path = writeVectorFile(vectors, 0);
        try (DirectIOVectorSource source = DirectIOVectorSource.open(path, reference(vectors))) {
            assertNotNull(source);
            final DirectIOVectorSource.Reader reader = source.newLoader(VectorScorerMode.RESCORE);
            expectThrows(IllegalArgumentException.class, () -> reader.read(-1));
            expectThrows(IllegalArgumentException.class, () -> reader.read(16));
        }
    }

    @SneakyThrows
    public void testReadFailsAfterClose() {
        assumeDirectIOWorksHere();
        final List<float[]> vectors = randomVectors(16);
        final Path path = writeVectorFile(vectors, 0);
        final DirectIOVectorSource source = DirectIOVectorSource.open(path, reference(vectors));
        assertNotNull(source);
        final DirectIOVectorSource.Reader reader = source.newLoader(VectorScorerMode.RESCORE);
        source.close();
        expectThrows(Exception.class, () -> reader.read(0));
    }

    // ----------------------------------------------------------------------------------------------
    // Staging. Every test here asserts the same thing from a different angle: read ahead changes when
    // bytes arrive and never which bytes. A ring with an off-by-one returns a neighbouring vector, which
    // is why these compare against the written contents rather than against a second read.
    // ----------------------------------------------------------------------------------------------

    /** The ordinals one batch of a rescore looks like: sparse, ascending, wider than one block. */
    private static int[] sparseOrdinals(final int count, final int stride, final int limit) {
        final int[] ords = new int[count];
        for (int i = 0; i < count; i++) {
            ords[i] = (i * stride) % limit;
        }
        return ords;
    }

    private void assertReadsInOrder(final DirectIOVectorSource.Reader reader, final int[] ords, final List<float[]> vectors)
        throws IOException {
        for (final int ord : ords) {
            assertArrayEquals("ordinal " + ord, vectors.get(ord), reader.read(ord), 0.0f);
        }
    }

    /**
     * A staged batch several times the default window, so the ring has to roll: slots are reused as reads
     * are consumed, and a slot handed to a new read before the old one finished writing it would show up
     * here as a vector from the wrong ordinal.
     */
    @SneakyThrows
    public void testAStagedBatchWiderThanTheRingReadsBackEveryVector() {
        assumeDirectIOWorksHere();
        final List<float[]> vectors = randomVectors(600);
        final Path path = writeVectorFile(vectors, 0);

        try (DirectIOVectorSource source = DirectIOVectorSource.open(path, reference(vectors))) {
            assertNotNull(source);
            final DirectIOVectorSource.Reader reader = source.newLoader(VectorScorerMode.RESCORE);
            // 200 is firstPassK on the benchmark index, i.e. a whole rescore candidate set.
            final int[] ords = sparseOrdinals(200, 7, vectors.size());
            reader.stage(ords, ords.length);
            assertReadsInOrder(reader, ords, vectors);
        }
    }

    /** A batch shorter than the window: every read is already in flight before the first is consumed. */
    @SneakyThrows
    public void testAStagedBatchNarrowerThanTheRingReadsBackEveryVector() {
        assumeDirectIOWorksHere();
        final List<float[]> vectors = randomVectors(64);
        final Path path = writeVectorFile(vectors, 0);

        try (DirectIOVectorSource source = DirectIOVectorSource.open(path, reference(vectors))) {
            assertNotNull(source);
            final DirectIOVectorSource.Reader reader = source.newLoader(VectorScorerMode.RESCORE);
            final int[] ords = { 61, 0, 33, 7, 12 };
            reader.stage(ords, ords.length);
            assertReadsInOrder(reader, ords, vectors);
        }
    }

    /**
     * Only the first {@code count} entries are staged, and the tail of the array must not be read — a
     * caller reusing one oversized array across batches is exactly what Lucene's bulk scorer does.
     */
    @SneakyThrows
    public void testStageLooksAtOnlyTheFirstCountOrdinals() {
        assumeDirectIOWorksHere();
        final List<float[]> vectors = randomVectors(64);
        final Path path = writeVectorFile(vectors, 0);

        try (DirectIOVectorSource source = DirectIOVectorSource.open(path, reference(vectors))) {
            assertNotNull(source);
            final DirectIOVectorSource.Reader reader = source.newLoader(VectorScorerMode.RESCORE);
            // The tail is out of range on purpose: staging it would decline the batch, and reading it would
            // throw. Neither may happen, because count says it is not part of this batch.
            final int[] ords = { 5, 9, 40, 12345, -3 };
            reader.stage(ords, 3);
            assertReadsInOrder(reader, new int[] { 5, 9, 40 }, vectors);
        }
    }

    /**
     * Staging is a prediction, and a wrong prediction has to cost latency rather than correctness. Here the
     * consumer reads an ordinal the batch did not put next, which drops the batch; the remaining reads then
     * come from blocking reads and must still be right.
     */
    @SneakyThrows
    public void testReadFallsBackToBlockingWhenTheConsumerLeavesTheStagedOrder() {
        assumeDirectIOWorksHere();
        final List<float[]> vectors = randomVectors(600);
        final Path path = writeVectorFile(vectors, 0);

        try (DirectIOVectorSource source = DirectIOVectorSource.open(path, reference(vectors))) {
            assertNotNull(source);
            final DirectIOVectorSource.Reader reader = source.newLoader(VectorScorerMode.RESCORE);
            final int[] ords = sparseOrdinals(64, 9, vectors.size());
            reader.stage(ords, ords.length);

            // Consume a little of the batch, then deviate, then read the whole batch anyway.
            assertArrayEquals(vectors.get(ords[0]), reader.read(ords[0]), 0.0f);
            assertArrayEquals(vectors.get(ords[1]), reader.read(ords[1]), 0.0f);
            assertArrayEquals(vectors.get(511), reader.read(511), 0.0f);
            assertReadsInOrder(reader, ords, vectors);
        }
    }

    /**
     * One reader serves many batches, and a batch that was abandoned part-read leaves reads in flight over
     * slots the next batch reuses. Quiescing them is the reason {@code stage} cancels and drains first; if
     * it did not, this test would intermittently read a vector from the previous batch.
     */
    @SneakyThrows
    public void testConsecutiveBatchesOnOneReaderAfterPartialConsumption() {
        assumeDirectIOWorksHere();
        final List<float[]> vectors = randomVectors(600);
        final Path path = writeVectorFile(vectors, 0);

        try (DirectIOVectorSource source = DirectIOVectorSource.open(path, reference(vectors))) {
            assertNotNull(source);
            final DirectIOVectorSource.Reader reader = source.newLoader(VectorScorerMode.RESCORE);
            for (int batch = 0; batch < 8; batch++) {
                final int[] ords = sparseOrdinals(64, 7 + batch, vectors.size());
                reader.stage(ords, ords.length);
                // Consume only part of it, so the next stage() has to clean up after this one.
                for (int i = 0; i < 3 + batch; i++) {
                    assertArrayEquals("batch " + batch + " ordinal " + ords[i], vectors.get(ords[i]), reader.read(ords[i]), 0.0f);
                }
            }
            // And after all that, the reader is still a correct reader.
            assertReadsInOrder(reader, sparseOrdinals(64, 13, vectors.size()), vectors);
        }
    }

    /**
     * A batch containing an ordinal outside the region is not staged at all, so the out-of-range ordinal is
     * still rejected by {@code read} with the ordinal in hand rather than swallowed by a background read.
     */
    @SneakyThrows
    public void testStageDeclinesABatchThatContainsAnOrdinalOutsideTheRegion() {
        assumeDirectIOWorksHere();
        final List<float[]> vectors = randomVectors(64);
        final Path path = writeVectorFile(vectors, 0);

        try (DirectIOVectorSource source = DirectIOVectorSource.open(path, reference(vectors))) {
            assertNotNull(source);
            final DirectIOVectorSource.Reader reader = source.newLoader(VectorScorerMode.RESCORE);
            reader.stage(new int[] { 1, 2, 64 }, 3);
            assertArrayEquals(vectors.get(1), reader.read(1), 0.0f);
            expectThrows(IllegalArgumentException.class, () -> reader.read(64));

            reader.stage(new int[] { 1, -1 }, 2);
            expectThrows(IllegalArgumentException.class, () -> reader.read(-1));
        }
    }

    @SneakyThrows
    public void testStageIsHarmlessForDegenerateBatches() {
        assumeDirectIOWorksHere();
        final List<float[]> vectors = randomVectors(64);
        final Path path = writeVectorFile(vectors, 0);

        try (DirectIOVectorSource source = DirectIOVectorSource.open(path, reference(vectors))) {
            assertNotNull(source);
            final DirectIOVectorSource.Reader reader = source.newLoader(VectorScorerMode.RESCORE);
            reader.stage(null, 8);
            reader.stage(new int[] { 3 }, 1);
            reader.stage(new int[] { 3, 4 }, 0);
            reader.stage(new int[0], 0);
            assertArrayEquals(vectors.get(3), reader.read(3), 0.0f);
        }
    }

    /**
     * The shared read pool means one file handle is read by many threads at once, and each reader's slots
     * are written by pool threads and decoded by the query thread. This is the test that would fail if
     * slots were shared between readers or if a slot were published without the {@code Future}'s
     * happens-before.
     */
    @SneakyThrows
    public void testManyReadersStageConcurrentlyWithoutCrossTalk() {
        assumeDirectIOWorksHere();
        final List<float[]> vectors = randomVectors(600);
        final Path path = writeVectorFile(vectors, 0);
        final int threads = 8;

        try (DirectIOVectorSource source = DirectIOVectorSource.open(path, reference(vectors))) {
            assertNotNull(source);
            final ExecutorService drivers = Executors.newFixedThreadPool(threads);
            try {
                final CountDownLatch start = new CountDownLatch(1);
                final List<Future<?>> running = new ArrayList<>(threads);
                for (int t = 0; t < threads; t++) {
                    final int stride = 3 + t;
                    running.add(drivers.submit(() -> {
                        start.await();
                        final DirectIOVectorSource.Reader reader = source.newLoader(VectorScorerMode.RESCORE);
                        for (int round = 0; round < 20; round++) {
                            final int[] ords = sparseOrdinals(64, stride, vectors.size());
                            reader.stage(ords, ords.length);
                            for (final int ord : ords) {
                                assertArrayEquals("stride " + stride + " ordinal " + ord, vectors.get(ord), reader.read(ord), 0.0f);
                            }
                        }
                        return null;
                    }));
                }
                start.countDown();
                for (final Future<?> future : running) {
                    future.get();
                }
            } finally {
                drivers.shutdownNow();
            }
        }
    }

    // ----------------------------------------------------------------------------------------------
    // The cache behind the loader seam. Staging and retention meet here, and the thing that breaks is
    // ordering: with a cache, a batch is a mix of positions that took a ring slot and positions that did
    // not, so batch position and slot index stop advancing together. Every test below therefore asserts
    // the vectors against the written contents, because an off-by-one in that arithmetic returns a
    // neighbouring candidate's vector and scores it as this one's.
    // ----------------------------------------------------------------------------------------------

    /** Bytes one cache entry costs at {@link #DIMENSION}, so a budget can be written in whole entries. */
    private static final long ENTRY_BYTES = LruVectorCache.ENTRY_OVERHEAD_BYTES + DIMENSION * (long) Float.BYTES;

    /** A source whose cache holds exactly {@code entries} vectors, bypassing the node setting. */
    private DirectIOVectorSource openWithCacheFor(final List<float[]> vectors, final Path path, final int entries) {
        final DirectIOVectorSource source = DirectIOVectorSource.open(path, reference(vectors), entries * ENTRY_BYTES);
        assertNotNull(source);
        return source;
    }

    /** Reads {@code ords} through {@code reader} without staging, which is what warms the cache. */
    private void warm(final DirectIOVectorSource.Reader reader, final int[] ords, final List<float[]> vectors) throws IOException {
        for (final int ord : ords) {
            assertArrayEquals(vectors.get(ord), reader.read(ord), 0.0f);
        }
    }

    /**
     * A budget of zero is no cache object at all, which is what makes "cache off" a control arm rather than a
     * cache that always misses: there is no lookup, no accounting and no counter on the read path. Asserted
     * through {@code cacheStats()} because that is the only way the absence is visible from outside.
     */
    @SneakyThrows
    public void testAZeroBudgetSourceHasNoCacheAndStillReadsEveryVector() {
        assumeDirectIOWorksHere();
        final List<float[]> vectors = randomVectors(600);
        final Path path = writeVectorFile(vectors, 0, ALIGNED_HEADER_LENGTH);

        try (DirectIOVectorSource source = DirectIOVectorSource.open(path, reference(vectors), 0L)) {
            assertNotNull(source);
            assertNull("a zero budget must leave no cache to report on", source.cacheStats());

            final DirectIOVectorSource.Reader reader = source.newLoader(VectorScorerMode.RESCORE);
            final int[] ords = sparseOrdinals(200, 7, vectors.size());
            // Twice, so anything that only shows up on the second pass over the same ordinals would show up.
            reader.stage(ords, ords.length);
            assertReadsInOrder(reader, ords, vectors);
            reader.stage(ords, ords.length);
            assertReadsInOrder(reader, ords, vectors);
            assertNull(source.cacheStats());
        }
    }

    /**
     * Verification reads through a loader with no cache, whatever the budget, so a source arrives with its
     * counters at zero. That matters for the benchmark rather than for correctness: the measured hit rate is
     * read against a prediction, and two startup misses charged to every segment is noise in it.
     */
    @SneakyThrows
    public void testANewSourceStartsWithAnEmptyCache() {
        assumeDirectIOWorksHere();
        final List<float[]> vectors = randomVectors(64);
        final Path path = writeVectorFile(vectors, 0);

        try (DirectIOVectorSource source = openWithCacheFor(vectors, path, 32)) {
            final LruVectorCache.Stats stats = source.cacheStats();
            assertNotNull(stats);
            assertEquals(0, stats.hits());
            assertEquals(0, stats.misses());
            assertEquals(0, stats.entries());
        }
    }

    /** Only a RESCORE loader caches. A SCORE loader over the same source must neither read nor fill it. */
    @SneakyThrows
    public void testOnlyRescoreLoadersCache() {
        assumeDirectIOWorksHere();
        final List<float[]> vectors = randomVectors(64);
        final Path path = writeVectorFile(vectors, 0);

        try (DirectIOVectorSource source = openWithCacheFor(vectors, path, 32)) {
            final DirectIOVectorSource.Reader score = source.newLoader(VectorScorerMode.SCORE);
            warm(score, new int[] { 1, 2, 3 }, vectors);
            assertEquals("a SCORE loader must not populate", 0, source.cacheStats().entries());
            assertEquals(0, source.cacheStats().misses());

            final DirectIOVectorSource.Reader rescore = source.newLoader(VectorScorerMode.RESCORE);
            warm(rescore, new int[] { 1, 2, 3 }, vectors);
            assertEquals(3, source.cacheStats().entries());
            assertEquals(3, source.cacheStats().misses());
        }
    }

    /**
     * The cache is shared by every reader over one source, which is the whole point: the reuse it recovers is
     * across queries, and a per-reader cache would recover none of it. So a second reader must find what the
     * first one read.
     */
    @SneakyThrows
    public void testTheCacheIsSharedBetweenReadersOverOneSource() {
        assumeDirectIOWorksHere();
        final List<float[]> vectors = randomVectors(64);
        final Path path = writeVectorFile(vectors, 0);

        try (DirectIOVectorSource source = openWithCacheFor(vectors, path, 32)) {
            warm(source.newLoader(VectorScorerMode.RESCORE), new int[] { 9, 10, 11 }, vectors);
            assertEquals(0, source.cacheStats().hits());

            warm(source.newLoader(VectorScorerMode.RESCORE), new int[] { 9, 10, 11 }, vectors);
            assertEquals("the second reader must have been served by the first reader's reads", 3, source.cacheStats().hits());
        }
    }

    /**
     * The headline of the wiring: a staged batch whose ordinals are all already held issues no device read at
     * all and still delivers every vector in declared order. Every position is cache-served, so the ring is
     * never touched - the case where slot arithmetic driven by batch position would read slot 0 repeatedly.
     */
    @SneakyThrows
    public void testAFullyCachedBatchIsServedWithoutTheRing() {
        assumeDirectIOWorksHere();
        final List<float[]> vectors = randomVectors(600);
        final Path path = writeVectorFile(vectors, 0, ALIGNED_HEADER_LENGTH);
        final int[] ords = sparseOrdinals(64, 9, vectors.size());

        try (DirectIOVectorSource source = openWithCacheFor(vectors, path, 600)) {
            final DirectIOVectorSource.Reader reader = source.newLoader(VectorScorerMode.RESCORE);
            reader.stage(ords, ords.length);
            assertReadsInOrder(reader, ords, vectors);
            final long missesAfterWarming = source.cacheStats().misses();

            reader.stage(ords, ords.length);
            assertReadsInOrder(reader, ords, vectors);

            assertEquals("a fully cached batch must read nothing from the device", missesAfterWarming, source.cacheStats().misses());
            assertEquals(ords.length, source.cacheStats().hits());
        }
    }

    /**
     * The fiddly case, in every shape that breaks it differently. A batch where only some positions were
     * already held has to interleave cache-served and device-served positions: hits at the head leave the
     * first device read in slot 0 rather than in the slot its batch position implies, hits at the tail leave
     * the ring draining early, and alternating hits shift slot from position by a growing offset. All three
     * return a neighbouring candidate's vector if the arithmetic is wrong, so all three are read back against
     * what was written.
     */
    @SneakyThrows
    public void testPartiallyCachedBatchesPreserveOrderAndValues() {
        assumeDirectIOWorksHere();
        final List<float[]> vectors = randomVectors(600);
        final Path path = writeVectorFile(vectors, 0, ALIGNED_HEADER_LENGTH);
        final int[] ords = sparseOrdinals(120, 7, vectors.size());

        // Which positions of the batch to warm first: head, tail, every other one, one, and all but one.
        final List<int[]> patterns = List.of(
            new int[] { 0, 1, 2, 3, 4, 5 },
            new int[] { 114, 115, 116, 117, 118, 119 },
            new int[] { 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22 },
            new int[] { 63 },
            new int[] { 0, 60, 119 }
        );
        for (final int[] warmPositions : patterns) {
            try (DirectIOVectorSource source = openWithCacheFor(vectors, path, 600)) {
                final DirectIOVectorSource.Reader warmer = source.newLoader(VectorScorerMode.RESCORE);
                for (final int position : warmPositions) {
                    assertArrayEquals(vectors.get(ords[position]), warmer.read(ords[position]), 0.0f);
                }
                final long hitsBefore = source.cacheStats().hits();

                final DirectIOVectorSource.Reader reader = source.newLoader(VectorScorerMode.RESCORE);
                reader.stage(ords, ords.length);
                assertReadsInOrder(reader, ords, vectors);

                assertEquals(
                    "positions " + Arrays.toString(warmPositions) + " should have been served from the cache",
                    hitsBefore + warmPositions.length,
                    source.cacheStats().hits()
                );
            }
        }
    }

    /**
     * A batch wider than the ring <em>and</em> partly cached, so slots are reused while the offset between
     * batch position and slot index keeps growing. This is the shape the benchmark actually runs: 200
     * candidates, a window of 48, and a hit rate around 40%.
     */
    @SneakyThrows
    public void testAPartiallyCachedBatchWiderThanTheRingRollsCorrectly() {
        assumeDirectIOWorksHere();
        final List<float[]> vectors = randomVectors(600);
        final Path path = writeVectorFile(vectors, 0, ALIGNED_HEADER_LENGTH);
        final int[] ords = sparseOrdinals(200, 7, vectors.size());

        try (DirectIOVectorSource source = openWithCacheFor(vectors, path, 600)) {
            // Warm a scattered 40% of the batch, which is roughly the measured steady-state hit rate.
            final DirectIOVectorSource.Reader warmer = source.newLoader(VectorScorerMode.RESCORE);
            int warmed = 0;
            for (int position = 0; position < ords.length; position++) {
                if (position % 5 == 1 || position % 5 == 3) {
                    assertArrayEquals(vectors.get(ords[position]), warmer.read(ords[position]), 0.0f);
                    warmed++;
                }
            }
            final long hitsBefore = source.cacheStats().hits();

            final DirectIOVectorSource.Reader reader = source.newLoader(VectorScorerMode.RESCORE);
            reader.stage(ords, ords.length);
            assertReadsInOrder(reader, ords, vectors);
            assertEquals(hitsBefore + warmed, source.cacheStats().hits());
        }
    }

    /**
     * A cache-served position whose entry is evicted between the stage-time lookup and the consume falls back
     * to a device read for its own ordinal - not to the ring, whose slots belong to other positions. Forced
     * here by a cache far smaller than the batch, so this batch's own misses evict the entries its own
     * stage-time lookup found; in production the stage-time touch makes that take a whole budget of other
     * traffic, which is why it needs forcing to be tested at all.
     */
    @SneakyThrows
    public void testAnEntryEvictedBetweenStageAndConsumeStillDeliversTheRightVector() {
        assumeDirectIOWorksHere();
        final List<float[]> vectors = randomVectors(64);
        final Path path = writeVectorFile(vectors, 0);

        try (DirectIOVectorSource source = openWithCacheFor(vectors, path, 2)) {
            final DirectIOVectorSource.Reader reader = source.newLoader(VectorScorerMode.RESCORE);
            warm(reader, new int[] { 5, 6 }, vectors);
            assertEquals(2, source.cacheStats().entries());

            // 5 and 6 are both resident when this is staged. Consuming 10 and 11 evicts them both, so the
            // last position is marked cache-served and then finds nothing there.
            final int[] ords = { 5, 10, 11, 6 };
            reader.stage(ords, ords.length);
            assertReadsInOrder(reader, ords, vectors);

            // And the reader is still a correct reader afterwards, with the batch intact for a later one.
            assertReadsInOrder(reader, new int[] { 11, 6, 5 }, vectors);
        }
    }

    /**
     * A batch that repeats an ordinal marks both positions cache-served or neither, and must still deliver
     * one vector per call in declared order. Rescore candidate sets do not repeat, but the ring must not be
     * the thing that assumes it.
     */
    @SneakyThrows
    public void testABatchThatRepeatsAnOrdinalIsServedPositionByPosition() {
        assumeDirectIOWorksHere();
        final List<float[]> vectors = randomVectors(64);
        final Path path = writeVectorFile(vectors, 0);

        try (DirectIOVectorSource source = openWithCacheFor(vectors, path, 32)) {
            final DirectIOVectorSource.Reader reader = source.newLoader(VectorScorerMode.RESCORE);
            warm(reader, new int[] { 3 }, vectors);

            final int[] ords = { 3, 8, 3, 9, 8, 3 };
            reader.stage(ords, ords.length);
            assertReadsInOrder(reader, ords, vectors);
        }
    }

    /**
     * Staging is a prediction and a wrong one must cost latency rather than correctness, cache or no cache.
     * Here the consumer deviates from the declared order in the middle of a partly cached batch, which drops
     * the batch and leaves every remaining read on the blocking path - where the cache still has to be
     * consulted, or an abandoned batch would silently turn every hit into a device read.
     */
    @SneakyThrows
    public void testACacheServedBatchStillFallsBackWhenTheConsumerLeavesTheStagedOrder() {
        assumeDirectIOWorksHere();
        final List<float[]> vectors = randomVectors(600);
        final Path path = writeVectorFile(vectors, 0, ALIGNED_HEADER_LENGTH);
        final int[] ords = sparseOrdinals(64, 9, vectors.size());

        try (DirectIOVectorSource source = openWithCacheFor(vectors, path, 600)) {
            final DirectIOVectorSource.Reader reader = source.newLoader(VectorScorerMode.RESCORE);
            // First pass fills the cache with the whole batch.
            reader.stage(ords, ords.length);
            assertReadsInOrder(reader, ords, vectors);
            final long missesAfterWarming = source.cacheStats().misses();

            // Second pass deviates after two positions, abandoning a batch every position of which was
            // marked cache-served.
            reader.stage(ords, ords.length);
            assertArrayEquals(vectors.get(ords[0]), reader.read(ords[0]), 0.0f);
            assertArrayEquals(vectors.get(ords[1]), reader.read(ords[1]), 0.0f);
            assertArrayEquals(vectors.get(511), reader.read(511), 0.0f);
            assertReadsInOrder(reader, ords, vectors);

            // 511 was the only ordinal not already held, so it is the only device read the second pass owed.
            assertEquals(missesAfterWarming + 1, source.cacheStats().misses());
        }
    }

    /**
     * One reader, many partly cached batches, each abandoned part-read by the next {@code stage}. This is the
     * shape that leaves reads in flight over slots the next batch reuses, and the shape where the cache keeps
     * changing which positions take a slot at all, so the offset between batch position and slot index is
     * different in every batch.
     */
    @SneakyThrows
    public void testConsecutiveMixedBatchesOnOneReaderAfterPartialConsumption() {
        assumeDirectIOWorksHere();
        final List<float[]> vectors = randomVectors(600);
        final Path path = writeVectorFile(vectors, 0, ALIGNED_HEADER_LENGTH);

        try (DirectIOVectorSource source = openWithCacheFor(vectors, path, 100)) {
            final DirectIOVectorSource.Reader reader = source.newLoader(VectorScorerMode.RESCORE);
            for (int batch = 0; batch < 12; batch++) {
                final int[] ords = sparseOrdinals(64, 7 + batch, vectors.size());
                reader.stage(ords, ords.length);
                for (int i = 0; i < 3 + batch; i++) {
                    assertArrayEquals("batch " + batch + " ordinal " + ords[i], vectors.get(ords[i]), reader.read(ords[i]), 0.0f);
                }
            }
            assertReadsInOrder(reader, sparseOrdinals(64, 13, vectors.size()), vectors);
            // A budget of 100 against 64-ordinal batches over 600 vectors must have been evicting.
            assertTrue("the budget should have been exercised", source.cacheStats().evictions() > 0);
        }
    }

    /**
     * The unit-level form of this phase's correctness oracle: a cached source and an uncached one, driven
     * through the same long sequence of staged batches, must produce identical vectors for every read. On the
     * bench node the same question is asked as "is recall still 0.8106 with identical top-100 identifier
     * sets"; here it is asked directly, against a cache small enough to be evicting throughout.
     */
    @SneakyThrows
    public void testACachedSourceReadsIdenticallyToAnUncachedOne() {
        assumeDirectIOWorksHere();
        final List<float[]> vectors = randomVectors(600);
        final Path path = writeVectorFile(vectors, 0, ALIGNED_HEADER_LENGTH);

        try (
            DirectIOVectorSource cached = openWithCacheFor(vectors, path, 40);
            DirectIOVectorSource uncached = DirectIOVectorSource.open(path, reference(vectors), 0L)
        ) {
            assertNotNull(uncached);
            final DirectIOVectorSource.Reader withCache = cached.newLoader(VectorScorerMode.RESCORE);
            final DirectIOVectorSource.Reader withoutCache = uncached.newLoader(VectorScorerMode.RESCORE);

            final Random random = new Random(7);
            for (int round = 0; round < 40; round++) {
                final int[] ords = new int[1 + random.nextInt(120)];
                for (int i = 0; i < ords.length; i++) {
                    ords[i] = random.nextInt(vectors.size());
                }
                withCache.stage(ords, ords.length);
                withoutCache.stage(ords, ords.length);
                for (final int ord : ords) {
                    final float[] fromCached = withCache.read(ord).clone();
                    final float[] fromUncached = withoutCache.read(ord);
                    assertArrayEquals("round " + round + " ordinal " + ord, fromUncached, fromCached, 0.0f);
                    assertArrayEquals(vectors.get(ord), fromCached, 0.0f);
                }
            }
            assertTrue("the run should have produced hits", cached.cacheStats().hits() > 0);
            assertTrue("and evictions, at 40 slots over 600 vectors", cached.cacheStats().evictions() > 0);
            assertNull(uncached.cacheStats());
        }
    }

    /**
     * Concurrent readers over one shared cache small enough to be evicting throughout, which is the
     * production shape: per-leaf scorers run in parallel, each with its own loader, all populating and
     * evicting the same LRU. A cache that published a half-written entry, or a stage-time lookup that raced
     * an eviction into serving the wrong slot, shows up here as a vector from another ordinal.
     */
    @SneakyThrows
    public void testManyReadersShareOneCacheWithoutCrossTalk() {
        assumeDirectIOWorksHere();
        final List<float[]> vectors = randomVectors(600);
        final Path path = writeVectorFile(vectors, 0, ALIGNED_HEADER_LENGTH);
        final int threads = 8;

        try (DirectIOVectorSource source = openWithCacheFor(vectors, path, 48)) {
            final ExecutorService drivers = Executors.newFixedThreadPool(threads);
            try {
                final CountDownLatch start = new CountDownLatch(1);
                final List<Future<?>> running = new ArrayList<>(threads);
                for (int t = 0; t < threads; t++) {
                    final int stride = 3 + t;
                    running.add(drivers.submit(() -> {
                        start.await();
                        final DirectIOVectorSource.Reader reader = source.newLoader(VectorScorerMode.RESCORE);
                        for (int round = 0; round < 20; round++) {
                            final int[] ords = sparseOrdinals(64, stride, vectors.size());
                            reader.stage(ords, ords.length);
                            for (final int ord : ords) {
                                assertArrayEquals("stride " + stride + " ordinal " + ord, vectors.get(ord), reader.read(ord), 0.0f);
                            }
                        }
                        return null;
                    }));
                }
                start.countDown();
                for (final Future<?> future : running) {
                    future.get();
                }
            } finally {
                drivers.shutdownNow();
            }
            final LruVectorCache.Stats stats = source.cacheStats();
            assertTrue("48 slots over 600 vectors and 8 threads must have evicted", stats.evictions() > 0);
            assertTrue("and must have hit", stats.hits() > 0);
            assertTrue("held bytes must never exceed the budget", stats.bytes() <= stats.budgetBytes());
        }
    }
}

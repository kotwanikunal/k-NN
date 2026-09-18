/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import lombok.SneakyThrows;
import org.apache.lucene.index.FloatVectorValues;
import org.opensearch.knn.KNNTestCase;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.OpenOption;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

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
 */
public class DirectIOVectorSourceTests extends KNNTestCase {

    private static final int DIMENSION = 8;
    private static final int VECTOR_BYTES = DIMENSION * Float.BYTES;
    /** Deliberately odd, so the vector region starts unaligned to any block or gcd boundary. */
    private static final int HEADER_LENGTH = 41;

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
    @SneakyThrows
    private Path writeVectorFile(final List<float[]> vectors, final int trailingPadding) {
        final Path path = createTempDir().resolve("_0_Test_0.vec");
        final ByteBuffer buffer = ByteBuffer.allocate(HEADER_LENGTH + vectors.size() * VECTOR_BYTES + trailingPadding + 16)
            .order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < HEADER_LENGTH; i++) {
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

            final DirectIOVectorSource.Reader reader = source.newReader();
            for (int ord = 0; ord < vectors.size(); ord++) {
                assertArrayEquals("ordinal " + ord, vectors.get(ord), reader.read(ord), 0.0f);
            }
            // Out of order, since a rescore reads a sparse candidate set rather than a scan.
            assertArrayEquals(vectors.get(599), reader.read(599), 0.0f);
            assertArrayEquals(vectors.get(3), reader.read(3), 0.0f);
            assertArrayEquals(vectors.get(412), reader.read(412), 0.0f);
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
            final DirectIOVectorSource.Reader first = source.newReader();
            final DirectIOVectorSource.Reader second = source.newReader();
            final float[] fromFirst = first.read(7);
            final float[] fromSecond = second.read(31);
            assertNotSame(fromFirst, fromSecond);
            assertArrayEquals(vectors.get(7), fromFirst, 0.0f);
            assertArrayEquals(vectors.get(31), fromSecond, 0.0f);
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
            assertArrayEquals(vectors.get(0), source.newReader().read(0), 0.0f);
        }
    }

    @SneakyThrows
    public void testReadRejectsOrdinalsOutsideTheRegion() {
        assumeDirectIOWorksHere();
        final List<float[]> vectors = randomVectors(16);
        final Path path = writeVectorFile(vectors, 0);
        try (DirectIOVectorSource source = DirectIOVectorSource.open(path, reference(vectors))) {
            assertNotNull(source);
            final DirectIOVectorSource.Reader reader = source.newReader();
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
        final DirectIOVectorSource.Reader reader = source.newReader();
        source.close();
        expectThrows(Exception.class, () -> reader.read(0));
    }
}

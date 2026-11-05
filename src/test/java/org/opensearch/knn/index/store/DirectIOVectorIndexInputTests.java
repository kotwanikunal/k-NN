/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.store.DirectIOTestUtils.Behaviour;
import org.opensearch.knn.index.store.DirectIOTestUtils.TestChannels;

import java.io.EOFException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.opensearch.knn.index.store.DirectIOTestUtils.INDEX_UUID;
import static org.opensearch.knn.index.store.DirectIOTestUtils.VEC_NAME;

/**
 * The aligned reads at every interesting offset through every read method a flat vector reader uses, the integrity
 * reads, and the fallback to the original input when an {@code O_DIRECT} read fails.
 */
public class DirectIOVectorIndexInputTests extends KNNTestCase {

    private static final int FILE_PAYLOAD_BYTES = 64 * 1024;

    private Path shardPath;
    private final AtomicBoolean on = new AtomicBoolean(true);
    private TestChannels channels;

    @Override
    public void setUp() throws Exception {
        super.setUp();
        shardPath = DirectIOTestUtils.defaultLayoutShardPath(createTempDir(), INDEX_UUID);
        on.set(true);
        // With a plain channel beside each O_DIRECT one, because a short read of 100 bytes on an O_DIRECT channel is
        // refused with EINVAL and would be indistinguishable from the throwing case.
        channels = new TestChannels(true);
        DirectIOTestUtils.assumeDirectIOSupported();
    }

    /** A routed input over a fresh file, with a counting channel under it, and the reference input beside it. */
    private final class Fixture implements AutoCloseable {
        private final Directory in;
        private final KNNDirectIODirectory directory;
        private final IndexInput file;
        private final IndexInput routed;
        private final IndexInput reference;
        /** How many device reads the content check made when this file settled, which the read counts are against. */
        private final int readsAtSettle;

        private Fixture() throws IOException {
            in = DirectIOTestUtils.fsDirectory(shardPath);
            DirectIOTestUtils.writeCodecFile(in, VEC_NAME, FILE_PAYLOAD_BYTES);
            directory = DirectIOTestUtils.directory(in, on::get, channels);
            file = directory.openInput(VEC_NAME, IOContext.DEFAULT);
            routed = file.clone();
            reference = in.openInput(VEC_NAME, IOContext.DEFAULT);
            readsAtSettle = channels.reads();
        }

        private int blockSize() throws IOException {
            return Math.toIntExact(Files.getFileStore(shardPath.resolve(VEC_NAME)).getBlockSize());
        }

        @Override
        public void close() throws IOException {
            reference.close();
            file.close();
            in.close();
        }
    }

    public void testReadsBytesAtEveryInterestingOffset() throws IOException {
        try (Fixture fixture = new Fixture()) {
            final int block = fixture.blockSize();
            final long length = fixture.routed.length();
            for (final long offset : new long[] { 0, block - 1, block, block + 1, 2L * block - 7, length - 32 }) {
                for (final int len : new int[] { 1, 7, 4096, 2 * block + 11 }) {
                    if (offset + len > length) {
                        continue;
                    }
                    final byte[] fromRoute = new byte[len];
                    final byte[] fromReference = new byte[len];
                    fixture.routed.seek(offset);
                    fixture.routed.readBytes(fromRoute, 0, len);
                    fixture.reference.seek(offset);
                    fixture.reference.readBytes(fromReference, 0, len);
                    assertArrayEquals("readBytes at " + offset + " of " + len, fromReference, fromRoute);
                    assertEquals(offset + len, fixture.routed.getFilePointer());
                }
            }
        }
    }

    public void testReadsFloatsAtEveryInterestingOffset() throws IOException {
        try (Fixture fixture = new Fixture()) {
            final int block = fixture.blockSize();
            for (final long offset : new long[] { 0, block - 4, block, block + 4, 2L * block - 8 }) {
                for (final int floats : new int[] { 1, 128, 1024, block }) {
                    if (offset + 4L * floats > fixture.routed.length()) {
                        continue;
                    }
                    final float[] fromRoute = new float[floats];
                    final float[] fromReference = new float[floats];
                    fixture.routed.seek(offset);
                    fixture.routed.readFloats(fromRoute, 0, floats);
                    fixture.reference.seek(offset);
                    fixture.reference.readFloats(fromReference, 0, floats);
                    assertArrayEquals("readFloats at " + offset + " of " + floats, fromReference, fromRoute, 0.0f);
                }
            }
        }
    }

    public void testReadsTheIntegerTypesAtEveryInterestingOffset() throws IOException {
        try (Fixture fixture = new Fixture()) {
            final int block = fixture.blockSize();
            for (final long offset : new long[] { 0, block - 1, block, block + 1, 4L * block - 3 }) {
                fixture.routed.seek(offset);
                fixture.reference.seek(offset);
                assertEquals("readByte at " + offset, fixture.reference.readByte(), fixture.routed.readByte());
                fixture.routed.seek(offset);
                fixture.reference.seek(offset);
                assertEquals("readShort at " + offset, fixture.reference.readShort(), fixture.routed.readShort());
                fixture.routed.seek(offset);
                fixture.reference.seek(offset);
                assertEquals("readInt at " + offset, fixture.reference.readInt(), fixture.routed.readInt());
                fixture.routed.seek(offset);
                fixture.reference.seek(offset);
                assertEquals("readLong at " + offset, fixture.reference.readLong(), fixture.routed.readLong());
            }
        }
    }

    public void testReadsLongerThanTheBuffer() throws IOException {
        try (Fixture fixture = new Fixture()) {
            final int len = 8 * fixture.blockSize() + 37;
            final byte[] fromRoute = new byte[len];
            final byte[] fromReference = new byte[len];
            fixture.routed.seek(11);
            fixture.routed.readBytes(fromRoute, 0, len);
            fixture.reference.seek(11);
            fixture.reference.readBytes(fromReference, 0, len);
            assertArrayEquals(fromReference, fromRoute);
        }
    }

    public void testSliceReadsAreOffsetCorrectly() throws IOException {
        try (Fixture fixture = new Fixture()) {
            final int block = fixture.blockSize();
            final long sliceOffset = block + 17;
            final long sliceLength = 3L * block;
            final IndexInput slice = fixture.file.slice("vector-data", sliceOffset, sliceLength);
            assertTrue(slice instanceof DirectIOVectorIndexInput);
            assertEquals(sliceLength, slice.length());
            final byte[] fromSlice = new byte[256];
            final byte[] fromReference = new byte[256];
            slice.seek(block);
            slice.readBytes(fromSlice, 0, fromSlice.length);
            fixture.reference.seek(sliceOffset + block);
            fixture.reference.readBytes(fromReference, 0, fromReference.length);
            assertArrayEquals(fromReference, fromSlice);
        }
    }

    public void testCloneHasItsOwnCursor() throws IOException {
        try (Fixture fixture = new Fixture()) {
            fixture.routed.seek(1024);
            final IndexInput clone = fixture.routed.clone();
            assertEquals(1024, clone.getFilePointer());
            clone.seek(4096);
            assertEquals("the clone's cursor is its own", 1024, fixture.routed.getFilePointer());
            assertEquals(4096, clone.getFilePointer());
        }
    }

    public void testReadPastEofThrowsEofAndIsNotAFailure() throws IOException {
        try (Fixture fixture = new Fixture()) {
            fixture.routed.seek(fixture.routed.length() - 4);
            final byte[] target = new byte[64];
            final int readsBefore = channels.reads();
            expectThrows(EOFException.class, () -> fixture.routed.readBytes(target, 0, target.length));
            assertEquals("the bounds check is thrown before any device read", readsBefore, channels.reads());
        }
    }

    public void testSeekPastEofThrowsEof() throws IOException {
        try (Fixture fixture = new Fixture()) {
            expectThrows(EOFException.class, () -> fixture.routed.seek(fixture.routed.length() + 1));
        }
    }

    public void testEveryDeviceReadIsBlockAligned() throws IOException {
        try (Fixture fixture = new Fixture()) {
            final int block = fixture.blockSize();
            final byte[] target = new byte[3 * block + 5];
            fixture.routed.seek(block - 3);
            fixture.routed.readBytes(target, 0, target.length);
            assertFalse("the read must have reached the device", channels.last().reads().isEmpty());
            for (final DirectIOTestUtils.Read read : channels.last().reads()) {
                assertEquals("offset " + read.position() + " is block aligned", 0, read.position() % block);
                assertEquals("length " + read.length() + " is block aligned", 0, read.length() % block);
            }
        }
    }

    public void testCheckIntegrityPassesThroughARoutedCloneAndThroughAnOffClone() throws IOException {
        try (Fixture fixture = new Fixture()) {
            final int beforeIntegrity = channels.reads();
            CodecUtil.checksumEntireFile(fixture.file.clone());
            assertTrue("checkIntegrity read with O_DIRECT", channels.reads() > beforeIntegrity);
            on.set(false);
            final int readsBefore = channels.reads();
            CodecUtil.checksumEntireFile(fixture.file.clone());
            assertEquals("with the index off the integrity read is the original input's", readsBefore, channels.reads());
        }
    }

    public void testTheFooterReadsTheSameOnBothRoutes() throws IOException {
        try (Fixture fixture = new Fixture()) {
            final IndexInput routedClone = fixture.file.clone();
            final long routedChecksum = CodecUtil.retrieveChecksum(routedClone);
            on.set(false);
            final IndexInput offClone = fixture.file.clone();
            assertEquals(routedChecksum, CodecUtil.retrieveChecksum(offClone));
        }
    }

    public void testAShortChannelReadFallsBackToTheDelegate() throws IOException {
        try (Fixture fixture = new Fixture()) {
            final byte[] fromRoute = new byte[8192];
            final byte[] fromReference = new byte[8192];
            channels.behave(Behaviour.SHORT);
            fixture.routed.seek(1024);
            fixture.routed.readBytes(fromRoute, 0, fromRoute.length);
            fixture.reference.seek(1024);
            fixture.reference.readBytes(fromReference, 0, fromReference.length);
            assertArrayEquals("a short read is served by the original input", fromReference, fromRoute);
            assertEquals(1024 + fromRoute.length, fixture.routed.getFilePointer());
            // The channel returned bytes rather than throwing, and fewer than the offset asked for, which is the
            // branch this case exists to cover: one device read was attempted, under one block, and the file then
            // reads through the original input.
            assertEquals("one device read was attempted", fixture.readsAtSettle + 1, channels.reads());
            assertTrue(DirectIOTestUtils.TestChannel.SHORT_READ_BYTES < fixture.blockSize());
            fixture.routed.seek(0);
            fixture.routed.readBytes(fromRoute, 0, fromRoute.length);
            assertEquals("a file whose channel failed never touches it again", fixture.readsAtSettle + 1, channels.reads());
        }
    }

    public void testAChannelThatThrowsFallsBackToTheDelegate() throws IOException {
        try (Fixture fixture = new Fixture()) {
            final byte[] fromRoute = new byte[4096];
            final byte[] fromReference = new byte[4096];
            channels.behave(Behaviour.THROW);
            fixture.routed.seek(2048);
            fixture.routed.readBytes(fromRoute, 0, fromRoute.length);
            fixture.reference.seek(2048);
            fixture.reference.readBytes(fromReference, 0, fromReference.length);
            assertArrayEquals(fromReference, fromRoute);
            assertEquals("one device read was attempted and then the file fell back", fixture.readsAtSettle + 1, channels.reads());
        }
    }

    public void testTheFallbackOverwritesBytesAPartialReadAlreadyCopied() throws IOException {
        try (Fixture fixture = new Fixture()) {
            final byte[] fromRoute = new byte[8192];
            final byte[] fromReference = new byte[8192];
            channels.behave(Behaviour.GARBAGE_THEN_THROW);
            fixture.routed.seek(0);
            fixture.routed.readBytes(fromRoute, 0, fromRoute.length);
            fixture.reference.seek(0);
            fixture.reference.readBytes(fromReference, 0, fromReference.length);
            assertArrayEquals("the whole request is read again from its start", fromReference, fromRoute);
        }
    }

    public void testTheFailedMarkIsSharedByEverySliceAndClone() throws IOException {
        try (Fixture fixture = new Fixture()) {
            final byte[] target = new byte[2048];
            channels.behave(Behaviour.THROW);
            fixture.routed.seek(0);
            fixture.routed.readBytes(target, 0, target.length);
            final int readsAfterFailure = channels.last().reads().size();
            channels.behave(Behaviour.NORMAL);

            final IndexInput anotherClone = fixture.file.clone();
            final IndexInput anotherSlice = fixture.file.slice("after", 0, fixture.file.length());
            anotherClone.seek(0);
            anotherClone.readBytes(target, 0, target.length);
            anotherSlice.seek(0);
            anotherSlice.readBytes(target, 0, target.length);
            fixture.routed.seek(4096);
            fixture.routed.readBytes(target, 0, target.length);

            assertEquals("no slice or clone of a failed file touches the channel again", readsAfterFailure, channels.last().reads().size());
        }
    }

    public void testAnInterruptedReaderFallsBackAndTheChannelStaysFailed() throws IOException {
        try (Fixture fixture = new Fixture()) {
            final byte[] fromRoute = new byte[4096];
            final byte[] fromReference = new byte[4096];
            fixture.reference.seek(0);
            fixture.reference.readBytes(fromReference, 0, fromReference.length);
            try {
                Thread.currentThread().interrupt();
                fixture.routed.seek(0);
                fixture.routed.readBytes(fromRoute, 0, fromRoute.length);
            } finally {
                Thread.interrupted();
            }
            assertArrayEquals("an interrupted read is served by the original input", fromReference, fromRoute);
            final int readsAfterFailure = channels.last().reads().size();
            fixture.routed.seek(8192);
            fixture.routed.readBytes(fromRoute, 0, fromRoute.length);
            assertEquals("the channel an interrupt closed never reopens", readsAfterFailure, channels.last().reads().size());
        }
    }

    public void testTheIntegerReadsFallBackAsWholeRequests() throws IOException {
        try (Fixture fixture = new Fixture()) {
            channels.behave(Behaviour.THROW);
            fixture.routed.seek(64);
            fixture.reference.seek(64);
            assertEquals(fixture.reference.readLong(), fixture.routed.readLong());
            assertEquals("one readLong of eight bytes advances eight, not one per fallback", 72L, fixture.routed.getFilePointer());

            assertEquals(fixture.reference.readInt(), fixture.routed.readInt());
            assertEquals(76L, fixture.routed.getFilePointer());
            assertEquals(fixture.reference.readShort(), fixture.routed.readShort());
            assertEquals(78L, fixture.routed.getFilePointer());

            final int[] ints = new int[16];
            final int[] referenceInts = new int[16];
            fixture.routed.readInts(ints, 0, ints.length);
            fixture.reference.readInts(referenceInts, 0, referenceInts.length);
            assertArrayEquals(referenceInts, ints);
            assertEquals("one readInts of sixteen is one request read whole", 78L + 64L, fixture.routed.getFilePointer());

            final long[] longs = new long[8];
            final long[] referenceLongs = new long[8];
            fixture.routed.readLongs(longs, 0, longs.length);
            fixture.reference.readLongs(referenceLongs, 0, referenceLongs.length);
            assertArrayEquals(referenceLongs, longs);
            assertEquals(78L + 64L + 64L, fixture.routed.getFilePointer());
            assertEquals(
                "every one of those requests went to the original input after the first failure",
                fixture.readsAtSettle + 1,
                channels.reads()
            );
        }
    }

    public void testReadsFloatsThatStraddleTheEndOfTheBuffer() throws IOException {
        try (Fixture fixture = new Fixture()) {
            // A slice whose offset is not a multiple of four, which is the shape Lucene's flat reader produces: the
            // vector data starts just after the codec header. Every buffer window then ends with one, two or three
            // bytes of a float, which is the hand-rolled reassembly path.
            final long sliceOffset = 3;
            final IndexInput slice = fixture.file.slice("unaligned", sliceOffset, fixture.file.length() - sliceOffset);
            assertTrue(slice instanceof DirectIOVectorIndexInput);
            final int block = fixture.blockSize();
            final int floats = 2 * block;
            final float[] fromRoute = new float[floats];
            final float[] fromReference = new float[floats];
            for (final long start : new long[] { 0, block - 7, 2L * block - 1, 4L * block - 3 }) {
                slice.seek(start);
                slice.readFloats(fromRoute, 0, floats);
                fixture.reference.seek(sliceOffset + start);
                fixture.reference.readFloats(fromReference, 0, floats);
                assertArrayEquals("readFloats at " + start + " of an unaligned slice", fromReference, fromRoute, 0.0f);
            }
            assertTrue("the straddling reads went to the device and never fell back", channels.reads() > fixture.readsAtSettle + 4);
        }
    }

    public void testContentMatchesIsFalseWhenTheFileOnDiskIsLonger() throws IOException {
        try (Directory in = DirectIOTestUtils.fsDirectory(shardPath)) {
            DirectIOTestUtils.writeCodecFile(in, VEC_NAME, FILE_PAYLOAD_BYTES);
            try (IndexInput delegate = in.openInput(VEC_NAME, IOContext.DEFAULT)) {
                // The delegate's length is the one it read when it was opened. The file grows underneath it, which is
                // what a layer that pads the bytes it stores looks like from the O_DIRECT side.
                java.nio.file.Files.write(
                    shardPath.resolve(VEC_NAME),
                    DirectIOTestUtils.payload(4096),
                    java.nio.file.StandardOpenOption.APPEND
                );
                final KNNDirectIODirectory directory = DirectIOTestUtils.directory(in, on::get);
                final DirectIOVectorIndexInput.Handle handle = directory.openDirect(shardPath.resolve(VEC_NAME), delegate.length());
                assertNotNull(handle);
                try {
                    assertFalse(
                        "the length the chain reports must be compared against the file on disk",
                        directory.contentMatches(handle, delegate)
                    );
                } finally {
                    handle.close();
                }
            }
        }
    }

    public void testContentMatchesIsFalseWhenTheHeaderDiffers() throws IOException {
        try (Directory in = DirectIOTestUtils.fsDirectory(shardPath)) {
            DirectIOTestUtils.writeCodecFile(in, VEC_NAME, FILE_PAYLOAD_BYTES);
            final KNNDirectIODirectory directory = DirectIOTestUtils.directory(in, on::get);
            final DirectIOVectorIndexInput.Handle handle = directory.openDirect(shardPath.resolve(VEC_NAME), in.fileLength(VEC_NAME));
            assertNotNull(handle);
            try (Directory crypto = new DirectIOTestUtils.CryptoShapedFsDirectory(shardPath)) {
                try (IndexInput transformed = crypto.openInput(VEC_NAME, IOContext.DEFAULT)) {
                    assertFalse(directory.contentMatches(handle, transformed));
                }
            } finally {
                handle.close();
            }
        }
    }

    public void testContentMatchesIsFalseWhenTheLengthDiffers() throws IOException {
        try (Directory in = DirectIOTestUtils.fsDirectory(shardPath)) {
            DirectIOTestUtils.writeCodecFile(in, VEC_NAME, FILE_PAYLOAD_BYTES);
            final KNNDirectIODirectory directory = DirectIOTestUtils.directory(in, on::get);
            final DirectIOVectorIndexInput.Handle handle = directory.openDirect(shardPath.resolve(VEC_NAME), in.fileLength(VEC_NAME));
            assertNotNull(handle);
            try (IndexInput full = in.openInput(VEC_NAME, IOContext.DEFAULT)) {
                final IndexInput shorter = full.slice("shorter", 0, full.length() - 1);
                assertFalse(directory.contentMatches(handle, shorter));
            } finally {
                handle.close();
            }
        }
    }

    public void testContentMatchesIsFalseWhenTheFooterDiffers() throws IOException {
        try (Directory in = DirectIOTestUtils.fsDirectory(shardPath)) {
            final byte[] bytes = DirectIOTestUtils.writeCodecFile(in, VEC_NAME, FILE_PAYLOAD_BYTES);
            final String tweaked = "footer-differs.vec";
            final byte[] copy = bytes.clone();
            copy[copy.length - 1] = (byte) (copy[copy.length - 1] ^ 0xFF);
            java.nio.file.Files.write(shardPath.resolve(tweaked), copy);
            final KNNDirectIODirectory directory = DirectIOTestUtils.directory(in, on::get);
            final DirectIOVectorIndexInput.Handle handle = directory.openDirect(shardPath.resolve(VEC_NAME), in.fileLength(VEC_NAME));
            assertNotNull(handle);
            try (IndexInput other = in.openInput(tweaked, IOContext.DEFAULT)) {
                assertFalse(
                    "the same length and the same first 4 KiB, a different last byte of the checksum",
                    directory.contentMatches(handle, other)
                );
            } finally {
                handle.close();
            }
        }
    }

    public void testContentMatchesIsTrueForAnInteriorOnlyTransform() throws IOException {
        try (Directory in = DirectIOTestUtils.fsDirectory(shardPath)) {
            final byte[] bytes = DirectIOTestUtils.writeCodecFile(in, VEC_NAME, FILE_PAYLOAD_BYTES);
            final KNNDirectIODirectory directory = DirectIOTestUtils.directory(in, on::get);
            final DirectIOVectorIndexInput.Handle handle = directory.openDirect(shardPath.resolve(VEC_NAME), in.fileLength(VEC_NAME));
            assertNotNull(handle);
            try (IndexInput interior = new InteriorFlippingIndexInput(in.openInput(VEC_NAME, IOContext.DEFAULT), bytes.length / 2)) {
                assertTrue(
                    "a transform that leaves the header and the footer alone is not detected, and the javadoc says so",
                    directory.contentMatches(handle, interior)
                );
            } finally {
                handle.close();
            }
        }
    }

    public void testHalfFloatVectorsReadEqualOnBothRoutes() throws IOException {
        // A half_float field reads one vector as 2 x dimension bytes through readBytes on a slice, rather than
        // through readFloats, and 1,023 dimensions straddle every block boundary.
        final int dimension = 1_023;
        final int vectorBytes = dimension * Short.BYTES;
        final int vectors = 64;
        try (Directory in = DirectIOTestUtils.fsDirectory(shardPath)) {
            DirectIOTestUtils.writeCodecFile(in, DirectIOTestUtils.HALF_FLOAT_VEC_NAME, vectorBytes * vectors);
            final KNNDirectIODirectory directory = DirectIOTestUtils.directory(in, on::get, channels);
            try (
                IndexInput file = directory.openInput(DirectIOTestUtils.HALF_FLOAT_VEC_NAME, IOContext.DEFAULT);
                IndexInput reference = in.openInput(DirectIOTestUtils.HALF_FLOAT_VEC_NAME, IOContext.DEFAULT)
            ) {
                final IndexInput routedSlice = file.slice("KNN1040HalfFloatFlatVectorsValuesSlice", 0, (long) vectorBytes * vectors);
                assertTrue(routedSlice instanceof DirectIOVectorIndexInput);
                final int afterSettle = channels.reads();
                final byte[] fromRoute = new byte[vectorBytes];
                final byte[] fromReference = new byte[vectorBytes];
                for (int ord = 0; ord < vectors; ord++) {
                    routedSlice.seek((long) ord * vectorBytes);
                    routedSlice.readBytes(fromRoute, 0, vectorBytes);
                    reference.seek((long) ord * vectorBytes);
                    reference.readBytes(fromReference, 0, vectorBytes);
                    assertArrayEquals("vector " + ord, fromReference, fromRoute);
                }
                assertTrue("the FP16 reads went to the device", channels.reads() > afterSettle);

                on.set(false);
                final IndexInput offSlice = file.slice("off", 0, (long) vectorBytes * vectors);
                assertFalse(offSlice instanceof DirectIOVectorIndexInput);
                for (int ord = 0; ord < vectors; ord++) {
                    offSlice.seek((long) ord * vectorBytes);
                    offSlice.readBytes(fromRoute, 0, vectorBytes);
                    reference.seek((long) ord * vectorBytes);
                    reference.readBytes(fromReference, 0, vectorBytes);
                    assertArrayEquals("vector " + ord + " with the index off", fromReference, fromRoute);
                }
            }
        }
    }

    /** An input that flips one interior byte, which the content check cannot see. */
    private static final class InteriorFlippingIndexInput extends IndexInput {
        private final IndexInput in;
        private final long flipAt;

        private InteriorFlippingIndexInput(final IndexInput in, final long flipAt) {
            super("interiorFlipping(" + in + ")");
            this.in = in;
            this.flipAt = flipAt;
        }

        @Override
        public byte readByte() throws IOException {
            final long position = in.getFilePointer();
            final byte value = in.readByte();
            return position == flipAt ? (byte) (value ^ 0xFF) : value;
        }

        @Override
        public void readBytes(final byte[] b, final int offset, final int len) throws IOException {
            final long position = in.getFilePointer();
            in.readBytes(b, offset, len);
            if (position <= flipAt && flipAt < position + len) {
                final int index = offset + (int) (flipAt - position);
                b[index] = (byte) (b[index] ^ 0xFF);
            }
        }

        @Override
        public void close() throws IOException {
            in.close();
        }

        @Override
        public long getFilePointer() {
            return in.getFilePointer();
        }

        @Override
        public void seek(final long pos) throws IOException {
            in.seek(pos);
        }

        @Override
        public long length() {
            return in.length();
        }

        @Override
        public IndexInput slice(final String sliceDescription, final long offset, final long length) throws IOException {
            return in.slice(sliceDescription, offset, length);
        }

        @Override
        public IndexInput clone() {
            return new InteriorFlippingIndexInput(in.clone(), flipAt);
        }
    }
}

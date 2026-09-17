/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.store.MergeInfo;
import org.apache.lucene.store.NIOFSDirectory;
import org.opensearch.knn.KNNTestCase;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.OptionalLong;
import java.util.Random;

/**
 * Tests the {@link KNNDirectIODirectory} routing gate, and one end to end round trip that asserts
 * Direct I/O was actually engaged rather than silently skipped.
 */
public class KNNDirectIODirectoryTests extends KNNTestCase {

    private static final long MIN_BYTES_DIRECT = 1024 * 1024;
    private static final int BUFFER_SIZE = 8192;
    private static final String VEC_FILE = "_0_NativeEngines990.vec";
    private static final String DIRECT_INPUT_CLASS = "DirectIOIndexInput";

    private static boolean gate(final String name, final IOContext context, final OptionalLong fileLength) {
        return KNNDirectIODirectory.shouldUseDirectIO(name, context, fileLength, MIN_BYTES_DIRECT);
    }

    private static IOContext mergeContext() {
        return IOContext.merge(new MergeInfo(1000, 100 * MIN_BYTES_DIRECT, false, 1));
    }

    private static KNNDirectIODirectory directory(final Path path) throws IOException {
        return new KNNDirectIODirectory(new NIOFSDirectory(path), BUFFER_SIZE, MIN_BYTES_DIRECT);
    }

    /**
     * Trap #1, at the boolean level: an absent file length marks an output request, and Lucene's
     * {@code DirectIOIndexOutput} opens with {@code CREATE_NEW}, so routing writes would break any
     * rewrite of an existing file.
     */
    public void testEmptyFileLengthIsNeverRouted() {
        assertFalse(gate(VEC_FILE, IOContext.DEFAULT, OptionalLong.empty()));
        assertFalse(gate(VEC_FILE, mergeContext(), OptionalLong.empty()));
    }

    /**
     * Trap #1, at the behavioural level: writing a {@code .vec} file through the wrapper must work,
     * be re-writable, and produce a file on disk of the right length. An assertion on the gate alone
     * would not catch a {@code CREATE_NEW} failure.
     */
    public void testVecFilesCanStillBeWrittenAndRewritten() throws IOException {
        final Path path = createTempDir();
        try (KNNDirectIODirectory dir = directory(path)) {
            writeBytes(dir, VEC_FILE, new byte[3072]);
            assertEquals(3072L, dir.fileLength(VEC_FILE));

            // a rewrite of the same name, which is what CREATE_NEW would reject
            dir.deleteFile(VEC_FILE);
            writeBytes(dir, VEC_FILE, new byte[100]);
            assertEquals(100L, dir.fileLength(VEC_FILE));
        }
        assertEquals(100L, Files.size(path.resolve(VEC_FILE)));
    }

    /**
     * Merges keep the delegate: their reads are sequential and the page cache serves them well, so
     * this override deliberately inverts the superclass' merge-only default.
     */
    public void testMergeContextIsNeverRouted() {
        final OptionalLong big = OptionalLong.of(10 * MIN_BYTES_DIRECT);
        assertFalse(gate(VEC_FILE, mergeContext(), big));
        // ... while the same file on a search context is routed, so the merge check is what made the
        // difference and not the name or the size.
        assertTrue(gate(VEC_FILE, IOContext.DEFAULT, big));
    }

    /**
     * The include list: only a final extension of exactly {@code vec} is routed.
     */
    public void testOnlyVecExtensionIsRouted() {
        final OptionalLong big = OptionalLong.of(10 * MIN_BYTES_DIRECT);

        assertTrue(gate(VEC_FILE, IOContext.DEFAULT, big));
        assertTrue(gate("_0_Lucene99.vec", IOContext.DEFAULT, big));

        assertFalse(gate("_0_NativeEngines990.vex", IOContext.DEFAULT, big));
        assertFalse(gate("_0_165.faiss", IOContext.DEFAULT, big));
        assertFalse(gate("_0.vecm", IOContext.DEFAULT, big));
        // A temporary file whose final extension is .tmp, not .vec. Worth an explicit case because
        // FileSwitchDirectory.getExtension — the obvious helper to reach for here — reports this one
        // as "vec": it looks through a trailing .tmp on purpose. The gate must not.
        assertFalse(gate("_0_NativeEngines990.vec.tmp", IOContext.DEFAULT, big));
        assertFalse(gate("_0_NativeEngines990.vec_0.tmp", IOContext.DEFAULT, big));
        // no extension at all
        assertFalse(gate("segments_3", IOContext.DEFAULT, big));
        assertFalse(gate("write.lock", IOContext.DEFAULT, big));
        // ".vec" appearing in a directory-like prefix rather than as the extension
        assertFalse(gate("some.vec/segments_3", IOContext.DEFAULT, big));
        assertTrue(gate("some.dir/_0.vec", IOContext.DEFAULT, big));
    }

    /**
     * The size floor is inclusive at the boundary, per {@code >=}.
     */
    public void testSizeFloorBoundary() {
        assertFalse(gate(VEC_FILE, IOContext.DEFAULT, OptionalLong.of(MIN_BYTES_DIRECT - 1)));
        assertTrue(gate(VEC_FILE, IOContext.DEFAULT, OptionalLong.of(MIN_BYTES_DIRECT)));
        assertTrue(gate(VEC_FILE, IOContext.DEFAULT, OptionalLong.of(MIN_BYTES_DIRECT + 1)));
        assertFalse(gate(VEC_FILE, IOContext.DEFAULT, OptionalLong.of(0)));
    }

    /**
     * A file below the floor and a file with the wrong extension are both opened by the delegate, so
     * their {@link IndexInput} class is the delegate's — the negative half of the engagement check.
     */
    public void testUnroutedFilesAreOpenedByTheDelegate() throws IOException {
        final Path path = createTempDir();
        try (KNNDirectIODirectory dir = directory(path); Directory plain = new NIOFSDirectory(path)) {
            writeBytes(dir, "small.vec", new byte[4096]);
            writeBytes(dir, "big.vex", randomBytes((int) MIN_BYTES_DIRECT + 7));

            assertSameInputClass(dir, plain, "small.vec");
            assertSameInputClass(dir, plain, "big.vex");
        }
    }

    /**
     * The round trip that matters: a {@code .vec} file of a length that is not a multiple of the
     * filesystem block size, written and read back byte for byte through the wrapper, with an
     * assertion that the read really went through Lucene's Direct I/O input and not the delegate.
     */
    public void testVecRoundTripActuallyUsesDirectIO() throws IOException {
        final Path path = createTempDir();
        assumeTrue("Direct I/O is not available here", isDirectIOAvailable(path));

        // deliberately not a block multiple, so the last read covers an unaligned tail
        final byte[] expected = randomBytes((int) MIN_BYTES_DIRECT + 1237);
        try (KNNDirectIODirectory dir = directory(path)) {
            writeBytes(dir, VEC_FILE, expected);

            try (IndexInput in = dir.openInput(VEC_FILE, IOContext.DEFAULT)) {
                assertEquals(
                    "expected Lucene's Direct I/O input, got " + in.getClass().getName(),
                    DIRECT_INPUT_CLASS,
                    in.getClass().getSimpleName()
                );
                assertEquals(expected.length, in.length());

                final byte[] actual = new byte[expected.length];
                in.readBytes(actual, 0, actual.length);
                assertArrayEquals(expected, actual);

                // a sparse, non sequential re-read, which is the shape of the rescore path and the
                // case where the buffer refill logic is actually exercised. Includes the unaligned
                // tail, since the first offset is length - 8.
                for (int offset = expected.length - 8; offset >= 0; offset -= 3072) {
                    in.seek(offset);
                    final byte[] chunk = new byte[8];
                    in.readBytes(chunk, 0, 8);
                    for (int i = 0; i < 8; i++) {
                        assertEquals("byte " + (offset + i), expected[offset + i], chunk[i]);
                    }
                }
            }
        }
    }

    /**
     * Direct I/O may be unavailable on the JDK or rejected by the filesystem under the temp dir. The
     * gate is filesystem independent, but the round trip is not.
     */
    private static boolean isDirectIOAvailable(final Path path) {
        try (KNNDirectIODirectory dir = directory(path)) {
            writeBytes(dir, "probe.vec", new byte[(int) MIN_BYTES_DIRECT + 3]);
            try (IndexInput in = dir.openInput("probe.vec", IOContext.DEFAULT)) {
                in.readByte();
                return DIRECT_INPUT_CLASS.equals(in.getClass().getSimpleName());
            } finally {
                dir.deleteFile("probe.vec");
            }
        } catch (IOException | UnsupportedOperationException e) {
            return false;
        }
    }

    private static byte[] randomBytes(final int length) {
        final byte[] bytes = new byte[length];
        new Random(length).nextBytes(bytes);
        return bytes;
    }

    private static void writeBytes(final Directory directory, final String name, final byte[] bytes) throws IOException {
        try (IndexOutput out = directory.createOutput(name, IOContext.DEFAULT)) {
            out.writeBytes(bytes, bytes.length);
        }
    }

    private static void assertSameInputClass(final Directory ours, final Directory delegate, final String name) throws IOException {
        try (IndexInput a = ours.openInput(name, IOContext.DEFAULT); IndexInput b = delegate.openInput(name, IOContext.DEFAULT)) {
            assertEquals("input class differs for " + name, b.getClass(), a.getClass());
        }
    }
}

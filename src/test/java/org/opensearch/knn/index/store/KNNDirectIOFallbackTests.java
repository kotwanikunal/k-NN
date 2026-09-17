/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.store.NIOFSDirectory;
import org.opensearch.Version;
import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.common.settings.Settings;
import org.opensearch.core.index.shard.ShardId;
import org.opensearch.index.IndexModule;
import org.opensearch.index.IndexSettings;
import org.opensearch.index.shard.ShardPath;
import org.opensearch.index.store.FsDirectoryFactory;
import org.opensearch.knn.KNNTestCase;

import java.io.IOException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;

import static org.opensearch.knn.index.store.KNNDirectIODirectoryFactory.KNN_DIRECT_IO_STORE_TYPE;

/**
 * Unit 13's job: Direct I/O being unavailable must degrade to today's path, never fail a shard open
 * and never fail a query. These tests inject each of the recoverable failures for real rather than
 * mocking them, and assert both that the fallback happened and that the bytes came back correct.
 * <p>
 * The one failure that is deliberately not defended against is a misaligned read mid-stream, which
 * throws from inside Lucene's {@code refill} with the {@link IndexInput} already in the caller's
 * hands. {@link #testNonBlockAlignedTailIsReadCorrectly()} is the test that this case cannot arise:
 * alignment is a property of the geometry, and the geometry is chosen by
 * {@link DirectIOBufferSizer}, so the defence is construction and not a run-time guard.
 */
public class KNNDirectIOFallbackTests extends KNNTestCase {

    private static final String TEST_INDEX = "test-index";
    private static final String VEC_FILE = "_0_NativeEngines990.vec";
    private static final long MIN_BYTES_DIRECT = 1024 * 1024;
    private static final int BUFFER_SIZE = 8192;
    private static final String DIRECT_INPUT_CLASS = "DirectIOIndexInput";

    /** A tmpfs mount. tmpfs answers {@code open(O_DIRECT)} with {@code EINVAL}, which is failure point 2. */
    private static final Path TMPFS = Path.of("/dev/shm");

    /**
     * A construction failure — {@code Files.getFileStore(...).getBlockSize()} throwing, which really
     * happens on filesystems that do not implement the attribute — must produce the stock directory,
     * not a shard-open failure.
     */
    public void testBlockSizeIOExceptionDegradesToStockDirectory() throws IOException {
        assertDegradesToStockDirectory(new IOException("no file store here"));
    }

    /**
     * The same, for the unchecked half. {@code getBlockSize} is documented to throw
     * {@link UnsupportedOperationException} when the file store does not support the attribute, and
     * {@code Math.toIntExact} throws {@link ArithmeticException} on an absurd one — both arrive as
     * {@link RuntimeException} and both must be caught.
     */
    public void testBlockSizeRuntimeExceptionDegradesToStockDirectory() throws IOException {
        assertDegradesToStockDirectory(new UnsupportedOperationException("block size unsupported"));
        assertDegradesToStockDirectory(new ArithmeticException("integer overflow"));
    }

    /**
     * With the block size unavailable, the factory must return a directory indistinguishable from the
     * node default — {@code isHybridFs}, same class as the stock factory's — and a {@code .vec} file
     * above the size floor must still open and read back byte for byte.
     */
    private void assertDegradesToStockDirectory(final Throwable failure) throws IOException {
        final Path root = createTempDir();
        final IndexSettings ourSettings = indexSettings(
            Settings.builder().put(IndexModule.INDEX_STORE_TYPE_SETTING.getKey(), KNN_DIRECT_IO_STORE_TYPE).build()
        );
        final IndexSettings stockSettings = indexSettings(Settings.EMPTY);
        final byte[] expected = randomBytes((int) MIN_BYTES_DIRECT + 4321);

        try (
            Directory ours = failingBlockSizeFactory(failure).newDirectory(ourSettings, shardPath(root, ourSettings));
            Directory stock = new FsDirectoryFactory().newDirectory(stockSettings, shardPath(root, stockSettings))
        ) {
            assertFalse("expected the stock directory, got " + ours.getClass().getName(), ours instanceof KNNDirectIODirectory);
            assertEquals(stock.getClass(), ours.getClass());
            assertTrue(FsDirectoryFactory.isHybridFs(ours));

            writeBytes(ours, VEC_FILE, expected);
            try (IndexInput in = ours.openInput(VEC_FILE, IOContext.DEFAULT)) {
                assertNotEquals(DIRECT_INPUT_CLASS, in.getClass().getSimpleName());
                assertArrayEquals(expected, readAll(in));
            }
            ours.deleteFile(VEC_FILE);
        }
    }

    /**
     * Failure point 2, injected for real: a filesystem that refuses {@code O_DIRECT}. The
     * {@link KNNDirectIODirectory} constructor succeeds there — tmpfs reports a block size perfectly
     * well — so the refusal only surfaces at {@code openInput}, per file, which is exactly the case
     * the per-file fallback exists for. Without it this would throw out of a query.
     */
    public void testFilesystemRefusingDirectIOFallsBackPerFile() throws IOException {
        assumeTrue("no writable tmpfs at " + TMPFS, Files.isDirectory(TMPFS) && Files.isWritable(TMPFS));
        final Path path = Files.createTempDirectory(TMPFS, "knn-dio-fallback-");
        try {
            assumeFalse("this tmpfs accepted O_DIRECT, so there is no refusal to fall back from", directIOWorks(path));

            final byte[] expected = randomBytes((int) MIN_BYTES_DIRECT + 999);
            try (KNNDirectIODirectory dir = directory(path); Directory plain = new NIOFSDirectory(path)) {
                writeBytes(dir, VEC_FILE, expected);
                assertFalse("nothing has failed yet", dir.isDirectIOUnavailable());

                // the gate selects this file, the Direct I/O open is refused, and the delegate serves it
                try (
                    IndexInput in = dir.openInput(VEC_FILE, IOContext.DEFAULT);
                    IndexInput ref = plain.openInput(VEC_FILE, IOContext.DEFAULT)
                ) {
                    assertEquals("expected the delegate's input class", ref.getClass(), in.getClass());
                    assertArrayEquals(expected, readAll(in));
                }
                assertTrue("the failure should have latched", dir.isDirectIOUnavailable());

                // and once latched, a second routed file is served by the delegate without another
                // failed open, still correctly
                final byte[] second = randomBytes((int) MIN_BYTES_DIRECT + 17);
                writeBytes(dir, "_1_NativeEngines990.vec", second);
                try (IndexInput in = dir.openInput("_1_NativeEngines990.vec", IOContext.DEFAULT)) {
                    assertArrayEquals(second, readAll(in));
                }
            }
        } finally {
            deleteScratchDirectory(path);
        }
    }

    /**
     * A file the gate does not route must behave identically whether or not Direct I/O is usable, and
     * must not set the latch — the latch means "Direct I/O failed", not "a file was delegated".
     */
    public void testDelegatedFilesDoNotSetTheLatch() throws IOException {
        final Path path = createTempDir();
        try (KNNDirectIODirectory dir = directory(path)) {
            writeBytes(dir, "segments_3", randomBytes(64));
            writeBytes(dir, "small.vec", randomBytes(4096));
            try (
                IndexInput a = dir.openInput("segments_3", IOContext.DEFAULT);
                IndexInput b = dir.openInput("small.vec", IOContext.DEFAULT)
            ) {
                assertEquals(64L, a.length());
                assertEquals(4096L, b.length());
            }
            assertFalse(dir.isDirectIOUnavailable());
        }
    }

    /**
     * The fallback must not swallow the delegate's own errors. A missing file has to come back as a
     * {@link java.nio.file.NoSuchFileException} from the delegate, not as a Direct I/O warning and a
     * second confusing exception, and it must not latch Direct I/O off.
     */
    public void testMissingFileStillThrowsFromTheDelegate() throws IOException {
        final Path path = createTempDir();
        try (KNNDirectIODirectory dir = directory(path)) {
            expectThrows(java.nio.file.NoSuchFileException.class, () -> dir.openInput("absent.vec", IOContext.DEFAULT));
            assertFalse("a missing file is not a Direct I/O failure", dir.isDirectIOUnavailable());
        }
    }

    /**
     * The highest-value test in Phase 4: a {@code .vec} whose length is not a whole number of
     * filesystem blocks, read through the Direct I/O path all the way into the partial tail block.
     * O_DIRECT requires block-aligned offsets and lengths, so the tail is where a hand-rolled reader
     * gets it wrong — the previous POC did. Lucene's input handles it by always reading whole blocks
     * and bounding the returned length, and this asserts that end to end.
     */
    public void testNonBlockAlignedTailIsReadCorrectly() throws IOException {
        final Path path = createTempDir();
        assumeTrue("Direct I/O is not available here", directIOWorks(path));
        final int blockSize = Math.toIntExact(Files.getFileStore(path).getBlockSize());

        // deliberately n whole blocks plus a 37 byte tail, and comfortably above the size floor
        final int blocks = (int) (MIN_BYTES_DIRECT / blockSize) + 3;
        final byte[] expected = randomBytes(blocks * blockSize + 37);

        try (KNNDirectIODirectory dir = directory(path)) {
            writeBytes(dir, VEC_FILE, expected);
            try (IndexInput in = dir.openInput(VEC_FILE, IOContext.DEFAULT)) {
                assertEquals("expected Lucene's Direct I/O input", DIRECT_INPUT_CLASS, in.getClass().getSimpleName());
                assertEquals(expected.length, in.length());

                // the whole file, which ends inside the partial block
                assertArrayEquals(expected, readAll(in));

                // the tail on its own, reached by a seek rather than by streaming into it
                in.seek(blocks * (long) blockSize);
                final byte[] tail = new byte[37];
                in.readBytes(tail, 0, 37);
                for (int i = 0; i < 37; i++) {
                    assertEquals("tail byte " + i, expected[blocks * blockSize + i], tail[i]);
                }

                // the last byte, and then one byte past the end
                in.seek(expected.length - 1);
                assertEquals(expected[expected.length - 1], in.readByte());
                expectThrows(java.io.EOFException.class, in::readByte);

                // a read straddling the final block boundary
                in.seek(blocks * (long) blockSize - 5);
                final byte[] straddle = new byte[42];
                in.readBytes(straddle, 0, 42);
                for (int i = 0; i < 42; i++) {
                    assertEquals("straddling byte " + i, expected[blocks * blockSize - 5 + i], straddle[i]);
                }
            }
            assertFalse("an unaligned tail is not a failure", dir.isDirectIOUnavailable());
        }
    }

    /**
     * Failure point 1 is a property of the runtime, so it cannot be injected here. What can be
     * asserted is the invariant the factory relies on: if the open option is missing then a real
     * Direct I/O open cannot succeed, which is what makes declining to wrap the right response.
     */
    public void testOpenOptionAvailabilityAgreesWithRealOpens() throws IOException {
        final Path path = createTempDir();
        if (KNNDirectIODirectory.isDirectIOOpenOptionAvailable() == false) {
            assertFalse("the option is absent, so no Direct I/O open can succeed", directIOWorks(path));
        }
    }

    private static KNNDirectIODirectory directory(final Path path) throws IOException {
        return new KNNDirectIODirectory(new NIOFSDirectory(path), BUFFER_SIZE, MIN_BYTES_DIRECT);
    }

    /**
     * A factory whose one filesystem-touching step fails, which is the only way to exercise the
     * construction fallback without a filesystem that has the defect.
     */
    private static KNNDirectIODirectoryFactory failingBlockSizeFactory(final Throwable failure) {
        return new KNNDirectIODirectoryFactory() {
            @Override
            int blockSize(final Path location) throws IOException {
                if (failure instanceof IOException) {
                    throw (IOException) failure;
                }
                throw (RuntimeException) failure;
            }
        };
    }

    private static boolean directIOWorks(final Path path) throws IOException {
        try (KNNDirectIODirectory dir = directory(path)) {
            writeBytes(dir, "probe.vec", new byte[(int) MIN_BYTES_DIRECT + 3]);
            try (IndexInput in = dir.openInput("probe.vec", IOContext.DEFAULT)) {
                in.readByte();
                return DIRECT_INPUT_CLASS.equals(in.getClass().getSimpleName()) && dir.isDirectIOUnavailable() == false;
            } finally {
                dir.deleteFile("probe.vec");
            }
        } catch (IOException | UnsupportedOperationException e) {
            return false;
        }
    }

    private static IndexSettings indexSettings(final Settings extraIndexSettings) {
        final Settings settings = Settings.builder()
            .put(IndexMetadata.SETTING_VERSION_CREATED, Version.CURRENT)
            .put(IndexMetadata.SETTING_NUMBER_OF_SHARDS, 1)
            .put(IndexMetadata.SETTING_NUMBER_OF_REPLICAS, 0)
            .put(extraIndexSettings)
            .build();
        return new IndexSettings(IndexMetadata.builder(TEST_INDEX).settings(settings).build(), Settings.EMPTY);
    }

    private static ShardPath shardPath(final Path root, final IndexSettings indexSettings) {
        final Path shardDir = root.resolve(indexSettings.getIndex().getUUID()).resolve("0");
        return new ShardPath(false, shardDir, shardDir, new ShardId(indexSettings.getIndex(), 0));
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

    private static byte[] readAll(final IndexInput in) throws IOException {
        in.seek(0);
        final byte[] bytes = new byte[Math.toIntExact(in.length())];
        in.readBytes(bytes, 0, bytes.length);
        return bytes;
    }

    /**
     * The tmpfs scratch directory is outside the test framework's temp-dir tracking, so it is cleaned
     * up here. Flat rather than recursive because only files are ever created in it.
     */
    private static void deleteScratchDirectory(final Path path) throws IOException {
        if (Files.isDirectory(path) == false) {
            return;
        }
        try (DirectoryStream<Path> entries = Files.newDirectoryStream(path)) {
            for (Path entry : entries) {
                Files.deleteIfExists(entry);
            }
        }
        Files.deleteIfExists(path);
    }
}

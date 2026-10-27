/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import com.carrotsearch.randomizedtesting.annotations.ThreadLeakFilters;
import lombok.SneakyThrows;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.DataAccessHint;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FileDataHint;
import org.apache.lucene.store.FileTypeHint;
import org.apache.lucene.store.FilterDirectory;
import org.apache.lucene.store.FlushInfo;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.store.NIOFSDirectory;
import org.apache.lucene.store.NRTCachingDirectory;
import org.opensearch.common.lucene.store.ByteArrayIndexInput;
import org.opensearch.common.settings.Settings;
import org.opensearch.core.index.shard.ShardId;
import org.opensearch.index.IndexModule;
import org.opensearch.index.IndexSettings;
import org.opensearch.index.shard.ShardPath;
import org.opensearch.index.store.FsDirectoryFactory;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.codec.KNN1040Codec.Faiss1040HalfFloatScalarQuantizedKnnVectorsFormat;
import org.opensearch.knn.index.codec.KNN1040Codec.Faiss1040ScalarQuantizedKnnVectorsFormat;
import org.opensearch.knn.index.codec.KNN990Codec.NativeEngines990HalfFloatKnnVectorsFormat;
import org.opensearch.knn.index.codec.KNN990Codec.NativeEngines990KnnVectorsFormat;
import org.opensearch.test.IndexSettingsModule;

import java.io.IOException;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.OpenOption;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.UnaryOperator;

/**
 * The dispatch rule of {@link KNNDirectIODirectory}, asserted one condition at a time, plus the two
 * properties that decide whether installing it is safe on an index that is not using the feature.
 *
 * <p>The rule has eight conditions and the interesting failures are all failures of <em>one</em> of them:
 * routing every {@code .vec} would put the Lucene engine's unquantized traversal on Direct I/O and off
 * its memory-segment SIMD scorer, which is outside what this feature is for; ignoring the setting would
 * turn the feature on everywhere including the substrates where the measurements say it loses; serving a
 * file whose bytes are not the bytes the delegate chain returns would serve a transforming layer's
 * ciphertext as vectors; ignoring the absence of a path would turn a remote-store index into a failed query
 * rather than an ordinary one; and ignoring the size floor is measured at 8,016x on a flush-sized file. So
 * each is tested by holding the others and breaking that one, and each decline is asserted to be counted
 * under its own reason, because a node that cannot tell them apart cannot act on them.
 *
 * <p>The other two properties are about what must <em>not</em> change. A file the rule does not name has
 * to come back as the delegate's own object, not a wrapper of it — a wrapper would displace Lucene's
 * memory-segment SIMD scorer for traversal, which is invisible to every correctness test and was
 * measured at 5.3× — and the class has to carry no node-wide mutable state, which its spike predecessor
 * did.
 */
@ThreadLeakFilters(defaultFilters = true, filters = { DirectIOReadPoolTests.ReadPoolThreadFilter.class })
public class KNNDirectIODirectoryTests extends KNNTestCase {

    /** A native-engine (faiss / MOS) fp32 flat file: the shape the rule routes. */
    private static final String VECTOR_DATA = "_0_" + NativeEngines990KnnVectorsFormat.FORMAT_NAME + "_0.vec";

    /** A faiss scalar-quantized fp32 flat file: the other shape the rule routes. */
    private static final String FAISS_SQ_VECTOR_DATA = "_0_" + Faiss1040ScalarQuantizedKnnVectorsFormat.FORMAT_NAME + "_0.vec";

    private static final int FILE_BYTES = 16 * 1024;

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

    /** Deterministic bytes, so a comparison failure names the offset rather than the seed. */
    private static byte[] contentOf(final int length) {
        final byte[] bytes = new byte[length];
        for (int i = 0; i < length; i++) {
            bytes[i] = (byte) (i * 31 + 7);
        }
        return bytes;
    }

    @SneakyThrows
    private static byte[] writeFile(final Directory directory, final String name, final int length) {
        final byte[] content = contentOf(length);
        try (IndexOutput output = directory.createOutput(name, IOContext.DEFAULT)) {
            output.writeBytes(content, content.length);
        }
        return content;
    }

    /** The context Lucene's flat vectors reader builds for its own {@code .vec} open. */
    private static IOContext vectorDataContext() {
        return IOContext.DEFAULT.withHints(FileTypeHint.DATA, FileDataHint.KNN_VECTORS, DataAccessHint.RANDOM);
    }

    private static byte[] readAll(final IndexInput input) throws IOException {
        final byte[] bytes = new byte[Math.toIntExact(input.length())];
        input.seek(0);
        input.readBytes(bytes, 0, bytes.length);
        return bytes;
    }

    /** No size floor, so the conditions under test are the only ones that can decline. */
    private KNNDirectIODirectory storageOver(final Directory delegate, final boolean enabled) {
        return new KNNDirectIODirectory(delegate, "_0", () -> enabled, () -> 0L);
    }

    // -------------------------------------------------------------------------------------------------
    // All three conditions hold
    // -------------------------------------------------------------------------------------------------

    /**
     * The routed case, for both faiss rows, and the only assertion that matters about it besides the
     * type: the bytes are the delegate's bytes. A route that returned different bytes would be a
     * correctness bug that no latency measurement would catch.
     */
    @SneakyThrows
    public void testFaissVectorDataIsServedWithDirectIOAndReadsTheSameBytes() {
        assumeDirectIOWorksHere();
        final Path path = createTempDir();
        try (MMapDirectory delegate = new MMapDirectory(path)) {
            final byte[] expected = writeFile(delegate, VECTOR_DATA, FILE_BYTES);
            writeFile(delegate, FAISS_SQ_VECTOR_DATA, FILE_BYTES);
            try (KNNDirectIODirectory storage = storageOver(delegate, true)) {
                for (final String name : new String[] { VECTOR_DATA, FAISS_SQ_VECTOR_DATA }) {
                    try (IndexInput routed = storage.openInput(name, vectorDataContext())) {
                        assertTrue(
                            name + " should be served with O_DIRECT, got " + routed.getClass().getName(),
                            routed instanceof DirectIOVectorIndexInput
                        );
                        assertEquals(expected.length, routed.length());
                        assertArrayEquals("the Direct I/O route must read the delegate's bytes exactly", expected, readAll(routed));
                    }
                }
                assertEquals(2, storage.routedOpens());
                assertEquals(0, storage.declinedOpens());
            }
        }
    }

    /**
     * The name is the whole signal, so the context is not consulted at all: the same file routes however
     * it is opened. Stated as an assertion because the previous shape of this class required a
     * plugin-defined hint on the {@link IOContext}, and the simplification this replaces it with is
     * exactly "the context no longer matters".
     */
    @SneakyThrows
    public void testRoutingDoesNotDependOnTheIOContext() {
        assumeDirectIOWorksHere();
        final Path path = createTempDir();
        try (MMapDirectory delegate = new MMapDirectory(path)) {
            writeFile(delegate, VECTOR_DATA, FILE_BYTES);
            try (KNNDirectIODirectory storage = storageOver(delegate, true)) {
                for (final IOContext context : new IOContext[] {
                    IOContext.DEFAULT,
                    IOContext.READONCE,
                    vectorDataContext(),
                    IOContext.merge(new org.apache.lucene.store.MergeInfo(10, 1024, true, 1)),
                    IOContext.flush(new org.apache.lucene.store.FlushInfo(10, 1024)) }) {
                    try (IndexInput input = storage.openInput(VECTOR_DATA, context)) {
                        assertTrue("a faiss .vec routes on its name whatever the context is", input instanceof DirectIOVectorIndexInput);
                    }
                }
                assertEquals(5, storage.routedOpens());
            }
        }
    }

    // -------------------------------------------------------------------------------------------------
    // Each condition broken in turn
    // -------------------------------------------------------------------------------------------------

    /**
     * Condition 1 broken, and the case the narrowed predicate exists for: a {@code .vec} that belongs to
     * a Lucene-engine field. A routed file is routed for all of its readers, so routing these would take
     * Lucene's unquantized HNSW traversal off its memory-segment SIMD scorer. They must come back
     * <em>unwrapped</em> — the delegate's own object, not a wrapper of it.
     */
    @SneakyThrows
    public void testALuceneEngineVectorDataFileIsTheDelegatesOwnInputUnwrapped() {
        final Path path = createTempDir();
        try (MMapDirectory delegate = new MMapDirectory(path)) {
            final String[] luceneVectorData = {
                "_0_Lucene99HnswVectorsFormat_0.vec",
                "_0_Lucene99FlatVectorsFormat_0.vec",
                "_0_Lucene104HnswBinaryQuantizedVectorsFormat_0.vec" };
            for (final String name : luceneVectorData) {
                writeFile(delegate, name, FILE_BYTES);
            }
            final Class<?> delegateInputClass;
            try (IndexInput reference = delegate.openInput(luceneVectorData[0], IOContext.DEFAULT)) {
                delegateInputClass = reference.getClass();
            }
            try (KNNDirectIODirectory storage = storageOver(delegate, true)) {
                for (final String name : luceneVectorData) {
                    try (IndexInput plain = storage.openInput(name, vectorDataContext())) {
                        assertFalse(name + " is not a faiss file and must not be routed", plain instanceof DirectIOVectorIndexInput);
                        assertSame(name + " must be the delegate's own object, not a wrapper of it", delegateInputClass, plain.getClass());
                    }
                }
                assertEquals(0, storage.routedOpens());
                assertEquals(0, storage.declinedOpens());
            }
        }
    }

    /**
     * Condition 1 broken the other way: a faiss field's other files. The quantized codes are the
     * traversal path's own file and the native index is read by the JNI layer; neither is what Direct I/O
     * exists for, and routing them would route the hottest reads in the system.
     */
    @SneakyThrows
    public void testNothingButTheFullPrecisionVectorDataIsRouted() {
        assumeDirectIOWorksHere();
        final Path path = createTempDir();
        final String[] otherFaissFiles = {
            "_0_" + Faiss1040ScalarQuantizedKnnVectorsFormat.FORMAT_NAME + "_0.veq",
            "_0_" + NativeEngines990KnnVectorsFormat.FORMAT_NAME + "_0.faiss",
            "_0_" + NativeEngines990KnnVectorsFormat.FORMAT_NAME + "_0.vemf",
            "_0.si" };
        try (MMapDirectory delegate = new MMapDirectory(path)) {
            for (final String name : otherFaissFiles) {
                writeFile(delegate, name, 4096);
            }
            try (KNNDirectIODirectory storage = storageOver(delegate, true)) {
                for (final String name : otherFaissFiles) {
                    try (IndexInput input = storage.openInput(name, vectorDataContext())) {
                        assertFalse(name + " must not be routed to Direct I/O", input instanceof DirectIOVectorIndexInput);
                    }
                }
                assertEquals(0, storage.routedOpens());
            }
        }
    }

    /** Condition 2 broken: the operator's switch is off, which is the default on every node. */
    @SneakyThrows
    public void testTheSettingIsTheOperatorSwitchAndOffMeansUnchanged() {
        final Path path = createTempDir();
        try (MMapDirectory delegate = new MMapDirectory(path)) {
            final byte[] expected = writeFile(delegate, VECTOR_DATA, FILE_BYTES);
            try (KNNDirectIODirectory storage = storageOver(delegate, false)) {
                try (IndexInput input = storage.openInput(VECTOR_DATA, vectorDataContext())) {
                    assertFalse("with the setting off nothing may be routed", input instanceof DirectIOVectorIndexInput);
                    assertArrayEquals(expected, readAll(input));
                }
                assertEquals(0, storage.routedOpens());
                assertEquals("declining on the setting is not a Direct I/O failure", 0, storage.declinedOpens());
            }
        }
    }

    /**
     * Condition 3 broken: a delegate with no filesystem underneath it — a remote-store or in-memory
     * directory. The answer must be an ordinary read, because the alternative is a failed query on a
     * substrate where Direct I/O was never available in the first place.
     */
    @SneakyThrows
    public void testADirectoryWithNoPathDeclinesRatherThanFails() {
        try (ByteBuffersDirectory delegate = new ByteBuffersDirectory()) {
            final byte[] expected = writeFile(delegate, VECTOR_DATA, FILE_BYTES);
            try (KNNDirectIODirectory storage = storageOver(delegate, true)) {
                try (IndexInput input = storage.openInput(VECTOR_DATA, vectorDataContext())) {
                    assertFalse(input instanceof DirectIOVectorIndexInput);
                    assertArrayEquals("declining must still read the right bytes", expected, readAll(input));
                }
                assertEquals(0, storage.routedOpens());
                assertEquals("a decline is counted, so a node can tell it from silence", 1, storage.declinedOpens());
                assertEquals("and the reason is that there is no FSDirectory below", 1, storage.declinedChain());
            }
        }
    }

    // -------------------------------------------------------------------------------------------------
    // The compound container is no longer a dispatch point
    // -------------------------------------------------------------------------------------------------

    /**
     * A {@code .cfs} is the delegate's own object. The store-factory design wrapped every compound
     * container in a 423-line {@code KNNVectorCompoundSliceInput} so that a compound segment's {@code .vec}
     * entry — which has no {@code openInput} of its own — could be routed as a four-argument slice. Phase
     * 14 deleted it on measurement, not on taste: a compound segment is a freshly flushed one, and phase
     * 13's E4 measured a flush-sized {@code .vec} at <b>64.7 ops/s</b> on {@code O_DIRECT} against
     * <b>518,681 ops/s</b> from the page cache. The code existed to route reads that must not be routed.
     *
     * <p>Pinned as a test because the failure it guards is a future edit re-adding the interception.
     */
    @SneakyThrows
    public void testTheCompoundContainerIsNotADispatchPoint() {
        final Path path = createTempDir();
        try (MMapDirectory delegate = new MMapDirectory(path)) {
            writeFile(delegate, "_0.cfs", FILE_BYTES);
            writeFile(delegate, "_0.cfe", 1024);
            final Class<?> delegateInputClass;
            try (IndexInput reference = delegate.openInput("_0.cfe", IOContext.DEFAULT)) {
                delegateInputClass = reference.getClass();
            }
            try (KNNDirectIODirectory storage = storageOver(delegate, true)) {
                for (final String name : new String[] { "_0.cfs", "_0.cfe" }) {
                    try (IndexInput input = storage.openInput(name, IOContext.DEFAULT)) {
                        assertSame(name + " must be the delegate's own object, not a wrapper of it", delegateInputClass, input.getClass());
                    }
                }
                try (IndexInput container = storage.openInput("_0.cfs", IOContext.DEFAULT)) {
                    assertArrayEquals(contentOf(FILE_BYTES), readAll(container));
                }
                assertEquals("a .cfs satisfies no condition of the rule, so nothing is counted", 0, storage.routedOpens());
                assertEquals(0, storage.declinedOpens());
            }
        }
    }

    // -------------------------------------------------------------------------------------------------
    // What must not be there, and what must still work
    // -------------------------------------------------------------------------------------------------

    /**
     * Joint 6 of the design, as an assertion rather than a note: the production directory keeps no
     * node-wide mutable state. Its spike predecessor has a static {@code AtomicBoolean INSTALLED_ON_NODE}
     * so a test could tell "never installed" from "installed but unreachable"; on a node that static
     * outlives every index that set it, and a later index's behaviour would depend on an earlier one's.
     * This is a structural test because the failure it guards is a future edit, not current behaviour.
     *
     * <p>A static {@link java.util.Collection} is admitted only if the <em>instance</em> rejects mutation,
     * which is checked rather than assumed, because a mutable collection would pass a check on the declared
     * type alone and is exactly the state joint 6 forbids. The class holds no such field, so the branch guards an
     * addition rather than the present.
     */
    @SneakyThrows
    public void testTheDirectoryKeepsNoNodeWideState() {
        for (final Field field : KNNDirectIODirectory.class.getDeclaredFields()) {
            if (Modifier.isStatic(field.getModifiers()) == false) {
                continue;
            }
            assertTrue("static field " + field.getName() + " must be final", Modifier.isFinal(field.getModifiers()));
            final Class<?> type = field.getType();
            // A logger carries nothing that varies with an index or a node's history, and it is generated
            // rather than written; everything else static would.
            boolean stateless = type.isPrimitive() || type == String.class || org.apache.logging.log4j.Logger.class.isAssignableFrom(type);
            if (stateless == false && java.util.Collection.class.isAssignableFrom(type)) {
                field.setAccessible(true);
                final java.util.Collection<?> value = (java.util.Collection<?>) field.get(null);
                expectThrows(UnsupportedOperationException.class, () -> ((java.util.Collection) value).clear());
                stateless = true;
            }
            assertTrue(
                "static field " + field.getName() + " is of mutable type " + type.getName() + "; node-wide mutable state is joint 6",
                stateless
            );
        }
        // And the shape it must not have, named: the spike's flag, which a later edit might copy across.
        for (final Field field : KNNDirectIODirectory.class.getDeclaredFields()) {
            assertFalse(
                "a node-wide installed flag is exactly what joint 6 forbids",
                field.getName().toLowerCase(java.util.Locale.ROOT).contains("installed")
            );
        }
    }

    /** Every other method is {@link FilterDirectory}'s pass-through, so an ordinary round trip still works. */
    @SneakyThrows
    public void testTheDirectoryIsOtherwiseTransparent() {
        final Path path = createTempDir();
        try (MMapDirectory delegate = new MMapDirectory(path)) {
            try (KNNDirectIODirectory storage = storageOver(delegate, true)) {
                final byte[] expected = writeFile(storage, "written-through.bin", 2048);
                assertEquals(2048, storage.fileLength("written-through.bin"));
                assertTrue(java.util.Arrays.asList(storage.listAll()).contains("written-through.bin"));
                try (IndexInput input = storage.openInput("written-through.bin", IOContext.DEFAULT)) {
                    assertArrayEquals(expected, readAll(input));
                }
                storage.deleteFile("written-through.bin");
                assertFalse(java.util.Arrays.asList(storage.listAll()).contains("written-through.bin"));
            }
        }
    }

    // -------------------------------------------------------------------------------------------------
    // Conditions 4 and 8: the delegate chain, and the bytes it serves
    // -------------------------------------------------------------------------------------------------

    /**
     * A {@link FilterDirectory} that transforms the bytes it serves. It stands in for the real thing this
     * guard exists for — an encryption-at-rest wrapper — in the only way that matters to the test: the
     * file on disk is not what a reader of this directory sees. Nothing but {@code openInput} needs to be
     * faithful, because nothing but {@code openInput} is what the guard decides.
     *
     * <p>{@code openChecksumInput} is <b>not</b> transformed: {@link FilterDirectory} delegates it straight
     * to the layer below, so a test that reaches {@code checkIntegrity()} through this fixture would read the
     * file's own bytes and pass for the wrong reason. No test here reaches it.
     */
    private static final class XorDirectory extends FilterDirectory {
        private static final byte KEY = (byte) 0x5a;

        XorDirectory(final Directory delegate) {
            super(delegate);
        }

        /** Writes the on-disk form of {@code plain}, which is what a transforming wrapper stores. */
        @SneakyThrows
        static byte[] writeTransformed(final Directory bottom, final String name, final byte[] plain) {
            final byte[] onDisk = xor(plain);
            try (IndexOutput output = bottom.createOutput(name, IOContext.DEFAULT)) {
                output.writeBytes(onDisk, onDisk.length);
            }
            return onDisk;
        }

        static byte[] xor(final byte[] bytes) {
            final byte[] out = new byte[bytes.length];
            for (int i = 0; i < bytes.length; i++) {
                out[i] = (byte) (bytes[i] ^ KEY);
            }
            return out;
        }

        @Override
        public IndexInput openInput(final String name, final IOContext context) throws IOException {
            final byte[] plain;
            try (IndexInput delegate = super.openInput(name, context)) {
                plain = new byte[Math.toIntExact(delegate.length())];
                delegate.readBytes(plain, 0, plain.length);
            }
            return new ByteArrayIndexInput(name, xor(plain));
        }
    }

    /**
     * The guard's reason for existing, and now the guard itself. A byte-transforming
     * {@link FilterDirectory} between the codec and the {@link org.apache.lucene.store.FSDirectory} must
     * <b>decline</b>: the path resolution walks past every layer with {@link FilterDirectory#unwrap}, so
     * without condition 8 this open would serve the raw on-disk file's ciphertext as vectors. Silently, and
     * only for the one file this routes, while every other read in the system stayed correct.
     *
     * <p>Three assertions, and the last is the one that makes the first two worth having: the open declines,
     * the reason is {@code CONTENT} rather than the chain, and the bytes that come back through the delegate
     * are the plaintext. Declining costs read-ahead; bypassing costs correctness.
     */
    @SneakyThrows
    public void testAByteTransformingFilterDirectoryDeclinesWithContent() {
        assumeDirectIOWorksHere();
        final Path path = createTempDir();
        try (MMapDirectory bottom = new MMapDirectory(path)) {
            try (XorDirectory transforming = new XorDirectory(bottom)) {
                final byte[] expected = contentOf(FILE_BYTES);
                final byte[] onDisk = XorDirectory.writeTransformed(bottom, VECTOR_DATA, expected);
                // The premise: the bytes on disk are not the bytes a reader of this directory sees.
                assertFalse("this stand-in must actually transform, or the test proves nothing", java.util.Arrays.equals(expected, onDisk));
                try (IndexInput through = transforming.openInput(VECTOR_DATA, IOContext.DEFAULT)) {
                    assertArrayEquals("and the wrapper is what undoes it", expected, readAll(through));
                }
                try (KNNDirectIODirectory storage = storageOver(transforming, true)) {
                    try (IndexInput input = storage.openInput(VECTOR_DATA, vectorDataContext())) {
                        assertFalse(
                            "a transforming layer must never be bypassed to reach the file under it",
                            input instanceof DirectIOVectorIndexInput
                        );
                        assertArrayEquals("declining must leave the transforming layer doing its job", expected, readAll(input));
                    }
                    assertEquals(0, storage.routedOpens());
                    assertEquals("the reason must be the content, not the chain", 1, storage.declinedContent());
                    assertEquals("the chain resolves: unwrap reaches the MMapDirectory under it", 0, storage.declinedChain());
                    assertEquals(0, storage.declinedTooSmall());
                    assertEquals(0, storage.declinedMissing());
                    assertEquals(0, storage.declinedOpenFailed());
                    assertEquals(1, storage.declinedOpens());
                }
            }
        }
    }

    /**
     * An {@link FSDirectory} <b>subclass</b> that transforms bytes on read: the shape of
     * {@code CryptoNIOFSDirectory} in {@code AWSOpenSearchStorageEncryptionPlugin}, which extends
     * {@code NIOFSDirectory} rather than wrapping a {@link Directory}.
     *
     * <p>This is the case an allowlist of {@link FilterDirectory} class names could not catch, and the
     * reason W2 replaced it with a byte comparison: the layer <em>is</em> the terminal {@link FSDirectory}
     * of the chain, so no walk ever asks whether it may be crossed, and {@code getDirectory().resolve(name)}
     * is the raw ciphertext file.
     */
    private static final class CryptoShapedDirectory extends NIOFSDirectory {
        CryptoShapedDirectory(final Path path) throws IOException {
            super(path);
        }

        @Override
        public IndexInput openInput(final String name, final IOContext context) throws IOException {
            final byte[] onDisk;
            try (IndexInput delegate = super.openInput(name, context)) {
                onDisk = new byte[Math.toIntExact(delegate.length())];
                delegate.readBytes(onDisk, 0, onDisk.length);
            }
            return new ByteArrayIndexInput(name, XorDirectory.xor(onDisk));
        }
    }

    /** The test the allowlist could not pass: a transforming {@link FSDirectory} subclass declines. */
    @SneakyThrows
    public void testACryptoShapedNioFsSubclassDeclinesWithContent() {
        assumeDirectIOWorksHere();
        final Path path = createTempDir();
        final byte[] expected = contentOf(FILE_BYTES);
        Files.write(path.resolve(VECTOR_DATA), XorDirectory.xor(expected));
        try (CryptoShapedDirectory crypto = new CryptoShapedDirectory(path)) {
            // The two premises: it is the terminal case of the chain, and it transforms.
            assertTrue("the premise: this layer is itself an FSDirectory", crypto instanceof org.apache.lucene.store.FSDirectory);
            assertEquals(
                "so the chain resolves to the raw file, with no layer to stop at",
                path.resolve(VECTOR_DATA),
                KNNDirectIODirectory.resolvePath(crypto, VECTOR_DATA)
            );
            try (IndexInput through = crypto.openInput(VECTOR_DATA, IOContext.DEFAULT)) {
                assertArrayEquals("and a reader of it sees the plaintext", expected, readAll(through));
            }
            try (KNNDirectIODirectory storage = storageOver(crypto, true)) {
                try (IndexInput input = storage.openInput(VECTOR_DATA, vectorDataContext())) {
                    assertFalse("the bytes on disk are not the bytes this directory serves", input instanceof DirectIOVectorIndexInput);
                    assertArrayEquals(expected, readAll(input));
                }
                assertEquals(1, storage.declinedContent());
                assertEquals(0, storage.routedOpens());
                assertEquals("and no counter but the content one moves", 1, storage.declinedOpens());
            }
        }
    }

    /** A {@link FilterDirectory} that adds nothing: the shape of every OpenSearch layer that only wraps. */
    private static final class PassthroughDirectory extends FilterDirectory {
        PassthroughDirectory(final Directory delegate) {
            super(delegate);
        }
    }

    /**
     * The opposite answer to the one a class-name rule gives: a
     * {@link FilterDirectory} nobody has ever named routes, because it passes the bytes through and
     * condition 8 is what says so. Under the allowlist this open declined, and a new layer in a production
     * chain cost the feature until someone read the class and added its name.
     */
    @SneakyThrows
    public void testAnUnknownPassthroughFilterDirectoryRoutes() {
        assumeDirectIOWorksHere();
        final Path path = createTempDir();
        try (MMapDirectory bottom = new MMapDirectory(path)) {
            final byte[] expected = writeFile(bottom, VECTOR_DATA, FILE_BYTES);
            try (PassthroughDirectory unknown = new PassthroughDirectory(bottom)) {
                try (KNNDirectIODirectory storage = storageOver(unknown, true)) {
                    try (IndexInput input = storage.openInput(VECTOR_DATA, vectorDataContext())) {
                        assertTrue(
                            "a layer that passes bytes through must route, got " + input.getClass().getName(),
                            input instanceof DirectIOVectorIndexInput
                        );
                        assertArrayEquals(expected, readAll(input));
                    }
                    assertEquals(1, storage.routedOpens());
                    assertEquals(0, storage.declinedOpens());
                }
            }
        }
    }

    /**
     * OpenSearch 3.9's real chain, read from a node's own decline log:
     * {@code Store$StoreDirectory -> ByteSizeCachingDirectory -> BucketedCompositeDirectory -> HybridDirectory}.
     * Three of the four are package private in OpenSearch and cannot be constructed from here, so the test
     * stands them in with passthrough layers of the same shape over the real {@code hybridfs} bottom. What
     * it pins is that a three-deep chain over an {@link FSDirectory} subclass still routes, which is the
     * production case the feature exists for.
     */
    @SneakyThrows
    public void testTheMeasuredThreeNineChainShapeRoutes() {
        assumeDirectIOWorksHere();
        final IndexSettings indexSettings = IndexSettingsModule.newIndexSettings(
            "dio",
            Settings.builder().put(IndexModule.INDEX_STORE_TYPE_SETTING.getKey(), "hybridfs").build()
        );
        final Path shardDir = createTempDir().resolve(indexSettings.getUUID()).resolve("0");
        Files.createDirectories(shardDir);
        final ShardPath shardPath = new ShardPath(false, shardDir, shardDir, new ShardId(indexSettings.getIndex(), 0));
        try (Directory hybrid = new FsDirectoryFactory().newDirectory(indexSettings, shardPath)) {
            final byte[] expected = writeFile(hybrid, VECTOR_DATA, FILE_BYTES);
            final Directory chain = new PassthroughDirectory(new PassthroughDirectory(new PassthroughDirectory(hybrid)));
            try (KNNDirectIODirectory storage = storageOver(chain, true)) {
                try (IndexInput input = storage.openInput(VECTOR_DATA, vectorDataContext())) {
                    assertTrue(
                        "the measured 3.9 chain shape must route, got " + input.getClass().getName(),
                        input instanceof DirectIOVectorIndexInput
                    );
                    assertArrayEquals(expected, readAll(input));
                }
                assertEquals(1, storage.routedOpens());
                assertEquals(0, storage.declinedOpens());
            }
        }
    }

    /**
     * Condition 4 as a unit: {@link FilterDirectory#unwrap} down to an {@link FSDirectory}, whatever the
     * layers are. There is no allowlist to parameterise any more, so what is left to pin is the terminal
     * case, the layers crossed and the two answers that are {@code null}.
     */
    @SneakyThrows
    public void testResolvePathUnwrapsToTheFsDirectory() {
        final Path path = createTempDir();
        try (MMapDirectory bottom = new MMapDirectory(path)) {
            assertEquals(
                "one layer is crossed",
                path.resolve(VECTOR_DATA),
                KNNDirectIODirectory.resolvePath(new PassthroughDirectory(bottom), VECTOR_DATA)
            );
            assertEquals(
                "and so are three, whatever they do: the bytes are checked later, not the class names",
                path.resolve(VECTOR_DATA),
                KNNDirectIODirectory.resolvePath(new XorDirectory(new PassthroughDirectory(new XorDirectory(bottom))), VECTOR_DATA)
            );
            assertEquals("the bottom itself answers too", path.resolve(VECTOR_DATA), KNNDirectIODirectory.resolvePath(bottom, VECTOR_DATA));
        }
        try (ByteBuffersDirectory noPath = new ByteBuffersDirectory()) {
            assertNull(
                "a chain with no FSDirectory at the bottom has no path",
                KNNDirectIODirectory.resolvePath(new PassthroughDirectory(noPath), VECTOR_DATA)
            );
        }
        assertNull("and neither has nothing at all", KNNDirectIODirectory.resolvePath(null, VECTOR_DATA));
    }

    /**
     * The chain description that goes into the decline log, which is the only thing that makes a chain
     * decline diagnosable: the counter can only ever say "nothing under this chain has a path", and the useful
     * question is which. Outermost first, stopping at the first non-{@link FilterDirectory}.
     *
     * <p>This is not cosmetic. The first S5 re-proof run routed nothing and declined 10 times on the chain;
     * the log line as originally written named only the <em>top</em> of the chain, which was a layer that passes
     * bytes through, so it pointed away from the cause. With the full chain the answer was immediate.
     */
    @SneakyThrows
    public void testTheDeclineLogDescribesTheWholeChainOutermostFirst() {
        try (ByteBuffersDirectory bottom = new ByteBuffersDirectory()) {
            final Directory chain = new XorDirectory(new XorDirectory(bottom));
            assertEquals(
                java.util.List.of(XorDirectory.class.getName(), XorDirectory.class.getName(), ByteBuffersDirectory.class.getName()),
                KNNDirectIODirectory.describeChain(chain)
            );
            assertEquals(java.util.List.of(ByteBuffersDirectory.class.getName()), KNNDirectIODirectory.describeChain(bottom));
        }
    }

    /**
     * OpenSearch's real {@code hybridfs} directory, built by the real {@code FsDirectoryFactory}: the
     * bottom of every production chain, and an {@code FSDirectory} <em>subclass</em> that overrides
     * {@code openInput} to pick mmap or nio per file. Both of its routes read
     * {@code getDirectory().resolve(name)}, so the terminal case of the resolution has to accept a subclass
     * rather than only the two concrete Lucene directories — which is what this pins.
     */
    @SneakyThrows
    public void testTheHybridfsDirectoryEngages() {
        assumeDirectIOWorksHere();
        final IndexSettings indexSettings = IndexSettingsModule.newIndexSettings(
            "dio",
            Settings.builder().put(IndexModule.INDEX_STORE_TYPE_SETTING.getKey(), "hybridfs").build()
        );
        final Path shardDir = createTempDir().resolve(indexSettings.getUUID()).resolve("0");
        Files.createDirectories(shardDir);
        final ShardPath shardPath = new ShardPath(false, shardDir, shardDir, new ShardId(indexSettings.getIndex(), 0));
        try (Directory hybrid = new FsDirectoryFactory().newDirectory(indexSettings, shardPath)) {
            assertTrue("the premise: hybridfs is an FSDirectory subclass", hybrid instanceof org.apache.lucene.store.FSDirectory);
            final byte[] expected = writeFile(hybrid, VECTOR_DATA, FILE_BYTES);
            try (KNNDirectIODirectory storage = storageOver(hybrid, true)) {
                try (IndexInput routed = storage.openInput(VECTOR_DATA, vectorDataContext())) {
                    assertTrue(
                        "hybridfs is the production bottom and must engage, got " + routed.getClass().getName(),
                        routed instanceof DirectIOVectorIndexInput
                    );
                    assertArrayEquals(expected, readAll(routed));
                }
                assertEquals(1, storage.routedOpens());
                assertEquals(0, storage.declinedOpens());
            }
        }
    }

    // -------------------------------------------------------------------------------------------------
    // Condition 6: the size floor (phase 13 audit FIX 1)
    // -------------------------------------------------------------------------------------------------

    /**
     * The size floor, and the measurement it was added for. Phase 13's E4 ran the same 16-thread random
     * rescore workload over a 0.98 MiB flushed {@code .vec} and measured <b>64.7 ops/s</b> on
     * {@code O_DIRECT} against <b>518,681 ops/s</b> on mmap — 8,016x — because mmap reads a small file from
     * the device once and answers from the page cache forever while {@code O_DIRECT} goes to the device for
     * every read. Below the floor the file is the delegate's, at or above it it is routed, and the boundary
     * is inclusive.
     */
    @SneakyThrows
    public void testAFileBelowTheSizeFloorIsNotRouted() {
        assumeDirectIOWorksHere();
        final Path path = createTempDir();
        try (MMapDirectory delegate = new MMapDirectory(path)) {
            final byte[] expected = writeFile(delegate, VECTOR_DATA, FILE_BYTES);
            // One directory, floor lowered between the two opens, because the setting is dynamic and is
            // read at every dispatch: a floor captured at construction would need a reopen to take effect.
            final long[] floor = { FILE_BYTES + 1 };
            try (KNNDirectIODirectory storage = new KNNDirectIODirectory(delegate, "_0", () -> true, () -> floor[0])) {
                try (IndexInput input = storage.openInput(VECTOR_DATA, vectorDataContext())) {
                    assertFalse("below the floor the page cache serves it", input instanceof DirectIOVectorIndexInput);
                    assertArrayEquals("declining on size must not change the bytes", expected, readAll(input));
                }
                assertEquals(0, storage.routedOpens());
                assertEquals("the reason must be the size, not the chain", 1, storage.declinedTooSmall());
                assertEquals(0, storage.declinedChain());
                assertEquals(1, storage.declinedOpens());

                // The boundary is inclusive: exactly at the floor routes.
                floor[0] = FILE_BYTES;
                try (IndexInput routed = storage.openInput(VECTOR_DATA, vectorDataContext())) {
                    assertTrue("a file exactly at the floor is routed", routed instanceof DirectIOVectorIndexInput);
                }
                assertEquals(1, storage.routedOpens());
                assertEquals("and the earlier decline is not re-counted", 1, storage.declinedTooSmall());
            }
        }
    }

    /** The shipped default, so the floor's value is pinned somewhere rather than only commented. */
    public void testTheShippedSizeFloorIsTenMegabytes() {
        assertEquals(10L * 1024 * 1024, KNNSettings.KNN_DIRECT_IO_MIN_FILE_SIZE_DEFAULT_VALUE.getBytes());
        assertEquals("knn.direct_io.min_file_size", KNNSettings.KNN_DIRECT_IO_MIN_FILE_SIZE_SETTING.getKey());
        assertTrue("the node bench has to sweep it without a rebuild", KNNSettings.KNN_DIRECT_IO_MIN_FILE_SIZE_SETTING.isDynamic());
    }

    /**
     * Condition 4, which is reachable in production: a {@code .vec} named in a segment that has since been
     * merged away, or a path the shard no longer owns. It must be an ordinary read and its own reason,
     * because unlike a size decline it means something is wrong.
     */
    @SneakyThrows
    public void testAMissingFileDeclinesUnderItsOwnReason() {
        final Path path = createTempDir();
        try (MMapDirectory delegate = new MMapDirectory(path)) {
            writeFile(delegate, VECTOR_DATA, FILE_BYTES);
            Files.delete(path.resolve(VECTOR_DATA));
            try (KNNDirectIODirectory storage = storageOver(delegate, true)) {
                expectThrows(IOException.class, () -> storage.openInput(VECTOR_DATA, vectorDataContext()));
                assertEquals("a missing file is not a size decline or a chain decline", 1, storage.declinedMissing());
                assertEquals(0, storage.declinedTooSmall());
                assertEquals(0, storage.declinedChain());
                assertEquals(0, storage.routedOpens());
            }
        }
    }

    // -------------------------------------------------------------------------------------------------
    // S4: the counters an operator actually reads are node wide, not per segment
    // -------------------------------------------------------------------------------------------------

    /**
     * Every dispatch outcome is counted <b>twice</b>: on this directory's own per-segment counter, and on
     * {@link DirectIOVectorIndexInput#STATS}, which is node wide and is what {@code _plugins/_knn/stats}
     * reports. Both are load bearing. The codec seam creates this directory per segment per reader and
     * discards it with that reader, so its instance counters can <em>never</em> be reached over REST — which
     * phase 13's re-proof notes recorded as a gap and this is the fix. The instance counters stay because
     * they are per segment, which a node-wide sum cannot be.
     *
     * <p>Driven through a real open of each kind, so the wiring is what is tested rather than restated.
     */
    @SneakyThrows
    public void testADeclineIsCountedNodeWideAsWellAsPerSegment() {
        assumeDirectIOWorksHere();
        DirectIOVectorIndexInput.STATS.reset();
        try {
            // A chain decline: no FSDirectory underneath.
            try (ByteBuffersDirectory noPath = new ByteBuffersDirectory()) {
                writeFile(noPath, VECTOR_DATA, FILE_BYTES);
                try (KNNDirectIODirectory storage = storageOver(noPath, true)) {
                    storage.openInput(VECTOR_DATA, vectorDataContext()).close();
                    assertEquals("the per-segment counter", 1, storage.declinedChain());
                }
            }
            assertEquals("and the node-wide one", 1L, DirectIOVectorIndexInput.STATS.declinedChain());

            // A size decline, then a routed open, on a real filesystem. One directory, because closing a
            // KNNDirectIODirectory closes its delegate -- which is exactly why routeVectorData never
            // closes the wrapper it builds.
            final Path path = createTempDir();
            try (MMapDirectory delegate = new MMapDirectory(path)) {
                writeFile(delegate, VECTOR_DATA, FILE_BYTES);
                final long[] floor = { FILE_BYTES + 1 };
                try (KNNDirectIODirectory storage = new KNNDirectIODirectory(delegate, "_0", () -> true, () -> floor[0])) {
                    storage.openInput(VECTOR_DATA, vectorDataContext()).close();
                    assertEquals(1L, DirectIOVectorIndexInput.STATS.declinedTooSmall());

                    floor[0] = 0L;
                    try (IndexInput routed = storage.openInput(VECTOR_DATA, vectorDataContext())) {
                        assertTrue(routed instanceof DirectIOVectorIndexInput);
                    }
                }
                assertEquals(1L, DirectIOVectorIndexInput.STATS.routedOpens());
            }

            // The reasons are kept apart node wide, exactly as they are per segment.
            assertEquals(2L, DirectIOVectorIndexInput.STATS.declinedOpens());
            assertEquals(0L, DirectIOVectorIndexInput.STATS.declinedMissing());
            assertEquals(0L, DirectIOVectorIndexInput.STATS.declinedOpenFailed());
            assertEquals("no compound directory is involved in this test", 0L, DirectIOVectorIndexInput.STATS.declinedCompound());
            assertTrue(
                "and they are in the stats line an operator sees at DEBUG: " + DirectIOVectorIndexInput.STATS,
                DirectIOVectorIndexInput.STATS.toString()
                    .contains("routedOpens=1 declinedCompound=0 declinedChain=1 declinedMissing=0 declinedTooSmall=1")
            );
        } finally {
            DirectIOVectorIndexInput.STATS.reset();
        }
    }

    /**
     * The predicate is name-based, so it is worth pinning to the names the real formats actually produce:
     * {@code PerFieldKnnVectorsFormat} builds each field's segment suffix from
     * {@code KnnVectorsFormat#getName()}, so a drift in either format's name would silently stop routing.
     */
    public void testThePredicateMatchesTheNamesTheRealFormatsProduce() {
        assertTrue(KNNDirectIODirectory.isFaissVectorData("_7_" + new NativeEngines990KnnVectorsFormat(0).getName() + "_0.vec"));
        assertTrue(KNNDirectIODirectory.isFaissVectorData("_7_" + new Faiss1040ScalarQuantizedKnnVectorsFormat().getName() + "_0.vec"));
        assertFalse(KNNDirectIODirectory.isFaissVectorData("_7_Lucene99HnswVectorsFormat_0.vec"));
        assertFalse(KNNDirectIODirectory.isFaissVectorData("_7_" + NativeEngines990KnnVectorsFormat.FORMAT_NAME + "_0.veq"));
        assertFalse(KNNDirectIODirectory.isFaissVectorData(null));
    }

    /**
     * O8. The two FP16 per-field formats declare their {@code FORMAT_NAME} private, so the rule carries the
     * strings; this is what fails if either format is renamed. Before O8 the inserted {@code HalfFloat} broke
     * the fp32 substrings and a half-float {@code .vec} was declined at condition 1, before any counter.
     */
    public void testHalfFloatFormatNamesMatchTheFormats() {
        assertEquals(KNNDirectIODirectory.HALF_FLOAT_FORMAT_NAME, new NativeEngines990HalfFloatKnnVectorsFormat().getName());
        assertEquals(KNNDirectIODirectory.HALF_FLOAT_SQ_FORMAT_NAME, new Faiss1040HalfFloatScalarQuantizedKnnVectorsFormat().getName());
        // And the reason the names had to be listed at all: neither contains the fp32 name it extends.
        assertFalse(KNNDirectIODirectory.HALF_FLOAT_FORMAT_NAME.contains(NativeEngines990KnnVectorsFormat.FORMAT_NAME));
        assertFalse(KNNDirectIODirectory.HALF_FLOAT_SQ_FORMAT_NAME.contains(Faiss1040ScalarQuantizedKnnVectorsFormat.FORMAT_NAME));
    }

    /** O8. A half-float {@code .vec}, of either FP16 format, is routed under exactly the same rule. */
    public void testTheHalfFloatVectorDataIsRouted() {
        assertTrue(KNNDirectIODirectory.isFaissVectorData("_7_" + new NativeEngines990HalfFloatKnnVectorsFormat().getName() + "_0.vec"));
        assertTrue(
            KNNDirectIODirectory.isFaissVectorData("_7_" + new Faiss1040HalfFloatScalarQuantizedKnnVectorsFormat().getName() + "_0.vec")
        );
        // Still only the .vec: the half-float formats' other files are no more routed than the fp32 ones'.
        assertFalse(KNNDirectIODirectory.isFaissVectorData("_7_" + KNNDirectIODirectory.HALF_FLOAT_FORMAT_NAME + "_0.vemf"));
        assertFalse(KNNDirectIODirectory.isFaissVectorData("_7_" + KNNDirectIODirectory.HALF_FLOAT_SQ_FORMAT_NAME + "_0.veq"));
    }

    // -------------------------------------------------------------------------------------------------
    // Condition 8, one failure of it at a time: what the byte check catches, and what it cannot
    // -------------------------------------------------------------------------------------------------

    /**
     * A {@link FilterDirectory} whose {@code .vec} input serves a transform of the file's bytes, so that one
     * test can break one part of the comparison at a time. Every other file is the delegate's own, and so is
     * {@code openChecksumInput} of this one: see {@link XorDirectory}.
     */
    private static final class TransformingDirectory extends FilterDirectory {
        private final UnaryOperator<byte[]> transform;

        TransformingDirectory(final Directory delegate, final UnaryOperator<byte[]> transform) {
            super(delegate);
            this.transform = transform;
        }

        @Override
        public IndexInput openInput(final String name, final IOContext context) throws IOException {
            if (name.endsWith(".vec") == false) {
                return super.openInput(name, context);
            }
            final byte[] onDisk;
            try (IndexInput delegate = super.openInput(name, context)) {
                onDisk = new byte[Math.toIntExact(delegate.length())];
                delegate.readBytes(onDisk, 0, onDisk.length);
            }
            return new ByteArrayIndexInput(name, transform.apply(onDisk));
        }
    }

    /**
     * Opens the {@code .vec} through {@code transform} and returns the directory, counters and all. Neither
     * the transforming layer nor the directory is closed: closing either would close the caller's own
     * {@code bottom} directory with it, which the caller is still holding.
     */
    @SneakyThrows
    private KNNDirectIODirectory openThrough(final Directory bottom, final UnaryOperator<byte[]> transform) {
        final TransformingDirectory transforming = new TransformingDirectory(bottom, transform);
        final KNNDirectIODirectory storage = new KNNDirectIODirectory(transforming, "_0", () -> true, () -> 0L);
        storage.openInput(VECTOR_DATA, vectorDataContext()).close();
        return storage;
    }

    /**
     * The length comparison, which is the cheapest of the three and the one a relocating layer fails: a
     * chain that serves a different number of bytes than the file holds is not serving the file.
     */
    @SneakyThrows
    public void testALengthMismatchDeclinesWithContent() {
        assumeDirectIOWorksHere();
        final Path path = createTempDir();
        try (MMapDirectory bottom = new MMapDirectory(path)) {
            writeFile(bottom, VECTOR_DATA, FILE_BYTES);
            final KNNDirectIODirectory storage = openThrough(bottom, bytes -> java.util.Arrays.copyOf(bytes, bytes.length - 8));
            assertEquals(1, storage.declinedContent());
            assertEquals(0, storage.routedOpens());
        }
    }

    /**
     * The footer half of the comparison, on its own: a layer that passes the first 4 KiB through and
     * transforms only the last 16 bytes. Lucene's codec footer is a checksum of everything before it, so a
     * transform of the body shows here even when the header does not move — which is why the check reads the
     * tail at all, and this is the test that pins it.
     */
    @SneakyThrows
    public void testAFooterOnlyTransformDeclinesWithContent() {
        assumeDirectIOWorksHere();
        final Path path = createTempDir();
        try (MMapDirectory bottom = new MMapDirectory(path)) {
            writeFile(bottom, VECTOR_DATA, FILE_BYTES);
            final KNNDirectIODirectory storage = openThrough(bottom, bytes -> {
                final byte[] out = bytes.clone();
                for (int i = out.length - DirectIOVectorIndexInput.CONTENT_CHECK_FOOTER_BYTES; i < out.length; i++) {
                    out[i] ^= (byte) 0x5a;
                }
                return out;
            });
            assertEquals("the footer is compared, not only the head", 1, storage.declinedContent());
            assertEquals(0, storage.routedOpens());
        }
    }

    /**
     * The documented limit, asserted so that the javadoc cannot drift away from the behaviour: a layer that
     * transforms <b>only interior bytes</b> — neither the head nor the footer — is routed, and would serve
     * transformed vectors. Sampling an interior block would narrow this without closing it, since a
     * transform may be the identity on any block sampled, so the opt-in guide carries it as a limitation line
     * instead.
     */
    @SneakyThrows
    public void testAnInteriorOnlyTransformIsNotCaught() {
        assumeDirectIOWorksHere();
        final Path path = createTempDir();
        try (MMapDirectory bottom = new MMapDirectory(path)) {
            writeFile(bottom, VECTOR_DATA, FILE_BYTES);
            final int interior = DirectIOVectorIndexInput.CONTENT_CHECK_HEAD_BYTES * 2;
            assertTrue("the transformed bytes must be outside both compared regions", interior < FILE_BYTES - 1024);
            final KNNDirectIODirectory storage = openThrough(bottom, bytes -> {
                final byte[] out = bytes.clone();
                out[interior] ^= (byte) 0x5a;
                return out;
            });
            assertEquals("an interior-only transform is the limit, not a decline", 0, storage.declinedContent());
            assertEquals(1, storage.routedOpens());
        }
    }

    /** An input whose length is honest and whose reads are not: the chain cannot be compared at all. */
    private static final class UnreadableIndexInput extends IndexInput {
        private final long length;

        UnreadableIndexInput(final String name, final long length) {
            super(name);
            this.length = length;
        }

        @Override
        public void close() {}

        @Override
        public long getFilePointer() {
            return 0L;
        }

        @Override
        public void seek(final long pos) {}

        @Override
        public long length() {
            return length;
        }

        @Override
        public IndexInput slice(final String sliceDescription, final long offset, final long length) throws IOException {
            throw new IOException("this input cannot be sliced either");
        }

        @Override
        public byte readByte() throws IOException {
            throw new IOException("this input cannot be read");
        }

        @Override
        public void readBytes(final byte[] b, final int offset, final int len) throws IOException {
            throw new IOException("this input cannot be read");
        }
    }

    /**
     * Fail closed on the comparison itself. A chain whose bytes cannot be read is a chain whose bytes have
     * not been shown to be the file's, so the open declines rather than routing unverified — the same rule
     * condition 7 applies to the delegate open that fails outright.
     */
    @SneakyThrows
    public void testADelegateThatCannotBeReadDeclinesWithContent() {
        assumeDirectIOWorksHere();
        final Path path = createTempDir();
        try (MMapDirectory bottom = new MMapDirectory(path)) {
            writeFile(bottom, VECTOR_DATA, FILE_BYTES);
            final Directory unreadable = new FilterDirectory(bottom) {
                @Override
                public IndexInput openInput(final String name, final IOContext context) throws IOException {
                    if (name.endsWith(".vec")) {
                        return new UnreadableIndexInput(name, FILE_BYTES);
                    }
                    return super.openInput(name, context);
                }
            };
            try (KNNDirectIODirectory storage = new KNNDirectIODirectory(unreadable, "_0", () -> true, () -> 0L)) {
                storage.openInput(VECTOR_DATA, vectorDataContext()).close();
                assertEquals("a comparison that could not be made has not passed", 1, storage.declinedContent());
                assertEquals(0, storage.routedOpens());
            }
        }
    }

    /**
     * Condition 7's second half: a delegate input that cannot be opened at all declines, where an earlier shape
     * of the rule routed on with no fallback. Without the delegate's input there is nothing to compare the file's
     * bytes with, so routing would be routing unverified.
     */
    @SneakyThrows
    public void testADelegateOpenFailureDeclinesWithOpenFailed() {
        assumeDirectIOWorksHere();
        final Path path = createTempDir();
        try (MMapDirectory bottom = new MMapDirectory(path)) {
            writeFile(bottom, VECTOR_DATA, FILE_BYTES);
            final Directory failing = new FilterDirectory(bottom) {
                @Override
                public IndexInput openInput(final String name, final IOContext context) throws IOException {
                    if (name.endsWith(".vec")) {
                        throw new IOException("no input for " + name);
                    }
                    return super.openInput(name, context);
                }
            };
            try (KNNDirectIODirectory storage = new KNNDirectIODirectory(failing, "_0", () -> true, () -> 0L)) {
                // The decline is counted, and then the delegate's own failure is what the caller sees: the
                // rule never turns a failing open into a routed one.
                expectThrows(IOException.class, () -> storage.openInput(VECTOR_DATA, vectorDataContext()));
                assertEquals(1, storage.declinedOpenFailed());
                assertEquals(0, storage.routedOpens());
                assertEquals(0, storage.declinedContent());
            }
        }
    }

    /**
     * A file that lives only in an {@link NRTCachingDirectory}'s RAM buffer: the chain resolves to a path
     * on disk, because {@code unwrap} walks past the caching layer, but nothing is there. It declines
     * {@code MISSING}, reads correctly through the delegate, and is the shape every composing layer takes —
     * a name whose bytes are not (yet) the file's.
     */
    @SneakyThrows
    public void testAFileOnlyInAnNrtCachingRamBufferDeclinesWithMissing() {
        final Path path = createTempDir();
        try (MMapDirectory bottom = new MMapDirectory(path)) {
            try (NRTCachingDirectory caching = new NRTCachingDirectory(bottom, 5.0, 60.0)) {
                // A flush context, because that is the only write this directory caches: with no flush or
                // merge info it writes straight through, and then there would be nothing in RAM to test.
                final byte[] expected = contentOf(FILE_BYTES);
                try (IndexOutput output = caching.createOutput(VECTOR_DATA, IOContext.flush(new FlushInfo(1, FILE_BYTES)))) {
                    output.writeBytes(expected, expected.length);
                }
                // The premise: the name is in the listing but not on disk.
                assertTrue(java.util.Arrays.asList(caching.listAll()).contains(VECTOR_DATA));
                assertFalse("this fixture must keep the file in RAM, or the test proves nothing", Files.exists(path.resolve(VECTOR_DATA)));
                try (KNNDirectIODirectory storage = storageOver(caching, true)) {
                    try (IndexInput input = storage.openInput(VECTOR_DATA, vectorDataContext())) {
                        assertFalse(input instanceof DirectIOVectorIndexInput);
                        assertArrayEquals("the RAM buffer serves it", expected, readAll(input));
                    }
                    assertEquals("a name with no file is missing, not a content decline", 1, storage.declinedMissing());
                    assertEquals(0, storage.declinedContent());
                    assertEquals(0, storage.routedOpens());
                }
            }
        }
    }

    /** An input over a byte array that counts what is read through it, and whose clones share the count. */
    private static final class CountingIndexInput extends IndexInput {
        private final byte[] content;
        private final AtomicLong reads;
        private final AtomicLong bytes;
        private long position;

        CountingIndexInput(final String name, final byte[] content, final AtomicLong reads, final AtomicLong bytes) {
            super(name);
            this.content = content;
            this.reads = reads;
            this.bytes = bytes;
        }

        @Override
        public void close() {}

        @Override
        public long getFilePointer() {
            return position;
        }

        @Override
        public void seek(final long pos) {
            position = pos;
        }

        @Override
        public long length() {
            return content.length;
        }

        @Override
        public IndexInput slice(final String sliceDescription, final long offset, final long length) {
            throw new UnsupportedOperationException("the byte check never slices");
        }

        @Override
        public byte readByte() {
            reads.incrementAndGet();
            bytes.incrementAndGet();
            return content[Math.toIntExact(position++)];
        }

        @Override
        public void readBytes(final byte[] b, final int offset, final int len) {
            reads.incrementAndGet();
            bytes.addAndGet(len);
            System.arraycopy(content, Math.toIntExact(position), b, offset, len);
            position += len;
        }

        @Override
        public IndexInput clone() {
            final CountingIndexInput copy = new CountingIndexInput(toString(), content, reads, bytes);
            copy.position = position;
            return copy;
        }
    }

    /**
     * What condition 8 costs at open, which is the clause of W2 the performance guard reads: <b>two</b>
     * block-aligned device reads and one {@value DirectIOVectorIndexInput#CONTENT_CHECK_HEAD_BYTES} plus
     * {@value DirectIOVectorIndexInput#CONTENT_CHECK_FOOTER_BYTES} byte read through the delegate chain, per
     * routed open and never per query.
     *
     * <p>The assertion that matters to every earlier measurement is the second one: the check reads through
     * the file handle with a buffer of its own, <b>not</b> through this input's instrumented read path, so
     * {@code blocking_reads}, {@code blocking_bytes}, {@code served_bytes} and {@code staged_reads} do not
     * move and the re-proof's counters stay comparable across the change.
     *
     * <p>The delegate side is pinned exactly, and that is the substantive assertion. The device side is read
     * from {@code /proc/self/io} where the runtime allows it, and its bound is deliberately loose rather than
     * {@code 2 x blockSize}: that counter is JVM wide, so any other thread's page-in during the window lands
     * in it. A loose bound still distinguishes two aligned regions from reading the file, which is the claim.
     */
    @SneakyThrows
    public void testTheByteCheckCostsTwoReadsAndMovesNoReadCounter() {
        assumeDirectIOWorksHere();
        final Path path = createTempDir();
        // Larger than the two compared regions by three orders of magnitude, so "it does not read the file"
        // is a claim the device bytes can actually distinguish.
        final int fileBytes = 4 * 1024 * 1024;
        final byte[] content = contentOf(fileBytes);
        Files.write(path.resolve(VECTOR_DATA), content);
        final AtomicLong reads = new AtomicLong();
        final AtomicLong bytes = new AtomicLong();
        DirectIOVectorIndexInput.STATS.reset();
        try (MMapDirectory bottom = new MMapDirectory(path)) {
            final Directory counting = new FilterDirectory(bottom) {
                @Override
                public IndexInput openInput(final String name, final IOContext context) throws IOException {
                    if (name.endsWith(".vec")) {
                        return new CountingIndexInput(name, content, reads, bytes);
                    }
                    return super.openInput(name, context);
                }
            };
            final long deviceBefore = deviceReadBytes();
            try (KNNDirectIODirectory storage = new KNNDirectIODirectory(counting, "_0", () -> true, () -> 0L)) {
                try (IndexInput input = storage.openInput(VECTOR_DATA, vectorDataContext())) {
                    assertTrue(input instanceof DirectIOVectorIndexInput);
                    assertEquals(1, storage.routedOpens());
                }
            }
            final long deviceAfter = deviceReadBytes();

            assertEquals("the head and the footer, one read each", 2L, reads.get());
            assertEquals(
                "and exactly their two lengths through the chain",
                DirectIOVectorIndexInput.CONTENT_CHECK_HEAD_BYTES + DirectIOVectorIndexInput.CONTENT_CHECK_FOOTER_BYTES,
                bytes.get()
            );
            assertEquals("no instrumented blocking read", 0L, DirectIOVectorIndexInput.STATS.blockingReads());
            assertEquals(0L, DirectIOVectorIndexInput.STATS.blockingBytes());
            assertEquals("and no staged read either", 0L, DirectIOVectorIndexInput.STATS.stagedReads());
            assertEquals("so no byte is counted as served", 0L, DirectIOVectorIndexInput.STATS.servedBytes());
            assertEquals(1L, DirectIOVectorIndexInput.STATS.routedOpens());
            if (deviceBefore >= 0 && deviceAfter >= 0) {
                final long delta = deviceAfter - deviceBefore;
                logger.info("the byte check read {} device bytes at one routed open of a {} B file", delta, fileBytes);
                assertTrue("the check must read two aligned regions, not the file: " + delta + " B of " + fileBytes, delta <= 64L * 1024);
            }
        } finally {
            DirectIOVectorIndexInput.STATS.reset();
        }
    }

    /** An input that counts how often it was closed, and whose clones share the count. */
    private static final class CloseCountingIndexInput extends IndexInput {
        private final IndexInput delegate;
        private final AtomicLong closes;

        CloseCountingIndexInput(final IndexInput delegate, final AtomicLong closes) {
            super(delegate.toString());
            this.delegate = delegate;
            this.closes = closes;
        }

        @Override
        public void close() throws IOException {
            closes.incrementAndGet();
            delegate.close();
        }

        @Override
        public long getFilePointer() {
            return delegate.getFilePointer();
        }

        @Override
        public void seek(final long pos) throws IOException {
            delegate.seek(pos);
        }

        @Override
        public long length() {
            return delegate.length();
        }

        @Override
        public IndexInput slice(final String sliceDescription, final long offset, final long length) throws IOException {
            return delegate.slice(sliceDescription, offset, length);
        }

        @Override
        public byte readByte() throws IOException {
            return delegate.readByte();
        }

        @Override
        public void readBytes(final byte[] b, final int offset, final int len) throws IOException {
            delegate.readBytes(b, offset, len);
        }

        @Override
        public IndexInput clone() {
            // The clone shares the counter but not the close: a clone of an IndexInput is never closed.
            return new CloseCountingIndexInput(delegate.clone(), closes);
        }
    }

    /**
     * What the content decline releases, and what it must <b>not</b>. It is the only decline taken after both
     * opens succeeded, so it is the only one that owns two resources, and it releases them indirectly through
     * the routed input's ownership of the delegate's input. Nothing else in the suite would notice a leak of
     * one file handle per routed {@code .vec}, on exactly the substrate where every open declines.
     *
     * <p>The second assertion is the one the design turns on: the release must not reach
     * {@code DirectIOVectorCache.releaseFile}, which is keyed on the file rather than on the handle and is not
     * reference counted, so an open that read nothing would otherwise evict the vectors another routed reader
     * of the same file is being served from. {@code cache.closed_files} is how that release is visible.
     */
    @SneakyThrows
    public void testTheContentDeclineReleasesBothInputsAndEvictsNothing() {
        assumeDirectIOWorksHere();
        final Path path = createTempDir();
        try (MMapDirectory bottom = new MMapDirectory(path)) {
            writeFile(bottom, VECTOR_DATA, FILE_BYTES);
            final AtomicLong closes = new AtomicLong();
            final Directory transforming = new FilterDirectory(bottom) {
                @Override
                public IndexInput openInput(final String name, final IOContext context) throws IOException {
                    if (name.endsWith(".vec") == false) {
                        return super.openInput(name, context);
                    }
                    final byte[] onDisk;
                    try (IndexInput delegate = super.openInput(name, context)) {
                        onDisk = new byte[Math.toIntExact(delegate.length())];
                        delegate.readBytes(onDisk, 0, onDisk.length);
                    }
                    return new CloseCountingIndexInput(new ByteArrayIndexInput(name, XorDirectory.xor(onDisk)), closes);
                }
            };
            final long closedFilesBefore = DirectIOVectorCache.closedFiles();
            final KNNDirectIODirectory storage = new KNNDirectIODirectory(transforming, "_0", () -> true, () -> 0L);
            // Not closed yet, so the count isolates the declined open's own input from the caller's.
            final IndexInput served = storage.openInput(VECTOR_DATA, vectorDataContext());

            assertEquals(1, storage.declinedContent());
            assertEquals(0, storage.routedOpens());
            assertEquals("the input the declined open took must be closed, exactly once", 1L, closes.get());
            served.close();
            assertEquals("and the one the caller got is its own, closed by the caller", 2L, closes.get());
            assertEquals(
                "and the file's cached vectors must not be released by an open that read nothing",
                closedFilesBefore,
                DirectIOVectorCache.closedFiles()
            );
        }
    }

    /**
     * The byte check over the file sizes its arithmetic bends at, each one routed and read back whole. A
     * {@code .vec} smaller than the compared head, one exactly a block, one a block plus a byte, and one whose
     * footer straddles two blocks all take a different path through the offset and aligned-length arithmetic
     * than the 16 KiB fixture the other tests use, and all of them are reachable the moment an operator lowers
     * {@code knn.direct_io.min_file_size}.
     */
    @SneakyThrows
    public void testTheByteCheckRoutesEveryFileGeometry() {
        assumeDirectIOWorksHere();
        for (final int fileBytes : new int[] {
            64,
            2048,
            DirectIOVectorIndexInput.CONTENT_CHECK_HEAD_BYTES,
            DirectIOVectorIndexInput.CONTENT_CHECK_HEAD_BYTES + 1,
            FILE_BYTES - 1,
            FILE_BYTES }) {
            final Path path = createTempDir();
            try (MMapDirectory delegate = new MMapDirectory(path)) {
                final byte[] expected = writeFile(delegate, VECTOR_DATA, fileBytes);
                try (KNNDirectIODirectory storage = storageOver(delegate, true)) {
                    try (IndexInput input = storage.openInput(VECTOR_DATA, vectorDataContext())) {
                        assertTrue(
                            fileBytes + " B must route, got " + input.getClass().getName() + " declined=" + storage.declinedOpens(),
                            input instanceof DirectIOVectorIndexInput
                        );
                        assertArrayEquals("and read the same bytes at " + fileBytes + " B", expected, readAll(input));
                    }
                    assertEquals(fileBytes + " B", 1, storage.routedOpens());
                    assertEquals(fileBytes + " B", 0, storage.declinedContent());
                }
            }
        }
    }

    /** A {@code .vec} of no bytes at all: nothing to compare, nothing a transform could hide in, so it routes. */
    @SneakyThrows
    public void testAnEmptyVectorDataFileRoutes() {
        assumeDirectIOWorksHere();
        final Path path = createTempDir();
        Files.write(path.resolve(VECTOR_DATA), new byte[0]);
        try (MMapDirectory delegate = new MMapDirectory(path)) {
            try (KNNDirectIODirectory storage = storageOver(delegate, true)) {
                try (IndexInput input = storage.openInput(VECTOR_DATA, vectorDataContext())) {
                    assertTrue(input instanceof DirectIOVectorIndexInput);
                    assertEquals(0L, input.length());
                }
                assertEquals(1, storage.routedOpens());
                assertEquals(0, storage.declinedContent());
            }
        }
    }

    /**
     * The fail-closed arm of the comparison that no transform can reach: the device could not deliver the
     * region. The file is truncated under an already-open input, so the channel returns fewer bytes than the
     * region needs while the delegate still reports the original length — and a region that could not be read
     * has not been shown to match.
     */
    @SneakyThrows
    public void testARegionThatCannotBeReadFromTheDeviceDoesNotMatch() {
        assumeDirectIOWorksHere();
        final Path path = createTempDir();
        final byte[] content = contentOf(FILE_BYTES);
        final Path file = path.resolve(VECTOR_DATA);
        Files.write(file, content);
        try (DirectIOVectorIndexInput routed = DirectIOVectorIndexInput.open(file)) {
            assertNotNull(routed);
            assertEquals(FILE_BYTES, routed.length());
            // Both sides still agree while the file is whole.
            assertTrue(routed.contentMatches(new ByteArrayIndexInput(VECTOR_DATA, content)));
            try (FileChannel truncate = FileChannel.open(file, StandardOpenOption.WRITE)) {
                truncate.truncate(1024);
            }
            assertFalse(
                "a region the channel cannot deliver must not be reported as matching",
                routed.contentMatches(new ByteArrayIndexInput(VECTOR_DATA, content))
            );
        }
    }

    /** This process's cumulative device read bytes, or -1 where the runtime will not let them be read. */
    private static long deviceReadBytes() {
        try {
            for (final String line : Files.readAllLines(Path.of("/proc/self/io"))) {
                if (line.startsWith("read_bytes:")) {
                    return Long.parseLong(line.substring("read_bytes:".length()).trim());
                }
            }
            return -1L;
        } catch (IOException | RuntimeException e) {
            return -1L;
        }
    }

    /** O8, end to end through the dispatch point: the open is routed and the counter sees it. */
    @SneakyThrows
    public void testAHalfFloatVectorDataOpenIsServedWithDirectIO() {
        assumeDirectIOWorksHere();
        final Path path = createTempDir();
        final String name = "_0_" + KNNDirectIODirectory.HALF_FLOAT_FORMAT_NAME + "_0.vec";
        try (MMapDirectory delegate = new MMapDirectory(path)) {
            final byte[] expected = writeFile(delegate, name, FILE_BYTES);
            try (KNNDirectIODirectory storage = storageOver(delegate, true)) {
                try (IndexInput input = storage.openInput(name, vectorDataContext())) {
                    assertTrue("a half-float .vec must be routed", input instanceof DirectIOVectorIndexInput);
                    assertArrayEquals("and read the same bytes", expected, readAll(input));
                }
                assertEquals(1, storage.routedOpens());
                assertEquals(0, storage.declinedOpens());
            }
        } finally {
            DirectIOVectorIndexInput.STATS.reset();
        }
    }
}

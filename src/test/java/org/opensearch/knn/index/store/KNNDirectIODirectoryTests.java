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
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.store.MMapDirectory;
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

/**
 * The dispatch rule of {@link KNNDirectIODirectory}, asserted one condition at a time, plus the two
 * properties that decide whether installing it is safe on an index that is not using the feature.
 *
 * <p>The rule has six conditions and the interesting failures are all failures of <em>one</em> of them:
 * routing every {@code .vec} would put the Lucene engine's unquantized traversal on Direct I/O and off
 * its memory-segment SIMD scorer, which is outside what this feature is for; ignoring the setting would
 * turn the feature on everywhere including the substrates where the measurements say it loses; walking
 * past an unknown layer of the delegate chain would serve a byte-transforming wrapper's ciphertext as
 * vectors; ignoring the absence of a path would turn a remote-store index into a failed query rather than
 * an ordinary one; and ignoring the size floor is measured at 8,016x on a flush-sized file. So each is
 * tested by holding the others and breaking that one, and each decline is asserted to be counted under
 * its own reason, because a node that cannot tell them apart cannot act on them.
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

    private KNNDirectIODirectory storageOver(final Directory delegate, final boolean enabled, final long minFileBytes) {
        return new KNNDirectIODirectory(delegate, "_0", () -> enabled, () -> minFileBytes);
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
     * which is checked rather than assumed: {@link KNNDirectIODirectory#PASSTHROUGH_FILTER_DIRECTORIES}
     * is a {@code Set.of(...)} of class names and so carries nothing that varies with a node's history, but
     * a later edit to a {@code HashSet} would be exactly the state joint 6 forbids and would pass a check
     * on the declared type alone.
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
    // Condition 3: the delegate chain, and what must never be walked past
    // -------------------------------------------------------------------------------------------------

    /**
     * A {@link FilterDirectory} that transforms the bytes it serves. It stands in for the real thing this
     * guard exists for — an encryption-at-rest wrapper — in the only way that matters to the test: the
     * file on disk is not what a reader of this directory sees. Nothing but {@code openInput} needs to be
     * faithful, because nothing but {@code openInput} is what the guard decides.
     */
    private static final class XorDirectory extends FilterDirectory {
        private static final byte KEY = (byte) 0x5a;

        XorDirectory(final Directory delegate) {
            super(delegate);
        }

        /** Writes the on-disk form of {@code plain}, which is what a transforming wrapper stores. */
        @SneakyThrows
        static byte[] writeTransformed(final Directory bottom, final String name, final byte[] plain) {
            final byte[] onDisk = new byte[plain.length];
            for (int i = 0; i < plain.length; i++) {
                onDisk[i] = (byte) (plain[i] ^ KEY);
            }
            try (IndexOutput output = bottom.createOutput(name, IOContext.DEFAULT)) {
                output.writeBytes(onDisk, onDisk.length);
            }
            return onDisk;
        }

        @Override
        public IndexInput openInput(final String name, final IOContext context) throws IOException {
            final byte[] plain;
            try (IndexInput delegate = super.openInput(name, context)) {
                plain = new byte[Math.toIntExact(delegate.length())];
                delegate.readBytes(plain, 0, plain.length);
            }
            for (int i = 0; i < plain.length; i++) {
                plain[i] ^= KEY;
            }
            return new ByteArrayIndexInput(name, plain);
        }
    }

    /**
     * The guard's reason for existing. A byte-transforming {@link FilterDirectory} between the codec and
     * the {@link org.apache.lucene.store.FSDirectory} must <b>decline</b>, because the previous shape of
     * this class used {@link FilterDirectory#unwrap} — which walks past every layer whatever it does — and
     * would therefore have opened the raw on-disk file and served ciphertext as vectors. Silently, and
     * only for the one file this routes, while every other read in the system stayed correct.
     *
     * <p>Two assertions, and the second is the one that makes the first worth having: the open declines
     * <em>and</em> the bytes that come back through the delegate are the plaintext. Declining costs
     * read-ahead; bypassing costs correctness.
     */
    @SneakyThrows
    public void testAByteTransformingFilterDirectoryDeclinesAndStillReadsTheRightBytes() {
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
                            "an unknown FilterDirectory must never be bypassed to reach the file under it",
                            input instanceof DirectIOVectorIndexInput
                        );
                        assertArrayEquals("declining must leave the transforming layer doing its job", expected, readAll(input));
                    }
                    assertEquals(0, storage.routedOpens());
                    assertEquals("the reason must be the chain, not the size or the open", 1, storage.declinedChain());
                    assertEquals(0, storage.declinedTooSmall());
                    assertEquals(0, storage.declinedMissing());
                    assertEquals(0, storage.declinedOpenFailed());
                }
            }
        }
    }

    /**
     * The walk itself, with the allowlist as a parameter: a layer named in it is crossed, an unnamed one
     * stops the walk. Parameterised rather than tested through the production constant because the two
     * classes the constant names — {@code Store$StoreDirectory} and {@code ByteSizeCachingDirectory} — are
     * package private in OpenSearch and cannot be constructed from here. The constant's contents are
     * pinned separately, below.
     */
    @SneakyThrows
    public void testTheChainWalkCrossesAllowlistedLayersAndStopsAtUnknownOnes() {
        final Path path = createTempDir();
        try (MMapDirectory bottom = new MMapDirectory(path)) {
            final Directory known = new XorDirectory(bottom);
            final java.util.Set<String> allowlist = java.util.Set.of(XorDirectory.class.getName());

            assertEquals(
                "a layer in the allowlist is crossed",
                path.resolve(VECTOR_DATA),
                KNNDirectIODirectory.resolvePath(known, VECTOR_DATA, allowlist)
            );
            assertEquals(
                "two of them are crossed too, so the walk is a walk and not a single step",
                path.resolve(VECTOR_DATA),
                KNNDirectIODirectory.resolvePath(new XorDirectory(known), VECTOR_DATA, allowlist)
            );
            assertNull(
                "a layer absent from the allowlist stops the walk",
                KNNDirectIODirectory.resolvePath(known, VECTOR_DATA, java.util.Set.of())
            );
            assertNull("and so does a null", KNNDirectIODirectory.resolvePath(null, VECTOR_DATA, allowlist));
        }
    }

    /**
     * The production allowlist, pinned by name. Two of the three are package private in OpenSearch, so the
     * only check available here is that the set says what it is documented to say; the claim behind each
     * name — that the class has no read path of its own — was verified by reading {@code Store.java},
     * {@code ByteSizeCachingDirectory.java} and {@code BucketedCompositeDirectory.java}, and is recorded
     * per entry in the constant's javadoc. This test exists so that an addition to the list is a
     * deliberate edit of a test rather than a one-line change nobody reviews — and the list has already
     * grown once, when the S5 re-proof's decline log revealed a fourth layer in OpenSearch 3.9's real
     * chain that nothing in the design notes had mentioned.
     */
    public void testTheAllowlistNamesExactlyTheVerifiedPassthroughLayers() {
        assertEquals(
            java.util.Set.of(
                "org.opensearch.index.store.Store$StoreDirectory",
                "org.opensearch.index.store.ByteSizeCachingDirectory",
                "org.opensearch.index.BucketedCompositeDirectory"
            ),
            KNNDirectIODirectory.PASSTHROUGH_FILTER_DIRECTORIES
        );
    }

    /**
     * The chain description that goes into the decline log, which is the only thing that makes a chain
     * decline diagnosable: the counter can only ever say "some layer is not allowlisted", and the useful
     * question is which. Outermost first, stopping at the first non-{@link FilterDirectory}.
     *
     * <p>This is not cosmetic. The first S5 re-proof run routed nothing and declined 10 times on the chain;
     * the log line as originally written named only the <em>top</em> of the chain, which was an allowlisted
     * class, so it pointed away from the cause. With the full chain the answer was immediate.
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
     * {@code getDirectory().resolve(name)}, so the terminal case of the walk has to accept a subclass
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
    // Condition 5: the size floor (phase 13 audit FIX 1)
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

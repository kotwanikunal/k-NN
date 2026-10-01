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
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.codec.KNN1040Codec.Faiss1040ScalarQuantizedKnnVectorsFormat;
import org.opensearch.knn.index.codec.KNN990Codec.NativeEngines990KnnVectorsFormat;

import java.io.IOException;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.OpenOption;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

/**
 * The dispatch rule of {@link KNNVectorStorageDirectory}, asserted one condition at a time, plus the two
 * properties that decide whether installing it is safe on an index that is not using the feature.
 *
 * <p>The rule has three conditions and the interesting failures are all failures of <em>one</em> of them:
 * routing every {@code .vec} would put the Lucene engine's unquantized traversal on Direct I/O and off
 * its memory-segment SIMD scorer, which is outside what this feature is for; ignoring the setting would
 * turn the feature on everywhere including the substrates where the measurements say it loses; and
 * ignoring the absence of a path would turn a remote-store index into a failed query rather than an
 * ordinary one. So each is tested by holding the other two and breaking that one.
 *
 * <p>The other two properties are about what must <em>not</em> change. A file the rule does not name has
 * to come back as the delegate's own object, not a wrapper of it — a wrapper would displace Lucene's
 * memory-segment SIMD scorer for traversal, which is invisible to every correctness test and was
 * measured at 5.3× — and the class has to carry no node-wide mutable state, which its spike predecessor
 * did.
 */
@ThreadLeakFilters(defaultFilters = true, filters = { DirectIOReadPoolTests.ReadPoolThreadFilter.class })
public class KNNVectorStorageDirectoryTests extends KNNTestCase {

    /** A native-engine (faiss / MOS) fp32 flat file: the shape the rule routes. */
    private static final String VECTOR_DATA = "_0_" + NativeEngines990KnnVectorsFormat.FORMAT_NAME + "_0.vec";

    /** A faiss scalar-quantized fp32 flat file: the other shape the rule routes. */
    private static final String FAISS_SQ_VECTOR_DATA = "_0_" + Faiss1040ScalarQuantizedKnnVectorsFormat.FORMAT_NAME + "_0.vec";

    private static final int FILE_BYTES = 16 * 1024;

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

    private KNNVectorStorageDirectory storageOver(final Directory delegate, final boolean enabled) {
        return new KNNVectorStorageDirectory(delegate, "test-index", () -> enabled);
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
            try (KNNVectorStorageDirectory storage = storageOver(delegate, true)) {
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
            try (KNNVectorStorageDirectory storage = storageOver(delegate, true)) {
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
            try (KNNVectorStorageDirectory storage = storageOver(delegate, true)) {
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
            try (KNNVectorStorageDirectory storage = storageOver(delegate, true)) {
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
            try (KNNVectorStorageDirectory storage = storageOver(delegate, false)) {
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
            try (KNNVectorStorageDirectory storage = storageOver(delegate, true)) {
                try (IndexInput input = storage.openInput(VECTOR_DATA, vectorDataContext())) {
                    assertFalse(input instanceof DirectIOVectorIndexInput);
                    assertArrayEquals("declining must still read the right bytes", expected, readAll(input));
                }
                assertEquals(0, storage.routedOpens());
                assertEquals("a decline is counted, so a node can tell it from silence", 1, storage.declinedOpens());
            }
        }
    }

    // -------------------------------------------------------------------------------------------------
    // The second dispatch point
    // -------------------------------------------------------------------------------------------------

    /**
     * Every compound container is wrapped, because a compound segment's {@code .vec} has no
     * {@code openInput} of its own to dispatch on — it arrives as a four-argument slice of this container.
     * Nothing else is wrapped: the container is the only file whose <em>entries</em> need a decision.
     */
    @SneakyThrows
    public void testTheCompoundContainerIsWrappedAndNothingElseIs() {
        final Path path = createTempDir();
        try (MMapDirectory delegate = new MMapDirectory(path)) {
            writeFile(delegate, "_0.cfs", FILE_BYTES);
            writeFile(delegate, "_0.cfe", 1024);
            writeFile(delegate, "_0_Lucene99FlatVectorsFormat_0.vec", FILE_BYTES);
            try (KNNVectorStorageDirectory storage = storageOver(delegate, true)) {
                try (IndexInput container = storage.openInput("_0.cfs", IOContext.DEFAULT)) {
                    assertTrue(
                        "the compound container is the compound segment's dispatch point",
                        container instanceof KNNVectorCompoundSliceInput
                    );
                    assertArrayEquals(contentOf(FILE_BYTES), readAll(container));
                }
                for (final String name : new String[] { "_0.cfe", "_0_Lucene99FlatVectorsFormat_0.vec" }) {
                    try (IndexInput input = storage.openInput(name, IOContext.DEFAULT)) {
                        assertFalse(name + " must not be wrapped", input instanceof KNNVectorCompoundSliceInput);
                    }
                }
                assertEquals(1, storage.wrappedContainers());
            }
        }
    }

    /**
     * The production container does not record its slices. Its predecessor did, into an unbounded
     * {@link java.util.concurrent.CopyOnWriteArrayList}, which is correct for a spike and is a leak plus an
     * O(n²) on a shard's real slice traffic. The counters are kept either way.
     */
    @SneakyThrows
    public void testTheProductionContainerCountsSlicesRatherThanRecordingThem() {
        final Path path = createTempDir();
        try (MMapDirectory delegate = new MMapDirectory(path)) {
            writeFile(delegate, "_0.cfs", FILE_BYTES);
            try (KNNVectorStorageDirectory storage = storageOver(delegate, true)) {
                try (IndexInput container = storage.openInput("_0.cfs", IOContext.DEFAULT)) {
                    final KNNVectorCompoundSliceInput wrapped = (KNNVectorCompoundSliceInput) container;
                    for (int i = 0; i < 64; i++) {
                        wrapped.slice("entry" + i, 0, 512, IOContext.DEFAULT).close();
                    }
                    assertTrue("a production container must not accumulate per-slice observations", wrapped.sliceObservations().isEmpty());
                    assertEquals("no entry here is a faiss .vec, so nothing is routed", 0, wrapped.routedSlices());
                    assertEquals(0, wrapped.declinedSlices());
                }
            }
        }
    }

    /**
     * The container's routing follows the setting while the container is open, which is what makes the
     * dynamic setting dynamic for compound segments: a container is opened once per segment and sliced
     * once per values object, so a decision captured at open time would need a reopen to take effect.
     */
    @SneakyThrows
    public void testTheContainersRoutingFollowsTheSettingWhileItIsOpen() {
        assumeDirectIOWorksHere();
        final Path path = createTempDir();
        try (MMapDirectory delegate = new MMapDirectory(path)) {
            writeFile(delegate, "_0.cfs", FILE_BYTES);
            final boolean[] enabled = { false };
            try (KNNVectorStorageDirectory storage = new KNNVectorStorageDirectory(delegate, "test-index", () -> enabled[0])) {
                try (IndexInput container = storage.openInput("_0.cfs", IOContext.DEFAULT)) {
                    final KNNVectorCompoundSliceInput wrapped = (KNNVectorCompoundSliceInput) container;
                    wrapped.slice(VECTOR_DATA, 1088, 4096, vectorDataContext()).close();
                    assertEquals("with the setting off the entry stays on the delegate's slice", 0, wrapped.routedSlices());

                    enabled[0] = true;
                    try (IndexInput entry = wrapped.slice(VECTOR_DATA, 1088, 4096, vectorDataContext())) {
                        assertTrue("with the setting on the entry is routed", entry instanceof DirectIOVectorIndexInput);
                    }
                    assertEquals(1, wrapped.routedSlices());
                }
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
     */
    public void testTheDirectoryKeepsNoNodeWideState() {
        for (final Field field : KNNVectorStorageDirectory.class.getDeclaredFields()) {
            if (Modifier.isStatic(field.getModifiers()) == false) {
                continue;
            }
            assertTrue("static field " + field.getName() + " must be final", Modifier.isFinal(field.getModifiers()));
            final Class<?> type = field.getType();
            // A logger carries nothing that varies with an index or a node's history, and it is generated
            // rather than written; everything else static would.
            final boolean stateless = type.isPrimitive()
                || type == String.class
                || org.apache.logging.log4j.Logger.class.isAssignableFrom(type);
            assertTrue(
                "static field " + field.getName() + " is of mutable type " + type.getName() + "; node-wide mutable state is joint 6",
                stateless
            );
        }
        // And the shape it must not have, named: the spike's flag, which a later edit might copy across.
        for (final Field field : KNNVectorStorageDirectory.class.getDeclaredFields()) {
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
            try (KNNVectorStorageDirectory storage = storageOver(delegate, true)) {
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

    /** {@link KNNVectorStorageDirectory#find} is a walk of the caller's chain and finds nothing when absent. */
    @SneakyThrows
    public void testFindWalksTheChainAndAnswersNullWhenNotInstalled() {
        try (ByteBuffersDirectory base = new ByteBuffersDirectory()) {
            assertNull(KNNVectorStorageDirectory.find(base));
            assertNull(KNNVectorStorageDirectory.find(null));
            try (KNNVectorStorageDirectory storage = storageOver(base, true)) {
                final Directory above = new FilterDirectory(new FilterDirectory(storage) {
                }) {
                };
                assertSame(storage, KNNVectorStorageDirectory.find(above));
                assertSame(storage, KNNVectorStorageDirectory.find(storage));
            }
        }
    }

    /**
     * The predicate is name-based, so it is worth pinning to the names the real formats actually produce:
     * {@code PerFieldKnnVectorsFormat} builds each field's segment suffix from
     * {@code KnnVectorsFormat#getName()}, so a drift in either format's name would silently stop routing.
     */
    public void testThePredicateMatchesTheNamesTheRealFormatsProduce() {
        assertTrue(KNNVectorStorageDirectory.isFaissVectorData("_7_" + new NativeEngines990KnnVectorsFormat(0).getName() + "_0.vec"));
        assertTrue(
            KNNVectorStorageDirectory.isFaissVectorData("_7_" + new Faiss1040ScalarQuantizedKnnVectorsFormat().getName() + "_0.vec")
        );
        assertFalse(KNNVectorStorageDirectory.isFaissVectorData("_7_Lucene99HnswVectorsFormat_0.vec"));
        assertFalse(KNNVectorStorageDirectory.isFaissVectorData("_7_" + NativeEngines990KnnVectorsFormat.FORMAT_NAME + "_0.veq"));
        assertFalse(KNNVectorStorageDirectory.isFaissVectorData(null));
    }
}

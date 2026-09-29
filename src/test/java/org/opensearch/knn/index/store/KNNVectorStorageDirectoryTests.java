/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import com.carrotsearch.randomizedtesting.annotations.ThreadLeakFilters;
import lombok.SneakyThrows;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FilterDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.store.MMapDirectory;
import org.opensearch.knn.KNNTestCase;

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
 * <p>The rule has four conditions and the interesting failures are all failures of <em>one</em> of them:
 * routing by name alone would put the traversal, merge, warmup and derived-source reads of {@code .vec}
 * on Direct I/O, which is the regression the whole design is shaped to avoid; routing by intent alone
 * would put the quantized codes and the native index there; ignoring the setting would turn the feature
 * on everywhere including the substrates where the phase-8 measurements say it loses; and ignoring the
 * absence of a path would turn a remote-store index into a failed query rather than an ordinary one. So
 * each is tested by holding the other three and breaking that one.
 *
 * <p>The other two properties are about what must <em>not</em> change. A {@code .vec} read without the
 * intent has to come back as the delegate's own object, not a wrapper of it — a wrapper would displace
 * Lucene's memory-segment SIMD scorer for traversal, which is invisible to every correctness test and
 * was measured at 5.3× — and the class has to carry no node-wide mutable state, which its spike
 * predecessor did.
 */
@ThreadLeakFilters(defaultFilters = true, filters = { DirectIOReadPoolTests.ReadPoolThreadFilter.class })
public class KNNVectorStorageDirectoryTests extends KNNTestCase {

    private static final String VECTOR_DATA = "_0_Lucene99FlatVectorsFormat_0.vec";
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

    /** The context Lucene's flat vectors reader builds, with the plugin's intent added on top. */
    private static IOContext rescoreContext() {
        return KNNVectorReadIntent.RESCORE.vectorDataContext();
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
    // All four conditions hold
    // -------------------------------------------------------------------------------------------------

    /**
     * The routed case, and the only assertion that matters about it besides the type: the bytes are the
     * delegate's bytes. A route that returned different bytes would be a correctness bug that no latency
     * measurement would catch.
     */
    @SneakyThrows
    public void testVectorDataWithTheRescoreIntentIsServedWithDirectIOAndReadsTheSameBytes() {
        assumeDirectIOWorksHere();
        final Path path = createTempDir();
        try (MMapDirectory delegate = new MMapDirectory(path)) {
            final byte[] expected = writeFile(delegate, VECTOR_DATA, FILE_BYTES);
            try (KNNVectorStorageDirectory storage = storageOver(delegate, true)) {
                try (IndexInput routed = storage.openInput(VECTOR_DATA, rescoreContext())) {
                    assertTrue(
                        "a .vec open carrying RESCORE should be served with O_DIRECT, got " + routed.getClass().getName(),
                        routed instanceof DirectIOVectorIndexInput
                    );
                    assertEquals(expected.length, routed.length());
                    assertArrayEquals("the Direct I/O route must read the delegate's bytes exactly", expected, readAll(routed));
                }
                assertEquals(1, storage.routedOpens());
                assertEquals(0, storage.declinedOpens());
            }
        }
    }

    // -------------------------------------------------------------------------------------------------
    // Each condition broken in turn
    // -------------------------------------------------------------------------------------------------

    /**
     * Condition 2 broken: the same file, opened the way every other reader of it opens it. This is the
     * case that must come back <em>unwrapped</em>, because traversal and re-score share neither the object
     * nor the mechanism, and the traversal object has to stay the one Lucene's SIMD scorer will bind to.
     */
    @SneakyThrows
    public void testVectorDataWithoutTheIntentIsTheDelegatesOwnInputUnwrapped() {
        final Path path = createTempDir();
        try (MMapDirectory delegate = new MMapDirectory(path)) {
            writeFile(delegate, VECTOR_DATA, FILE_BYTES);
            final Class<?> delegateInputClass;
            try (IndexInput reference = delegate.openInput(VECTOR_DATA, IOContext.DEFAULT)) {
                delegateInputClass = reference.getClass();
            }
            try (KNNVectorStorageDirectory storage = storageOver(delegate, true)) {
                try (IndexInput plain = storage.openInput(VECTOR_DATA, IOContext.DEFAULT)) {
                    assertFalse("a .vec read with no intent must not be routed", plain instanceof DirectIOVectorIndexInput);
                    assertSame(
                        "a .vec read with no intent must be the delegate's own object, not a wrapper of it",
                        delegateInputClass,
                        plain.getClass()
                    );
                }
                assertEquals(0, storage.routedOpens());
                assertEquals(0, storage.declinedOpens());
            }
        }
    }

    /**
     * Condition 1 broken: the intent held, on a file that is not the full-precision vector data. The
     * quantized codes are the traversal path's own file and the native index is read by the JNI layer;
     * neither is what Direct I/O exists for, and an intent that leaked onto them would route the hottest
     * reads in the system.
     */
    @SneakyThrows
    public void testTheIntentRoutesNothingButTheFullPrecisionVectorData() {
        assumeDirectIOWorksHere();
        final Path path = createTempDir();
        try (MMapDirectory delegate = new MMapDirectory(path)) {
            for (final String name : new String[] {
                "_0_Lucene104ScalarQuantizedVectorsFormat_0.veq",
                "_0_NativeEngines990KnnVectorsFormat_0.faiss",
                "_0_Lucene99FlatVectorsFormat_0.vemf",
                "_0.si" }) {
                writeFile(delegate, name, 4096);
            }
            try (KNNVectorStorageDirectory storage = storageOver(delegate, true)) {
                for (final String name : delegate.listAll()) {
                    try (IndexInput input = storage.openInput(name, rescoreContext())) {
                        assertFalse(
                            name + " must not be routed to Direct I/O however the context is tagged",
                            input instanceof DirectIOVectorIndexInput
                        );
                    }
                }
                assertEquals(0, storage.routedOpens());
            }
        }
    }

    /** Condition 3 broken: the operator's switch is off, which is the default on every node. */
    @SneakyThrows
    public void testTheSettingIsTheOperatorSwitchAndOffMeansUnchanged() {
        final Path path = createTempDir();
        try (MMapDirectory delegate = new MMapDirectory(path)) {
            final byte[] expected = writeFile(delegate, VECTOR_DATA, FILE_BYTES);
            try (KNNVectorStorageDirectory storage = storageOver(delegate, false)) {
                try (IndexInput input = storage.openInput(VECTOR_DATA, rescoreContext())) {
                    assertFalse("with the setting off nothing may be routed", input instanceof DirectIOVectorIndexInput);
                    assertArrayEquals(expected, readAll(input));
                }
                assertEquals(0, storage.routedOpens());
                assertEquals("declining on the setting is not a Direct I/O failure", 0, storage.declinedOpens());
            }
        }
    }

    /**
     * Condition 4 broken: a delegate with no filesystem underneath it — a remote-store or in-memory
     * directory. The answer must be an ordinary read, because the alternative is a failed query on a
     * substrate where Direct I/O was never available in the first place.
     */
    @SneakyThrows
    public void testADirectoryWithNoPathDeclinesRatherThanFails() {
        try (ByteBuffersDirectory delegate = new ByteBuffersDirectory()) {
            final byte[] expected = writeFile(delegate, VECTOR_DATA, FILE_BYTES);
            try (KNNVectorStorageDirectory storage = storageOver(delegate, true)) {
                try (IndexInput input = storage.openInput(VECTOR_DATA, rescoreContext())) {
                    assertFalse(input instanceof DirectIOVectorIndexInput);
                    assertArrayEquals("declining must still read the right bytes", expected, readAll(input));
                }
                assertEquals(0, storage.routedOpens());
                assertEquals("a decline is counted, so a node can tell it from silence", 1, storage.declinedOpens());
            }
        }
    }

    /**
     * A merge read of {@code .vec} cannot be routed, and it is Lucene that guarantees it rather than a
     * check here: {@code IOContext.merge(..).withHints(..)} returns the context with an empty hint set, so
     * condition 2 is structurally unsatisfiable on the one read pattern that must stay sequential.
     */
    @SneakyThrows
    public void testMergeAndFlushReadsAreStructurallyUnroutable() {
        final Path path = createTempDir();
        try (MMapDirectory delegate = new MMapDirectory(path)) {
            writeFile(delegate, VECTOR_DATA, FILE_BYTES);
            try (KNNVectorStorageDirectory storage = storageOver(delegate, true)) {
                for (final IOContext context : new IOContext[] {
                    IOContext.merge(new org.apache.lucene.store.MergeInfo(10, 1024, true, 1)).withHints(KNNVectorReadIntent.RESCORE),
                    IOContext.flush(new org.apache.lucene.store.FlushInfo(10, 1024)).withHints(KNNVectorReadIntent.RESCORE) }) {
                    assertNull("a merge or flush context cannot carry the intent at all", KNNVectorReadIntent.of(context));
                    try (IndexInput input = storage.openInput(VECTOR_DATA, context)) {
                        assertFalse(input instanceof DirectIOVectorIndexInput);
                    }
                }
                assertEquals(0, storage.routedOpens());
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
            writeFile(delegate, VECTOR_DATA, FILE_BYTES);
            try (KNNVectorStorageDirectory storage = storageOver(delegate, true)) {
                try (IndexInput container = storage.openInput("_0.cfs", IOContext.DEFAULT)) {
                    assertTrue(
                        "the compound container is the compound segment's dispatch point",
                        container instanceof KNNVectorCompoundSliceInput
                    );
                    assertArrayEquals(contentOf(FILE_BYTES), readAll(container));
                }
                for (final String name : new String[] { "_0.cfe", VECTOR_DATA }) {
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
                    assertEquals("no entry here is a re-score .vec, so nothing is routed", 0, wrapped.routedSlices());
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
                    wrapped.slice(VECTOR_DATA, 1088, 4096, rescoreContext()).close();
                    assertEquals("with the setting off the entry stays on the delegate's slice", 0, wrapped.routedSlices());

                    enabled[0] = true;
                    try (IndexInput entry = wrapped.slice(VECTOR_DATA, 1088, 4096, rescoreContext())) {
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
}

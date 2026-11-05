/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import org.apache.lucene.codecs.Codec;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.misc.store.DirectIODirectory;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.DataAccessHint;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FilterDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.Version;
import org.opensearch.core.index.Index;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.codec.KNN1040Codec.Faiss1040HalfFloatScalarQuantizedKnnVectorsFormat;
import org.opensearch.knn.index.codec.KNN1040Codec.Faiss1040ScalarQuantizedKnnVectorsFormat;
import org.opensearch.knn.index.codec.KNN990Codec.NativeEngines990HalfFloatKnnVectorsFormat;
import org.opensearch.knn.index.codec.KNN990Codec.NativeEngines990KnnVectorsFormat;
import org.opensearch.knn.index.store.DirectIOTestUtils.CryptoShapedFsDirectory;
import org.opensearch.knn.index.store.DirectIOTestUtils.GhostNameFsDirectory;
import org.opensearch.knn.index.store.DirectIOTestUtils.PassThroughDirectory;
import org.opensearch.knn.index.store.DirectIOTestUtils.TestChannels;
import org.opensearch.knn.index.store.DirectIOTestUtils.TestCompoundDirectory;

import java.io.IOException;
import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.opensearch.knn.index.store.DirectIOTestUtils.FAISS_NAME;
import static org.opensearch.knn.index.store.DirectIOTestUtils.HALF_FLOAT_SQ_VEC_NAME;
import static org.opensearch.knn.index.store.DirectIOTestUtils.HALF_FLOAT_VEC_NAME;
import static org.opensearch.knn.index.store.DirectIOTestUtils.INDEX_UUID;
import static org.opensearch.knn.index.store.DirectIOTestUtils.LUCENE_VEC_NAME;
import static org.opensearch.knn.index.store.DirectIOTestUtils.META_NAME;
import static org.opensearch.knn.index.store.DirectIOTestUtils.QUANTIZED_NAME;
import static org.opensearch.knn.index.store.DirectIOTestUtils.SQ_VEC_NAME;
import static org.opensearch.knn.index.store.DirectIOTestUtils.VEC_NAME;

/**
 * The directory's own work: each of the six eligibility checks on its own and a pass (U1), the index UUID read from
 * the shard path and the index found from it (U6), the constants the rule cannot read from the class that owns them,
 * the {@code DIRECT} open option, the buffer sizes, and {@code routeVectorData} on both faiss formats.
 */
public class KNNDirectIODirectoryTests extends KNNTestCase {

    private static final int FILE_PAYLOAD_BYTES = 20_000;
    private static final String INDEX_NAME = "target-index";

    private Path shardPath;
    private final AtomicBoolean on = new AtomicBoolean(true);

    @Override
    public void setUp() throws Exception {
        super.setUp();
        shardPath = DirectIOTestUtils.defaultLayoutShardPath(createTempDir(), INDEX_UUID);
        on.set(true);
    }

    public void testRoutesAFaissVectorDataFileThatPassesEveryCheck() throws IOException {
        DirectIOTestUtils.assumeDirectIOSupported();
        try (Directory in = DirectIOTestUtils.fsDirectory(shardPath)) {
            DirectIOTestUtils.writeCodecFile(in, VEC_NAME, FILE_PAYLOAD_BYTES);
            final TestChannels channels = new TestChannels();
            final KNNDirectIODirectory directory = DirectIOTestUtils.directory(in, on::get, channels);
            try (IndexInput file = directory.openInput(VEC_NAME, IOContext.DEFAULT)) {
                assertTrue("the delegate hands out a switch", file instanceof SwitchableIndexInput);
                final IndexInput slice = file.slice("vector-data", 0, file.length());
                assertTrue("a slice taken while the index is on reads with O_DIRECT", slice instanceof DirectIOVectorIndexInput);
                assertEquals("one channel per settled file", 1, channels.channels().size());
                assertTrue("the content check read the file through that channel", channels.reads() > 0);
            }
        }
    }

    public void testACompoundSegmentGetsTheOriginalDirectorysOwnInput() throws IOException {
        try (Directory in = DirectIOTestUtils.fsDirectory(shardPath)) {
            DirectIOTestUtils.writeCodecFile(in, VEC_NAME, FILE_PAYLOAD_BYTES);
            try (Directory compound = new TestCompoundDirectory(in)) {
                assertOriginalInput(compound, DirectIOTestUtils.directory(compound, on::get), VEC_NAME);
            }
        }
    }

    public void testADirectoryWithNoPathGetsTheOriginalDirectorysOwnInput() throws IOException {
        try (Directory in = new ByteBuffersDirectory()) {
            DirectIOTestUtils.writeCodecFile(in, VEC_NAME, FILE_PAYLOAD_BYTES);
            assertOriginalInput(in, DirectIOTestUtils.directory(in, on::get), VEC_NAME);
        }
    }

    public void testAMissingFileGetsTheOriginalDirectorysOwnInput() throws IOException {
        try (Directory backing = DirectIOTestUtils.fsDirectory(shardPath)) {
            DirectIOTestUtils.writeCodecFile(backing, SQ_VEC_NAME, FILE_PAYLOAD_BYTES);
        }
        try (Directory in = new GhostNameFsDirectory(shardPath, VEC_NAME, SQ_VEC_NAME)) {
            assertOriginalInput(in, DirectIOTestUtils.directory(in, on::get), VEC_NAME);
        }
    }

    public void testAFileUnderTheFloorGetsTheOriginalDirectorysOwnInput() throws IOException {
        try (Directory in = DirectIOTestUtils.fsDirectory(shardPath)) {
            DirectIOTestUtils.writeCodecFile(in, VEC_NAME, 16);
            final KNNDirectIODirectory directory = new KNNDirectIODirectory(
                in,
                INDEX_UUID,
                () -> KNNDirectIODirectory.MIN_FILE_SIZE_BYTES,
                on::get,
                KNNDirectIODirectory.DEFAULT_CHANNEL_FACTORY,
                KNNDirectIODirectory.DIRECT_OPEN_OPTION
            );
            // Being the original directory's own input is the whole statement: it answers every question that input
            // answers, so nothing makes Lucene open the file a second time for a merge or a sequential search.
            // Asserting isEfficientFor and isLoaded against a second input of the same class would assert nothing.
            assertOriginalInput(in, directory, VEC_NAME);
        }
    }

    public void testAnAbsentDirectOptionGetsTheOriginalDirectorysOwnInput() throws IOException {
        try (Directory in = DirectIOTestUtils.fsDirectory(shardPath)) {
            DirectIOTestUtils.writeCodecFile(in, VEC_NAME, FILE_PAYLOAD_BYTES);
            // A JDK without com.sun.nio.file.ExtendedOpenOption.DIRECT: nothing is wrapped, and the warn line was
            // logged once by the static initializer rather than once per file.
            final KNNDirectIODirectory directory = DirectIOTestUtils.directory(in, on::get, path -> {
                throw new AssertionError("a runtime without the DIRECT option must not open a channel");
            }, null);
            assertOriginalInput(in, directory, VEC_NAME);
        }
    }

    public void testAFileWhoseDirectOpenFailsReadsThroughTheOriginalInput() throws IOException {
        try (Directory in = DirectIOTestUtils.fsDirectory(shardPath)) {
            final byte[] expected = DirectIOTestUtils.writeCodecFile(in, VEC_NAME, FILE_PAYLOAD_BYTES);
            final KNNDirectIODirectory directory = DirectIOTestUtils.directory(
                in,
                on::get,
                path -> { throw new IOException("no O_DIRECT here"); }
            );
            try (IndexInput file = directory.openInput(VEC_NAME, IOContext.DEFAULT)) {
                // The file system is a per-file fact, so a file that passed the checks at open is still wrapped: the
                // open failure is learned at the first slice, and the slice is then the original input's own.
                assertTrue(file instanceof SwitchableIndexInput);
                final IndexInput slice = file.slice("vector-data", 0, file.length());
                assertFalse(slice instanceof DirectIOVectorIndexInput);
                final byte[] read = new byte[expected.length];
                slice.seek(0);
                slice.readBytes(read, 0, read.length);
                assertArrayEquals(expected, read);
            }
        }
    }

    public void testDeclinesACryptoShapedNioFsSubclassWithContent() throws IOException {
        DirectIOTestUtils.assumeDirectIOSupported();
        try (Directory plain = DirectIOTestUtils.fsDirectory(shardPath)) {
            DirectIOTestUtils.writeCodecFile(plain, VEC_NAME, FILE_PAYLOAD_BYTES);
        }
        try (Directory in = new CryptoShapedFsDirectory(shardPath)) {
            final KNNDirectIODirectory directory = DirectIOTestUtils.directory(in, on::get);
            try (IndexInput file = directory.openInput(VEC_NAME, IOContext.DEFAULT)) {
                final IndexInput slice = file.slice("vector-data", 0, file.length());
                assertFalse("a layer that transforms bytes stays on the original input", slice instanceof DirectIOVectorIndexInput);
                assertFalse("and never settles routed", ((SwitchableIndexInput) file).isRouted());
            }
        }
    }

    public void testRoutesThroughAPassThroughFilterDirectory() throws IOException {
        DirectIOTestUtils.assumeDirectIOSupported();
        try (Directory bottom = DirectIOTestUtils.fsDirectory(shardPath)) {
            DirectIOTestUtils.writeCodecFile(bottom, VEC_NAME, FILE_PAYLOAD_BYTES);
            try (Directory in = new PassThroughDirectory(bottom)) {
                final KNNDirectIODirectory directory = DirectIOTestUtils.directory(in, on::get);
                try (IndexInput file = directory.openInput(VEC_NAME, IOContext.DEFAULT)) {
                    final IndexInput slice = file.slice("vector-data", 0, file.length());
                    assertTrue("a pass-through layer is crossed and routed", slice instanceof DirectIOVectorIndexInput);
                }
            }
        }
    }

    public void testOffersNothingForALuceneEngineVectorDataFile() throws IOException {
        try (Directory in = DirectIOTestUtils.fsDirectory(shardPath)) {
            DirectIOTestUtils.writeCodecFile(in, LUCENE_VEC_NAME, FILE_PAYLOAD_BYTES);
            assertOriginalInput(in, DirectIOTestUtils.directory(in, on::get), LUCENE_VEC_NAME);
            assertFalse(KNNDirectIODirectory.isFaissVectorData(LUCENE_VEC_NAME));
        }
    }

    public void testOffersNothingForAFileThatIsNotVectorData() throws IOException {
        try (Directory in = DirectIOTestUtils.fsDirectory(shardPath)) {
            for (final String name : new String[] { META_NAME, QUANTIZED_NAME, FAISS_NAME }) {
                DirectIOTestUtils.writeCodecFile(in, name, FILE_PAYLOAD_BYTES);
                assertFalse(name + " is not vector data", KNNDirectIODirectory.isFaissVectorData(name));
                assertOriginalInput(in, DirectIOTestUtils.directory(in, on::get), name);
            }
        }
    }

    public void testRoutesTheTwoHalfFloatFormatNames() throws IOException {
        DirectIOTestUtils.assumeDirectIOSupported();
        try (Directory in = DirectIOTestUtils.fsDirectory(shardPath)) {
            for (final String name : new String[] { HALF_FLOAT_VEC_NAME, HALF_FLOAT_SQ_VEC_NAME }) {
                DirectIOTestUtils.writeCodecFile(in, name, FILE_PAYLOAD_BYTES);
                final KNNDirectIODirectory directory = DirectIOTestUtils.directory(in, on::get);
                try (IndexInput file = directory.openInput(name, IOContext.DEFAULT)) {
                    final IndexInput slice = file.slice("vector-data", 0, file.length());
                    assertTrue(name + " must route", slice instanceof DirectIOVectorIndexInput);
                }
            }
        }
    }

    public void testRouteVectorDataOnBothFaissFormats() throws IOException {
        DirectIOTestUtils.assumeDirectIOSupported();
        try (Directory in = DirectIOTestUtils.fsDirectory(shardPath)) {
            for (final String name : new String[] { VEC_NAME, SQ_VEC_NAME }) {
                DirectIOTestUtils.writeCodecFile(in, name, FILE_PAYLOAD_BYTES);
            }
            for (final String suffix : new String[] {
                NativeEngines990KnnVectorsFormat.FORMAT_NAME + "_0",
                Faiss1040ScalarQuantizedKnnVectorsFormat.FORMAT_NAME + "_0" }) {
                final SegmentReadState state = new SegmentReadState(in, segmentInfo(in), FieldInfos.EMPTY, IOContext.DEFAULT, suffix);
                final SegmentReadState routed = KNNDirectIODirectory.routeVectorData(state);
                assertTrue("the state of " + suffix + " carries the delegate", routed.directory instanceof KNNDirectIODirectory);
                assertSame(state.segmentInfo, routed.segmentInfo);
            }
            // And the file each format writes is wrapped by that delegate.
            final KNNDirectIODirectory directory = DirectIOTestUtils.directory(in, on::get);
            for (final String name : new String[] { VEC_NAME, SQ_VEC_NAME }) {
                try (IndexInput file = directory.openInput(name, IOContext.DEFAULT)) {
                    assertTrue(name, file instanceof SwitchableIndexInput);
                }
            }
        }
    }

    public void testAnIndexThatIsOffOpensNoChannel() throws IOException {
        on.set(false);
        try (Directory in = DirectIOTestUtils.fsDirectory(shardPath)) {
            final byte[] expected = DirectIOTestUtils.writeCodecFile(in, VEC_NAME, FILE_PAYLOAD_BYTES);
            final KNNDirectIODirectory directory = DirectIOTestUtils.directory(in, on::get, path -> {
                throw new AssertionError("an index that is off must not open an O_DIRECT channel");
            });
            try (IndexInput file = directory.openInput(VEC_NAME, IOContext.DEFAULT)) {
                final IndexInput slice = file.slice("vector-data", 0, file.length());
                final IndexInput clone = file.clone();
                final byte[] read = new byte[expected.length];
                clone.seek(0);
                clone.readBytes(read, 0, read.length);
                assertArrayEquals(expected, read);
                final byte[] fromSlice = new byte[expected.length];
                slice.seek(0);
                slice.readBytes(fromSlice, 0, fromSlice.length);
                assertArrayEquals(expected, fromSlice);
                assertFalse(((SwitchableIndexInput) file).isRouted());
            }
        }
    }

    public void testTheDelegateIsInstalledOnAnIndexThatIsOff() throws IOException {
        DirectIOTestUtils.assumeDirectIOSupported();
        on.set(false);
        try (Directory in = DirectIOTestUtils.fsDirectory(shardPath)) {
            assertEquals("the UUID comes from the shard path whatever the setting says", INDEX_UUID, KNNDirectIODirectory.indexUuidOf(in));
            assertSame(in, FilterDirectory.unwrap(DirectIOTestUtils.directory(in, on::get)));
            DirectIOTestUtils.writeCodecFile(in, VEC_NAME, FILE_PAYLOAD_BYTES);
            final KNNDirectIODirectory directory = DirectIOTestUtils.directory(in, on::get);
            // FOLLOW-UPS M7, the decision of 2026-11-02 and not one of the mutations M1 to M6: the switch the
            // delegate hands out is there even while the index is off, which is what lets a later flip reach a
            // segment that is already open, with no close and open of the index.
            try (IndexInput file = directory.openInput(VEC_NAME, IOContext.DEFAULT)) {
                assertTrue(file instanceof SwitchableIndexInput);
                assertFalse("and it routes nothing", file.clone() instanceof DirectIOVectorIndexInput);
                on.set(true);
                assertTrue("until the index is turned on", file.clone() instanceof DirectIOVectorIndexInput);
            }
        }
    }

    public void testIndexUuidOfReadsBothDataPathLayouts() throws IOException {
        final Path root = createTempDir();
        final Path defaultLayout = DirectIOTestUtils.defaultLayoutShardPath(root, INDEX_UUID);
        final Path customLayout = DirectIOTestUtils.customLayoutShardPath(root, INDEX_UUID);
        try (Directory in = DirectIOTestUtils.fsDirectory(defaultLayout)) {
            assertEquals(INDEX_UUID, KNNDirectIODirectory.indexUuidOf(in));
        }
        try (Directory in = DirectIOTestUtils.fsDirectory(customLayout)) {
            assertEquals("a custom index.data_path has no indices component", INDEX_UUID, KNNDirectIODirectory.indexUuidOf(in));
        }
    }

    public void testIndexUuidOfIsNullForAnythingElse() throws IOException {
        final Path root = createTempDir();
        final Path notIndexFolder = Files.createDirectories(root.resolve(INDEX_UUID).resolve("0").resolve("data"));
        final Path nonNumericShard = Files.createDirectories(root.resolve(INDEX_UUID).resolve("shard").resolve("index"));
        final Path tooShallow = Files.createDirectories(root.resolve("index"));
        for (final Path path : new Path[] { notIndexFolder, nonNumericShard, tooShallow }) {
            try (Directory in = DirectIOTestUtils.fsDirectory(path)) {
                assertNull("no UUID for " + path, KNNDirectIODirectory.indexUuidOf(in));
            }
        }
    }

    public void testRouteVectorDataTakesTheUuidFromTheSegmentInfoDirectory() throws IOException {
        try (Directory shardDirectory = DirectIOTestUtils.fsDirectory(shardPath)) {
            try (Directory compound = new TestCompoundDirectory(shardDirectory)) {
                final SegmentReadState state = new SegmentReadState(
                    compound,
                    segmentInfo(shardDirectory),
                    FieldInfos.EMPTY,
                    IOContext.DEFAULT,
                    "NativeEngines990KnnVectorsFormat_0"
                );
                final SegmentReadState routed = KNNDirectIODirectory.routeVectorData(state);
                assertNotSame("a compound segment of a known index is still wrapped", state, routed);
                assertTrue(routed.directory instanceof KNNDirectIODirectory);
                assertEquals(INDEX_UUID, ((KNNDirectIODirectory) routed.directory).indexUuid());
                assertSame("the wrapped directory is the one the codec was handed", compound, FilterDirectory.unwrap(routed.directory));
                assertSame(state.segmentInfo, routed.segmentInfo);
                assertSame(state.fieldInfos, routed.fieldInfos);
                assertEquals(state.segmentSuffix, routed.segmentSuffix);
            }
        }
    }

    public void testRouteVectorDataReturnsTheCallersStateForAPathWithNoIndexUuid() throws IOException {
        try (Directory in = new ByteBuffersDirectory()) {
            assertNull("a directory with no path has no index UUID", KNNDirectIODirectory.indexUuidOf(in));
            final SegmentReadState state = new SegmentReadState(in, segmentInfo(in), FieldInfos.EMPTY, IOContext.DEFAULT, "x");
            assertSame(state, KNNDirectIODirectory.routeVectorData(state));
        }
    }

    public void testTheIndexIsResolvedOncePerSegment() throws IOException {
        DirectIOTestUtils.publishIndex(clusterService, INDEX_NAME, INDEX_UUID, true, null);
        try (Directory in = DirectIOTestUtils.fsDirectory(shardPath)) {
            final KNNDirectIODirectory directory = new KNNDirectIODirectory(in, INDEX_UUID);
            final Index first = directory.index();
            final Index second = directory.index();
            assertNotNull(first);
            assertSame("the Index is memoed", first, second);
            assertEquals(INDEX_NAME, first.getName());
            assertEquals(INDEX_UUID, first.getUUID());
        }
    }

    public void testHalfFloatFormatNamesMatchTheFormats() {
        assertEquals(KNNDirectIODirectory.HALF_FLOAT_FORMAT_NAME, new NativeEngines990HalfFloatKnnVectorsFormat().getName());
        assertEquals(KNNDirectIODirectory.HALF_FLOAT_SQ_FORMAT_NAME, new Faiss1040HalfFloatScalarQuantizedKnnVectorsFormat().getName());
    }

    public void testTheTwoFullPrecisionFormatNamesAreReadFromTheFormats() {
        assertEquals(NativeEngines990KnnVectorsFormat.FORMAT_NAME, new NativeEngines990KnnVectorsFormat().getName());
        assertEquals(Faiss1040ScalarQuantizedKnnVectorsFormat.FORMAT_NAME, new Faiss1040ScalarQuantizedKnnVectorsFormat().getName());
    }

    public void testMinFileSizeMatchesLucene() {
        assertEquals(DirectIODirectory.DEFAULT_MIN_BYTES_DIRECT, KNNDirectIODirectory.MIN_FILE_SIZE_BYTES);
        assertEquals(10L * 1024 * 1024, KNNDirectIODirectory.MIN_FILE_SIZE_BYTES);
    }

    public void testTheDirectOptionIsLookedUpOncePerJvm() throws Exception {
        final Field field = KNNDirectIODirectory.class.getDeclaredField("DIRECT_OPEN_OPTION");
        assertTrue("the option is a constant, not a lookup per call", Modifier.isStatic(field.getModifiers()));
        assertTrue(Modifier.isFinal(field.getModifiers()));
        if (DirectIOTestUtils.isDirectIOSupported()) {
            assertNotNull(KNNDirectIODirectory.DIRECT_OPEN_OPTION);
            assertEquals("DIRECT", KNNDirectIODirectory.DIRECT_OPEN_OPTION.toString());
        }
    }

    public void testBufferSizeForRandomAndSequentialReaders() {
        for (final int blockSize : new int[] { 512, 4096 }) {
            assertEquals("a random reader takes two blocks", 2 * blockSize, KNNDirectIODirectory.bufferSizeFor(false, blockSize));
            assertEquals(
                "a sequential reader takes Lucene's merge buffer",
                KNNDirectIODirectory.SEQUENTIAL_BUFFER_BYTES,
                KNNDirectIODirectory.bufferSizeFor(true, blockSize)
            );
            assertEquals(0, KNNDirectIODirectory.bufferSizeFor(true, blockSize) % blockSize);
            assertEquals(0, KNNDirectIODirectory.bufferSizeFor(false, blockSize) % blockSize);
        }
        // A block larger than the merge buffer still gets whole blocks.
        final int hugeBlock = 1 << 19;
        assertEquals(hugeBlock, KNNDirectIODirectory.bufferSizeFor(true, hugeBlock));
        assertTrue(KNNDirectIODirectory.isSequential(IOContext.DEFAULT.withHints(DataAccessHint.SEQUENTIAL)));
        assertFalse(KNNDirectIODirectory.isSequential(IOContext.DEFAULT));
        assertFalse(KNNDirectIODirectory.isSequential(null));
    }

    /**
     * A file that fails one of the checks {@code openInput} can make without a device read is the original
     * directory's own input, of the same class and carrying the same bytes, so an index that can never route costs
     * nothing. Written against the production surface both trees have, so that it compiles at HEAD and fails there on
     * this assertion (SPEC section 9.7).
     *
     * @param delegate  the directory the delegate wraps
     * @param directory the delegate
     * @param name      the file name
     * @throws IOException if either open or read fails
     */
    private void assertOriginalInput(final Directory delegate, final KNNDirectIODirectory directory, final String name) throws IOException {
        try (
            IndexInput fromDelegate = delegate.openInput(name, IOContext.DEFAULT);
            IndexInput fromKnn = directory.openInput(name, IOContext.DEFAULT)
        ) {
            assertEquals(
                "a file that cannot be routed must be the original directory's own input",
                fromDelegate.getClass(),
                fromKnn.getClass()
            );
            assertArrayEquals(allBytes(fromDelegate), allBytes(fromKnn));
        }
    }

    private static byte[] allBytes(final IndexInput in) throws IOException {
        final IndexInput clone = in.clone();
        clone.seek(0);
        final byte[] bytes = new byte[(int) clone.length()];
        clone.readBytes(bytes, 0, bytes.length);
        return bytes;
    }

    private static SegmentInfo segmentInfo(final Directory dir) {
        return new SegmentInfo(
            dir,
            Version.LATEST,
            Version.LATEST,
            "_0",
            1,
            false,
            false,
            Codec.getDefault(),
            Collections.emptyMap(),
            new byte[16],
            Map.of(),
            null
        );
    }

}

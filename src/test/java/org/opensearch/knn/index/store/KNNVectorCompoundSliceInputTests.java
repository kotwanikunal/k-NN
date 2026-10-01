/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import com.carrotsearch.randomizedtesting.annotations.ThreadLeakFilters;
import lombok.SneakyThrows;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.CompoundDirectory;
import org.apache.lucene.codecs.hnsw.FlatVectorScorerUtil;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.codecs.lucene104.Lucene104Codec;
import org.apache.lucene.codecs.lucene95.OffHeapFloatVectorValues;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.SegmentCommitInfo;
import org.apache.lucene.index.SegmentInfos;
import org.apache.lucene.index.TieredMergePolicy;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.DataAccessHint;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.FileDataHint;
import org.apache.lucene.store.FileTypeHint;
import org.apache.lucene.store.FilterDirectory;
import org.apache.lucene.store.FilterIndexInput;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.KNN990Codec.NativeEngines990KnnVectorsFormat;
import org.opensearch.knn.index.codec.util.UnitTestCodec;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.OpenOption;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.nio.channels.FileChannel;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * Whether a compound segment's {@code .vec} can be dispatched on, and whether the dispatch is confined to
 * the faiss/MOS rows.
 *
 * <p>Looking only at {@code Directory} says it cannot be: the {@code .vec} bytes are served from inside
 * the {@code .cfs} and {@code KNN80CompoundDirectory} is not a {@code FilterDirectory} to walk, so a
 * directory below never sees the name {@code .vec}. Since every freshly flushed segment is compound, that
 * limit would apply to the newest data in every index.
 *
 * <p>These tests show the limit was an artefact of stopping at {@code Directory}. Lucene's compound
 * reader opens the container on the outer directory — the plugin's — and hands each entry out as
 * {@code handle.slice(name, offset, length, context)}: the plugin's own {@code IndexInput} is given the
 * logical file name. The assertions here are made against Lucene's real {@code Lucene90CompoundReader}
 * over real flushed compound segments, not a stand-in, because a stand-in compound directory that opens
 * its entries from a nested directory is exactly what makes this read the wrong answer.
 *
 * <p>Two segments are used, and the pair is the point: one written by the plugin's native-engine format,
 * whose {@code .vec} must route, and one written by stock Lucene, whose {@code .vec} must not. The
 * discriminator is the per-field format name that {@code PerFieldKnnVectorsFormat} puts in the file name.
 *
 * <p>They also pin the two traps in wrapping an {@code IndexInput} at all, both of which are silent:
 * {@link FilterIndexInput} inherits a no-op {@code prefetch} and a per-{@code float} {@code readFloats},
 * and a production {@code FilterIndexInput} subclass is <em>not</em> unwrapped by
 * {@code unwrapOnlyTest}, so it displaces the SIMD scorer whether or not that was wanted.
 */
@ThreadLeakFilters(defaultFilters = true, filters = { DirectIOReadPoolTests.ReadPoolThreadFilter.class })
public class KNNVectorCompoundSliceInputTests extends KNNTestCase {

    private static final int DIMENSION = 32;
    private static final int DOCS = 64;
    private static final String VECTOR_FIELD = "v";

    /**
     * The plugin's native-engine format with the graph build skipped. A negative approximate threshold
     * makes {@code AbstractNativeEnginesKnnVectorsWriter.shouldSkipBuildingVectorDataStructure} answer
     * true for any doc count, so the segment holds the fp32 {@code .vec} and its {@code .vemf} sidecar
     * and no {@code .faiss} — which is all these tests read, and keeps the fixture off the JNI layer.
     */
    private static final Codec FAISS_CODEC = new UnitTestCodec(() -> new NativeEngines990KnnVectorsFormat(-1));

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

    /**
     * The minimum outer {@link Directory} these tests need: one that records the names it is asked to
     * open and wraps every {@code .cfs} in the class under test, keeping a handle on the container so the
     * slices taken out of it are observable.
     *
     * <p>It is a test harness rather than the production {@link KNNVectorStorageDirectory} on purpose.
     * Production keeps bounded counters and no per-slice record — which is the right trade on a shard's
     * real slice traffic and the wrong one here, where the assertion <em>is</em> the record. The
     * dispatch rule itself is pinned against the production directory in
     * {@link KNNVectorStorageDirectoryTests}; what these tests pin is the channel one level down.
     */
    private static final class ObservingDirectory extends FilterDirectory {

        private final List<String> opens = new CopyOnWriteArrayList<>();
        private final Map<String, KNNVectorCompoundSliceInput> containers = new ConcurrentHashMap<>();

        private ObservingDirectory(final Directory delegate) {
            super(delegate);
        }

        @Override
        public IndexInput openInput(final String name, final IOContext context) throws IOException {
            opens.add(name);
            final IndexInput input = in.openInput(name, context);
            if (name.endsWith(".cfs")) {
                // Exactly what KNNVectorStorageDirectory does, in its observing form: every entry of the
                // container is a four-argument slice of THIS input, carrying the name.
                final KNNVectorCompoundSliceInput wrapped = new KNNVectorCompoundSliceInput(input, name, resolvePath(name), true);
                containers.put(name, wrapped);
                return wrapped;
            }
            return input;
        }

        private Path resolvePath(final String name) {
            final Directory bottom = FilterDirectory.unwrap(in);
            return bottom instanceof FSDirectory fsDirectory ? fsDirectory.getDirectory().resolve(name) : null;
        }

        private List<String> opens() {
            return List.copyOf(opens);
        }

        private void clearOpens() {
            opens.clear();
        }

        private Map<String, KNNVectorCompoundSliceInput> compoundContainers() {
            return Map.copyOf(containers);
        }
    }

    private static float[] vector(final Random random) {
        final float[] values = new float[DIMENSION];
        for (int d = 0; d < DIMENSION; d++) {
            values[d] = random.nextFloat() * 20 - 10;
        }
        return values;
    }

    /** A native-engine vector field, so the per-field format name in the file name is the faiss one. */
    private static FieldType nativeEngineVectorField() {
        final FieldType fieldType = new FieldType();
        fieldType.setTokenized(false);
        fieldType.setIndexOptions(IndexOptions.NONE);
        fieldType.putAttribute(KNNVectorFieldMapper.KNN_FIELD, "true");
        fieldType.putAttribute(KNNConstants.KNN_METHOD, KNNConstants.METHOD_HNSW);
        fieldType.putAttribute(KNNConstants.KNN_ENGINE, KNNEngine.FAISS.getName());
        fieldType.putAttribute(KNNConstants.SPACE_TYPE, SpaceType.L2.getValue());
        fieldType.putAttribute(KNNConstants.HNSW_ALGO_M, "32");
        fieldType.putAttribute(KNNConstants.HNSW_ALGO_EF_CONSTRUCTION, "512");
        fieldType.putAttribute(KNNConstants.VECTOR_DATA_TYPE_FIELD, VectorDataType.FLOAT.getValue());
        fieldType.setVectorAttributes(DIMENSION, VectorEncoding.FLOAT32, VectorSimilarityFunction.EUCLIDEAN);
        fieldType.freeze();
        return fieldType;
    }

    /**
     * A real single-segment compound index with a float vector field. The codec is pinned rather than
     * taken from {@code Codec.getDefault()} because the Lucene test framework randomises that, and a
     * codec that does not write a {@code .vec} would make these tests pass for the wrong reason.
     */
    private static void writeCompoundSegmentWithVectors(final ObservingDirectory directory, final Codec codec, final FieldType fieldType)
        throws IOException {
        final IndexWriterConfig config = new IndexWriterConfig();
        config.setCodec(codec);
        config.setUseCompoundFile(true);
        final TieredMergePolicy mergePolicy = new TieredMergePolicy();
        // Without this a flushed segment large relative to the index is left non-compound, which is the
        // one thing this fixture must not be.
        mergePolicy.setNoCFSRatio(1.0);
        config.setMergePolicy(mergePolicy);

        final Random random = new Random(20260929L);
        try (IndexWriter writer = new IndexWriter(directory, config)) {
            for (int i = 0; i < DOCS; i++) {
                final Document document = new Document();
                document.add(new KnnFloatVectorField(VECTOR_FIELD, vector(random), fieldType));
                writer.addDocument(document);
            }
            writer.commit();
        }
    }

    /** The faiss fixture: a compound segment whose {@code .vec} the rule must route. */
    private static void writeFaissCompoundSegment(final ObservingDirectory directory) throws IOException {
        writeCompoundSegmentWithVectors(directory, FAISS_CODEC, nativeEngineVectorField());
    }

    /** The control fixture: a compound segment written by stock Lucene, whose {@code .vec} must not route. */
    private static void writeLuceneCompoundSegment(final ObservingDirectory directory) throws IOException {
        final FieldType fieldType = new FieldType(KnnFloatVectorField.createFieldType(DIMENSION, VectorSimilarityFunction.EUCLIDEAN));
        fieldType.freeze();
        writeCompoundSegmentWithVectors(directory, new Lucene104Codec(), fieldType);
    }

    private static SegmentCommitInfo onlySegment(final ObservingDirectory directory) throws IOException {
        final SegmentInfos infos = SegmentInfos.readLatestCommit(directory);
        assertEquals("the fixture must be one segment", 1, infos.size());
        return infos.info(0);
    }

    /** The {@code .vec} entry of a compound segment, by the only thing a slice description ever is. */
    private static String vecEntryName(final CompoundDirectory compound) throws IOException {
        final List<String> entries = List.of(compound.listAll());
        return entries.stream()
            .filter(name -> name.endsWith(".vec"))
            .findFirst()
            .orElseThrow(() -> new AssertionError("no .vec entry in " + entries));
    }

    private static Optional<KNNVectorCompoundSliceInput.SliceObservation> vecSlice(
        final List<KNNVectorCompoundSliceInput.SliceObservation> observations
    ) {
        return observations.stream().filter(o -> o.name().endsWith(".vec")).findFirst();
    }

    private static final IOContext VECTOR_DATA_CONTEXT = IOContext.DEFAULT.withHints(
        FileTypeHint.DATA,
        FileDataHint.KNN_VECTORS,
        DataAccessHint.RANDOM
    );

    /**
     * The channel. On a real compound segment the plugin's {@code IndexInput} for the container is handed
     * the {@code .vec} entry by name, with its extent — the signal that looking only at {@code openInput}
     * could not find is present one level down, at {@code slice}. And because the name is now the whole
     * signal, that one call is enough to decide: the faiss entry is routed on Lucene's own read of it.
     */
    public void testCompoundSegmentVecEntryArrivesAtThePluginByName() throws IOException {
        assumeDirectIOWorksHere();
        final Path path = createTempDir();
        try (ObservingDirectory outer = new ObservingDirectory(new MMapDirectory(path))) {
            writeFaissCompoundSegment(outer);
            final SegmentCommitInfo commit = onlySegment(outer);
            assertTrue("the fixture must be a compound segment", commit.info.getUseCompoundFile());

            outer.clearOpens();
            try (CompoundDirectory compound = commit.info.getCodec().compoundFormat().getCompoundReader(outer, commit.info)) {
                // The container itself is an ordinary openInput on the plugin's directory, which is what
                // makes the slices below reachable at all.
                assertTrue(
                    "the .cfs must be opened on the plugin directory, saw " + outer.opens(),
                    outer.opens().stream().anyMatch(name -> name.endsWith(".cfs"))
                );
                final KNNVectorCompoundSliceInput container = outer.compoundContainers()
                    .values()
                    .stream()
                    .findFirst()
                    .orElseThrow(() -> new AssertionError("the .cfs was not wrapped"));

                final String vecName = vecEntryName(compound);
                assertTrue(
                    "the entry name must carry the per-field format name, saw " + vecName,
                    vecName.contains(NativeEngines990KnnVectorsFormat.FORMAT_NAME)
                );

                container.clearSliceObservations();
                try (IndexInput entry = compound.openInput(vecName, VECTOR_DATA_CONTEXT)) {
                    assertNotNull(entry);
                    assertTrue("a faiss .vec entry routes on its name alone", entry instanceof DirectIOVectorIndexInput);
                }
                final KNNVectorCompoundSliceInput.SliceObservation observed = vecSlice(container.sliceObservations()).orElseThrow(
                    () -> new AssertionError("the .vec read did not reach the container, saw " + container.sliceObservations())
                );
                assertTrue("the four-argument slice is the channel", observed.carriedContext());
                assertEquals(
                    "the caller's hints arrive with the slice",
                    Set.of(FileTypeHint.DATA, FileDataHint.KNN_VECTORS, DataAccessHint.RANDOM),
                    observed.hints()
                );
                assertTrue("the entry's extent is given too", observed.offset() > 0 && observed.length() > 0);
                assertTrue(observed.routedToDirectIO());
                assertEquals(1, container.routedSlices());
            }
        }
    }

    /**
     * The control, and the one external behaviour change this design makes explicit: a stock Lucene
     * segment's {@code .vec} is <em>not</em> routed, so the Lucene engine stays on mmap. A routed file is
     * routed for all of its readers, and Lucene's unquantized traversal reads this file through its
     * memory-segment SIMD scorer, so routing it would be a regression on a path this feature is not for.
     */
    public void testALuceneSegmentVecEntryIsNotRouted() throws IOException {
        final Path path = createTempDir();
        try (ObservingDirectory outer = new ObservingDirectory(new MMapDirectory(path))) {
            writeLuceneCompoundSegment(outer);
            final SegmentCommitInfo commit = onlySegment(outer);
            assertTrue(commit.info.getUseCompoundFile());

            try (CompoundDirectory compound = commit.info.getCodec().compoundFormat().getCompoundReader(outer, commit.info)) {
                final KNNVectorCompoundSliceInput container = outer.compoundContainers()
                    .values()
                    .stream()
                    .findFirst()
                    .orElseThrow(() -> new AssertionError("the .cfs was not wrapped"));
                final String vecName = vecEntryName(compound);
                assertFalse(
                    "a stock Lucene entry must not carry a plugin format name, saw " + vecName,
                    vecName.contains(NativeEngines990KnnVectorsFormat.FORMAT_NAME)
                );

                container.clearSliceObservations();
                try (IndexInput entry = compound.openInput(vecName, VECTOR_DATA_CONTEXT)) {
                    assertFalse("a Lucene-engine .vec entry must stay on the delegate", entry instanceof DirectIOVectorIndexInput);
                    assertSame("and must not be wrapped at all", entry, FilterIndexInput.unwrap(entry));
                }
                assertEquals(0, container.routedSlices());
                assertEquals(0, container.declinedSlices());
            }
        }
    }

    /**
     * Having reached the entry, the wrapper can actually serve it with {@code O_DIRECT}, and the bytes are
     * the delegate's bytes. A dispatch that reaches the right region and returns different data would be
     * worse than no dispatch at all. The mmap control is the same entry read through a container with the
     * gate off, so the comparison is between the two routes and nothing else.
     */
    public void testRoutedVecEntryIsServedByDirectIOAndReadsIdenticalBytes() throws IOException {
        assumeDirectIOWorksHere();
        final Path path = createTempDir();
        try (ObservingDirectory outer = new ObservingDirectory(new MMapDirectory(path))) {
            writeFaissCompoundSegment(outer);
            final SegmentCommitInfo commit = onlySegment(outer);
            assertTrue(commit.info.getUseCompoundFile());

            try (CompoundDirectory compound = commit.info.getCodec().compoundFormat().getCompoundReader(outer, commit.info)) {
                final String vecName = vecEntryName(compound);
                final KNNVectorCompoundSliceInput container = outer.compoundContainers()
                    .values()
                    .stream()
                    .findFirst()
                    .orElseThrow(() -> new AssertionError("the .cfs was not wrapped"));
                final KNNVectorCompoundSliceInput.SliceObservation extent;
                try (IndexInput probe = compound.openInput(vecName, VECTOR_DATA_CONTEXT)) {
                    assertNotNull(probe);
                }
                extent = vecSlice(container.sliceObservations()).orElseThrow();

                // The same region, read both ways: the delegate's own slice of the .cfs, and the routed one.
                try (
                    IndexInput containerHandle = new MMapDirectory(path).openInput(commit.info.name + ".cfs", IOContext.DEFAULT);
                    IndexInput mmap = containerHandle.slice(vecName, extent.offset(), extent.length());
                    IndexInput routed = compound.openInput(vecName, VECTOR_DATA_CONTEXT)
                ) {
                    assertTrue(
                        "the faiss entry must be served by Direct I/O, got " + routed.getClass().getName(),
                        routed instanceof DirectIOVectorIndexInput
                    );
                    assertEquals("both views must span the same entry", mmap.length(), routed.length());

                    final int length = Math.toIntExact(mmap.length());
                    final byte[] expected = new byte[length];
                    final byte[] actual = new byte[length];
                    mmap.seek(0);
                    mmap.readBytes(expected, 0, length);
                    routed.seek(0);
                    routed.readBytes(actual, 0, length);
                    assertArrayEquals("Direct I/O must deliver the delegate's bytes", expected, actual);

                    // And out of order, which is the access pattern re-score actually has.
                    final Random random = new Random(11L);
                    for (int i = 0; i < 64; i++) {
                        final int position = random.nextInt(length - Float.BYTES);
                        mmap.seek(position);
                        routed.seek(position);
                        assertEquals("position " + position, mmap.readInt(), routed.readInt());
                    }
                }
            }
        }
    }

    /**
     * Every entry the rule does not name keeps the delegate's own input, unchanged and unwrapped. This is
     * what makes the wrapper's cost one virtual call per {@code slice} — once per values object — rather
     * than anything per read, and it is why the entries that must stay on mmap (the quantized codes, the
     * metadata sidecars, and every non-vector file in the segment) are not merely allowed to but forced
     * to.
     */
    public void testEveryOtherEntryKeepsTheDelegateInputUnwrapped() throws IOException {
        final Path path = createTempDir();
        try (ObservingDirectory outer = new ObservingDirectory(new MMapDirectory(path))) {
            writeFaissCompoundSegment(outer);
            final SegmentCommitInfo commit = onlySegment(outer);

            try (CompoundDirectory compound = commit.info.getCodec().compoundFormat().getCompoundReader(outer, commit.info)) {
                for (final String name : compound.listAll()) {
                    if (KNNVectorStorageDirectory.isFaissVectorData(name)) {
                        continue;
                    }
                    try (IndexInput entry = compound.openInput(name, IOContext.DEFAULT)) {
                        assertFalse(
                            name + " must not be wrapped",
                            entry instanceof KNNVectorCompoundSliceInput || entry instanceof DirectIOVectorIndexInput
                        );
                        assertSame("and must not be a filter of anything", entry, FilterIndexInput.unwrap(entry));
                    }
                }
            }
        }
    }

    /**
     * The first silent trap. {@link FilterIndexInput} overrides neither {@link IndexInput#prefetch} nor
     * {@code readFloats}, so a wrapper that forgets them drops the shipped prefetch path
     * ({@code PrefetchableFlatVectorScorer} → {@code PrefetchHelper} → {@code IndexInput.prefetch}) and
     * turns one bulk copy into four reads per byte. Neither changes a byte, so neither would be caught by
     * a correctness test — which is why they are asserted here by construction.
     */
    public void testFilterIndexInputDropsPrefetchAndBulkReadsUnlessOverridden() throws IOException {
        final Path path = createTempDir();
        final Path file = path.resolve("bytes");
        final byte[] content = new byte[4096];
        new Random(3L).nextBytes(content);
        Files.write(file, content);

        try (MMapDirectory directory = new MMapDirectory(path); IndexInput delegate = directory.openInput("bytes", IOContext.DEFAULT)) {
            final CountingInput counting = new CountingInput(delegate.clone());
            final FilterIndexInput naive = new FilterIndexInput("naive", counting);

            // A naive wrapper's prefetch reaches nothing.
            naive.prefetch(0, 1024);
            assertEquals("FilterIndexInput inherits IndexInput's no-op prefetch", 0, counting.prefetches);

            // And its readFloats is DataInput's loop, in terms of readByte.
            naive.seek(0);
            naive.readFloats(new float[16], 0, 16);
            assertEquals("a naive wrapper's readFloats is not delegated", 0, counting.bulkFloatReads);
            assertTrue("it becomes a per-byte loop instead, saw " + counting.byteReads, counting.byteReads >= 64);

            // The class under test delegates both.
            final CountingInput countingForWrapper = new CountingInput(delegate.clone());
            try (KNNVectorCompoundSliceInput wrapper = new KNNVectorCompoundSliceInput(countingForWrapper, "bytes", null, false)) {
                wrapper.prefetch(0, 1024);
                assertEquals(1, countingForWrapper.prefetches);
                wrapper.seek(0);
                wrapper.readFloats(new float[16], 0, 16);
                assertEquals(1, countingForWrapper.bulkFloatReads);
                assertEquals("and costs no per-byte reads at all", 0, countingForWrapper.byteReads);
            }
        }
    }

    /**
     * The second silent trap. {@code unwrapOnlyTest} unwraps only classes registered through
     * {@code TestSecrets}, whose setter Lucene restricts to its own test framework, so in production the
     * registry is empty and the method is the identity. A plugin {@link FilterIndexInput} subclass
     * therefore does <em>not</em> get seen through: it displaces the memory-segment SIMD scorer exactly
     * as a non-filter input would.
     *
     * <p>That is the reason this class returns the <em>raw</em> delegate slice for every entry it does not
     * route, and the reason the routing predicate is narrowed to the faiss rows: a routed file is routed
     * for all of its readers, so a file whose traversal depends on the SIMD scorer must not be named by
     * the rule in the first place.
     */
    public void testProductionFilterIndexInputIsNotUnwrappedAndDisplacesTheSimdScorer() throws IOException {
        final Path path = createTempDir();
        final Path file = path.resolve("vectors");
        final int count = 16;
        final Random random = new Random(5L);
        // Real finite floats, little-endian, exactly as Lucene's flat vector file holds them. Random bytes
        // would decode to NaN and infinity, which Lucene's own assertions in VectorUtil reject.
        final java.nio.ByteBuffer buffer = java.nio.ByteBuffer.allocate(count * DIMENSION * Float.BYTES)
            .order(java.nio.ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < count * DIMENSION; i++) {
            buffer.putFloat(random.nextFloat() * 20 - 10);
        }
        final byte[] content = buffer.array();
        Files.write(file, content);

        final FlatVectorsScorer scorer = FlatVectorScorerUtil.getLucene99FlatVectorsScorer();
        final float[] query = vector(new Random(6L));

        try (MMapDirectory directory = new MMapDirectory(path); IndexInput delegate = directory.openInput("vectors", IOContext.DEFAULT)) {
            final IndexInput wrapper = new KNNVectorCompoundSliceInput(delegate.clone(), "vectors", null, false);
            assertSame("a production FilterIndexInput subclass is not a test filter", wrapper, FilterIndexInput.unwrapOnlyTest(wrapper));
            assertNotSame("but the non-test unwrap does see through it", wrapper, FilterIndexInput.unwrap(wrapper));

            final OffHeapFloatVectorValues.DenseOffHeapVectorValues mmapValues = new OffHeapFloatVectorValues.DenseOffHeapVectorValues(
                DIMENSION,
                count,
                delegate.slice("vector-data", 0, content.length),
                DIMENSION * Float.BYTES,
                scorer,
                VectorSimilarityFunction.EUCLIDEAN
            );
            final OffHeapFloatVectorValues.DenseOffHeapVectorValues wrappedValues = new OffHeapFloatVectorValues.DenseOffHeapVectorValues(
                DIMENSION,
                count,
                new KNNVectorCompoundSliceInput(delegate.slice("vector-data", 0, content.length), "vectors", null, false),
                DIMENSION * Float.BYTES,
                scorer,
                VectorSimilarityFunction.EUCLIDEAN
            );

            final RandomVectorScorer mmapScorer = scorer.getRandomVectorScorer(VectorSimilarityFunction.EUCLIDEAN, mmapValues, query);
            final RandomVectorScorer wrappedScorer = scorer.getRandomVectorScorer(VectorSimilarityFunction.EUCLIDEAN, wrappedValues, query);

            assertTrue(
                "the mmap input binds the memory-segment scorer, got " + mmapScorer.getClass().getName(),
                mmapScorer.getClass().getName().contains("MemorySegment")
            );
            assertFalse(
                "a wrapped input must not, got " + wrappedScorer.getClass().getName(),
                wrappedScorer.getClass().getName().contains("MemorySegment")
            );

            // Displaced, but not different: both decode the same fp32 bytes.
            for (int ord = 0; ord < count; ord++) {
                assertEquals("ordinal " + ord, mmapScorer.score(ord), wrappedScorer.score(ord), 0.0f);
            }
            wrapper.close();
        }
    }

    /**
     * A clone must not share this object's delegate. {@link IndexInput#clone} is {@code Object.clone},
     * so an un-overridden clone of a delegating input would move the original's file pointer when it
     * seeks — and {@code CodecUtil.checksumEntireFile} clones a {@code .cfs} handle and seeks it to zero
     * on a real segment open, so this is reached in practice rather than in theory.
     */
    public void testCloneDoesNotShareTheDelegateCursor() throws IOException {
        final Path path = createTempDir();
        final byte[] content = new byte[256];
        for (int i = 0; i < content.length; i++) {
            content[i] = (byte) i;
        }
        Files.write(path.resolve("bytes"), content);

        try (MMapDirectory directory = new MMapDirectory(path); IndexInput delegate = directory.openInput("bytes", IOContext.DEFAULT)) {
            try (KNNVectorCompoundSliceInput wrapper = new KNNVectorCompoundSliceInput(delegate.clone(), "bytes", null, false)) {
                wrapper.seek(100);
                final IndexInput clone = wrapper.clone();
                clone.seek(0);
                assertEquals("the clone reads from its own position", 0, clone.readByte());
                assertEquals("and the original keeps its own", 100, wrapper.readByte());
            }
        }
    }

    /**
     * The three-argument {@code slice} is not a dispatch point, and that is recorded rather than argued.
     * Lucene's compound reader serves every entry through the four-argument overload, so a three-argument
     * slice is always a sub-slice of an entry already dispatched — and routing one would hand a second
     * {@code O_DIRECT} handle to a caller that already has the right input.
     */
    public void testThreeArgumentSliceIsNotADispatchPoint() throws IOException {
        final Path path = createTempDir();
        Files.write(path.resolve("bytes"), new byte[256]);

        try (MMapDirectory directory = new MMapDirectory(path); IndexInput delegate = directory.openInput("bytes", IOContext.DEFAULT)) {
            try (KNNVectorCompoundSliceInput wrapper = new KNNVectorCompoundSliceInput(delegate.clone(), "bytes", null, true)) {
                wrapper.slice("_0_" + NativeEngines990KnnVectorsFormat.FORMAT_NAME + "_0.vec", 0, 128).close();
                final KNNVectorCompoundSliceInput.SliceObservation observation = vecSlice(wrapper.sliceObservations()).orElseThrow();
                assertFalse("no IOContext, so not the dispatch channel", observation.carriedContext());
                assertFalse(observation.routedToDirectIO());
                assertEquals(0, wrapper.routedSlices());
            }
        }
    }

    /**
     * The queue depth survives being reached through a compound-file offset.
     *
     * <p>A routed entry is a {@link DirectIOVectorIndexInput#slice} at the entry's offset inside the
     * {@code .cfs} — an offset that is neither zero nor block aligned (1088 in the segment above). What
     * the block-layer measurements did not show is that the staging still works when every position is
     * shifted by an arbitrary amount, which is the only thing the compound route changes. So: a
     * 64-ordinal burst through the entry's own slice, and both the staged hits and a queue depth greater
     * than one.
     */
    public void testStagingSurvivesACompoundEntryOffset() throws IOException {
        assumeDirectIOWorksHere();
        final int dimension = 768;
        final int vectorBytes = dimension * Float.BYTES;
        final int vectors = 600;
        // The .vec entry offset observed in a real compound segment: not zero, not a multiple of any
        // block size.
        final long entryOffset = 1088;

        final Path path = createTempDir();
        final Path file = path.resolve("container.cfs");
        final java.nio.ByteBuffer buffer = java.nio.ByteBuffer.allocate(Math.toIntExact(entryOffset) + vectors * vectorBytes + 16)
            .order(java.nio.ByteOrder.LITTLE_ENDIAN);
        final Random random = new Random(20260929L);
        for (long i = 0; i < entryOffset; i++) {
            buffer.put((byte) 0xA5);
        }
        for (int i = 0; i < vectors * dimension; i++) {
            buffer.putFloat(random.nextFloat() * 20 - 10);
        }
        Files.write(file, buffer.array());

        DirectIOReadPool.resetForTesting();
        DirectIOVectorIndexInput.STATS.reset();
        try (DirectIOVectorIndexInput container = DirectIOVectorIndexInput.open(file, 0, 64, 132 * 1024)) {
            assertNotNull(container);
            // Exactly what KNNVectorCompoundSliceInput hands back for a routed entry.
            final IndexInput entry = container.slice(
                "_0_" + NativeEngines990KnnVectorsFormat.FORMAT_NAME + "_0.vec",
                entryOffset,
                (long) vectors * vectorBytes
            );

            final int[] ords = new int[64];
            for (int i = 0; i < ords.length; i++) {
                ords[i] = i * 9;
            }
            for (final int ord : ords) {
                entry.prefetch((long) ord * vectorBytes, vectorBytes);
            }
            final float[] destination = new float[dimension];
            for (final int ord : ords) {
                entry.seek((long) ord * vectorBytes);
                entry.readFloats(destination, 0, dimension);
                // The bytes are the file's, shifted by the entry offset -- the thing an offset could break.
                final int expectedIndex = Math.toIntExact(entryOffset) + ord * vectorBytes;
                assertEquals(
                    "ordinal " + ord,
                    java.nio.ByteBuffer.wrap(buffer.array(), expectedIndex, Float.BYTES).order(java.nio.ByteOrder.LITTLE_ENDIAN).getFloat(),
                    destination[0],
                    0.0f
                );
            }

            logger.info("staging through a compound entry offset: {}", DirectIOVectorIndexInput.STATS);
            assertTrue("reads must be served from staged ranges", DirectIOVectorIndexInput.STATS.stagedHits() > 0);
            assertTrue(
                "the queue depth must survive the offset, saw " + DirectIOVectorIndexInput.STATS.inFlightPeak(),
                DirectIOVectorIndexInput.STATS.inFlightPeak() > 1
            );
        }
    }

    /** Counts what a wrapper does or does not delegate. */
    private static final class CountingInput extends FilterIndexInput {
        private int prefetches;
        private int bulkFloatReads;
        private int byteReads;

        private CountingInput(final IndexInput delegate) {
            super("CountingInput", delegate);
        }

        @Override
        public void prefetch(final long offset, final long length) {
            prefetches++;
        }

        @Override
        public void readFloats(final float[] destination, final int offset, final int length) throws IOException {
            bulkFloatReads++;
            in.readFloats(destination, offset, length);
        }

        @Override
        public byte readByte() throws IOException {
            byteReads++;
            return in.readByte();
        }

        @Override
        public IndexInput clone() {
            throw new UnsupportedOperationException("not needed by these tests");
        }
    }
}

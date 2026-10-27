/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import com.carrotsearch.randomizedtesting.annotations.ThreadLeakFilters;
import lombok.SneakyThrows;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.lucene95.HasIndexSlice;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.MMapDirectory;
import org.opensearch.common.settings.ClusterSettings;
import org.opensearch.common.settings.Setting;
import org.opensearch.common.settings.Settings;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.KNN990Codec.NativeEngines990HalfFloatKnnVectorsFormat;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexBuildStrategyFactory;
import org.opensearch.knn.index.codec.scorer.PrefetchHelper;
import org.opensearch.knn.index.codec.util.UnitTestCodec;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

import static org.mockito.Mockito.when;

/**
 * O8's identity proof, end to end through the real codec with nothing mocked: a {@code half_float} native
 * engine segment has its {@code .vec} served with {@code O_DIRECT}, the vectors that come back are
 * bit-for-bit the memory-mapped reader's, and the slab store serves and admits them.
 *
 * <p>This is the FP16 counterpart of {@link FaissVectorDataIdentityTests} and it is the test that would have
 * failed before O8 in <b>two</b> different ways. The dispatch rule asks for the per-field format's name in
 * the file name, and a half-float field's is {@code NativeEngines990HalfFloatKnnVectorsFormat}, which does
 * not contain the fp32 name it extends, so the file was declined at condition 1 and {@code routedOpens} was
 * 0. And even once routed, the FP16 values read a vector with
 * {@code KNN1040HalfFloatFlatVectorsValues#readRawVectorBytes}, which is {@code readBytes} and not
 * {@code readFloats}, so the store had no serve site on that path: the second test below would have found
 * zero hits and zero entries while every byte was still correct.
 */
@ThreadLeakFilters(defaultFilters = true, filters = { DirectIOReadPoolTests.ReadPoolThreadFilter.class })
public class HalfFloatVectorDataIdentityTests extends KNNTestCase {

    private static final int DIMENSION = 32;
    private static final int DOCS = 64;
    private static final String FIELD = "v";
    private static final int TEST_SLAB_BYTES = 64 * 1024;

    /**
     * The plugin's FP16 native-engine format with the graph build skipped, exactly as
     * {@link FaissVectorDataIdentityTests} does for fp32: a negative approximate threshold makes
     * {@code AbstractNativeEnginesKnnVectorsWriter.shouldSkipBuildingVectorDataStructure} answer true for any
     * doc count, so the segment holds the FP16 {@code .vec} and its {@code .vemf} sidecar and no
     * {@code .faiss}, which keeps the fixture off the JNI layer.
     *
     * <p>It is this <b>per-field</b> format and not the nested {@code KNN1040HalfFloatFlatVectorsFormat} that
     * matters here: {@code PerFieldKnnVectorsFormat} builds the segment suffix from the per-field format's
     * name, so this is the name the dispatch rule has to recognise.
     */
    private static final Codec HALF_FLOAT_CODEC = new UnitTestCodec(
        () -> new NativeEngines990HalfFloatKnnVectorsFormat(-1, new NativeIndexBuildStrategyFactory())
    );

    @Override
    public void tearDown() throws Exception {
        DirectIOVectorCache.resetForTesting();
        DirectIOReadPool.resetForTesting();
        DirectIOVectorIndexInput.STATS.reset();
        super.tearDown();
    }

    // -----------------------------------------------------------------------------------------------

    /** The route and the bytes: routed once, and identical vectors on either channel. */
    @SneakyThrows
    public void testHalfFloatVectorValuesAreRoutedAndIdenticalToMmap() {
        assumeDirectIOWorksHere();
        configure(0);
        final Path path = createTempDir();
        try (MMapDirectory delegate = new MMapDirectory(path)) {
            writeSegment(delegate);

            final List<float[]> expected;
            try (DirectoryReader mmapReader = DirectoryReader.open(delegate)) {
                expected = readAll(onlyLeafValues(mmapReader));
            }
            assertEquals(DOCS, expected.size());

            try (KNNDirectIODirectory storage = new KNNDirectIODirectory(delegate, "_0", () -> true, () -> 0L)) {
                final List<float[]> actual;
                try (DirectoryReader routedReader = DirectoryReader.open(storage)) {
                    assertEquals(
                        "the half-float .vec should have been served with O_DIRECT, declined=" + storage.declinedOpens(),
                        1,
                        storage.routedOpens()
                    );
                    assertEquals(0, storage.declinedOpens());
                    actual = readAll(onlyLeafValues(routedReader));
                }
                assertEquals(expected.size(), actual.size());
                for (int ordinal = 0; ordinal < expected.size(); ordinal++) {
                    assertArrayEquals(
                        "ordinal " + ordinal + " differs between the mmap reader and the O_DIRECT reader",
                        expected.get(ordinal),
                        actual.get(ordinal),
                        0.0f
                    );
                }
            }
        }
    }

    /**
     * And the store, on the path the scorer actually takes: the prefetch helper is handed the routed values'
     * own slice, which declares the field's vector grid, and the reads that follow are served and admitted
     * through {@code readBytes}.
     *
     * <p>The vector byte length assertion is the one that says this is really the FP16 layout and not an fp32
     * file read by accident: it is {@code 2 x dimension}, and it is also the store's slot length, which is why
     * the store needed no new key or layout for FP16.
     */
    @SneakyThrows
    public void testTheStoreServesAndAdmitsTheHalfFloatPath() {
        assumeDirectIOWorksHere();
        configure(8);
        final Path path = createTempDir();
        try (MMapDirectory delegate = new MMapDirectory(path)) {
            writeSegment(delegate);

            final List<float[]> expected;
            try (DirectoryReader mmapReader = DirectoryReader.open(delegate)) {
                expected = readAll(onlyLeafValues(mmapReader));
            }

            try (
                KNNDirectIODirectory storage = new KNNDirectIODirectory(delegate, "_0", () -> true, () -> 0L);
                DirectoryReader routedReader = DirectoryReader.open(storage)
            ) {
                assertEquals(1, storage.routedOpens());
                final FloatVectorValues values = onlyLeafValues(routedReader);
                final IndexInput slice = ((HasIndexSlice) values).getSlice();
                assertTrue("the values' slice must be the Direct I/O input", slice instanceof DirectIOVectorIndexInput);
                assertEquals("two bytes a dimension, which is the whole point", DIMENSION * Short.BYTES, values.getVectorByteLength());

                final int[] batch = new int[DOCS];
                for (int ordinal = 0; ordinal < DOCS; ordinal++) {
                    batch[ordinal] = ordinal;
                }
                // Exactly the call PrefetchableVectorValuesHelper#doPrefetch makes for these values.
                PrefetchHelper.prefetch(slice, 0, values.getVectorByteLength(), batch.clone(), batch.length);
                final List<float[]> firstPass = readByOrdinal(values);

                assertTrue("the FP16 reads were admitted", DirectIOVectorCache.entries() > 0);
                final long hitsAfterFirst = DirectIOVectorCache.hits();

                // A second burst over the same candidates: the filter now drops them all, so nothing is
                // staged and every read is served from the store.
                final long stagedAfterFirst = DirectIOVectorIndexInput.STATS.stagedReads();
                PrefetchHelper.prefetch(slice, 0, values.getVectorByteLength(), batch.clone(), batch.length);
                final List<float[]> secondPass = readByOrdinal(values);

                assertTrue("the second pass was served from the store", DirectIOVectorCache.hits() > hitsAfterFirst);
                assertEquals("and needed no further staged read", stagedAfterFirst, DirectIOVectorIndexInput.STATS.stagedReads());

                for (int ordinal = 0; ordinal < DOCS; ordinal++) {
                    assertArrayEquals("miss path, ordinal " + ordinal, expected.get(ordinal), firstPass.get(ordinal), 0.0f);
                    assertArrayEquals("store hit, ordinal " + ordinal, expected.get(ordinal), secondPass.get(ordinal), 0.0f);
                }
                DirectIOVectorCache.checkInvariantsForTesting();
            }
        }
    }

    // -----------------------------------------------------------------------------------------------

    @SneakyThrows
    private void assumeDirectIOWorksHere() {
        final Path probe = createTempDir().resolve("probe");
        Files.write(probe, new byte[8192]);
        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(probe)) {
            assumeTrue("O_DIRECT is not available here", input != null);
        }
    }

    private static float[] vector(final Random random) {
        final float[] values = new float[DIMENSION];
        for (int d = 0; d < DIMENSION; d++) {
            values[d] = random.nextFloat() * 20 - 10;
        }
        return values;
    }

    /** A {@code half_float} native-engine vector field, so the per-field format is the FP16 one. */
    private static FieldType halfFloatVectorField() {
        final FieldType fieldType = new FieldType();
        fieldType.setTokenized(false);
        fieldType.setIndexOptions(IndexOptions.NONE);
        fieldType.putAttribute(KNNVectorFieldMapper.KNN_FIELD, "true");
        fieldType.putAttribute(KNNConstants.KNN_METHOD, KNNConstants.METHOD_HNSW);
        fieldType.putAttribute(KNNConstants.KNN_ENGINE, KNNEngine.FAISS.getName());
        fieldType.putAttribute(KNNConstants.SPACE_TYPE, SpaceType.L2.getValue());
        fieldType.putAttribute(KNNConstants.HNSW_ALGO_M, "32");
        fieldType.putAttribute(KNNConstants.HNSW_ALGO_EF_CONSTRUCTION, "512");
        fieldType.putAttribute(KNNConstants.VECTOR_DATA_TYPE_FIELD, VectorDataType.HALF_FLOAT.getValue());
        fieldType.setVectorAttributes(DIMENSION, VectorEncoding.FLOAT32, VectorSimilarityFunction.EUCLIDEAN);
        fieldType.freeze();
        return fieldType;
    }

    /** One non-compound segment, so the {@code .vec} arrives at the directory's own {@code openInput}. */
    private static void writeSegment(final MMapDirectory directory) throws IOException {
        final IndexWriterConfig config = new IndexWriterConfig();
        config.setCodec(HALF_FLOAT_CODEC);
        config.setUseCompoundFile(false);
        config.setMergePolicy(NoMergePolicy.INSTANCE);

        final FieldType fieldType = halfFloatVectorField();
        final Random random = new Random(20261014L);
        try (IndexWriter writer = new IndexWriter(directory, config)) {
            for (int i = 0; i < DOCS; i++) {
                final Document document = new Document();
                document.add(new KnnFloatVectorField(FIELD, vector(random), fieldType));
                writer.addDocument(document);
            }
            writer.commit();
        }
    }

    private static List<float[]> readAll(final FloatVectorValues values) throws IOException {
        final List<float[]> vectors = new ArrayList<>();
        final KnnVectorValues.DocIndexIterator iterator = values.iterator();
        for (int doc = iterator.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = iterator.nextDoc()) {
            vectors.add(values.vectorValue(iterator.index()).clone());
        }
        return vectors;
    }

    /** Every ordinal in order, which is how the scorer consumes a sorted batch. */
    private static List<float[]> readByOrdinal(final FloatVectorValues values) throws IOException {
        final List<float[]> vectors = new ArrayList<>(DOCS);
        for (int ordinal = 0; ordinal < DOCS; ordinal++) {
            vectors.add(values.vectorValue(ordinal).clone());
        }
        return vectors;
    }

    private static FloatVectorValues onlyLeafValues(final DirectoryReader reader) throws IOException {
        assertEquals("the fixture must be one segment", 1, reader.leaves().size());
        final LeafReader leaf = reader.leaves().get(0).reader();
        final FloatVectorValues values = leaf.getFloatVectorValues(FIELD);
        assertNotNull(values);
        return values;
    }

    private void withNodeSettings(final Settings settings) {
        final Set<Setting<?>> registered = new HashSet<>(ClusterSettings.BUILT_IN_CLUSTER_SETTINGS);
        registered.addAll(
            KNNSettings.state()
                .getSettings()
                .stream()
                .filter(setting -> setting.getProperties().contains(Setting.Property.NodeScope))
                .toList()
        );
        when(clusterService.getClusterSettings()).thenReturn(new ClusterSettings(settings, registered));
        KNNSettings.state().setClusterService(clusterService);
    }

    private void configure(final int slabs) {
        final long maxBytes = (long) slabs * TEST_SLAB_BYTES;
        withNodeSettings(
            Settings.builder()
                .put(KNNSettings.KNN_DIRECT_IO_RESCORE_CACHE_MAX_BYTES, maxBytes + "b")
                .put(KNNSettings.KNN_DIRECT_IO_READAHEAD_WINDOW, 0)
                .build()
        );
        DirectIOVectorCache.resetForTesting();
        DirectIOVectorCache.setSlabBytesForTesting(TEST_SLAB_BYTES);
        DirectIOVectorCache.configure(maxBytes);
        DirectIOVectorIndexInput.STATS.reset();
    }
}

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import com.carrotsearch.randomizedtesting.annotations.ThreadLeakFilters;
import lombok.SneakyThrows;
import org.apache.lucene.codecs.Codec;
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
import org.apache.lucene.search.VectorScorer;
import org.apache.lucene.store.MMapDirectory;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.KNN990Codec.NativeEngines990KnnVectorsFormat;
import org.opensearch.knn.index.codec.util.UnitTestCodec;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;
import org.opensearch.knn.index.query.scorers.VectorScorerMode;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * The identity proof for the design, end to end, with nothing mocked between the codec and the kernel:
 * a real native-engine segment read through a {@link KNNVectorStorageDirectory} has its {@code .vec}
 * served with {@code O_DIRECT}, and the vectors and the scores that come back are bit-for-bit the ones
 * the ordinary memory-mapped reader returns.
 *
 * <p>This is the assertion that could actually fail, and the only one in the suite where the two sides
 * read the same file through different system calls. It is made on the <em>base</em>
 * {@code getFloatVectorValues} path rather than on a second, selectively-tagged view of the file,
 * because after this phase there is no second view: the file name is the whole dispatch signal, so the
 * object the rescore path holds is the object every reader of that segment holds.
 *
 * <p>Both halves are asserted. The counter, because identical bytes are also exactly what a silently
 * un-routed read returns, so an identity assertion alone would pass if the route were never taken; and
 * the values, because a route that reached the right file and returned different data would be worse
 * than no route at all.
 */
@ThreadLeakFilters(defaultFilters = true, filters = { DirectIOReadPoolTests.ReadPoolThreadFilter.class })
public class FaissVectorDataIdentityTests extends KNNTestCase {

    private static final int DIMENSION = 32;
    private static final int DOCS = 64;
    private static final String FIELD = "v";

    /**
     * The plugin's native-engine format with the graph build skipped. A negative approximate threshold
     * makes {@code AbstractNativeEnginesKnnVectorsWriter.shouldSkipBuildingVectorDataStructure} answer
     * true for any doc count, so the segment holds the fp32 {@code .vec} and its {@code .vemf} sidecar
     * and no {@code .faiss} — which is all this proof reads, and keeps the fixture off the JNI layer.
     */
    private static final Codec FAISS_CODEC = new UnitTestCodec(() -> new NativeEngines990KnnVectorsFormat(-1));

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
     * One non-compound segment, so the {@code .vec} arrives at the directory's own {@code openInput}
     * rather than as a slice of a {@code .cfs} — the compound dispatch point is proven separately in
     * {@link KNNVectorCompoundSliceInputTests}.
     */
    private static void writeSegment(final MMapDirectory directory) throws IOException {
        final IndexWriterConfig config = new IndexWriterConfig();
        config.setCodec(FAISS_CODEC);
        config.setUseCompoundFile(false);
        config.setMergePolicy(NoMergePolicy.INSTANCE);

        final FieldType fieldType = nativeEngineVectorField();
        final Random random = new Random(20261001L);
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

    private static FloatVectorValues onlyLeafValues(final DirectoryReader reader) throws IOException {
        assertEquals("the fixture must be one segment", 1, reader.leaves().size());
        final LeafReader leaf = reader.leaves().get(0).reader();
        final FloatVectorValues values = leaf.getFloatVectorValues(FIELD);
        assertNotNull(values);
        return values;
    }

    /** The vectors, read both ways. */
    @SneakyThrows
    public void testBaseFloatVectorValuesAreRoutedAndIdenticalToMmap() {
        assumeDirectIOWorksHere();
        final Path path = createTempDir();
        try (MMapDirectory delegate = new MMapDirectory(path)) {
            writeSegment(delegate);

            final List<float[]> expected;
            try (DirectoryReader mmapReader = DirectoryReader.open(delegate)) {
                expected = readAll(onlyLeafValues(mmapReader));
            }
            assertEquals(DOCS, expected.size());

            try (KNNVectorStorageDirectory storage = new KNNVectorStorageDirectory(delegate, "test-index", () -> true)) {
                final List<float[]> actual;
                try (DirectoryReader routedReader = DirectoryReader.open(storage)) {
                    assertEquals(
                        "the segment's .vec should have been served with O_DIRECT, declined=" + storage.declinedOpens(),
                        1,
                        storage.routedOpens()
                    );
                    assertEquals(0, storage.declinedOpens());
                    actual = readAll(onlyLeafValues(routedReader));
                }

                assertEquals(expected.size(), actual.size());
                for (int ord = 0; ord < expected.size(); ord++) {
                    assertArrayEquals(
                        "ordinal " + ord + " differs between the mmap reader and the O_DIRECT reader",
                        expected.get(ord),
                        actual.get(ord),
                        0.0f
                    );
                }
            }
        }
    }

    /**
     * And the scores, through the mode the query path actually uses.
     * {@link VectorScorerMode#RESCORE} is {@code values.rescorer(target)}, so this is Lucene's own scorer
     * over the same fp32 bytes arriving by two different mechanisms — exact equality, not a delta.
     */
    @SneakyThrows
    public void testRescoreScoresAreIdenticalToMmap() {
        assumeDirectIOWorksHere();
        final Path path = createTempDir();
        try (MMapDirectory delegate = new MMapDirectory(path)) {
            writeSegment(delegate);
            final float[] target = vector(new Random(7L));

            try (
                MMapDirectory control = new MMapDirectory(path);
                DirectoryReader mmapReader = DirectoryReader.open(control);
                KNNVectorStorageDirectory storage = new KNNVectorStorageDirectory(delegate, "test-index", () -> true);
                DirectoryReader routedReader = DirectoryReader.open(storage)
            ) {
                assertEquals(1, storage.routedOpens());

                final VectorScorer expected = VectorScorerMode.RESCORE.createScorer(onlyLeafValues(mmapReader), target);
                final VectorScorer actual = VectorScorerMode.RESCORE.createScorer(onlyLeafValues(routedReader), target);
                assertNotNull(expected);
                assertNotNull(actual);

                final DocIdSetIterator expectedDocs = expected.iterator();
                final DocIdSetIterator actualDocs = actual.iterator();
                int scored = 0;
                for (int doc = expectedDocs.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = expectedDocs.nextDoc()) {
                    assertEquals(doc, actualDocs.nextDoc());
                    assertEquals("doc " + doc, expected.score(), actual.score(), 0.0f);
                    scored++;
                }
                assertEquals(DOCS, scored);
                assertEquals(DocIdSetIterator.NO_MORE_DOCS, actualDocs.nextDoc());
            }
        }
    }

    /**
     * The gate off is the default on every node, and then the segment reads exactly as it did before the
     * directory was installed: nothing routed, and the same vectors.
     */
    @SneakyThrows
    public void testWithTheGateOffTheSegmentIsReadAsBefore() {
        final Path path = createTempDir();
        try (MMapDirectory delegate = new MMapDirectory(path)) {
            writeSegment(delegate);

            final List<float[]> expected;
            try (DirectoryReader mmapReader = DirectoryReader.open(delegate)) {
                expected = readAll(onlyLeafValues(mmapReader));
            }

            try (KNNVectorStorageDirectory storage = new KNNVectorStorageDirectory(delegate, "test-index", () -> false)) {
                try (DirectoryReader reader = DirectoryReader.open(storage)) {
                    assertEquals(0, storage.routedOpens());
                    assertEquals("declining on the gate is not a Direct I/O failure", 0, storage.declinedOpens());
                    final List<float[]> actual = readAll(onlyLeafValues(reader));
                    assertEquals(expected.size(), actual.size());
                    for (int ord = 0; ord < expected.size(); ord++) {
                        assertArrayEquals("ordinal " + ord, expected.get(ord), actual.get(ord), 0.0f);
                    }
                }
            }
        }
    }
}

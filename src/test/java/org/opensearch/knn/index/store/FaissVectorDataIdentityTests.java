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
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FilterDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.MMapDirectory;
import org.opensearch.common.lucene.store.ByteArrayIndexInput;
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
 * a real native-engine segment read through a {@link KNNDirectIODirectory} has its {@code .vec}
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
     * One non-compound segment, so the {@code .vec} arrives at the directory's own {@code openInput} —
     * the only dispatch point there is. A compound segment declines by construction, which is asserted in
     * {@link KNNDirectIOCodecSeamTests#testACompoundSegmentDeclinesAndStillReadsTheRightBytes}.
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

            try (KNNDirectIODirectory storage = new KNNDirectIODirectory(delegate, "_0", () -> true, () -> 0L)) {
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
                KNNDirectIODirectory storage = new KNNDirectIODirectory(delegate, "_0", () -> true, () -> 0L);
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
     * A byte-transforming layer over a <b>real</b> segment: W2's content decline, asserted where it
     * matters, which is the answers rather than the counter.
     *
     * <p>The fixture is the shape of encryption at rest. The segment is written normally, its {@code .vec}
     * is then rewritten on disk in transformed form, and the layer transforms it back on read — so a reader
     * of the chain sees the right vectors while the file holds something else. The path resolution walks
     * past the layer (W2 replaced the allowlist with {@link FilterDirectory#unwrap}), so without the byte
     * check the {@code O_DIRECT} route would read the file and score transformed bytes.
     *
     * <p>Both halves are asserted: the open declines with {@code CONTENT}, and the vectors and the scores
     * equal the ones the same chain returns with the feature off, which is the only comparison that says
     * the decline cost correctness nothing.
     */
    @SneakyThrows
    public void testAByteTransformingLayerDeclinesWithContentAndScoresTheSameAsWithTheFeatureOff() {
        assumeDirectIOWorksHere();
        final Path path = createTempDir();
        try (MMapDirectory delegate = new MMapDirectory(path)) {
            writeSegment(delegate);
            final String vectorData = onlyVectorDataFileName(path);
            // The on-disk form a transforming layer would have stored. From here the file's bytes are not
            // the bytes any reader of the chain sees, which is the premise of the whole test.
            final byte[] plain = Files.readAllBytes(path.resolve(vectorData));
            Files.write(path.resolve(vectorData), xor(plain));

            final float[] target = vector(new Random(7L));
            try (XorOnReadDirectory transforming = new XorOnReadDirectory(delegate)) {
                // The control: the same chain with the feature off, which is what a node runs today.
                final List<float[]> expectedVectors;
                final List<Float> expectedScores;
                final KNNDirectIODirectory off = new KNNDirectIODirectory(transforming, "_0", () -> false, () -> 0L);
                try (DirectoryReader reader = DirectoryReader.open(off)) {
                    expectedVectors = readAll(onlyLeafValues(reader));
                    expectedScores = scoreAll(onlyLeafValues(reader), target);
                }
                assertEquals(DOCS, expectedVectors.size());
                assertEquals(0, off.routedOpens());

                final KNNDirectIODirectory on = new KNNDirectIODirectory(transforming, "_0", () -> true, () -> 0L);
                try (DirectoryReader reader = DirectoryReader.open(on)) {
                    assertEquals("the transforming layer must not be served with O_DIRECT", 0, on.routedOpens());
                    assertEquals("and the reason must be the bytes", 1, on.declinedContent());
                    final List<float[]> actualVectors = readAll(onlyLeafValues(reader));
                    final List<Float> actualScores = scoreAll(onlyLeafValues(reader), target);
                    assertEquals(expectedVectors.size(), actualVectors.size());
                    for (int ord = 0; ord < expectedVectors.size(); ord++) {
                        assertArrayEquals("ordinal " + ord, expectedVectors.get(ord), actualVectors.get(ord), 0.0f);
                    }
                    assertEquals("0 differing score pairs, or the decline cost a correct answer", expectedScores, actualScores);
                }
            }
        }
    }

    /** The one {@code .vec} the fixture's segment holds. */
    @SneakyThrows
    private static String onlyVectorDataFileName(final Path path) {
        final List<String> names = new ArrayList<>();
        try (var files = Files.list(path)) {
            files.map(file -> file.getFileName().toString()).filter(name -> name.endsWith(".vec")).forEach(names::add);
        }
        assertEquals("the fixture must hold exactly one .vec, saw " + names, 1, names.size());
        return names.get(0);
    }

    private static byte[] xor(final byte[] bytes) {
        final byte[] out = new byte[bytes.length];
        for (int i = 0; i < bytes.length; i++) {
            out[i] = (byte) (bytes[i] ^ 0x5a);
        }
        return out;
    }

    /**
     * A layer that undoes {@link #xor} on every {@code .vec} read, so that the file on disk is transformed
     * and the chain serves the original bytes: the shape of an encryption-at-rest wrapper.
     *
     * <p>{@code openChecksumInput} is not transformed, because {@link FilterDirectory} delegates it to the
     * layer below. This fixture's readers never reach it: the flat vectors reader uses it for the {@code .vemf}
     * meta file and in {@code checkIntegrity()}, and this test does neither.
     */
    private static final class XorOnReadDirectory extends FilterDirectory {
        XorOnReadDirectory(final Directory delegate) {
            super(delegate);
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
            return new ByteArrayIndexInput(name, xor(onDisk));
        }
    }

    /** Every document's score under the mode the query path uses, in doc order. */
    private static List<Float> scoreAll(final FloatVectorValues values, final float[] target) throws IOException {
        final VectorScorer scorer = VectorScorerMode.RESCORE.createScorer(values, target);
        assertNotNull(scorer);
        final List<Float> scores = new ArrayList<>();
        final DocIdSetIterator docs = scorer.iterator();
        for (int doc = docs.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = docs.nextDoc()) {
            scores.add(scorer.score());
        }
        return scores;
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

            try (KNNDirectIODirectory storage = new KNNDirectIODirectory(delegate, "_0", () -> false, () -> 0L)) {
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

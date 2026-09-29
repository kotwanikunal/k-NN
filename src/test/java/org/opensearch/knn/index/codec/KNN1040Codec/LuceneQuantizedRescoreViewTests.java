/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.hnsw.HnswGraphProvider;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsReader;
import org.apache.lucene.index.DocValuesSkipIndexType;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.VectorScorer;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.util.InfoStream;
import org.apache.lucene.util.StringHelper;
import org.apache.lucene.util.Version;
import org.apache.lucene.util.quantization.QuantizedVectorsReader;
import org.mockito.Mock;
import org.opensearch.common.settings.ClusterSettings;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.codec.scorer.HasRescoreVectorsReader;
import org.opensearch.knn.index.query.scorers.VectorScorerMode;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.mockito.Mockito.when;
import static org.opensearch.knn.index.KNNSettings.KNN_DIRECT_IO_RESCORE_ENABLED_SETTING;

/**
 * Closes the <b>Lucene-engine quantized row</b> of the coverage matrix: k-NN's own rescore over a Lucene
 * quantized index, where traversal scores the {@code .veq} codes and rescore reads the fp32 {@code .vec}.
 *
 * <p>This is the hardest row to reach, and the reason is structural rather than about encodings:
 * {@code KNN1040HnswScalarQuantizedVectorsFormat} builds a <em>stock</em> {@code Lucene99HnswVectorsReader}
 * over a <em>stock</em> {@code Lucene104ScalarQuantizedVectorsReader}, the HNSW reader is {@code final}, and
 * the flat reader it holds is private with no accessor — so before this commit there was no plugin-owned
 * object anywhere between the segment and this row's {@code .vec}.
 * {@link KNN1040RescoreAwareHnswVectorsReader} is that object, and it is a <em>reader</em> delegate precisely
 * so that every values object on the traversal, merge, warmup and fetch paths stays the one Lucene built.
 *
 * <p>Two of the assertions here are not about the rescore path at all. {@link #testTheMergeInstanceIsTheStockReader()}
 * and {@link #testTheWrapperStillPresentsTheInterfacesMergeTestsFor()} pin the failure mode a reader delegate
 * actually has: Lucene decides whether a merge can reuse an existing graph, and reuse a quantized scorer
 * supplier, by {@code instanceof} on the reader it is handed. A delegate that hid either interface would turn
 * incremental merges into full rebuilds while producing <em>identical resulting bytes</em> — invisible to every
 * correctness test, and visible only as merge cost.
 */
public class LuceneQuantizedRescoreViewTests extends KNNTestCase {

    private static final String FIELD_NAME = "vector";
    private static final int DIMENSION = 16;
    private static final int NUM_VECTORS = 40;

    @Mock
    ClusterSettings clusterSettings;

    @Override
    public void setUp() throws Exception {
        super.setUp();
        when(clusterService.getClusterSettings()).thenReturn(clusterSettings);
        KNNSettings.state().setClusterService(clusterService);
        setRescoreEnabled(true);
    }

    private void setRescoreEnabled(final boolean enabled) {
        when(clusterSettings.get(KNN_DIRECT_IO_RESCORE_ENABLED_SETTING)).thenReturn(enabled);
    }

    private static FieldInfo fieldInfo() {
        return new FieldInfo(
            FIELD_NAME,
            0,
            false,
            false,
            false,
            IndexOptions.NONE,
            DocValuesType.NONE,
            DocValuesSkipIndexType.NONE,
            -1,
            Map.of(),
            0,
            0,
            0,
            DIMENSION,
            VectorEncoding.FLOAT32,
            VectorSimilarityFunction.EUCLIDEAN,
            false,
            false
        );
    }

    /** A segment written by the real Lucene-engine quantized HNSW writer this format selects. */
    private static SegmentReadState writeSegment(final MMapDirectory dir) throws Exception {
        final FieldInfo fieldInfo = fieldInfo();
        final FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { fieldInfo });
        final SegmentInfo segmentInfo = new SegmentInfo(
            dir,
            Version.LATEST,
            Version.LATEST,
            "_0",
            NUM_VECTORS,
            false,
            false,
            null,
            Collections.emptyMap(),
            StringHelper.randomId(),
            new HashMap<>(),
            null
        );
        final SegmentWriteState writeState = new SegmentWriteState(
            InfoStream.NO_OUTPUT,
            dir,
            segmentInfo,
            fieldInfos,
            null,
            IOContext.DEFAULT
        );

        try (KnnVectorsWriter writer = new KNN1040HnswScalarQuantizedVectorsFormat().fieldsWriter(writeState)) {
            @SuppressWarnings("unchecked")
            final KnnFieldVectorsWriter<float[]> fieldWriter = (KnnFieldVectorsWriter<float[]>) writer.addField(fieldInfo);
            for (int i = 0; i < NUM_VECTORS; i++) {
                fieldWriter.addValue(i, randomVector());
            }
            writer.flush(NUM_VECTORS, null);
            writer.finish();
        }
        return new SegmentReadState(dir, segmentInfo, fieldInfos, IOContext.DEFAULT);
    }

    private static float[] randomVector() {
        final float[] v = new float[DIMENSION];
        for (int i = 0; i < DIMENSION; i++) {
            v[i] = random().nextFloat() * 2 - 1;
        }
        return v;
    }

    private static List<float[]> readAll(final FloatVectorValues values) throws IOException {
        final List<float[]> vectors = new ArrayList<>(values.size());
        for (int ord = 0; ord < values.size(); ord++) {
            vectors.add(values.vectorValue(ord).clone());
        }
        return vectors;
    }

    private static FloatVectorValues viewFrom(final KnnVectorsReader reader) {
        return ((HasRescoreVectorsReader) reader).rescoreVectorValues(FIELD_NAME);
    }

    /**
     * The row closes, and the vectors are the same ones. The view is built on the <em>raw</em> fp32 format
     * nested inside the quantized one, so it reads {@code .vec} and never opens a second handle on the
     * {@code .veq} codes the rescore path will not look at.
     */
    @SneakyThrows
    public void testTheLuceneQuantizedReaderOffersARescoreViewOfTheSameVectors() {
        try (MMapDirectory dir = new MMapDirectory(createTempDir())) {
            final SegmentReadState state = writeSegment(dir);
            try (KnnVectorsReader reader = new KNN1040HnswScalarQuantizedVectorsFormat().fieldsReader(state)) {
                assertTrue(reader.getClass().getName(), reader instanceof KNN1040RescoreAwareHnswVectorsReader);
                assertTrue(reader.getClass().getName(), reader instanceof HasRescoreVectorsReader);

                final FloatVectorValues values = reader.getFloatVectorValues(FIELD_NAME);
                final FloatVectorValues view = viewFrom(reader);
                assertNotNull("the Lucene-engine quantized row must offer a rescore view when the setting is on", view);

                assertEquals(NUM_VECTORS, view.size());
                assertEquals(DIMENSION, view.dimension());
                assertEquals(values.getVectorByteLength(), view.getVectorByteLength());
                assertEquals(values.getEncoding(), view.getEncoding());

                final List<float[]> expected = readAll(values);
                final List<float[]> actual = readAll(view);
                for (int ord = 0; ord < expected.size(); ord++) {
                    assertArrayEquals("ordinal " + ord, expected.get(ord), actual.get(ord), 0.0f);
                }
            }
        }
    }

    /**
     * The identity oracle at unit scale for this row. {@code VectorScorerMode.RESCORE} is literally
     * {@code values.rescorer(target)}, which on this row is Lucene's fp32 rescorer over {@code .vec} — so
     * substituting the view must change no score at all, and does not.
     */
    @SneakyThrows
    public void testScoresThroughTheViewAreIdenticalForTheLuceneQuantizedRow() {
        try (MMapDirectory dir = new MMapDirectory(createTempDir())) {
            final SegmentReadState state = writeSegment(dir);
            try (KnnVectorsReader reader = new KNN1040HnswScalarQuantizedVectorsFormat().fieldsReader(state)) {
                final FloatVectorValues values = reader.getFloatVectorValues(FIELD_NAME);
                final FloatVectorValues view = viewFrom(reader);
                assertNotNull(view);

                final float[] target = randomVector();
                final VectorScorer expected = VectorScorerMode.RESCORE.createScorer(values, target);
                final VectorScorer actual = VectorScorerMode.RESCORE.createScorer(view, target);
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
                assertEquals(NUM_VECTORS, scored);
                assertEquals(DocIdSetIterator.NO_MORE_DOCS, actualDocs.nextDoc());
            }
        }
    }

    /**
     * Traversal scoring is untouched: {@code scorer()} on this row's values scores the quantized codes, and the
     * values are the object Lucene built, not a plugin wrapper. This is the "must NOT become Direct I/O" half of
     * the row — the reason the dispatch is per read rather than per file.
     */
    @SneakyThrows
    public void testTraversalValuesAreLucenesOwnAndCarryNoCapability() {
        try (MMapDirectory dir = new MMapDirectory(createTempDir())) {
            final SegmentReadState state = writeSegment(dir);
            try (KnnVectorsReader reader = new KNN1040HnswScalarQuantizedVectorsFormat().fieldsReader(state)) {
                final FloatVectorValues values = reader.getFloatVectorValues(FIELD_NAME);
                assertFalse(values.getClass().getName(), values instanceof HasRescoreVectorsReader);
                assertTrue(values.getClass().getName(), values.getClass().getName().startsWith("org.apache.lucene."));
                assertNotNull("traversal must still have a quantized scorer", values.scorer(randomVector()));
            }
        }
    }

    /**
     * The merge instance is the stock reader, not this wrapper. Merge has no use for a rescore view, and Lucene
     * decides whether it can reuse an existing graph by {@code instanceof HnswGraphProvider} on the object
     * {@code MergeState} hands it — which is the result of {@code getMergeInstance()}.
     */
    @SneakyThrows
    public void testTheMergeInstanceIsTheStockReader() {
        try (MMapDirectory dir = new MMapDirectory(createTempDir())) {
            final SegmentReadState state = writeSegment(dir);
            try (KnnVectorsReader reader = new KNN1040HnswScalarQuantizedVectorsFormat().fieldsReader(state)) {
                final KnnVectorsReader mergeInstance = reader.getMergeInstance();

                assertTrue(mergeInstance.getClass().getName(), mergeInstance instanceof Lucene99HnswVectorsReader);
                assertFalse(mergeInstance instanceof KNN1040RescoreAwareHnswVectorsReader);
                assertTrue(mergeInstance instanceof HnswGraphProvider);
                assertTrue(mergeInstance instanceof QuantizedVectorsReader);
            }
        }
    }

    /**
     * Belt and braces for the same hazard: even the wrapper itself answers to both interfaces, so a caller that
     * tests the search-time reader rather than the merge instance also sees what it expects.
     */
    @SneakyThrows
    public void testTheWrapperStillPresentsTheInterfacesMergeTestsFor() {
        try (MMapDirectory dir = new MMapDirectory(createTempDir())) {
            final SegmentReadState state = writeSegment(dir);
            try (KnnVectorsReader reader = new KNN1040HnswScalarQuantizedVectorsFormat().fieldsReader(state)) {
                assertTrue(reader instanceof HnswGraphProvider);
                assertTrue(reader instanceof QuantizedVectorsReader);
                assertNotNull(((HnswGraphProvider) reader).getGraph(FIELD_NAME));
                assertNotNull(((QuantizedVectorsReader) reader).getQuantizedVectorValues(FIELD_NAME));
                assertTrue(reader.getOffHeapByteSize(fieldInfo()).isEmpty() == false);
            }
        }
    }

    /** With the setting off there is no view, so a default node never acquires a second handle on {@code .vec}. */
    @SneakyThrows
    public void testWithTheSettingOffTheLuceneQuantizedRowOffersNoView() {
        setRescoreEnabled(false);
        try (MMapDirectory dir = new MMapDirectory(createTempDir())) {
            final SegmentReadState state = writeSegment(dir);
            try (KnnVectorsReader reader = new KNN1040HnswScalarQuantizedVectorsFormat().fieldsReader(state)) {
                assertNull(viewFrom(reader));
                setRescoreEnabled(true);
                assertNotNull("the setting is dynamic and declining is not remembered as failure", viewFrom(reader));
            }
        }
    }

    /** Closing the reader closes the view with it. */
    @SneakyThrows
    public void testClosingTheLuceneQuantizedReaderClosesTheView() {
        try (MMapDirectory dir = new MMapDirectory(createTempDir())) {
            final SegmentReadState state = writeSegment(dir);
            final KnnVectorsReader reader = new KNN1040HnswScalarQuantizedVectorsFormat().fieldsReader(state);
            assertNotNull(viewFrom(reader));

            reader.close();

            assertNull(viewFrom(reader));
        }
    }

    /** An unknown field is an ordinary decline, not a throw. */
    @SneakyThrows
    public void testAnUnknownFieldOffersNoView() {
        try (MMapDirectory dir = new MMapDirectory(createTempDir())) {
            final SegmentReadState state = writeSegment(dir);
            try (KnnVectorsReader reader = new KNN1040HnswScalarQuantizedVectorsFormat().fieldsReader(state)) {
                assertNull(((HasRescoreVectorsReader) reader).rescoreVectorValues("no_such_field"));
            }
        }
    }
}

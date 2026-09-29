/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.VectorScorer;
import org.apache.lucene.store.MMapDirectory;
import org.mockito.Mock;
import org.opensearch.common.settings.ClusterSettings;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.codec.scorer.HasRescoreVectorValues;
import org.opensearch.knn.index.query.scorers.VectorScorerMode;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.mockito.Mockito.when;
import static org.opensearch.knn.index.KNNSettings.KNN_DIRECT_IO_RESCORE_ENABLED_SETTING;
import static org.opensearch.knn.index.codec.KNN1040Codec.KNN1040ScalarQuantizedTestUtils.DIMENSION;
import static org.opensearch.knn.index.codec.KNN1040Codec.KNN1040ScalarQuantizedTestUtils.FIELD_NAME;
import static org.opensearch.knn.index.codec.KNN1040Codec.KNN1040ScalarQuantizedTestUtils.NUM_VECTORS;

/**
 * The <em>what</em>-signal, end to end on a real segment: the faiss SQ format builds a rescore view, the
 * flat reader hands it to the values object, and the values object offers it through
 * {@link HasRescoreVectorValues} — so a rescore query reads the same full-precision vectors through an
 * {@link org.apache.lucene.store.IndexInput} the storage layer chose, while every other read of the same
 * file is untouched.
 *
 * <p>Where {@code KNNRescoreVectorsReaderTests} pins the composition in isolation, this pins the
 * <em>wiring</em>: that it is reached from the format the codec actually selects, on a segment written by the
 * real quantized writer, and that the vectors it yields are the same ones. Those are different failures —
 * the composition can work perfectly while nothing calls it, which is exactly the state the previous commit
 * deliberately left the tree in.
 *
 * <p>The one thing every assertion here rests on is that the two views are interchangeable. They must be:
 * the whole reason a failure anywhere in this path can be treated as a silent fallback is that both read the
 * same bytes of the same file at the same ordinals.
 */
public class ScalarQuantizedRescoreVectorValuesTests extends KNNTestCase {

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

    private static List<float[]> readAll(final FloatVectorValues values) throws IOException {
        final List<float[]> vectors = new ArrayList<>(values.size());
        for (int ord = 0; ord < values.size(); ord++) {
            vectors.add(values.vectorValue(ord).clone());
        }
        return vectors;
    }

    /** A segment written by the real quantized writer, as an index lays one out. */
    private static SegmentReadState segment(final MMapDirectory dir) throws Exception {
        return KNN1040ScalarQuantizedTestUtils.writeQuantizedVectors(dir, new KNN1040ScalarQuantizedVectorsFormat(), random());
    }

    /**
     * The values object the rescore path reaches, obtained through the format the codec selects rather than
     * by constructing the reader by hand — so the test fails if the format stops passing the view down.
     */
    private static FloatVectorValues valuesFrom(final KnnVectorsReader reader) throws IOException {
        final FlatVectorsReader flatReader = ((Faiss1040ScalarQuantizedKnnVectorsReader) reader).getFlatVectorsReader();
        return flatReader.getFloatVectorValues(FIELD_NAME);
    }

    /**
     * The wiring, and the identity it has to preserve. The format hands the flat reader a rescore view, the
     * values offer it, and it yields the same vectors at the same ordinals.
     */
    @SneakyThrows
    public void testTheValuesOfferARescoreViewOfTheSameVectors() {
        try (MMapDirectory dir = new MMapDirectory(createTempDir())) {
            final SegmentReadState state = segment(dir);
            try (KnnVectorsReader reader = new Faiss1040ScalarQuantizedKnnVectorsFormat().fieldsReader(state)) {
                final FloatVectorValues values = valuesFrom(reader);
                assertTrue(values.getClass().getName(), values instanceof HasRescoreVectorValues);

                final FloatVectorValues view = ((HasRescoreVectorValues) values).rescoreVectorValues();
                assertNotNull("the faiss SQ row must offer a rescore view when the setting is on", view);

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
     * The view and the ordinary values are interchangeable for scoring too, which is the property
     * {@link VectorScorerMode#RESCORE} relies on: it is literally {@code values.rescorer(target)}, so
     * substituting the view one level up must produce the same scores. Exact equality, not a delta — both
     * sides are the same fp32 bytes through the same Lucene scorer.
     */
    @SneakyThrows
    public void testScoresThroughTheViewAreIdenticalToScoresThroughTheOrdinaryValues() {
        try (MMapDirectory dir = new MMapDirectory(createTempDir())) {
            final SegmentReadState state = segment(dir);
            try (KnnVectorsReader reader = new Faiss1040ScalarQuantizedKnnVectorsFormat().fieldsReader(state)) {
                final FloatVectorValues values = valuesFrom(reader);
                final FloatVectorValues view = ((HasRescoreVectorValues) values).rescoreVectorValues();
                assertNotNull(view);

                final float[] target = KNN1040ScalarQuantizedTestUtils.randomVector(DIMENSION, random());
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
     * With the setting off there is no view and nothing was opened to find that out. This is the
     * "default node is bit-identical to no rescore path at all" guarantee, asserted where it can fail:
     * the setting is read by the view, not by the caller, so a wiring change that made the view eager
     * would show up only here.
     */
    @SneakyThrows
    public void testWithTheSettingOffThereIsNoView() {
        setRescoreEnabled(false);
        try (MMapDirectory dir = new MMapDirectory(createTempDir())) {
            final SegmentReadState state = segment(dir);
            try (KnnVectorsReader reader = new Faiss1040ScalarQuantizedKnnVectorsFormat().fieldsReader(state)) {
                final FloatVectorValues values = valuesFrom(reader);
                assertTrue(values instanceof HasRescoreVectorValues);
                assertNull(((HasRescoreVectorValues) values).rescoreVectorValues());
            }
        }
    }

    /**
     * The setting is dynamic, and declining is not remembered as a failure: a node that turns it on gets
     * the view on the next query without reopening the index.
     */
    @SneakyThrows
    public void testTurningTheSettingOnAfterwardsStillYieldsTheView() {
        setRescoreEnabled(false);
        try (MMapDirectory dir = new MMapDirectory(createTempDir())) {
            final SegmentReadState state = segment(dir);
            try (KnnVectorsReader reader = new Faiss1040ScalarQuantizedKnnVectorsFormat().fieldsReader(state)) {
                assertNull(((HasRescoreVectorValues) valuesFrom(reader)).rescoreVectorValues());
                setRescoreEnabled(true);
                assertNotNull(((HasRescoreVectorValues) valuesFrom(reader)).rescoreVectorValues());
            }
        }
    }

    /**
     * Each call yields its own values. {@link FloatVectorValues} carries a cursor and is not thread safe, so
     * a shared instance would corrupt concurrent rescorers — the thing that is shared and segment-scoped is
     * the reader behind them, not these.
     */
    @SneakyThrows
    public void testEachCallYieldsItsOwnValues() {
        try (MMapDirectory dir = new MMapDirectory(createTempDir())) {
            final SegmentReadState state = segment(dir);
            try (KnnVectorsReader reader = new Faiss1040ScalarQuantizedKnnVectorsFormat().fieldsReader(state)) {
                final FloatVectorValues values = valuesFrom(reader);
                final FloatVectorValues first = ((HasRescoreVectorValues) values).rescoreVectorValues();
                final FloatVectorValues second = ((HasRescoreVectorValues) values).rescoreVectorValues();
                assertNotNull(first);
                assertNotNull(second);
                assertNotSame(first, second);
            }
        }
    }

    /**
     * {@code copy()} exists so a caller can iterate the same vectors independently, and it must carry the
     * capability with it. Dropping the supplier there would silently disable the rescore path for every
     * caller that copies first, with nothing failing.
     */
    @SneakyThrows
    public void testCopyKeepsTheRescoreView() {
        try (MMapDirectory dir = new MMapDirectory(createTempDir())) {
            final SegmentReadState state = segment(dir);
            try (KnnVectorsReader reader = new Faiss1040ScalarQuantizedKnnVectorsFormat().fieldsReader(state)) {
                final FloatVectorValues copy = valuesFrom(reader).copy();
                assertTrue(copy instanceof HasRescoreVectorValues);
                assertNotNull(((HasRescoreVectorValues) copy).rescoreVectorValues());
            }
        }
    }

    /**
     * Closing the reader closes the view's handles with it, and the values it had already handed out stop
     * offering one. A second handle on {@code .vec} that outlived its reader would leak a file descriptor
     * per segment on every node with the setting on.
     */
    @SneakyThrows
    public void testClosingTheReaderClosesTheView() {
        try (MMapDirectory dir = new MMapDirectory(createTempDir())) {
            final SegmentReadState state = segment(dir);
            final KnnVectorsReader reader = new Faiss1040ScalarQuantizedKnnVectorsFormat().fieldsReader(state);
            final FloatVectorValues values = valuesFrom(reader);
            assertNotNull(((HasRescoreVectorValues) values).rescoreVectorValues());

            reader.close();

            assertNull(((HasRescoreVectorValues) values).rescoreVectorValues());
        }
    }

    /**
     * An empty segment's values are built without either capability, because Lucene exposes no quantized
     * delegate for one. The rescore path must decline rather than throw.
     */
    @SneakyThrows
    public void testEmptyValuesOfferNoView() {
        final ScalarQuantizedFloatVectorValues empty = new ScalarQuantizedFloatVectorValues(
            new org.opensearch.knn.index.vectorvalues.TestVectorValues.PreDefinedFloatVectorValues(List.of()),
            null
        );
        assertNull(empty.rescoreVectorValues());
        assertNull(empty.vectorLoaderSource());
    }
}

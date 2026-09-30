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
import org.opensearch.knn.index.codec.scorer.HasRescoreVectorsReader;
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
 * flat reader holds it, and the segment's {@code KnnVectorsReader} offers it through
 * {@link HasRescoreVectorsReader} — so a rescore query reads the same full-precision vectors through an
 * {@link org.apache.lucene.store.IndexInput} the storage layer chose, while every other read of the same
 * file is untouched.
 *
 * <p>The capability is on the <em>reader</em> and deliberately not on the values: two of the four
 * rescore-reachable encodings have no plugin values class, and a values wrapper per encoding is the one shape
 * this design avoids. {@link #testTheValuesDoNotCarryTheCapability()} is the regression guard for that.
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

    /** The rescore view as the query layer reaches it: from the segment's reader, by field name. */
    private static FloatVectorValues viewFrom(final KnnVectorsReader reader) {
        return ((HasRescoreVectorsReader) reader).rescoreVectorValues(FIELD_NAME);
    }

    /**
     * The wiring, and the identity it has to preserve. The format hands the flat reader a rescore view, the
     * values offer it, and it yields the same vectors at the same ordinals.
     */
    @SneakyThrows
    public void testTheReaderOffersARescoreViewOfTheSameVectors() {
        try (MMapDirectory dir = new MMapDirectory(createTempDir())) {
            final SegmentReadState state = segment(dir);
            try (KnnVectorsReader reader = new Faiss1040ScalarQuantizedKnnVectorsFormat().fieldsReader(state)) {
                final FloatVectorValues values = valuesFrom(reader);
                assertTrue(reader.getClass().getName(), reader instanceof HasRescoreVectorsReader);

                final FloatVectorValues view = viewFrom(reader);
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
                final FloatVectorValues view = viewFrom(reader);
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
                assertTrue(reader instanceof HasRescoreVectorsReader);
                assertNull(viewFrom(reader));
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
                assertNull(viewFrom(reader));
                setRescoreEnabled(true);
                assertNotNull(viewFrom(reader));
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
                final FloatVectorValues first = viewFrom(reader);
                final FloatVectorValues second = viewFrom(reader);
                assertNotNull(first);
                assertNotNull(second);
                assertNotSame(first, second);
            }
        }
    }

    /**
     * The values do <em>not</em> carry the capability, and neither does a copy of them. This is the regression
     * guard for the design decision the reader-level route exists to enforce: the moment a values class can be
     * asked for the view, the two encodings that have no plugin values class stop being reachable by the same
     * mechanism, and the pressure is to introduce a per-encoding values wrapper — the one shape that can lose
     * Lucene's SIMD binding while still returning identical bytes, which measured 5.3x slower and which no
     * correctness test can catch.
     */
    @SneakyThrows
    public void testTheValuesDoNotCarryTheCapability() {
        try (MMapDirectory dir = new MMapDirectory(createTempDir())) {
            final SegmentReadState state = segment(dir);
            try (KnnVectorsReader reader = new Faiss1040ScalarQuantizedKnnVectorsFormat().fieldsReader(state)) {
                final FloatVectorValues values = valuesFrom(reader);
                assertFalse(values.getClass().getName(), values instanceof HasRescoreVectorsReader);
                assertFalse(values.copy().getClass().getName(), values.copy() instanceof HasRescoreVectorsReader);
                // and the reader still does, so the route is not simply gone
                assertNotNull(viewFrom(reader));
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
            assertNotNull(viewFrom(reader));

            reader.close();

            assertNull(viewFrom(reader));
        }
    }

    /**
     * A field the reader knows nothing about yields no view and does not throw. The query layer asks by name,
     * so a name that does not resolve has to be an ordinary decline.
     */
    @SneakyThrows
    public void testAnUnknownFieldYieldsNoView() {
        try (MMapDirectory dir = new MMapDirectory(createTempDir())) {
            final SegmentReadState state = segment(dir);
            try (KnnVectorsReader reader = new Faiss1040ScalarQuantizedKnnVectorsFormat().fieldsReader(state)) {
                assertNull(((HasRescoreVectorsReader) reader).rescoreVectorValues("no_such_field"));
            }
        }
    }
}

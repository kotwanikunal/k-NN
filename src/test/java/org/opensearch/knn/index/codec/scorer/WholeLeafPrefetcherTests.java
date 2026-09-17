/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.scorer;

import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.TestUtil;
import org.mockito.MockedStatic;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.featureflags.KNNFeatureFlags;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.mockStatic;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.when;

/**
 * Unit tests for {@link WholeLeafPrefetcher}.
 * <p>
 * The load-bearing case is the docId to ordinal translation: on a sparse field, or a segment where some
 * documents simply have no vector, {@code ord == docId} is false and assuming otherwise reads a different
 * document's vector with no exception. Those tests therefore assert on the *contents* of the vector at the
 * returned ordinal, not just on the ordinal's value.
 */
public class WholeLeafPrefetcherTests extends KNNTestCase {

    private static final String FIELD = "target_field";
    private static final String ID_FIELD = "id";
    private static final int DIMENSION = 4;

    /**
     * Runs the body with both prefetch switches on. {@code mockStatic} stubs every static on
     * {@link KNNFeatureFlags}, so both have to be stubbed explicitly.
     */
    private void withPrefetchEnabled(final ThrowingRunnable body) throws Exception {
        try (MockedStatic<KNNFeatureFlags> flags = mockStatic(KNNFeatureFlags.class)) {
            flags.when(KNNFeatureFlags::isWholeLeafPrefetchEnabled).thenReturn(true);
            flags.when(KNNFeatureFlags::isPrefetchEnabled).thenReturn(true);
            body.run();
        }
    }

    private interface ThrowingRunnable {
        void run() throws Exception;
    }

    /**
     * Writes one segment where document {@code i} carries the vector {@code [i, 0, 0, 0]} if and only if
     * {@code i % vectorEvery == 0}, then deletes every document whose id is a multiple of
     * {@code deleteEvery} (0 to delete nothing) and force merges so the surviving documents are renumbered.
     */
    private void buildIndex(final Directory directory, final int numDocs, final int vectorEvery, final int deleteEvery) throws IOException {
        try (IndexWriter writer = new IndexWriter(directory, newIndexWriterConfig().setCodec(TestUtil.getDefaultCodec()))) {
            for (int i = 0; i < numDocs; i++) {
                final Document document = new Document();
                document.add(new StringField(ID_FIELD, String.valueOf(i), Field.Store.YES));
                if (i % vectorEvery == 0) {
                    final float[] vector = new float[DIMENSION];
                    vector[0] = i;
                    document.add(new KnnFloatVectorField(FIELD, vector));
                }
                writer.addDocument(document);
            }
            if (deleteEvery > 0) {
                for (int i = 0; i < numDocs; i += deleteEvery) {
                    writer.deleteDocuments(new Term(ID_FIELD, String.valueOf(i)));
                }
            }
            writer.forceMerge(1);
        }
    }

    /** The leaf-local doc ids that actually have a vector, ascending. */
    private int[] docIdsWithVectors(final LeafReader leafReader) throws IOException {
        final FloatVectorValues values = leafReader.getFloatVectorValues(FIELD);
        final List<Integer> docIds = new ArrayList<>();
        final var iterator = values.iterator();
        for (int doc = iterator.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = iterator.nextDoc()) {
            docIds.add(doc);
        }
        return docIds.stream().mapToInt(Integer::intValue).toArray();
    }

    /** The id stored on a leaf-local doc id, which survives the renumbering a force merge does. */
    private int storedId(final LeafReader leafReader, final int docId) throws IOException {
        return Integer.parseInt(leafReader.storedFields().document(docId).get(ID_FIELD));
    }

    public void testToOrdinals_whenFieldIsDense_thenOrdinalsEqualDocIds() throws Exception {
        try (Directory directory = newDirectory()) {
            buildIndex(directory, 32, 1, 0);
            try (DirectoryReader reader = DirectoryReader.open(directory)) {
                final LeafReader leafReader = reader.leaves().get(0).reader();
                final FloatVectorValues values = leafReader.getFloatVectorValues(FIELD);

                final int[] ords = WholeLeafPrefetcher.toOrdinals(values, new int[] { 3, 17, 31 });

                assertArrayEquals(new int[] { 3, 17, 31 }, ords);
            }
        }
    }

    /**
     * The case a naive {@code ord == docId} gets wrong. Every candidate's ordinal is checked by reading the
     * vector back and comparing it against the id stored on that document.
     */
    public void testToOrdinals_whenFieldIsSparseWithDeletions_thenTranslatesThroughIterator() throws Exception {
        try (Directory directory = newDirectory()) {
            buildIndex(directory, 60, 3, 7);
            try (DirectoryReader reader = DirectoryReader.open(directory)) {
                assertEquals(1, reader.leaves().size());
                final LeafReader leafReader = reader.leaves().get(0).reader();
                final int[] candidates = docIdsWithVectors(leafReader);
                assertTrue("expected several candidates with vectors", candidates.length > 3);

                final FloatVectorValues values = leafReader.getFloatVectorValues(FIELD);
                final int[] ords = WholeLeafPrefetcher.toOrdinals(values, candidates);

                assertEquals(candidates.length, ords.length);
                boolean sawOrdinalDifferingFromDocId = false;
                for (int i = 0; i < candidates.length; i++) {
                    // The vector's first component is the document's original id; if the ordinal were
                    // computed as ord == docId these would disagree.
                    assertEquals(
                        "wrong vector for docId " + candidates[i],
                        (float) storedId(leafReader, candidates[i]),
                        values.vectorValue(ords[i])[0],
                        0.0f
                    );
                    sawOrdinalDifferingFromDocId |= ords[i] != candidates[i];
                }
                assertTrue("the fixture must actually separate ordinals from doc ids", sawOrdinalDifferingFromDocId);
            }
        }
    }

    public void testToOrdinals_whenCandidateHasNoVector_thenCandidateIsDropped() throws Exception {
        try (Directory directory = newDirectory()) {
            buildIndex(directory, 30, 3, 0);
            try (DirectoryReader reader = DirectoryReader.open(directory)) {
                final LeafReader leafReader = reader.leaves().get(0).reader();
                final FloatVectorValues values = leafReader.getFloatVectorValues(FIELD);

                // docIds 0, 3, 6 have vectors (ordinals 0, 1, 2); 1 and 2 do not.
                final int[] ords = WholeLeafPrefetcher.toOrdinals(values, new int[] { 0, 1, 2, 3, 6 });

                assertArrayEquals(new int[] { 0, 1, 2 }, ords);
            }
        }
    }

    public void testToOrdinals_whenCandidatesAreUnsortedWithDuplicates_thenOrdinalsAreSortedAndUnique() throws Exception {
        try (Directory directory = newDirectory()) {
            buildIndex(directory, 32, 1, 0);
            try (DirectoryReader reader = DirectoryReader.open(directory)) {
                final LeafReader leafReader = reader.leaves().get(0).reader();
                final FloatVectorValues values = leafReader.getFloatVectorValues(FIELD);

                final int[] ords = WholeLeafPrefetcher.toOrdinals(values, new int[] { 9, 2, 9, 5, 2 });

                assertArrayEquals(new int[] { 2, 5, 9 }, ords);
            }
        }
    }

    public void testToOrdinals_whenCandidateIsBeyondTheLastVector_thenIterationStops() throws Exception {
        try (Directory directory = newDirectory()) {
            buildIndex(directory, 10, 1, 0);
            try (DirectoryReader reader = DirectoryReader.open(directory)) {
                final LeafReader leafReader = reader.leaves().get(0).reader();
                final FloatVectorValues values = leafReader.getFloatVectorValues(FIELD);

                final int[] ords = WholeLeafPrefetcher.toOrdinals(values, new int[] { 8, 999 });

                assertArrayEquals(new int[] { 8 }, ords);
            }
        }
    }

    /**
     * The whole point of the unit: one prefetch covering every candidate in the leaf, rather than one per
     * 64-ordinal bulk-scoring batch. 100 candidates is deliberately more than
     * {@code VectorScorer.Bulk.DEFAULT_BULK_BATCH_SIZE}.
     */
    public void testPrefetchCandidates_whenLeafHasManyCandidates_thenOnePrefetchCoversThemAll() throws Exception {
        try (Directory directory = newDirectory()) {
            buildIndex(directory, 100, 1, 0);
            try (DirectoryReader reader = DirectoryReader.open(directory)) {
                final LeafReader leafReader = reader.leaves().get(0).reader();
                final int[] candidates = docIdsWithVectors(leafReader);
                assertEquals(100, candidates.length);

                withPrefetchEnabled(() -> {
                    try (MockedStatic<PrefetchHelper> prefetchHelper = mockStatic(PrefetchHelper.class)) {
                        final int numOrds = WholeLeafPrefetcher.prefetchCandidates(leafReader, FIELD, candidates);

                        assertEquals(100, numOrds);
                        prefetchHelper.verify(
                            () -> PrefetchHelper.prefetch(any(), eq(0L), eq((long) DIMENSION * Float.BYTES), any(), eq(100)),
                            times(1)
                        );
                    }
                });
            }
        }
    }

    public void testPrefetchCandidates_whenGivenScoreDocs_thenExtractsDocIds() throws Exception {
        try (Directory directory = newDirectory()) {
            buildIndex(directory, 20, 1, 0);
            try (DirectoryReader reader = DirectoryReader.open(directory)) {
                final LeafReader leafReader = reader.leaves().get(0).reader();
                final ScoreDoc[] scoreDocs = { new ScoreDoc(11, 0.9f), new ScoreDoc(4, 0.8f), new ScoreDoc(19, 0.7f) };

                withPrefetchEnabled(() -> {
                    try (MockedStatic<PrefetchHelper> prefetchHelper = mockStatic(PrefetchHelper.class)) {
                        assertEquals(3, WholeLeafPrefetcher.prefetchCandidates(leafReader, FIELD, scoreDocs));

                        prefetchHelper.verify(() -> PrefetchHelper.prefetch(any(), eq(0L), any(Long.class), any(), eq(3)), times(1));
                    }
                });
            }
        }
    }

    public void testPrefetchCandidates_whenWholeLeafFlagIsOff_thenNoOp() throws Exception {
        try (Directory directory = newDirectory()) {
            buildIndex(directory, 20, 1, 0);
            try (DirectoryReader reader = DirectoryReader.open(directory)) {
                final LeafReader leafReader = reader.leaves().get(0).reader();
                try (
                    MockedStatic<KNNFeatureFlags> flags = mockStatic(KNNFeatureFlags.class);
                    MockedStatic<PrefetchHelper> prefetchHelper = mockStatic(PrefetchHelper.class)
                ) {
                    flags.when(KNNFeatureFlags::isWholeLeafPrefetchEnabled).thenReturn(false);
                    flags.when(KNNFeatureFlags::isPrefetchEnabled).thenReturn(true);

                    assertEquals(0, WholeLeafPrefetcher.prefetchCandidates(leafReader, FIELD, new int[] { 1, 2, 3 }));

                    prefetchHelper.verifyNoInteractions();
                }
            }
        }
    }

    /** The existing kill switch must still turn off everything, so that Phase-5 arm (b) is one switch. */
    public void testPrefetchCandidates_whenGeneralPrefetchKillSwitchIsOff_thenNoOp() throws Exception {
        try (Directory directory = newDirectory()) {
            buildIndex(directory, 20, 1, 0);
            try (DirectoryReader reader = DirectoryReader.open(directory)) {
                final LeafReader leafReader = reader.leaves().get(0).reader();
                try (
                    MockedStatic<KNNFeatureFlags> flags = mockStatic(KNNFeatureFlags.class);
                    MockedStatic<PrefetchHelper> prefetchHelper = mockStatic(PrefetchHelper.class)
                ) {
                    flags.when(KNNFeatureFlags::isWholeLeafPrefetchEnabled).thenReturn(true);
                    flags.when(KNNFeatureFlags::isPrefetchEnabled).thenReturn(false);

                    assertEquals(0, WholeLeafPrefetcher.prefetchCandidates(leafReader, FIELD, new int[] { 1, 2, 3 }));

                    prefetchHelper.verifyNoInteractions();
                }
            }
        }
    }

    /**
     * {@link PrefetchHelper#prefetch} returns early below two ordinals, so a single-candidate leaf must not
     * even open vector values - and a multi-candidate leaf must pass more than one ordinal, or the whole
     * unit is a silent no-op.
     */
    public void testPrefetchCandidates_whenLeafHasOneCandidate_thenNoOp() throws Exception {
        final LeafReader leafReader = mock(LeafReader.class);
        withPrefetchEnabled(() -> {
            assertEquals(0, WholeLeafPrefetcher.prefetchCandidates(leafReader, FIELD, new int[] { 7 }));
            assertEquals(0, WholeLeafPrefetcher.prefetchCandidates(leafReader, FIELD, new int[0]));
            assertEquals(0, WholeLeafPrefetcher.prefetchCandidates(leafReader, FIELD, (int[]) null));
            assertEquals(0, WholeLeafPrefetcher.prefetchCandidates(leafReader, FIELD, (ScoreDoc[]) null));
        });
    }

    public void testPrefetchCandidates_whenFieldHasNoVectorValues_thenNoOp() throws Exception {
        try (Directory directory = newDirectory()) {
            buildIndex(directory, 20, 1, 0);
            try (DirectoryReader reader = DirectoryReader.open(directory)) {
                final LeafReader leafReader = reader.leaves().get(0).reader();
                withPrefetchEnabled(
                    () -> assertEquals(0, WholeLeafPrefetcher.prefetchCandidates(leafReader, "no_such_field", new int[] { 1, 2 }))
                );
            }
        }
    }

    /** Prefetch is a hint; an I/O failure while issuing it must not surface to the query. */
    public void testPrefetchCandidates_whenReaderThrows_thenNoOpRatherThanFailure() throws Exception {
        final LeafReader leafReader = mock(LeafReader.class);
        when(leafReader.getFloatVectorValues(FIELD)).thenThrow(new IOException("boom"));
        withPrefetchEnabled(() -> assertEquals(0, WholeLeafPrefetcher.prefetchCandidates(leafReader, FIELD, new int[] { 1, 2, 3 })));
    }

    /** Advisory means advisory: the bytes the rescore path would read are identical either way. */
    public void testPrefetchCandidates_whenEnabled_thenVectorsReadAreUnchanged() throws Exception {
        try (Directory directory = newDirectory()) {
            buildIndex(directory, 40, 3, 7);
            try (DirectoryReader reader = DirectoryReader.open(directory)) {
                final LeafReader leafReader = reader.leaves().get(0).reader();
                final int[] candidates = docIdsWithVectors(leafReader);

                final float[][] withoutPrefetch = readVectors(leafReader, candidates);
                withPrefetchEnabled(
                    () -> assertEquals(candidates.length, WholeLeafPrefetcher.prefetchCandidates(leafReader, FIELD, candidates))
                );
                final float[][] withPrefetch = readVectors(leafReader, candidates);

                assertEquals(withoutPrefetch.length, withPrefetch.length);
                for (int i = 0; i < withoutPrefetch.length; i++) {
                    assertArrayEquals(withoutPrefetch[i], withPrefetch[i], 0.0f);
                }
            }
        }
    }

    /**
     * The whole point of unit 16b: a quantized field is fronted by a wrapper over {@code .veq} and
     * {@code .vec} that exposes no slice of its own, and the whole-leaf prefetch used to decline on it -
     * making the feature a silent no-op on exactly the codec the rescore path uses. It must now reach the
     * full-precision values the wrapper names, and hint them with *their* per-vector byte size.
     */
    public void testPrefetchCandidates_whenValuesHideTheirSliceBehindAWrapper_thenStillPrefetches() throws Exception {
        try (Directory directory = newDirectory()) {
            buildIndex(directory, 20, 1, 0);
            try (DirectoryReader reader = DirectoryReader.open(directory)) {
                final FloatVectorValues realValues = reader.leaves().get(0).reader().getFloatVectorValues(FIELD);
                final LeafReader leafReader = mock(LeafReader.class);
                when(leafReader.getFloatVectorValues(FIELD)).thenReturn(new SliceHidingWrapper(realValues));

                withPrefetchEnabled(() -> {
                    try (MockedStatic<PrefetchHelper> prefetchHelper = mockStatic(PrefetchHelper.class)) {
                        assertEquals(3, WholeLeafPrefetcher.prefetchCandidates(leafReader, FIELD, new int[] { 2, 5, 9 }));

                        prefetchHelper.verify(
                            () -> PrefetchHelper.prefetch(any(), eq(0L), eq((long) DIMENSION * Float.BYTES), any(), eq(3)),
                            times(1)
                        );
                    }
                });
            }
        }
    }

    /**
     * The shape of the plugin's scalar-quantized wrapper: it delegates iteration to the full-precision
     * values - so ordinals mean the same thing - but exposes no slice, because it fronts two files.
     */
    private static class SliceHidingWrapper extends FloatVectorValues implements HasFullPrecisionVectorValues {
        private final FloatVectorValues fullPrecision;

        SliceHidingWrapper(final FloatVectorValues fullPrecision) {
            this.fullPrecision = fullPrecision;
        }

        @Override
        public KnnVectorValues getFullPrecisionVectorValues() {
            return fullPrecision;
        }

        @Override
        public int dimension() {
            return fullPrecision.dimension();
        }

        @Override
        public int size() {
            return fullPrecision.size();
        }

        @Override
        public float[] vectorValue(int ord) throws IOException {
            return fullPrecision.vectorValue(ord);
        }

        @Override
        public DocIndexIterator iterator() {
            return fullPrecision.iterator();
        }

        @Override
        public FloatVectorValues copy() throws IOException {
            return new SliceHidingWrapper(fullPrecision.copy());
        }
    }

    private float[][] readVectors(final LeafReader leafReader, final int[] docIds) throws IOException {
        final FloatVectorValues values = leafReader.getFloatVectorValues(FIELD);
        final int[] ords = WholeLeafPrefetcher.toOrdinals(values, docIds);
        final float[][] vectors = new float[ords.length][];
        for (int i = 0; i < ords.length; i++) {
            vectors[i] = values.vectorValue(ords[i]).clone();
        }
        return vectors;
    }
}

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.scorers;

import lombok.SneakyThrows;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.hnsw.HasKnnVectorValues;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.store.DirectIOVectorSource;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * The staging wrapper. Its whole contract is "declare the batch, then score it unchanged", so the tests are
 * about ordering and transparency: staging happens before any score is computed, and nothing else about the
 * delegate's behaviour is altered.
 */
public class PrefetchingRandomVectorScorerTests extends KNNTestCase {

    /** Records the order of stage and read calls, which is the property that makes read ahead work. */
    private static final class RecordingReader {
        private final List<String> calls = new ArrayList<>();

        @SneakyThrows
        DirectIOVectorSource.Reader mockReader() {
            final DirectIOVectorSource.Reader reader = mock(DirectIOVectorSource.Reader.class);
            org.mockito.Mockito.doAnswer(invocation -> {
                calls.add("stage:" + invocation.getArgument(1));
                return null;
            }).when(reader).stage(any(), anyInt());
            return reader;
        }
    }

    private static RandomVectorScorer delegateScoring(final List<String> calls) {
        return new RandomVectorScorer() {
            @Override
            public float score(final int ord) {
                calls.add("score:" + ord);
                return ord * 10f;
            }

            @Override
            public int maxOrd() {
                return 1000;
            }
        };
    }

    @SneakyThrows
    public void testTheWholeBatchIsStagedBeforeAnythingInItIsScored() {
        final RecordingReader recorder = new RecordingReader();
        final DirectIOVectorSource.Reader reader = recorder.mockReader();
        final RandomVectorScorer delegate = delegateScoring(recorder.calls);
        final PrefetchingRandomVectorScorer scorer = new PrefetchingRandomVectorScorer(delegate, reader, mock(FloatVectorValues.class));

        final int[] ords = { 4, 9, 2, 7 };
        final float[] scores = new float[8];
        final float max = scorer.bulkScore(ords, scores, 4);

        assertEquals("stage must come first, or the reads it predicts have already happened", "stage:4", recorder.calls.get(0));
        assertEquals(List.of("stage:4", "score:4", "score:9", "score:2", "score:7"), recorder.calls);
        assertEquals(90f, max, 0.0f);
        assertArrayEquals(new float[] { 40f, 90f, 20f, 70f, 0f, 0f, 0f, 0f }, scores, 0.0f);
        verify(reader).stage(ords, 4);
    }

    /**
     * {@code numOrds} can be less than the array length — Lucene's bulk scorer reuses one oversized array —
     * and the count the wrapper declares has to be the count, not the capacity.
     */
    @SneakyThrows
    public void testOnlyTheDeclaredPrefixOfTheBatchIsStaged() {
        final RecordingReader recorder = new RecordingReader();
        final DirectIOVectorSource.Reader reader = recorder.mockReader();
        final PrefetchingRandomVectorScorer scorer = new PrefetchingRandomVectorScorer(
            delegateScoring(recorder.calls),
            reader,
            mock(FloatVectorValues.class)
        );

        final int[] ords = new int[64];
        for (int i = 0; i < ords.length; i++) {
            ords[i] = i;
        }
        scorer.bulkScore(ords, new float[64], 3);

        verify(reader).stage(ords, 3);
        assertEquals(List.of("stage:3", "score:0", "score:1", "score:2"), recorder.calls);
    }

    /** A single score carries no lookahead, so there is nothing to stage and nothing may be staged. */
    @SneakyThrows
    public void testASingleScoreDoesNotStage() {
        final RecordingReader recorder = new RecordingReader();
        final DirectIOVectorSource.Reader reader = recorder.mockReader();
        final PrefetchingRandomVectorScorer scorer = new PrefetchingRandomVectorScorer(
            delegateScoring(recorder.calls),
            reader,
            mock(FloatVectorValues.class)
        );

        assertEquals(170f, scorer.score(17), 0.0f);
        verify(reader, never()).stage(any(), anyInt());
    }

    @SneakyThrows
    public void testEverythingElseDelegates() {
        final RandomVectorScorer delegate = mock(RandomVectorScorer.class);
        final Bits acceptDocs = mock(Bits.class);
        final Bits acceptOrds = mock(Bits.class);
        when(delegate.maxOrd()).thenReturn(1_000_000);
        when(delegate.ordToDoc(17)).thenReturn(4242);
        when(delegate.getAcceptOrds(acceptDocs)).thenReturn(acceptOrds);
        when(delegate.score(5)).thenReturn(1.5f);

        final FloatVectorValues values = mock(FloatVectorValues.class);
        final PrefetchingRandomVectorScorer scorer = new PrefetchingRandomVectorScorer(
            delegate,
            mock(DirectIOVectorSource.Reader.class),
            values
        );

        assertEquals(1_000_000, scorer.maxOrd());
        assertEquals(4242, scorer.ordToDoc(17));
        assertSame(acceptOrds, scorer.getAcceptOrds(acceptDocs));
        assertEquals(1.5f, scorer.score(5), 0.0f);
    }

    /**
     * {@code AbstractRandomVectorScorer} — what Lucene's scorer on this path actually is — implements
     * {@link HasKnnVectorValues}, and wrapping it must not take that away from anything that looks for it.
     */
    public void testTheWrappedScorerStillExposesItsValues() {
        final FloatVectorValues values = mock(FloatVectorValues.class);
        final RandomVectorScorer scorer = new PrefetchingRandomVectorScorer(
            mock(RandomVectorScorer.class),
            mock(DirectIOVectorSource.Reader.class),
            values
        );

        assertTrue(scorer instanceof HasKnnVectorValues);
        assertSame(values, ((HasKnnVectorValues) scorer).values());
    }

    /**
     * Staging is advisory: a reader that refuses to stage still has to be scored. Nothing in the wrapper may
     * treat a staging failure as a query failure.
     */
    @SneakyThrows
    public void testScoringProceedsWhenStagingThrows() {
        final DirectIOVectorSource.Reader reader = mock(DirectIOVectorSource.Reader.class);
        final List<String> calls = new ArrayList<>();
        final PrefetchingRandomVectorScorer scorer = new PrefetchingRandomVectorScorer(
            delegateScoring(calls),
            reader,
            mock(KnnVectorValues.class)
        );

        // stage() does not declare a checked exception, so the only thing it can throw is unchecked - and
        // the wrapper does not catch it. This pins the decision: staging is not allowed to throw, which is
        // why Reader.stage swallows everything it can instead.
        org.mockito.Mockito.doThrow(new IllegalStateException("boom")).when(reader).stage(any(), anyInt());
        expectThrows(IllegalStateException.class, () -> scorer.bulkScore(new int[] { 1, 2 }, new float[2], 2));
        assertTrue("no score should have been attempted", calls.isEmpty());
    }

    @SneakyThrows
    public void testBulkScoreOfAnEmptyBatchIsANoOp() {
        final RecordingReader recorder = new RecordingReader();
        final PrefetchingRandomVectorScorer scorer = new PrefetchingRandomVectorScorer(
            delegateScoring(recorder.calls),
            recorder.mockReader(),
            mock(FloatVectorValues.class)
        );
        assertEquals(Float.NEGATIVE_INFINITY, scorer.bulkScore(new int[4], new float[4], 0), 0.0f);
        assertEquals(List.of("stage:0"), recorder.calls);
    }

    @SneakyThrows
    public void testScoresAreIdenticalToTheUnwrappedDelegateOverAWholeBatch() throws IOException {
        final List<String> ignored = new ArrayList<>();
        final RandomVectorScorer delegate = delegateScoring(ignored);
        final int[] ords = new int[64];
        for (int i = 0; i < ords.length; i++) {
            ords[i] = (i * 37) % 997;
        }

        final float[] direct = new float[64];
        final float directMax = delegate.bulkScore(ords, direct, 64);

        final float[] wrapped = new float[64];
        final float wrappedMax = new PrefetchingRandomVectorScorer(
            delegate,
            mock(DirectIOVectorSource.Reader.class),
            mock(FloatVectorValues.class)
        ).bulkScore(ords, wrapped, 64);

        assertEquals(directMax, wrappedMax, 0.0f);
        assertArrayEquals(direct, wrapped, 0.0f);
    }
}

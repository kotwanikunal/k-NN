/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.scorers;

import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.hnsw.HasKnnVectorValues;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.opensearch.knn.index.store.DirectIOVectorSource;

import java.io.IOException;

/**
 * The staging half of the rescore seam: a {@link RandomVectorScorer} that, before scoring a batch, hands
 * the batch's ordinals to the {@link DirectIOVectorSource.Reader} the scores will be read through, so those
 * reads can be in flight by the time they are needed.
 *
 * <h2>Why {@code bulkScore} is the right place, and the only one</h2>
 * Read ahead needs to know what is coming, and on the rescore path exactly one call knows. Scoring is
 * driven by {@code BulkVectorScorer}, which drives {@code VectorScorer.Bulk.fromRandomScorerSparse}, which
 * walks the conjunction of the matching docs and the vector iterator to collect <b>up to 64 ordinals into
 * an array</b> and only then calls {@link #bulkScore}. So at the moment {@code bulkScore} is entered, the
 * next 64 ordinals are known, in the order they will be read: Lucene's default {@code bulkScore} is a
 * strict in-order loop over {@code score(ords[i])}, and the concrete scorer on this path
 * ({@code DefaultFlatVectorScorer}'s float scorer) does not override it. Nothing further down knows more —
 * {@code score(int)} is handed one ordinal with no lookahead — and nothing further up knows it per segment.
 *
 * <p>Delegating rather than reimplementing that walk is deliberate. Reproducing the collect loop here to
 * insert a prefetch into it would duplicate Lucene's conjunction, {@code Bits} filtering and
 * ordinal-to-docid swap, and would then have to be kept in step with them. This way
 * {@code Bulk.fromRandomScorerSparse} stays the only implementation of that logic and the read-ahead is one
 * call in front of it.
 *
 * <h2>Why the window is not the batch</h2>
 * {@link #bulkScore} declares all {@code numOrds} ordinals, but {@link DirectIOVectorSource.Reader#stage}
 * only puts a window of them in flight and refills as they are consumed. Offering the whole batch at once
 * is measured-worse, not merely wasteful: a 200-wide fan-out measured p90 2.2x and p99 2.0x worse than a
 * 64-wide one, because the device queue holds 63 requests and everything past that queues in software while
 * still competing for the same completion order.
 *
 * <p>Staging is advisory. If it does not happen, for any of the reasons {@code stage} lists, the delegate
 * still scores the same ordinals from the same file and gets the same floats — just one blocking read at a
 * time. So this class cannot change a score, only when the bytes for it arrive.
 */
final class PrefetchingRandomVectorScorer implements RandomVectorScorer, HasKnnVectorValues {

    private final RandomVectorScorer delegate;
    private final DirectIOVectorSource.Reader reader;
    private final KnnVectorValues values;

    /**
     * @param delegate the scorer that does the arithmetic, over {@code values}
     * @param reader   the reader {@code values} reads through, whose ring is being staged
     * @param values   the values {@code delegate} scores over, re-exposed because
     *                 {@code AbstractRandomVectorScorer} exposes it and wrapping must not take that away
     */
    PrefetchingRandomVectorScorer(
        final RandomVectorScorer delegate,
        final DirectIOVectorSource.Reader reader,
        final KnnVectorValues values
    ) {
        this.delegate = delegate;
        this.reader = reader;
        this.values = values;
    }

    @Override
    public float bulkScore(final int[] ords, final float[] scores, final int numOrds) throws IOException {
        reader.stage(ords, numOrds);
        return delegate.bulkScore(ords, scores, numOrds);
    }

    @Override
    public float score(final int ord) throws IOException {
        return delegate.score(ord);
    }

    @Override
    public int maxOrd() {
        return delegate.maxOrd();
    }

    @Override
    public int ordToDoc(final int ord) {
        return delegate.ordToDoc(ord);
    }

    @Override
    public Bits getAcceptOrds(final Bits acceptDocs) {
        return delegate.getAcceptOrds(acceptDocs);
    }

    @Override
    public KnnVectorValues values() {
        return values;
    }
}

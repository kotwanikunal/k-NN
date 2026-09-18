/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.scorers;

import org.apache.lucene.codecs.lucene95.HasIndexSlice;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.codecs.hnsw.FlatVectorScorerUtil;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.VectorScorer;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.opensearch.knn.index.store.DirectIOVectorSource;

import java.io.IOException;

/**
 * The {@link FloatVectorValues} the rescore path scores through when Direct I/O is engaged: identical to
 * the codec's values in every respect except that {@link #vectorValue(int)} reads the vector from a
 * {@link DirectIOVectorSource} instead of from Lucene's memory mapping.
 *
 * <h2>Why not implementing {@link HasIndexSlice} is the mechanism</h2>
 * Lucene binds {@code Lucene99MemorySegmentFloatVectorScorer} - which reads fp32 vectors straight out of
 * a {@code MemorySegment} and never calls {@code vectorValue(int)} at all - when the values it is given
 * implement {@link HasIndexSlice} over a whole-file {@code MemorySegmentAccessInput}. Phase 0 confirmed
 * that scorer binds on the default rescore path, so a wrapper that kept the slice would be handed to that
 * scorer, scored out of the mapping, and this class' reads would never run. Not implementing
 * {@link HasIndexSlice} fails the first term of that predicate, so Lucene falls through to the scorer that
 * calls {@code vectorValue(int)} per ordinal.
 *
 * <p>That fall-through keeps SIMD: it lands on {@code DefaultFlatVectorScorer}, which scores through
 * {@code VectorUtil}, which dispatches to Lucene's Panama implementation. What is given up is zero-copy
 * access to the mapping, measured at +0.46 ms median on 200 queries against dio-1m - inside the 1.5 ms
 * budget the design allows. Do not "fix" this by re-exposing the slice: doing so silently restores the
 * mmap reads this path exists to remove, and the only symptom is that the Direct I/O counters stay flat.
 *
 * <h2>Ordinals, copies and buffers</h2>
 * Everything except {@code vectorValue} delegates, so ordinals, {@code ordToDoc} and iteration are the
 * codec's and cannot drift from them. {@link #copy()} pairs a fresh delegate copy with a fresh
 * {@link DirectIOVectorSource.Reader}, because Lucene gives each per-leaf scoring task its own copy and
 * those tasks run concurrently - one buffer shared between them would be a data race. The buffer is
 * allocated on the reader's first read and dropped when it becomes unreachable; nothing is retained
 * between reads, so this is a staging area and not a cache.
 */
final class DirectIOFloatVectorValues extends FloatVectorValues {

    private final FloatVectorValues delegate;
    private final DirectIOVectorSource source;
    private final DirectIOVectorSource.Reader reader;
    private final VectorSimilarityFunction similarityFunction;

    /**
     * @param delegate           the codec's values, which own iteration and the ordinal-to-doc mapping
     * @param source             the Direct I/O source for the same vectors, sharing {@code delegate}'s
     *                           ordinal space
     * @param similarityFunction the function to score with, taken from the field
     */
    DirectIOFloatVectorValues(
        final FloatVectorValues delegate,
        final DirectIOVectorSource source,
        final VectorSimilarityFunction similarityFunction
    ) {
        this.delegate = delegate;
        this.source = source;
        this.reader = source.newReader();
        this.similarityFunction = similarityFunction;
    }

    /**
     * Whether {@code source} can stand in for {@code values}: same number of vectors, same dimension and
     * same on-disk vector size. A mismatch means the source was verified against a different field or a
     * different segment generation and must not be used, so the seam falls back.
     */
    static boolean isCompatible(final FloatVectorValues values, final DirectIOVectorSource source) {
        return source.size() == values.size()
            && source.dimension() == values.dimension()
            && source.vectorByteLength() == values.getVectorByteLength();
    }

    @Override
    public float[] vectorValue(final int ord) throws IOException {
        return reader.read(ord);
    }

    @Override
    public FloatVectorValues copy() throws IOException {
        return new DirectIOFloatVectorValues(delegate.copy(), source, similarityFunction);
    }

    @Override
    public int dimension() {
        return delegate.dimension();
    }

    @Override
    public int size() {
        return delegate.size();
    }

    @Override
    public int getVectorByteLength() {
        return delegate.getVectorByteLength();
    }

    @Override
    public VectorEncoding getEncoding() {
        return delegate.getEncoding();
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
    public DocIndexIterator iterator() {
        return delegate.iterator();
    }

    /**
     * Mirrors {@code OffHeapFloatVectorValues.DenseOffHeapVectorValues#scorer}: score over a private copy,
     * so the iterator this scorer advances is not shared with whoever else holds these values, and so the
     * reads go through a buffer no other thread touches.
     *
     * <p>{@code rescorer(float[])} inherits this, because {@link FloatVectorValues#rescorer(float[])}
     * delegates to {@code scorer}. The rescore path is the only caller that reaches here.
     */
    @Override
    public VectorScorer scorer(final float[] target) throws IOException {
        final FloatVectorValues scoringCopy = copy();
        final DocIndexIterator iterator = scoringCopy.iterator();
        final RandomVectorScorer randomVectorScorer = FlatVectorScorerUtil.getLucene99FlatVectorsScorer()
            .getRandomVectorScorer(similarityFunction, scoringCopy, target);
        return new VectorScorer() {
            @Override
            public float score() throws IOException {
                return randomVectorScorer.score(iterator.index());
            }

            @Override
            public DocIdSetIterator iterator() {
                return iterator;
            }

            @Override
            public Bulk bulk(final DocIdSetIterator matchingDocs) {
                return Bulk.fromRandomScorerSparse(randomVectorScorer, iterator, matchingDocs);
            }
        };
    }
}

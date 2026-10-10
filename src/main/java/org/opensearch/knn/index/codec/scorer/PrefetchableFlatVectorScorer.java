/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.scorer;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.hnsw.RandomVectorScorerSupplier;

import java.io.IOException;

/**
 * A Prefetchable {@link FlatVectorsScorer} that wraps another scorer and adds prefetching capabilities for doing
 * search. For all other cases it just delegates to the underlying scorer.
 */
@Log4j2
public class PrefetchableFlatVectorScorer implements FlatVectorsScorer {

    private final FlatVectorsScorer delegateScorer;

    /**
     * Constructs a new prefetchable scorer wrapper.
     *
     * @param delegateScorer the underlying scorer to delegate to
     */
    public PrefetchableFlatVectorScorer(final FlatVectorsScorer delegateScorer) {
        this.delegateScorer = delegateScorer;
    }

    @Override
    public RandomVectorScorerSupplier getRandomVectorScorerSupplier(
        VectorSimilarityFunction similarityFunction,
        KnnVectorValues vectorValues
    ) throws IOException {
        return delegateScorer.getRandomVectorScorerSupplier(similarityFunction, vectorValues);
    }

    @Override
    public RandomVectorScorer getRandomVectorScorer(
        VectorSimilarityFunction similarityFunction,
        KnnVectorValues vectorValues,
        float[] target
    ) throws IOException {
        final RandomVectorScorer.AbstractRandomVectorScorer delegate = (RandomVectorScorer.AbstractRandomVectorScorer) delegateScorer
            .getRandomVectorScorer(similarityFunction, vectorValues, target);
        // The one place the vector cache's key is in reach: the scope the caller installed is live for the
        // duration of this call and nowhere else. See VectorCacheScope.
        return new PrefetchableRandomVectorScorer(
            delegate,
            VectorCacheBatchScorer.createOrNull(similarityFunction, vectorValues, target, delegate)
        );
    }

    @Override
    public RandomVectorScorer getRandomVectorScorer(
        VectorSimilarityFunction similarityFunction,
        KnnVectorValues vectorValues,
        byte[] target
    ) throws IOException {
        // No cache on the byte path: the cache holds full-precision vectors, and a byte target means the
        // delegate is scoring something else.
        return new PrefetchableRandomVectorScorer(
            (RandomVectorScorer.AbstractRandomVectorScorer) delegateScorer.getRandomVectorScorer(similarityFunction, vectorValues, target),
            null
        );
    }

    @Override
    public String toString() {
        return "PrefetchableFlatVectorScorer()";
    }

    /**
     * A {@link RandomVectorScorer} that prefetches vector data before bulk scoring operations during search,
     * and serves what the node's vector cache already holds without reading it again.
     *
     * <p>This scorer delegates all operations to an underlying scorer, but intercepts {@link #bulkScore} and
     * {@link #score} to look the candidates up in the cache, prefetch and read only the misses, and fill the
     * cache with them. Where the cache cannot serve this scorer, which is every scorer built outside a
     * {@code VectorCacheScope} and every scorer whose kernel is not a full-precision float compare, both
     * methods behave exactly as they did before: prefetch everything, then delegate.
     */
    @Log4j2
    public static class PrefetchableRandomVectorScorer extends RandomVectorScorer.AbstractRandomVectorScorer {

        private final RandomVectorScorer.AbstractRandomVectorScorer delegate;

        /** The cache path for this scorer, or {@code null} when the cache cannot serve it at all. */
        private final VectorCacheBatchScorer cached;

        /**
         * Constructs a new prefetchable random vector scorer.
         *
         * @param delegate the underlying scorer to delegate to
         */
        public PrefetchableRandomVectorScorer(final RandomVectorScorer.AbstractRandomVectorScorer delegate) {
            this(delegate, null);
        }

        /**
         * Constructs a new prefetchable random vector scorer.
         *
         * @param delegate the underlying scorer to delegate to
         * @param cached the vector cache path for this scorer, or {@code null} for none
         */
        public PrefetchableRandomVectorScorer(
            final RandomVectorScorer.AbstractRandomVectorScorer delegate,
            final VectorCacheBatchScorer cached
        ) {
            super(delegate.values());
            this.delegate = delegate;
            this.cached = cached;
        }

        @Override
        public float score(int node) throws IOException {
            if (cached != null && cached.engages(node)) {
                return cached.score(node);
            }
            return delegate.score(node);
        }

        /**
         * Scores multiple nodes, serving the cached ones from the vector cache and prefetching the rest.
         *
         * @param nodes    array of node ordinals to score
         * @param scores   output array for computed scores
         * @param numNodes number of nodes to score
         * @return the maximum score computed
         * @throws IOException if an I/O error occurs
         */
        @Override
        public float bulkScore(int[] nodes, float[] scores, int numNodes) throws IOException {
            if (numNodes > 0 && cached != null && cached.engages(nodes[0])) {
                return cached.bulkScore(nodes, scores, numNodes);
            }
            PrefetchableVectorValuesHelper.doPrefetch(values(), nodes, numNodes);
            return delegate.bulkScore(nodes, scores, numNodes);
        }

        @Override
        public int maxOrd() {
            return delegate.maxOrd();
        }

        @Override
        public int ordToDoc(int ord) {
            return delegate.ordToDoc(ord);
        }

        @Override
        public Bits getAcceptOrds(Bits acceptDocs) {
            return delegate.getAcceptOrds(acceptDocs);
        }

        @Override
        public KnnVectorValues values() {
            return delegate.values();
        }
    }
}

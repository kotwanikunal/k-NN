/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN9120Codec;

import java.lang.foreign.MemorySegment;

import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.codecs.lucene95.OffHeapFloatVectorValues;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.index.KnnVectorValues;

import org.apache.lucene.store.MemorySegmentAccessInput;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.hnsw.RandomVectorScorerSupplier;
import org.opensearch.knn.index.KNNVectorSimilarityFunction;
import org.opensearch.knn.plugin.script.KNNScoringUtil;

import java.io.IOException;

public class KNN9120NativeLuceneScorer implements FlatVectorsScorer {

    @Override
    public RandomVectorScorerSupplier getRandomVectorScorerSupplier(
        VectorSimilarityFunction similarityFunction,
        KnnVectorValues vectorValues
    ) throws IOException {
        return new NativeLuceneRandomVectorScorerSupplier((FloatVectorValues) vectorValues);
    }

    @Override
    public RandomVectorScorer getRandomVectorScorer(
        VectorSimilarityFunction similarityFunction,
        KnnVectorValues vectorValues,
        float[] target
    ) throws IOException {
        return new NativeLuceneVectorScorer((FloatVectorValues) vectorValues, target);
    }

    @Override
    public RandomVectorScorer getRandomVectorScorer(
        VectorSimilarityFunction similarityFunction,
        KnnVectorValues vectorValues,
        byte[] target
    ) throws IOException {
        throw new IllegalArgumentException("native lucene vectors do not support byte[] targets");
    }

    static class NativeLuceneVectorScorer implements RandomVectorScorer {
        private final FloatVectorValues vectorValues;
        private final float[] queryVector;
        private final int dimension;
        private final int FLOAT_SZ = 4;
        private final boolean isOffHeap;
        private final long queryVectorAddress;
        // private final Arena arena;
        // private final MemorySegment segment;

        NativeLuceneVectorScorer(FloatVectorValues vectorValues, float[] query) {
            this.isOffHeap = vectorValues instanceof OffHeapFloatVectorValues;
            this.queryVector = query;
            this.vectorValues = vectorValues;
            this.dimension = vectorValues.dimension();

            if (this.isOffHeap) {
                // this.arena = Arena.ofAuto();
                // this.segment = arena.allocate(this.dimension * Float.BYTES);
                // MemorySegment.copy(
                // MemorySegment.ofArray(query), // source
                // 0, // source offset
                // segment, // target
                // 0, // target offset
                // this.dimension * Float.BYTES // bytes to copy
                // );
                // this.queryVectorAddress = segment.address();
                this.queryVectorAddress = KNNScoringUtil.allocatePinnedQueryVector(query, this.dimension);
            } else {
                this.queryVectorAddress = 0;
                // this.arena = null;
                // this.segment = null;
            }
        }

        @Override
        public float score(int node) throws IOException {
            // vectorValues are not always offheap.
            // for instance due to deferred segment creation the vectorValues are on-heap during indexing.
            // from testing it seems that vectorValues are only off-heap during search.
            if (this.isOffHeap) {
                // access vectorValues slice.
                MemorySegmentAccessInput slice = (MemorySegmentAccessInput) ((OffHeapFloatVectorValues) vectorValues).getSlice();
                // get MemorySegment from slice.
                MemorySegment seg = slice.segmentSliceOrNull(0, slice.length());
                // see
                // https://github.com/apache/lucene/blob/27079706ef1f8341b2033efde767e95045c91f6c/lucene/core/src/java21/org/apache/lucene/internal/vectorization/PanamaVectorizationProvider.java#L91

                long baseSegmentAddress = seg.address();
                // given segment's address, find the actual vector's address.
                // Since MemorySegments are contiguous, this is probably given by
                // address + node * sizeof float * dimension
                long vectorAddress = baseSegmentAddress + (long) node * FLOAT_SZ * vectorValues.dimension();
                return KNNScoringUtil.innerProductScaledNativeOffHeapPinnedQuery(this.queryVectorAddress, vectorAddress, dimension);
            } else {
                // need a way to avoid allocating the memory here for the query vector.
                // vectors are on java heap so do not call JNI function and waste a copy.
                return KNNVectorSimilarityFunction.MAXIMUM_INNER_PRODUCT.compare(queryVector, vectorValues.vectorValue(node));
            }
        }

        @Override
        public int maxOrd() {
            return vectorValues.size();
        }

        @Override
        public int ordToDoc(int ord) {
            return vectorValues.ordToDoc(ord);
        }
    }

    static class NativeLuceneRandomVectorScorerSupplier implements RandomVectorScorerSupplier {
        protected final FloatVectorValues vectorValues;
        protected final FloatVectorValues vectorValues1;
        protected final FloatVectorValues vectorValues2;

        public NativeLuceneRandomVectorScorerSupplier(FloatVectorValues vectorValues) throws IOException {
            this.vectorValues = vectorValues;
            this.vectorValues1 = vectorValues.copy();
            this.vectorValues2 = vectorValues.copy();
        }

        @Override
        public RandomVectorScorer scorer(int ord) throws IOException {
            float[] queryVector = vectorValues1.vectorValue(ord);
            return new KNN9120NativeLuceneScorer.NativeLuceneVectorScorer(vectorValues2, queryVector);
        }

        // TODO: issues here?
        @Override
        public RandomVectorScorerSupplier copy() throws IOException {
            return new KNN9120NativeLuceneScorer.NativeLuceneRandomVectorScorerSupplier(vectorValues.copy());
        }
    }
}

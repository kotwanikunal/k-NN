/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN9120Codec;

//import jdk.incubator.foreign.MemorySegment;
//import jdk.incubator.foreign.MemorySegment;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.Arena;
import java.lang.foreign.ValueLayout;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.codecs.lucene95.OffHeapFloatVectorValues;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
//import org.apache.lucene.util.hnsw.RandomAccessVectorValues;
import org.apache.lucene.index.KnnVectorValues;

import org.apache.lucene.store.MemorySegmentAccessInput;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.hnsw.RandomVectorScorerSupplier;
import org.opensearch.knn.index.KNNVectorSimilarityFunction;
import org.opensearch.knn.plugin.script.KNNScoringUtil;
import org.opensearch.knn.ffm.FFMInterface;

import java.io.IOException;

public class KNN9120NativeLuceneScorer implements FlatVectorsScorer {
       // TODO Finn
    // extends Closeable

    @Override
    public RandomVectorScorerSupplier getRandomVectorScorerSupplier(VectorSimilarityFunction similarityFunction, KnnVectorValues vectorValues) throws IOException {
        return new NativeLuceneRandomVectorScorerSupplier((FloatVectorValues) vectorValues);
    }

    @Override
    public RandomVectorScorer getRandomVectorScorer(VectorSimilarityFunction similarityFunction, KnnVectorValues vectorValues, float[] target) throws IOException {
        return new NativeLuceneVectorScorer((FloatVectorValues) vectorValues, target);
    }

    @Override
    public RandomVectorScorer getRandomVectorScorer(VectorSimilarityFunction similarityFunction, KnnVectorValues vectorValues, byte[] target) throws IOException {
        throw new IllegalArgumentException("native lucene vectors do not support byte[] targets");
    }

    static class NativeLuceneVectorScorer implements RandomVectorScorer {
        private final FloatVectorValues vectorValues;
        private final float[] queryVector;
        private final MemorySegment queryVectorMemorySegment;
        private final int dimension;
        private final int FLOAT_SZ = 4;
        NativeLuceneVectorScorer(FloatVectorValues vectorValues, float[] query) {
            this.queryVector = query;
            this.vectorValues = vectorValues;
            this.dimension = vectorValues.dimension();
            int BYTE_ALIGN = 8; // TODO check this. should maybe be 4?
            this.queryVectorMemorySegment = Arena.ofAuto().allocate(FLOAT_SZ * dimension, BYTE_ALIGN);
            MemorySegment castedQueryVector = MemorySegment.ofArray(query); // TODO probably more efficient way to get
            // the queryVector values off-heap. Here we use ofArray() call as it is linked to the thread's lifetime so
            // it will be GCed and not leaked. Alternatively we could implement Closeable and deallocate the memory
            // there.

            MemorySegment.copy(
                    castedQueryVector,                          // Source array
                    0,                                   // Source offset
                    this.queryVectorMemorySegment,            // Destination memory segment
                    0,                                   // Destination offset
                    this.dimension * Float.BYTES // Number of bytes to copy
            );
        }

        @Override
        public float score(int node) throws IOException {
            // vectorValues are not always offheap.
            // for instance due to deferred segment creation the vectorValues are on-heap during indexing.
            // from testing it seems that vectorValues are only off-heap during search.
            if (vectorValues instanceof OffHeapFloatVectorValues) {
                // access vectorValues slice.
                MemorySegmentAccessInput slice = (MemorySegmentAccessInput) ((OffHeapFloatVectorValues) vectorValues).getSlice();
                // get MemorySegment from slice.
                MemorySegment seg = slice.segmentSliceOrNull(0, slice.length());
                // see https://github.com/apache/lucene/blob/27079706ef1f8341b2033efde767e95045c91f6c/lucene/core/src/java21/org/apache/lucene/internal/vectorization/PanamaVectorizationProvider.java#L91

                long baseSegmentAddress = seg.address();
                // given segment's address, find the actual vector's address.
                // Since MemorySegments are contiguous, this is probably given by
                // address + node * sizeof float * dimension
                long vectorAddress = baseSegmentAddress + (long) node * FLOAT_SZ * vectorValues.dimension();

                return KNNScoringUtil.innerProductScaledNativeOffHeapPinnedQuery(
                        queryVectorMemorySegment.address(),
                        vectorAddress, dimension
                );
            } else {
                // vectors are on java heap so do not call JNI function and waste a copy.
                return KNNVectorSimilarityFunction.MAXIMUM_INNER_PRODUCT.compare(queryVector, vectorValues.vectorValue(node));
            }

            // **UNFINISHED** FFM Implementation
//            if (vectorValues instanceof OffHeapFloatVectorValues) {
//                MemorySegmentAccessInput slice = (MemorySegmentAccessInput) ((OffHeapFloatVectorValues) vectorValues).getSlice();
//                MemorySegment seg = slice.segmentSliceOrNull(0, slice.length());
//                long vectorAddress = seg.address() + (long) node * 4 * vectorValues.dimension();
//
//                return FFMInterface.computeInnerProduct(
//                        queryVectorMemorySegment.address(),
//                        vectorAddress,
//                        vectorValues.dimension()
//                );
//            } else {
//                return KNNVectorSimilarityFunction.MAXIMUM_INNER_PRODUCT.compare(queryVector, vectorValues.vectorValue(node));
//            }
        }
//        }

        // TODO Finn -- probably unnecessary since GC will clean up queryVectorMemorySegment,
        //  but we should consider if we need to manually deallocate the queryVectorMemorySegment.
//        @Override
//        public void close() {
//
//        }
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

        @Override
        public RandomVectorScorerSupplier copy() throws IOException {
            return new KNN9120NativeLuceneScorer.NativeLuceneRandomVectorScorerSupplier(vectorValues.copy());
        }
    }
}
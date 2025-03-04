/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN9120Codec;

//import jdk.incubator.foreign.MemorySegment;
//import jdk.incubator.foreign.MemorySegment;

import java.lang.foreign.MemorySegment;

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

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.io.IOException;

public class KNN9120NativeLuceneScorer implements FlatVectorsScorer {
    // TODO Finn
    // extends Closeable
    // private static final Claner CLEANER = Cleaner.create();

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
        // going to take finalizer approach
        // grab resources
        // allocate memorysegment

        // use jni to pin the queryVector, native function returns mem addr
        // NewNativeLuceneVectorScorer((FloatVectorValues) vectorValues, target_addr);
        return new NativeLuceneVectorScorer((FloatVectorValues) vectorValues, target);
        // use jni to release the queryVector address memory.
        // on kNN Search, collector/query level drop the query vector.
        // kNNWeight
        // auto release resources
        // somehow with some java feature have the memorysegment auto close/deallocate.
    }

    @Override
    public RandomVectorScorer getRandomVectorScorer(
        VectorSimilarityFunction similarityFunction,
        KnnVectorValues vectorValues,
        byte[] target
    ) throws IOException {
        throw new IllegalArgumentException("native lucene vectors do not support byte[] targets");
    }

    // static class to wrap the memorysegment.
    // queryVector -> memory location. (think about overhead)

    static class NativeLuceneVectorScorer implements RandomVectorScorer {
        private final FloatVectorValues vectorValues;
        // private final Arena arena; // inner static class
        private final float[] queryVector;
        private final int dimension;
        private final int FLOAT_SZ = 4;
        private final boolean isOffHeap;
        // private final MemorySegment queryVectorMemorySegment;
        Logger logger = LogManager.getLogger(NativeLuceneVectorScorer.class);
        private final long queryVectorAddress;

        NativeLuceneVectorScorer(FloatVectorValues vectorValues, float[] query) {
            this.isOffHeap = vectorValues instanceof OffHeapFloatVectorValues;
            this.queryVector = query;
            this.vectorValues = vectorValues;
            this.dimension = vectorValues.dimension();
            int BYTE_ALIGN = 8; // TODO check this. should maybe be 4?
            // logger.info("IN CTOR");
            if (this.isOffHeap) {

                this.queryVectorAddress = KNNScoringUtil.allocatePinnedQueryVector(query, this.dimension);
                // see if we can do memory allocation through JNI, and hold.
                // explicitly call .close()?
                // this.arena = Arena.ofAuto();
                // this.arena.close() -> error
                // this.queryVectorMemorySegment = arena.allocateFrom(ValueLayout.JAVA_FLOAT, query);

            } else {
                this.queryVectorAddress = 0;
                // this.arena = null;
                // this.queryVectorMemorySegment = null;
            }

            // this.queryVectorMemorySegment = this.arena.allocate(FLOAT_SZ * dimension, BYTE_ALIGN);

            // MemorySegment castedQueryVector = MemorySegment.ofArray(query); // TODO probably more efficient way to get
            // // the queryVector values off-heap. Here we use ofArray() call as it is linked to the thread's lifetime so
            // // it will be GCed and not leaked. Alternatively we could implement Closeable and deallocate the memory
            //
            // // there.
            // MemorySegment.copy(
            // castedQueryVector, // Source array
            // 0, // Source offset
            // this.queryVectorMemorySegment, // Destination memory segment
            // 0, // Destination offset
            // this.dimension * Float.BYTES // Number of bytes to copy
            // );

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
                // return KNNScoringUtil.innerProductScaledNativeOffHeap(queryVector, vectorAddress);
                // return KNNScoringUtil.innerProductScaledNativeOffHeapPinnedQuery(
                // queryVectorMemorySegment.address(),
                // vectorAddress,
                // dimension
                // );
                return KNNScoringUtil.innerProductScaledNativeOffHeapPinnedQuery(this.queryVectorAddress, vectorAddress, dimension);
            } else {
                // need a way to avoid allocating the memory here for the query vector.
                // vectors are on java heap so do not call JNI function and waste a copy.
                return KNNVectorSimilarityFunction.MAXIMUM_INNER_PRODUCT.compare(queryVector, vectorValues.vectorValue(node));
            }

            // **UNFINISHED** FFM Implementation
            // if (vectorValues instanceof OffHeapFloatVectorValues) {
            // MemorySegmentAccessInput slice = (MemorySegmentAccessInput) ((OffHeapFloatVectorValues) vectorValues).getSlice();
            // MemorySegment seg = slice.segmentSliceOrNull(0, slice.length());
            // long vectorAddress = seg.address() + (long) node * 4 * vectorValues.dimension();
            //
            // return FFMInterface.computeInnerProduct(
            // queryVectorMemorySegment.address(),
            // vectorAddress,
            // vectorValues.dimension()
            // );
            // } else {
            // return KNNVectorSimilarityFunction.MAXIMUM_INNER_PRODUCT.compare(queryVector, vectorValues.vectorValue(node));
            // }
        }
        // }

        // TODO Finn -- probably unnecessary since GC will clean up queryVectorMemorySegment,
        // but we should consider if we need to manually deallocate the queryVectorMemorySegment.
        // @Override
        // public void close() {
        //
        // if (this.isOffHeap) {
        // logger.error("ABOUT TO CLOSE ARENA");
        //// this.queryVectorMemorySegment.deallocate();
        //// this.arena.close(); // wrong
        // }
        // else {
        // logger.error("ARENA NOT CLOSED");
        // }
        // }

        @Override
        protected void finalize() throws Throwable {
            logger.info("Object being finalized!!!");
            if (this.isOffHeap) {
                logger.info("ABOUT TO DEALLOCATE PINNED QUERY");
                KNNScoringUtil.deallocatePinnedQueryVector(this.queryVectorAddress);
            } else {
                logger.info("NOT DEALLOCATING PINNED QUERY");
            }
            super.finalize();
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

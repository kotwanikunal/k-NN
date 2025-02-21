/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.opensearch.knn.plugin.script.KNNScoringUtil;

/**
 * Wrapper class of VectorSimilarityFunction to support more function than what Lucene provides
 */
public enum KNNVectorSimilarityFunction {
    EUCLIDEAN(VectorSimilarityFunction.EUCLIDEAN),
    DOT_PRODUCT(VectorSimilarityFunction.DOT_PRODUCT),
    COSINE(VectorSimilarityFunction.COSINE),
    MAXIMUM_INNER_PRODUCT(VectorSimilarityFunction.MAXIMUM_INNER_PRODUCT),
    MAXIMUM_INNER_PRODUCT_NATIVE(null) {
        @Override
        public float compare(float[] v1, float[] v2) {


            return KNNScoringUtil.innerProductScaledNative(v1, v2);


        }


        @Override


        public float compare(byte[] v1, byte[] v2) {


            throw new IllegalStateException("can't binary compare in native for Maximum Inner Product Native space");


        }

//        public float compareOffHeap(float[] v1, long address) {
//            return
//        }


        @Override
        public VectorSimilarityFunction getVectorSimilarityFunction() {


            throw new IllegalStateException("VectorSimilarityFunction is not available for Maximum Inner Product Native space");


        }


    },
    HAMMING(null) {
        @Override
        public float compare(float[] v1, float[] v2) {
            throw new IllegalStateException("Hamming space is not supported with float vectors");
        }

        @Override
        public float compare(byte[] v1, byte[] v2) {
            return 1.0f / (1 + KNNScoringUtil.calculateHammingBit(v1, v2));
        }

        @Override
        public VectorSimilarityFunction getVectorSimilarityFunction() {
            // For binary vectors using Lucene engine we instead implement a custom BinaryVectorScorer
            throw new IllegalStateException("VectorSimilarityFunction is not available for Hamming space");
        }
    };

    private final VectorSimilarityFunction vectorSimilarityFunction;

    KNNVectorSimilarityFunction(final VectorSimilarityFunction vectorSimilarityFunction) {
        this.vectorSimilarityFunction = vectorSimilarityFunction;
    }

    public VectorSimilarityFunction getVectorSimilarityFunction() {
        return vectorSimilarityFunction;
    }

    public float compare(float[] var1, float[] var2) {
        return vectorSimilarityFunction.compare(var1, var2);
    }

    public float compare(byte[] var1, byte[] var2) {
        return vectorSimilarityFunction.compare(var1, var2);
    }
}

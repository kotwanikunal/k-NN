/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.nativeindex;

import lombok.AccessLevel;
import lombok.NoArgsConstructor;
import org.apache.lucene.codecs.lucene104.QuantizedByteVectorValues;
import org.apache.lucene.util.quantization.OptimizedScalarQuantizer;
import org.opensearch.knn.index.codec.nativeindex.model.BuildIndexParams;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.jni.JNIService;

import java.io.IOException;
import java.security.AccessController;
import java.security.PrivilegedAction;
import java.util.Map;

import static org.apache.lucene.search.DocIdSetIterator.NO_MORE_DOCS;
import static org.opensearch.knn.index.codec.util.KNNCodecUtil.initializeVectorValues;

/**
 * Memory-optimized build strategy for Faiss BBQ (Binary Quantized) HNSW indexes.
 *
 * <p>Expects {@link BuildIndexParams#getQuantizedVectorValuesSupplier()} to be populated by the writer
 * with pre-read quantized vectors from Lucene's on-disk format. This strategy handles:
 * <ol>
 *   <li>Initializing the native Faiss BBQ index via JNI</li>
 *   <li>Transferring quantized vectors and correction factors to off-heap memory</li>
 *   <li>Building the HNSW graph over those quantized vectors</li>
 *   <li>Writing the HNSW graph to disk (without duplicating flat vector storage)</li>
 * </ol>
 *
 * @see NativeIndexBuildStrategy
 */
@NoArgsConstructor(access = AccessLevel.PRIVATE)
public class MemOptimizedBBQIndexBuildStrategy implements NativeIndexBuildStrategy {
    private static final MemOptimizedBBQIndexBuildStrategy INSTANCE = new MemOptimizedBBQIndexBuildStrategy();

    public static MemOptimizedBBQIndexBuildStrategy getInstance() {
        return INSTANCE;
    }

    @Override
    public void buildAndWriteIndex(final BuildIndexParams indexInfo) throws IOException, IndexBuildAbortedException {
        final KNNVectorValues<?> knnVectorValues = indexInfo.getKnnVectorValuesSupplier().get();
        // Advance the iterator to the first document so we can read the vector dimension.
        // Without this, dimension() may return 0 since no vector has been loaded yet.
        initializeVectorValues(knnVectorValues);

        final QuantizedByteVectorValues quantizedValues = indexInfo.getQuantizedVectorValuesSupplier().get();
        if (quantizedValues == null) {
            throw new IllegalStateException("BBQ build strategy requires quantizedVectorValuesSupplier in BuildIndexParams");
        }

        final Map<String, Object> indexParameters = indexInfo.getParameters();
        final KNNEngine knnEngine = indexInfo.getKnnEngine();
        final int quantizedVecBytes = quantizedValues.vectorValue(0).length;
        final float centroidDp = quantizedValues.getCentroidDP();

        final long indexMemoryAddress = AccessController.doPrivileged(
            (PrivilegedAction<Long>) () -> JNIService.initFaissBBQIndex(
                indexInfo.getTotalLiveDocs(),
                knnVectorValues.dimension(),
                indexParameters,
                centroidDp,
                quantizedVecBytes,
                knnEngine
            )
        );

        try {
            transferQuantizedVectors(indexMemoryAddress, quantizedValues, quantizedVecBytes, knnEngine);
            buildGraph(indexMemoryAddress, knnVectorValues, knnEngine);

            AccessController.doPrivileged((PrivilegedAction<Void>) () -> {
                JNIService.writeIndex(
                    indexInfo.getIndexOutputWithBuffer(), indexMemoryAddress, knnEngine, indexParameters, true
                );
                return null;
            });
        } catch (final Exception e) {
            JNIService.releaseBBQIndex(indexMemoryAddress, knnEngine);
            throw e;
        }
    }

    /**
     * Transfers quantized vectors and correction factors to off-heap memory in ~64KB batches.
     * Layout per vector: [binaryCode | lowerInterval | upperInterval | additionalCorrection | quantizedComponentSum]
     * All values in little-endian byte order.
     */
    private void transferQuantizedVectors(
        long indexMemoryAddress, QuantizedByteVectorValues quantizedValues, int quantizedVecBytes, KNNEngine knnEngine
    ) throws IOException {
        final int oneBlockSize = quantizedVecBytes + Integer.BYTES * 4;
        final int batchSize = Math.max(1, 65536 / oneBlockSize);
        byte[] buffer = null;

        for (int i = 0; i < quantizedValues.size();) {
            final int loopSize = Math.min(quantizedValues.size() - i, batchSize);
            int o = 0;
            for (int j = 0; j < loopSize; ++j) {
                final byte[] binaryVector = quantizedValues.vectorValue(i + j);
                if (buffer == null) {
                    buffer = new byte[(binaryVector.length + Integer.BYTES * 4) * batchSize];
                }
                final OptimizedScalarQuantizer.QuantizationResult qr = quantizedValues.getCorrectiveTerms(i + j);

                System.arraycopy(binaryVector, 0, buffer, o, binaryVector.length);
                o += binaryVector.length;
                o = writeLittleEndianFloat(buffer, o, qr.lowerInterval());
                o = writeLittleEndianFloat(buffer, o, qr.upperInterval());
                o = writeLittleEndianFloat(buffer, o, qr.additionalCorrection());
                o = writeLittleEndianInt(buffer, o, qr.quantizedComponentSum());
            }
            JNIService.passBBQVectorsWithCorrectionFactors(indexMemoryAddress, buffer, loopSize, knnEngine);
            i += loopSize;
        }
    }

    private void buildGraph(long indexMemoryAddress, KNNVectorValues<?> knnVectorValues, KNNEngine knnEngine) throws IOException {
        final int batchSize = 16 * 1024;
        final int[] docIds = new int[batchSize];
        int numAdded = 0;
        while (knnVectorValues.docId() != NO_MORE_DOCS) {
            int i = 0;
            while (i < batchSize && knnVectorValues.docId() != NO_MORE_DOCS) {
                docIds[i++] = knnVectorValues.docId();
                knnVectorValues.nextDoc();
            }
            JNIService.addDocsToBBQIndex(indexMemoryAddress, docIds, i, numAdded, knnEngine);
            numAdded += i;
        }
    }

    private static int writeLittleEndianFloat(byte[] buffer, int offset, float value) {
        return writeLittleEndianInt(buffer, offset, Float.floatToRawIntBits(value));
    }

    private static int writeLittleEndianInt(byte[] buffer, int offset, int value) {
        buffer[offset++] = (byte) (value);
        buffer[offset++] = (byte) (value >>> 8);
        buffer[offset++] = (byte) (value >>> 16);
        buffer[offset++] = (byte) (value >>> 24);
        return offset;
    }
}

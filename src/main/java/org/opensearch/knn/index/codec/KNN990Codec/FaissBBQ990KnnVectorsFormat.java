/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN990Codec;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorsFormat;
import org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorsFormat.ScalarEncoding;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.engine.KNNEngine;

import java.io.IOException;

/**
 * Vector format for Faiss BBQ (Binary Quantized) fields. Uses Lucene's 1-bit scalar quantization
 * ({@link Lucene104ScalarQuantizedVectorsFormat} with {@link ScalarEncoding#SINGLE_BIT_QUERY_NIBBLE})
 * for flat vector storage while delegating HNSW graph building to native Faiss engine.
 */
@Log4j2
public class FaissBBQ990KnnVectorsFormat extends KnnVectorsFormat {

    private static final String FORMAT_NAME = "FaissBBQ990KnnVectorsFormat";
    private static final Lucene104ScalarQuantizedVectorsFormat bbqFlatFormat =
        new Lucene104ScalarQuantizedVectorsFormat(ScalarEncoding.SINGLE_BIT_QUERY_NIBBLE);
    private final int approximateThreshold;

    public FaissBBQ990KnnVectorsFormat() {
        this(KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD_DEFAULT_VALUE);
    }

    public FaissBBQ990KnnVectorsFormat(int approximateThreshold) {
        super(FORMAT_NAME);
        this.approximateThreshold = approximateThreshold;
    }

    @Override
    public KnnVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
        return new FaissBBQ990KnnVectorsWriter(state, bbqFlatFormat.fieldsWriter(state), approximateThreshold);
    }

    @Override
    public KnnVectorsReader fieldsReader(SegmentReadState state) throws IOException {
        return new FaissBBQ990KnnVectorsReader(state, bbqFlatFormat.fieldsReader(state));
    }

    @Override
    public int getMaxDimensions(String fieldName) {
        return KNNEngine.getMaxDimensionByEngine(KNNEngine.LUCENE);
    }

    @Override
    public String toString() {
        return "FaissBBQ990KnnVectorsFormat(name=" + this.getClass().getSimpleName()
            + ", approximateThreshold=" + approximateThreshold + ")";
    }
}

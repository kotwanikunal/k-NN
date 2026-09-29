/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import java.util.Locale;

import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.lucene104.Lucene104HnswScalarQuantizedVectorsFormat;
import org.apache.lucene.util.quantization.QuantizedByteVectorValues.ScalarEncoding;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsReader;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsWriter;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.search.TaskExecutor;
import org.opensearch.knn.index.codec.KNNRescoreVectorsReader;
import org.opensearch.knn.index.engine.KNNEngine;

import java.io.IOException;
import java.util.concurrent.ExecutorService;

import static org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat.DEFAULT_BEAM_WIDTH;
import static org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat.DEFAULT_MAX_CONN;
import static org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat.DEFAULT_NUM_MERGE_WORKER;
import static org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsFormat.HNSW_GRAPH_THRESHOLD;

/**
 * HNSW + scalar quantization format that extends {@link Lucene104HnswScalarQuantizedVectorsFormat}
 * and overrides flat vector operations to use {@link KNN1040ScalarQuantizedVectorsFormat},
 * inheriting its SIMD-accelerated {@link KNN1040ScalarQuantizedVectorScorer} for graph traversal scoring.
 */
public class KNN1040HnswScalarQuantizedVectorsFormat extends Lucene104HnswScalarQuantizedVectorsFormat {

    private final int maxConn;
    private final int beamWidth;
    private final int tinySegmentsThreshold;
    private final int numMergeWorkers;
    private final TaskExecutor mergeExec;
    private final KNN1040ScalarQuantizedVectorsFormat flatVectorsFormat;

    public KNN1040HnswScalarQuantizedVectorsFormat() {
        this(ScalarEncoding.SINGLE_BIT_QUERY_NIBBLE, DEFAULT_MAX_CONN, DEFAULT_BEAM_WIDTH, DEFAULT_NUM_MERGE_WORKER, null);
    }

    public KNN1040HnswScalarQuantizedVectorsFormat(
        ScalarEncoding encoding,
        int maxConn,
        int beamWidth,
        int numMergeWorkers,
        ExecutorService mergeExec
    ) {
        this(encoding, maxConn, beamWidth, numMergeWorkers, mergeExec, HNSW_GRAPH_THRESHOLD);
    }

    public KNN1040HnswScalarQuantizedVectorsFormat(
        ScalarEncoding encoding,
        int maxConn,
        int beamWidth,
        int numMergeWorkers,
        ExecutorService mergeExec,
        int tinySegmentsThreshold
    ) {
        super(encoding, maxConn, beamWidth, numMergeWorkers, mergeExec, tinySegmentsThreshold);
        this.maxConn = maxConn;
        this.beamWidth = beamWidth;
        this.tinySegmentsThreshold = tinySegmentsThreshold;
        this.numMergeWorkers = numMergeWorkers;
        this.mergeExec = mergeExec != null ? new TaskExecutor(mergeExec) : null;
        this.flatVectorsFormat = new KNN1040ScalarQuantizedVectorsFormat(encoding);
    }

    @Override
    public KnnVectorsWriter fieldsWriter(SegmentWriteState state) throws IOException {
        return new Lucene99HnswVectorsWriter(
            state,
            maxConn,
            beamWidth,
            flatVectorsFormat,
            flatVectorsFormat.fieldsWriter(state),
            numMergeWorkers,
            mergeExec,
            tinySegmentsThreshold
        );
    }

    /**
     * Returns the stock HNSW reader wrapped so that this row can offer a rescore view of the
     * full-precision {@code .vec} vectors.
     *
     * <p>The wrapper is the only plugin-owned object on the path from the segment to this row's {@code .vec}
     * — {@link Lucene99HnswVectorsReader} is final and the flat reader it holds is private — and it changes
     * nothing that Lucene returns: see {@link KNN1040RescoreAwareHnswVectorsReader} for why it is a reader
     * delegate rather than a values wrapper, and for the two interfaces merge depends on.
     *
     * <p>The view is built on the <em>raw</em> fp32 format nested inside the quantized one, not on the
     * quantized format: the rescore path reads {@code .vec} and its {@code .vemf} sidecar and never the
     * {@code .veq} codes. Creating it opens nothing.
     */
    @Override
    public KnnVectorsReader fieldsReader(SegmentReadState state) throws IOException {
        return new KNN1040RescoreAwareHnswVectorsReader(
            new Lucene99HnswVectorsReader(state, flatVectorsFormat.fieldsReader(state)),
            KNNRescoreVectorsReader.create(flatVectorsFormat.rawVectorsFormat(), state)
        );
    }

    @Override
    public int getMaxDimensions(String fieldName) {
        return KNNEngine.getMaxDimensionByEngine(KNNEngine.LUCENE);
    }

    @Override
    public String getName() {
        return getClass().getSimpleName();
    }

    @Override
    public String toString() {
        return String.format(
            Locale.ROOT,
            "%s(maxConn=%d, beamWidth=%d, tinySegmentsThreshold=%d, flatVectorFormat=%s)",
            getClass().getSimpleName(),
            maxConn,
            beamWidth,
            tinySegmentsThreshold,
            flatVectorsFormat
        );
    }
}

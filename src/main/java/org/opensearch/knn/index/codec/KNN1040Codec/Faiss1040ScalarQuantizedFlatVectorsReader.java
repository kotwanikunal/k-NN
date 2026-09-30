/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.opensearch.common.Nullable;
import org.opensearch.knn.index.codec.KNNRescoreVectorsReader;
import org.opensearch.knn.index.codec.scorer.HasRescoreVectorsReader;

import java.io.IOException;

/**
 * A {@link FlatVectorsReader} wrapper for Faiss SQ vector fields that exposes both the
 * full-precision floats and the quantized codes on the {@link FloatVectorValues} returned by
 * {@link #getFloatVectorValues(String)}.
 *
 * <p>Lucene's {@code Lucene104ScalarQuantizedVectorsReader} returns a {@code ScalarQuantizedVectorValues}
 * that hides its quantized byte delegate behind a private field. Callers on the plugin side
 * (warmup, native scoring) need both the full-precision {@code .vec} floats and the quantized
 * {@code .veq} codes, so this reader wraps the delegate's values with
 * {@link ScalarQuantizedFloatVectorValues}, which carries both delegates explicitly.
 *
 * <p>The resulting reader hierarchy is:
 * <pre>
 *   Faiss1040ScalarQuantizedKnnVectorsReader
 *     └─ Faiss1040ScalarQuantizedFlatVectorsReader  (this class)
 *          └─ Lucene104ScalarQuantizedVectorsReader  (delegate)
 * </pre>
 *
 * <p>All other operations are delegated directly to the underlying reader.
 */
@Log4j2
public class Faiss1040ScalarQuantizedFlatVectorsReader extends FlatVectorsReader implements HasRescoreVectorsReader {

    private final FlatVectorsReader delegateFlatVectorsReader;

    /**
     * The segment's second, intent-carrying view of the same full-precision vectors, or {@code null} when
     * this reader was built without one. Opens nothing until a rescore query asks for values and the
     * Direct I/O rescore setting is on.
     * <p>
     * The view needs nothing from the plugin beyond the segment it was created for, because Lucene's own
     * flat vectors format does the layout work over a {@link org.apache.lucene.store.Directory} that adds
     * the intent — which is why it reaches every rescore-capable encoding rather than one.
     */
    @Nullable
    private final KNNRescoreVectorsReader rescoreVectorsReader;

    /**
     * @param lucene104ScalarQuantizedVectorsReader the delegate reader whose {@link FloatVectorValues}
     *                                              will be wrapped to implement {@code HasIndexSlice}
     */
    protected Faiss1040ScalarQuantizedFlatVectorsReader(final FlatVectorsReader lucene104ScalarQuantizedVectorsReader) {
        this(lucene104ScalarQuantizedVectorsReader, null);
    }

    /**
     * @param lucene104ScalarQuantizedVectorsReader the delegate reader whose {@link FloatVectorValues}
     *                                              will be wrapped to implement {@code HasIndexSlice}
     * @param rescoreVectorsReader the segment's intent-carrying second view of the full-precision
     *              vectors, or {@code null} to offer none. Holding it here rather than in the
     *              {@code KnnVectorsReader} above is deliberate: this is the class whose
     *              {@link #getFloatVectorValues(String)} the rescore path reaches, so it is the class that
     *              can hand the view to the values object without anything in between having to know.
     */
    protected Faiss1040ScalarQuantizedFlatVectorsReader(
        final FlatVectorsReader lucene104ScalarQuantizedVectorsReader,
        @Nullable final KNNRescoreVectorsReader rescoreVectorsReader
    ) {
        super();
        this.delegateFlatVectorsReader = lucene104ScalarQuantizedVectorsReader;
        this.rescoreVectorsReader = rescoreVectorsReader;
    }

    /**
     * A second view of {@code field}'s full-precision vectors whose reads carry the rescore intent, or
     * {@code null} when this segment offers none — which is the default, because the Direct I/O rescore
     * setting is off by default and the view is what checks it.
     * <p>
     * Fresh values on every call, deliberately: {@link FloatVectorValues} carries a cursor and is not
     * thread safe, so they cannot be cached the way a loader source can. The shared, segment-scoped thing
     * is the reader behind them.
     */
    @Override
    @Nullable
    public FloatVectorValues rescoreVectorValues(final String field) {
        return rescoreVectorsReader == null ? null : rescoreVectorsReader.floatVectorValues(field);
    }

    @Override
    public RandomVectorScorer getRandomVectorScorer(String field, float[] target) throws IOException {
        return delegateFlatVectorsReader.getRandomVectorScorer(field, target);
    }

    @Override
    public RandomVectorScorer getRandomVectorScorer(String field, byte[] target) throws IOException {
        return delegateFlatVectorsReader.getRandomVectorScorer(field, target);
    }

    @Override
    public void checkIntegrity() throws IOException {
        delegateFlatVectorsReader.checkIntegrity();
    }

    /**
     * Returns {@link FloatVectorValues} wrapped with {@link ScalarQuantizedFloatVectorValues},
     * which exposes both the full-precision float delegate and the quantized byte delegate via
     * dedicated getters. Empty values are wrapped with no quantized backing because Lucene does
     * not expose one.
     */
    @Override
    public FloatVectorValues getFloatVectorValues(String field) throws IOException {
        final FloatVectorValues floatVectorValues = delegateFlatVectorsReader.getFloatVectorValues(field);
        if (floatVectorValues == null) {
            return null;
        }

        if (floatVectorValues.size() == 0) {
            return new ScalarQuantizedFloatVectorValues(floatVectorValues, null);
        }

        // The rescore view is deliberately NOT offered on the values. It is offered on this reader, via
        // HasRescoreVectorsReader, because two of the four rescore-reachable encodings have no plugin values
        // class to put it on and a values wrapper per encoding is the one shape this design must avoid.
        return new ScalarQuantizedFloatVectorValues(
            floatVectorValues,
            KNN1040ScalarQuantizedUtils.extractQuantizedByteVectorValues(floatVectorValues)
        );
    }

    @Override
    public ByteVectorValues getByteVectorValues(String field) throws IOException {
        return delegateFlatVectorsReader.getByteVectorValues(field);
    }

    /**
     * Closes the delegate and the rescore view. The delegate is closed even if the view fails to close,
     * since it owns every file the default path reads.
     */
    @Override
    public void close() throws IOException {
        try {
            if (rescoreVectorsReader != null) {
                try {
                    rescoreVectorsReader.close();
                } catch (IOException e) {
                    log.warn("Failed to close the rescore view of the full-precision vectors", e);
                }
            }
        } finally {
            delegateFlatVectorsReader.close();
        }
    }

    @Override
    public long ramBytesUsed() {
        return delegateFlatVectorsReader.ramBytesUsed();
    }

    @Override
    public FlatVectorsScorer getFlatVectorScorer(String field) throws IOException {
        return delegateFlatVectorsReader.getFlatVectorScorer(field);
    }

}

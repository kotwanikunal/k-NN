/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.Getter;
import org.apache.lucene.util.quantization.QuantizedByteVectorValues;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.search.VectorScorer;
import org.opensearch.common.Nullable;
import org.opensearch.knn.index.codec.scorer.HasRescoreVectorValues;
import org.opensearch.knn.index.codec.scorer.HasVectorLoaderSource;
import org.opensearch.knn.index.codec.scorer.HasFullPrecisionVectorValues;
import org.opensearch.knn.index.store.VectorLoaderSource;

import java.io.IOException;
import java.util.function.Supplier;

/**
 * A {@link FloatVectorValues} wrapper that holds both the full-precision float delegate (backed by
 * the {@code .vec} file) and the underlying quantized byte delegate (backed by the {@code .veq} file).
 *
 * <p>The wrapper exists so callers can reach either representation without reflection: the search
 * scorer needs the quantized codes; warmup needs the full-precision bytes. Consumers pick via
 * {@link #getFloatVectorValues()} or {@link #getQuantizedVectorValues()} rather than relying on a
 * single ambiguous slice.
 *
 * <p>Historically this class implemented {@code HasIndexSlice#getSlice()}. It was removed because
 * the delegating {@code vectorValue(ord)} reads full-precision floats from {@code .vec} while the
 * quantized values expose the {@code .veq} slice — a generic prefetch caller trusting
 * {@code HasIndexSlice} would warm the wrong file, defeating the fetch-phase prefetch entirely.
 * A caller that specifically wants the {@code .vec} side asks for it by name instead, through
 * {@link HasFullPrecisionVectorValues}.
 *
 * <p>For an empty vector segment, the quantized delegate may be {@code null}.
 */
@Getter
class ScalarQuantizedFloatVectorValues extends FloatVectorValues
    implements
        HasFullPrecisionVectorValues,
        HasVectorLoaderSource,
        HasRescoreVectorValues {
    /**
     * The full-precision float delegate (reads the {@code .vec} file).
     */
    private final FloatVectorValues floatVectorValues;
    /**
     * The quantized byte delegate (reads the {@code .veq} file), or {@code null} for empty
     * segments where Lucene does not expose one.
     */
    private final QuantizedByteVectorValues quantizedVectorValues;
    /**
     * The values that own the {@code .vec} slice, resolved once at construction, or {@code null} when they
     * cannot be reached. {@link #floatVectorValues} is itself a two-representation wrapper - Lucene's
     * {@code ScalarQuantizedVectorValues} - so it exposes no slice either and one more unwrap is needed to
     * reach the file.
     */
    private final KnnVectorValues fullPrecisionVectorValues;
    /**
     * Supplies the segment-scoped loader seam for the {@code .vec} file, or {@code null} when this values
     * object was built without one. Held as a supplier rather than a source so that nothing is established
     * until a query that knows its reads have no reuse asks: these values are constructed on every search,
     * the source is owned by the reader, and the reader must not open a file handle on a node whose Direct
     * I/O rescore flag is off.
     */
    @Getter(lombok.AccessLevel.NONE)
    @Nullable
    private final Supplier<VectorLoaderSource> vectorLoaderSourceSupplier;
    /**
     * Supplies a second view of the {@code .vec} vectors whose reads carry the rescore intent, or {@code null}
     * when this values object was built without one. A supplier rather than the values themselves for the same
     * reason {@link #vectorLoaderSourceSupplier} is one — nothing may be opened on a node whose Direct I/O
     * rescore setting is off — and because {@link FloatVectorValues} carry a cursor, so each rescorer needs
     * their own.
     */
    @Getter(lombok.AccessLevel.NONE)
    @Nullable
    private final Supplier<FloatVectorValues> rescoreVectorValuesSupplier;

    ScalarQuantizedFloatVectorValues(final FloatVectorValues floatVectorValues, final QuantizedByteVectorValues quantizedVectorValues) {
        this(floatVectorValues, quantizedVectorValues, null, null);
    }

    ScalarQuantizedFloatVectorValues(
        final FloatVectorValues floatVectorValues,
        final QuantizedByteVectorValues quantizedVectorValues,
        @Nullable final Supplier<VectorLoaderSource> vectorLoaderSourceSupplier
    ) {
        this(floatVectorValues, quantizedVectorValues, vectorLoaderSourceSupplier, null);
    }

    ScalarQuantizedFloatVectorValues(
        final FloatVectorValues floatVectorValues,
        final QuantizedByteVectorValues quantizedVectorValues,
        @Nullable final Supplier<VectorLoaderSource> vectorLoaderSourceSupplier,
        @Nullable final Supplier<FloatVectorValues> rescoreVectorValuesSupplier
    ) {
        this.floatVectorValues = floatVectorValues;
        this.quantizedVectorValues = quantizedVectorValues;
        this.fullPrecisionVectorValues = KNN1040ScalarQuantizedUtils.extractRawFloatVectorValues(floatVectorValues);
        this.vectorLoaderSourceSupplier = vectorLoaderSourceSupplier;
        this.rescoreVectorValuesSupplier = rescoreVectorValuesSupplier;
    }

    /**
     * The loader seam for the {@code .vec} vectors this wrapper serves through {@link #vectorValue(int)}, or
     * {@code null} when there is none. Shares this wrapper's ordinal space, for the same reason
     * {@link #getFullPrecisionVectorValues()} does.
     */
    @Override
    public VectorLoaderSource vectorLoaderSource() {
        return vectorLoaderSourceSupplier == null ? null : vectorLoaderSourceSupplier.get();
    }

    /**
     * A second view of the same {@code .vec} vectors this wrapper serves, whose reads carry the rescore
     * intent, or {@code null} when there is none. Shares this wrapper's ordinal space, because it is the
     * same file read by the same format over the same segment — only the {@code IndexInput} differs.
     */
    @Override
    public FloatVectorValues rescoreVectorValues() {
        return rescoreVectorValuesSupplier == null ? null : rescoreVectorValuesSupplier.get();
    }

    /**
     * Returns the values backed by the {@code .vec} file, so an advisory prefetch caller can warm the
     * full-precision vectors the rescore path reads rather than the quantized codes. They share this
     * wrapper's ordinal space, because every iteration method here delegates to
     * {@link #floatVectorValues}, which in turn delegates to them.
     *
     * @return the full-precision values, or {@code null} when they cannot be reached
     */
    @Override
    public KnnVectorValues getFullPrecisionVectorValues() {
        return fullPrecisionVectorValues;
    }

    @Override
    public int dimension() {
        return floatVectorValues.dimension();
    }

    @Override
    public int size() {
        return floatVectorValues.size();
    }

    @Override
    public float[] vectorValue(int ord) throws IOException {
        return floatVectorValues.vectorValue(ord);
    }

    @Override
    public FloatVectorValues copy() throws IOException {
        return new ScalarQuantizedFloatVectorValues(
            floatVectorValues.copy(),
            quantizedVectorValues == null ? null : quantizedVectorValues.copy(),
            vectorLoaderSourceSupplier,
            rescoreVectorValuesSupplier
        );
    }

    @Override
    public VectorEncoding getEncoding() {
        return floatVectorValues.getEncoding();
    }

    @Override
    public DocIndexIterator iterator() {
        return floatVectorValues.iterator();
    }

    /**
     * Returns a {@link VectorScorer} that scores the query against the quantized codes in
     * {@code .veq}. Scoring is the quantized-vs-quantized path, not the full-precision one — the
     * {@code .vec} floats are reserved for rescore/fetch via {@link #rescorer(float[])}.
     */
    @Override
    public VectorScorer scorer(float[] target) throws IOException {
        return quantizedVectorValues == null ? null : quantizedVectorValues.scorer(target);
    }

    /**
     * Returns a {@link VectorScorer} for rescoring candidates against the given target vector
     * using full-precision vectors. Delegates to the underlying {@link FloatVectorValues}.
     *
     * @param target the query vector to score against
     * @return a {@link VectorScorer} for exact rescoring, or {@code null} if not supported
     * @throws IOException if an I/O error occurs
     */
    @Override
    public VectorScorer rescorer(final float[] target) throws IOException {
        return floatVectorValues.rescorer(target);
    }
}

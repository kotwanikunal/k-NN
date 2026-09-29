/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.scorer;

import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.index.FloatVectorValues;
import org.opensearch.common.Nullable;

/**
 * Implemented by a codec reader that can offer a <em>second view of a field's full-precision vectors</em>
 * whose reads carry the rescore intent, so that the storage layer can serve them differently from the
 * traversal, merge, warmup and fetch reads of the same {@code .vec}.
 *
 * <h2>Why the capability lives on the reader and not on the values</h2>
 * The obvious place to put it is the {@link FloatVectorValues} the rescore path already holds — and that is
 * where it started. It does not generalise, because <b>two of the four rescore-reachable encodings have no
 * plugin values class to put it on</b>:
 * <ul>
 *   <li>the native engines' unquantized fp32 path hands back Lucene's {@code OffHeapFloatVectorValues}
 *       ({@code NativeEngines990KnnVectorsFormat} nests a bare {@code Lucene99FlatVectorsFormat}), and</li>
 *   <li>the Lucene-engine quantized path hands back Lucene's {@code ScalarQuantizedVectorValues} from a
 *       stock {@code Lucene104ScalarQuantizedVectorsReader}.</li>
 * </ul>
 * Reaching those from the values side means introducing a plugin {@code FloatVectorValues} wrapper per
 * encoding — exactly the per-encoding class this design exists to avoid, and on the one object the
 * traversal, merge and warmup paths also hold. Such a wrapper is also the single most dangerous shape in
 * this area: a wrapper that forgets to delegate one bulk accessor still returns <em>identical bytes</em>
 * while losing Lucene's SIMD binding, which measured 5.3x slower and which no correctness test can catch.
 *
 * <p>A reader, by contrast, is already one per field per segment, is already a plugin class (or can be
 * delegated to by one without touching any values object), and is the thing that holds the
 * {@link org.apache.lucene.index.SegmentReadState} the view has to be built from. Every values object on
 * every path stays exactly the object Lucene built.
 *
 * <h2>How the query layer reaches it</h2>
 * {@link RescoreVectorValuesSelector#select} walks a segment's {@link KnnVectorsReader} — through
 * {@code PerFieldKnnVectorsFormat.FieldsReader} to the field's own reader — and asks for the view. The
 * query layer therefore contributes the <em>what</em> (this read is a rescore read) and the storage layer
 * owns the <em>how</em>; nothing in between decodes, offsets or sizes anything.
 *
 * <h2>Contract</h2>
 * <ul>
 *   <li><b>Same bytes, same ordinals.</b> The returned values must serve exactly the vectors
 *       {@link KnnVectorsReader#getFloatVectorValues(String)} serves for the same field, at the same
 *       ordinals, from the same file. They differ only in which {@code IndexInput} the bytes arrive
 *       through. Nothing here permits a transformation, and the safety of treating every failure as a
 *       silent fallback depends on that.</li>
 *   <li><b>Lazy.</b> Nothing may be opened until this method is first called. The Direct I/O rescore
 *       setting is dynamic and off by default, and a node with it off must not acquire a second handle on
 *       {@code .vec}.</li>
 *   <li><b>Fresh per call.</b> {@link FloatVectorValues} carries a cursor and is not thread safe, so each
 *       caller gets its own. The <em>reader</em> behind them is the shared, segment-scoped thing.</li>
 * </ul>
 *
 * <p>{@code null} is an ordinary answer, not an error: it means this segment cannot offer the second view
 * — the setting is off, the directory has no rescore route, the format did not recognise the field — and
 * the caller reads the vectors the way it would have anyway.
 *
 * @see org.opensearch.knn.index.codec.KNNRescoreVectorsReader
 * @see RescoreVectorValuesSelector
 */
public interface HasRescoreVectorsReader {

    /**
     * A second view of {@code field}'s full-precision vectors whose reads carry the rescore intent, or
     * {@code null} when this reader cannot offer one.
     *
     * @param field the vector field
     * @return values over the same bytes and the same ordinals, or {@code null} to mean "use the default
     *         path"
     */
    @Nullable
    FloatVectorValues rescoreVectorValues(String field);
}

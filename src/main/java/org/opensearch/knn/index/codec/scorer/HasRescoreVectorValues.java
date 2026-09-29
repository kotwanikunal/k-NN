/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.scorer;

import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.KnnVectorValues;

/**
 * Implemented by {@link KnnVectorValues} that can offer a <em>second view of the same
 * full-precision vectors</em> whose reads carry the rescore intent, so that the storage layer can serve
 * them differently from the traversal, merge, warmup and fetch reads of the same file.
 *
 * <h2>How this differs from {@link HasVectorLoaderSource}, which it is meant to replace</h2>
 * {@link HasVectorLoaderSource} hands the query layer a <em>loader</em> — a plugin-owned file handle,
 * a plugin-owned offset calculation, and a plugin-owned notion of an entry size — which the query layer
 * then has to wrap the values in. That is why it has exactly one implementor: every encoding whose
 * {@code .vec} layout has not been reverse-engineered by hand is outside it.
 *
 * <p>This interface hands back {@link FloatVectorValues} instead, and the values are Lucene's own,
 * built by the segment's own flat vectors format over a {@link org.apache.lucene.store.Directory} view
 * that adds the intent. The plugin therefore contributes no decoding, no offset arithmetic and no entry
 * sizing, and one mechanism covers every encoding whose full-precision vectors live in a {@code .vec} —
 * scalar quantization at any code width, the native engines' unquantized fp32, and the Lucene-engine
 * quantized format alike.
 *
 * <h2>Contract</h2>
 * <ul>
 *   <li><b>Same bytes, same ordinals.</b> The returned values must serve exactly the vectors this values
 *       object serves, at the same ordinals, from the same file. They differ only in which
 *       {@code IndexInput} the bytes arrive through. Nothing in this contract permits a transformation,
 *       and the safety of treating every failure as a silent fallback depends on that.</li>
 *   <li><b>Lazy.</b> Nothing may be opened until this method is first called. The Direct I/O rescore
 *       setting is dynamic and off by default, and a node with it off must not acquire a second handle
 *       on {@code .vec}; that is what makes "setting off is bit-identical to today" a property of the
 *       code rather than a claim about it.</li>
 *   <li><b>Fresh per call.</b> Unlike a loader source, which is segment-scoped and shared, the values
 *       returned here are per-caller: {@link FloatVectorValues} carries a cursor and is not thread
 *       safe. The <em>reader</em> behind them is the shared, segment-scoped thing.</li>
 * </ul>
 *
 * <p>{@code null} is an ordinary answer, not an error: it means this segment cannot offer the second
 * view — the setting is off, the directory has no rescore route, the format did not recognise the
 * segment — and the caller reads the vectors the way it would have anyway.
 *
 * @see HasVectorLoaderSource
 * @see org.opensearch.knn.index.codec.KNNRescoreVectorsReader
 */
public interface HasRescoreVectorValues {

    /**
     * A second view of these values' full-precision vectors whose reads carry the rescore intent, or
     * {@code null} when this segment cannot offer one.
     *
     * @return values over the same bytes and the same ordinals, or {@code null} to mean "use the
     *         default path"
     */
    FloatVectorValues rescoreVectorValues();
}

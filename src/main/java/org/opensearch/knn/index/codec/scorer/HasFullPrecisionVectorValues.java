/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.scorer;

import org.apache.lucene.codecs.lucene95.HasIndexSlice;
import org.apache.lucene.index.KnnVectorValues;

/**
 * Implemented by a {@link KnnVectorValues} wrapper that fronts more than one on-disk representation of the
 * same vectors, to name the full-precision one explicitly.
 * <p>
 * A quantized field is served by two files - the quantized codes and the full-precision floats - and a
 * wrapper over both cannot implement {@link HasIndexSlice}, because it has two slices and a caller asking
 * for "the" slice would get an arbitrary one. Warming the wrong file is worse than not warming at all:
 * it evicts the pages that were wanted. So those wrappers deliberately do not implement
 * {@link HasIndexSlice}, and a prefetch caller that only understands {@link HasIndexSlice} silently
 * declines on them.
 * <p>
 * This interface is the narrow, explicit way out: a caller that specifically wants the full-precision
 * vectors - which is what the rescore path reads - asks for them by name. The returned values are the ones
 * that own the full-precision slice, so the caller can go on to use {@link HasIndexSlice} and
 * {@link KnnVectorValues#getVectorByteLength()} on them as usual.
 * <p>
 * The returned values share the implementor's ordinal space, so an ordinal computed against the wrapper is
 * valid against them.
 *
 * @see PrefetchableVectorValuesHelper#doPrefetch
 */
public interface HasFullPrecisionVectorValues {

    /**
     * Returns the values that read the full-precision vectors, or {@code null} when they cannot be
     * reached. A {@code null} return is not an error: the caller declines to prefetch, exactly as it does
     * for values that expose no slice at all.
     *
     * @return the full-precision values, in the implementor's own ordinal space, or {@code null}
     */
    KnnVectorValues getFullPrecisionVectorValues();
}

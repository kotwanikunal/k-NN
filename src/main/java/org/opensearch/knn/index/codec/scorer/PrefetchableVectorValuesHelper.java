/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.scorer;

import lombok.AccessLevel;
import lombok.NoArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.lucene95.HasIndexSlice;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.store.IndexInput;
import org.opensearch.knn.index.store.DirectIOVectorIndexInput;

import java.io.IOException;

/**
 * Utility class that performs prefetching of vector data for upcoming node accesses during graph traversal.
 * <p>
 * Prefetching hints the underlying storage to load vector data into memory ahead of time, reducing I/O latency
 * during scoring operations. This is particularly beneficial for memory-optimized (off-heap) search where vector
 * data is read directly from disk-backed index inputs.
 * <p>
 * Supports prefetching for any {@link KnnVectorValues} implementation that also implements {@link HasIndexSlice},
 * which provides access to the underlying sliced index input for prefetch operations, and for wrappers over
 * several representations of the same vectors that name their full-precision one through
 * {@link HasFullPrecisionVectorValues}.
 */
@Log4j2
@NoArgsConstructor(access = AccessLevel.PRIVATE)
class PrefetchableVectorValuesHelper {

    /**
     * How many {@link HasFullPrecisionVectorValues} unwraps to follow before giving up. One is enough for
     * every implementation in this plugin; the bound exists only so that a wrapper cycle cannot spin.
     */
    private static final int MAX_UNWRAP_DEPTH = 4;

    /**
     * Attempts to prefetch vector data for the given node ordinals.
     * <p>
     * If the provided {@link KnnVectorValues} implementation supports prefetching, this method will issue
     * prefetch hints for the specified nodes. Otherwise, it logs an informational message and returns without
     * performing any prefetch.
     * <p>
     * Values that expose no slice of their own but do implement {@link HasFullPrecisionVectorValues} are
     * unwrapped first: a wrapper over a quantized and a full-precision file cannot expose a single
     * unambiguous slice, so it names the full-precision one instead. The unwrapped values share the
     * wrapper's ordinal space, so {@code nodes} needs no translation. If unwrapping yields nothing that
     * exposes a slice, this declines and warns exactly as it does for any other unsupported type.
     *
     * @param vectorValues the vector values instance to prefetch from
     * @param nodes        array of node ordinals whose vector data should be prefetched
     * @param numNodes     number of valid entries in the {@code nodes} array to prefetch
     * @throws IOException if an I/O error occurs during prefetching
     */
    public static void doPrefetch(final KnnVectorValues vectorValues, final int[] nodes, final int numNodes) throws IOException {
        final KnnVectorValues sliced = withSlice(vectorValues);
        if (sliced == null) {
            log.warn("Not able to do prefetch on instance {}", vectorValues.getClass().getSimpleName());
            return;
        }
        // passing base offset as 0, since the index input is a slice and its base offset is 0.
        PrefetchHelper.prefetch(((HasIndexSlice) sliced).getSlice(), 0, sliced.getVectorByteLength(), nodes, numNodes);
    }

    /**
     * The values in {@code vectorValues}' unwrap chain that expose a slice, or {@code null} when none does.
     *
     * <p>Factored out of {@link #doPrefetch} so that span fill (F5) reaches the <em>same</em> input the
     * prefetch staged into. Two rules that differed would be a defect that only showed as a cache that
     * admitted nothing, so they are one rule.
     */
    static KnnVectorValues withSlice(final KnnVectorValues vectorValues) {
        KnnVectorValues candidate = vectorValues;
        for (int depth = 0; depth < MAX_UNWRAP_DEPTH && candidate != null; depth++) {
            if (candidate instanceof HasIndexSlice vectorValuesWithSlice && vectorValuesWithSlice.getSlice() != null) {
                return candidate;
            }
            if (candidate instanceof HasFullPrecisionVectorValues == false) {
                return null;
            }
            final KnnVectorValues fullPrecision = ((HasFullPrecisionVectorValues) candidate).getFullPrecisionVectorValues();
            if (fullPrecision == candidate) {
                return null;
            }
            candidate = fullPrecision;
        }
        return null;
    }

    /**
     * The {@link DirectIOVectorIndexInput} the given values read through, or {@code null} when they read
     * through anything else. Span fill is for staged spans only, so values served by mmap answer null here
     * and nothing about their admission changes.
     */
    static DirectIOVectorIndexInput directIOSliceOrNull(final KnnVectorValues vectorValues) {
        final KnnVectorValues sliced = withSlice(vectorValues);
        if (sliced == null) {
            return null;
        }
        final IndexInput slice = ((HasIndexSlice) sliced).getSlice();
        return slice instanceof DirectIOVectorIndexInput ? (DirectIOVectorIndexInput) slice : null;
    }

}

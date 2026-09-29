/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.scorer;

import lombok.AccessLevel;
import lombok.NoArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.SegmentReader;
import org.opensearch.common.Nullable;

/**
 * Finds a segment's rescore view of a field's full-precision vectors, if its codec reader offers one.
 *
 * <p>This is the whole of the bridge between the query layer's <em>what</em> signal and the storage
 * layer's <em>how</em>: given the segment reader the rescore path already holds and the field it is about
 * to score, it walks down to the field's own {@link KnnVectorsReader} and asks whether that reader can
 * hand back vectors whose reads carry the rescore intent. It reads nothing, decodes nothing and knows
 * nothing about {@code O_DIRECT}; a reader that does not implement {@link HasRescoreVectorsReader} — which
 * is every reader on a default node, and every reader for an encoding whose vectors are not an fp32
 * {@code .vec} — produces {@code null} and the caller keeps the path it was on.
 *
 * <p>The walk is the same one {@code MemoryOptimizedSearchWarmup} does: a segment's vector reader is
 * Lucene's {@code PerFieldKnnVectorsFormat.FieldsReader}, and the per-field reader behind it is the one
 * the field's own format built. Both steps are defended rather than asserted, because a caller on the
 * query path must degrade to the default read rather than fail a search if a future Lucene arranges the
 * readers differently.
 */
@Log4j2
@NoArgsConstructor(access = AccessLevel.PRIVATE)
public final class RescoreVectorValuesSelector {

    /**
     * The rescore view of {@code field}'s full-precision vectors in {@code segmentReader}'s segment, or
     * {@code null} when this segment offers none.
     *
     * <p>Calling this is what makes the view open a second handle on {@code .vec}, so it must only be
     * called once the caller has decided the read really is a rescore read — see
     * {@code DirectIORescoreSeam}, which owns that decision and is the only caller.
     *
     * @param segmentReader the segment being scored
     * @param field         the vector field being scored
     * @return values over the same bytes and ordinals as the field's ordinary values, whose reads carry the
     *         rescore intent, or {@code null} to mean "use the default path"
     */
    @Nullable
    public static FloatVectorValues select(@Nullable final SegmentReader segmentReader, @Nullable final String field) {
        if (segmentReader == null || field == null) {
            return null;
        }
        final KnnVectorsReader fieldReader = fieldReaderFor(segmentReader, field);
        if ((fieldReader instanceof HasRescoreVectorsReader) == false) {
            // The ordinary answer on a default node and on every encoding that is not an fp32 .vec.
            return null;
        }
        try {
            return ((HasRescoreVectorsReader) fieldReader).rescoreVectorValues(field);
        } catch (RuntimeException e) {
            // Both views read the same bytes, so declining can cost the isolation but never a result.
            log.warn("[KNN] rescore view lookup failed for field [{}]: {}", field, e.toString());
            return null;
        }
    }

    /**
     * The {@link KnnVectorsReader} the field's own format built, or {@code null} if it cannot be reached.
     */
    @Nullable
    private static KnnVectorsReader fieldReaderFor(final SegmentReader segmentReader, final String field) {
        final KnnVectorsReader vectorReader = segmentReader.getVectorReader();
        if (vectorReader == null) {
            return null;
        }
        if (vectorReader instanceof PerFieldKnnVectorsFormat.FieldsReader perField) {
            return perField.getFieldReader(field);
        }
        // A reader that is not per-field is already the field's own reader.
        return vectorReader;
    }
}

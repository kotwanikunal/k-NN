/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.scorers;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.opensearch.knn.common.featureflags.KNNFeatureFlags;
import org.opensearch.knn.index.codec.scorer.HasDirectIOVectorSource;
import org.opensearch.knn.index.store.DirectIOVectorSource;

/**
 * The query side seam where the rescore path may swap Lucene's mmap backed full-precision vector
 * values for a Direct I/O backed source of the same bytes.
 *
 * <p>Rescore reads are the one place where a Direct I/O bypass is worth taking: the full-precision
 * {@code .vec} vectors a rescore touches are read once per query and never reused, so keeping them out
 * of the page cache costs nothing and stops them evicting the graph and quantized vectors that
 * traversal does reuse. Traversal reads have the opposite profile and are deliberately not routed here.
 *
 * <h2>What this class gates, and why here</h2>
 * The seam sits inside {@code VectorScorers#getBaseScorer}, which is the only point on the rescore
 * chain that has all three things the decision needs at once: the {@link FloatVectorValues} for the
 * segment, the {@link VectorScorerMode} that says whether this scorer is a rescorer, and a field whose
 * vectors are known to be fp32 {@code .vec} rather than {@code BinaryDocValues} or quantized bytes.
 * Codec level hooks cannot see the mode at all.
 *
 * <h2>Three conditions, all required</h2>
 * <ol>
 *   <li><b>Flag on.</b> {@link KNNFeatureFlags#isDirectIORescoreEnabled()}, a dynamic node setting that
 *       is off by default. With it off this class returns its input unchanged, so the default path is
 *       byte for byte what it is without this code.</li>
 *   <li><b>Mode is {@link VectorScorerMode#RESCORE}.</b> This is the whole safety argument, and it must
 *       never be widened. {@code SCORE} covers primary scoring and the exact-search fallback, both of
 *       which read quantized vectors or reuse full-precision vectors across many documents.</li>
 *   <li><b>Not a radial query.</b> Radial search needs its own exclusion because the mode gate does
 *       <em>not</em> exclude it: {@code RescoreRadialSearchQuery} builds its exact-search context with
 *       {@code useQuantizedVectorsForSearch(false)}, which {@code ExactSearcher} turns into
 *       {@code RESCORE}. Radial scores every candidate that clears a threshold rather than a bounded
 *       top-k candidate set, so its read pattern is not the one measured here. Callers that cannot tell
 *       whether a query is radial say so by passing {@code radialSearch = true}, which keeps them on the
 *       default path.</li>
 * </ol>
 *
 * <h2>What the engaged branch does</h2>
 * It asks the values for a {@link org.opensearch.knn.index.store.DirectIOVectorSource} through
 * {@link HasDirectIOVectorSource} and, if it gets one, wraps them in {@link DirectIOFloatVectorValues} so
 * every vector the rescorer reads comes off the device rather than out of the page cache. Three further
 * things send a query back to the default path, and none of them is an error: values that name no source
 * (any other vector format), a source that could not be opened or verified (a compound segment, a
 * filesystem that refuses {@code O_DIRECT}), and a source whose shape does not match the values.
 *
 * <p>Reads are still one synchronous {@code pread} per vector here. The parallel fetch that makes that
 * competitive is the next phase; the bar for this one is that the ranking is unchanged.
 */
@Log4j2
public final class DirectIORescoreSeam {

    private DirectIORescoreSeam() {}

    /**
     * Whether the Direct I/O rescore path applies to the scorer about to be built. See the class
     * javadoc for why all three conditions are required.
     *
     * @param vectorScorerMode the mode the scorer is being built in
     * @param radialSearch     true if this scorer serves a radial (min-score) query, or if the caller
     *                         cannot tell
     * @return true if the reads for this scorer may be served with Direct I/O
     */
    public static boolean isEngaged(final VectorScorerMode vectorScorerMode, final boolean radialSearch) {
        if (vectorScorerMode != VectorScorerMode.RESCORE || radialSearch) {
            return false;
        }
        return KNNFeatureFlags.isDirectIORescoreEnabled();
    }

    /**
     * Returns the vector values the rescore scorer should read through.
     *
     * <p>Returns {@code values} itself whenever {@link #isEngaged} is false, so a disabled or
     * ineligible path is identical to having no seam at all rather than merely equivalent to it.
     *
     * @param values           the codec's vector values for the segment
     * @param vectorScorerMode the mode the scorer is being built in
     * @param radialSearch     true if this scorer serves a radial query, or if the caller cannot tell
     * @param fieldInfo        the vector field being scored, for logging
     * @return the values to score against
     */
    public static FloatVectorValues vectorValuesForRescore(
        final FloatVectorValues values,
        final VectorScorerMode vectorScorerMode,
        final boolean radialSearch,
        final FieldInfo fieldInfo
    ) {
        if (isEngaged(vectorScorerMode, radialSearch) == false) {
            return values;
        }
        if ((values instanceof HasDirectIOVectorSource) == false) {
            // Every other vector format keeps the default path. This is not a gap to close later: the fp32
            // .vec rescore chain the project measures resolves to values that do implement it, and a format
            // that does not is one whose file layout has not been verified.
            log.debug(
                "[KNN] Direct I/O rescore seam declined field [{}]: values [{}] name no Direct I/O source",
                fieldInfo.name,
                values.getClass().getName()
            );
            return values;
        }

        final DirectIOVectorSource source = ((HasDirectIOVectorSource) values).directIOVectorSource();
        if (source == null) {
            // The source logs its own reason once per segment; a compound segment or a filesystem that
            // refuses O_DIRECT lands here on every query and must stay cheap and quiet.
            return values;
        }
        if (DirectIOFloatVectorValues.isCompatible(values, source) == false) {
            log.warn(
                "[KNN] Direct I/O rescore seam declined field [{}]: source has {} vectors of {} bytes at dimension {}, "
                    + "values have {} of {} at {}",
                fieldInfo.name,
                source.size(),
                source.vectorByteLength(),
                source.dimension(),
                values.size(),
                values.getVectorByteLength(),
                values.dimension()
            );
            return values;
        }

        log.debug(
            "[KNN] Direct I/O rescore seam engaged for field [{}], values [{}], reading [{}]",
            fieldInfo.name,
            values.getClass().getName(),
            source.path()
        );
        return new DirectIOFloatVectorValues(values, source, fieldInfo.getVectorSimilarityFunction());
    }
}

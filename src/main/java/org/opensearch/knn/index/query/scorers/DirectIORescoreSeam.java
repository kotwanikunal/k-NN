/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.scorers;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.opensearch.common.Nullable;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.codec.scorer.HasRescoreVectorsReader;
import org.opensearch.knn.index.codec.scorer.RescoreVectorValuesSelector;

import java.util.function.Supplier;

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
 *   <li><b>Flag on.</b> {@link KNNSettings#isDirectIORescoreEnabled()}, a dynamic node setting that
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
 * <h2>What the engaged branch does — the rescore view</h2>
 * It asks the segment's codec reader ({@link HasRescoreVectorsReader}) for a second view of the same
 * {@code .vec} vectors whose reads carry the rescore intent, and returns it. The view is Lucene's own
 * {@code FloatVectorValues}, built by the segment's own flat vectors format over a
 * {@link org.apache.lucene.store.Directory} that adds the intent below the codec, so the scorer built from it
 * one line later in {@link VectorScorers} is Lucene's own too — including the shipped bulk-prefetch path — and
 * the plugin contributes no decoding, no offset arithmetic and no entry sizing for any encoding. Nothing here
 * is wrapped: the values object the traversal path holds is returned to no one and its SIMD binding cannot be
 * disturbed. The view is looked up from the <em>reader</em> rather than the values, by
 * {@link RescoreVectorValuesSelector}, because two of the four rescore-reachable encodings have no plugin
 * values class and a values wrapper per encoding is the one shape this design avoids.
 *
 * <p>Several things send a query back to the default path, and none of them is an error: a caller with no view
 * to offer, a view that could not be established (the setting off, a directory with no rescore route, a
 * filesystem that refuses {@code O_DIRECT}), and a view whose shape does not match the values.
 *
 * <p>This class knows nothing about {@code O_DIRECT}. It decides <em>whether</em> a read is a rescore read;
 * <em>how</em> the bytes then arrive is decided below it, by the storage layer.
 *
 * <h2>Why this class still exists once the view covers everything</h2>
 * The substitution itself needs no query-layer help: {@code VectorScorerMode.RESCORE} is
 * literally {@code values.rescorer(target)}, so Lucene's own API already names <em>what</em> the read is, and
 * a codec class could return the view from {@code rescorer()} with no call from here at all. Two things
 * cannot be moved, and they are the reason for the gate: <b>the radial exclusion is query-layer
 * knowledge</b>, and so is the segment reader the view has to be looked up from.
 * {@code radialSearch} is a parameter of {@code VectorScorers#getBaseScorer}, threaded from
 * {@code ExactSearcher}, and {@code rescorer()} has no way to see it — while a radial rescore does arrive in
 * {@code RESCORE} mode. So the gate stays here, where all three conditions are visible at once, and the
 * codec layer offers a capability rather than deciding to use it.
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
        return KNNSettings.isDirectIORescoreEnabled();
    }

    /**
     * Returns the vector values the rescore scorer should read through: the segment's rescore view when the
     * caller can supply one and it checks out, and {@code values} itself otherwise.
     *
     * <p>Returns {@code values} itself whenever {@link #isEngaged} is false, so a disabled or ineligible path
     * is identical to having no seam at all rather than merely equivalent to it.
     *
     * <p>The view arrives as a {@link Supplier} rather than as values, and that is load bearing: asking for
     * it is what makes a segment open a second handle on {@code .vec}, so it must not be asked for on a
     * query this seam is going to decline. Only the one call site that reaches fp32 {@code .vec} values in
     * {@link VectorScorerMode#RESCORE} — {@code ExactSearcher}, which is also the only place that holds the
     * {@code SegmentReader} the view is looked up from — supplies one; every other caller passes
     * {@code null} and keeps the default path.
     *
     * @param values             the codec's vector values for the segment
     * @param vectorScorerMode   the mode the scorer is being built in
     * @param radialSearch       true if this scorer serves a radial query, or if the caller cannot tell
     * @param fieldInfo          the vector field being scored
     * @param rescoreViewSupplier supplies the segment's rescore view of the same vectors, or {@code null}
     * @return the values to score against
     */
    public static FloatVectorValues vectorValuesForRescore(
        final FloatVectorValues values,
        final VectorScorerMode vectorScorerMode,
        final boolean radialSearch,
        final FieldInfo fieldInfo,
        @Nullable final Supplier<FloatVectorValues> rescoreViewSupplier
    ) {
        if (isEngaged(vectorScorerMode, radialSearch) == false) {
            return values;
        }

        // The view returns Lucene's own values over an intent-carrying IndexInput, so the scorer built from
        // them one line later in VectorScorers is Lucene's own too - including the shipped bulk-prefetch path
        // - and the plugin contributes no decoding for any encoding.
        final FloatVectorValues rescoreValues = rescoreVectorValues(values, fieldInfo, rescoreViewSupplier);
        return rescoreValues != null ? rescoreValues : values;
    }

    /**
     * The segment's intent-carrying view of the same full-precision vectors, or {@code null} when there is
     * none and the caller should keep the default path.
     *
     * <p>{@code null} is the ordinary answer on a node with no rescore-aware directory installed, on a
     * caller that supplies no view, on a codec reader that offers none, and on any failure below — the
     * view's whole safety argument is that both views read the same bytes, so declining it can only cost
     * the isolation, never a result.
     *
     * <p>The shape check is not defensive boilerplate. The view is built by handing the segment's own flat
     * vectors format a different {@link org.apache.lucene.store.Directory}, so a view that disagrees with
     * the values on ordinal count, dimension or entry size means the two are not reading the same segment's
     * vectors — a wrong {@code segmentSuffix}, a format that resolved a different field — and scoring
     * through it would silently score the wrong vectors. Lucene catches most of that loudly at
     * {@code CodecUtil.checkIndexHeader}; this catches the rest quietly and declines.
     */
    @Nullable
    private static FloatVectorValues rescoreVectorValues(
        final FloatVectorValues values,
        final FieldInfo fieldInfo,
        @Nullable final Supplier<FloatVectorValues> rescoreViewSupplier
    ) {
        if (rescoreViewSupplier == null) {
            return null;
        }
        final FloatVectorValues rescoreValues = rescoreViewSupplier.get();
        if (rescoreValues == null) {
            // The view logs its own reason once per segment. With the setting off this is every query, so
            // it must stay cheap and quiet.
            return null;
        }
        if (rescoreValues.size() != values.size()
            || rescoreValues.dimension() != values.dimension()
            || rescoreValues.getVectorByteLength() != values.getVectorByteLength()
            || rescoreValues.getEncoding() != values.getEncoding()) {
            log.warn(
                "[KNN] Direct I/O rescore declined the rescore view for field [{}]: view has {} vectors of {} bytes at "
                    + "dimension {} encoded {}, values have {} of {} at {} encoded {}",
                fieldInfo.name,
                rescoreValues.size(),
                rescoreValues.getVectorByteLength(),
                rescoreValues.dimension(),
                rescoreValues.getEncoding(),
                values.size(),
                values.getVectorByteLength(),
                values.dimension(),
                values.getEncoding()
            );
            return null;
        }
        log.debug(
            "[KNN] Direct I/O rescore engaged the rescore view for field [{}], values [{}], view [{}]",
            fieldInfo.name,
            values.getClass().getName(),
            rescoreValues.getClass().getName()
        );
        return rescoreValues;
    }
}

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.scorers;

import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.mockito.Mock;
import org.opensearch.common.settings.ClusterSettings;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.vectorvalues.TestVectorValues;

import java.util.List;
import java.util.function.Supplier;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.index.KNNSettings.KNN_DIRECT_IO_RESCORE_ENABLED_SETTING;

/**
 * Pins the three conditions of the Direct I/O rescore gate, that the seam is a pass-through in every case
 * that fails one of them, and that the one case which passes all of them scores through the segment's
 * rescore view.
 */
public class DirectIORescoreSeamTests extends KNNTestCase {

    private static final List<float[]> VECTORS = List.of(new float[] { 1.0f, 2.0f });

    @Mock
    ClusterSettings clusterSettings;

    private final FieldInfo fieldInfo = mock(FieldInfo.class);

    @Override
    public void setUp() throws Exception {
        super.setUp();
        when(clusterService.getClusterSettings()).thenReturn(clusterSettings);
        KNNSettings.state().setClusterService(clusterService);
    }

    private void setFlag(final boolean enabled) {
        when(clusterSettings.get(KNN_DIRECT_IO_RESCORE_ENABLED_SETTING)).thenReturn(enabled);
    }

    private static FloatVectorValues values() {
        return new TestVectorValues.PreDefinedFloatVectorValues(VECTORS);
    }

    public void testIsEngaged_whenFlagIsOff_thenFalseInEveryMode() {
        setFlag(false);
        assertFalse(DirectIORescoreSeam.isEngaged(VectorScorerMode.RESCORE, false));
        assertFalse(DirectIORescoreSeam.isEngaged(VectorScorerMode.RESCORE, true));
        assertFalse(DirectIORescoreSeam.isEngaged(VectorScorerMode.SCORE, false));
        assertFalse(DirectIORescoreSeam.isEngaged(VectorScorerMode.SCORE, true));
    }

    public void testIsEngaged_whenFlagIsOnAndModeIsRescore_thenTrue() {
        setFlag(true);
        assertTrue(DirectIORescoreSeam.isEngaged(VectorScorerMode.RESCORE, false));
    }

    /**
     * The gate is the entire safety argument, so SCORE must not reach the Direct I/O path. SCORE covers
     * primary scoring and the exact-search fallback, whose reads are reused across documents.
     */
    public void testIsEngaged_whenModeIsScore_thenFalseEvenWithFlagOn() {
        setFlag(true);
        assertFalse(DirectIORescoreSeam.isEngaged(VectorScorerMode.SCORE, false));
    }

    /**
     * Radial rescore arrives with mode RESCORE, so only the explicit radial flag keeps it out.
     */
    public void testIsEngaged_whenRadial_thenFalseEvenWithFlagOnAndRescoreMode() {
        setFlag(true);
        assertFalse(DirectIORescoreSeam.isEngaged(VectorScorerMode.RESCORE, true));
    }

    /** A mode that is neither of the two constants (a custom implementation) is not a rescore. */
    public void testIsEngaged_whenModeIsNeitherConstant_thenFalse() {
        setFlag(true);
        assertFalse(DirectIORescoreSeam.isEngaged(mock(VectorScorerMode.class), false));
    }

    /** No cluster service yet - the flag must read as its default rather than throwing. */
    public void testIsEngaged_whenClusterServiceIsNotSet_thenDefaultsToOff() {
        KNNSettings.state().setClusterService(null);
        assertFalse(DirectIORescoreSeam.isEngaged(VectorScorerMode.RESCORE, false));
    }

    /**
     * With the flag off the seam hands back what it was given in every mode, so a disabled path is identical
     * to having no seam at all rather than merely equivalent to it.
     */
    public void testVectorValuesForRescore_whenFlagIsOff_thenReturnsSameInstance() {
        setFlag(false);
        final FloatVectorValues values = values();
        assertSame(values, DirectIORescoreSeam.vectorValuesForRescore(values, VectorScorerMode.RESCORE, false, fieldInfo, () -> values()));
        assertSame(values, DirectIORescoreSeam.vectorValuesForRescore(values, VectorScorerMode.SCORE, false, fieldInfo, () -> values()));
    }

    public void testVectorValuesForRescore_whenRadial_thenReturnsSameInstance() {
        setFlag(true);
        final FloatVectorValues values = values();
        assertSame(values, DirectIORescoreSeam.vectorValuesForRescore(values, VectorScorerMode.RESCORE, true, fieldInfo, () -> values()));
    }

    /**
     * A caller that offers no view at all — every {@code createScorer} overload except the one
     * {@code ExactSearcher} uses — keeps the default path. This is what makes the parameter a no-op for every
     * other caller.
     */
    public void testVectorValuesForRescore_whenNoSupplierIsGiven_thenReturnsSameInstance() {
        setFlag(true);
        final FloatVectorValues values = values();
        assertSame(values, DirectIORescoreSeam.vectorValuesForRescore(values, VectorScorerMode.RESCORE, false, fieldInfo, null));
    }

    /**
     * No view on offer — the setting off, a directory with no rescore route, a codec reader that offers none —
     * is an ordinary answer of {@code null}, not an error, and must not disturb the query.
     */
    public void testVectorValuesForRescore_whenTheSupplierOffersNoView_thenReturnsSameInstance() {
        setFlag(true);
        final FloatVectorValues values = values();
        assertSame(values, DirectIORescoreSeam.vectorValuesForRescore(values, VectorScorerMode.RESCORE, false, fieldInfo, () -> null));
    }

    /**
     * The engaged case, and the only place the whole chain is asserted end to end. The seam returns the view
     * itself — not a wrapper around anything — so the scorer built from it one line later in
     * {@code VectorScorers} is Lucene's own.
     */
    public void testVectorValuesForRescore_whenAViewIsOffered_thenReturnsTheViewUnwrapped() {
        setFlag(true);
        final FloatVectorValues view = values();

        final FloatVectorValues values = values();
        final FloatVectorValues chosen = DirectIORescoreSeam.vectorValuesForRescore(
            values,
            VectorScorerMode.RESCORE,
            false,
            fieldInfo,
            () -> view
        );

        assertSame(view, chosen);
    }

    /**
     * A view that disagrees with the values on ordinal count is not reading the same segment's vectors — a
     * wrong segment suffix, a format that resolved a different field — and scoring through it would silently
     * score the wrong vectors. The seam declines and neither route engages.
     */
    public void testVectorValuesForRescore_whenTheViewShapeDisagrees_thenReturnsSameInstance() {
        setFlag(true);
        final FloatVectorValues shorter = new TestVectorValues.PreDefinedFloatVectorValues(List.of());
        final FloatVectorValues values = values();

        assertSame(values, DirectIORescoreSeam.vectorValuesForRescore(values, VectorScorerMode.RESCORE, false, fieldInfo, () -> shorter));
    }

    /**
     * The two conditions that cannot be seen from inside the codec layer still gate the view: with the
     * setting off, and on a radial query, the view is never even asked for. That matters beyond tidiness —
     * asking is what opens a second handle on {@code .vec} — and the radial half is the reason this class
     * still exists, since a codec-level hook has no way to see that flag.
     */
    public void testVectorValuesForRescore_whenFlagOffOrRadial_thenTheViewIsNeverAsked() {
        final FloatVectorValues values = values();
        final CountingViewSupplier supplier = new CountingViewSupplier(values());

        setFlag(false);
        assertSame(values, DirectIORescoreSeam.vectorValuesForRescore(values, VectorScorerMode.RESCORE, false, fieldInfo, supplier));
        assertEquals(0, supplier.timesAsked);

        setFlag(true);
        assertSame(values, DirectIORescoreSeam.vectorValuesForRescore(values, VectorScorerMode.RESCORE, true, fieldInfo, supplier));
        assertEquals(0, supplier.timesAsked);

        assertSame(values, DirectIORescoreSeam.vectorValuesForRescore(values, VectorScorerMode.SCORE, false, fieldInfo, supplier));
        assertEquals(0, supplier.timesAsked);
    }

    /**
     * Counts how many times the view was asked for, because "never opened with the setting off" is a property
     * of who calls whom, not of what is returned.
     */
    private static final class CountingViewSupplier implements Supplier<FloatVectorValues> {

        private final FloatVectorValues view;
        private int timesAsked;

        private CountingViewSupplier(final FloatVectorValues view) {
            this.view = view;
        }

        @Override
        public FloatVectorValues get() {
            timesAsked++;
            return view;
        }
    }
}

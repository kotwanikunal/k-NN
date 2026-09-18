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

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.common.featureflags.KNNFeatureFlags.KNN_DIRECT_IO_RESCORE_ENABLED_SETTING;

/**
 * Pins the three conditions of the Direct I/O rescore gate, and that the seam is a pass-through in
 * every one of the resulting cases.
 */
public class DirectIORescoreSeamTests extends KNNTestCase {

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
        return new TestVectorValues.PreDefinedFloatVectorValues(List.of(new float[] { 1.0f, 2.0f }));
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

    public void testVectorValuesForRescore_whenFlagIsOff_thenReturnsSameInstance() {
        setFlag(false);
        final FloatVectorValues values = values();
        assertSame(values, DirectIORescoreSeam.vectorValuesForRescore(values, VectorScorerMode.RESCORE, false, fieldInfo));
        assertSame(values, DirectIORescoreSeam.vectorValuesForRescore(values, VectorScorerMode.SCORE, false, fieldInfo));
    }

    /**
     * The engaged branch is a pass-through for now, which is what makes "switching the flag on cannot
     * change a score" a property of the code rather than of a measurement.
     */
    public void testVectorValuesForRescore_whenEngaged_thenStillReturnsSameInstance() {
        setFlag(true);
        final FloatVectorValues values = values();
        assertSame(values, DirectIORescoreSeam.vectorValuesForRescore(values, VectorScorerMode.RESCORE, false, fieldInfo));
    }

    public void testVectorValuesForRescore_whenRadial_thenReturnsSameInstance() {
        setFlag(true);
        final FloatVectorValues values = values();
        assertSame(values, DirectIORescoreSeam.vectorValuesForRescore(values, VectorScorerMode.RESCORE, true, fieldInfo));
    }
}

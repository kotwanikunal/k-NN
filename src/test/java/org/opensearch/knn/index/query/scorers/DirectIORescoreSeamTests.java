/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.scorers;

import lombok.SneakyThrows;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.mockito.Mock;
import org.opensearch.common.settings.ClusterSettings;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.codec.scorer.HasDirectIOVectorSource;
import org.opensearch.knn.index.store.DirectIOVectorSource;
import org.opensearch.knn.index.vectorvalues.TestVectorValues;

import java.util.List;

import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.common.featureflags.KNNFeatureFlags.KNN_DIRECT_IO_RESCORE_ENABLED_SETTING;

/**
 * Pins the three conditions of the Direct I/O rescore gate, that the seam is a pass-through in every case
 * that fails one of them, and that the one case which passes all of them reads through
 * {@link DirectIOFloatVectorValues}.
 */
public class DirectIORescoreSeamTests extends KNNTestCase {

    private static final int DIMENSION = 2;
    private static final List<float[]> VECTORS = List.of(new float[] { 1.0f, 2.0f });
    private static final int SIZE = VECTORS.size();
    /** What the stub source answers, chosen to differ from {@link #VECTORS} so a swap is observable. */
    private static final float[] SOURCE_VECTOR = { 30.0f, 40.0f };

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

    public void testVectorValuesForRescore_whenFlagIsOff_thenReturnsSameInstance() {
        setFlag(false);
        final FloatVectorValues values = values();
        assertSame(values, DirectIORescoreSeam.vectorValuesForRescore(values, VectorScorerMode.RESCORE, false, fieldInfo));
        assertSame(values, DirectIORescoreSeam.vectorValuesForRescore(values, VectorScorerMode.SCORE, false, fieldInfo));
    }

    /**
     * Any vector format whose values do not name a Direct I/O source keeps the default path. There is no
     * fallback to arrange here: the seam simply hands back what it was given.
     */
    public void testVectorValuesForRescore_whenValuesNameNoSource_thenReturnsSameInstance() {
        setFlag(true);
        final FloatVectorValues values = values();
        assertSame(values, DirectIORescoreSeam.vectorValuesForRescore(values, VectorScorerMode.RESCORE, false, fieldInfo));
    }

    public void testVectorValuesForRescore_whenRadial_thenReturnsSameInstance() {
        setFlag(true);
        final FloatVectorValues values = values();
        assertSame(values, DirectIORescoreSeam.vectorValuesForRescore(values, VectorScorerMode.RESCORE, true, fieldInfo));
    }

    /**
     * A source that could not be opened or verified - a compound segment, a filesystem that refuses
     * {@code O_DIRECT} - is an ordinary answer of {@code null}, not an error, and must not disturb the query.
     */
    public void testVectorValuesForRescore_whenSourceIsNull_thenReturnsSameInstance() {
        setFlag(true);
        final FloatVectorValues values = new SourceNamingValues(null);
        assertSame(values, DirectIORescoreSeam.vectorValuesForRescore(values, VectorScorerMode.RESCORE, false, fieldInfo));
    }

    /**
     * A source whose shape disagrees with the values was verified against something else, so reading
     * through it would score the wrong vectors. The seam falls back rather than trusting it.
     */
    public void testVectorValuesForRescore_whenSourceShapeDisagrees_thenReturnsSameInstance() {
        setFlag(true);
        final DirectIOVectorSource source = source(SIZE + 1, DIMENSION, DIMENSION * Float.BYTES);
        final FloatVectorValues values = new SourceNamingValues(source);
        assertSame(values, DirectIORescoreSeam.vectorValuesForRescore(values, VectorScorerMode.RESCORE, false, fieldInfo));
    }

    /**
     * The one case that swaps the values, and the only place the whole chain is asserted end to end: with
     * the flag on, a matching source, and a non-radial rescore, reads go through
     * {@code DirectIOFloatVectorValues}.
     */
    @SneakyThrows
    public void testVectorValuesForRescore_whenSourceMatches_thenReadsThroughDirectIO() {
        setFlag(true);
        when(fieldInfo.getVectorSimilarityFunction()).thenReturn(VectorSimilarityFunction.EUCLIDEAN);
        final FloatVectorValues values = new SourceNamingValues(source(SIZE, DIMENSION, DIMENSION * Float.BYTES));

        final FloatVectorValues wrapped = DirectIORescoreSeam.vectorValuesForRescore(values, VectorScorerMode.RESCORE, false, fieldInfo);

        assertNotSame(values, wrapped);
        assertTrue(wrapped.getClass().getSimpleName(), wrapped instanceof DirectIOFloatVectorValues);
        assertArrayEquals(SOURCE_VECTOR, wrapped.vectorValue(0), 0.0f);
        assertEquals(values.size(), wrapped.size());
        assertEquals(values.dimension(), wrapped.dimension());
    }

    /**
     * The same values that would be swapped with the flag on are handed straight back with it off. This is
     * the "flag off is identical to no seam at all" guarantee, asserted where it can actually fail.
     */
    @SneakyThrows
    public void testVectorValuesForRescore_whenFlagIsOffAndSourceMatches_thenReturnsSameInstance() {
        setFlag(false);
        final FloatVectorValues values = new SourceNamingValues(source(SIZE, DIMENSION, DIMENSION * Float.BYTES));
        assertSame(values, DirectIORescoreSeam.vectorValuesForRescore(values, VectorScorerMode.RESCORE, false, fieldInfo));
    }

    /** A stub source of a given shape whose reader answers {@link #SOURCE_VECTOR} for every ordinal. */
    @SneakyThrows
    private static DirectIOVectorSource source(final int size, final int dimension, final int vectorByteLength) {
        final DirectIOVectorSource source = mock(DirectIOVectorSource.class);
        when(source.size()).thenReturn(size);
        when(source.dimension()).thenReturn(dimension);
        when(source.vectorByteLength()).thenReturn(vectorByteLength);
        when(source.newReader()).thenAnswer(invocation -> {
            final DirectIOVectorSource.Reader reader = mock(DirectIOVectorSource.Reader.class);
            when(reader.read(anyInt())).thenReturn(SOURCE_VECTOR);
            return reader;
        });
        return source;
    }

    /**
     * Stands in for {@code ScalarQuantizedFloatVectorValues}: the shape of {@link #values()} plus the one
     * interface the seam looks for.
     */
    private static final class SourceNamingValues extends TestVectorValues.PreDefinedFloatVectorValues implements HasDirectIOVectorSource {

        private final DirectIOVectorSource source;

        private SourceNamingValues(final DirectIOVectorSource source) {
            super(VECTORS);
            this.source = source;
        }

        @Override
        public DirectIOVectorSource directIOVectorSource() {
            return source;
        }
    }
}

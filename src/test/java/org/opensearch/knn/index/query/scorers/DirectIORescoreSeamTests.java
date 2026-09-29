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
import org.opensearch.knn.index.codec.scorer.HasVectorLoaderSource;
import org.opensearch.knn.index.store.VectorLoaderSource;
import org.opensearch.knn.index.vectorvalues.TestVectorValues;

import java.util.List;
import java.util.function.Supplier;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.index.KNNSettings.KNN_DIRECT_IO_RESCORE_ENABLED_SETTING;

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
        final VectorLoaderSource source = source(SIZE + 1, DIMENSION, DIMENSION * Float.BYTES);
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
     * The mode is both the gate and the loader seam's reuse hint, and the hint half is easy to drop: nothing
     * in this plugin reads it, so only this assertion notices if it stops arriving. A future cache at the
     * loader seam needs it to know that rescore reads have no reuse and must not be retained.
     */
    @SneakyThrows
    public void testVectorValuesForRescore_whenEngaged_thenPassesTheModeDownAsTheReuseHint() {
        setFlag(true);
        when(fieldInfo.getVectorSimilarityFunction()).thenReturn(VectorSimilarityFunction.EUCLIDEAN);
        final VectorLoaderSource source = source(SIZE, DIMENSION, DIMENSION * Float.BYTES);

        DirectIORescoreSeam.vectorValuesForRescore(new SourceNamingValues(source), VectorScorerMode.RESCORE, false, fieldInfo);

        verify(source).newLoader(VectorScorerMode.RESCORE);
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

    // ---------------------------------------------------------------------------------------------------
    // The rescore view route. Preferred over the loader seam because it is Lucene's own values over an
    // intent-carrying IndexInput: no plugin decoding, no offset arithmetic, and one mechanism per file
    // rather than one per encoding.
    // ---------------------------------------------------------------------------------------------------

    /**
     * The engaged case for the new route. The seam returns the view itself — not a wrapper around anything —
     * so the scorer built from it one line later in {@code VectorScorers} is Lucene's own.
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
     * When both routes are on offer the view wins. It has to: it is the route that reaches every encoding,
     * and running both would put two Direct I/O mechanisms on the same reads.
     */
    public void testVectorValuesForRescore_whenBothRoutesAreOffered_thenTheViewWins() {
        setFlag(true);
        final FloatVectorValues view = values();
        final FloatVectorValues values = new SourceNamingValues(source(SIZE, DIMENSION, DIMENSION * Float.BYTES));

        final FloatVectorValues chosen = DirectIORescoreSeam.vectorValuesForRescore(
            values,
            VectorScorerMode.RESCORE,
            false,
            fieldInfo,
            () -> view
        );

        assertSame(view, chosen);
        assertFalse(chosen instanceof DirectIOFloatVectorValues);
    }

    /**
     * No view — the setting off, a directory with no rescore route, a codec reader that offers none — falls
     * through to the loader seam rather than to the default path, so the encoding the seam already covered
     * keeps working while the view is being rolled out across the others.
     */
    @SneakyThrows
    public void testVectorValuesForRescore_whenNoViewButASourceMatches_thenFallsBackToTheLoaderSeam() {
        setFlag(true);
        when(fieldInfo.getVectorSimilarityFunction()).thenReturn(VectorSimilarityFunction.EUCLIDEAN);
        final FloatVectorValues values = new SourceNamingValues(source(SIZE, DIMENSION, DIMENSION * Float.BYTES));

        final FloatVectorValues chosen = DirectIORescoreSeam.vectorValuesForRescore(
            values,
            VectorScorerMode.RESCORE,
            false,
            fieldInfo,
            () -> null
        );

        assertTrue(chosen.getClass().getSimpleName(), chosen instanceof DirectIOFloatVectorValues);
        assertArrayEquals(SOURCE_VECTOR, chosen.vectorValue(0), 0.0f);
    }

    /**
     * A caller that offers no supplier at all - every {@code createScorer} overload except the one
     * {@code ExactSearcher} uses - reaches the loader seam exactly as before. This is what makes adding the
     * parameter a no-op for every existing caller.
     */
    @SneakyThrows
    public void testVectorValuesForRescore_whenNoSupplierIsGiven_thenBehavesAsBefore() {
        setFlag(true);
        when(fieldInfo.getVectorSimilarityFunction()).thenReturn(VectorSimilarityFunction.EUCLIDEAN);
        final FloatVectorValues values = new SourceNamingValues(source(SIZE, DIMENSION, DIMENSION * Float.BYTES));

        final FloatVectorValues chosen = DirectIORescoreSeam.vectorValuesForRescore(values, VectorScorerMode.RESCORE, false, fieldInfo);

        assertTrue(chosen.getClass().getSimpleName(), chosen instanceof DirectIOFloatVectorValues);
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

    /**
     * A stub source of a given shape whose loaders answer {@link #SOURCE_VECTOR} for every ordinal.
     * <p>
     * Mocked as the {@link VectorLoaderSource} seam rather than as the Direct I/O implementation, which is
     * itself an assertion: if the seam ever needed something only the implementation offers, this would stop
     * compiling.
     */
    @SneakyThrows
    private static VectorLoaderSource source(final int size, final int dimension, final int vectorByteLength) {
        final VectorLoaderSource source = mock(VectorLoaderSource.class);
        when(source.size()).thenReturn(size);
        when(source.dimension()).thenReturn(dimension);
        when(source.vectorByteLength()).thenReturn(vectorByteLength);
        when(source.newLoader(any())).thenAnswer(invocation -> {
            final VectorLoaderSource.Loader loader = mock(VectorLoaderSource.Loader.class);
            when(loader.read(anyInt())).thenReturn(SOURCE_VECTOR);
            when(loader.reuseHint()).thenReturn(invocation.getArgument(0));
            return loader;
        });
        return source;
    }

    /**
     * Stands in for {@code ScalarQuantizedFloatVectorValues}: the shape of {@link #values()} plus the one
     * interface the seam looks for.
     */
    private static final class SourceNamingValues extends TestVectorValues.PreDefinedFloatVectorValues implements HasVectorLoaderSource {

        private final VectorLoaderSource source;

        private SourceNamingValues(final VectorLoaderSource source) {
            super(VECTORS);
            this.source = source;
        }

        @Override
        public VectorLoaderSource vectorLoaderSource() {
            return source;
        }
    }
}

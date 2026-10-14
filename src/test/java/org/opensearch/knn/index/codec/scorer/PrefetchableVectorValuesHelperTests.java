/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.scorer;

import org.apache.lucene.codecs.lucene95.OffHeapFloatVectorValues;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.store.IndexInput;
import org.mockito.MockedStatic;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.memoryoptsearch.faiss.FaissIndexScalarQuantizedFlat;
import org.opensearch.knn.memoryoptsearch.faiss.vectorvalues.FaissByteVectorValues;
import org.opensearch.knn.memoryoptsearch.faiss.vectorvalues.FaissFloatVectorValues;
import org.opensearch.knn.memoryoptsearch.faiss.vectorvalues.FaissFloatVectorValues.SparseFloatVectorValuesImpl;

import java.io.IOException;
import java.util.function.Supplier;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.mockStatic;
import static org.mockito.Mockito.when;

public class PrefetchableVectorValuesHelperTests extends KNNTestCase {

    private final int[] nodes = { 0, 1, 2 };
    private final int numNodes = 3;

    public void testDoPrefetch_whenFloatVectorValuesImpl_thenPrefetchesViaHasIndexSlice() throws IOException {
        IndexInput mockSlice = mock(IndexInput.class);
        when(mockSlice.length()).thenReturn(200L * 1024);
        int vectorByteLength = 512;

        FaissFloatVectorValues floatImpl = mock(FaissFloatVectorValues.class);
        when(floatImpl.getSlice()).thenReturn(mockSlice);
        when(floatImpl.getVectorByteLength()).thenReturn(vectorByteLength);

        try (MockedStatic<PrefetchHelper> mockedPrefetchHelper = mockStatic(PrefetchHelper.class)) {
            PrefetchableVectorValuesHelper.doPrefetch(floatImpl, nodes, numNodes);

            mockedPrefetchHelper.verify(() -> PrefetchHelper.prefetch(mockSlice, 0, vectorByteLength, nodes, numNodes));
        }
    }

    public void testDoPrefetch_whenQuantizedFloatVectorValuesImpl_thenPrefetchesViaHasIndexSlice() throws IOException {
        IndexInput mockSlice = mock(IndexInput.class);
        when(mockSlice.length()).thenReturn(200L * 1024);
        int vectorByteLength = 512;

        FaissIndexScalarQuantizedFlat.FloatVectorValuesImpl floatImpl = mock(FaissIndexScalarQuantizedFlat.FloatVectorValuesImpl.class);
        when(floatImpl.getSlice()).thenReturn(mockSlice);
        when(floatImpl.getVectorByteLength()).thenReturn(vectorByteLength);

        try (MockedStatic<PrefetchHelper> mockedPrefetchHelper = mockStatic(PrefetchHelper.class)) {
            PrefetchableVectorValuesHelper.doPrefetch(floatImpl, nodes, numNodes);

            mockedPrefetchHelper.verify(() -> PrefetchHelper.prefetch(mockSlice, 0, vectorByteLength, nodes, numNodes));
        }
    }

    public void testDoPrefetch_whenByteVectorValuesImpl_thenPrefetchesViaHasIndexSlice() throws IOException {
        IndexInput mockSlice = mock(IndexInput.class);
        when(mockSlice.length()).thenReturn(200L * 1024);
        int vectorByteLength = 64;

        FaissByteVectorValues binaryImpl = mock(FaissByteVectorValues.class);
        when(binaryImpl.getSlice()).thenReturn(mockSlice);
        when(binaryImpl.getVectorByteLength()).thenReturn(vectorByteLength);

        try (MockedStatic<PrefetchHelper> mockedPrefetchHelper = mockStatic(PrefetchHelper.class)) {
            PrefetchableVectorValuesHelper.doPrefetch(binaryImpl, nodes, numNodes);

            mockedPrefetchHelper.verify(() -> PrefetchHelper.prefetch(mockSlice, 0, vectorByteLength, nodes, numNodes));
        }
    }

    public void testDoPrefetch_whenSparseFloatVectorValuesImpl_thenPrefetchesViaHasIndexSlice() throws IOException {
        IndexInput mockSlice = mock(IndexInput.class);
        when(mockSlice.length()).thenReturn(200L * 1024);
        int vectorByteLength = 512;

        SparseFloatVectorValuesImpl sparseImpl = mock(SparseFloatVectorValuesImpl.class);
        when(sparseImpl.getSlice()).thenReturn(mockSlice);
        when(sparseImpl.getVectorByteLength()).thenReturn(vectorByteLength);

        try (MockedStatic<PrefetchHelper> mockedPrefetchHelper = mockStatic(PrefetchHelper.class)) {
            PrefetchableVectorValuesHelper.doPrefetch(sparseImpl, nodes, numNodes);

            mockedPrefetchHelper.verify(() -> PrefetchHelper.prefetch(mockSlice, 0, vectorByteLength, nodes, numNodes));
        }
    }

    public void testDoPrefetch_whenOffHeapFloatVectorValues_thenCallsPrefetchHelper() throws IOException {
        IndexInput mockSlice = mock(IndexInput.class);
        when(mockSlice.length()).thenReturn(200L * 1024);
        int vectorByteLength = 16;

        OffHeapFloatVectorValues hasSliceValues = mock(OffHeapFloatVectorValues.class);
        when(hasSliceValues.getSlice()).thenReturn(mockSlice);
        when(hasSliceValues.getVectorByteLength()).thenReturn(vectorByteLength);

        try (MockedStatic<PrefetchHelper> mockedPrefetchHelper = mockStatic(PrefetchHelper.class)) {
            PrefetchableVectorValuesHelper.doPrefetch(hasSliceValues, nodes, numNodes);

            mockedPrefetchHelper.verify(() -> PrefetchHelper.prefetch(mockSlice, 0, vectorByteLength, nodes, numNodes));
        }
    }

    public void testDoPrefetch_whenUnsupportedType_thenNoException() throws IOException {
        KnnVectorValues unsupported = mock(FloatVectorValues.class);

        try (MockedStatic<PrefetchHelper> mockedPrefetchHelper = mockStatic(PrefetchHelper.class)) {
            PrefetchableVectorValuesHelper.doPrefetch(unsupported, nodes, numNodes);

            mockedPrefetchHelper.verifyNoInteractions();
        }
    }

    /**
     * The case unit 16b exists for: a wrapper over a quantized and a full-precision file exposes no slice
     * of its own, and must be prefetched through the full-precision values it names - with *their* byte
     * length, not the wrapper's.
     */
    public void testDoPrefetch_whenWrapperNamesFullPrecisionValues_thenPrefetchesInnerSlice() throws IOException {
        IndexInput mockSlice = mock(IndexInput.class);
        when(mockSlice.length()).thenReturn(200L * 1024);
        int vectorByteLength = 3072;

        OffHeapFloatVectorValues fullPrecision = mock(OffHeapFloatVectorValues.class);
        when(fullPrecision.getSlice()).thenReturn(mockSlice);
        when(fullPrecision.getVectorByteLength()).thenReturn(vectorByteLength);

        KnnVectorValues wrapper = twoSliceWrapper(fullPrecision);

        try (MockedStatic<PrefetchHelper> mockedPrefetchHelper = mockStatic(PrefetchHelper.class)) {
            PrefetchableVectorValuesHelper.doPrefetch(wrapper, nodes, numNodes);

            mockedPrefetchHelper.verify(() -> PrefetchHelper.prefetch(mockSlice, 0, vectorByteLength, nodes, numNodes));
        }
    }

    /** Unwrapping is followed more than once if it has to be, since the plugin's wrapper wraps Lucene's. */
    public void testDoPrefetch_whenWrapperIsNested_thenStillReachesTheSlice() throws IOException {
        IndexInput mockSlice = mock(IndexInput.class);
        when(mockSlice.length()).thenReturn(200L * 1024);
        int vectorByteLength = 3072;

        OffHeapFloatVectorValues fullPrecision = mock(OffHeapFloatVectorValues.class);
        when(fullPrecision.getSlice()).thenReturn(mockSlice);
        when(fullPrecision.getVectorByteLength()).thenReturn(vectorByteLength);

        KnnVectorValues nested = twoSliceWrapper(twoSliceWrapper(fullPrecision));

        try (MockedStatic<PrefetchHelper> mockedPrefetchHelper = mockStatic(PrefetchHelper.class)) {
            PrefetchableVectorValuesHelper.doPrefetch(nested, nodes, numNodes);

            mockedPrefetchHelper.verify(() -> PrefetchHelper.prefetch(mockSlice, 0, vectorByteLength, nodes, numNodes));
        }
    }

    /** The decline path has to survive a wrapper that cannot reach its full-precision values. */
    public void testDoPrefetch_whenWrapperNamesNothing_thenDeclinesWithoutException() throws IOException {
        KnnVectorValues wrapper = twoSliceWrapper(null);

        try (MockedStatic<PrefetchHelper> mockedPrefetchHelper = mockStatic(PrefetchHelper.class)) {
            PrefetchableVectorValuesHelper.doPrefetch(wrapper, nodes, numNodes);

            mockedPrefetchHelper.verifyNoInteractions();
        }
    }

    /** A self-referential or cyclic wrapper must terminate rather than spin. */
    public void testDoPrefetch_whenWrapperNamesItself_thenDeclinesWithoutSpinning() throws IOException {
        final KnnVectorValues[] holder = new KnnVectorValues[1];
        holder[0] = new TwoSliceWrapper(() -> holder[0]);

        try (MockedStatic<PrefetchHelper> mockedPrefetchHelper = mockStatic(PrefetchHelper.class)) {
            PrefetchableVectorValuesHelper.doPrefetch(holder[0], nodes, numNodes);

            mockedPrefetchHelper.verifyNoInteractions();
        }
    }

    /** A wrapper cycle longer than one hop is bounded too. */
    public void testDoPrefetch_whenWrappersCycle_thenDeclinesWithoutSpinning() throws IOException {
        final KnnVectorValues[] holder = new KnnVectorValues[2];
        holder[0] = new TwoSliceWrapper(() -> holder[1]);
        holder[1] = new TwoSliceWrapper(() -> holder[0]);

        try (MockedStatic<PrefetchHelper> mockedPrefetchHelper = mockStatic(PrefetchHelper.class)) {
            PrefetchableVectorValuesHelper.doPrefetch(holder[0], nodes, numNodes);

            mockedPrefetchHelper.verifyNoInteractions();
        }
    }

    public void testDoPrefetch_whenZeroNumNodes_thenDelegatesAsIs() throws IOException {
        IndexInput mockSlice = mock(IndexInput.class);
        when(mockSlice.length()).thenReturn(200L * 1024);
        int vectorByteLength = 64;

        FaissByteVectorValues binaryImpl = mock(FaissByteVectorValues.class);
        when(binaryImpl.getSlice()).thenReturn(mockSlice);
        when(binaryImpl.getVectorByteLength()).thenReturn(vectorByteLength);

        try (MockedStatic<PrefetchHelper> mockedPrefetchHelper = mockStatic(PrefetchHelper.class)) {
            PrefetchableVectorValuesHelper.doPrefetch(binaryImpl, nodes, 0);

            mockedPrefetchHelper.verify(() -> PrefetchHelper.prefetch(mockSlice, 0, vectorByteLength, nodes, 0));
        }
    }

    private KnnVectorValues twoSliceWrapper(final KnnVectorValues fullPrecision) {
        return new TwoSliceWrapper(() -> fullPrecision);
    }

    /**
     * Stands in for a wrapper over two on-disk representations: it exposes no slice of its own - so it
     * cannot implement {@code HasIndexSlice} - and names its full-precision values instead. A supplier is
     * used rather than a field so that a test can build a cycle.
     */
    private static class TwoSliceWrapper extends FloatVectorValues implements HasFullPrecisionVectorValues {
        private final Supplier<KnnVectorValues> fullPrecision;

        TwoSliceWrapper(final Supplier<KnnVectorValues> fullPrecision) {
            this.fullPrecision = fullPrecision;
        }

        @Override
        public KnnVectorValues getFullPrecisionVectorValues() {
            return fullPrecision.get();
        }

        @Override
        public int dimension() {
            return 768;
        }

        @Override
        public int size() {
            return 1;
        }

        @Override
        public float[] vectorValue(int ord) {
            return new float[dimension()];
        }

        @Override
        public FloatVectorValues copy() {
            return this;
        }
    }
}

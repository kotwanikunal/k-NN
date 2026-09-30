/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.lucene95.HasIndexSlice;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.search.VectorScorer;
import org.apache.lucene.util.quantization.QuantizedByteVectorValues;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.codec.scorer.HasFullPrecisionVectorValues;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

public class ScalarQuantizedFloatVectorValuesTests extends KNNTestCase {

    @SneakyThrows
    public void testDimension_thenDelegatesToFloatVectorValues() {
        FloatVectorValues fvv = mock(FloatVectorValues.class);
        when(fvv.dimension()).thenReturn(128);
        var wrapper = new ScalarQuantizedFloatVectorValues(fvv, mock(QuantizedByteVectorValues.class));
        assertEquals(128, wrapper.dimension());
        verify(fvv).dimension();
    }

    @SneakyThrows
    public void testSize_thenDelegatesToFloatVectorValues() {
        FloatVectorValues fvv = mock(FloatVectorValues.class);
        when(fvv.size()).thenReturn(400);
        var wrapper = new ScalarQuantizedFloatVectorValues(fvv, mock(QuantizedByteVectorValues.class));
        assertEquals(400, wrapper.size());
        verify(fvv).size();
    }

    @SneakyThrows
    public void testVectorValue_thenDelegatesToFloatVectorValues() {
        FloatVectorValues fvv = mock(FloatVectorValues.class);
        float[] expected = { 1.0f, 2.0f, 3.0f };
        when(fvv.vectorValue(5)).thenReturn(expected);
        var wrapper = new ScalarQuantizedFloatVectorValues(fvv, mock(QuantizedByteVectorValues.class));
        assertSame(expected, wrapper.vectorValue(5));
        verify(fvv).vectorValue(5);
    }

    @SneakyThrows
    public void testCopy_thenReturnsNewWrappedInstance() {
        FloatVectorValues fvv = mock(FloatVectorValues.class);
        FloatVectorValues fvvCopy = mock(FloatVectorValues.class);
        QuantizedByteVectorValues qbvv = mock(QuantizedByteVectorValues.class);
        QuantizedByteVectorValues qbvvCopy = mock(QuantizedByteVectorValues.class);
        when(fvv.copy()).thenReturn(fvvCopy);
        when(qbvv.copy()).thenReturn(qbvvCopy);

        var wrapper = new ScalarQuantizedFloatVectorValues(fvv, qbvv);
        FloatVectorValues copied = wrapper.copy();

        assertNotSame(wrapper, copied);
        assertTrue(copied instanceof ScalarQuantizedFloatVectorValues);
        assertSame(fvvCopy, ((ScalarQuantizedFloatVectorValues) copied).getFloatVectorValues());
        assertSame(qbvvCopy, ((ScalarQuantizedFloatVectorValues) copied).getQuantizedVectorValues());
        verify(fvv).copy();
        verify(qbvv).copy();
    }

    @SneakyThrows
    public void testCopy_whenQuantizedValuesAreNull_thenReturnsNewWrappedInstance() {
        FloatVectorValues fvv = mock(FloatVectorValues.class);
        FloatVectorValues fvvCopy = mock(FloatVectorValues.class);
        when(fvv.copy()).thenReturn(fvvCopy);

        var wrapper = new ScalarQuantizedFloatVectorValues(fvv, null);
        FloatVectorValues copied = wrapper.copy();

        assertNotSame(wrapper, copied);
        assertTrue(copied instanceof ScalarQuantizedFloatVectorValues);
        assertNull(((ScalarQuantizedFloatVectorValues) copied).getQuantizedVectorValues());
        verify(fvv).copy();
    }

    @SneakyThrows
    public void testGetEncoding_thenDelegatesToFloatVectorValues() {
        FloatVectorValues fvv = mock(FloatVectorValues.class);
        when(fvv.getEncoding()).thenReturn(VectorEncoding.FLOAT32);
        var wrapper = new ScalarQuantizedFloatVectorValues(fvv, mock(QuantizedByteVectorValues.class));
        assertEquals(VectorEncoding.FLOAT32, wrapper.getEncoding());
        verify(fvv).getEncoding();
    }

    public void testGetFloatVectorValues_thenReturnsConstructorArg() {
        FloatVectorValues fvv = mock(FloatVectorValues.class);
        QuantizedByteVectorValues qbvv = mock(QuantizedByteVectorValues.class);
        var wrapper = new ScalarQuantizedFloatVectorValues(fvv, qbvv);
        assertSame(fvv, wrapper.getFloatVectorValues());
    }

    public void testGetQuantizedVectorValues_thenReturnsConstructorArg() {
        FloatVectorValues fvv = mock(FloatVectorValues.class);
        QuantizedByteVectorValues qbvv = mock(QuantizedByteVectorValues.class);
        var wrapper = new ScalarQuantizedFloatVectorValues(fvv, qbvv);
        assertSame(qbvv, wrapper.getQuantizedVectorValues());
    }

    public void testGetQuantizedVectorValues_whenNull_thenReturnsNull() {
        var wrapper = new ScalarQuantizedFloatVectorValues(mock(FloatVectorValues.class), null);
        assertNull(wrapper.getQuantizedVectorValues());
    }

    /**
     * The full-precision delegate is itself a two-representation wrapper - Lucene's
     * {@code ScalarQuantizedVectorValues} - which keeps the {@code .vec}-backed values in a private
     * {@code rawVectorValues} field. Unwrapping that is what lets a prefetch caller reach {@code .vec}
     * instead of declining, so this asserts the unwrap happened rather than that the getter returns a
     * constructor argument.
     */
    @SneakyThrows
    public void testGetFullPrecisionVectorValues_thenUnwrapsTheRawValues() {
        FloatVectorValues rawValues = mock(FloatVectorValues.class);
        LuceneWrapperStub luceneWrapper = new LuceneWrapperStub();
        java.lang.reflect.Field field = LuceneWrapperStub.class.getDeclaredField("rawVectorValues");
        field.setAccessible(true);
        field.set(luceneWrapper, rawValues);

        var wrapper = new ScalarQuantizedFloatVectorValues(luceneWrapper, mock(QuantizedByteVectorValues.class));

        assertSame(rawValues, wrapper.getFullPrecisionVectorValues());
        assertTrue(wrapper instanceof HasFullPrecisionVectorValues);
    }

    /** No raw values reachable means the prefetch caller declines; it must not mean an exception. */
    public void testGetFullPrecisionVectorValues_whenNotReachable_thenNull() {
        var wrapper = new ScalarQuantizedFloatVectorValues(mock(FloatVectorValues.class), mock(QuantizedByteVectorValues.class));
        assertNull(wrapper.getFullPrecisionVectorValues());
    }

    /** A copy is a fresh wrapper over a fresh delegate, so it has to unwrap again rather than share. */
    @SneakyThrows
    public void testCopy_thenUnwrapsTheCopiedRawValues() {
        FloatVectorValues rawValues = mock(FloatVectorValues.class);
        LuceneWrapperStub luceneWrapperCopy = new LuceneWrapperStub();
        java.lang.reflect.Field field = LuceneWrapperStub.class.getDeclaredField("rawVectorValues");
        field.setAccessible(true);
        field.set(luceneWrapperCopy, rawValues);

        FloatVectorValues fvv = mock(FloatVectorValues.class);
        when(fvv.copy()).thenReturn(luceneWrapperCopy);

        var copied = (ScalarQuantizedFloatVectorValues) new ScalarQuantizedFloatVectorValues(fvv, null).copy();

        assertSame(rawValues, copied.getFullPrecisionVectorValues());
    }

    /**
     * Stands in for {@code Lucene104ScalarQuantizedVectorsReader.ScalarQuantizedVectorValues}, which is not
     * constructible from here: what matters is only the private {@code rawVectorValues} field it holds.
     */
    private static class LuceneWrapperStub extends FloatVectorValues {
        private FloatVectorValues rawVectorValues;

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

    public void testDoesNotImplementHasIndexSlice() {
        var wrapper = new ScalarQuantizedFloatVectorValues(mock(FloatVectorValues.class), mock(QuantizedByteVectorValues.class));
        assertFalse(
            "Wrapper must not implement HasIndexSlice because its vectorValue() reads .vec while the quantized slice"
                + " points at .veq — a mismatch that misleads generic prefetch consumers.",
            wrapper instanceof HasIndexSlice
        );
    }

    @SneakyThrows
    public void testIterator_thenDelegatesToFloatVectorValues() {
        FloatVectorValues fvv = mock(FloatVectorValues.class);
        FloatVectorValues.DocIndexIterator expectedIterator = mock(FloatVectorValues.DocIndexIterator.class);
        when(fvv.iterator()).thenReturn(expectedIterator);

        var wrapper = new ScalarQuantizedFloatVectorValues(fvv, mock(QuantizedByteVectorValues.class));
        assertSame(expectedIterator, wrapper.iterator());
        verify(fvv).iterator();
    }

    @SneakyThrows
    public void testScorer_thenDelegatesToQuantizedValues() {
        QuantizedByteVectorValues qbvv = mock(QuantizedByteVectorValues.class);
        VectorScorer expectedScorer = mock(VectorScorer.class);
        float[] target = { 1.0f, 2.0f };
        when(qbvv.scorer(target)).thenReturn(expectedScorer);

        var wrapper = new ScalarQuantizedFloatVectorValues(mock(FloatVectorValues.class), qbvv);
        assertSame(expectedScorer, wrapper.scorer(target));
        verify(qbvv).scorer(target);
    }

    @SneakyThrows
    public void testScorer_whenQuantizedValuesAreNull_thenReturnsNull() {
        var wrapper = new ScalarQuantizedFloatVectorValues(mock(FloatVectorValues.class), null);
        assertNull(wrapper.scorer(new float[] { 1.0f, 2.0f }));
    }
}

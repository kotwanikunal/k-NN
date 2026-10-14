/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.SneakyThrows;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.util.quantization.QuantizedByteVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.VectorEncoding;
import org.opensearch.knn.KNNTestCase;

import java.io.IOException;

import static org.mockito.Mockito.mock;

@Log4j2
public class KNN1040ScalarQuantizedUtilsTests extends KNNTestCase {

    /**
     * A concrete stub extending KnnVectorValues that declares the private field
     * {@code quantizedVectorValues} so that reflection-based extraction succeeds.
     */
    static class StubVectorValues extends KnnVectorValues {
        private QuantizedByteVectorValues quantizedVectorValues;

        @Override
        public int dimension() {
            return 0;
        }

        @Override
        public int size() {
            return 0;
        }

        @Override
        public KnnVectorValues copy() throws IOException {
            return this;
        }

        @Override
        public VectorEncoding getEncoding() {
            return VectorEncoding.FLOAT32;
        }
    }

    @SneakyThrows
    public void testExtractQuantizedByteVectorValues_whenFieldExists_thenReturnsValue() {
        // Arrange: create stub and set the private field via reflection
        StubVectorValues stub = new StubVectorValues();
        QuantizedByteVectorValues expected = mock(QuantizedByteVectorValues.class);

        java.lang.reflect.Field field = StubVectorValues.class.getDeclaredField("quantizedVectorValues");
        field.setAccessible(true);
        field.set(stub, expected);

        // Act
        QuantizedByteVectorValues result = KNN1040ScalarQuantizedUtils.extractQuantizedByteVectorValues(stub);

        // Assert: the returned reference is the exact same object
        assertSame(expected, result);
    }

    public void testExtractQuantizedByteVectorValues_whenFieldMissing_thenThrowsIOException() {
        // Arrange: a Mockito mock of KnnVectorValues lacks the quantizedVectorValues field
        KnnVectorValues mockValues = mock(KnnVectorValues.class);

        // Act & Assert
        IOException exception = expectThrows(
            IOException.class,
            () -> KNN1040ScalarQuantizedUtils.extractQuantizedByteVectorValues(mockValues)
        );

        assertTrue(exception.getMessage().contains("incompatible Lucene version"));
        assertTrue(exception.getCause() instanceof NoSuchFieldException);
    }

    /**
     * A stub standing in for Lucene's {@code ScalarQuantizedVectorValues}, which holds the full-precision
     * values in a private {@code rawVectorValues} field with no accessor.
     */
    static class StubWrapperWithRawValues extends KnnVectorValues {
        private KnnVectorValues rawVectorValues;

        @Override
        public int dimension() {
            return 0;
        }

        @Override
        public int size() {
            return 0;
        }

        @Override
        public KnnVectorValues copy() throws IOException {
            return this;
        }

        @Override
        public VectorEncoding getEncoding() {
            return VectorEncoding.FLOAT32;
        }
    }

    @SneakyThrows
    public void testExtractRawFloatVectorValues_whenFieldExists_thenReturnsValue() {
        StubWrapperWithRawValues stub = new StubWrapperWithRawValues();
        KnnVectorValues expected = mock(KnnVectorValues.class);

        java.lang.reflect.Field field = StubWrapperWithRawValues.class.getDeclaredField("rawVectorValues");
        field.setAccessible(true);
        field.set(stub, expected);

        assertSame(expected, KNN1040ScalarQuantizedUtils.extractRawFloatVectorValues(stub));
    }

    /**
     * Losing the unwrap must never fail a read - its only caller is issuing an advisory prefetch hint - so
     * this returns null where {@link KNN1040ScalarQuantizedUtils#extractQuantizedByteVectorValues} throws.
     */
    public void testExtractRawFloatVectorValues_whenFieldMissing_thenReturnsNull() {
        assertNull(KNN1040ScalarQuantizedUtils.extractRawFloatVectorValues(mock(KnnVectorValues.class)));
    }

    public void testExtractRawFloatVectorValues_whenFieldIsUnset_thenReturnsNull() {
        assertNull(KNN1040ScalarQuantizedUtils.extractRawFloatVectorValues(new StubWrapperWithRawValues()));
    }

    public void testExtractRawFloatVectorValues_whenArgumentIsNull_thenReturnsNull() {
        assertNull(KNN1040ScalarQuantizedUtils.extractRawFloatVectorValues(null));
    }

}

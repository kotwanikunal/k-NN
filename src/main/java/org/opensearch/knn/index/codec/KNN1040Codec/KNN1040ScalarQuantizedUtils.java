/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.experimental.UtilityClass;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.util.quantization.QuantizedByteVectorValues;
import org.apache.lucene.index.KnnVectorValues;

import java.io.IOException;
import java.lang.reflect.Field;
import java.util.Locale;

/**
 * Utility class for extracting quantized vector values from Lucene's internal reader structures.
 *
 * <p>Lucene's {@code Lucene104ScalarQuantizedVectorsReader} returns a {@code ScalarQuantizedVectorValues}
 * from {@code getFloatVectorValues()}, which wraps both raw float vectors and their quantized counterparts.
 * The quantized values are stored in a private {@code quantizedVectorValues} field and are not exposed
 * through any public API. This utility uses reflection to access that field.
 *
 * <p>This is used by both the write path ({@link Faiss1040ScalarQuantizedKnnVectorsWriter}) to extract
 * quantized vectors for native HNSW graph construction, and the search path
 * ({@link KNN1040ScalarQuantizedVectorScorer}) to obtain quantized vectors for SIMD-accelerated scoring.
 */
@UtilityClass
@Log4j2
class KNN1040ScalarQuantizedUtils {
    private static final String QUANTIZED_VECTOR_VALUES_FIELD_NAME = "quantizedVectorValues";
    private static final String RAW_VECTOR_VALUES_FIELD_NAME = "rawVectorValues";

    /**
     * Extracts {@link QuantizedByteVectorValues} from the given {@link KnnVectorValues} via reflection.
     *
     * <p>The {@code floatVectorValues} parameter is expected to be a {@code ScalarQuantizedVectorValues}
     * instance returned by {@code Lucene104ScalarQuantizedVectorsReader.getFloatVectorValues()}.
     * This wrapper holds a private {@code quantizedVectorValues} field containing the 1-bit binary
     * quantized codes and their correction factors (lower/upper intervals, additional correction,
     * and quantized component sum).
     *
     * @param floatVectorValues the vector values instance to extract quantized values from;
     *                          typically a {@code ScalarQuantizedVectorValues}
     * @return the extracted {@link QuantizedByteVectorValues}, or {@code null} if not found
     * and {@code throwExceptionIfNotFound} is {@code false}
     * @throws IOException if extraction fails and {@code throwExceptionIfNotFound} is {@code true}
     */

    public static QuantizedByteVectorValues extractQuantizedByteVectorValues(final KnnVectorValues floatVectorValues) throws IOException {
        try {
            final Field f = floatVectorValues.getClass().getDeclaredField(QUANTIZED_VECTOR_VALUES_FIELD_NAME);
            f.setAccessible(true);
            return (QuantizedByteVectorValues) f.get(floatVectorValues);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            throw new IOException(
                String.format(
                    Locale.ROOT,
                    "Failed to extract QuantizedByteVectorValues from floatVectorValues [%s/%s]."
                        + " This may indicate an incompatible Lucene version.",
                    floatVectorValues.getClass().getSimpleName(),
                    floatVectorValues
                ),
                e
            );
        }
    }

    /**
     * Extracts the raw, full-precision {@link KnnVectorValues} from the given {@link KnnVectorValues} via
     * reflection.
     *
     * <p>The {@code floatVectorValues} parameter is expected to be a {@code ScalarQuantizedVectorValues}
     * instance returned by {@code Lucene104ScalarQuantizedVectorsReader.getFloatVectorValues()}. That
     * wrapper holds the full-precision values - the ones backed by the {@code .vec} file, which is what the
     * rescore path reads - in a private {@code rawVectorValues} field with no accessor, exactly as it holds
     * the quantized values in {@code quantizedVectorValues}.
     *
     * <p>Unlike {@link #extractQuantizedByteVectorValues}, this returns {@code null} rather than throwing
     * when the field is absent or inaccessible. Its only caller wants the values in order to issue an
     * advisory prefetch hint, and losing a hint must never fail a read; a different Lucene layout, or a
     * segment whose float values are not that wrapper at all, simply means no hint.
     *
     * @param floatVectorValues the vector values instance to unwrap
     * @return the raw full-precision values, or {@code null} if they cannot be reached
     */
    public static KnnVectorValues extractRawFloatVectorValues(final KnnVectorValues floatVectorValues) {
        if (floatVectorValues == null) {
            return null;
        }
        try {
            final Field f = floatVectorValues.getClass().getDeclaredField(RAW_VECTOR_VALUES_FIELD_NAME);
            f.setAccessible(true);
            final Object raw = f.get(floatVectorValues);
            return raw instanceof KnnVectorValues rawValues ? rawValues : null;
        } catch (NoSuchFieldException | IllegalAccessException | RuntimeException e) {
            log.debug("Could not reach the raw full-precision values of [{}]", floatVectorValues.getClass().getSimpleName(), e);
            return null;
        }
    }
}

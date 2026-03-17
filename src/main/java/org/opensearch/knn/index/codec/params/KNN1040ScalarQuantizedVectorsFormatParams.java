/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.params;

import lombok.Getter;
import org.opensearch.knn.index.engine.MethodComponentContext;

import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.ENCODER_BBQ;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;

/**
 * Params class for the Lucene BBQ HNSW format. Used by {@code LuceneCodecFormatResolver} to
 * detect whether a field should be routed to the BBQ format type.
 *
 * <p>This is a minimal stub — the full implementation with bit encoding resolution and
 * validation is tracked in https://github.com/opensearch-project/k-NN/pull/3144
 */
@Getter
public class KNN1040ScalarQuantizedVectorsFormatParams extends KNNVectorsFormatParams {

    private final String encoderName;

    public KNN1040ScalarQuantizedVectorsFormatParams(Map<String, Object> params, int defaultMaxConnections, int defaultBeamWidth) {
        super(params, defaultMaxConnections, defaultBeamWidth);
        this.encoderName = resolveEncoderName(params);
    }

    /**
     * Returns true if the encoder is BBQ. Used by the format resolver to decide between
     * the BBQ format type and the standard SQ/HNSW format types.
     */
    public boolean validate(Map<String, Object> params) {
        return ENCODER_BBQ.equals(encoderName);
    }

    /**
     * Safely extracts the encoder name from the method parameters, handling the case where
     * the encoder parameter might be missing or not a MethodComponentContext (which can
     * happen during BwC when old mappings are re-parsed).
     */
    private static String resolveEncoderName(Map<String, Object> params) {
        if (params == null || params.containsKey(METHOD_ENCODER_PARAMETER) == false) {
            return null;
        }
        Object encoder = params.get(METHOD_ENCODER_PARAMETER);
        if (encoder instanceof MethodComponentContext ctx) {
            return ctx.getName();
        }
        return null;
    }
}

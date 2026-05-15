/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn;

import lombok.AllArgsConstructor;
import lombok.Getter;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.Mode;
import org.opensearch.knn.quantization.enums.ScalarQuantizationType;

/**
 * Test configurations for dual-path coverage of FP32 (no compression) and SQ 1-bit (32x compression).
 */
@Getter
@AllArgsConstructor
public enum CompressionTestConfig {
    FP32(CompressionLevel.x1, Mode.IN_MEMORY, null, 0.95f),
    SQ_1BIT(CompressionLevel.x32, Mode.ON_DISK, ScalarQuantizationType.ONE_BIT, 0.70f);

    private final CompressionLevel compressionLevel;
    private final Mode mode;
    private final ScalarQuantizationType expectedQuantizationType;
    private final float minimumRecallThreshold;

    public String getCompressionLevelName() {
        return compressionLevel.getName();
    }

    public String getModeName() {
        return mode.getName();
    }

    public boolean isCompressed() {
        return this != FP32;
    }
}

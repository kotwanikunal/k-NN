/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.faiss;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import org.opensearch.Version;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.Encoder;
import org.opensearch.knn.index.engine.KNNLibraryIndexingContextImpl;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.MethodComponent;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.engine.Parameter;
import org.opensearch.knn.index.mapper.CompressionLevel;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Set;
import java.util.stream.Collectors;

import static org.opensearch.knn.common.KNNConstants.ENCODER_FAISS_BBQ;
import static org.opensearch.knn.common.KNNConstants.FAISS_FLAT_DESCRIPTION;
import static org.opensearch.knn.common.KNNConstants.INDEX_DESCRIPTION_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.NAME;

/**
 * Encoder for Faiss BBQ (Better Binary Quantization).
 *
 * <p>This encoder is the default for 32x compression on Faiss starting with OpenSearch 3.6.0,
 * replacing the older {@link QFrameBitEncoder} for new indices. It uses a dedicated per-field
 * codec format ({@link org.opensearch.knn.index.codec.KNN1040Codec.Faiss104ScalarQuantizedKnnVectorsFormat})
 * where Lucene handles 1-bit quantized flat vector storage while the HNSW graph is built
 * natively by Faiss.
 *
 * <p>Only 1-bit quantization is supported currently (see {@link Bits#ONE}). The encoder is
 * version-gated to 3.6.0+ to prevent accidental use on older indices during rolling upgrades.
 */
public class FaissBBQEncoder implements Encoder {

    public static final String BITCOUNT_PARAM = "bits";
    public static final int DEFAULT_BITS = Bits.ONE.getValue();

    private static final Set<VectorDataType> SUPPORTED_DATA_TYPES = Set.of(VectorDataType.FLOAT);
    private static final Set<Integer> VALID_BIT_COUNTS = Arrays.stream(Bits.values())
        .map(Bits::getValue)
        .collect(Collectors.toUnmodifiableSet());

    /**
     * Supported bit counts for BBQ quantization. Each value maps to a specific scalar encoding
     * in Lucene's quantized vectors format. Only 1-bit is supported for now — additional bit
     * widths can be added here as Lucene gains support for them.
     */
    @Getter
    @RequiredArgsConstructor
    public enum Bits {
        ONE(1);

        private final int value;
    }

    private static final MethodComponent METHOD_COMPONENT = MethodComponent.Builder.builder(ENCODER_FAISS_BBQ)
        .addSupportedDataTypes(SUPPORTED_DATA_TYPES)
        .addParameter(BITCOUNT_PARAM, new Parameter.IntegerParameter(BITCOUNT_PARAM, DEFAULT_BITS, (v, context) -> {
            // Reject on pre-3.6.0 indices — this encoder didn't exist before then, and allowing
            // it would produce segments that older nodes can't read.
            if (context != null && context.getVersionCreated() != null && context.getVersionCreated().before(Version.V_3_6_0)) {
                return false;
            }
            return VALID_BIT_COUNTS.contains(v);
        }))
        .setKnnLibraryIndexingContextGenerator(((methodComponent, methodComponentContext, knnMethodConfigContext) -> {
            // We include the encoder name and bits in the parameters map so they get serialized
            // into the segment's PARAMETERS field attribute. This makes segments self-describing —
            // any code inspecting the segment can identify the encoder without needing the mapping.
            int bits = (int) methodComponentContext.getParameters().getOrDefault(BITCOUNT_PARAM, DEFAULT_BITS);
            return KNNLibraryIndexingContextImpl.builder().parameters(new HashMap<>() {
                {
                    put(INDEX_DESCRIPTION_PARAMETER, FAISS_FLAT_DESCRIPTION);
                    put(NAME, ENCODER_FAISS_BBQ);
                    put(BITCOUNT_PARAM, bits);
                }
            }).build();
        }))
        .setRequiresTraining(false)
        .build();

    @Override
    public MethodComponent getMethodComponent() {
        return METHOD_COMPONENT;
    }

    @Override
    public CompressionLevel calculateCompressionLevel(
        MethodComponentContext encoderContext,
        KNNMethodConfigContext knnMethodConfigContext
    ) {
        return CompressionLevel.x32;
    }
}

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.lucene;

import com.google.common.collect.ImmutableSet;
import org.opensearch.Version;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.Encoder;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.MethodComponent;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.engine.Parameter;
import org.opensearch.knn.index.mapper.CompressionLevel;

import java.util.Set;

import static org.opensearch.knn.common.KNNConstants.ENCODER_BBQ;
import static org.opensearch.knn.common.KNNConstants.LUCENE_BBQ_BITS;
import static org.opensearch.knn.common.KNNConstants.LUCENE_BBQ_DEFAULT_BITS;

/**
 * Lucene-side BBQ encoder definition. Uses {@code "bbq"} as the encoder name to distinguish
 * it from the Faiss-side {@code "faiss_bbq"} encoder — each engine has its own encoder name
 * even though both represent the same 1-bit quantization concept.
 *
 * <p>This is a stub that provides the encoder registration, parameter validation, and
 * compression level calculation needed by the mapping and method resolution layers.
 * The actual codec format routing and write path are being implemented in a separate PR.
 *
 * <p>Version-gated to 3.6.0+ — same as the Faiss BBQ encoder.
 *
 * TODO: Integrate with full Lucene BBQ implementation from
 *  https://github.com/opensearch-project/k-NN/pull/3144
 * TODO: Wire up KNN1040ScalarQuantizedVectorsFormatParams for codec format routing
 * TODO: Add KNNLibraryIndexingContext generation once Lucene BBQ write path is implemented
 */
public class LuceneBBQEncoder implements Encoder {

    private static final Set<Integer> VALID_BIT_COUNTS = ImmutableSet.of(1);
    private static final Set<VectorDataType> SUPPORTED_DATA_TYPES = ImmutableSet.of(VectorDataType.FLOAT);

    private static final MethodComponent METHOD_COMPONENT = MethodComponent.Builder.builder(ENCODER_BBQ)
        .addSupportedDataTypes(SUPPORTED_DATA_TYPES)
        .addParameter(LUCENE_BBQ_BITS, new Parameter.IntegerParameter(LUCENE_BBQ_BITS, LUCENE_BBQ_DEFAULT_BITS, (v, context) -> {
            if (context != null && context.getVersionCreated() != null && context.getVersionCreated().before(Version.V_3_6_0)) {
                return false;
            }
            return VALID_BIT_COUNTS.contains(v);
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

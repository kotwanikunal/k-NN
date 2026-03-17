/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.faiss;

import com.google.common.collect.ImmutableMap;
import org.opensearch.Version;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNLibraryIndexingContext;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.MethodComponent;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.engine.qframe.QuantizationConfig;
import org.opensearch.knn.index.mapper.CompressionLevel;

import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.ENCODER_FAISS_BBQ;
import static org.opensearch.knn.common.KNNConstants.FAISS_FLAT_DESCRIPTION;
import static org.opensearch.knn.common.KNNConstants.INDEX_DESCRIPTION_PARAMETER;

public class FaissBBQEncoderTests extends KNNTestCase {

    public void testGetLibraryIndexingContext() {
        FaissBBQEncoder encoder = new FaissBBQEncoder();
        MethodComponent methodComponent = encoder.getMethodComponent();
        KNNMethodConfigContext context = KNNMethodConfigContext.builder()
            .versionCreated(Version.CURRENT)
            .vectorDataType(VectorDataType.FLOAT)
            .dimension(128)
            .build();

        MethodComponentContext methodComponentContext = new MethodComponentContext(ENCODER_FAISS_BBQ, ImmutableMap.of("bits", 1));
        KNNLibraryIndexingContext indexingContext = methodComponent.getKNNLibraryIndexingContext(methodComponentContext, context);

        Map<String, Object> params = indexingContext.getLibraryParameters();
        assertEquals(FAISS_FLAT_DESCRIPTION, params.get(INDEX_DESCRIPTION_PARAMETER));
        assertEquals(ENCODER_FAISS_BBQ, params.get("name"));
        assertEquals(1, params.get("bits"));
        // BBQ should NOT produce a QuantizationConfig — quantization is handled by Lucene's flat format
        assertEquals(QuantizationConfig.EMPTY, indexingContext.getQuantizationConfig());
    }

    public void testCalculateCompressionLevel() {
        FaissBBQEncoder encoder = new FaissBBQEncoder();
        assertEquals(CompressionLevel.x32, encoder.calculateCompressionLevel(null, null));
    }

    public void testValidate_whenValidConfig_thenNoError() {
        FaissBBQEncoder encoder = new FaissBBQEncoder();
        MethodComponent methodComponent = encoder.getMethodComponent();
        KNNMethodConfigContext context = KNNMethodConfigContext.builder()
            .versionCreated(Version.CURRENT)
            .vectorDataType(VectorDataType.FLOAT)
            .dimension(128)
            .build();

        MethodComponentContext methodComponentContext = new MethodComponentContext(ENCODER_FAISS_BBQ, Map.of("bits", 1));
        assertNull(methodComponent.validate(methodComponentContext, context));
    }

    public void testValidate_whenInvalidBits_thenError() {
        FaissBBQEncoder encoder = new FaissBBQEncoder();
        MethodComponent methodComponent = encoder.getMethodComponent();
        KNNMethodConfigContext context = KNNMethodConfigContext.builder()
            .versionCreated(Version.CURRENT)
            .vectorDataType(VectorDataType.FLOAT)
            .dimension(128)
            .build();

        // bits=2 is not valid for BBQ
        MethodComponentContext methodComponentContext = new MethodComponentContext(ENCODER_FAISS_BBQ, Map.of("bits", 2));
        assertNotNull(methodComponent.validate(methodComponentContext, context));
    }

    public void testValidate_whenPreV360_thenError() {
        FaissBBQEncoder encoder = new FaissBBQEncoder();
        MethodComponent methodComponent = encoder.getMethodComponent();
        KNNMethodConfigContext context = KNNMethodConfigContext.builder()
            .versionCreated(Version.V_3_5_0)
            .vectorDataType(VectorDataType.FLOAT)
            .dimension(128)
            .build();

        MethodComponentContext methodComponentContext = new MethodComponentContext(ENCODER_FAISS_BBQ, Map.of("bits", 1));
        assertNotNull(methodComponent.validate(methodComponentContext, context));
    }

    public void testValidate_whenByteDataType_thenError() {
        FaissBBQEncoder encoder = new FaissBBQEncoder();
        MethodComponent methodComponent = encoder.getMethodComponent();
        KNNMethodConfigContext context = KNNMethodConfigContext.builder()
            .versionCreated(Version.CURRENT)
            .vectorDataType(VectorDataType.BYTE)
            .dimension(128)
            .build();

        MethodComponentContext methodComponentContext = new MethodComponentContext(ENCODER_FAISS_BBQ, Map.of("bits", 1));
        assertNotNull(methodComponent.validate(methodComponentContext, context));
    }

    public void testIsTrainingRequired() {
        FaissBBQEncoder encoder = new FaissBBQEncoder();
        assertFalse(encoder.getMethodComponent().isTrainingRequired(new MethodComponentContext(ENCODER_FAISS_BBQ, Map.of("bits", 1))));
    }

    public void testGetName() {
        FaissBBQEncoder encoder = new FaissBBQEncoder();
        assertEquals(ENCODER_FAISS_BBQ, encoder.getName());
    }
}

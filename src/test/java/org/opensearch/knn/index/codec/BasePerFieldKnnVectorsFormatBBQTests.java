/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.KnnVectorsFormat;
import org.opensearch.index.IndexSettings;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.codec.KNN990Codec.FaissBBQ990KnnVectorsFormat;
import org.opensearch.knn.index.codec.KNN990Codec.NativeEngines990KnnVectorsFormat;
import org.opensearch.knn.index.codec.KNN9120Codec.KNN9120PerFieldKnnVectorsFormat;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.mapper.KNNMappingConfig;
import org.opensearch.knn.index.mapper.KNNVectorFieldType;

import java.util.Map;
import java.util.Optional;

import static org.opensearch.knn.common.KNNConstants.ENCODER_FAISS_BBQ;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Tests for BasePerFieldKnnVectorsFormat's Faiss BBQ routing logic.
 */
public class BasePerFieldKnnVectorsFormatBBQTests extends KNNTestCase {

    /**
     * When a field has faiss engine with faiss_bbq encoder, getKnnVectorsFormatForField
     * should return FaissBBQ990KnnVectorsFormat.
     */
    @SneakyThrows
    public void testGetKnnVectorsFormatForField_whenFaissBBQEncoder_thenReturnsFaissBBQFormat() {
        final String fieldName = "bbq_field";

        // Build the encoder parameter: { "encoder": { "name": "faiss_bbq", "parameters": { "bits": 1 } } }
        final MethodComponentContext encoderContext = new MethodComponentContext(ENCODER_FAISS_BBQ, Map.of("bits", 1));
        final Map<String, Object> params = Map.of(METHOD_ENCODER_PARAMETER, encoderContext);
        final MethodComponentContext methodComponentContext = new MethodComponentContext("hnsw", params);

        // Mock KNNMethodContext
        final KNNMethodContext knnMethodContext = mock(KNNMethodContext.class);
        when(knnMethodContext.getKnnEngine()).thenReturn(KNNEngine.FAISS);
        when(knnMethodContext.getMethodComponentContext()).thenReturn(methodComponentContext);

        // Mock KNNMappingConfig
        final KNNMappingConfig knnMappingConfig = mock(KNNMappingConfig.class);
        when(knnMappingConfig.getModelId()).thenReturn(Optional.empty());
        when(knnMappingConfig.getKnnMethodContext()).thenReturn(Optional.of(knnMethodContext));
        when(knnMappingConfig.getKnnLibraryIndexingContext()).thenReturn(null);

        // Mock KNNVectorFieldType
        final KNNVectorFieldType fieldType = mock(KNNVectorFieldType.class);
        when(fieldType.getKnnMappingConfig()).thenReturn(knnMappingConfig);

        // Mock MapperService
        final MapperService mapperService = mock(MapperService.class);
        when(mapperService.fieldType(fieldName)).thenReturn(fieldType);

        // Mock IndexSettings for approximate threshold
        final IndexSettings indexSettings = mock(IndexSettings.class);
        when(indexSettings.getValue(KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD_SETTING))
            .thenReturn(KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD_DEFAULT_VALUE);
        when(mapperService.getIndexSettings()).thenReturn(indexSettings);

        final KNN9120PerFieldKnnVectorsFormat perFieldFormat = new KNN9120PerFieldKnnVectorsFormat(Optional.of(mapperService));
        final KnnVectorsFormat format = perFieldFormat.getKnnVectorsFormatForField(fieldName);

        assertTrue(
            "Expected FaissBBQ990KnnVectorsFormat but got " + format.getClass().getSimpleName(),
            format instanceof FaissBBQ990KnnVectorsFormat
        );
    }

    /**
     * When a field has faiss engine WITHOUT faiss_bbq encoder, getKnnVectorsFormatForField
     * should return NativeEngines990KnnVectorsFormat (not BBQ).
     */
    @SneakyThrows
    public void testGetKnnVectorsFormatForField_whenFaissWithoutBBQ_thenReturnsNativeFormat() {
        final String fieldName = "regular_faiss_field";

        // Regular HNSW params without BBQ encoder
        final Map<String, Object> params = Map.of();
        final MethodComponentContext methodComponentContext = new MethodComponentContext("hnsw", params);

        final KNNMethodContext knnMethodContext = mock(KNNMethodContext.class);
        when(knnMethodContext.getKnnEngine()).thenReturn(KNNEngine.FAISS);
        when(knnMethodContext.getMethodComponentContext()).thenReturn(methodComponentContext);

        final KNNMappingConfig knnMappingConfig = mock(KNNMappingConfig.class);
        when(knnMappingConfig.getModelId()).thenReturn(Optional.empty());
        when(knnMappingConfig.getKnnMethodContext()).thenReturn(Optional.of(knnMethodContext));
        when(knnMappingConfig.getKnnLibraryIndexingContext()).thenReturn(null);

        final KNNVectorFieldType fieldType = mock(KNNVectorFieldType.class);
        when(fieldType.getKnnMappingConfig()).thenReturn(knnMappingConfig);

        final MapperService mapperService = mock(MapperService.class);
        when(mapperService.fieldType(fieldName)).thenReturn(fieldType);

        final IndexSettings indexSettings = mock(IndexSettings.class);
        when(indexSettings.getValue(KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD_SETTING))
            .thenReturn(KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD_DEFAULT_VALUE);
        when(mapperService.getIndexSettings()).thenReturn(indexSettings);

        final KNN9120PerFieldKnnVectorsFormat perFieldFormat = new KNN9120PerFieldKnnVectorsFormat(Optional.of(mapperService));
        final KnnVectorsFormat format = perFieldFormat.getKnnVectorsFormatForField(fieldName);

        assertTrue(
            "Expected NativeEngines990KnnVectorsFormat but got " + format.getClass().getSimpleName(),
            format instanceof NativeEngines990KnnVectorsFormat
        );
    }

    /**
     * When a field has a non-BBQ encoder (e.g. some other encoder name), it should NOT return BBQ format.
     */
    @SneakyThrows
    public void testGetKnnVectorsFormatForField_whenFaissWithNonBBQEncoder_thenReturnsNativeFormat() {
        final String fieldName = "non_bbq_field";

        final MethodComponentContext encoderContext = new MethodComponentContext("some_other_encoder", Map.of());
        final Map<String, Object> params = Map.of(METHOD_ENCODER_PARAMETER, encoderContext);
        final MethodComponentContext methodComponentContext = new MethodComponentContext("hnsw", params);

        final KNNMethodContext knnMethodContext = mock(KNNMethodContext.class);
        when(knnMethodContext.getKnnEngine()).thenReturn(KNNEngine.FAISS);
        when(knnMethodContext.getMethodComponentContext()).thenReturn(methodComponentContext);

        final KNNMappingConfig knnMappingConfig = mock(KNNMappingConfig.class);
        when(knnMappingConfig.getModelId()).thenReturn(Optional.empty());
        when(knnMappingConfig.getKnnMethodContext()).thenReturn(Optional.of(knnMethodContext));
        when(knnMappingConfig.getKnnLibraryIndexingContext()).thenReturn(null);

        final KNNVectorFieldType fieldType = mock(KNNVectorFieldType.class);
        when(fieldType.getKnnMappingConfig()).thenReturn(knnMappingConfig);

        final MapperService mapperService = mock(MapperService.class);
        when(mapperService.fieldType(fieldName)).thenReturn(fieldType);

        final IndexSettings indexSettings = mock(IndexSettings.class);
        when(indexSettings.getValue(KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD_SETTING))
            .thenReturn(KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD_DEFAULT_VALUE);
        when(mapperService.getIndexSettings()).thenReturn(indexSettings);

        final KNN9120PerFieldKnnVectorsFormat perFieldFormat = new KNN9120PerFieldKnnVectorsFormat(Optional.of(mapperService));
        final KnnVectorsFormat format = perFieldFormat.getKnnVectorsFormatForField(fieldName);

        assertTrue(
            "Expected NativeEngines990KnnVectorsFormat but got " + format.getClass().getSimpleName(),
            format instanceof NativeEngines990KnnVectorsFormat
        );
    }
}

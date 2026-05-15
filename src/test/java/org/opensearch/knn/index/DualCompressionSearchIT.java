/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import lombok.SneakyThrows;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.client.Response;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.index.query.QueryBuilders;
import org.opensearch.knn.CompressionTestConfig;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.KNNResult;
import org.opensearch.knn.TestUtils;
import org.opensearch.knn.index.query.KNNQueryBuilder;

import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import static org.opensearch.knn.common.KNNConstants.COMPRESSION_LEVEL_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.FAISS_NAME;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.KNN_METHOD;
import static org.opensearch.knn.common.KNNConstants.LUCENE_NAME;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_SEARCH;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_M;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.MODE_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.index.KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD;
import static org.opensearch.knn.index.KNNSettings.KNN_INDEX;

/**
 * Parameterized search/accuracy tests that validate k-NN search with both FP32 and SQ 1-bit (32x) compression
 * for Faiss and Lucene engines. Uses the sub-case pattern: each test method calls a helper with each config.
 */
public class DualCompressionSearchIT extends KNNRestTestCase {

    private static final int DIMENSION = 128;
    private static final int DOC_COUNT = 100;
    private static final int QUERY_COUNT = 10;
    private static final int K = 10;
    private static final int HNSW_M = 16;
    private static final int HNSW_EF_CONSTRUCTION = 100;
    private static final int HNSW_EF_SEARCH = 100;

    private static final float[][] INDEX_VECTORS = TestUtils.getIndexVectors(DOC_COUNT, DIMENSION, true);
    private static final float[][] QUERY_VECTORS = TestUtils.getQueryVectors(QUERY_COUNT, DIMENSION, DOC_COUNT, true);
    private static final List<Set<String>> GROUND_TRUTH_L2 = TestUtils.computeGroundTruthValues(
        INDEX_VECTORS,
        QUERY_VECTORS,
        SpaceType.L2,
        K
    );
    private static final List<Set<String>> GROUND_TRUTH_COSINE = TestUtils.computeGroundTruthValues(
        INDEX_VECTORS,
        QUERY_VECTORS,
        SpaceType.COSINESIMIL,
        K
    );
    private static final List<Set<String>> GROUND_TRUTH_IP = TestUtils.computeGroundTruthValues(
        INDEX_VECTORS,
        QUERY_VECTORS,
        SpaceType.INNER_PRODUCT,
        K
    );

    @SneakyThrows
    public void testDualCompression_faiss_kSearch_l2() {
        for (CompressionTestConfig config : CompressionTestConfig.values()) {
            runKSearchTest("faiss_dual_l2_" + config.name().toLowerCase(Locale.ROOT), FAISS_NAME, SpaceType.L2, GROUND_TRUTH_L2, config);
        }
    }

    @SneakyThrows
    public void testDualCompression_lucene_kSearch_l2() {
        for (CompressionTestConfig config : CompressionTestConfig.values()) {
            runKSearchTest("lucene_dual_l2_" + config.name().toLowerCase(Locale.ROOT), LUCENE_NAME, SpaceType.L2, GROUND_TRUTH_L2, config);
        }
    }

    @SneakyThrows
    public void testDualCompression_faiss_kSearch_cosine() {
        for (CompressionTestConfig config : CompressionTestConfig.values()) {
            runKSearchTest(
                "faiss_dual_cosine_" + config.name().toLowerCase(Locale.ROOT),
                FAISS_NAME,
                SpaceType.COSINESIMIL,
                GROUND_TRUTH_COSINE,
                config
            );
        }
    }

    @SneakyThrows
    public void testDualCompression_lucene_kSearch_cosine() {
        for (CompressionTestConfig config : CompressionTestConfig.values()) {
            runKSearchTest(
                "lucene_dual_cosine_" + config.name().toLowerCase(Locale.ROOT),
                LUCENE_NAME,
                SpaceType.COSINESIMIL,
                GROUND_TRUTH_COSINE,
                config
            );
        }
    }

    @SneakyThrows
    public void testDualCompression_faiss_kSearch_innerProduct() {
        for (CompressionTestConfig config : CompressionTestConfig.values()) {
            runKSearchTest(
                "faiss_dual_ip_" + config.name().toLowerCase(Locale.ROOT),
                FAISS_NAME,
                SpaceType.INNER_PRODUCT,
                GROUND_TRUTH_IP,
                config
            );
        }
    }

    @SneakyThrows
    public void testDualCompression_lucene_kSearch_innerProduct() {
        for (CompressionTestConfig config : CompressionTestConfig.values()) {
            runKSearchTest(
                "lucene_dual_ip_" + config.name().toLowerCase(Locale.ROOT),
                LUCENE_NAME,
                SpaceType.INNER_PRODUCT,
                GROUND_TRUTH_IP,
                config
            );
        }
    }

    @SneakyThrows
    public void testDualCompression_faiss_filteredSearch() {
        for (CompressionTestConfig config : CompressionTestConfig.values()) {
            runFilteredSearchTest("faiss_dual_filter_" + config.name().toLowerCase(Locale.ROOT), FAISS_NAME, config);
        }
    }

    @SneakyThrows
    public void testDualCompression_lucene_filteredSearch() {
        for (CompressionTestConfig config : CompressionTestConfig.values()) {
            runFilteredSearchTest("lucene_dual_filter_" + config.name().toLowerCase(Locale.ROOT), LUCENE_NAME, config);
        }
    }

    @SneakyThrows
    private void runKSearchTest(
        String indexName,
        String engine,
        SpaceType spaceType,
        List<Set<String>> groundTruth,
        CompressionTestConfig config
    ) {
        createHnswIndex(indexName, engine, spaceType, config);
        bulkAddKnnDocs(indexName, FIELD_NAME, INDEX_VECTORS, DOC_COUNT);
        forceMergeKnnIndex(indexName, 1);

        List<List<String>> searchResults = bulkSearch(indexName, FIELD_NAME, QUERY_VECTORS, K);
        double recall = TestUtils.calculateRecallValue(searchResults, groundTruth, K);
        logger.info("{} {} {} recall: {}", engine, config.name(), spaceType.getValue(), recall);
        assertTrue(
            engine
                + " "
                + config.name()
                + " "
                + spaceType.getValue()
                + " recall should be >= "
                + config.getMinimumRecallThreshold()
                + " but was "
                + recall,
            recall >= config.getMinimumRecallThreshold()
        );

        deleteKNNIndex(indexName);
    }

    @SneakyThrows
    public void testDualCompression_faiss_lifecycle() {
        for (CompressionTestConfig config : CompressionTestConfig.values()) {
            runLifecycleTest("faiss_lifecycle_" + config.name().toLowerCase(Locale.ROOT), FAISS_NAME, config);
        }
    }

    @SneakyThrows
    public void testDualCompression_lucene_lifecycle() {
        for (CompressionTestConfig config : CompressionTestConfig.values()) {
            runLifecycleTest("lucene_lifecycle_" + config.name().toLowerCase(Locale.ROOT), LUCENE_NAME, config);
        }
    }

    @SneakyThrows
    private void runLifecycleTest(String indexName, String engine, CompressionTestConfig config) {
        createHnswIndex(indexName, engine, SpaceType.L2, config);
        bulkAddKnnDocs(indexName, FIELD_NAME, INDEX_VECTORS, DOC_COUNT);
        forceMergeKnnIndex(indexName, 1);

        List<List<String>> resultsAfterMerge = bulkSearch(indexName, FIELD_NAME, QUERY_VECTORS, K);
        double recallAfterMerge = TestUtils.calculateRecallValue(resultsAfterMerge, GROUND_TRUTH_L2, K);
        assertTrue(
            engine
                + " "
                + config.name()
                + " recall after force merge should be >= "
                + config.getMinimumRecallThreshold()
                + " but was "
                + recallAfterMerge,
            recallAfterMerge >= config.getMinimumRecallThreshold()
        );

        closeIndex(indexName);
        openIndex(indexName);
        ensureGreen(indexName);

        List<List<String>> resultsAfterReopen = bulkSearch(indexName, FIELD_NAME, QUERY_VECTORS, K);
        double recallAfterReopen = TestUtils.calculateRecallValue(resultsAfterReopen, GROUND_TRUTH_L2, K);
        assertTrue(
            engine
                + " "
                + config.name()
                + " recall after close/reopen should be >= "
                + config.getMinimumRecallThreshold()
                + " but was "
                + recallAfterReopen,
            recallAfterReopen >= config.getMinimumRecallThreshold()
        );

        String repositoryName = indexName + "-repo-" + randomLowerCaseString();
        String snapshotName = indexName + "-snap-" + getTestName().toLowerCase(Locale.ROOT);
        String pathRepo = System.getProperty("tests.path.repo");
        Settings repoSettings = Settings.builder().put("compress", randomBoolean()).put("location", pathRepo).build();
        registerRepository(repositoryName, "fs", true, repoSettings);
        createSnapshot(repositoryName, snapshotName, true);

        deleteKNNIndex(indexName);

        String restoreSuffix = "-restored";
        restoreSnapshot(restoreSuffix, List.of(indexName), repositoryName, snapshotName, true);
        String restoredIndexName = indexName + restoreSuffix;
        ensureGreen(restoredIndexName);

        List<List<String>> resultsAfterRestore = bulkSearch(restoredIndexName, FIELD_NAME, QUERY_VECTORS, K);
        double recallAfterRestore = TestUtils.calculateRecallValue(resultsAfterRestore, GROUND_TRUTH_L2, K);
        assertTrue(
            engine
                + " "
                + config.name()
                + " recall after snapshot/restore should be >= "
                + config.getMinimumRecallThreshold()
                + " but was "
                + recallAfterRestore,
            recallAfterRestore >= config.getMinimumRecallThreshold()
        );

        deleteKNNIndex(restoredIndexName);
    }

    @SneakyThrows
    private void runFilteredSearchTest(String indexName, String engine, CompressionTestConfig config) {
        createHnswIndexWithFilterField(indexName, engine, SpaceType.L2, config);

        int halfCount = DOC_COUNT / 2;
        for (int i = 0; i < DOC_COUNT; i++) {
            String category = (i < halfCount) ? "alpha" : "beta";
            addKnnDocWithAttributes(indexName, String.valueOf(i), FIELD_NAME, INDEX_VECTORS[i], Map.of("category", category));
        }
        forceMergeKnnIndex(indexName, 1);

        float[] queryVector = QUERY_VECTORS[0];
        KNNQueryBuilder filteredQuery = KNNQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(queryVector)
            .k(K)
            .filter(QueryBuilders.termQuery("category", "alpha"))
            .build();
        Response response = searchKNNIndex(indexName, filteredQuery, K);
        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), FIELD_NAME);

        assertFalse(engine + " " + config.name() + " filtered search should return results", results.isEmpty());
        assertTrue(results.size() <= K);
        List<String> returnedIds = results.stream().map(KNNResult::getDocId).collect(Collectors.toList());
        for (String id : returnedIds) {
            int docId = Integer.parseInt(id);
            assertTrue(
                engine + " " + config.name() + " filtered result should only contain alpha docs (id < " + halfCount + ")",
                docId < halfCount
            );
        }

        deleteKNNIndex(indexName);
    }

    @SneakyThrows
    private void createHnswIndex(String indexName, String engine, SpaceType spaceType, CompressionTestConfig config) {
        Settings settings = Settings.builder()
            .put("number_of_shards", 1)
            .put("number_of_replicas", 0)
            .put(KNN_INDEX, true)
            .put(INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD, 0)
            .build();

        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .field(COMPRESSION_LEVEL_PARAMETER, config.getCompressionLevelName());

        if (config.isCompressed()) {
            builder.field(MODE_PARAMETER, config.getModeName());
        }

        builder.startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(KNN_ENGINE, engine)
            .field(METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_M, HNSW_M)
            .field(METHOD_PARAMETER_EF_CONSTRUCTION, HNSW_EF_CONSTRUCTION);

        if (FAISS_NAME.equals(engine)) {
            builder.field(METHOD_PARAMETER_EF_SEARCH, HNSW_EF_SEARCH);
        }

        builder.endObject().endObject().endObject().endObject().endObject();

        createKnnIndex(indexName, settings, builder.toString());
    }

    @SneakyThrows
    private void createHnswIndexWithFilterField(String indexName, String engine, SpaceType spaceType, CompressionTestConfig config) {
        Settings settings = Settings.builder()
            .put("number_of_shards", 1)
            .put("number_of_replicas", 0)
            .put(KNN_INDEX, true)
            .put(INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD, 0)
            .build();

        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .field(COMPRESSION_LEVEL_PARAMETER, config.getCompressionLevelName());

        if (config.isCompressed()) {
            builder.field(MODE_PARAMETER, config.getModeName());
        }

        builder.startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(KNN_ENGINE, engine)
            .field(METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_M, HNSW_M)
            .field(METHOD_PARAMETER_EF_CONSTRUCTION, HNSW_EF_CONSTRUCTION);

        if (FAISS_NAME.equals(engine)) {
            builder.field(METHOD_PARAMETER_EF_SEARCH, HNSW_EF_SEARCH);
        }

        builder.endObject().endObject().endObject().startObject("category").field("type", "keyword").endObject().endObject().endObject();

        createKnnIndex(indexName, settings, builder.toString());
    }
}

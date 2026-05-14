/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import lombok.SneakyThrows;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.client.ResponseException;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.index.query.MatchAllQueryBuilder;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.index.query.QueryBuilders;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.KNNResult;
import org.opensearch.knn.TestUtils;
import org.opensearch.knn.index.query.KNNQueryBuilder;
import org.opensearch.knn.index.query.rescore.RescoreContext;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import static org.opensearch.knn.common.KNNConstants.COMPRESSION_LEVEL_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.FAISS_NAME;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.KNN_METHOD;
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

public class FaissX32CompressionIT extends KNNRestTestCase {

    private static final int DIMENSION = 128;
    private static final int DOC_COUNT = 100;
    private static final int QUERY_COUNT = 10;
    private static final int K = 10;
    private static final int HNSW_M = 16;
    private static final int HNSW_EF_CONSTRUCTION = 100;
    private static final int HNSW_EF_SEARCH = 100;
    private static final double MIN_RECALL_X32 = 0.70;
    private static final double MIN_RECALL_FP32 = 0.95;

    private static final int DIMENSION_SMALL = 64;
    private static final String FIELD_NAME_2 = "test_field_2";

    private static final int BASELINE_DOC_COUNT = 200;
    private static final int BASELINE_QUERY_COUNT = 20;

    private static final float[][] INDEX_VECTORS = TestUtils.getIndexVectors(DOC_COUNT, DIMENSION, true);
    private static final float[][] QUERY_VECTORS = TestUtils.getQueryVectors(QUERY_COUNT, DIMENSION, DOC_COUNT, true);
    private static final float[][] INDEX_VECTORS_SMALL = TestUtils.getIndexVectors(DOC_COUNT, DIMENSION_SMALL, true);
    private static final float[][] QUERY_VECTORS_SMALL = TestUtils.getQueryVectors(QUERY_COUNT, DIMENSION_SMALL, DOC_COUNT, true);
    private static final float[][] BASELINE_INDEX_VECTORS = TestUtils.getIndexVectors(BASELINE_DOC_COUNT, DIMENSION, true);
    private static final float[][] BASELINE_QUERY_VECTORS = TestUtils.getQueryVectors(
        BASELINE_QUERY_COUNT,
        DIMENSION,
        BASELINE_DOC_COUNT,
        true
    );
    private static final List<Set<String>> GROUND_TRUTH_L2 = TestUtils.computeGroundTruthValues(
        INDEX_VECTORS,
        QUERY_VECTORS,
        SpaceType.L2,
        K
    );
    private static final List<Set<String>> GROUND_TRUTH_IP = TestUtils.computeGroundTruthValues(
        INDEX_VECTORS,
        QUERY_VECTORS,
        SpaceType.INNER_PRODUCT,
        K
    );
    private static final List<Set<String>> GROUND_TRUTH_COSINE = TestUtils.computeGroundTruthValues(
        INDEX_VECTORS,
        QUERY_VECTORS,
        SpaceType.COSINESIMIL,
        K
    );
    private static final List<Set<String>> GROUND_TRUTH_SMALL = TestUtils.computeGroundTruthValues(
        INDEX_VECTORS_SMALL,
        QUERY_VECTORS_SMALL,
        SpaceType.L2,
        K
    );
    private static final List<Set<String>> GROUND_TRUTH_BASELINE = TestUtils.computeGroundTruthValues(
        BASELINE_INDEX_VECTORS,
        BASELINE_QUERY_VECTORS,
        SpaceType.L2,
        K
    );

    @SneakyThrows
    public void testX32_faiss_explicitOverride() {
        String indexName = "faiss_x32_explicit";
        createFaissX32Index(indexName, SpaceType.L2);
        bulkAddKnnDocs(indexName, FIELD_NAME, INDEX_VECTORS, DOC_COUNT);
        forceMergeKnnIndex(indexName, 1);

        List<List<String>> searchResults = bulkSearch(indexName, FIELD_NAME, QUERY_VECTORS, K);
        double recall = TestUtils.calculateRecallValue(searchResults, GROUND_TRUTH_L2, K);
        logger.info("Faiss x32 explicit recall: {}", recall);
        assertTrue("Faiss x32 recall should be >= " + MIN_RECALL_X32 + " but was " + recall, recall >= MIN_RECALL_X32);
    }

    // TODO: [DEFAULT_FLIP] Add testX32_faiss_implicitDefault: create index with NO mode, NO compression,
    // assert recall ~0.70 proving default resolved to x32

    @SneakyThrows
    public void testX32_faiss_optOut() {
        String indexName = "faiss_fp32_optout";
        createFaissFP32Index(indexName, SpaceType.L2);
        bulkAddKnnDocs(indexName, FIELD_NAME, INDEX_VECTORS, DOC_COUNT);
        forceMergeKnnIndex(indexName, 1);

        List<List<String>> searchResults = bulkSearch(indexName, FIELD_NAME, QUERY_VECTORS, K);
        double recall = TestUtils.calculateRecallValue(searchResults, GROUND_TRUTH_L2, K);
        logger.info("Faiss FP32 opt-out recall: {}", recall);
        assertTrue("FP32 recall should be >= " + MIN_RECALL_FP32 + " but was " + recall, recall >= MIN_RECALL_FP32);
    }

    @SneakyThrows
    public void testX32_faiss_spaceTypes() {
        String indexL2 = "faiss_x32_l2";
        String indexIP = "faiss_x32_ip";
        String indexCosine = "faiss_x32_cosine";

        createFaissX32Index(indexL2, SpaceType.L2);
        createFaissX32Index(indexIP, SpaceType.INNER_PRODUCT);
        createFaissX32Index(indexCosine, SpaceType.COSINESIMIL);

        bulkAddKnnDocs(indexL2, FIELD_NAME, INDEX_VECTORS, DOC_COUNT);
        bulkAddKnnDocs(indexIP, FIELD_NAME, INDEX_VECTORS, DOC_COUNT);
        bulkAddKnnDocs(indexCosine, FIELD_NAME, INDEX_VECTORS, DOC_COUNT);

        forceMergeKnnIndex(indexL2, 1);
        forceMergeKnnIndex(indexIP, 1);
        forceMergeKnnIndex(indexCosine, 1);

        List<List<String>> resultsL2 = bulkSearch(indexL2, FIELD_NAME, QUERY_VECTORS, K);
        double recallL2 = TestUtils.calculateRecallValue(resultsL2, GROUND_TRUTH_L2, K);
        logger.info("Faiss x32 L2 recall: {}", recallL2);
        assertTrue("L2 recall should be >= " + MIN_RECALL_X32 + " but was " + recallL2, recallL2 >= MIN_RECALL_X32);

        List<List<String>> resultsIP = bulkSearch(indexIP, FIELD_NAME, QUERY_VECTORS, K);
        double recallIP = TestUtils.calculateRecallValue(resultsIP, GROUND_TRUTH_IP, K);
        logger.info("Faiss x32 innerproduct recall: {}", recallIP);
        assertTrue("IP recall should be >= " + MIN_RECALL_X32 + " but was " + recallIP, recallIP >= MIN_RECALL_X32);

        List<List<String>> resultsCosine = bulkSearch(indexCosine, FIELD_NAME, QUERY_VECTORS, K);
        double recallCosine = TestUtils.calculateRecallValue(resultsCosine, GROUND_TRUTH_COSINE, K);
        logger.info("Faiss x32 cosine recall: {}", recallCosine);
        assertTrue("Cosine recall should be >= " + MIN_RECALL_X32 + " but was " + recallCosine, recallCosine >= MIN_RECALL_X32);
    }

    @SneakyThrows
    public void testX32_faiss_multiField() {
        String indexName = "faiss_x32_multifield";
        createFaissX32MultiFieldIndex(indexName);

        for (int i = 0; i < DOC_COUNT; i++) {
            addKnnDoc(
                indexName,
                String.valueOf(i),
                Arrays.asList(FIELD_NAME, FIELD_NAME_2),
                Arrays.asList(toObjectArray(INDEX_VECTORS[i]), toObjectArray(INDEX_VECTORS_SMALL[i]))
            );
        }
        refreshIndex(indexName);
        forceMergeKnnIndex(indexName, 1);

        List<List<String>> resultsField1 = bulkSearch(indexName, FIELD_NAME, QUERY_VECTORS, K);
        double recallField1 = TestUtils.calculateRecallValue(resultsField1, GROUND_TRUTH_L2, K);
        logger.info("Faiss x32 multi-field field1 recall: {}", recallField1);
        assertTrue("Field1 recall should be >= " + MIN_RECALL_X32 + " but was " + recallField1, recallField1 >= MIN_RECALL_X32);

        List<List<String>> resultsField2 = bulkSearch(indexName, FIELD_NAME_2, QUERY_VECTORS_SMALL, K);
        double recallField2 = TestUtils.calculateRecallValue(resultsField2, GROUND_TRUTH_SMALL, K);
        logger.info("Faiss x32 multi-field field2 recall: {}", recallField2);
        assertTrue("Field2 recall should be >= " + MIN_RECALL_X32 + " but was " + recallField2, recallField2 >= MIN_RECALL_X32);
    }

    @SneakyThrows
    public void testX32_faiss_lifecycle() {
        String indexName = "faiss_x32_lifecycle";
        createFaissX32Index(indexName, SpaceType.L2);
        bulkAddKnnDocs(indexName, FIELD_NAME, INDEX_VECTORS, DOC_COUNT);
        forceMergeKnnIndex(indexName, 1);

        List<List<String>> resultsBeforeMerge = bulkSearch(indexName, FIELD_NAME, QUERY_VECTORS, K);
        double recallAfterMerge = TestUtils.calculateRecallValue(resultsBeforeMerge, GROUND_TRUTH_L2, K);
        assertTrue(
            "Recall after force merge should be >= " + MIN_RECALL_X32 + " but was " + recallAfterMerge,
            recallAfterMerge >= MIN_RECALL_X32
        );

        closeIndex(indexName);
        openIndex(indexName);
        ensureGreen(indexName);

        List<List<String>> resultsAfterReopen = bulkSearch(indexName, FIELD_NAME, QUERY_VECTORS, K);
        double recallAfterReopen = TestUtils.calculateRecallValue(resultsAfterReopen, GROUND_TRUTH_L2, K);
        assertTrue(
            "Recall after close/reopen should be >= " + MIN_RECALL_X32 + " but was " + recallAfterReopen,
            recallAfterReopen >= MIN_RECALL_X32
        );

        String repositoryName = "faiss-x32-repo-" + randomLowerCaseString();
        String snapshotName = "faiss-x32-snap-" + getTestName().toLowerCase(Locale.ROOT);
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
            "Recall after snapshot/restore should be >= " + MIN_RECALL_X32 + " but was " + recallAfterRestore,
            recallAfterRestore >= MIN_RECALL_X32
        );
    }

    @SneakyThrows
    public void testX32_faiss_radialBlocked() {
        String indexName = "faiss_x32_radial_blocked";
        createFaissX32Index(indexName, SpaceType.L2);
        addKnnDoc(indexName, "0", FIELD_NAME, INDEX_VECTORS[0]);
        refreshIndex(indexName);

        float[] queryVector = QUERY_VECTORS[0];

        String maxDistanceQuery = buildRadialSearchQuery(FIELD_NAME, queryVector, "max_distance", 100.0f);
        Request maxDistanceRequest = new Request("POST", "/" + indexName + "/_search");
        maxDistanceRequest.setJsonEntity(maxDistanceQuery);
        ResponseException maxDistEx = expectThrows(ResponseException.class, () -> client().performRequest(maxDistanceRequest));
        assertTrue(maxDistEx.getMessage().contains("Radial search is not supported for indices which have quantization enabled"));

        String minScoreQuery = buildRadialSearchQuery(FIELD_NAME, queryVector, "min_score", 0.01f);
        Request minScoreRequest = new Request("POST", "/" + indexName + "/_search");
        minScoreRequest.setJsonEntity(minScoreQuery);
        ResponseException minScoreEx = expectThrows(ResponseException.class, () -> client().performRequest(minScoreRequest));
        assertTrue(minScoreEx.getMessage().contains("Radial search is not supported for indices which have quantization enabled"));
    }

    @SneakyThrows
    public void testX32_faiss_scriptScoring() {
        String x32IndexName = "faiss_x32_script";
        String fp32IndexName = "faiss_fp32_script";
        int docCount = 20;
        int dimension = 16;

        float[][] vectors = TestUtils.getIndexVectors(docCount, dimension, true);
        float[] queryVector = TestUtils.getQueryVectors(1, dimension, docCount, true)[0];

        createFaissIndexForScriptScoring(x32IndexName, SpaceType.L2, dimension, true);
        createFaissIndexForScriptScoring(fp32IndexName, SpaceType.L2, dimension, false);

        for (int i = 0; i < docCount; i++) {
            addKnnDoc(x32IndexName, String.valueOf(i), FIELD_NAME, vectors[i]);
            addKnnDoc(fp32IndexName, String.valueOf(i), FIELD_NAME, vectors[i]);
        }
        forceMergeKnnIndex(x32IndexName, 1);
        forceMergeKnnIndex(fp32IndexName, 1);

        QueryBuilder qb = new MatchAllQueryBuilder();
        Map<String, Object> params = new HashMap<>();
        params.put("field", FIELD_NAME);
        params.put("query_value", queryVector);
        params.put("space_type", SpaceType.L2.getValue());

        Request x32Request = constructKNNScriptQueryRequest(x32IndexName, qb, params, docCount);
        Response x32Response = client().performRequest(x32Request);
        List<KNNResult> x32Results = parseSearchResponse(EntityUtils.toString(x32Response.getEntity()), FIELD_NAME);

        Request fp32Request = constructKNNScriptQueryRequest(fp32IndexName, qb, params, docCount);
        Response fp32Response = client().performRequest(fp32Request);
        List<KNNResult> fp32Results = parseSearchResponse(EntityUtils.toString(fp32Response.getEntity()), FIELD_NAME);

        assertEquals(docCount, x32Results.size());
        assertEquals(docCount, fp32Results.size());

        for (int i = 0; i < x32Results.size(); i++) {
            assertEquals(fp32Results.get(i).getDocId(), x32Results.get(i).getDocId());
            assertEquals(fp32Results.get(i).getScore(), x32Results.get(i).getScore(), 0.001f);
        }
    }

    @SneakyThrows
    public void testX32_faiss_recallBaseline() {
        String indexName = "faiss_x32_recall_baseline";
        createFaissX32Index(indexName, SpaceType.L2);
        bulkAddKnnDocs(indexName, FIELD_NAME, BASELINE_INDEX_VECTORS, BASELINE_DOC_COUNT);
        forceMergeKnnIndex(indexName, 1);

        List<List<String>> searchResults = bulkSearch(indexName, FIELD_NAME, BASELINE_QUERY_VECTORS, K);
        double recall = TestUtils.calculateRecallValue(searchResults, GROUND_TRUTH_BASELINE, K);
        logger.info("Faiss x32 recall baseline (200 vectors, 128-dim): {}", recall);
        assertTrue("Faiss x32 recall baseline should be >= " + MIN_RECALL_X32 + " but was " + recall, recall >= MIN_RECALL_X32);
    }

    @SneakyThrows
    public void testX32_faiss_inMemory() {
        String indexName = "faiss_x32_in_memory";
        createFaissX32InMemoryIndex(indexName, SpaceType.L2);
        bulkAddKnnDocs(indexName, FIELD_NAME, INDEX_VECTORS, DOC_COUNT);
        forceMergeKnnIndex(indexName, 1);

        List<List<String>> searchResults = bulkSearch(indexName, FIELD_NAME, QUERY_VECTORS, K);
        double recall = TestUtils.calculateRecallValue(searchResults, GROUND_TRUTH_L2, K);
        logger.info("Faiss x32 in-memory recall: {}", recall);
        assertTrue("Faiss x32 in-memory recall should be >= " + MIN_RECALL_X32 + " but was " + recall, recall >= MIN_RECALL_X32);
    }

    @SneakyThrows
    public void testX32_faiss_rescoreWithInMemory() {
        String indexName = "faiss_x32_in_memory_rescore";
        createFaissX32InMemoryIndex(indexName, SpaceType.L2);
        bulkAddKnnDocs(indexName, FIELD_NAME, INDEX_VECTORS, DOC_COUNT);
        forceMergeKnnIndex(indexName, 1);

        List<List<KNNResult>> noRescoreResults = searchWithRescore(
            indexName,
            QUERY_VECTORS,
            K,
            RescoreContext.builder().rescoreEnabled(false).build()
        );

        List<List<KNNResult>> withRescoreResults = searchWithRescore(
            indexName,
            QUERY_VECTORS,
            K,
            RescoreContext.builder().oversampleFactor(3.0f).build()
        );

        for (int q = 0; q < QUERY_COUNT; q++) {
            List<KNNResult> noRescore = noRescoreResults.get(q);
            List<KNNResult> withRescore = withRescoreResults.get(q);
            assertEquals(noRescore.size(), withRescore.size());
            for (int i = 0; i < noRescore.size(); i++) {
                assertEquals(
                    "In-memory mode: doc IDs should match regardless of rescore setting",
                    noRescore.get(i).getDocId(),
                    withRescore.get(i).getDocId()
                );
                assertEquals(
                    "In-memory mode: scores should match regardless of rescore setting",
                    noRescore.get(i).getScore(),
                    withRescore.get(i).getScore(),
                    0.001f
                );
            }
        }
    }

    @SneakyThrows
    public void testX32_faiss_filtering() {
        String indexName = "faiss_x32_filter";
        createFaissX32IndexWithFilterField(indexName, SpaceType.L2);

        int halfCount = DOC_COUNT / 2;
        for (int i = 0; i < DOC_COUNT; i++) {
            String category = (i < halfCount) ? "alpha" : "beta";
            addKnnDocWithAttributes(indexName, String.valueOf(i), FIELD_NAME, INDEX_VECTORS[i], Map.of("category", category));
        }
        forceMergeKnnIndex(indexName, 1);

        float[] queryVector = QUERY_VECTORS[0];
        KNNQueryBuilder filteredQuery = new KNNQueryBuilder(FIELD_NAME, queryVector, K, QueryBuilders.termQuery("category", "alpha"));
        Response response = searchKNNIndex(indexName, filteredQuery, K);
        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), FIELD_NAME);

        assertFalse(results.isEmpty());
        assertTrue(results.size() <= K);
        List<String> returnedIds = results.stream().map(KNNResult::getDocId).collect(Collectors.toList());
        for (String id : returnedIds) {
            int docId = Integer.parseInt(id);
            assertTrue("Filtered result should only contain alpha docs (id < " + halfCount + ")", docId < halfCount);
        }
    }

    @SneakyThrows
    public void testX32_faiss_filteringWithRescore() {
        String indexName = "faiss_x32_filter_rescore";
        createFaissX32IndexWithFilterField(indexName, SpaceType.L2);

        int halfCount = BASELINE_DOC_COUNT / 2;
        for (int i = 0; i < BASELINE_DOC_COUNT; i++) {
            String category = (i < halfCount) ? "alpha" : "beta";
            addKnnDocWithAttributes(indexName, String.valueOf(i), FIELD_NAME, BASELINE_INDEX_VECTORS[i], Map.of("category", category));
        }
        forceMergeKnnIndex(indexName, 1);

        float[] queryVector = BASELINE_QUERY_VECTORS[0];
        QueryBuilder filter = QueryBuilders.termQuery("category", "alpha");

        KNNQueryBuilder filteredRescoreQuery = KNNQueryBuilder.builder()
            .fieldName(FIELD_NAME)
            .vector(queryVector)
            .k(K)
            .filter(filter)
            .rescoreContext(RescoreContext.builder().oversampleFactor(3.0f).build())
            .build();

        Response firstResponse = searchKNNIndex(indexName, filteredRescoreQuery, K);
        List<KNNResult> firstResults = parseSearchResponse(EntityUtils.toString(firstResponse.getEntity()), FIELD_NAME);

        Response secondResponse = searchKNNIndex(indexName, filteredRescoreQuery, K);
        List<KNNResult> secondResults = parseSearchResponse(EntityUtils.toString(secondResponse.getEntity()), FIELD_NAME);

        assertFalse(firstResults.isEmpty());
        assertTrue(firstResults.size() <= K);

        for (KNNResult result : firstResults) {
            int docId = Integer.parseInt(result.getDocId());
            assertTrue("Filtered+rescored result should only contain alpha docs (id < " + halfCount + ")", docId < halfCount);
        }

        List<String> firstIds = firstResults.stream().map(KNNResult::getDocId).collect(Collectors.toList());
        List<String> secondIds = secondResults.stream().map(KNNResult::getDocId).collect(Collectors.toList());
        assertEquals("Rescored results should be stable across repeated queries", firstIds, secondIds);

        for (int i = 0; i < firstResults.size() - 1; i++) {
            assertTrue("Scores should be in descending order", firstResults.get(i).getScore() >= firstResults.get(i + 1).getScore());
        }
    }

    @SneakyThrows
    public void testX32_faiss_nested() {
        String indexName = "faiss_x32_nested";
        String nestedPath = "nested_field";
        String nestedFieldPath = "nested_field.vector";
        int nestedDimension = 16;
        int numDocs = 10;
        int k = 5;

        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(nestedPath)
            .field("type", "nested")
            .startObject("properties")
            .startObject("vector")
            .field("type", "knn_vector")
            .field("dimension", nestedDimension)
            .field(MODE_PARAMETER, "on_disk")
            .field(COMPRESSION_LEVEL_PARAMETER, "32x")
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(KNN_ENGINE, FAISS_NAME)
            .field(METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue())
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_M, HNSW_M)
            .field(METHOD_PARAMETER_EF_CONSTRUCTION, HNSW_EF_CONSTRUCTION)
            .field(METHOD_PARAMETER_EF_SEARCH, HNSW_EF_SEARCH)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        Settings settings = Settings.builder()
            .put("number_of_shards", 1)
            .put("number_of_replicas", 0)
            .put(KNN_INDEX, true)
            .put(INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD, 0)
            .build();

        createKnnIndex(indexName, settings, builder.toString());
        bulkIngestRandomVectorsWithNestedField(indexName, nestedFieldPath, numDocs, nestedDimension);
        refreshIndex(indexName);
        forceMergeKnnIndex(indexName, 1);

        assertEquals(numDocs, getDocCount(indexName));

        float[] queryVector = new float[nestedDimension];
        Arrays.fill(queryVector, 0.5f);
        XContentBuilder queryBuilder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("nested")
            .field("path", nestedPath)
            .startObject("query")
            .startObject("knn")
            .startObject(nestedFieldPath)
            .field("vector", queryVector)
            .field("k", k)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        Response searchResponse = searchKNNIndex(indexName, queryBuilder, k);
        List<Object> hits = parseSearchResponseHits(EntityUtils.toString(searchResponse.getEntity()));
        assertFalse(hits.isEmpty());
        assertTrue(hits.size() <= k);
    }

    @SneakyThrows
    public void testX32_faiss_rescoreVariations() {
        String indexName = "faiss_x32_rescore_variations";
        createFaissX32Index(indexName, SpaceType.L2);
        bulkAddKnnDocs(indexName, FIELD_NAME, BASELINE_INDEX_VECTORS, BASELINE_DOC_COUNT);
        forceMergeKnnIndex(indexName, 1);

        List<Set<String>> groundTruth = computeScriptScoreGroundTruth(indexName, BASELINE_QUERY_VECTORS, K);

        List<List<KNNResult>> noRescoreResults = searchWithRescore(
            indexName,
            BASELINE_QUERY_VECTORS,
            K,
            RescoreContext.builder().rescoreEnabled(false).build()
        );
        double recallNoRescore = calculateRecallFromResults(noRescoreResults, groundTruth, K);
        logger.info("Faiss x32 rescore=false recall: {}", recallNoRescore);

        List<List<KNNResult>> defaultRescoreResults = searchWithRescore(
            indexName,
            BASELINE_QUERY_VECTORS,
            K,
            RescoreContext.builder().oversampleFactor(3.0f).build()
        );
        double recallDefaultRescore = calculateRecallFromResults(defaultRescoreResults, groundTruth, K);
        logger.info("Faiss x32 rescore=true (3.0x) recall: {}", recallDefaultRescore);
        assertTrue(
            "Default rescore recall should be >= " + MIN_RECALL_X32 + " but was " + recallDefaultRescore,
            recallDefaultRescore >= MIN_RECALL_X32
        );

        List<List<KNNResult>> highOversampleResults = searchWithRescore(
            indexName,
            BASELINE_QUERY_VECTORS,
            K,
            RescoreContext.builder().oversampleFactor(5.0f).build()
        );
        double recallHighOversample = calculateRecallFromResults(highOversampleResults, groundTruth, K);
        logger.info("Faiss x32 rescore oversample=5.0 recall: {}", recallHighOversample);
        assertTrue("High oversample recall should be >= 0.75 but was " + recallHighOversample, recallHighOversample >= 0.75);

        List<List<KNNResult>> lowOversampleResults = searchWithRescore(
            indexName,
            BASELINE_QUERY_VECTORS,
            K,
            RescoreContext.builder().oversampleFactor(1.0f).build()
        );
        double recallLowOversample = calculateRecallFromResults(lowOversampleResults, groundTruth, K);
        logger.info("Faiss x32 rescore oversample=1.0 recall: {}", recallLowOversample);
        assertTrue("Low oversample recall should be >= 0.60 but was " + recallLowOversample, recallLowOversample >= 0.60);

        assertTrue("Rescore recall should be >= no-rescore recall", recallDefaultRescore >= recallNoRescore);
        assertTrue("High oversample recall should be >= default rescore recall", recallHighOversample >= recallDefaultRescore);
    }

    @SneakyThrows
    private List<Set<String>> computeScriptScoreGroundTruth(String indexName, float[][] queryVectors, int k) {
        List<Set<String>> groundTruth = new ArrayList<>();
        QueryBuilder qb = new MatchAllQueryBuilder();
        for (float[] queryVector : queryVectors) {
            Map<String, Object> params = new HashMap<>();
            params.put("field", FIELD_NAME);
            params.put("query_value", queryVector);
            params.put("space_type", SpaceType.L2.getValue());

            Request request = constructKNNScriptQueryRequest(indexName, qb, params, k);
            Response response = client().performRequest(request);
            List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), FIELD_NAME);
            Set<String> topK = results.stream().map(KNNResult::getDocId).collect(Collectors.toSet());
            groundTruth.add(topK);
        }
        return groundTruth;
    }

    @SneakyThrows
    private List<List<KNNResult>> searchWithRescore(String indexName, float[][] queryVectors, int k, RescoreContext rescoreContext) {
        List<List<KNNResult>> allResults = new ArrayList<>();
        for (float[] queryVector : queryVectors) {
            KNNQueryBuilder query = KNNQueryBuilder.builder()
                .fieldName(FIELD_NAME)
                .vector(queryVector)
                .k(k)
                .rescoreContext(rescoreContext)
                .build();
            Response response = searchKNNIndex(indexName, query, k);
            List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), FIELD_NAME);
            allResults.add(results);
        }
        return allResults;
    }

    private double calculateRecallFromResults(List<List<KNNResult>> searchResults, List<Set<String>> groundTruth, int k) {
        int totalRelevant = 0;
        int totalExpected = 0;
        for (int i = 0; i < searchResults.size(); i++) {
            Set<String> truth = groundTruth.get(i);
            List<KNNResult> results = searchResults.get(i);
            for (KNNResult result : results) {
                if (truth.contains(result.getDocId())) {
                    totalRelevant++;
                }
            }
            totalExpected += Math.min(k, truth.size());
        }
        return totalExpected == 0 ? 0.0 : (double) totalRelevant / totalExpected;
    }

    @SneakyThrows
    private void createFaissX32IndexWithFilterField(String indexName, SpaceType spaceType) {
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
            .field(MODE_PARAMETER, "on_disk")
            .field(COMPRESSION_LEVEL_PARAMETER, "32x")
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(KNN_ENGINE, FAISS_NAME)
            .field(METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_M, HNSW_M)
            .field(METHOD_PARAMETER_EF_CONSTRUCTION, HNSW_EF_CONSTRUCTION)
            .field(METHOD_PARAMETER_EF_SEARCH, HNSW_EF_SEARCH)
            .endObject()
            .endObject()
            .endObject()
            .startObject("category")
            .field("type", "keyword")
            .endObject()
            .endObject()
            .endObject();

        createKnnIndex(indexName, settings, builder.toString());
    }

    @SneakyThrows
    private void createFaissX32InMemoryIndex(String indexName, SpaceType spaceType) {
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
            .field(MODE_PARAMETER, "in_memory")
            .field(COMPRESSION_LEVEL_PARAMETER, "32x")
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(KNN_ENGINE, FAISS_NAME)
            .field(METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_M, HNSW_M)
            .field(METHOD_PARAMETER_EF_CONSTRUCTION, HNSW_EF_CONSTRUCTION)
            .field(METHOD_PARAMETER_EF_SEARCH, HNSW_EF_SEARCH)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        createKnnIndex(indexName, settings, builder.toString());
    }

    @SneakyThrows
    private void createFaissIndexForScriptScoring(String indexName, SpaceType spaceType, int dimension, boolean useX32) {
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
            .field("dimension", dimension);

        if (useX32) {
            builder.field(MODE_PARAMETER, "on_disk");
            builder.field(COMPRESSION_LEVEL_PARAMETER, "32x");
        } else {
            builder.field(COMPRESSION_LEVEL_PARAMETER, "1x");
        }

        builder.startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(KNN_ENGINE, FAISS_NAME)
            .field(METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_M, HNSW_M)
            .field(METHOD_PARAMETER_EF_CONSTRUCTION, HNSW_EF_CONSTRUCTION)
            .field(METHOD_PARAMETER_EF_SEARCH, HNSW_EF_SEARCH)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        createKnnIndex(indexName, settings, builder.toString());
    }

    @SneakyThrows
    private String buildRadialSearchQuery(String fieldName, float[] vector, String thresholdType, float thresholdValue) {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("knn")
            .startObject(fieldName)
            .field("vector", vector)
            .field(thresholdType, thresholdValue)
            .endObject()
            .endObject()
            .endObject()
            .endObject();
        return builder.toString();
    }

    @SneakyThrows
    private void createFaissX32MultiFieldIndex(String indexName) {
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
            .field(MODE_PARAMETER, "on_disk")
            .field(COMPRESSION_LEVEL_PARAMETER, "32x")
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(KNN_ENGINE, FAISS_NAME)
            .field(METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue())
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_M, HNSW_M)
            .field(METHOD_PARAMETER_EF_CONSTRUCTION, HNSW_EF_CONSTRUCTION)
            .field(METHOD_PARAMETER_EF_SEARCH, HNSW_EF_SEARCH)
            .endObject()
            .endObject()
            .endObject()
            .startObject(FIELD_NAME_2)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION_SMALL)
            .field(MODE_PARAMETER, "on_disk")
            .field(COMPRESSION_LEVEL_PARAMETER, "32x")
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(KNN_ENGINE, FAISS_NAME)
            .field(METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue())
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_M, HNSW_M)
            .field(METHOD_PARAMETER_EF_CONSTRUCTION, HNSW_EF_CONSTRUCTION)
            .field(METHOD_PARAMETER_EF_SEARCH, HNSW_EF_SEARCH)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        createKnnIndex(indexName, settings, builder.toString());
    }

    private Object[] toObjectArray(float[] vector) {
        Object[] result = new Object[vector.length];
        for (int i = 0; i < vector.length; i++) {
            result[i] = vector[i];
        }
        return result;
    }

    @SneakyThrows
    private void createFaissX32Index(String indexName, SpaceType spaceType) {
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
            .field(MODE_PARAMETER, "on_disk")
            .field(COMPRESSION_LEVEL_PARAMETER, "32x")
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(KNN_ENGINE, FAISS_NAME)
            .field(METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_M, HNSW_M)
            .field(METHOD_PARAMETER_EF_CONSTRUCTION, HNSW_EF_CONSTRUCTION)
            .field(METHOD_PARAMETER_EF_SEARCH, HNSW_EF_SEARCH)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        createKnnIndex(indexName, settings, builder.toString());
    }

    @SneakyThrows
    private void createFaissFP32Index(String indexName, SpaceType spaceType) {
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
            .field(COMPRESSION_LEVEL_PARAMETER, "1x")
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(KNN_ENGINE, FAISS_NAME)
            .field(METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_M, HNSW_M)
            .field(METHOD_PARAMETER_EF_CONSTRUCTION, HNSW_EF_CONSTRUCTION)
            .field(METHOD_PARAMETER_EF_SEARCH, HNSW_EF_SEARCH)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        createKnnIndex(indexName, settings, builder.toString());
    }
}

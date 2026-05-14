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
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.KNN_METHOD;
import static org.opensearch.knn.common.KNNConstants.LUCENE_NAME;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_M;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.MODE_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.index.KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD;
import static org.opensearch.knn.index.KNNSettings.KNN_INDEX;

public class LuceneX32CompressionIT extends KNNRestTestCase {

    private static final int DIMENSION = 128;
    private static final int DOC_COUNT = 100;
    private static final int QUERY_COUNT = 10;
    private static final int K = 10;
    private static final int HNSW_M = 16;
    private static final int HNSW_EF_CONSTRUCTION = 100;
    private static final double MIN_RECALL_X32 = 0.70;

    private static final float[][] INDEX_VECTORS = TestUtils.getIndexVectors(DOC_COUNT, DIMENSION, true);
    private static final float[][] QUERY_VECTORS = TestUtils.getQueryVectors(QUERY_COUNT, DIMENSION, DOC_COUNT, true);
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

    private static final int BASELINE_DOC_COUNT = 200;
    private static final int BASELINE_QUERY_COUNT = 20;
    private static final float[][] BASELINE_INDEX_VECTORS = TestUtils.getIndexVectors(BASELINE_DOC_COUNT, DIMENSION, true);
    private static final float[][] BASELINE_QUERY_VECTORS = TestUtils.getQueryVectors(
        BASELINE_QUERY_COUNT,
        DIMENSION,
        BASELINE_DOC_COUNT,
        true
    );

    @SneakyThrows
    public void testX32_lucene_kSearch() {
        String indexName = "lucene_x32_ksearch";
        createLuceneX32Index(indexName, SpaceType.L2);
        bulkAddKnnDocs(indexName, FIELD_NAME, INDEX_VECTORS, DOC_COUNT);
        forceMergeKnnIndex(indexName, 1);

        List<List<String>> searchResults = bulkSearch(indexName, FIELD_NAME, QUERY_VECTORS, K);
        double recall = TestUtils.calculateRecallValue(searchResults, GROUND_TRUTH_L2, K);
        logger.info("Lucene x32 k-search recall: {}", recall);
        assertTrue("Lucene x32 recall should be >= " + MIN_RECALL_X32 + " but was " + recall, recall >= MIN_RECALL_X32);
    }

    @SneakyThrows
    public void testX32_lucene_spaceTypes() {
        String indexL2 = "lucene_x32_l2";
        String indexIP = "lucene_x32_ip";
        String indexCosine = "lucene_x32_cosine";

        createLuceneX32Index(indexL2, SpaceType.L2);
        createLuceneX32Index(indexIP, SpaceType.INNER_PRODUCT);
        createLuceneX32Index(indexCosine, SpaceType.COSINESIMIL);

        bulkAddKnnDocs(indexL2, FIELD_NAME, INDEX_VECTORS, DOC_COUNT);
        bulkAddKnnDocs(indexIP, FIELD_NAME, INDEX_VECTORS, DOC_COUNT);
        bulkAddKnnDocs(indexCosine, FIELD_NAME, INDEX_VECTORS, DOC_COUNT);

        forceMergeKnnIndex(indexL2, 1);
        forceMergeKnnIndex(indexIP, 1);
        forceMergeKnnIndex(indexCosine, 1);

        List<List<String>> resultsL2 = bulkSearch(indexL2, FIELD_NAME, QUERY_VECTORS, K);
        double recallL2 = TestUtils.calculateRecallValue(resultsL2, GROUND_TRUTH_L2, K);
        logger.info("Lucene x32 L2 recall: {}", recallL2);
        assertTrue("L2 recall should be >= " + MIN_RECALL_X32 + " but was " + recallL2, recallL2 >= MIN_RECALL_X32);

        List<List<String>> resultsIP = bulkSearch(indexIP, FIELD_NAME, QUERY_VECTORS, K);
        double recallIP = TestUtils.calculateRecallValue(resultsIP, GROUND_TRUTH_IP, K);
        logger.info("Lucene x32 innerproduct recall: {}", recallIP);
        assertTrue("IP recall should be >= " + MIN_RECALL_X32 + " but was " + recallIP, recallIP >= MIN_RECALL_X32);

        List<List<String>> resultsCosine = bulkSearch(indexCosine, FIELD_NAME, QUERY_VECTORS, K);
        double recallCosine = TestUtils.calculateRecallValue(resultsCosine, GROUND_TRUTH_COSINE, K);
        logger.info("Lucene x32 cosine recall: {}", recallCosine);
        assertTrue("Cosine recall should be >= " + MIN_RECALL_X32 + " but was " + recallCosine, recallCosine >= MIN_RECALL_X32);
    }

    @SneakyThrows
    public void testX32_lucene_filtering() {
        String indexName = "lucene_x32_filter";
        createLuceneX32IndexWithFilterField(indexName, SpaceType.L2);

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
    public void testX32_lucene_nested() {
        String indexName = "lucene_x32_nested";
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
            .field(KNN_ENGINE, LUCENE_NAME)
            .field(METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue())
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_M, HNSW_M)
            .field(METHOD_PARAMETER_EF_CONSTRUCTION, HNSW_EF_CONSTRUCTION)
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
    public void testX32_lucene_lifecycle() {
        String indexName = "lucene_x32_lifecycle";
        createLuceneX32Index(indexName, SpaceType.L2);
        bulkAddKnnDocs(indexName, FIELD_NAME, INDEX_VECTORS, DOC_COUNT);
        forceMergeKnnIndex(indexName, 1);

        List<List<String>> resultsAfterMerge = bulkSearch(indexName, FIELD_NAME, QUERY_VECTORS, K);
        double recallAfterMerge = TestUtils.calculateRecallValue(resultsAfterMerge, GROUND_TRUTH_L2, K);
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

        String repositoryName = "lucene-x32-repo-" + randomLowerCaseString();
        String snapshotName = "lucene-x32-snap-" + getTestName().toLowerCase(Locale.ROOT);
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
    public void testX32_lucene_dimensions() {
        int highDim = 768;
        int lowDim = 3;
        int docCount = 50;
        int queryCount = 5;

        float[][] highDimVectors = TestUtils.getIndexVectors(docCount, highDim, true);
        float[][] highDimQueries = TestUtils.getQueryVectors(queryCount, highDim, docCount, true);
        float[][] lowDimVectors = TestUtils.getIndexVectors(docCount, lowDim, true);
        float[][] lowDimQueries = TestUtils.getQueryVectors(queryCount, lowDim, docCount, true);

        String highDimIndex = "lucene_x32_highdim";
        createLuceneX32IndexWithDimension(highDimIndex, SpaceType.L2, highDim);
        bulkAddKnnDocs(highDimIndex, FIELD_NAME, highDimVectors, docCount);
        forceMergeKnnIndex(highDimIndex, 1);

        String lowDimIndex = "lucene_x32_lowdim";
        createLuceneX32IndexWithDimension(lowDimIndex, SpaceType.L2, lowDim);
        bulkAddKnnDocs(lowDimIndex, FIELD_NAME, lowDimVectors, docCount);
        forceMergeKnnIndex(lowDimIndex, 1);

        List<Set<String>> groundTruthHighDim = TestUtils.computeGroundTruthValues(highDimVectors, highDimQueries, SpaceType.L2, K);
        List<List<String>> highDimResults = bulkSearch(highDimIndex, FIELD_NAME, highDimQueries, K);
        double recallHighDim = TestUtils.calculateRecallValue(highDimResults, groundTruthHighDim, K);
        logger.info("Lucene x32 high-dim ({}) recall: {}", highDim, recallHighDim);
        assertTrue("High-dim recall should be >= " + MIN_RECALL_X32 + " but was " + recallHighDim, recallHighDim >= MIN_RECALL_X32);

        List<Set<String>> groundTruthLowDim = TestUtils.computeGroundTruthValues(lowDimVectors, lowDimQueries, SpaceType.L2, K);
        List<List<String>> lowDimResults = bulkSearch(lowDimIndex, FIELD_NAME, lowDimQueries, K);
        double recallLowDim = TestUtils.calculateRecallValue(lowDimResults, groundTruthLowDim, K);
        logger.info("Lucene x32 low-dim ({}) recall: {}", lowDim, recallLowDim);
        assertTrue("Low-dim recall should be >= " + MIN_RECALL_X32 + " but was " + recallLowDim, recallLowDim >= MIN_RECALL_X32);
    }

    @SneakyThrows
    public void testX32_lucene_radialBlocked() {
        String indexName = "lucene_x32_radial_blocked";
        createLuceneX32Index(indexName, SpaceType.L2);
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
    public void testX32_lucene_inMemory() {
        String indexName = "lucene_x32_in_memory";
        createLuceneX32InMemoryIndex(indexName, SpaceType.L2);
        bulkAddKnnDocs(indexName, FIELD_NAME, INDEX_VECTORS, DOC_COUNT);
        forceMergeKnnIndex(indexName, 1);

        List<List<String>> searchResults = bulkSearch(indexName, FIELD_NAME, QUERY_VECTORS, K);
        double recall = TestUtils.calculateRecallValue(searchResults, GROUND_TRUTH_L2, K);
        logger.info("Lucene x32 in-memory recall: {}", recall);
        assertTrue("Lucene x32 in-memory recall should be >= " + MIN_RECALL_X32 + " but was " + recall, recall >= MIN_RECALL_X32);
    }

    @SneakyThrows
    public void testX32_lucene_scriptScoring() {
        String x32IndexName = "lucene_x32_script";
        String fp32IndexName = "lucene_fp32_script";
        int docCount = 20;
        int dimension = 16;

        float[][] vectors = TestUtils.getIndexVectors(docCount, dimension, true);
        float[] queryVector = TestUtils.getQueryVectors(1, dimension, docCount, true)[0];

        createLuceneIndexForScriptScoring(x32IndexName, SpaceType.L2, dimension, true);
        createLuceneIndexForScriptScoring(fp32IndexName, SpaceType.L2, dimension, false);

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
    public void testX32_lucene_rescoreVariations() {
        String indexName = "lucene_x32_rescore_variations";
        createLuceneX32Index(indexName, SpaceType.L2);
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
        logger.info("Lucene x32 rescore=false recall: {}", recallNoRescore);

        List<List<KNNResult>> defaultRescoreResults = searchWithRescore(
            indexName,
            BASELINE_QUERY_VECTORS,
            K,
            RescoreContext.builder().oversampleFactor(3.0f).build()
        );
        double recallDefaultRescore = calculateRecallFromResults(defaultRescoreResults, groundTruth, K);
        logger.info("Lucene x32 rescore=true (3.0x) recall: {}", recallDefaultRescore);
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
        logger.info("Lucene x32 rescore oversample=5.0 recall: {}", recallHighOversample);
        assertTrue("High oversample recall should be >= 0.75 but was " + recallHighOversample, recallHighOversample >= 0.75);

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
    private void createLuceneX32InMemoryIndex(String indexName, SpaceType spaceType) {
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
            .field(KNN_ENGINE, LUCENE_NAME)
            .field(METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_M, HNSW_M)
            .field(METHOD_PARAMETER_EF_CONSTRUCTION, HNSW_EF_CONSTRUCTION)
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
    private void createLuceneX32IndexWithDimension(String indexName, SpaceType spaceType, int dimension) {
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
            .field("dimension", dimension)
            .field(MODE_PARAMETER, "on_disk")
            .field(COMPRESSION_LEVEL_PARAMETER, "32x")
            .startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(KNN_ENGINE, LUCENE_NAME)
            .field(METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_M, HNSW_M)
            .field(METHOD_PARAMETER_EF_CONSTRUCTION, HNSW_EF_CONSTRUCTION)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        createKnnIndex(indexName, settings, builder.toString());
    }

    @SneakyThrows
    private void createLuceneX32IndexWithFilterField(String indexName, SpaceType spaceType) {
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
            .field(KNN_ENGINE, LUCENE_NAME)
            .field(METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_M, HNSW_M)
            .field(METHOD_PARAMETER_EF_CONSTRUCTION, HNSW_EF_CONSTRUCTION)
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
    private void createLuceneX32Index(String indexName, SpaceType spaceType) {
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
            .field(KNN_ENGINE, LUCENE_NAME)
            .field(METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_M, HNSW_M)
            .field(METHOD_PARAMETER_EF_CONSTRUCTION, HNSW_EF_CONSTRUCTION)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        createKnnIndex(indexName, settings, builder.toString());
    }

    @SneakyThrows
    private void createLuceneIndexForScriptScoring(String indexName, SpaceType spaceType, int dimension, boolean useX32) {
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
            .field(KNN_ENGINE, LUCENE_NAME)
            .field(METHOD_PARAMETER_SPACE_TYPE, spaceType.getValue())
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_M, HNSW_M)
            .field(METHOD_PARAMETER_EF_CONSTRUCTION, HNSW_EF_CONSTRUCTION)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        createKnnIndex(indexName, settings, builder.toString());
    }
}

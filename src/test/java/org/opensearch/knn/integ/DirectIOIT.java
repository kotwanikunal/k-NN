/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ;

import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.client.ResponseException;
import org.opensearch.common.settings.Settings;
import org.opensearch.knn.KNNRestTestCase;

import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.faiss.QFrameBitEncoder;

import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.StringJoiner;

import static org.opensearch.knn.common.KNNConstants.FAISS_NAME;
import static org.opensearch.knn.common.KNNConstants.LUCENE_NAME;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;

/**
 * The integration tests of {@code index.knn.direct_io.rescore.enabled}, one class for every scenario of the design's
 * coverage table. Each case asserts results: the same document ids in the same order, scores within
 * {@link #SCORE_ULP_TOLERANCE} float32 units in the last place, a reconstructed {@code _source} equal to the bit, and
 * the setting's own behaviour.
 *
 * <p>These tests do not assert which input a file was read through. A node that cannot open an {@code O_DIRECT}
 * descriptor serves the index from the page cache and returns exactly these answers, so there is nothing here to skip
 * on. That the reads really do go through {@code O_DIRECT} is proven by mutation M5, which corrupts every byte the
 * {@code O_DIRECT} buffer serves and must make every case below that reads vector data fail, and routing itself is
 * proven by {@code DirectIORoutingTests} and the unit tests under {@code index.store}.
 */
public class DirectIOIT extends KNNRestTestCase {

    /** The setting under test, read from the production constant so there is one copy of the key. */
    private static final String DIRECT_IO_SETTING = KNNSettings.INDEX_KNN_DIRECT_IO_RESCORE_ENABLED;

    /**
     * A dimension and a document count whose product is over the 10 MiB floor: 1,024 by 3,000 floats is 12,288,000
     * bytes of vector data.
     */
    private static final int ROUTABLE_DIMENSION = 1024;

    /** How many documents the routable fixture indexes. */
    private static final int ROUTABLE_DOC_COUNT = 3000;

    /** 1,024 half floats is 2,048 bytes, so 6,000 documents is 12,288,000 bytes of vector data. */
    private static final int HALF_FLOAT_DOC_COUNT = 6000;

    private static final int K = 10;
    private static final String SECOND_FIELD = "second_field";
    private static final String NESTED_PATH = "test_nested";
    private static final String NESTED_VECTOR_FIELD = "test_vector";

    /**
     * How many float32 units in the last place two scores of the same query may differ by when one side read the
     * vector data with {@code O_DIRECT}. The ids and their order never change, but Lucene binds its memory-segment
     * scorer only to an input it can take a memory segment from, so vector data read with {@code O_DIRECT} can be
     * scored by another float32 kernel with a different accumulation order. A few units in the last place is that, and
     * anything larger is a wrong read.
     */
    private static final int SCORE_ULP_TOLERANCE = 32;

    // S1 and S2: an index created on, and the default of off, answer the same.

    public void testAnIndexCreatedOnAnswersAsAnIndexThatIsOff() throws Exception {
        final float[][] vectors = createRoutableFaissIndex("direct-io-on", true);
        final float[][] offVectors = createRoutableFaissIndex("direct-io-off", false);
        assertArrayEquals("the two fixtures must hold the same vectors", vectors[0], offVectors[0], 0.0f);
        final String onBody = knnSearch("direct-io-on", FIELD_NAME, vectors[0], K);
        final String offBody = knnSearch("direct-io-off", FIELD_NAME, vectors[0], K);
        assertEquals(K, parseIds(onBody).size());
        assertEquals("the ids and their order must not change", parseIds(offBody), parseIds(onBody));
        assertScoresEqualWithinUlps(parseScores(offBody), parseScores(onBody), "an index created on");

        // The approximate search above scores from the faiss index rather than from the vector data file, so it
        // cannot see a wrong read of that file (FINDINGS F-10). These two read it: an exact search forced by a filter,
        // which scores the candidates from the file, and the derived source reconstruction of one document.
        final String onExact = exactSearch("direct-io-on", vectors[0], 0);
        final String offExact = exactSearch("direct-io-off", offVectors[0], 0);
        assertEquals("the exact search ids must not change", parseIds(offExact), parseIds(onExact));
        assertScoresEqualWithinUlps(parseScores(offExact), parseScores(onExact), "an exact search on an index created on");
        assertEquals(
            "the reconstructed document must be the same",
            vectorFieldOf(getDocument("direct-io-off", "0", "")),
            vectorFieldOf(getDocument("direct-io-on", "0", ""))
        );
    }

    // S3: off to on in place, with no close and open.

    public void testTheFlipToOnKeepsTheSameAnswers() throws Exception {
        final String index = "direct-io-flip-on";
        final float[][] vectors = createRoutableFaissIndex(index, false);
        final String offBody = knnSearch(index, FIELD_NAME, vectors[0], K);
        setDirectIO(index, true);
        final String onBody = knnSearch(index, FIELD_NAME, vectors[0], K);
        assertEquals("the ids and their order must not change", parseIds(offBody), parseIds(onBody));
        assertScoresEqualWithinUlps(parseScores(offBody), parseScores(onBody), "the flip to on");
    }

    // S4: on to off in place.

    public void testResultsEqualAfterRollback() throws Exception {
        final String index = "direct-io-rollback";
        final float[][] vectors = createRoutableFaissIndex(index, true);
        final String onBody = knnSearch(index, FIELD_NAME, vectors[7], K);
        setDirectIO(index, false);
        final String offBody = knnSearch(index, FIELD_NAME, vectors[7], K);
        assertEquals("the ids and their order must not change", parseIds(onBody), parseIds(offBody));
        assertScoresEqualWithinUlps(parseScores(onBody), parseScores(offBody), "the rollback");
    }

    // S5: two indices on one node, one on and one off.

    public void testTwoIndicesOneOnOneOffAnswerEqual() throws Exception {
        final float[][] onVectors = createRoutableFaissIndex("direct-io-pair-on", true);
        final float[][] offVectors = createRoutableFaissIndex("direct-io-pair-off", false);
        final String onBody = knnSearch("direct-io-pair-on", FIELD_NAME, onVectors[2], K);
        final String offBody = knnSearch("direct-io-pair-off", FIELD_NAME, offVectors[2], K);
        assertEquals(parseIds(offBody), parseIds(onBody));
        assertScoresEqualWithinUlps(parseScores(offBody), parseScores(onBody), "two indices, one on and one off");
        // The wildcard rollback of the design, which must leave the answers alone.
        setDirectIO("*", false);
        final String afterBody = knnSearch("direct-io-pair-on", FIELD_NAME, onVectors[2], K);
        assertEquals(parseIds(onBody), parseIds(afterBody));
        assertScoresEqualWithinUlps(parseScores(onBody), parseScores(afterBody), "after the wildcard rollback");
    }

    // S6: an index whose files can never be routed answers as on main.

    public void testADeclinedIndexAnswersAsOnMain() throws Exception {
        // A compound segment: no force merge, so every segment is a flushed one and compound at the shipped defaults.
        final float[][] compound = createFaissIndex("direct-io-compound", ROUTABLE_DIMENSION, 200, true, false);
        final String compoundBody = knnSearch("direct-io-compound", FIELD_NAME, compound[0], K);
        assertFalse("a compound index must still answer", parseIds(compoundBody).isEmpty());

        // A vector data file under the 10 MiB floor, in a segment that is not compound: 16 by 500 floats is 32,000
        // bytes, far under 10,485,760.
        final float[][] small = createSmallNonCompoundFaissIndex("direct-io-small", 16, 500);
        final String smallBody = knnSearch("direct-io-small", FIELD_NAME, small[0], K);
        assertFalse(parseIds(smallBody).isEmpty());

        // A Lucene engine field, which is never offered to the delegate at all.
        final Settings settings = Settings.builder()
            .put("index.knn", true)
            .put("number_of_shards", 1)
            .put("number_of_replicas", 0)
            .put(DIRECT_IO_SETTING, true)
            .build();
        createKnnIndex(
            "direct-io-lucene",
            settings,
            createKnnIndexMapping(FIELD_NAME, ROUTABLE_DIMENSION, METHOD_HNSW, LUCENE_NAME, SpaceType.L2.getValue(), true)
        );
        final float[][] vectors = randomVectors(ROUTABLE_DOC_COUNT, ROUTABLE_DIMENSION);
        bulkAddKnnDocsInChunks("direct-io-lucene", FIELD_NAME, vectors);
        refreshAllIndices();
        forceMergeKnnIndex("direct-io-lucene", 1);
        final String luceneOn = knnSearch("direct-io-lucene", FIELD_NAME, vectors[0], K);
        setDirectIO("direct-io-lucene", false);
        final String luceneOff = knnSearch("direct-io-lucene", FIELD_NAME, vectors[0], K);
        assertEquals("a Lucene engine field answers the same whatever the setting says", parseIds(luceneOff), parseIds(luceneOn));
        assertEquals(parseScores(luceneOff), parseScores(luceneOn));
    }

    // S19: one faiss and one Lucene field in one index, and two faiss fields.

    public void testAMixedFieldIndexAnswersEqual() throws Exception {
        final String index = "direct-io-mixed";
        final Settings settings = Settings.builder()
            .put("index.knn", true)
            .put("number_of_shards", 1)
            .put("number_of_replicas", 0)
            .put(DIRECT_IO_SETTING, true)
            .build();
        final String mapping = "{\"properties\":{\""
            + FIELD_NAME
            + "\":{\"type\":\"knn_vector\",\"dimension\":"
            + ROUTABLE_DIMENSION
            + ",\"method\":{\"name\":\"hnsw\",\"space_type\":\"l2\",\"engine\":\""
            + FAISS_NAME
            + "\"}},\""
            + SECOND_FIELD
            + "\":{\"type\":\"knn_vector\",\"dimension\":"
            + ROUTABLE_DIMENSION
            + ",\"method\":{\"name\":\"hnsw\",\"space_type\":\"l2\",\"engine\":\""
            + LUCENE_NAME
            + "\"}}}}";
        createKnnIndex(index, settings, mapping);
        final float[][] vectors = randomVectors(ROUTABLE_DOC_COUNT, ROUTABLE_DIMENSION);
        bulkAddTwoFieldDocs(index, FIELD_NAME, SECOND_FIELD, vectors);
        refreshAllIndices();
        forceMergeKnnIndex(index, 1);

        final String faissOn = knnSearch(index, FIELD_NAME, vectors[0], K);
        final String luceneOn = knnSearch(index, SECOND_FIELD, vectors[0], K, false);
        setDirectIO(index, false);
        final String faissOff = knnSearch(index, FIELD_NAME, vectors[0], K);
        final String luceneOff = knnSearch(index, SECOND_FIELD, vectors[0], K, false);
        assertEquals("the faiss field answers the same", parseIds(faissOff), parseIds(faissOn));
        assertScoresEqualWithinUlps(parseScores(faissOff), parseScores(faissOn), "the faiss field of a mixed index");
        assertEquals("the Lucene field answers the same", parseIds(luceneOff), parseIds(luceneOn));
        assertEquals(parseScores(luceneOff), parseScores(luceneOn));
    }

    public void testTwoFaissFieldsAnswerEqual() throws Exception {
        final String index = "direct-io-two-faiss";
        final Settings settings = Settings.builder()
            .put("index.knn", true)
            .put("number_of_shards", 1)
            .put("number_of_replicas", 0)
            .put(DIRECT_IO_SETTING, true)
            .build();
        createKnnIndex(
            index,
            settings,
            createKnnIndexMapping(List.of(FIELD_NAME, SECOND_FIELD), List.of(ROUTABLE_DIMENSION, ROUTABLE_DIMENSION))
        );
        final float[][] vectors = randomVectors(ROUTABLE_DOC_COUNT, ROUTABLE_DIMENSION);
        bulkAddTwoFieldDocs(index, FIELD_NAME, SECOND_FIELD, vectors);
        refreshAllIndices();
        forceMergeKnnIndex(index, 1);

        final String firstOn = knnSearch(index, FIELD_NAME, vectors[0], K);
        final String secondOn = knnSearch(index, SECOND_FIELD, vectors[0], K);
        setDirectIO(index, false);
        final String firstOff = knnSearch(index, FIELD_NAME, vectors[0], K);
        final String secondOff = knnSearch(index, SECOND_FIELD, vectors[0], K);
        assertEquals(parseIds(firstOff), parseIds(firstOn));
        assertEquals(parseIds(secondOff), parseIds(secondOn));
        assertScoresEqualWithinUlps(parseScores(firstOff), parseScores(firstOn), "the first faiss field");
        assertScoresEqualWithinUlps(parseScores(secondOff), parseScores(secondOn), "the second faiss field");
    }

    // S17: the setting itself.

    public void testAnInvalidValueIsRejected() throws Exception {
        final String index = "direct-io-invalid";
        createRoutableFaissIndex(index, false);
        final Request request = new Request("PUT", "/" + index + "/_settings");
        request.setJsonEntity(String.format(Locale.ROOT, "{\"%s\": \"not-a-boolean\"}", DIRECT_IO_SETTING));
        final ResponseException exception = expectThrows(ResponseException.class, () -> client().performRequest(request));
        assertEquals(400, exception.getResponse().getStatusLine().getStatusCode());
    }

    public void testSetAtCreation() throws Exception {
        final String index = "direct-io-created-on";
        createRoutableFaissIndex(index, true);
        assertEquals("true", getIndexSettingByName(index, DIRECT_IO_SETTING));
    }

    public void testSetThroughATemplate() throws Exception {
        final Request template = new Request("PUT", "/_index_template/direct-io-template");
        template.setJsonEntity(
            String.format(
                Locale.ROOT,
                "{\"index_patterns\": [\"direct-io-templated*\"], \"template\": {\"settings\": {\"index.knn\": true, \"%s\": true}}}",
                DIRECT_IO_SETTING
            )
        );
        client().performRequest(template);
        try {
            final Request create = new Request("PUT", "/direct-io-templated-1");
            create.setJsonEntity("{}");
            client().performRequest(create);
            assertEquals("true", getIndexSettingByName("direct-io-templated-1", DIRECT_IO_SETTING));
        } finally {
            client().performRequest(new Request("DELETE", "/_index_template/direct-io-template"));
        }
    }

    // S18: the warmup API on an index that is on.

    public void testWarmupDoesNotErrorOnAnIndexThatIsOn() throws Exception {
        final String index = "direct-io-warmup";
        createRoutableFaissIndex(index, true);
        final Response response = knnWarmup(List.of(index));
        assertEquals(200, response.getStatusLine().getStatusCode());
    }

    // S8: rescore over quantized fields, which is the reason the setting carries rescore in its name.

    public void testOnDisk32xRescoreAnswersEqual() throws Exception {
        assertRescoreEqual("direct-io-on-disk-32x", "32x", false);
    }

    public void testOnDisk16xRescoreAnswersEqual() throws Exception {
        assertRescoreEqual("direct-io-on-disk-16x", "16x", false);
    }

    public void testOnDisk8xRescoreAnswersEqual() throws Exception {
        assertRescoreEqual("direct-io-on-disk-8x", "8x", false);
    }

    public void testAdcResultsEqual() throws Exception {
        assertRescoreEqual("direct-io-on-disk-adc", "32x", true);
    }

    // S9: the query paths that read the vector data for a reason other than a plain approximate search.

    public void testFilteredExactSearch() throws Exception {
        final float[][] vectors = createRoutableFaissIndex("direct-io-query-paths", true);
        final String query = "{\"size\":"
            + K
            + ",\"query\":{\"knn\":{\""
            + FIELD_NAME
            + "\":{\"vector\":"
            + vectorAsJson(vectors[0])
            + ",\"k\":"
            + K
            + ",\"filter\":{\"terms\":{\"_id\":[\"0\",\"1\",\"2\",\"3\",\"4\",\"5\",\"6\",\"7\",\"8\",\"9\",\"10\"]}}}}}}";
        assertQueryEqual("direct-io-query-paths", query, "filtered exact search");
    }

    public void testRadialByMaxDistance() throws Exception {
        final float[][] vectors = createRoutableFaissIndex("direct-io-query-paths", true);
        final String query = "{\"size\":"
            + K
            + ",\"query\":{\"knn\":{\""
            + FIELD_NAME
            + "\":{\"vector\":"
            + vectorAsJson(vectors[0])
            + ",\"max_distance\":175.0}}}}";
        assertQueryEqual("direct-io-query-paths", query, "radial by max_distance");
    }

    public void testRadialByMinScore() throws Exception {
        final float[][] vectors = createRoutableFaissIndex("direct-io-query-paths", true);
        final String query = "{\"size\":"
            + K
            + ",\"query\":{\"knn\":{\""
            + FIELD_NAME
            + "\":{\"vector\":"
            + vectorAsJson(vectors[0])
            + ",\"min_score\":0.0057}}}}";
        assertQueryEqual("direct-io-query-paths", query, "radial by min_score");
    }

    public void testRadialWithFilter() throws Exception {
        final float[][] vectors = createRoutableFaissIndex("direct-io-query-paths", true);
        final String query = "{\"size\":"
            + K
            + ",\"query\":{\"knn\":{\""
            + FIELD_NAME
            + "\":{\"vector\":"
            + vectorAsJson(vectors[0])
            + ",\"max_distance\":175.0,\"filter\":{\"terms\":{\"_id\":[\"0\",\"1\",\"2\",\"3\",\"4\"]}}}}}}";
        assertQueryEqual("direct-io-query-paths", query, "radial with a filter");
    }

    public void testKnnScoreScript() throws Exception {
        final float[][] vectors = createRoutableFaissIndex("direct-io-query-paths", true);
        final String query = "{\"size\":"
            + K
            + ",\"query\":{\"script_score\":{\"query\":{\"match_all\":{}},\"script\":{\"source\":\"knn_score\",\"lang\":\"knn\","
            + "\"params\":{\"field\":\""
            + FIELD_NAME
            + "\",\"query_value\":"
            + vectorAsJson(vectors[0])
            + ",\"space_type\":\"l2\"}}}}}";
        assertQueryEqual("direct-io-query-paths", query, "the knn_score script");
    }

    // S10: half_float, and memory-optimized search on fp32 and on FP16.

    public void testHalfFloatAnswersEqual() throws Exception {
        final String index = "direct-io-half-float";
        final float[][] vectors = createTypedIndex(index, VectorDataType.HALF_FLOAT, HALF_FLOAT_DOC_COUNT, false);
        final String onBody = knnSearch(index, FIELD_NAME, vectors[0], K);
        setDirectIO(index, false);
        final String offBody = knnSearch(index, FIELD_NAME, vectors[0], K);
        assertEquals("the ids and their order must not change", parseIds(offBody), parseIds(onBody));
        assertScoresEqualWithinUlps(parseScores(offBody), parseScores(onBody), "half_float");
    }

    public void testMemoryOptimizedFp32() throws Exception {
        final String index = "direct-io-mos-fp32";
        final float[][] vectors = createTypedIndex(index, VectorDataType.FLOAT, ROUTABLE_DOC_COUNT, true);
        assertOnAndOffEqual(index, vectors[1], "memory optimized search on fp32");
    }

    public void testMemoryOptimizedHalfFloat() throws Exception {
        final String index = "direct-io-mos-fp16";
        final float[][] vectors = createTypedIndex(index, VectorDataType.HALF_FLOAT, HALF_FLOAT_DOC_COUNT, true);
        assertOnAndOffEqual(index, vectors[1], "memory optimized search on FP16");
    }

    // S11: nested fields with expand_nested_docs and inner hits.

    public void testNestedResultsEqual() throws Exception {
        final float[][] vectors = createNestedIndex("direct-io-nested");
        final String onBody = nestedSearch("direct-io-nested", vectors[0]);
        setDirectIO("direct-io-nested", false);
        final String offBody = nestedSearch("direct-io-nested", vectors[0]);
        assertEquals("the ids and their order must not change", parseIds(offBody), parseIds(onBody));
        assertScoresEqualWithinUlps(parseScores(offBody), parseScores(onBody), "nested search");
    }

    public void testInnerHitsEqual() throws Exception {
        final float[][] vectors = createNestedIndex("direct-io-nested");
        final String onBody = nestedSearch("direct-io-nested", vectors[1]);
        setDirectIO("direct-io-nested", false);
        final String offBody = nestedSearch("direct-io-nested", vectors[1]);
        // Both assertions are needed. parseInnerHits returns a Multimap keyed by parent id, whose equality ignores
        // the order of the parents, and the order of the parents is what M4 moves: with only the Multimap compared
        // this case stopped failing under M4 (output/REVIEW.md, finding 27).
        assertEquals("the parent ids and their order must not change", parseIds(offBody), parseIds(onBody));
        assertEquals("the inner hits must be the same", parseInnerHits(offBody, NESTED_PATH), parseInnerHits(onBody, NESTED_PATH));
        assertFalse("the query must return inner hits, or it asserts nothing", parseInnerHits(onBody, NESTED_PATH).isEmpty());
    }

    // S12: derived source, where fetching a document reconstructs its vectors from the vector data file.

    public void testSourceEqualsBitForBit() throws Exception {
        final String index = "direct-io-derived-source";
        final float[][] vectors = createRoutableFaissIndex(index, true);
        final String onSource = getDocument(index, "7", "");
        setDirectIO(index, false);
        final String offSource = getDocument(index, "7", "");
        assertEquals("the reconstructed document must be the same text", offSource, onSource);
        assertEquals("and must hold the vector that was indexed", vectorAsJson(vectors[7]), vectorFieldOf(onSource));
    }

    public void testSourceIncludesEqual() throws Exception {
        final String index = "direct-io-derived-source";
        createRoutableFaissIndex(index, true);
        assertFetchEqual(index, "?_source_includes=" + FIELD_NAME, "11");
    }

    public void testSourceExcludesEqual() throws Exception {
        final String index = "direct-io-derived-source";
        createRoutableFaissIndex(index, true);
        // Excluding the vector field itself would leave nothing in the response that comes from the vector data
        // file, so the comparison could not see a wrong read. Excluding another field keeps the reconstruction in.
        final String body = assertFetchEqual(index, "?_source_excludes=not_a_field", "12");
        assertTrue("the excluded-field response must still carry the reconstructed vector", body.contains(FIELD_NAME));
    }

    public void testDocValueFieldsEqual() throws Exception {
        final String index = "direct-io-derived-source";
        final float[][] vectors = createRoutableFaissIndex(index, true);
        final String query = "{\"size\":3,\"docvalue_fields\":[\""
            + FIELD_NAME
            + "\"],\"query\":{\"knn\":{\""
            + FIELD_NAME
            + "\":{\"vector\":"
            + vectorAsJson(vectors[0])
            + ",\"k\":3}}}}";
        assertSearchBodyEqual(index, query, "docvalue_fields");
    }

    public void testFieldsEqual() throws Exception {
        final String index = "direct-io-derived-source";
        final float[][] vectors = createRoutableFaissIndex(index, true);
        final String query = "{\"size\":3,\"fields\":[\""
            + FIELD_NAME
            + "\"],\"query\":{\"knn\":{\""
            + FIELD_NAME
            + "\":{\"vector\":"
            + vectorAsJson(vectors[0])
            + ",\"k\":3}}}}";
        assertSearchBodyEqual(index, query, "fields");
    }

    // S13: a force merge while on, and the merge's own reads of the vector data.

    public void testForceMergeWhileOnAnswersEqual() throws Exception {
        final String index = "direct-io-merge";
        final float[][] vectors = createFaissIndex(index, ROUTABLE_DIMENSION, ROUTABLE_DOC_COUNT, true, true);
        final String beforeBody = knnSearch(index, FIELD_NAME, vectors[3], K);

        // A second batch, then a merge of the two segments while the index is on.
        final float[][] more = randomVectors(ROUTABLE_DOC_COUNT, ROUTABLE_DIMENSION);
        bulkAddKnnDocsInChunks(index, FIELD_NAME, more);
        refreshAllIndices();
        forceMergeKnnIndex(index, 1);

        final String afterBody = knnSearch(index, FIELD_NAME, vectors[3], K);
        assertFalse(parseIds(afterBody).isEmpty());
        assertEquals("the top hit is still the queried document", "3", parseIds(afterBody).get(0));
        assertEquals("the ids the first batch answered are still answered", parseIds(beforeBody).get(0), parseIds(afterBody).get(0));

        setDirectIO(index, false);
        final String offBody = knnSearch(index, FIELD_NAME, vectors[3], K);
        assertEquals("the merged segment answers the same with the index off", parseIds(afterBody), parseIds(offBody));
        assertScoresEqualWithinUlps(parseScores(afterBody), parseScores(offBody), "after the merge");
    }

    public void testNoCorruptionAfterMerge() throws Exception {
        final String index = "direct-io-merge-integrity";
        final float[][] vectors = createFaissIndex(index, ROUTABLE_DIMENSION, ROUTABLE_DOC_COUNT, true, true);
        forceMergeKnnIndex(index, 1);
        flushIndex(index);
        final Response stats = client().performRequest(new Request("GET", "/" + index + "/_stats"));
        assertEquals(200, stats.getStatusLine().getStatusCode());
        assertEquals(ROUTABLE_DOC_COUNT, getDocCount(index));
        // A status code and a document count would pass over a vector data file whose every interior byte is wrong,
        // so the two reads that come from the file itself are compared too.
        final String onExact = exactSearch(index, vectors[11], 11);
        final String onSource = getDocument(index, "11", "");
        setDirectIO(index, false);
        assertEquals(
            "the exact search ids must not change after the merge",
            parseIds(exactSearch(index, vectors[11], 11)),
            parseIds(onExact)
        );
        assertEquals("the reconstructed vector must not change after the merge", getDocument(index, "11", ""), onSource);
    }

    // S14: delete while on.

    public void testDeleteWhileOnDoesNotError() throws Exception {
        final String index = "direct-io-lifecycle";
        final float[][] vectors = createRoutableFaissIndex(index, true);
        knnSearch(index, FIELD_NAME, vectors[0], K);
        deleteKNNIndex(index);
        final Response health = client().performRequest(new Request("GET", "/_cluster/health"));
        assertEquals(200, health.getStatusLine().getStatusCode());
    }

    // S15: snapshot and restore of an index that is on.

    public void testARestoredIndexAnswersEqual() throws Exception {
        final String index = "direct-io-snapshot";
        final String restored = "direct-io-restored";
        final String repository = "direct-io-repo";
        final String snapshotName = "direct-io-snapshot-1";
        final String pathRepo = System.getProperty("tests.path.repo");
        assertNotNull("the integTest cluster must have path.repo set", pathRepo);
        registerRepository(repository, "fs", true, Settings.builder().put("location", pathRepo).build());

        final float[][] vectors = createRoutableFaissIndex(index, true);
        final String originalBody = knnSearch(index, FIELD_NAME, vectors[0], K);

        final Request snapshot = new Request("PUT", "/_snapshot/" + repository + "/" + snapshotName);
        snapshot.addParameter("wait_for_completion", "true");
        snapshot.setJsonEntity("{\"indices\": \"" + index + "\"}");
        assertEquals(200, client().performRequest(snapshot).getStatusLine().getStatusCode());

        final Request restore = new Request("POST", "/_snapshot/" + repository + "/" + snapshotName + "/_restore");
        restore.addParameter("wait_for_completion", "true");
        restore.setJsonEntity(
            "{\"indices\": \"" + index + "\", \"rename_pattern\": \"" + index + "\", \"rename_replacement\": \"" + restored + "\"}"
        );
        assertEquals(200, client().performRequest(restore).getStatusLine().getStatusCode());
        ensureGreen(restored);

        final String restoredBody = knnSearch(restored, FIELD_NAME, vectors[0], K);
        assertEquals("the restored index answers the same ids", parseIds(originalBody), parseIds(restoredBody));
        assertScoresEqualWithinUlps(parseScores(originalBody), parseScores(restoredBody), "the restored index");
        assertEquals("and it is still on", "true", getIndexSettingByName(restored, DIRECT_IO_SETTING));
    }

    // S16: concurrent segment search, several segments, deleted and updated documents.

    public void testConcurrentSegmentSearchResultsEqual() throws Exception {
        final String index = "direct-io-concurrent";
        final float[][] vectors = createSeveralSegments(index);
        final String onBody = knnSearch(index, FIELD_NAME, vectors[0], K);
        setDirectIO(index, false);
        final String offBody = knnSearch(index, FIELD_NAME, vectors[0], K);
        assertEquals("the ids and their order must not change", parseIds(offBody), parseIds(onBody));
        assertScoresEqualWithinUlps(parseScores(offBody), parseScores(onBody), "concurrent segment search");
    }

    public void testWithDeletedDocuments() throws Exception {
        final String index = "direct-io-concurrent";
        final float[][] vectors = createSeveralSegments(index);
        for (int id = 0; id < 50; id++) {
            deleteKnnDoc(index, Integer.toString(id));
        }
        final String onBody = knnSearch(index, FIELD_NAME, vectors[100], K);
        assertFalse("a deleted document must not come back", parseIds(onBody).contains("0"));
        setDirectIO(index, false);
        final String offBody = knnSearch(index, FIELD_NAME, vectors[100], K);
        assertEquals(parseIds(offBody), parseIds(onBody));
        assertScoresEqualWithinUlps(parseScores(offBody), parseScores(onBody), "with deleted documents");
    }

    public void testWithUpdatedDocuments() throws Exception {
        final String index = "direct-io-concurrent";
        final float[][] vectors = createSeveralSegments(index);
        final float[] replacement = vectors[ROUTABLE_DOC_COUNT + 1];
        final Float[] boxed = new Float[replacement.length];
        for (int i = 0; i < replacement.length; i++) {
            boxed[i] = replacement[i];
        }
        updateKnnDoc(index, "5", FIELD_NAME, boxed);

        final String onBody = knnSearch(index, FIELD_NAME, replacement, K);
        assertTrue("the updated document is a neighbour of its new vector", parseIds(onBody).contains("5"));
        setDirectIO(index, false);
        final String offBody = knnSearch(index, FIELD_NAME, replacement, K);
        assertEquals(parseIds(offBody), parseIds(onBody));
        assertScoresEqualWithinUlps(parseScores(offBody), parseScores(onBody), "with updated documents");
    }

    // S20: reads across a block boundary.

    public void testDimension129() throws Exception {
        // 129 floats is 516 bytes, so vectors straddle every 4 KiB boundary.
        assertResultsEqualOnBothRoutes("direct-io-boundary-129", 129, 22_000, 0);
    }

    public void testDimension1023() throws Exception {
        // 1,023 floats is 4,092 bytes, four bytes short of a block, so each vector moves four bytes further in. The
        // ordinal is in the middle of the file, which leaves the first and the last to the two cases below.
        assertResultsEqualOnBothRoutes("direct-io-boundary-1023", 1023, 3_000, 1_500);
    }

    public void testTheFirstVectorAfterTheHeader() throws Exception {
        // Ordinal 0 starts at the codec index header length, which is not a multiple of four or of a block, so this
        // is the vector whose first float straddles the end of the first aligned window.
        final String index = "direct-io-boundary-first";
        final float[][] vectors = createFaissIndex(index, 1023, 3_000, true, true);
        final String onSource = getDocument(index, "0", "");
        setDirectIO(index, false);
        final String offSource = getDocument(index, "0", "");
        assertEquals("the first vector in the file must be reconstructed the same way", offSource, onSource);
        for (int i = 0; i < 8; i++) {
            assertTrue(
                "the reconstructed vector must hold the indexed value " + vectors[0][i],
                onSource.contains(Float.toString(vectors[0][i]))
            );
        }
    }

    public void testTheLastVectorInTheTailBlock() throws Exception {
        assertResultsEqualOnBothRoutes("direct-io-boundary-last", 1023, 3_000, 2_999);
    }

    // The fixtures and the assertions.

    /**
     * Creates a faiss fp32 index whose vector data file is over the 10 MiB floor, indexes it, and force merges it to
     * one segment, which is not compound because a single merged segment is the whole index against a no-CFS ratio of
     * 0.1. That is the shape a {@code .vec} has to have to be routed at all.
     *
     * @param index           the index name
     * @param directIOEnabled whether to create it with Direct I/O already on
     * @return the vectors that were indexed, document id {@code i} holding row {@code i}
     * @throws Exception if any request fails
     */
    private float[][] createRoutableFaissIndex(final String index, final boolean directIOEnabled) throws Exception {
        return createFaissIndex(index, ROUTABLE_DIMENSION, ROUTABLE_DOC_COUNT, directIOEnabled, true);
    }

    /**
     * Creates a faiss fp32 index of any shape, which is how the decline and the block boundary scenarios get a file
     * that is under the floor, a segment that is compound, or vectors that straddle a block.
     *
     * @param index           the index name
     * @param dimension       the vector dimension
     * @param docCount        how many documents to index
     * @param directIOEnabled whether to create it with Direct I/O already on
     * @param forceMerge      whether to force merge to one segment, which is what makes the segment non-compound
     * @return the vectors that were indexed
     * @throws Exception if any request fails
     */
    private float[][] createFaissIndex(
        final String index,
        final int dimension,
        final int docCount,
        final boolean directIOEnabled,
        final boolean forceMerge
    ) throws Exception {
        final Settings settings = Settings.builder()
            .put("index.knn", true)
            .put("number_of_shards", 1)
            .put("number_of_replicas", 0)
            .put(DIRECT_IO_SETTING, directIOEnabled)
            .build();
        createKnnIndex(
            index,
            settings,
            createKnnIndexMapping(FIELD_NAME, dimension, METHOD_HNSW, FAISS_NAME, SpaceType.L2.getValue(), true)
        );
        final float[][] vectors = randomVectors(docCount, dimension);
        bulkAddKnnDocsInChunks(index, FIELD_NAME, vectors);
        refreshAllIndices();
        if (forceMerge) {
            forceMergeKnnIndex(index, 1);
        }
        return vectors;
    }

    /**
     * Creates a faiss index of one vector data type, optionally with memory-optimized search on.
     *
     * @param index           the index name
     * @param dataType        the vector data type
     * @param docCount        how many documents
     * @param memoryOptimized whether to turn memory-optimized search on
     * @return the vectors that were indexed
     * @throws Exception if any request fails
     */
    private float[][] createTypedIndex(final String index, final VectorDataType dataType, final int docCount, final boolean memoryOptimized)
        throws Exception {
        final Settings.Builder settings = Settings.builder()
            .put("index.knn", true)
            .put("number_of_shards", 1)
            .put("number_of_replicas", 0)
            .put(DIRECT_IO_SETTING, true);
        if (memoryOptimized) {
            settings.put("index.knn.memory_optimized_search", true);
        }
        createKnnIndex(
            index,
            settings.build(),
            createKnnIndexMapping(FIELD_NAME, ROUTABLE_DIMENSION, METHOD_HNSW, FAISS_NAME, SpaceType.L2.getValue(), true, dataType)
        );
        final float[][] vectors = randomVectors(docCount, ROUTABLE_DIMENSION);
        bulkAddKnnDocsInChunks(index, FIELD_NAME, vectors);
        refreshAllIndices();
        forceMergeKnnIndex(index, 1);
        return vectors;
    }

    /**
     * Creates a faiss index whose vector data file is under the 10 MiB floor but whose segment is not compound, which
     * takes two flushes and then a merge: a single flushed segment is compound at the shipped defaults and a force
     * merge of one segment is a no-op, so the floor would never be the reason it declined.
     *
     * @param index     the index name
     * @param dimension the vector dimension, small enough to stay under the floor
     * @param docCount  how many documents, split over two flushes
     * @return the vectors that were indexed
     * @throws Exception if any request fails
     */
    private float[][] createSmallNonCompoundFaissIndex(final String index, final int dimension, final int docCount) throws Exception {
        final Settings settings = Settings.builder()
            .put("index.knn", true)
            .put("number_of_shards", 1)
            .put("number_of_replicas", 0)
            .put(DIRECT_IO_SETTING, true)
            .build();
        createKnnIndex(
            index,
            settings,
            createKnnIndexMapping(FIELD_NAME, dimension, METHOD_HNSW, FAISS_NAME, SpaceType.L2.getValue(), true)
        );
        final float[][] vectors = randomVectors(docCount, dimension);
        final int half = docCount / 2;
        final float[][] first = new float[half][];
        System.arraycopy(vectors, 0, first, 0, half);
        bulkAddKnnDocs(index, FIELD_NAME, first, half);
        flushIndex(index);
        final float[][] second = new float[docCount - half][];
        System.arraycopy(vectors, half, second, 0, docCount - half);
        final Request bulk = new Request("POST", "/_bulk");
        bulk.addParameter("refresh", "true");
        final StringBuilder body = new StringBuilder();
        for (int i = 0; i < second.length; i++) {
            body.append("{ \"index\" : { \"_index\" : \"").append(index).append("\", \"_id\" : \"").append(half + i).append("\" } }\n");
            body.append("{ \"").append(FIELD_NAME).append("\" : ").append(Arrays.toString(second[i])).append(" }\n");
        }
        bulk.setJsonEntity(body.toString());
        assertEquals(200, client().performRequest(bulk).getStatusLine().getStatusCode());
        flushIndex(index);
        refreshAllIndices();
        forceMergeKnnIndex(index, 1);
        return vectors;
    }

    /**
     * Two merged segments, each over the size floor and each therefore not compound, with concurrent segment search
     * on so that both are searched on their own threads.
     *
     * @param index the index name
     * @return the vectors that were indexed, in document id order
     * @throws Exception if any request fails
     */
    private float[][] createSeveralSegments(final String index) throws Exception {
        final Settings settings = Settings.builder()
            .put("index.knn", true)
            .put("number_of_shards", 1)
            .put("number_of_replicas", 0)
            .put("index.search.concurrent_segment_search.mode", "all")
            .put(DIRECT_IO_SETTING, true)
            .build();
        createKnnIndex(
            index,
            settings,
            createKnnIndexMapping(FIELD_NAME, ROUTABLE_DIMENSION, METHOD_HNSW, FAISS_NAME, SpaceType.L2.getValue(), true)
        );
        final float[][] vectors = randomVectors(2 * ROUTABLE_DOC_COUNT, ROUTABLE_DIMENSION);
        final float[][] first = new float[ROUTABLE_DOC_COUNT][];
        System.arraycopy(vectors, 0, first, 0, ROUTABLE_DOC_COUNT);
        bulkAddKnnDocsInChunks(index, FIELD_NAME, first);
        refreshAllIndices();
        forceMergeKnnIndex(index, 1);

        for (int from = ROUTABLE_DOC_COUNT; from < 2 * ROUTABLE_DOC_COUNT; from += 500) {
            final int to = Math.min(from + 500, 2 * ROUTABLE_DOC_COUNT);
            final Request bulk = new Request("POST", "/_bulk");
            bulk.addParameter("refresh", "true");
            final StringBuilder body = new StringBuilder();
            for (int i = from; i < to; i++) {
                body.append("{ \"index\" : { \"_index\" : \"").append(index).append("\", \"_id\" : \"").append(i).append("\" } }\n");
                body.append("{ \"").append(FIELD_NAME).append("\" : ").append(Arrays.toString(vectors[i])).append(" }\n");
            }
            bulk.setJsonEntity(body.toString());
            assertEquals(200, client().performRequest(bulk).getStatusLine().getStatusCode());
        }
        refreshAllIndices();
        forceMergeKnnIndex(index, 2);
        return vectors;
    }

    /**
     * One index of 1,000 documents with three nested vectors each, which is 3,000 vectors of 1,024 floats and
     * 12,288,000 bytes of vector data, force merged to one segment. Every nested field is sparse, so Lucene reads its
     * ord-to-doc table through a random access slice, which stays on the original input.
     *
     * @param index the index name
     * @return the nested vectors in the order they were indexed
     * @throws Exception if any request fails
     */
    private float[][] createNestedIndex(final String index) throws Exception {
        final int parents = 1000;
        final int children = 3;
        final Settings settings = Settings.builder()
            .put("index.knn", true)
            .put("number_of_shards", 1)
            .put("number_of_replicas", 0)
            .put(DIRECT_IO_SETTING, true)
            .build();
        final String mapping = "{\"properties\":{\""
            + NESTED_PATH
            + "\":{\"type\":\"nested\",\"properties\":{\""
            + NESTED_VECTOR_FIELD
            + "\":{\"type\":\"knn_vector\",\"dimension\":"
            + ROUTABLE_DIMENSION
            + ",\"method\":{\"name\":\"hnsw\",\"engine\":\"faiss\",\"space_type\":\"l2\"}}}}}}";
        createKnnIndex(index, settings, mapping);

        final float[][] vectors = randomVectors(parents * children, ROUTABLE_DIMENSION);
        for (int from = 0; from < parents; from += 100) {
            final int to = Math.min(from + 100, parents);
            final Request bulk = new Request("POST", "/_bulk");
            bulk.addParameter("refresh", "true");
            final StringBuilder body = new StringBuilder();
            for (int parent = from; parent < to; parent++) {
                body.append("{ \"index\" : { \"_index\" : \"").append(index).append("\", \"_id\" : \"").append(parent).append("\" } }\n");
                body.append("{ \"").append(NESTED_PATH).append("\" : [");
                for (int child = 0; child < children; child++) {
                    if (child > 0) {
                        body.append(',');
                    }
                    body.append("{ \"")
                        .append(NESTED_VECTOR_FIELD)
                        .append("\" : ")
                        .append(Arrays.toString(vectors[parent * children + child]))
                        .append(" }");
                }
                body.append("] }\n");
            }
            bulk.setJsonEntity(body.toString());
            assertEquals(200, client().performRequest(bulk).getStatusLine().getStatusCode());
        }
        refreshAllIndices();
        forceMergeKnnIndex(index, 1);
        return vectors;
    }

    /**
     * Indexes one vector field in chunks, because a single bulk request carrying thousands of 1,024-float vectors is
     * answered with 429.
     *
     * @param index   the index
     * @param field   the vector field
     * @param vectors the vectors, one row per document
     * @throws Exception if a bulk request fails
     */
    private void bulkAddKnnDocsInChunks(final String index, final String field, final float[][] vectors) throws Exception {
        final int chunk = 500;
        for (int from = 0; from < vectors.length; from += chunk) {
            final int to = Math.min(from + chunk, vectors.length);
            final Request request = new Request("POST", "/_bulk");
            request.addParameter("refresh", "true");
            final StringBuilder body = new StringBuilder();
            for (int i = from; i < to; i++) {
                body.append("{ \"index\" : { \"_index\" : \"").append(index).append("\", \"_id\" : \"").append(i).append("\" } }\n");
                body.append("{ \"").append(field).append("\" : ").append(Arrays.toString(vectors[i])).append(" }\n");
            }
            request.setJsonEntity(body.toString());
            assertEquals(200, client().performRequest(request).getStatusLine().getStatusCode());
        }
    }

    /**
     * Indexes documents that carry the same vector in two fields, in one bulk request, which is what the mixed-field
     * scenarios need and what {@code bulkAddKnnDocs} does not do.
     *
     * @param index   the index
     * @param fieldA  the first vector field
     * @param fieldB  the second vector field
     * @param vectors the vectors, one row per document
     * @throws Exception if the bulk request fails
     */
    private void bulkAddTwoFieldDocs(final String index, final String fieldA, final String fieldB, final float[][] vectors)
        throws Exception {
        // In chunks, because one request carrying two copies of 3,000 vectors of 1,024 floats is tens of megabytes
        // and the node answers 429.
        final int chunk = 250;
        for (int from = 0; from < vectors.length; from += chunk) {
            final int to = Math.min(from + chunk, vectors.length);
            final Request request = new Request("POST", "/_bulk");
            request.addParameter("refresh", "true");
            final StringBuilder body = new StringBuilder();
            for (int i = from; i < to; i++) {
                final String vector = Arrays.toString(vectors[i]);
                body.append("{ \"index\" : { \"_index\" : \"").append(index).append("\", \"_id\" : \"").append(i).append("\" } }\n");
                body.append("{ \"")
                    .append(fieldA)
                    .append("\" : ")
                    .append(vector)
                    .append(", \"")
                    .append(fieldB)
                    .append("\" : ")
                    .append(vector)
                    .append(" }\n");
            }
            request.setJsonEntity(body.toString());
            assertEquals(200, client().performRequest(request).getStatusLine().getStatusCode());
        }
    }

    /**
     * Turns Direct I/O on or off for an index, which is dynamic and reaches segments that are already open.
     *
     * @param index   the index, or a pattern such as {@code *}
     * @param enabled the value to set
     * @throws Exception if the request fails
     */
    private void setDirectIO(final String index, final boolean enabled) throws Exception {
        updateIndexSettings(index, Settings.builder().put(DIRECT_IO_SETTING, enabled));
    }

    private void assertOnAndOffEqual(final String index, final float[] vector, final String what) throws Exception {
        final String onBody = knnSearch(index, FIELD_NAME, vector, K);
        setDirectIO(index, false);
        final String offBody = knnSearch(index, FIELD_NAME, vector, K);
        assertEquals(what + ": the ids must not change", parseIds(offBody), parseIds(onBody));
        assertFalse(what + " returned nothing, so it asserts nothing", parseIds(onBody).isEmpty());
    }

    private void assertQueryEqual(final String index, final String query, final String what) throws Exception {
        final String onBody = EntityUtils.toString(searchKNNIndex(index, query, K).getEntity());
        setDirectIO(index, false);
        final String offBody = EntityUtils.toString(searchKNNIndex(index, query, K).getEntity());
        assertEquals(what + ": the ids and their order must not change", parseIds(offBody), parseIds(onBody));
        final int worstUlps = assertScoresEqualWithinUlps(parseScores(offBody), parseScores(onBody), what);
        logger.info("[direct-io] {}: the worst score difference was {} float32 ulps", what, worstUlps);
        assertFalse(what + " returned nothing, so it asserts nothing", parseIds(onBody).isEmpty());
    }

    private void assertSearchBodyEqual(final String index, final String query, final String what) throws Exception {
        final String onBody = EntityUtils.toString(searchKNNIndex(index, query, 3).getEntity());
        setDirectIO(index, false);
        final String offBody = EntityUtils.toString(searchKNNIndex(index, query, 3).getEntity());
        assertEquals(what + " must return the same ids", parseIds(offBody), parseIds(onBody));
        // The scores are compared with the same tolerance as everywhere else, because a routed vector data file can
        // be scored by another float32 kernel. The rest of the body, which carries the reconstructed vectors, must
        // be equal to the bit.
        assertScoresEqualWithinUlps(parseScores(offBody), parseScores(onBody), what);
        assertEquals(what + " must return the same body", stripScores(stripTook(offBody)), stripScores(stripTook(onBody)));
    }

    private String assertFetchEqual(final String index, final String parameters, final String docId) throws Exception {
        final String onBody = getDocument(index, docId, parameters);
        setDirectIO(index, false);
        final String offBody = getDocument(index, docId, parameters);
        assertEquals("the fetched document must be the same with " + parameters, offBody, onBody);
        return onBody;
    }

    private void assertRescoreEqual(final String index, final String compressionLevel, final boolean adc) throws Exception {
        final Settings settings = Settings.builder()
            .put("index.knn", true)
            .put("number_of_shards", 1)
            .put("number_of_replicas", 0)
            .put(DIRECT_IO_SETTING, true)
            .build();
        final String mapping = adc
            ? adcMapping()
            : "{\"properties\":{\""
                + FIELD_NAME
                + "\":{\"type\":\"knn_vector\",\"dimension\":"
                + ROUTABLE_DIMENSION
                + ",\"space_type\":\"l2\",\"mode\":\"on_disk\",\"compression_level\":\""
                + compressionLevel
                + "\"}}}";
        createKnnIndex(index, settings, mapping);
        final float[][] vectors = randomVectors(ROUTABLE_DOC_COUNT, ROUTABLE_DIMENSION);
        bulkAddKnnDocsInChunks(index, FIELD_NAME, vectors);
        refreshAllIndices();
        forceMergeKnnIndex(index, 1);

        final String onBody = rescoreSearch(index, vectors[0]);
        setDirectIO(index, false);
        final String offBody = rescoreSearch(index, vectors[0]);
        assertFalse(compressionLevel + " returned nothing, so it asserts nothing", parseIds(onBody).isEmpty());
        assertEquals(compressionLevel + ": the ids and their order must not change", parseIds(offBody), parseIds(onBody));
        assertScoresEqualWithinUlps(parseScores(offBody), parseScores(onBody), compressionLevel + " rescore");
    }

    /**
     * The mapping of a 1-bit quantized faiss field with asymmetric distance computation on, which is how
     * {@code ADCIT} builds one: the switch is a parameter of the binary encoder, not of the field.
     *
     * @return the mapping
     */
    private String adcMapping() {
        return "{\"properties\":{\""
            + FIELD_NAME
            + "\":{\"type\":\"knn_vector\",\"dimension\":"
            + ROUTABLE_DIMENSION
            + ",\"method\":{\"space_type\":\"l2\",\"engine\":\"faiss\",\"name\":\"hnsw\",\"parameters\":{\"encoder\":{\"name\":\"binary\","
            + "\"parameters\":{\"bits\":1,\""
            + QFrameBitEncoder.ENABLE_ADC_PARAM
            + "\":true}}}}}}}";
    }

    private String rescoreSearch(final String index, final float[] vector) throws Exception {
        final String query = "{\"size\":"
            + K
            + ",\"_source\":false,\"query\":{\"knn\":{\""
            + FIELD_NAME
            + "\":{\"vector\":"
            + vectorAsJson(vector)
            + ",\"k\":"
            + K
            + ",\"rescore\":{\"oversample_factor\":2.0}}}}}";
        return EntityUtils.toString(searchKNNIndex(index, query, K).getEntity());
    }

    /**
     * Queries with the vector of one document, first with the index on and then with it off, and asserts the ids, the
     * order and the scores are the same on the three read paths a wrong read would show: an approximate search, an
     * exact search forced by a filter, which scores from the vector data file rather than from the native index, and
     * the derived source reconstruction of one document.
     *
     * @param index     the index to create
     * @param dimension the vector dimension
     * @param docCount  how many documents
     * @param queryOrd  which document's vector to query with
     * @throws Exception if any request fails
     */
    private void assertResultsEqualOnBothRoutes(final String index, final int dimension, final int docCount, final int queryOrd)
        throws Exception {
        final float[][] vectors = createFaissIndex(index, dimension, docCount, true, true);
        final String onBody = knnSearch(index, FIELD_NAME, vectors[queryOrd], K);
        final String onExact = exactSearch(index, vectors[queryOrd], queryOrd);
        final String onSource = getDocument(index, Integer.toString(queryOrd), "");

        setDirectIO(index, false);
        final String offBody = knnSearch(index, FIELD_NAME, vectors[queryOrd], K);
        final String offExact = exactSearch(index, vectors[queryOrd], queryOrd);
        final String offSource = getDocument(index, Integer.toString(queryOrd), "");

        assertEquals("the ids must not change at dimension " + dimension, parseIds(offBody), parseIds(onBody));
        assertScoresEqualWithinUlps(parseScores(offBody), parseScores(onBody), "dimension " + dimension);
        assertEquals("the queried document is its own nearest neighbour", Integer.toString(queryOrd), parseIds(onBody).get(0));

        assertEquals("the exact search ids must not change at dimension " + dimension, parseIds(offExact), parseIds(onExact));
        assertScoresEqualWithinUlps(parseScores(offExact), parseScores(onExact), "exact search at dimension " + dimension);
        assertEquals("the reconstructed vector must be the same at dimension " + dimension, offSource, onSource);
    }

    /**
     * An exact search over a handful of documents, forced by a filter, which scores the candidates by reading the
     * vector data file.
     *
     * @param index    the index
     * @param vector   the query vector
     * @param queryOrd the document whose vector it is, which must be among the filtered ids
     * @return the response body
     * @throws Exception if the search fails
     */
    private String exactSearch(final String index, final float[] vector, final int queryOrd) throws Exception {
        final StringBuilder ids = new StringBuilder();
        for (int i = Math.max(0, queryOrd - 3); i <= queryOrd + 3; i++) {
            if (ids.length() > 0) {
                ids.append(',');
            }
            ids.append('"').append(i).append('"');
        }
        final String query = "{\"size\":5,\"_source\":false,\"query\":{\"knn\":{\""
            + FIELD_NAME
            + "\":{\"vector\":"
            + vectorAsJson(vector)
            + ",\"k\":5,\"filter\":{\"terms\":{\"_id\":["
            + ids
            + "]}}}}}}";
        return EntityUtils.toString(searchKNNIndex(index, query, 5).getEntity());
    }

    private String nestedSearch(final String index, final float[] vector) throws Exception {
        final String query = "{\"size\":"
            + K
            + ",\"_source\":false,\"query\":{\"nested\":{\"path\":\""
            + NESTED_PATH
            + "\",\"query\":{\"knn\":{\""
            + NESTED_PATH
            + "."
            + NESTED_VECTOR_FIELD
            + "\":{\"vector\":"
            + vectorAsJson(vector)
            + ",\"k\":"
            + K
            + ",\"expand_nested_docs\":true}}},\"inner_hits\":{}}}}";
        return EntityUtils.toString(performSearch(index, query, "size=" + K).getEntity());
    }

    /**
     * Runs a k-NN query and returns the response body.
     *
     * @param index  the index
     * @param field  the vector field
     * @param vector the query vector
     * @param k      how many neighbours
     * @return the response body
     * @throws Exception if the search fails
     */
    private String knnSearch(final String index, final String field, final float[] vector, final int k) throws Exception {
        return knnSearch(index, field, vector, k, true);
    }

    /**
     * Runs a k-NN query, optionally without {@code _source}. Leaving {@code _source} out matters when a test looks at
     * one field only: with derived source on, which is the default, fetching {@code _source} reconstructs every
     * vector field of the hits from its vector data file, so a query on one field reads the files of the others too.
     *
     * @param index         the index
     * @param field         the vector field
     * @param vector        the query vector
     * @param k             how many neighbours
     * @param includeSource whether to fetch {@code _source}
     * @return the response body
     * @throws Exception if the search fails
     */
    private String knnSearch(final String index, final String field, final float[] vector, final int k, final boolean includeSource)
        throws Exception {
        final String query = "{\"size\":"
            + k
            + (includeSource ? "" : ",\"_source\":false")
            + ",\"query\":{\"knn\":{\""
            + field
            + "\":{\"vector\":"
            + vectorAsJson(vector)
            + ",\"k\":"
            + k
            + "}}}}";
        return EntityUtils.toString(searchKNNIndex(index, query, k).getEntity());
    }

    private String getDocument(final String index, final String docId, final String parameters) throws Exception {
        final Request request = new Request("GET", "/" + index + "/_doc/" + docId + parameters);
        return EntityUtils.toString(client().performRequest(request).getEntity());
    }

    @SuppressWarnings("unchecked")
    private String vectorFieldOf(final String documentBody) throws Exception {
        final Map<String, Object> body = createParser(
            org.opensearch.core.xcontent.MediaTypeRegistry.getDefaultMediaType().xContent(),
            documentBody
        ).map();
        final Map<String, Object> source = (Map<String, Object>) body.get("_source");
        final List<Object> vector = (List<Object>) source.get(FIELD_NAME);
        final StringBuilder json = new StringBuilder("[");
        for (int i = 0; i < vector.size(); i++) {
            if (i > 0) {
                json.append(',');
            }
            json.append(((Number) vector.get(i)).floatValue());
        }
        return json.append(']').toString();
    }

    private String vectorAsJson(final float[] vector) {
        final StringJoiner joiner = new StringJoiner(",", "[", "]");
        for (final float value : vector) {
            joiner.add(Float.toString(value));
        }
        return joiner.toString();
    }

    /** The response body without the {@code took} field, which changes between two runs of the same query. */
    private String stripTook(final String body) {
        return body.replaceAll("\"took\":[0-9]+,", "");
    }

    /**
     * The response body without the scores, which are compared separately with the float32 tolerance of
     * {@link #SCORE_ULP_TOLERANCE}.
     *
     * @param body the response body
     * @return the body with every {@code _score} and {@code max_score} value removed
     */
    private String stripScores(final String body) {
        return body.replaceAll("\"(_score|max_score)\":[0-9eE.+-]+", "\"$1\":0");
    }

    /**
     * Deterministic vectors, so that two arms of a comparison query the same data.
     *
     * @param count     how many vectors
     * @param dimension how many floats each
     * @return the vectors
     */
    private float[][] randomVectors(final int count, final int dimension) {
        final Random random = new Random(1_234L);
        final float[][] vectors = new float[count][dimension];
        for (int i = 0; i < count; i++) {
            for (int j = 0; j < dimension; j++) {
                vectors[i][j] = random.nextFloat();
            }
        }
        return vectors;
    }

    /**
     * Asserts two score lists agree to within {@link #SCORE_ULP_TOLERANCE} float32 units in the last place, and
     * returns the largest distance seen so a test can report it.
     *
     * @param expected the scores of one arm
     * @param actual   the scores of the other
     * @param what     what the query was, for the failure message
     * @return the largest distance in float32 units in the last place
     */
    private int assertScoresEqualWithinUlps(final List<Double> expected, final List<Double> actual, final String what) {
        assertEquals(what + ": the number of hits", expected.size(), actual.size());
        // Two empty lists compare equal, so a query that returned nothing would assert nothing.
        assertFalse(what + " returned no hits, so it asserts nothing", expected.isEmpty());
        int worst = 0;
        for (int i = 0; i < expected.size(); i++) {
            final float one = expected.get(i).floatValue();
            final float other = actual.get(i).floatValue();
            // k-NN scores are positive, so the bit patterns are ordered and their difference is the distance in
            // units in the last place.
            assertTrue(what + ": score " + i + " is not positive", one > 0 && other > 0);
            final int distance = Math.abs(Float.floatToIntBits(one) - Float.floatToIntBits(other));
            assertTrue(
                what + ": score " + i + " moved " + distance + " float32 ulps, " + one + " against " + other,
                distance <= SCORE_ULP_TOLERANCE
            );
            worst = Math.max(worst, distance);
        }
        return worst;
    }
}

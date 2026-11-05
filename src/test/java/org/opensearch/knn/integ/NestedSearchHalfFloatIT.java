/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ;

import lombok.SneakyThrows;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.client.Response;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.KNNJsonIndexMappingsBuilder;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.KNNResult;
import org.opensearch.knn.NestedKnnDocBuilder;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.common.settings.Settings;
import org.opensearch.knn.KNNJsonQueryBuilder;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;

public class NestedSearchHalfFloatIT extends KNNRestTestCase {

    private static final String INDEX_NAME = "half_float_nested_test_index";
    private static final String NESTED_FIELD_NAME = "test_nested";
    private static final String VECTOR_FIELD_NAME = "test_vector";
    private static final String NESTED_VECTOR_PATH = NESTED_FIELD_NAME + "." + VECTOR_FIELD_NAME;
    private static final int DIMENSION = 4;

    @SneakyThrows
    public void testNestedSearch_whenHalfFloatFlat_thenReturnsExpectedResults() {
        createKnnIndex(INDEX_NAME, buildHalfFloatNestedMapping("flat", null));
        indexNestedTestDocs();

        float[] queryVector = { 0.0f, 0.0f, 0.0f, 0.0f };
        Response response = searchKNNIndex(INDEX_NAME, buildNestedSearchQuery(queryVector, 3), 3);
        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), NESTED_VECTOR_PATH);

        assertEquals(3, results.size());
        // Doc 1's vector is nearest to the query vector (all values closest to the origin).
        assertEquals("1", results.get(0).getDocId());
    }

    @SneakyThrows
    public void testNestedSearch_whenHalfFloatHnsw_thenReturnsExpectedResults() {
        createKnnIndex(INDEX_NAME, buildHalfFloatNestedMapping("hnsw", "lucene"));
        indexNestedTestDocs();

        float[] queryVector = { 0.0f, 0.0f, 0.0f, 0.0f };
        Response response = searchKNNIndex(INDEX_NAME, buildNestedSearchQuery(queryVector, 3), 3);
        List<KNNResult> results = parseSearchResponse(EntityUtils.toString(response.getEntity()), NESTED_VECTOR_PATH);

        assertEquals(3, results.size());
        assertEquals("1", results.get(0).getDocId());
    }

    /**
     * A nested half_float field answered by exact search, because the segment is written without a graph: every
     * parent comes back with its own best child's score, checked against the vectors this test indexed after
     * rounding them to fp16, which is how they are stored.
     */
    @SneakyThrows
    public void testNestedSearchHalfFloat_whenExactSearch_thenFp16RoundedScoresMatchReference() {
        for (SpaceType space : List.of(SpaceType.L2, SpaceType.INNER_PRODUCT)) {
            final String indexName = "nested-exact-half-float-" + space.getValue();
            createKnnIndex(
                indexName,
                Settings.builder()
                    .put(getKNNDefaultIndexSettings())
                    .put(KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD, NO_GRAPH_THRESHOLD)
                    .build(),
                buildHalfFloatNestedMapping(METHOD_HNSW, KNNEngine.FAISS.getName(), space.getValue(), DIMENSION_EXACT)
            );

            for (int group = 0; group < GROUPS.length; group++) {
                final NestedKnnDocBuilder builder = NestedKnnDocBuilder.create(NESTED_FIELD_NAME);
                for (int offset = 0; offset < GROUPS[group][0]; offset++) {
                    if (offset == GROUPS[group][2]) {
                        builder.addMetadata(Map.of(TAG_FIELD, "t" + offset));
                        continue;
                    }
                    builder.addVectors(VECTOR_FIELD_NAME, box(childVector(group, offset)));
                }
                addKnnDoc(indexName, "p" + group, builder.build());
            }
            refreshIndex(indexName);

            final int size = GROUPS.length + 5;
            final String query = KNNJsonQueryBuilder.builder()
                .nestedFieldName(NESTED_FIELD_NAME)
                .fieldName(VECTOR_FIELD_NAME)
                .vector(box(queryVector()))
                .k(size)
                .build()
                .getQueryString();
            final String entity = EntityUtils.toString(searchKNNIndex(indexName, query, size).getEntity());

            final List<String> expectedIds = new ArrayList<>();
            for (int group = 0; group < GROUPS.length; group++) {
                if (GROUPS[group][0] > 0) {
                    expectedIds.add("p" + group);
                }
            }
            assertEquals("half_float " + space.getValue() + ": the parents and their order", expectedIds, parseIds(entity));
            final List<Double> scores = parseScores(entity);
            for (int i = 0; i < expectedIds.size(); i++) {
                final int group = Integer.parseInt(expectedIds.get(i).substring(1));
                final float expected = space.getKnnVectorSimilarityFunction().compare(queryVector(), childVector(group, GROUPS[group][1]));
                assertEquals(
                    "half_float " + space.getValue() + ": score of p" + group,
                    expected,
                    scores.get(i),
                    1.0e-5 * Math.max(1.0, expected)
                );
            }

            deleteKNNIndex(indexName);
        }
    }

    // ---- the parent groups the exact search method indexes: child count, the offset that must win, and
    // the offset that carries no vector ----
    private static final int[][] GROUPS = { { 1, 0, -1 }, { 3, 0, -1 }, { 4, 3, -1 }, { 5, 2, 0 }, { 0, -1, -1 }, { 6, 1, 5 } };
    private static final int DIMENSION_EXACT = 8;
    private static final float QUERY_VALUE = 3.0f;
    private static final int NO_GRAPH_THRESHOLD = 100000;
    /** What a child with no vector carries instead, which is how a sparse nested field looks. */
    private static final String TAG_FIELD = "tag";

    private float[] queryVector() {
        final float[] query = new float[DIMENSION_EXACT];
        Arrays.fill(query, roundToFp16(QUERY_VALUE));
        return query;
    }

    /**
     * The group's named best child is the query with one coordinate pulled in by a step that grows with the group,
     * so the winners' scores fall with the group index and every one of them beats any other child, whose
     * coordinates alternate in sign. Every value is rounded to fp16 first, which is how the field stores it.
     */
    private float[] childVector(final int group, final int offset) {
        final float[] vector = queryVector();
        if (offset == GROUPS[group][1]) {
            vector[0] = roundToFp16(QUERY_VALUE * (0.7f - 0.05f * group));
            return vector;
        }
        for (int i = 0; i < DIMENSION_EXACT; i++) {
            final float magnitude = 0.5f + ((group * 31 + offset * 7 + i) % 13) / 12.0f;
            vector[i] = roundToFp16(QUERY_VALUE * magnitude * ((i % 2 == (offset % 2)) ? 1.0f : -1.0f));
        }
        return vector;
    }

    private static float roundToFp16(final float value) {
        return Float.float16ToFloat(Float.floatToFloat16(value));
    }

    private static Object[] box(final float[] vector) {
        final Float[] boxed = new Float[vector.length];
        for (int i = 0; i < vector.length; i++) {
            boxed[i] = vector[i];
        }
        return boxed;
    }

    private void indexNestedTestDocs() throws Exception {
        addKnnDocWithNestedField(INDEX_NAME, "1", NESTED_VECTOR_PATH, new Float[] { 1.0f, 1.0f, 1.0f, 1.0f });
        addKnnDocWithNestedField(INDEX_NAME, "2", NESTED_VECTOR_PATH, new Float[] { 2.0f, 2.0f, 2.0f, 2.0f });
        addKnnDocWithNestedField(INDEX_NAME, "3", NESTED_VECTOR_PATH, new Float[] { 5.0f, 5.0f, 5.0f, 5.0f });
    }

    private String buildHalfFloatNestedMapping(String methodName, String engine) throws Exception {
        return buildHalfFloatNestedMapping(methodName, engine, "l2", DIMENSION);
    }

    private String buildHalfFloatNestedMapping(String methodName, String engine, String spaceType, int dimension) throws Exception {
        KNNJsonIndexMappingsBuilder.Method.MethodBuilder methodBuilder = KNNJsonIndexMappingsBuilder.Method.builder()
            .methodName(methodName)
            .spaceType(spaceType);
        if (engine != null) {
            methodBuilder.engine(engine);
        }
        return KNNJsonIndexMappingsBuilder.builder()
            .fieldName(VECTOR_FIELD_NAME)
            .nestedFieldName(NESTED_FIELD_NAME)
            .dimension(dimension)
            .vectorDataType("half_float")
            .method(methodBuilder.build())
            .build()
            .getIndexMapping();
    }

    private XContentBuilder buildNestedSearchQuery(float[] queryVector, int k) throws Exception {
        return XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("nested")
            .field("path", NESTED_FIELD_NAME)
            .startObject("query")
            .startObject("knn")
            .startObject(NESTED_VECTOR_PATH)
            .field("vector", queryVector)
            .field("k", k)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();
    }
}

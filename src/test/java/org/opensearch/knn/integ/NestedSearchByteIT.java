/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ;

import java.util.Locale;

import lombok.SneakyThrows;
import lombok.extern.log4j.Log4j2;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.client.Response;
import org.opensearch.common.settings.Settings;
import org.opensearch.knn.KNNJsonIndexMappingsBuilder;
import org.opensearch.knn.KNNJsonQueryBuilder;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.NestedKnnDocBuilder;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngine;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;

@Log4j2
public class NestedSearchByteIT extends KNNRestTestCase {
    @SneakyThrows
    public void testNestedSearchWithHnswByte_whenKIsTwo_thenReturnTwoResults() {
        for (KNNEngine engine : List.of(KNNEngine.FAISS, KNNEngine.LUCENE)) {
            String indexName = INDEX_NAME + "_" + engine.getName().toLowerCase(Locale.ROOT);
            String nestedFieldName = "nested";
            createKnnByteIndexWithNestedField(indexName, nestedFieldName, FIELD_NAME, 2, engine);

            int totalDocCount = 15;
            for (byte i = 0; i < totalDocCount; i++) {
                String doc = NestedKnnDocBuilder.create(nestedFieldName)
                    .addVectors(FIELD_NAME, new Byte[] { i, i }, new Byte[] { i, i })
                    .build();
                addKnnDoc(indexName, String.valueOf(i), doc);
            }

            refreshIndex(indexName);
            forceMergeKnnIndex(indexName);

            Byte[] queryVector = { 14, 14 };
            String query = KNNJsonQueryBuilder.builder()
                .nestedFieldName(nestedFieldName)
                .fieldName(FIELD_NAME)
                .vector(queryVector)
                .k(2)
                .build()
                .getQueryString();
            Response response = searchKNNIndex(indexName, query, 2);
            String entity = EntityUtils.toString(response.getEntity());

            assertEquals(2, parseHits(entity));
            assertEquals(2, parseTotalSearchHits(entity));
            assertEquals("14", parseIds(entity).get(0));
            assertNotEquals("14", parseIds(entity).get(1));

            deleteKNNIndex(indexName);
        }
    }

    /**
     * {
     * 	"query": {
     * 		"nested": {
     * 			"path": "test_nested",
     * 			"query": {
     * 				"knn": {
     * 					"test_nested.test_vector": {
     * 						"vector": [
     * 							1, 1, 1
     * 						],
     * 						"k": 3,
     * 						"filter": {
     * 							"term": {
     * 								"parking": "true"
     *                          }
     *                      }
     * 					}
     * 				}
     * 			}
     * 		}
     * 	 }
     * }
     *
     */
    @SneakyThrows
    public void testNestedSearchWithFaissHnswByte_whenDoingExactSearch_thenReturnCorrectResults() {
        String nestedFieldName = "nested";
        String filterFieldName = "parking";
        createKnnByteIndexWithNestedField(INDEX_NAME, nestedFieldName, FIELD_NAME, 3, KNNEngine.FAISS);

        for (byte i = 1; i < 4; i++) {
            String doc = NestedKnnDocBuilder.create(nestedFieldName)
                .addVectors(FIELD_NAME, new Byte[] { i, i, i }, new Byte[] { i, i, i }, new Byte[] { i, i, i })
                .addTopLevelField(filterFieldName, i % 2 == 1 ? "true" : "false")
                .build();
            addKnnDoc(INDEX_NAME, String.valueOf(i), doc);
        }
        refreshIndex(INDEX_NAME);
        forceMergeKnnIndex(INDEX_NAME);

        // Make it as an exact search by setting the threshold larger than size of filteredIds(6)
        updateIndexSettings(INDEX_NAME, Settings.builder().put(KNNSettings.ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD, 100));

        Byte[] queryVector = { 3, 3, 3 };
        String query = KNNJsonQueryBuilder.builder()
            .nestedFieldName(nestedFieldName)
            .fieldName(FIELD_NAME)
            .vector(queryVector)
            .k(3)
            .filterFieldName(filterFieldName)
            .filterValue("true")
            .build()
            .getQueryString();
        Response response = searchKNNIndex(INDEX_NAME, query, 3);
        String entity = EntityUtils.toString(response.getEntity());
        List<String> docIds = parseIds(entity);
        assertEquals(2, docIds.size());
        assertEquals("3", docIds.get(0));
        assertEquals("1", docIds.get(1));
        assertEquals(2, parseTotalSearchHits(entity));
    }

    @SneakyThrows
    public void testNestedSearchWithLuceneByte_whenDocWithoutNestedObjectInSeparateSegment_thenSucceed() {
        String indexName = INDEX_NAME + "_lucene_byte_npe";
        String nestedFieldName = "nested";
        createKnnByteIndexWithNestedField(indexName, nestedFieldName, FIELD_NAME, 4, KNNEngine.LUCENE);

        String doc = NestedKnnDocBuilder.create(nestedFieldName)
            .addVectors(FIELD_NAME, new Byte[] { 1, 2, 3, 4 }, new Byte[] { 5, 6, 7, 8 })
            .build();
        addKnnDoc(indexName, "1", doc);
        flushIndex(indexName);

        addKnnDoc(indexName, "2", "{}");
        flushIndex(indexName);

        refreshIndex(indexName);

        Byte[] queryVector = { 1, 2, 3, 4 };
        String query = KNNJsonQueryBuilder.builder()
            .nestedFieldName(nestedFieldName)
            .fieldName(FIELD_NAME)
            .vector(queryVector)
            .k(1)
            .build()
            .getQueryString();
        Response response = searchKNNIndex(indexName, query, 1);
        String entity = EntityUtils.toString(response.getEntity());
        assertEquals(1, parseHits(entity));
        assertEquals("1", parseIds(entity).get(0));

        deleteKNNIndex(indexName);
    }

    /**
     * A nested byte field answered by exact search, because the segment is written without a graph: every parent
     * comes back with its own best child's score, checked against the vectors this test indexed with the space
     * type's own similarity function.
     */
    @SneakyThrows
    public void testNestedSearchByte_whenExactSearch_thenScoresMatchReference() {
        for (SpaceType space : List.of(SpaceType.L2, SpaceType.INNER_PRODUCT)) {
            final String indexName = "nested-exact-byte-" + space.getValue();
            final String nestedFieldName = "nested";
            createKnnByteIndexWithNestedField(
                indexName,
                nestedFieldName,
                FIELD_NAME,
                DIMENSION_EXACT,
                KNNEngine.FAISS,
                space,
                Settings.builder()
                    .put(getKNNDefaultIndexSettings())
                    .put(KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD, NO_GRAPH_THRESHOLD)
                    .build()
            );

            for (int group = 0; group < GROUPS.length; group++) {
                final NestedKnnDocBuilder builder = NestedKnnDocBuilder.create(nestedFieldName);
                for (int offset = 0; offset < GROUPS[group][0]; offset++) {
                    if (offset == GROUPS[group][2]) {
                        builder.addMetadata(Map.of(TAG_FIELD, "t" + offset));
                        continue;
                    }
                    builder.addVectors(FIELD_NAME, box(childVector(group, offset)));
                }
                addKnnDoc(indexName, "p" + group, builder.build());
            }
            refreshIndex(indexName);

            final int size = GROUPS.length + 5;
            final String query = KNNJsonQueryBuilder.builder()
                .nestedFieldName(nestedFieldName)
                .fieldName(FIELD_NAME)
                .vector(box(queryVector()))
                .k(size)
                .build()
                .getQueryString();
            final Response response = searchKNNIndex(indexName, query, size);
            final String entity = EntityUtils.toString(response.getEntity());

            final List<String> expectedIds = new ArrayList<>();
            for (int group = 0; group < GROUPS.length; group++) {
                if (GROUPS[group][0] > 0) {
                    expectedIds.add("p" + group);
                }
            }
            assertEquals("byte " + space.getValue() + ": the parents and their order", expectedIds, parseIds(entity));
            final List<Double> scores = parseScores(entity);
            for (int i = 0; i < expectedIds.size(); i++) {
                final int group = Integer.parseInt(expectedIds.get(i).substring(1));
                final float expected = space.getKnnVectorSimilarityFunction().compare(queryVector(), childVector(group, GROUPS[group][1]));
                assertEquals(
                    "byte " + space.getValue() + ": score of p" + group,
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
    private static final byte QUERY_VALUE = 100;
    private static final int NO_GRAPH_THRESHOLD = 100000;
    /** What a child with no vector carries instead, which is how a sparse nested field looks. */
    private static final String TAG_FIELD = "tag";

    private byte[] queryVector() {
        final byte[] query = new byte[DIMENSION_EXACT];
        Arrays.fill(query, QUERY_VALUE);
        return query;
    }

    /**
     * The group's named best child is the query with one coordinate pulled in by a step that grows with the group,
     * so the winners' scores fall with the group index and every one of them beats any other child, whose
     * coordinates alternate in sign.
     */
    private byte[] childVector(final int group, final int offset) {
        final byte[] vector = queryVector();
        if (offset == GROUPS[group][1]) {
            vector[0] = (byte) (QUERY_VALUE - 2 * (group + 1));
            return vector;
        }
        for (int i = 0; i < DIMENSION_EXACT; i++) {
            final int magnitude = 60 + ((group * 7 + offset * 3 + i) % 30);
            vector[i] = (byte) ((i % 2 == (offset % 2)) ? magnitude : -magnitude);
        }
        return vector;
    }

    private static Object[] box(final byte[] vector) {
        final Integer[] boxed = new Integer[vector.length];
        for (int i = 0; i < vector.length; i++) {
            boxed[i] = (int) vector[i];
        }
        return boxed;
    }

    private void createKnnByteIndexWithNestedField(
        final String indexName,
        final String nestedFieldName,
        final String fieldName,
        final int dimension,
        final KNNEngine knnEngine
    ) throws Exception {
        createKnnByteIndexWithNestedField(indexName, nestedFieldName, fieldName, dimension, knnEngine, null, null);
    }

    private void createKnnByteIndexWithNestedField(
        final String indexName,
        final String nestedFieldName,
        final String fieldName,
        final int dimension,
        final KNNEngine knnEngine,
        final SpaceType spaceType,
        final Settings settings
    ) throws Exception {
        KNNJsonIndexMappingsBuilder.Method method = KNNJsonIndexMappingsBuilder.Method.builder()
            .methodName(METHOD_HNSW)
            .engine(knnEngine.getName())
            .spaceType(spaceType == null ? null : spaceType.getValue())
            .build();

        String knnIndexMapping = KNNJsonIndexMappingsBuilder.builder()
            .nestedFieldName(nestedFieldName)
            .fieldName(fieldName)
            .dimension(dimension)
            .vectorDataType(VectorDataType.BYTE.getValue())
            .method(method)
            .build()
            .getIndexMapping();

        if (settings == null) {
            createKnnIndex(indexName, knnIndexMapping);
        } else {
            createKnnIndex(indexName, settings, knnIndexMapping);
        }
    }
}

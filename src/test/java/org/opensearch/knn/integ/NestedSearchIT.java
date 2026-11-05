/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ;

import lombok.SneakyThrows;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.CompressionTestConfig;
import org.opensearch.knn.KNNCompressionRestTestCase;
import org.opensearch.knn.KNNJsonIndexMappingsBuilder;
import org.opensearch.knn.KNNJsonQueryBuilder;
import org.opensearch.knn.NestedKnnDocBuilder;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.Mode;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.function.IntPredicate;

import static org.opensearch.knn.common.Constants.FIELD_FILTER;
import static org.opensearch.knn.common.Constants.FIELD_TERM;
import static org.opensearch.knn.common.KNNConstants.DIMENSION;
import static org.opensearch.knn.common.KNNConstants.K;
import static org.opensearch.knn.common.KNNConstants.KNN;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.KNN_METHOD;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_M;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.MIN_SCORE;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;
import static org.opensearch.knn.common.KNNConstants.PATH;
import static org.opensearch.knn.common.KNNConstants.QUERY;
import static org.opensearch.knn.common.KNNConstants.TYPE;
import static org.opensearch.knn.common.KNNConstants.TYPE_KNN_VECTOR;
import static org.opensearch.knn.common.KNNConstants.TYPE_NESTED;
import static org.opensearch.knn.common.KNNConstants.VECTOR;
import static org.opensearch.knn.index.query.parser.RescoreParser.RESCORE_OVERSAMPLE_PARAMETER;
import static org.opensearch.knn.index.query.parser.RescoreParser.RESCORE_PARAMETER;

public class NestedSearchIT extends KNNCompressionRestTestCase {
    private static final String INDEX_NAME = "test-index-nested-search";
    private static final String FIELD_NAME_NESTED = "test_nested";
    private static final String FIELD_NAME_VECTOR = "test_vector";
    private static final String FIELD_NAME_PARKING = "parking";
    private static final String FIELD_VALUE_TRUE = "true";
    private static final String FIELD_VALUE_FALSE = "false";
    private static final String PROPERTIES_FIELD = "properties";
    private static final int EF_CONSTRUCTION = 128;
    private static final int M = 16;
    private static final SpaceType SPACE_TYPE = SpaceType.L2;

    public NestedSearchIT(CompressionTestConfig compressionConfig) {
        super(compressionConfig);
    }

    @SneakyThrows
    public void testNestedSearchWithLucene_whenKIsTwo_thenReturnTwoResults() {
        createKnnIndex(2, KNNEngine.LUCENE.getName());

        int totalDocCount = 15;
        for (int i = 0; i < totalDocCount; i++) {
            String doc = NestedKnnDocBuilder.create(FIELD_NAME_NESTED)
                .addVectors(FIELD_NAME_VECTOR, new Float[] { (float) i, (float) i }, new Float[] { (float) i, (float) i })
                .build();
            addKnnDoc(INDEX_NAME, String.valueOf(i), doc);
        }

        refreshIndex(INDEX_NAME);
        forceMergeKnnIndex(INDEX_NAME);

        Float[] queryVector = { 14f, 14f };
        Response response = queryNestedField(INDEX_NAME, 2, queryVector);
        String entity = EntityUtils.toString(response.getEntity());
        assertEquals(2, parseHits(entity));
        assertEquals(2, parseTotalSearchHits(entity));
        assertEquals("14", parseIds(entity).get(0));
        assertEquals("13", parseIds(entity).get(1));
    }

    @SneakyThrows
    public void testNestedSearchWithFaiss_whenKIsTwo_thenReturnTwoResults() {
        createKnnIndex(2, KNNEngine.FAISS.getName());

        int totalDocCount = 15;
        for (int i = 0; i < totalDocCount; i++) {
            String doc = NestedKnnDocBuilder.create(FIELD_NAME_NESTED)
                .addVectors(FIELD_NAME_VECTOR, new Float[] { (float) i, (float) i }, new Float[] { (float) i, (float) i })
                .build();
            addKnnDoc(INDEX_NAME, String.valueOf(i), doc);
        }

        refreshIndex(INDEX_NAME);
        forceMergeKnnIndex(INDEX_NAME);

        Float[] queryVector = { 14f, 14f };
        Response response = queryNestedField(INDEX_NAME, 2, queryVector);
        String entity = EntityUtils.toString(response.getEntity());
        assertEquals(2, parseHits(entity));
        assertEquals(2, parseTotalSearchHits(entity));
        assertEquals("14", parseIds(entity).get(0));
        assertEquals("13", parseIds(entity).get(1));
    }

    @SneakyThrows
    public void testNestedSearchWithLucene_whenDocWithoutNestedObjectInSeparateSegment_thenSucceed() {
        createKnnIndex(2, KNNEngine.LUCENE.getName());

        String doc = NestedKnnDocBuilder.create(FIELD_NAME_NESTED)
            .addVectors(FIELD_NAME_VECTOR, new Float[] { 1f, 1f }, new Float[] { 2f, 2f })
            .build();
        addKnnDoc(INDEX_NAME, "1", doc);
        flushIndex(INDEX_NAME);

        addKnnDoc(INDEX_NAME, "2", "{}");
        flushIndex(INDEX_NAME);

        refreshIndex(INDEX_NAME);

        Float[] queryVector = { 1f, 1f };
        Response response = queryNestedField(INDEX_NAME, 1, queryVector);
        String entity = EntityUtils.toString(response.getEntity());
        assertEquals(1, parseHits(entity));
        assertEquals("1", parseIds(entity).get(0));
    }

    @SneakyThrows
    public void testNestedSearchWithFaiss_whenDocWithoutNestedObjectInSeparateSegment_thenSucceed() {
        createKnnIndex(2, KNNEngine.FAISS.getName());

        String doc = NestedKnnDocBuilder.create(FIELD_NAME_NESTED)
            .addVectors(FIELD_NAME_VECTOR, new Float[] { 1f, 1f }, new Float[] { 2f, 2f })
            .build();
        addKnnDoc(INDEX_NAME, "1", doc);
        flushIndex(INDEX_NAME);

        addKnnDoc(INDEX_NAME, "2", "{}");
        flushIndex(INDEX_NAME);

        refreshIndex(INDEX_NAME);

        Float[] queryVector = { 1f, 1f };
        Response response = queryNestedField(INDEX_NAME, 1, queryVector);
        String entity = EntityUtils.toString(response.getEntity());
        assertEquals(1, parseHits(entity));
        assertEquals("1", parseIds(entity).get(0));
    }

    @SneakyThrows
    public void testNestedSearchWithFaissMOS_whenDocWithoutNestedObjectInSeparateSegment_thenSucceed() {
        createKnnIndex(2, KNNEngine.FAISS.getName(), true);

        String doc = NestedKnnDocBuilder.create(FIELD_NAME_NESTED)
            .addVectors(FIELD_NAME_VECTOR, new Float[] { 1f, 1f }, new Float[] { 2f, 2f })
            .build();
        addKnnDoc(INDEX_NAME, "1", doc);
        flushIndex(INDEX_NAME);

        addKnnDoc(INDEX_NAME, "2", "{}");
        flushIndex(INDEX_NAME);

        refreshIndex(INDEX_NAME);

        Float[] queryVector = { 1f, 1f };
        Response response = queryNestedField(INDEX_NAME, 1, queryVector);
        String entity = EntityUtils.toString(response.getEntity());
        assertEquals(1, parseHits(entity));
        assertEquals("1", parseIds(entity).get(0));
    }

    @SneakyThrows
    public void testNestedSearchWithFaiss_whenKIsTwo_SomeNestedDocsHasNoVectors_thenReturnTwoResults() {
        createKnnIndex(2, KNNEngine.FAISS.getName());
        indexAndTestKNNIndexWithVectorAndNonVectorField();
    }

    @SneakyThrows
    public void testNestedSearchWithFaiss_whenRescoreEnabled_thenSucceed() {
        createKnnIndex(2, KNNEngine.FAISS.getName());

        int totalDocCount = 15;
        for (int i = 0; i < totalDocCount; i++) {
            String doc = NestedKnnDocBuilder.create(FIELD_NAME_NESTED)
                .addVectors(FIELD_NAME_VECTOR, new Float[] { (float) i, (float) i }, new Float[] { (float) i, (float) i })
                .build();
            addKnnDoc(INDEX_NAME, String.valueOf(i), doc);
        }

        refreshIndex(INDEX_NAME);
        forceMergeKnnIndex(INDEX_NAME);

        Float[] queryVector = { 14f, 14f };
        Response response = queryNestedField(INDEX_NAME, 2, queryVector, null, null, null, 2.0f);

        String entity = EntityUtils.toString(response.getEntity());
        assertEquals(2, parseHits(entity));
        assertEquals(2, parseTotalSearchHits(entity));
        assertEquals("14", parseIds(entity).get(0));
        assertEquals("13", parseIds(entity).get(1));
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
    public void testNestedSearchWithFaiss_whenDoingExactSearch_thenReturnCorrectResults() {
        createKnnIndex(3, KNNEngine.FAISS.getName());

        for (int i = 1; i < 4; i++) {
            float value = (float) i;
            String doc = NestedKnnDocBuilder.create(FIELD_NAME_NESTED)
                .addVectors(
                    FIELD_NAME_VECTOR,
                    new Float[] { value, value, value },
                    new Float[] { value, value, value },
                    new Float[] { value, value, value }
                )
                .addTopLevelField(FIELD_NAME_PARKING, i % 2 == 1 ? FIELD_VALUE_TRUE : FIELD_VALUE_FALSE)
                .build();
            addKnnDoc(INDEX_NAME, String.valueOf(i), doc);
        }
        refreshIndex(INDEX_NAME);
        forceMergeKnnIndex(INDEX_NAME);

        // Make it as an exact search by setting the threshold larger than size of filteredIds(6)
        updateIndexSettings(INDEX_NAME, Settings.builder().put(KNNSettings.ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD, 100));

        Float[] queryVector = { 3f, 3f, 3f };
        Response response = queryNestedField(INDEX_NAME, 3, queryVector, FIELD_NAME_PARKING, FIELD_VALUE_TRUE, null, null);
        String entity = EntityUtils.toString(response.getEntity());
        List<String> docIds = parseIds(entity);
        assertEquals(2, docIds.size());
        assertEquals("3", docIds.get(0));
        assertEquals("1", docIds.get(1));
        assertEquals(2, parseTotalSearchHits(entity));
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
     * 						"min_score": 0.00001,
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
    public void testNestedWithFaiss_whenFilter_whenDoRadialSearch_thenReturnCorrectResults() {
        assumeTrue("Radial search is not supported on quantized indices", isRadialSearchSupported());
        createKnnIndex(3, KNNEngine.FAISS.getName());

        for (int i = 1; i < 4; i++) {
            float value = (float) i;
            String doc = NestedKnnDocBuilder.create(FIELD_NAME_NESTED)
                .addVectors(
                    FIELD_NAME_VECTOR,
                    new Float[] { value, value, value },
                    new Float[] { value, value, value },
                    new Float[] { value, value, value }
                )
                .addTopLevelField(FIELD_NAME_PARKING, i % 2 == 1 ? FIELD_VALUE_TRUE : FIELD_VALUE_FALSE)
                .build();
            addKnnDoc(INDEX_NAME, String.valueOf(i), doc);
        }
        refreshIndex(INDEX_NAME);
        forceMergeKnnIndex(INDEX_NAME);

        Float[] queryVector = { 3f, 3f, 3f };
        Float minScore = 0.00001f;
        Response response = queryNestedField(INDEX_NAME, null, queryVector, FIELD_NAME_PARKING, FIELD_VALUE_TRUE, minScore, null);

        String entity = EntityUtils.toString(response.getEntity());
        List<String> docIds = parseIds(entity);
        assertEquals(2, docIds.size());
        assertEquals("3", docIds.get(0));
        assertEquals("1", docIds.get(1));
        assertEquals(2, parseTotalSearchHits(entity));
    }

    @SneakyThrows
    public void testNestedSearch_whenExactSearchByParentFilter_thenEachParentScoresItsBestChild() {
        assumeTrue(FULL_PRECISION_ONLY, isFullPrecision());
        for (final SpaceType space : List.of(SpaceType.L2, SpaceType.INNER_PRODUCT, SpaceType.COSINESIMIL)) {
            final String index = prefix() + space.getValue();
            createNestedIndex(
                index,
                nestedMapping(space, KNNEngine.FAISS.getName(), null, null),
                Settings.builder().put(KNNSettings.ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD, EXACT_SEARCH_THRESHOLD).build()
            );
            indexGroups(index, 0);
            forceMergeKnnIndex(index);

            final List<Integer> groups = groupsWhere(group -> FIELD_VALUE_TRUE.equals(bucketOf(group)));
            final int size = groups.size() + 5;
            final String query = nestedQuery().k(size)
                .filterFieldName(FIELD_NAME_PARKING)
                .filterValue(FIELD_VALUE_TRUE)
                .build()
                .getQueryString();
            final String entity = EntityUtils.toString(searchKNNIndex(index, query, size).getEntity());
            assertBestChildPerParent(space + " parent filter", space, groups, entity);
            deleteKNNIndex(index);
        }
    }

    @SneakyThrows
    public void testNestedSearch_whenExactSearchByChildFieldFilter_thenEachParentScoresItsBestAcceptedChild() {
        assumeTrue(FULL_PRECISION_ONLY, isFullPrecision());
        createNestedIndex(
            INDEX_NAME,
            nestedMapping(SPACE_TYPE, KNNEngine.FAISS.getName(), null, null),
            Settings.builder().put(KNNSettings.ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD, EXACT_SEARCH_THRESHOLD).build()
        );
        indexGroups(INDEX_NAME, 0);
        forceMergeKnnIndex(INDEX_NAME);

        // Even offsets only, so the filter keeps a group's named best child where that offset is even and leaves
        // the group with a fallback child where it is odd.
        final String shade = shadeOf(0);
        final int size = GROUPS.size() + 5;
        final String query = nestedQuery().k(size)
            .childFilterFieldName(FIELD_NAME_NESTED + "." + FIELD_NAME_SHADE + ".keyword")
            .filterValue(shade)
            .build()
            .getQueryString();
        final String entity = EntityUtils.toString(searchKNNIndex(INDEX_NAME, query, size).getEntity());
        assertBestAcceptedChildPerParent("child filter", SPACE_TYPE, offset -> shade.equals(shadeOf(offset)), entity);
    }

    @SneakyThrows
    public void testNestedSearch_whenSegmentHasNoGraph_thenEachParentScoresItsBestChild() {
        assumeTrue(FULL_PRECISION_ONLY, isFullPrecision());
        createNestedIndex(INDEX_NAME, nestedMapping(SPACE_TYPE, KNNEngine.FAISS.getName(), null, null), noGraphSettings());
        indexGroups(INDEX_NAME, 0);

        final List<Integer> groups = scoringGroups();
        final int size = groups.size() + 5;
        final String query = nestedQuery().k(size).build().getQueryString();
        final String entity = EntityUtils.toString(searchKNNIndex(INDEX_NAME, query, size).getEntity());
        assertBestChildPerParent("no graph", SPACE_TYPE, groups, entity);
    }

    @SneakyThrows
    public void testNestedSearch_whenRadialByMaxDistance_thenEachParentScoresItsBestChild() {
        assumeTrue("Radial search is not supported on quantized indices", isRadialSearchSupported());
        assertRadialByCut(false);
    }

    @SneakyThrows
    public void testNestedSearch_whenRadialByMinScore_thenEachParentScoresItsBestChild() {
        assumeTrue("Radial search is not supported on quantized indices", isRadialSearchSupported());
        assertRadialByCut(true);
    }

    @SneakyThrows
    public void testNestedSearch_whenMemoryOptimizedWithFilterAndWithRescore_thenEachParentScoresItsBestChild() {
        assumeTrue(FULL_PRECISION_ONLY, isFullPrecision());
        createNestedIndex(
            INDEX_NAME,
            nestedMapping(SPACE_TYPE, KNNEngine.FAISS.getName(), null, null),
            Settings.builder()
                .put(KNNSettings.MEMORY_OPTIMIZED_KNN_SEARCH_MODE, true)
                .put(KNNSettings.ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD, EXACT_SEARCH_THRESHOLD)
                .build()
        );
        indexGroups(INDEX_NAME, 0);
        forceMergeKnnIndex(INDEX_NAME);

        final List<Integer> groups = groupsWhere(group -> FIELD_VALUE_FALSE.equals(bucketOf(group)));
        final int size = groups.size() + 5;
        final String filtered = nestedQuery().k(size)
            .filterFieldName(FIELD_NAME_PARKING)
            .filterValue(FIELD_VALUE_FALSE)
            .build()
            .getQueryString();
        assertBestChildPerParent(
            "memory optimized with filter",
            SPACE_TYPE,
            groups,
            EntityUtils.toString(searchKNNIndex(INDEX_NAME, filtered, size).getEntity())
        );

        final String rescoreIndex = prefix() + "mos-rescore";
        createNestedIndex(
            rescoreIndex,
            nestedMapping(SPACE_TYPE, null, Mode.ON_DISK.getName(), CompressionLevel.x32.getName()),
            Settings.builder().put(KNNSettings.MEMORY_OPTIMIZED_KNN_SEARCH_MODE, true).build()
        );
        indexGroups(rescoreIndex, 0);
        forceMergeKnnIndex(rescoreIndex);

        final String rescored = nestedQuery().k(RESCORE_K).oversampleFactor(3.0f).build().getQueryString();
        assertRescoredParents(
            "memory optimized with rescore",
            SPACE_TYPE,
            EntityUtils.toString(searchKNNIndex(rescoreIndex, rescored, RESCORE_K).getEntity())
        );
        deleteKNNIndex(rescoreIndex);
    }

    @SneakyThrows
    public void testNestedSearch_whenPrefetchDisabled_thenIdenticalIdsAndScores() {
        createNestedIndex(INDEX_NAME, nestedMapping(SPACE_TYPE, KNNEngine.FAISS.getName(), null, null), noGraphSettings());
        indexGroups(INDEX_NAME, 0);

        final int size = GROUPS.size() + 5;
        final List<String> queries = List.of(
            nestedQuery().k(size).build().getQueryString(),
            nestedQuery().k(RESCORE_K).oversampleFactor(3.0f).build().getQueryString(),
            nestedQuery().k(RESCORE_K).expandNested(true).scoreMode("max").innerHitsSize(MAX_INNER_HITS).build().getQueryString()
        );
        try {
            for (final String query : queries) {
                updateClusterSettings(PREFETCH_SETTING, true);
                final String with = EntityUtils.toString(searchKNNIndex(INDEX_NAME, query, size).getEntity());
                updateClusterSettings(PREFETCH_SETTING, false);
                final String without = EntityUtils.toString(searchKNNIndex(INDEX_NAME, query, size).getEntity());

                assertEquals("the parents", parseIds(with), parseIds(without));
                assertEquals("the scores", parseScores(with), parseScores(without));
            }
        } finally {
            updateClusterSettings(PREFETCH_SETTING, (Object) null);
        }
    }

    @SneakyThrows
    public void testNestedSearch_whenAFilterRejectsWholeGroups_thenThoseParentsAreAbsent() {
        assumeTrue(FULL_PRECISION_ONLY, isFullPrecision());
        createNestedIndex(
            INDEX_NAME,
            nestedMapping(SPACE_TYPE, KNNEngine.FAISS.getName(), null, null),
            Settings.builder().put(KNNSettings.ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD, EXACT_SEARCH_THRESHOLD).build()
        );
        indexGroups(INDEX_NAME, 0);
        forceMergeKnnIndex(INDEX_NAME);

        final List<Integer> groups = groupsWhere(group -> FIELD_VALUE_TRUE.equals(bucketOf(group)));
        final int size = GROUPS.size() + 5;
        final String query = nestedQuery().k(size)
            .filterFieldName(FIELD_NAME_PARKING)
            .filterValue(FIELD_VALUE_TRUE)
            .build()
            .getQueryString();
        final String entity = EntityUtils.toString(searchKNNIndex(INDEX_NAME, query, size).getEntity());
        assertBestChildPerParent("filter keeps one bucket", SPACE_TYPE, groups, entity);
        for (int group = 0; group < GROUPS.size(); group++) {
            if (FIELD_VALUE_TRUE.equals(bucketOf(group)) == false) {
                assertFalse("a rejected parent came back", parseIds(entity).contains(parentId(group)));
            }
        }
    }

    @SneakyThrows
    public void testNestedSearch_whenAFilterMatchesNothing_thenNoHits() {
        createNestedIndex(
            INDEX_NAME,
            nestedMapping(SPACE_TYPE, KNNEngine.FAISS.getName(), null, null),
            Settings.builder().put(KNNSettings.ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD, EXACT_SEARCH_THRESHOLD).build()
        );
        indexGroups(INDEX_NAME, 0);
        forceMergeKnnIndex(INDEX_NAME);

        final String query = nestedQuery().k(10).filterFieldName(FIELD_NAME_PARKING).filterValue("nosuchbucket").build().getQueryString();
        assertEquals(0, parseHits(EntityUtils.toString(searchKNNIndex(INDEX_NAME, query, 10).getEntity())));
    }

    @SneakyThrows
    public void testNestedSearch_whenParentsDeletedAndUpdated_thenLiveParentsScoreTheirBestChild() {
        assumeTrue(FULL_PRECISION_ONLY, isFullPrecision());
        createNestedIndex(INDEX_NAME, nestedMapping(SPACE_TYPE, KNNEngine.FAISS.getName(), null, null), noGraphSettings());
        indexGroups(INDEX_NAME, 0);

        // Two parents deleted and two reindexed with another group's children, refreshed without a force merge, so
        // the deleted children are still in the segment the scorer walks.
        final int deletedA = 2;
        final int deletedB = 9;
        final int updatedA = 1;
        final int updatedB = 11;
        deleteKnnDoc(INDEX_NAME, parentId(deletedA));
        deleteKnnDoc(INDEX_NAME, parentId(deletedB));
        addKnnDoc(INDEX_NAME, parentId(updatedA), groupDocument(6));
        addKnnDoc(INDEX_NAME, parentId(updatedB), groupDocument(8));
        refreshIndex(INDEX_NAME);

        final int size = GROUPS.size() + 5;
        final String query = nestedQuery().k(size).build().getQueryString();
        final String entity = EntityUtils.toString(searchKNNIndex(INDEX_NAME, query, size).getEntity());
        final List<String> ids = parseIds(entity);
        assertFalse("a deleted parent came back", ids.contains(parentId(deletedA)));
        assertFalse("a deleted parent came back", ids.contains(parentId(deletedB)));
        assertTrue("an updated parent is missing", ids.contains(parentId(updatedA)));
        assertTrue("an updated parent is missing", ids.contains(parentId(updatedB)));
        for (final int group : scoringGroups()) {
            if (group == deletedA || group == deletedB || group == updatedA || group == updatedB) {
                continue;
            }
            final float expected = expectedBestScore(SPACE_TYPE, group);
            assertEquals("score of " + parentId(group), expected, scoreOf(entity, parentId(group)), allowance(expected));
        }
    }

    @SneakyThrows
    public void testNestedSearch_whenSeveralSegments_thenEachParentScoresItsBestChild() {
        assumeTrue(FULL_PRECISION_ONLY, isFullPrecision());
        createNestedIndex(INDEX_NAME, nestedMapping(SPACE_TYPE, KNNEngine.FAISS.getName(), null, null), noGraphSettings());
        // A flush every four parents, so the groups land in several segments.
        indexGroups(INDEX_NAME, 4);
        // A segment with no nested object at all.
        addKnnDoc(INDEX_NAME, "no-nested", "{}");
        flushIndex(INDEX_NAME);
        // A segment whose only parent's children all lack the vector.
        addKnnDoc(INDEX_NAME, "no-vectors", groupDocument(VECTORLESS_GROUP));
        flushIndex(INDEX_NAME);
        refreshIndex(INDEX_NAME);

        final List<Integer> groups = scoringGroups();
        final int size = groups.size() + 5;
        final String query = nestedQuery().k(size).build().getQueryString();
        final String entity = EntityUtils.toString(searchKNNIndex(INDEX_NAME, query, size).getEntity());
        assertBestChildPerParent("several segments", SPACE_TYPE, groups, entity);
    }

    @SneakyThrows
    public void testNestedSearch_whenTheIndexAlsoHasATopLevelVectorField_thenNeitherAffectsTheOther() {
        assumeTrue(FULL_PRECISION_ONLY, isFullPrecision());
        final String mapping = addCompressionMappingFields(
            KNNJsonIndexMappingsBuilder.builder()
                .nestedFieldName(FIELD_NAME_NESTED)
                .fieldName(FIELD_NAME_VECTOR)
                .topLevelFieldName(FIELD_NAME_TOP_LEVEL_VECTOR)
                .dimension(DIMENSION_NESTED)
        ).build().getIndexMapping();
        createNestedIndex(INDEX_NAME, mapping, noGraphSettings());
        indexGroups(INDEX_NAME, 0);
        // One document carrying only the top-level field, so a query on it has something of its own to find.
        addKnnDoc(INDEX_NAME, "top-only", topLevelDocument());
        refreshIndex(INDEX_NAME);

        final List<Integer> groups = scoringGroups();
        final int size = groups.size() + 5;
        final String nested = nestedQuery().k(size).build().getQueryString();
        final String entity = EntityUtils.toString(searchKNNIndex(INDEX_NAME, nested, size).getEntity());
        assertBestChildPerParent("nested query beside a top level field", SPACE_TYPE, groups, entity);
        assertFalse("the top-level-only document answered a nested query", parseIds(entity).contains("top-only"));

        final String topLevel = KNNJsonQueryBuilder.builder()
            .fieldName(FIELD_NAME_TOP_LEVEL_VECTOR)
            .vector(queryVector())
            .k(3)
            .build()
            .getQueryString();
        final String topEntity = EntityUtils.toString(searchKNNIndex(INDEX_NAME, topLevel, 3).getEntity());
        assertEquals("only the top-level document has that field", List.of("top-only"), parseIds(topEntity));
    }

    @SneakyThrows
    public void testNestedSearch_whenTwoNestedVectorFields_thenEachQueryIsIndependent() {
        assumeTrue(FULL_PRECISION_ONLY, isFullPrecision());
        final String mapping = addCompressionMappingFields(
            KNNJsonIndexMappingsBuilder.builder()
                .nestedFieldName(FIELD_NAME_NESTED)
                .fieldName(FIELD_NAME_VECTOR)
                .secondFieldName(FIELD_NAME_SECOND_VECTOR)
                .dimension(DIMENSION_NESTED)
        ).build().getIndexMapping();
        createNestedIndex(INDEX_NAME, mapping, noGraphSettings());
        indexGroups(INDEX_NAME, 0);

        final List<Integer> groups = scoringGroups();
        final int size = groups.size() + 5;
        final String populated = nestedQuery().k(size).build().getQueryString();
        assertBestChildPerParent(
            "the populated field",
            SPACE_TYPE,
            groups,
            EntityUtils.toString(searchKNNIndex(INDEX_NAME, populated, size).getEntity())
        );

        // The second field has no values at all, so a query on it finds nothing and does not disturb the first.
        final String second = KNNJsonQueryBuilder.builder()
            .nestedFieldName(FIELD_NAME_NESTED)
            .fieldName(FIELD_NAME_SECOND_VECTOR)
            .vector(queryVector())
            .k(10)
            .build()
            .getQueryString();
        assertEquals(0, parseHits(EntityUtils.toString(searchKNNIndex(INDEX_NAME, second, 10).getEntity())));
        assertBestChildPerParent(
            "the populated field again",
            SPACE_TYPE,
            groups,
            EntityUtils.toString(searchKNNIndex(INDEX_NAME, populated, size).getEntity())
        );
    }

    @SneakyThrows
    public void testNestedSearch_whenKBelowEqualAndAboveTheMatchingParents_thenKParentsComeBack() {
        assumeTrue(FULL_PRECISION_ONLY, isFullPrecision());
        createNestedIndex(INDEX_NAME, nestedMapping(SPACE_TYPE, KNNEngine.FAISS.getName(), null, null), noGraphSettings());
        indexGroups(INDEX_NAME, 0);

        final List<Integer> groups = scoringGroups();
        final int matching = groups.size();
        for (final int k : List.of(3, matching, matching + 10)) {
            final int size = Math.min(k, matching);
            final String query = nestedQuery().k(k).build().getQueryString();
            final String entity = EntityUtils.toString(searchKNNIndex(INDEX_NAME, query, size).getEntity());
            assertEquals("hits for k " + k, size, parseIds(entity).size());
            assertBestChildPerParent("k " + k, SPACE_TYPE, groups.subList(0, size), entity);
        }

        // size below k returns size hits, still the reference's prefix.
        final String query = nestedQuery().k(10).build().getQueryString();
        final String entity = EntityUtils.toString(searchKNNIndex(INDEX_NAME, query, 2).getEntity());
        assertEquals(2, parseIds(entity).size());
        assertBestChildPerParent("size below k", SPACE_TYPE, groups.subList(0, 2), entity);
    }

    @SneakyThrows
    public void testNestedSearch_whenOnDiskQuantizedRescore_thenEachParentScoresItsBestChild() {
        assumeTrue(OWN_COMPRESSION, isFullPrecision());
        for (final String compression : List.of("32x", "16x", "8x")) {
            final String index = prefix() + compression;
            createNestedIndex(index, nestedMapping(SPACE_TYPE, null, Mode.ON_DISK.getName(), compression), null);
            indexGroups(index, 0);
            forceMergeKnnIndex(index);

            for (final Float oversample : List.of(1.0f, 3.0f)) {
                final String query = nestedQuery().k(RESCORE_K).oversampleFactor(oversample).build().getQueryString();
                assertRescoredParents(
                    compression + " oversample " + oversample,
                    SPACE_TYPE,
                    EntityUtils.toString(searchKNNIndex(index, query, RESCORE_K).getEntity())
                );
            }
            deleteKNNIndex(index);
        }
    }

    @SneakyThrows
    public void testNestedSearch_whenOneBitAdcRescore_thenEachParentScoresItsBestChild() {
        assumeTrue(OWN_COMPRESSION, isFullPrecision());
        for (final SpaceType space : List.of(SpaceType.L2, SpaceType.INNER_PRODUCT, SpaceType.COSINESIMIL)) {
            final String index = prefix() + "adc-" + space.getValue();
            createNestedIndex(index, adcMapping(space), null);
            indexGroups(index, 0);
            forceMergeKnnIndex(index);

            final String query = nestedQuery().k(RESCORE_K).oversampleFactor(5.0f).build().getQueryString();
            assertRescoredParents(
                "adc " + space.getValue(),
                space,
                EntityUtils.toString(searchKNNIndex(index, query, RESCORE_K).getEntity())
            );
            deleteKNNIndex(index);
        }
    }

    @SneakyThrows
    public void testNestedSearch_whenConcurrentSegmentSearch_thenEachParentScoresItsBestChild() {
        assumeTrue(FULL_PRECISION_ONLY, isFullPrecision());
        createNestedIndex(INDEX_NAME, nestedMapping(SPACE_TYPE, KNNEngine.FAISS.getName(), null, null), noGraphSettings());
        indexGroups(INDEX_NAME, 3);
        updateIndexSettings(INDEX_NAME, Settings.builder().put("index.search.concurrent_segment_search.mode", "all"));

        final List<Integer> groups = scoringGroups();
        final int size = groups.size() + 5;
        final String query = nestedQuery().k(size).build().getQueryString();
        assertBestChildPerParent(
            "concurrent segment search",
            SPACE_TYPE,
            groups,
            EntityUtils.toString(searchKNNIndex(INDEX_NAME, query, size).getEntity())
        );
    }

    private void assertRadialByCut(final boolean byMinScore) throws Exception {
        createNestedIndex(
            INDEX_NAME,
            nestedMapping(SPACE_TYPE, KNNEngine.FAISS.getName(), null, null),
            Settings.builder()
                .put(KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD, EXACT_SEARCH_THRESHOLD)
                .put(KNNSettings.ADVANCED_FILTERED_EXACT_SEARCH_THRESHOLD, EXACT_SEARCH_THRESHOLD)
                .build()
        );
        indexGroups(INDEX_NAME, 0);

        // A radius strictly between two neighbouring reference scores, so no parent sits on the boundary and
        // whether it comes back cannot turn on the last bit of its score.
        final List<Integer> all = scoringGroups();
        final int middle = all.size() / 2;
        final float cut = (expectedBestScore(SPACE_TYPE, all.get(middle)) + expectedBestScore(SPACE_TYPE, all.get(middle + 1))) / 2.0f;

        final List<Integer> above = all.stream().filter(group -> expectedBestScore(SPACE_TYPE, group) >= cut).toList();
        final String unfiltered = radialQuery(byMinScore, cut, null).getQueryString();
        assertBestChildPerParent(
            "radial no filter",
            SPACE_TYPE,
            above,
            EntityUtils.toString(searchKNNIndex(INDEX_NAME, unfiltered, RADIAL_SIZE).getEntity())
        );

        final List<Integer> filtered = above.stream().filter(group -> FIELD_VALUE_TRUE.equals(bucketOf(group))).toList();
        final String withFilter = radialQuery(byMinScore, cut, FIELD_VALUE_TRUE).getQueryString();
        assertBestChildPerParent(
            "radial with filter",
            SPACE_TYPE,
            filtered,
            EntityUtils.toString(searchKNNIndex(INDEX_NAME, withFilter, RADIAL_SIZE).getEntity())
        );
    }

    private KNNJsonQueryBuilder radialQuery(final boolean byMinScore, final float cut, final String bucket) {
        final KNNJsonQueryBuilder.KNNJsonQueryBuilderBuilder query = nestedQuery();
        if (byMinScore) {
            query.minScore(cut);
        } else {
            query.maxDistance(SPACE_TYPE.scoreToDistanceTranslation(cut));
        }
        if (bucket != null) {
            query.filterFieldName(FIELD_NAME_PARKING).filterValue(bucket);
        }
        return query.build();
    }

    @SneakyThrows
    private void indexAndTestKNNIndexWithVectorAndNonVectorField() {
        int totalDocCount = 15;
        for (int i = 0; i < totalDocCount; i++) {
            final String doc = String.format(
                Locale.ROOT,
                "{\"title\": \"Document $i\",\"test_nested\": [{\"title\": \"Document $i\","
                    + "\"test_vector\": "
                    + "[%d, %d]},{\"title\": \"Document %d - 2\"}]}",
                i,
                i,
                i
            );
            addKnnDoc(INDEX_NAME, String.valueOf(i), doc);
        }

        refreshIndex(INDEX_NAME);
        forceMergeKnnIndex(INDEX_NAME);

        Float[] queryVector = { 14f, 14f };
        Response response = queryNestedField(INDEX_NAME, 2, queryVector, null, null, null, 2.0f);
        String entity = EntityUtils.toString(response.getEntity());
        assertEquals(2, parseHits(entity));
        assertEquals(2, parseTotalSearchHits(entity));
        assertEquals("14", parseIds(entity).get(0));
        assertEquals("13", parseIds(entity).get(1));
    }

    /**
     * {
     *      "properties": {
     *          "test_nested": {
     *              "type": "nested",
     *              "properties": {
     *                  "test_vector": {
     *                      "type": "knn_vector",
     *                      "dimension": 3,
     *                      "method": {
     *                          "name": "hnsw",
     *                          "space_type": "l2",
     *                          "engine": "lucene",
     *                          "parameters": {
     *                              "ef_construction": 128,
     *                              "m": 24
     *                          }
     *                      }
     *                  }
     *              }
     *          },
     *          "title": { "type": "text"}
     *      }
     *  }
     */
    private void createKnnIndex(final int dimension, final String engine) throws Exception {
        createKnnIndex(dimension, engine, false);
    }

    private void createKnnIndex(final int dimension, final String engine, boolean memoryOptimizedSearch) throws Exception {
        XContentBuilder builder = XContentFactory.jsonBuilder()
            .startObject()
            .startObject(PROPERTIES_FIELD)
            .startObject(FIELD_NAME_NESTED)
            .field(TYPE, TYPE_NESTED)
            .startObject(PROPERTIES_FIELD)
            .startObject("title")
            .field("type", "text")
            .endObject()
            .startObject(FIELD_NAME_VECTOR)
            .field(TYPE, TYPE_KNN_VECTOR)
            .field(DIMENSION, dimension);
        addCompressionMappingFields(builder);
        builder.startObject(KNN_METHOD)
            .field(NAME, METHOD_HNSW)
            .field(METHOD_PARAMETER_SPACE_TYPE, SPACE_TYPE)
            .field(KNN_ENGINE, engine)
            .startObject(PARAMETERS)
            .field(METHOD_PARAMETER_M, M)
            .field(METHOD_PARAMETER_EF_CONSTRUCTION, EF_CONSTRUCTION)
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject()
            .endObject();

        String mapping = builder.toString();
        if (memoryOptimizedSearch) {
            Settings settings = Settings.builder()
                .put(getKNNDefaultIndexSettings())
                .put(KNNSettings.MEMORY_OPTIMIZED_KNN_SEARCH_MODE, true)
                .build();
            createKnnIndex(INDEX_NAME, settings, mapping);
        } else {
            createKnnIndex(INDEX_NAME, mapping);
        }
    }

    private Response queryNestedField(final String index, final int k, final Object[] vector) throws IOException {
        return queryNestedField(index, k, vector, null, null, null, null);
    }

    private Response queryNestedField(
        final String index,
        final Integer k,
        final Object[] vector,
        final String filterName,
        final String filterValue,
        final Float minScore,
        final Float oversampleFactor
    ) throws IOException {
        XContentBuilder builder = XContentFactory.jsonBuilder().startObject().startObject(QUERY);
        builder.startObject(TYPE_NESTED);
        builder.field(PATH, FIELD_NAME_NESTED);
        builder.startObject(QUERY).startObject(KNN).startObject(FIELD_NAME_NESTED + "." + FIELD_NAME_VECTOR);
        builder.field(VECTOR, vector);
        if (minScore != null) {
            builder.field(MIN_SCORE, minScore);
        } else if (k != null) {
            builder.field(K, k);
        } else {
            throw new IllegalArgumentException("k or minScore must be provided in the query");
        }
        if (filterName != null && filterValue != null) {
            builder.startObject(FIELD_FILTER);
            builder.startObject(FIELD_TERM);
            builder.field(filterName, filterValue);
            builder.endObject();
            builder.endObject();
        }
        if (oversampleFactor != null) {
            builder.startObject(RESCORE_PARAMETER);
            builder.field(RESCORE_OVERSAMPLE_PARAMETER, oversampleFactor);
            builder.endObject();
        }

        builder.endObject().endObject().endObject().endObject().endObject().endObject();

        Request request = new Request("POST", "/" + index + "/_search");
        request.setJsonEntity(builder.toString());

        Response response = client().performRequest(request);
        assertEquals(request.getEndpoint() + ": failed", RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));

        return response;
    }

    // ---- the parent groups the exact search and rescore methods above index ----

    /** One parent group: how many children it has, which child must win, and which children carry no vector. */
    private record Group(int childCount, int bestOffset, Set<Integer> withoutVector) {
        static Group of(final int childCount, final int bestOffset, final Integer... withoutVector) {
            return new Group(childCount, bestOffset, Set.of(withoutVector));
        }

        boolean hasVector(final int offset) {
            return withoutVector.contains(offset) == false;
        }
    }

    private static final List<Group> GROUPS = List.of(
        Group.of(1, 0),                           // a single child
        Group.of(3, 0),                           // the first child wins
        Group.of(5, 4),                           // the last child wins
        Group.of(7, 3, 0, 4),                     // the first child and a middle child have no vector
        Group.of(0, -1),                          // a parent with no nested documents
        Group.of(4, -1, 0, 1, 2, 3),              // every child lacks the vector
        Group.of(63, 31),                         // one below a bulk batch
        Group.of(64, 0),                          // exactly a bulk batch, the first child wins
        Group.of(65, 64),                         // one above a bulk batch, the last child wins
        Group.of(130, 65),                        // two batches and a bit
        Group.of(200, 137),                       // several batches, the winner in the third
        Group.of(6, 2, 5),                        // the last child has no vector
        Group.of(5, 3, 1),                        // the child nearest the query has no vector
        Group.of(2, 0),                           // four small groups, together inside one batch
        Group.of(3, 1),
        Group.of(2, 0),
        Group.of(3, 1)
    );

    /** The group whose children all lack the vector. */
    private static final int VECTORLESS_GROUP = 5;
    private static final String FIELD_NAME_SHADE = "shade";
    private static final String FIELD_NAME_SECOND_VECTOR = "test_vector_two";
    private static final String FIELD_NAME_TOP_LEVEL_VECTOR = "top_vector";
    private static final String PREFETCH_SETTING = "knn.feature.prefetch.enabled";
    private static final String FULL_PRECISION_ONLY = "a reference score needs the uncompressed configuration";
    private static final String OWN_COMPRESSION = "the mapping carries its own compression, so one configuration is enough";
    private static final int DIMENSION_NESTED = 8;
    private static final float QUERY_VALUE = 3.0f;
    /** Above any group's child count, so a filtered or thresholded query is answered by exact search. */
    private static final int EXACT_SEARCH_THRESHOLD = 100000;
    /** Enough hits for a radial query, which has no k of its own to size the result set from. */
    private static final int RADIAL_SIZE = 1000;
    /** The default {@code index.max_inner_result_window}, which an inner hits request may not exceed. */
    private static final int MAX_INNER_HITS = 100;
    private static final int RESCORE_K = 10;
    private static final float SCORE_TOLERANCE = 1.0e-5f;

    private boolean isFullPrecision() {
        return compressionConfig.isCompressed() == false;
    }

    private Settings noGraphSettings() {
        return Settings.builder().put(KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD, EXACT_SEARCH_THRESHOLD).build();
    }

    private void createNestedIndex(final String index, final String mapping, final Settings extra) throws Exception {
        final Settings.Builder settings = Settings.builder().put(getKNNDefaultIndexSettings());
        if (extra != null) {
            settings.put(extra);
        }
        createKnnIndex(index, settings.build(), mapping);
    }

    private String nestedMapping(final SpaceType space, final String engine, final String mode, final String compression)
        throws IOException {
        final KNNJsonIndexMappingsBuilder.KNNJsonIndexMappingsBuilderBuilder builder = KNNJsonIndexMappingsBuilder.builder()
            .nestedFieldName(FIELD_NAME_NESTED)
            .fieldName(FIELD_NAME_VECTOR)
            .dimension(DIMENSION_NESTED);
        if (engine != null) {
            builder.method(
                KNNJsonIndexMappingsBuilder.Method.builder()
                    .methodName(METHOD_HNSW)
                    .engine(engine)
                    .spaceType(space.getValue())
                    .parameters(KNNJsonIndexMappingsBuilder.Method.Parameters.builder().m(M).efConstruction(EF_CONSTRUCTION).build())
                    .build()
            );
        }
        if (mode != null) {
            return builder.mode(mode).compressionLevel(compression).build().getIndexMapping();
        }
        return addCompressionMappingFields(builder).build().getIndexMapping();
    }

    /** A one-bit binary encoder with asymmetric distance computation on, which is how the ADC path is reached. */
    private String adcMapping(final SpaceType space) throws IOException {
        return KNNJsonIndexMappingsBuilder.builder()
            .nestedFieldName(FIELD_NAME_NESTED)
            .fieldName(FIELD_NAME_VECTOR)
            .dimension(DIMENSION_NESTED)
            .method(
                KNNJsonIndexMappingsBuilder.Method.builder()
                    .methodName(METHOD_HNSW)
                    .engine(KNNEngine.FAISS.getName())
                    .spaceType(space.getValue())
                    .parameters(
                        KNNJsonIndexMappingsBuilder.Method.Parameters.builder()
                            .encoder(
                                KNNJsonIndexMappingsBuilder.Method.Parameters.Encoder.builder()
                                    .name("binary")
                                    .bits(1)
                                    .enableAdc(true)
                                    .build()
                            )
                            .build()
                    )
                    .build()
            )
            .build()
            .getIndexMapping();
    }

    private KNNJsonQueryBuilder.KNNJsonQueryBuilderBuilder nestedQuery() {
        return KNNJsonQueryBuilder.builder().nestedFieldName(FIELD_NAME_NESTED).fieldName(FIELD_NAME_VECTOR).vector(queryVector());
    }

    private void indexGroups(final String index, final int flushEvery) throws Exception {
        for (int group = 0; group < GROUPS.size(); group++) {
            addKnnDoc(index, parentId(group), groupDocument(group));
            if (flushEvery > 0 && (group + 1) % flushEvery == 0) {
                flushIndex(index);
            }
        }
        refreshIndex(index);
    }

    private String groupDocument(final int group) throws IOException {
        final NestedKnnDocBuilder builder = NestedKnnDocBuilder.create(FIELD_NAME_NESTED);
        final Group shape = GROUPS.get(group);
        for (int offset = 0; offset < shape.childCount(); offset++) {
            if (shape.hasVector(offset)) {
                builder.addVectorWithMetadata(FIELD_NAME_VECTOR, box(childVector(group, offset)), FIELD_NAME_SHADE, shadeOf(offset));
            } else {
                builder.addMetadata(Map.of(FIELD_NAME_SHADE, shadeOf(offset)));
            }
        }
        return builder.addTopLevelField(FIELD_NAME_PARKING, bucketOf(group)).build();
    }

    private String topLevelDocument() throws IOException {
        return XContentFactory.jsonBuilder().startObject().field(FIELD_NAME_TOP_LEVEL_VECTOR, queryVector()).endObject().toString();
    }

    /**
     * The named best child is the query with one coordinate pulled in by a step that grows with the group, so every
     * group's winner has a distinct score, the winners' scores fall with the group index, and all of them beat any
     * other child. Every other child points away from the query: half of its coordinates carry the opposite sign.
     */
    private float[] childVector(final int group, final int offset) {
        final float[] vector = new float[DIMENSION_NESTED];
        if (offset == GROUPS.get(group).bestOffset()) {
            Arrays.fill(vector, QUERY_VALUE);
            vector[0] = QUERY_VALUE * (0.7f - 0.02f * group);
            return vector;
        }
        // Half the coordinates carry the opposite sign, and the magnitude grows with the group and with the
        // offset, so a group that loses its named child to a filter still has a best accepted child of its own
        // whose score no other group's comes close to.
        for (int i = 0; i < DIMENSION_NESTED; i++) {
            final float magnitude = 1.0f + 0.05f * group + 0.01f * offset;
            vector[i] = QUERY_VALUE * magnitude * ((i % 2 == 0) ? 1.0f : -1.0f);
        }
        return vector;
    }

    private float[] queryFloats() {
        final float[] query = new float[DIMENSION_NESTED];
        Arrays.fill(query, QUERY_VALUE);
        return query;
    }

    private Object[] queryVector() {
        return box(queryFloats());
    }

    private static Object[] box(final float[] vector) {
        final Float[] boxed = new Float[vector.length];
        for (int i = 0; i < vector.length; i++) {
            boxed[i] = vector[i];
        }
        return boxed;
    }

    /** The parent document id of a group, zero padded so the ids sort the way the groups are numbered. */
    private String parentId(final int group) {
        return String.format(Locale.ROOT, "p%03d", group);
    }

    private int groupOf(final String parentId) {
        return Integer.parseInt(parentId.substring(1));
    }

    private String bucketOf(final int group) {
        return group % 2 == 0 ? FIELD_VALUE_TRUE : FIELD_VALUE_FALSE;
    }

    private String shadeOf(final int offset) {
        return "s" + (offset % 2);
    }

    /** The groups a query must return, in the order their best children score, which is the group order. */
    private List<Integer> scoringGroups() {
        return groupsWhere(group -> true);
    }

    private List<Integer> groupsWhere(final IntPredicate accepted) {
        final List<Integer> groups = new ArrayList<>();
        for (int group = 0; group < GROUPS.size(); group++) {
            if (accepted.test(group) && hasAChildWithAVector(group)) {
                groups.add(group);
            }
        }
        return groups;
    }

    private boolean hasAChildWithAVector(final int group) {
        final Group shape = GROUPS.get(group);
        for (int offset = 0; offset < shape.childCount(); offset++) {
            if (shape.hasVector(offset)) {
                return true;
            }
        }
        return false;
    }

    /** The group's best accepted child, or -1 when it has none: the ground truth for one parent. */
    private int bestAcceptedChild(final SpaceType space, final int group, final IntPredicate acceptedOffset) {
        final Group shape = GROUPS.get(group);
        int best = -1;
        float bestScore = Float.NEGATIVE_INFINITY;
        for (int offset = 0; offset < shape.childCount(); offset++) {
            if (shape.hasVector(offset) == false || acceptedOffset.test(offset) == false) {
                continue;
            }
            final float score = expectedScore(space, group, offset);
            if (score > bestScore) {
                bestScore = score;
                best = offset;
            }
        }
        return best;
    }

    private float expectedScore(final SpaceType space, final int group, final int offset) {
        return space.getKnnVectorSimilarityFunction().compare(queryFloats(), childVector(group, offset));
    }

    private float expectedBestScore(final SpaceType space, final int group) {
        return expectedScore(space, group, bestAcceptedChild(space, group, offset -> true));
    }

    private double allowance(final float score) {
        return SCORE_TOLERANCE * Math.max(1.0, Math.abs(score));
    }

    private void assertBestChildPerParent(final String what, final SpaceType space, final List<Integer> groups, final String entity)
        throws IOException {
        final List<String> expectedIds = groups.stream().map(this::parentId).toList();
        assertEquals(what + ": the parents and their order", expectedIds, parseIds(entity));
        final List<Double> scores = parseScores(entity);
        assertEquals(what + ": a score for every hit", expectedIds.size(), scores.size());
        for (int i = 0; i < groups.size(); i++) {
            final float expected = expectedBestScore(space, groups.get(i));
            assertEquals(what + ": score of " + expectedIds.get(i), expected, scores.get(i), allowance(expected));
        }
    }

    private void assertBestAcceptedChildPerParent(
        final String what,
        final SpaceType space,
        final IntPredicate acceptedOffset,
        final String entity
    ) throws IOException {
        final List<Map.Entry<String, Float>> expected = new ArrayList<>();
        for (int group = 0; group < GROUPS.size(); group++) {
            final int best = bestAcceptedChild(space, group, acceptedOffset);
            if (best >= 0) {
                expected.add(Map.entry(parentId(group), expectedScore(space, group, best)));
            }
        }
        expected.sort(Comparator.comparingDouble((Map.Entry<String, Float> entry) -> -entry.getValue()).thenComparing(Map.Entry::getKey));
        // Two parents closer together than two comparison allowances would make the order below ambiguous, so an
        // unlucky layout fails here rather than flakily.
        for (int i = 1; i < expected.size(); i++) {
            final float above = expected.get(i - 1).getValue();
            final float below = expected.get(i).getValue();
            assertTrue(
                what + ": " + expected.get(i - 1).getKey() + " and " + expected.get(i).getKey() + " score too close to order",
                above - below > 2 * allowance(above)
            );
        }

        final List<String> expectedIds = expected.stream().map(Map.Entry::getKey).toList();
        assertEquals(what + ": the parents and their order", expectedIds, parseIds(entity));
        final List<Double> scores = parseScores(entity);
        for (int i = 0; i < expected.size(); i++) {
            final float want = expected.get(i).getValue();
            assertEquals(what + ": score of " + expectedIds.get(i), want, scores.get(i), allowance(want));
        }
    }

    /**
     * An approximate path followed by a rescore returns some of the reference's parents, each with that parent's
     * best child's score, in descending reference order. It may miss a parent, which is what makes it approximate.
     */
    private void assertRescoredParents(final String what, final SpaceType space, final String entity) throws IOException {
        final List<String> ids = parseIds(entity);
        final List<Double> scores = parseScores(entity);
        assertFalse(what + ": no hits at all", ids.isEmpty());
        final List<Integer> order = scoringGroups();
        int previous = -1;
        for (int i = 0; i < ids.size(); i++) {
            final int rank = order.indexOf(groupOf(ids.get(i)));
            assertTrue(what + ": " + ids.get(i) + " is not a parent the reference returns", rank >= 0);
            final float expected = expectedBestScore(space, groupOf(ids.get(i)));
            assertEquals(what + ": score of " + ids.get(i), expected, scores.get(i), allowance(expected));
            assertTrue(what + ": " + ids.get(i) + " came back out of reference order", rank > previous);
            previous = rank;
        }
    }

    private double scoreOf(final String entity, final String parentId) throws IOException {
        final List<String> ids = parseIds(entity);
        final int at = ids.indexOf(parentId);
        assertTrue("a live parent is missing: " + parentId, at >= 0);
        return parseScores(entity).get(at);
    }
}

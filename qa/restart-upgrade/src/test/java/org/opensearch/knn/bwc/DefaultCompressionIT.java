/*
 *  Copyright OpenSearch Contributors
 *  SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.bwc;

import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.Mode;

import java.util.Map;

import static org.opensearch.knn.TestUtils.KNN_VECTOR;
import static org.opensearch.knn.TestUtils.NODES_BWC_CLUSTER;
import static org.opensearch.knn.TestUtils.PROPERTIES;
import static org.opensearch.knn.TestUtils.VECTOR_TYPE;
import static org.opensearch.knn.common.KNNConstants.COMPRESSION_LEVEL_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.DIMENSION;
import static org.opensearch.knn.common.KNNConstants.FAISS_NAME;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.MODE_PARAMETER;

public class DefaultCompressionIT extends AbstractRestartUpgradeTestCase {
    private static final String TEST_FIELD = "test-field";
    private static final int DIMENSIONS = 5;
    private static final int K = 5;
    private static final int NUM_DOCS = 10;

    @SuppressWarnings("unchecked")
    public void testRestartUpgrade_defaultCompression() throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);
        final String explicitX32Index = testIndex + "-explicit-x32";

        if (isRunningAgainstOldCluster()) {
            String explicitMapping = XContentFactory.jsonBuilder()
                .startObject()
                .startObject(PROPERTIES)
                .startObject(TEST_FIELD)
                .field(VECTOR_TYPE, KNN_VECTOR)
                .field(DIMENSION, DIMENSIONS)
                .field(COMPRESSION_LEVEL_PARAMETER, CompressionLevel.x32.getName())
                .field(MODE_PARAMETER, Mode.ON_DISK.getName())
                .endObject()
                .endObject()
                .endObject()
                .toString();
            createKnnIndex(explicitX32Index, getKNNDefaultIndexSettings(), explicitMapping);
            addKNNDocs(explicitX32Index, TEST_FIELD, DIMENSIONS, 0, NUM_DOCS);

            createKnnIndex(testIndex, getKNNDefaultIndexSettings(), createKnnIndexMapping(TEST_FIELD, DIMENSIONS, METHOD_HNSW, FAISS_NAME));
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 0, NUM_DOCS);
            flush(testIndex, true);
            flush(explicitX32Index, true);
        } else {
            validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, K);
            validateKNNSearch(explicitX32Index, TEST_FIELD, DIMENSIONS, NUM_DOCS, K);

            Map<String, Object> defaultMappings = getIndexMappingAsMap(testIndex);
            Map<String, Object> defaultFieldProps = (Map<String, Object>) ((Map<String, Object>) defaultMappings.get(PROPERTIES)).get(
                TEST_FIELD
            );
            assertNull(defaultFieldProps.get(COMPRESSION_LEVEL_PARAMETER));

            Map<String, Object> x32Mappings = getIndexMappingAsMap(explicitX32Index);
            Map<String, Object> x32FieldProps = (Map<String, Object>) ((Map<String, Object>) x32Mappings.get(PROPERTIES)).get(TEST_FIELD);
            assertEquals(CompressionLevel.x32.getName(), x32FieldProps.get(COMPRESSION_LEVEL_PARAMETER));
            assertEquals(Mode.ON_DISK.getName(), x32FieldProps.get(MODE_PARAMETER));

            // TODO: [DEFAULT_FLIP] After Step 4, add test that post-restart NEW index creation (implicit) defaults to SQ 1-bit

            deleteKNNIndex(testIndex);
            deleteKNNIndex(explicitX32Index);
        }
    }
}

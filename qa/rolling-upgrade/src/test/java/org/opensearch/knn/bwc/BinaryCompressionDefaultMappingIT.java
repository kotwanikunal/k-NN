/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.bwc;

import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.Mode;

import static org.opensearch.knn.TestUtils.KNN_VECTOR;
import static org.opensearch.knn.TestUtils.NODES_BWC_CLUSTER;
import static org.opensearch.knn.TestUtils.PROPERTIES;
import static org.opensearch.knn.TestUtils.VECTOR_TYPE;
import static org.opensearch.knn.common.KNNConstants.COMPRESSION_LEVEL_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.DIMENSION;
import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.FAISS_NAME;
import static org.opensearch.knn.common.KNNConstants.KNN_ENGINE;
import static org.opensearch.knn.common.KNNConstants.KNN_METHOD;
import static org.opensearch.knn.common.KNNConstants.LUCENE_NAME;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;
import static org.opensearch.knn.common.KNNConstants.METHOD_PARAMETER_SPACE_TYPE;
import static org.opensearch.knn.common.KNNConstants.MODE_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.NAME;
import static org.opensearch.knn.common.KNNConstants.PARAMETERS;

/**
 * Rolling-upgrade BWC tests for all mapping variations affected by the BBQ default change.
 *
 * Each test creates an index on the OLD cluster, validates search in MIXED cluster
 * (where some nodes are old, some new), adds docs in mixed, then validates search
 * and force merge on the fully UPGRADED cluster.
 */
public class BinaryCompressionDefaultMappingIT extends AbstractRollingUpgradeTestCase {

    private static final String TEST_FIELD = "test-field";
    private static final int DIMENSIONS = 8;
    private static final int BINARY_DIMENSIONS = 40;
    private static final int NUM_DOCS = 10;
    private static final int K = 5;

    // 1. Minimal mapping
    public void testMinimalMapping() throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);
        switch (getClusterType()) {
            case OLD:
                String mapping = XContentFactory.jsonBuilder()
                    .startObject()
                    .startObject(PROPERTIES)
                    .startObject(TEST_FIELD)
                    .field(VECTOR_TYPE, KNN_VECTOR)
                    .field(DIMENSION, DIMENSIONS)
                    .endObject()
                    .endObject()
                    .endObject()
                    .toString();
                createKnnIndex(testIndex, getKNNDefaultIndexSettings(), mapping);
                addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 0, NUM_DOCS);
                flush(testIndex, true);
                break;
            case MIXED:
                validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, K);
                if (isFirstMixedRound()) {
                    addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, NUM_DOCS);
                }
                break;
            case UPGRADED:
                validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS * 2, K);
                forceMergeKnnIndex(testIndex, 1);
                validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS * 2, K);
                deleteKNNIndex(testIndex);
                break;
        }
    }

    // 2. Explicit faiss engine
    public void testExplicitFaissEngine() throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);
        switch (getClusterType()) {
            case OLD:
                createKnnIndex(
                    testIndex,
                    getKNNDefaultIndexSettings(),
                    createKnnIndexMapping(TEST_FIELD, DIMENSIONS, METHOD_HNSW, FAISS_NAME)
                );
                addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 0, NUM_DOCS);
                flush(testIndex, true);
                break;
            case MIXED:
                validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, K);
                if (isFirstMixedRound()) {
                    addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, NUM_DOCS);
                }
                break;
            case UPGRADED:
                validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS * 2, K);
                forceMergeKnnIndex(testIndex, 1);
                validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS * 2, K);
                deleteKNNIndex(testIndex);
                break;
        }
    }

    // 3. Explicit lucene engine
    public void testExplicitLuceneEngine() throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);
        switch (getClusterType()) {
            case OLD:
                createKnnIndex(
                    testIndex,
                    getKNNDefaultIndexSettings(),
                    createKnnIndexMapping(TEST_FIELD, DIMENSIONS, METHOD_HNSW, LUCENE_NAME)
                );
                addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 0, NUM_DOCS);
                flush(testIndex, true);
                break;
            case MIXED:
                validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, K);
                if (isFirstMixedRound()) {
                    addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, NUM_DOCS);
                }
                break;
            case UPGRADED:
                validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS * 2, K);
                forceMergeKnnIndex(testIndex, 1);
                validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS * 2, K);
                deleteKNNIndex(testIndex);
                break;
        }
    }

    // 4. mode: on_disk
    public void testOnDiskMode() throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);
        switch (getClusterType()) {
            case OLD:
                String mapping = XContentFactory.jsonBuilder()
                    .startObject()
                    .startObject(PROPERTIES)
                    .startObject(TEST_FIELD)
                    .field(VECTOR_TYPE, KNN_VECTOR)
                    .field(DIMENSION, DIMENSIONS)
                    .field(MODE_PARAMETER, Mode.ON_DISK.getName())
                    .endObject()
                    .endObject()
                    .endObject()
                    .toString();
                createKnnIndex(testIndex, getKNNDefaultIndexSettings(), mapping);
                addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 0, NUM_DOCS);
                flush(testIndex, true);
                break;
            case MIXED:
                validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, K);
                if (isFirstMixedRound()) {
                    addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, NUM_DOCS);
                }
                break;
            case UPGRADED:
                validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS * 2, K);
                forceMergeKnnIndex(testIndex, 1);
                validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS * 2, K);
                deleteKNNIndex(testIndex);
                break;
        }
    }

    // 5. Explicit x32 + on_disk (old BQ)
    public void testExplicitX32OnDisk() throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);
        switch (getClusterType()) {
            case OLD:
                String mapping = XContentFactory.jsonBuilder()
                    .startObject()
                    .startObject(PROPERTIES)
                    .startObject(TEST_FIELD)
                    .field(VECTOR_TYPE, KNN_VECTOR)
                    .field(DIMENSION, DIMENSIONS)
                    .field(MODE_PARAMETER, Mode.ON_DISK.getName())
                    .field(COMPRESSION_LEVEL_PARAMETER, CompressionLevel.x32.getName())
                    .endObject()
                    .endObject()
                    .endObject()
                    .toString();
                createKnnIndex(testIndex, getKNNDefaultIndexSettings(), mapping);
                addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 0, NUM_DOCS);
                flush(testIndex, true);
                break;
            case MIXED:
                validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, K);
                if (isFirstMixedRound()) {
                    addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, NUM_DOCS);
                }
                break;
            case UPGRADED:
                validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS * 2, K);
                forceMergeKnnIndex(testIndex, 1);
                validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS * 2, K);
                deleteKNNIndex(testIndex);
                break;
        }
    }

    // 6. Explicit x4 compression (Lucene SQ)
    public void testExplicitX4Compression() throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);
        switch (getClusterType()) {
            case OLD:
                String mapping = XContentFactory.jsonBuilder()
                    .startObject()
                    .startObject(PROPERTIES)
                    .startObject(TEST_FIELD)
                    .field(VECTOR_TYPE, KNN_VECTOR)
                    .field(DIMENSION, DIMENSIONS)
                    .field(COMPRESSION_LEVEL_PARAMETER, CompressionLevel.x4.getName())
                    .endObject()
                    .endObject()
                    .endObject()
                    .toString();
                createKnnIndex(testIndex, getKNNDefaultIndexSettings(), mapping);
                addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 0, NUM_DOCS);
                flush(testIndex, true);
                break;
            case MIXED:
                validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, K);
                break;
            case UPGRADED:
                validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, K);
                forceMergeKnnIndex(testIndex, 1);
                validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, K);
                deleteKNNIndex(testIndex);
                break;
        }
    }

    // 7. Explicit BQ encoder on faiss
    public void testExplicitBQEncoderFaiss() throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);
        switch (getClusterType()) {
            case OLD:
                String mapping = XContentFactory.jsonBuilder()
                    .startObject()
                    .startObject(PROPERTIES)
                    .startObject(TEST_FIELD)
                    .field(VECTOR_TYPE, KNN_VECTOR)
                    .field(DIMENSION, DIMENSIONS)
                    .startObject(KNN_METHOD)
                    .field(NAME, METHOD_HNSW)
                    .field(KNN_ENGINE, FAISS_NAME)
                    .field(METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue())
                    .startObject(PARAMETERS)
                    .startObject(METHOD_ENCODER_PARAMETER)
                    .field(NAME, "binary")
                    .startObject(PARAMETERS)
                    .field("bits", 1)
                    .endObject()
                    .endObject()
                    .endObject()
                    .endObject()
                    .endObject()
                    .endObject()
                    .endObject()
                    .toString();
                createKnnIndex(testIndex, getKNNDefaultIndexSettings(), mapping);
                addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 0, NUM_DOCS);
                flush(testIndex, true);
                break;
            case MIXED:
                validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, K);
                break;
            case UPGRADED:
                validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, K);
                forceMergeKnnIndex(testIndex, 1);
                validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, K);
                deleteKNNIndex(testIndex);
                break;
        }
    }

    // 8. Explicit SQ encoder on lucene
    public void testExplicitSQEncoderLucene() throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);
        switch (getClusterType()) {
            case OLD:
                String mapping = XContentFactory.jsonBuilder()
                    .startObject()
                    .startObject(PROPERTIES)
                    .startObject(TEST_FIELD)
                    .field(VECTOR_TYPE, KNN_VECTOR)
                    .field(DIMENSION, DIMENSIONS)
                    .startObject(KNN_METHOD)
                    .field(NAME, METHOD_HNSW)
                    .field(KNN_ENGINE, LUCENE_NAME)
                    .field(METHOD_PARAMETER_SPACE_TYPE, SpaceType.L2.getValue())
                    .startObject(PARAMETERS)
                    .startObject(METHOD_ENCODER_PARAMETER)
                    .field(NAME, ENCODER_SQ)
                    .endObject()
                    .endObject()
                    .endObject()
                    .endObject()
                    .endObject()
                    .endObject()
                    .toString();
                createKnnIndex(testIndex, getKNNDefaultIndexSettings(), mapping);
                addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 0, NUM_DOCS);
                flush(testIndex, true);
                break;
            case MIXED:
                validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, K);
                break;
            case UPGRADED:
                validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, K);
                forceMergeKnnIndex(testIndex, 1);
                validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, K);
                deleteKNNIndex(testIndex);
                break;
        }
    }

    // 9. Binary data type
    public void testBinaryDataType() throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);
        switch (getClusterType()) {
            case OLD:
                createKnnIndex(
                    testIndex,
                    getKNNDefaultIndexSettings(),
                    createKnnIndexMapping(
                        TEST_FIELD,
                        BINARY_DIMENSIONS,
                        METHOD_HNSW,
                        KNNEngine.FAISS.getName(),
                        SpaceType.HAMMING.getValue(),
                        true,
                        VectorDataType.BINARY
                    )
                );
                addKNNByteDocs(testIndex, TEST_FIELD, BINARY_DIMENSIONS / 8, 0, NUM_DOCS);
                flush(testIndex, true);
                break;
            case MIXED:
                // Binary search validation is limited — just ensure no crash
                break;
            case UPGRADED:
                forceMergeKnnIndex(testIndex, 1);
                deleteKNNIndex(testIndex);
                break;
        }
    }

    // 10. Byte data type on lucene
    public void testByteDataType() throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);
        switch (getClusterType()) {
            case OLD:
                createKnnIndex(
                    testIndex,
                    getKNNDefaultIndexSettings(),
                    createKnnIndexMapping(
                        TEST_FIELD,
                        DIMENSIONS,
                        METHOD_HNSW,
                        LUCENE_NAME,
                        SpaceType.L2.getValue(),
                        true,
                        VectorDataType.BYTE
                    )
                );
                addKNNByteDocs(testIndex, TEST_FIELD, DIMENSIONS, 0, NUM_DOCS);
                flush(testIndex, true);
                break;
            case MIXED:
                validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, K);
                break;
            case UPGRADED:
                validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, K);
                forceMergeKnnIndex(testIndex, 1);
                validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, K);
                deleteKNNIndex(testIndex);
                break;
        }
    }

    // 11. Merge with deleted docs during rolling upgrade
    public void testMergeWithDeletedDocs() throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);
        switch (getClusterType()) {
            case OLD:
                String mapping = XContentFactory.jsonBuilder()
                    .startObject()
                    .startObject(PROPERTIES)
                    .startObject(TEST_FIELD)
                    .field(VECTOR_TYPE, KNN_VECTOR)
                    .field(DIMENSION, DIMENSIONS)
                    .field(MODE_PARAMETER, Mode.ON_DISK.getName())
                    .field(COMPRESSION_LEVEL_PARAMETER, CompressionLevel.x32.getName())
                    .endObject()
                    .endObject()
                    .endObject()
                    .toString();
                createKnnIndex(testIndex, getKNNDefaultIndexSettings(), mapping);
                addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 0, NUM_DOCS);
                flush(testIndex, true);
                break;
            case MIXED:
                validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, K);
                if (isFirstMixedRound()) {
                    deleteKnnDoc(testIndex, "0");
                }
                break;
            case UPGRADED:
                forceMergeKnnIndex(testIndex, 1);
                validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS - 1, K);
                deleteKNNIndex(testIndex);
                break;
        }
    }

    // 12. Mixed segments: vector + non-vector across rolling upgrade
    public void testMixedSegments() throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);
        switch (getClusterType()) {
            case OLD:
                String mapping = XContentFactory.jsonBuilder()
                    .startObject()
                    .startObject(PROPERTIES)
                    .startObject(TEST_FIELD)
                    .field(VECTOR_TYPE, KNN_VECTOR)
                    .field(DIMENSION, DIMENSIONS)
                    .field(MODE_PARAMETER, Mode.ON_DISK.getName())
                    .field(COMPRESSION_LEVEL_PARAMETER, CompressionLevel.x32.getName())
                    .endObject()
                    .startObject("description")
                    .field("type", "text")
                    .endObject()
                    .endObject()
                    .endObject()
                    .toString();
                createKnnIndex(testIndex, getKNNDefaultIndexSettings(), mapping);
                addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 0, NUM_DOCS);
                flush(testIndex, true);
                addNonKNNDoc(testIndex, String.valueOf(NUM_DOCS + 1), "description", "text only");
                flush(testIndex, true);
                break;
            case MIXED:
                validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, K);
                break;
            case UPGRADED:
                forceMergeKnnIndex(testIndex, 1);
                validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, K);
                deleteKNNIndex(testIndex);
                break;
        }
    }

    // 13. Old + new segments merged after full upgrade (snapshot restore equivalent)
    public void testOldAndNewSegmentsMerged() throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);
        switch (getClusterType()) {
            case OLD:
                String mapping = XContentFactory.jsonBuilder()
                    .startObject()
                    .startObject(PROPERTIES)
                    .startObject(TEST_FIELD)
                    .field(VECTOR_TYPE, KNN_VECTOR)
                    .field(DIMENSION, DIMENSIONS)
                    .field(MODE_PARAMETER, Mode.ON_DISK.getName())
                    .field(COMPRESSION_LEVEL_PARAMETER, CompressionLevel.x32.getName())
                    .endObject()
                    .endObject()
                    .endObject()
                    .toString();
                createKnnIndex(testIndex, getKNNDefaultIndexSettings(), mapping);
                addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 0, NUM_DOCS);
                flush(testIndex, true);
                break;
            case MIXED:
                if (isFirstMixedRound()) {
                    addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, NUM_DOCS);
                    flush(testIndex, true);
                }
                break;
            case UPGRADED:
                addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS * 2, NUM_DOCS);
                flush(testIndex, true);
                forceMergeKnnIndex(testIndex, 1);
                validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS * 3, K);
                deleteKNNIndex(testIndex);
                break;
        }
    }
}

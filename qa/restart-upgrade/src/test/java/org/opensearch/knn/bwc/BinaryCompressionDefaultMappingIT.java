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
 * Restart-upgrade BWC tests for all mapping variations affected by the BBQ default change.
 *
 * Each test creates an index on the old cluster with a specific mapping variation,
 * then validates search, doc addition, and force merge work after restart upgrade.
 * This ensures that indices created before 3.6.0 (which use old defaults) remain
 * fully functional when opened on 3.6.0+ nodes with new defaults.
 */
public class BinaryCompressionDefaultMappingIT extends AbstractRestartUpgradeTestCase {

    private static final String TEST_FIELD = "test-field";
    private static final int DIMENSIONS = 8;
    private static final int BINARY_DIMENSIONS = 40;
    private static final int NUM_DOCS = 10;
    private static final int K = 5;

    // 1. Minimal mapping: just type + dimension (pre-3.6.0 = flat/x1/nmslib or faiss)
    public void testMinimalMapping() throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);
        if (isRunningAgainstOldCluster()) {
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
        } else {
            validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, K);
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, NUM_DOCS);
            forceMergeKnnIndex(testIndex, 1);
            validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS * 2, K);
            deleteKNNIndex(testIndex);
        }
    }

    // 2. Explicit faiss engine, no compression/mode
    public void testExplicitFaissEngine() throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);
        if (isRunningAgainstOldCluster()) {
            createKnnIndex(testIndex, getKNNDefaultIndexSettings(), createKnnIndexMapping(TEST_FIELD, DIMENSIONS, METHOD_HNSW, FAISS_NAME));
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 0, NUM_DOCS);
            flush(testIndex, true);
        } else {
            validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, K);
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, NUM_DOCS);
            forceMergeKnnIndex(testIndex, 1);
            validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS * 2, K);
            deleteKNNIndex(testIndex);
        }
    }

    // 3. Explicit lucene engine, no compression/mode
    public void testExplicitLuceneEngine() throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);
        if (isRunningAgainstOldCluster()) {
            createKnnIndex(
                testIndex,
                getKNNDefaultIndexSettings(),
                createKnnIndexMapping(TEST_FIELD, DIMENSIONS, METHOD_HNSW, LUCENE_NAME)
            );
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 0, NUM_DOCS);
            flush(testIndex, true);
        } else {
            validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, K);
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, NUM_DOCS);
            forceMergeKnnIndex(testIndex, 1);
            validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS * 2, K);
            deleteKNNIndex(testIndex);
        }
    }

    // 4. mode: on_disk, no compression — pre-3.6.0 uses BQ/x32
    public void testOnDiskMode() throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);
        if (isRunningAgainstOldCluster()) {
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
        } else {
            validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, K);
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, NUM_DOCS);
            forceMergeKnnIndex(testIndex, 1);
            validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS * 2, K);
            deleteKNNIndex(testIndex);
        }
    }

    // 5. mode: in_memory, no compression
    public void testInMemoryMode() throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);
        if (isRunningAgainstOldCluster()) {
            String mapping = XContentFactory.jsonBuilder()
                .startObject()
                .startObject(PROPERTIES)
                .startObject(TEST_FIELD)
                .field(VECTOR_TYPE, KNN_VECTOR)
                .field(DIMENSION, DIMENSIONS)
                .field(MODE_PARAMETER, Mode.IN_MEMORY.getName())
                .field(COMPRESSION_LEVEL_PARAMETER, CompressionLevel.x1.getName())
                .endObject()
                .endObject()
                .endObject()
                .toString();
            createKnnIndex(testIndex, getKNNDefaultIndexSettings(), mapping);
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, 0, NUM_DOCS);
            flush(testIndex, true);
        } else {
            validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, K);
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, NUM_DOCS);
            forceMergeKnnIndex(testIndex, 1);
            validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS * 2, K);
            deleteKNNIndex(testIndex);
        }
    }

    // 6. Explicit x32 + on_disk — old cluster uses BQ (QFrameBitEncoder)
    public void testExplicitX32OnDisk() throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);
        if (isRunningAgainstOldCluster()) {
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
        } else {
            validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, K);
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, NUM_DOCS);
            forceMergeKnnIndex(testIndex, 1);
            validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS * 2, K);
            deleteKNNIndex(testIndex);
        }
    }

    // 7. Explicit x4 compression — Lucene SQ, should stay SQ after upgrade
    public void testExplicitX4Compression() throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);
        if (isRunningAgainstOldCluster()) {
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
        } else {
            validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, K);
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, NUM_DOCS);
            forceMergeKnnIndex(testIndex, 1);
            validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS * 2, K);
            deleteKNNIndex(testIndex);
        }
    }

    // 8. Explicit BQ encoder (binary, bits:1) on faiss
    public void testExplicitBQEncoderFaiss() throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);
        if (isRunningAgainstOldCluster()) {
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
        } else {
            validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, K);
            forceMergeKnnIndex(testIndex, 1);
            validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, K);
            deleteKNNIndex(testIndex);
        }
    }

    // 9. Explicit SQ encoder on lucene
    public void testExplicitSQEncoderLucene() throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);
        if (isRunningAgainstOldCluster()) {
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
        } else {
            validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, K);
            forceMergeKnnIndex(testIndex, 1);
            validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, K);
            deleteKNNIndex(testIndex);
        }
    }

    // 10. Binary data type — should not get compression, unaffected by BBQ defaults
    public void testBinaryDataType() throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);
        if (isRunningAgainstOldCluster()) {
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
        } else {
            forceMergeKnnIndex(testIndex, 1);
            deleteKNNIndex(testIndex);
        }
    }

    // 11. Byte data type — should not get compression, unaffected by BBQ defaults
    public void testByteDataType() throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);
        if (isRunningAgainstOldCluster()) {
            createKnnIndex(
                testIndex,
                getKNNDefaultIndexSettings(),
                createKnnIndexMapping(TEST_FIELD, DIMENSIONS, METHOD_HNSW, LUCENE_NAME, SpaceType.L2.getValue(), true, VectorDataType.BYTE)
            );
            addKNNByteDocs(testIndex, TEST_FIELD, DIMENSIONS, 0, NUM_DOCS);
            flush(testIndex, true);
        } else {
            validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, K);
            forceMergeKnnIndex(testIndex, 1);
            validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, K);
            deleteKNNIndex(testIndex);
        }
    }

    // 12. Merge with deleted docs — validates BBQ handles tombstones across upgrade
    public void testMergeWithDeletedDocs() throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);
        if (isRunningAgainstOldCluster()) {
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
        } else {
            // Delete some docs then force merge — old segments with deletes + new tombstones
            deleteKnnDoc(testIndex, "0");
            deleteKnnDoc(testIndex, "1");
            forceMergeKnnIndex(testIndex, 1);
            validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS - 2, K);
            deleteKNNIndex(testIndex);
        }
    }

    // 13. Mixed segments: vector + non-vector docs across upgrade
    public void testMixedSegmentsAcrossUpgrade() throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);
        if (isRunningAgainstOldCluster()) {
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
            // Second segment with only non-vector doc
            addNonKNNDoc(testIndex, String.valueOf(NUM_DOCS + 1), "description", "text only");
            flush(testIndex, true);
        } else {
            validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, K);
            // Force merge mixed segments
            forceMergeKnnIndex(testIndex, 1);
            validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, K);
            deleteKNNIndex(testIndex);
        }
    }

    // 14. Snapshot restore simulation — create, flush, add more after upgrade, merge
    // (actual snapshot/restore requires cluster-level setup not available in this harness,
    // but this validates the equivalent: old segments + new segments merged post-upgrade)
    public void testOldAndNewSegmentsMergedAfterUpgrade() throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);
        if (isRunningAgainstOldCluster()) {
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
        } else {
            // Add new docs on upgraded cluster — creates new segments with potentially different format
            addKNNDocs(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS, NUM_DOCS);
            flush(testIndex, true);
            // Merge old (pre-upgrade) + new (post-upgrade) segments
            forceMergeKnnIndex(testIndex, 1);
            validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS * 2, K);
            deleteKNNIndex(testIndex);
        }
    }

    // 15. Vector field removal by update — doc had vector, updated to remove it
    public void testVectorFieldRemovalAfterUpgrade() throws Exception {
        waitForClusterHealthGreen(NODES_BWC_CLUSTER);
        if (isRunningAgainstOldCluster()) {
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
        } else {
            // Update a doc to remove vector field — creates segment with doc missing vector
            addNonKNNDoc(testIndex, "0", "description", "updated without vector");
            forceMergeKnnIndex(testIndex, 1);
            // Should find NUM_DOCS - 1 vector docs (doc 0 lost its vector)
            validateKNNSearch(testIndex, TEST_FIELD, DIMENSIONS, NUM_DOCS - 1, K);
            deleteKNNIndex(testIndex);
        }
    }
}

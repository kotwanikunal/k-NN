/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import org.opensearch.cluster.metadata.MappingMetadata;
import org.opensearch.index.mapper.MapperService;
import org.opensearch.knn.KNNTestCase;

import java.util.Map;

/**
 * The buffer-sizing rule, tested at the data points where it was measured. Every expectation with a
 * 4096 block size below was verified against a real {@code .vec} at 200 sorted sparse ordinals: the
 * stated size is the smallest one that achieves exactly 1.00 reads per vector.
 */
public class DirectIOBufferSizerTests extends KNNTestCase {

    private static final int BLOCK = 4096;
    private static final long CLAMP_32K = 32 * 1024L;

    /**
     * The measured table: dim 768 needs 8192, 1024 needs only 4096, 1792 needs 12288.
     */
    public void testRequiredBufferSizeAtTheMeasuredDataPoints() {
        assertEquals(8192, DirectIOBufferSizer.requiredBufferSize(768 * 4, BLOCK));
        assertEquals(4096, DirectIOBufferSizer.requiredBufferSize(1024 * 4, BLOCK));
        assertEquals(8192, DirectIOBufferSizer.requiredBufferSize(1025 * 4, BLOCK));
        assertEquals(8192, DirectIOBufferSizer.requiredBufferSize(1280 * 4, BLOCK));
        assertEquals(8192, DirectIOBufferSizer.requiredBufferSize(1536 * 4, BLOCK));
        assertEquals(12288, DirectIOBufferSizer.requiredBufferSize(1792 * 4, BLOCK));
        assertEquals(69632, DirectIOBufferSizer.requiredBufferSize(16000 * 4, BLOCK));
    }

    /**
     * The {@code gcd} term is what makes 1025-dim and 1536-dim come out at 8192. A
     * {@code blockSize - 1} rule — the obvious wrong implementation — gives 12288 for both, so this
     * pair is the test that the gcd is actually there.
     */
    public void testGcdTermIsImplemented() {
        assertNotEquals(
            "a blockSize-1 rule would give 12288 for 1025 dimensions",
            12288,
            DirectIOBufferSizer.requiredBufferSize(1025 * 4, BLOCK)
        );
        assertNotEquals(
            "a blockSize-1 rule would give 12288 for 1536 dimensions",
            12288,
            DirectIOBufferSizer.requiredBufferSize(1536 * 4, BLOCK)
        );
        // and for a vector whose size shares no factor with the block, the gcd is 1 and the two
        // rules do agree
        assertEquals(8192, DirectIOBufferSizer.requiredBufferSize(4095, BLOCK));
    }

    /**
     * A vector shorter than a block never needs more than two blocks, and a block-aligned one never
     * needs more than its own length.
     */
    public void testRequiredBufferSizeIsAlwaysAtLeastOneBlockAndAMultipleOfIt() {
        for (int vectorBytes = 1; vectorBytes <= 20_000; vectorBytes++) {
            final int size = DirectIOBufferSizer.requiredBufferSize(vectorBytes, BLOCK);
            assertTrue("not a multiple of the block size for " + vectorBytes, size % BLOCK == 0);
            assertTrue("smaller than a block for " + vectorBytes, size >= BLOCK);
            assertTrue("cannot serve one vector for " + vectorBytes, size >= vectorBytes);
        }
    }

    public void testRequiredBufferSizeHonoursANonDefaultBlockSize() {
        // 512-byte blocks: a 3072-byte vector is block aligned, so one block-sized read serves it
        assertEquals(3072, DirectIOBufferSizer.requiredBufferSize(3072, 512));
        // 16 KiB blocks: a 3072-byte vector can start 15360 bytes in, so it needs two blocks
        assertEquals(32768, DirectIOBufferSizer.requiredBufferSize(3072, 16384));
    }

    public void testSingleFloatFieldMapping() {
        assertEquals(8192, DirectIOBufferSizer.readBufferSize(mapping(knnVector(768, null)), BLOCK, CLAMP_32K));
    }

    /**
     * The largest field wins, because one {@code .vec} holds every field's vectors and one buffer
     * serves the whole directory.
     */
    public void testMultipleFieldsTakeTheMaximum() {
        final MappingMetadata mapping = mappingOf(
            Map.of("small", knnVector(768, null), "large", knnVector(1792, null), "text", Map.of("type", "text"))
        );
        assertEquals(12288, DirectIOBufferSizer.readBufferSize(mapping, BLOCK, CLAMP_32K));
    }

    /**
     * A {@code knn_vector} nested inside an object field must still be found.
     */
    public void testNestedFieldIsFound() {
        final Map<String, Object> parent = Map.of("type", "nested", "properties", Map.of("vector", knnVector(1792, null)));
        assertEquals(12288, DirectIOBufferSizer.readBufferSize(mappingOf(Map.of("parent", parent)), BLOCK, CLAMP_32K));
    }

    public void testByteDataType() {
        // 4096 byte-typed dimensions are 4096 bytes on disk, i.e. block aligned
        assertEquals(4096, DirectIOBufferSizer.readBufferSize(mapping(knnVector(4096, "byte")), BLOCK, CLAMP_32K));
        // 768 byte-typed dimensions are 768 bytes on disk, so two blocks
        assertEquals(8192, DirectIOBufferSizer.readBufferSize(mapping(knnVector(768, "byte")), BLOCK, CLAMP_32K));
    }

    public void testBinaryDataType() {
        // binary dimensions are bits: 32768 bits are 4096 bytes on disk
        assertEquals(4096, DirectIOBufferSizer.readBufferSize(mapping(knnVector(32768, "binary")), BLOCK, CLAMP_32K));
        // 3072 bits are 384 bytes, which straddles a block boundary
        assertEquals(8192, DirectIOBufferSizer.readBufferSize(mapping(knnVector(3072, "binary")), BLOCK, CLAMP_32K));
    }

    /**
     * At the legal maximum dimension the required buffer is 69632 bytes, which would be allocated
     * per clone — per segment per query thread. The clamp bounds that; the price is one extra
     * syscall per vector, which is the intended behaviour.
     */
    public void testClampAtTheLegalMaximumDimension() {
        assertEquals(32768, DirectIOBufferSizer.readBufferSize(mapping(knnVector(16000, null)), BLOCK, CLAMP_32K));
    }

    /**
     * A clamp that is not a multiple of the block size is aligned down, and one below a block is
     * raised to a block — Lucene aligns the buffer to a block and cannot honour anything else.
     */
    public void testClampIsBlockAlignedWithABlockFloor() {
        assertEquals(8192, DirectIOBufferSizer.readBufferSize(mapping(knnVector(16000, null)), BLOCK, 12287));
        assertEquals(BLOCK, DirectIOBufferSizer.readBufferSize(mapping(knnVector(16000, null)), BLOCK, 100));
    }

    /**
     * The clamp never inflates a buffer the mapping does not need.
     */
    public void testClampDoesNotInflateASmallBuffer() {
        assertEquals(8192, DirectIOBufferSizer.readBufferSize(mapping(knnVector(768, null)), BLOCK, 1024 * 1024L));
    }

    /**
     * Everything that leaves us with no resolvable dimension falls back to two blocks — today's
     * behaviour before this class existed.
     */
    public void testFallbackWhenNoDimensionIsResolvable() {
        assertEquals("null mapping", 2 * BLOCK, DirectIOBufferSizer.readBufferSize((MappingMetadata) null, BLOCK, CLAMP_32K));
        assertEquals("empty mapping", 2 * BLOCK, DirectIOBufferSizer.readBufferSize(mappingOf(Map.of()), BLOCK, CLAMP_32K));
        assertEquals(
            "no knn_vector field",
            2 * BLOCK,
            DirectIOBufferSizer.readBufferSize(mappingOf(Map.of("text", Map.of("type", "text"))), BLOCK, CLAMP_32K)
        );
        assertEquals(
            "unknown data_type",
            2 * BLOCK,
            DirectIOBufferSizer.readBufferSize(mapping(knnVector(768, "float16")), BLOCK, CLAMP_32K)
        );
        assertEquals(
            "non-numeric dimension",
            2 * BLOCK,
            DirectIOBufferSizer.readBufferSize(mappingOf(Map.of("v", Map.of("type", "knn_vector", "dimension", "768"))), BLOCK, CLAMP_32K)
        );
    }

    /**
     * A model-based field declares its dimension in the model, not in the mapping. It must be
     * skipped rather than read as dimension 0, which would size the buffer for the wrong field.
     */
    public void testModelBasedFieldWithNoDimensionIsSkipped() {
        final Map<String, Object> modelBased = Map.of("type", "knn_vector", "model_id", "my-model");
        assertEquals(2 * BLOCK, DirectIOBufferSizer.readBufferSize(mappingOf(Map.of("v", modelBased)), BLOCK, CLAMP_32K));

        // and a sibling field that does declare one still decides the size
        assertEquals(
            8192,
            DirectIOBufferSizer.readBufferSize(mappingOf(Map.of("v", modelBased, "w", knnVector(768, null))), BLOCK, CLAMP_32K)
        );
    }

    private static Map<String, Object> knnVector(final int dimension, final String dataType) {
        if (dataType == null) {
            return Map.of("type", "knn_vector", "dimension", dimension);
        }
        return Map.of("type", "knn_vector", "dimension", dimension, "data_type", dataType);
    }

    private static MappingMetadata mapping(final Map<String, Object> singleField) {
        return mappingOf(Map.of("vector", singleField));
    }

    private static MappingMetadata mappingOf(final Map<String, Object> properties) {
        return new MappingMetadata(MapperService.SINGLE_MAPPING_NAME, Map.of("properties", properties));
    }
}

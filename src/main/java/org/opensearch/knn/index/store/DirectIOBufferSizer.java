/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import lombok.extern.log4j.Log4j2;
import org.opensearch.cluster.metadata.MappingMetadata;
import org.opensearch.index.IndexSettings;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.VectorDataType;

import java.util.Map;

/**
 * Derives the Direct I/O read buffer size for one index from its mapping.
 * <p>
 * Lucene's {@code DirectIOIndexInput} aligns every read down to a filesystem block and then refills
 * exactly one buffer capacity, so a vector at ordinal {@code n} starts
 * {@code delta = (n * vectorBytes) % blockSize} bytes into the buffer. Serving any vector in a
 * single syscall therefore needs {@code bufferSize >= delta_max + vectorBytes}, and because
 * {@code delta} only ever takes multiples of {@code gcd(vectorBytes, blockSize)}:
 *
 * <pre>
 *   delta_max       = blockSize - gcd(vectorBytes, blockSize)      // not blockSize - 1
 *   requiredBufSize = alignUp(delta_max + vectorBytes, blockSize)
 * </pre>
 *
 * The {@code gcd} term is load bearing rather than a refinement: a {@code blockSize - 1} rule
 * predicts that 1025-dim and 1536-dim float vectors need 12288 bytes at a 4096-byte block size,
 * when both are in fact served in one read from 8192.
 * <p>
 * No flat constant works. 8192 is too small for a 1792-dim float field (needs 12288) and twice as
 * large as necessary for a 1024-dim one (whose vectors are block aligned, so 4096 gives one read
 * per vector with zero amplification), and this plugin permits dimensions up to 16,000, whose
 * required buffer is 69632 bytes. Since the buffer is allocated per {@code IndexInput} clone — that
 * is, per segment per query thread — sizing every index for the legal maximum is not an option;
 * the size is therefore computed per index here, from the dimensions the mapping declares.
 * <p>
 * Above {@link KNNSettings#getDirectIOMaxBufferSize()} the size is clamped, and the cost of the
 * clamp is one extra syscall per vector, not incorrect reads. That is the intended behaviour and
 * not a bug: amplification is roughly {@code 1 + blockSize / vectorBytes}, so large vectors were
 * never the amplification problem — for them the pressure is buffer size and direct memory.
 * <p>
 * When the mapping declares no usable {@code knn_vector} dimension the result is
 * {@code 2 * blockSize}, the smallest size that reads any sub-block vector in one syscall however
 * it straddles a block boundary.
 */
@Log4j2
final class DirectIOBufferSizer {

    private DirectIOBufferSizer() {}

    /**
     * The read buffer size for the index these settings belong to, in bytes. The clamp comes from
     * the node setting {@code knn.direct_io.max_buffer_size}.
     *
     * @param indexSettings the settings the shard is being opened with
     * @param blockSize     the filesystem block size, supplied by the caller rather than read here
     *                      so that this class does no filesystem I/O and cannot throw
     * @return a positive multiple of {@code blockSize}
     */
    static int readBufferSize(final IndexSettings indexSettings, final int blockSize) {
        return readBufferSize(indexSettings.getIndexMetadata().mapping(), blockSize, KNNSettings.getDirectIOMaxBufferSize().getBytes());
    }

    /**
     * The read buffer size for a mapping, as a pure function of its three inputs.
     *
     * @param mapping       the index mapping, or null when the index has none
     * @param blockSize     the filesystem block size in bytes
     * @param maxBufferSize the clamp, in bytes. Aligned down to a block, with a floor of one block,
     *                      so that an operator cannot configure a buffer Lucene could not align.
     * @return a positive multiple of {@code blockSize}
     */
    static int readBufferSize(final MappingMetadata mapping, final int blockSize, final long maxBufferSize) {
        final int fallback = 2 * blockSize;
        final int largestVectorBytes = largestVectorBytes(mapping);
        if (largestVectorBytes <= 0) {
            log.debug("No knn_vector dimension is resolvable from the mapping; using a {} byte Direct I/O read buffer", fallback);
            return fallback;
        }
        final int clamp = Math.max(blockSize, alignDown(Math.min(maxBufferSize, Integer.MAX_VALUE), blockSize));
        return Math.min(requiredBufferSize(largestVectorBytes, blockSize), clamp);
    }

    /**
     * The smallest multiple of {@code blockSize} that serves any single vector of
     * {@code vectorBytes} bytes in one read, whatever its alignment within the file.
     */
    static int requiredBufferSize(final int vectorBytes, final int blockSize) {
        final long maxDelta = blockSize - gcd(vectorBytes, blockSize);
        return Math.max(blockSize, alignUp(maxDelta + vectorBytes, blockSize));
    }

    /**
     * The largest on-disk vector size over every {@code knn_vector} field in the mapping, or 0 when
     * the mapping declares none whose size can be resolved.
     */
    private static int largestVectorBytes(final MappingMetadata mapping) {
        if (mapping == null) {
            return 0;
        }
        try {
            return largestVectorBytes(mapping.sourceAsMap());
        } catch (Exception e) {
            // A mapping we cannot parse must not fail shard open; fall back to the default buffer.
            log.warn("Could not read the mapping to size the Direct I/O read buffer", e);
            return 0;
        }
    }

    /**
     * Walks a mapping node and every map nested under it, so that fields inside {@code properties},
     * inside object fields and inside nested fields are all found without this method having to
     * know which container it is in.
     */
    @SuppressWarnings("unchecked")
    private static int largestVectorBytes(final Map<String, Object> node) {
        int largest = 0;
        if (KNNConstants.TYPE_KNN_VECTOR.equals(node.get(KNNConstants.TYPE))) {
            largest = vectorBytes(node);
        }
        for (final Object value : node.values()) {
            if (value instanceof Map) {
                largest = Math.max(largest, largestVectorBytes((Map<String, Object>) value));
            }
        }
        return largest;
    }

    /**
     * The on-disk size of one vector of a {@code knn_vector} field, or 0 when the field does not
     * declare enough to compute it.
     * <p>
     * A field with no {@code dimension} is skipped rather than read as zero: a model-based field
     * carries its dimension in the model rather than in the mapping, and guessing would size the
     * buffer wrongly for it. Skipping leaves it to the fallback, which costs at most an extra
     * syscall per vector.
     */
    private static int vectorBytes(final Map<String, Object> field) {
        final Object dimension = field.get(KNNConstants.DIMENSION);
        if (dimension instanceof Number == false) {
            return 0;
        }
        final int dimensions = ((Number) dimension).intValue();
        if (dimensions <= 0) {
            return 0;
        }
        final VectorDataType dataType = dataType(field);
        if (dataType == null) {
            return 0;
        }
        final long vectorBytes;
        switch (dataType) {
            case BINARY:
                // dimension counts bits for a binary field
                vectorBytes = dimensions / Byte.SIZE;
                break;
            case BYTE:
                vectorBytes = dimensions;
                break;
            case FLOAT:
                vectorBytes = (long) dimensions * Float.BYTES;
                break;
            default:
                return 0;
        }
        // Guards a hand-written mapping that declares a dimension the mapper would have rejected;
        // the arithmetic downstream is in ints and the clamp would swallow anything this large.
        return vectorBytes > Integer.MAX_VALUE / 2 ? 0 : (int) vectorBytes;
    }

    /**
     * The field's {@code data_type}, defaulting to {@code float} when it is absent, or null when it
     * names something this version does not know.
     */
    private static VectorDataType dataType(final Map<String, Object> field) {
        final Object dataType = field.get(KNNConstants.VECTOR_DATA_TYPE_FIELD);
        if (dataType == null) {
            return KNNConstants.DEFAULT_VECTOR_DATA_TYPE_FIELD;
        }
        for (final VectorDataType candidate : VectorDataType.values()) {
            if (candidate.getValue().equals(dataType)) {
                return candidate;
            }
        }
        return null;
    }

    private static int gcd(final int a, final int b) {
        int x = a;
        int y = b;
        while (y != 0) {
            final int remainder = x % y;
            x = y;
            y = remainder;
        }
        return x;
    }

    private static int alignUp(final long value, final int blockSize) {
        return Math.toIntExact(((value + blockSize - 1) / blockSize) * blockSize);
    }

    private static int alignDown(final long value, final int blockSize) {
        return Math.toIntExact((value / blockSize) * blockSize);
    }
}

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Random;

/**
 * The nested index W4 measures Direct I/O against mmap on: clustered child vectors under synthetic parents, in
 * two child-count shapes and three vector data types, from one recorded seed.
 *
 * <p>Clustered and not uniform, because a nested query's cost is the number of candidate <em>children</em> it
 * scores and uniform vectors make the candidate set noise. Every child is a cluster centre plus a small jitter,
 * so the ranking a query produces is stable and the same documents are the candidates on every run.
 *
 * <p>The two shapes are the measurement: {@link Shape#SMALL} is the common nested mapping, a handful of children
 * per parent, and {@link Shape#LARGE} is the shape that makes one parent group a long run of contiguous child
 * ordinals, which is what a per-group prefetch would coalesce into one read.
 *
 * <p>Only the faiss engine routes to {@code O_DIRECT}, so every mapping here is a faiss mapping: the two codec
 * formats that call the seam are the native-engine ones. {@code float} and {@code half_float} are mapped
 * {@code on_disk} so that their queries carry a rescore pass over the full-precision {@code .vec}; {@code byte}
 * has no {@code on_disk} mode, so its {@code .vec} is read by exact search and by the child expansion instead.
 */
public final class NestedDirectIOFixture {

    /** The one seed every index and query set here comes from. Recorded so a re-run is the same index. */
    public static final long SEED = 20261027L;

    /** The dimension of every field here. 128 so that one fp32 vector is 512 bytes and one FP16 vector 256. */
    public static final int DIMENSION = 128;

    /** Cluster centres. Enough that neighbouring parents are not all one another's candidates. */
    public static final int CENTRES = 16;

    public static final String NESTED_FIELD = "nested";
    public static final String VECTOR_FIELD = "vector";
    /** A parent-level boolean that selects about half the parents. */
    public static final String PARENT_FILTER_FIELD = "parent_match";
    /** A parent-level boolean that selects about one parent in a hundred, which forces exact search. */
    public static final String PARENT_RARE_FIELD = "parent_rare";
    /** A child-level boolean that selects about half the children of every parent. */
    public static final String CHILD_FILTER_FIELD = "child_match";

    /** How many parents, and how many children each of them carries. */
    public enum Shape {
        SMALL(300, 1, 3),
        LARGE(120, 20, 24);

        private final int parents;
        private final int minChildren;
        private final int maxChildren;

        Shape(final int parents, final int minChildren, final int maxChildren) {
            this.parents = parents;
            this.minChildren = minChildren;
            this.maxChildren = maxChildren;
        }

        /** How many parent documents this shape indexes. */
        public int parents() {
            return parents;
        }
    }

    /** The three vector data types of the matrix, each with the faiss mapping that routes. */
    public enum DataType {
        /** fp32 vectors, mapped {@code on_disk} so a query carries a rescore pass. */
        FLOAT,
        /** FP16 vectors, read as raw bytes rather than floats, also mapped {@code on_disk}. */
        HALF_FLOAT,
        /** Signed byte vectors, which have no {@code on_disk} mode and so no rescore pass. */
        BYTE;

        /** Whether a field of this type has a rescore pass to turn off or oversample. */
        public boolean hasRescore() {
            return this != BYTE;
        }
    }

    private NestedDirectIOFixture() {}

    /**
     * The mapping: a nested object holding the knn_vector and the child filter field, plus the two parent filter
     * fields beside it.
     */
    public static String mapping(final DataType dataType) {
        final String vector = switch (dataType) {
            // on_disk resolves float to 32x and half_float to 16x on faiss, so both carry a rescore pass.
            case FLOAT -> String.format(
                Locale.ROOT,
                "{\"type\":\"knn_vector\",\"dimension\":%d,\"mode\":\"on_disk\","
                    + "\"method\":{\"name\":\"hnsw\",\"engine\":\"faiss\",\"space_type\":\"l2\"}}",
                DIMENSION
            );
            case HALF_FLOAT -> String.format(
                Locale.ROOT,
                "{\"type\":\"knn_vector\",\"dimension\":%d,\"data_type\":\"half_float\",\"mode\":\"on_disk\","
                    + "\"method\":{\"name\":\"hnsw\",\"engine\":\"faiss\",\"space_type\":\"l2\"}}",
                DIMENSION
            );
            case BYTE -> String.format(
                Locale.ROOT,
                "{\"type\":\"knn_vector\",\"dimension\":%d,\"data_type\":\"byte\","
                    + "\"method\":{\"name\":\"hnsw\",\"engine\":\"faiss\",\"space_type\":\"l2\"}}",
                DIMENSION
            );
        };
        return String.format(
            Locale.ROOT,
            "{\"properties\":{\"%s\":{\"type\":\"nested\",\"properties\":{\"%s\":%s,\"%s\":{\"type\":\"boolean\"}}},"
                + "\"%s\":{\"type\":\"boolean\"},\"%s\":{\"type\":\"boolean\"}}}",
            NESTED_FIELD,
            VECTOR_FIELD,
            vector,
            CHILD_FILTER_FIELD,
            PARENT_FILTER_FIELD,
            PARENT_RARE_FIELD
        );
    }

    /**
     * One generated index: the documents in index order, the document at position {@code i} belonging under id
     * {@code i}, and how many children each of them carries. The child counts come back with the documents
     * because they are the denominator of every per-read number W4 reports.
     */
    public record Documents(List<String> documents, int[] childCounts, int childrenOfMatchingParents) {
        public int children() {
            int total = 0;
            for (final int count : childCounts) {
                total += count;
            }
            return total;
        }

    }

    /**
     * The documents of one index. Every call with the same arguments returns the same documents, because the
     * only randomness is {@link #SEED}.
     */
    public static Documents documents(final DataType dataType, final Shape shape) throws IOException {
        final Random random = new Random(SEED);
        final float[][] centres = centres(dataType, random);
        final List<String> documents = new ArrayList<>(shape.parents);
        final int[] childCounts = new int[shape.parents];
        int childrenOfMatchingParents = 0;
        for (int parent = 0; parent < shape.parents; parent++) {
            final float[] centre = centres[parent % CENTRES];
            final int children = shape.minChildren + random.nextInt(shape.maxChildren - shape.minChildren + 1);
            childCounts[parent] = children;
            final boolean parentMatches = parent % 2 == 0;
            if (parentMatches) {
                childrenOfMatchingParents += children;
            }
            final NestedKnnDocBuilder builder = NestedKnnDocBuilder.create(NESTED_FIELD);
            for (int child = 0; child < children; child++) {
                builder.addVectorWithMetadata(VECTOR_FIELD, child(dataType, centre, random), CHILD_FILTER_FIELD, child % 2 == 0);
            }
            documents.add(
                builder.addTopLevelField(PARENT_FILTER_FIELD, parentMatches).addTopLevelField(PARENT_RARE_FIELD, parent % 100 == 0).build()
            );
        }
        return new Documents(documents, childCounts, childrenOfMatchingParents);
    }

    /**
     * {@code count} query vectors, one per cluster centre in turn so that every query has candidates, from a
     * seed of their own so that adding a query does not change the index.
     */
    public static List<Object[]> queries(final DataType dataType, final int count) {
        final Random layout = new Random(SEED);
        final float[][] centres = centres(dataType, layout);
        final Random random = new Random(SEED + 1);
        final List<Object[]> queries = new ArrayList<>(count);
        for (int i = 0; i < count; i++) {
            queries.add(child(dataType, centres[i % CENTRES], random));
        }
        return queries;
    }

    /**
     * The centres, drawn from {@code random} before any child is, so that the layout is the same whatever is
     * drawn after it.
     */
    private static float[][] centres(final DataType dataType, final Random random) {
        final float spread = dataType == DataType.BYTE ? 60.0f : 40.0f;
        final float[][] centres = new float[CENTRES][DIMENSION];
        for (final float[] centre : centres) {
            for (int i = 0; i < DIMENSION; i++) {
                centre[i] = Math.round((random.nextFloat() - 0.5f) * 2.0f * spread);
            }
        }
        return centres;
    }

    /**
     * One child of {@code centre}: the centre plus a jitter small against the distance between centres, so a
     * query near one centre takes its candidates from that cluster. {@code byte} values are whole and clamped
     * into the signed byte range; the other two types stay well inside half_float's range.
     */
    private static Object[] child(final DataType dataType, final float[] centre, final Random random) {
        final Object[] vector = new Object[DIMENSION];
        for (int i = 0; i < DIMENSION; i++) {
            final float jitter = (random.nextFloat() - 0.5f) * 4.0f;
            if (dataType == DataType.BYTE) {
                vector[i] = (byte) Math.max(Byte.MIN_VALUE, Math.min(Byte.MAX_VALUE, Math.round(centre[i] + jitter)));
            } else {
                vector[i] = centre[i] + jitter;
            }
        }
        return vector;
    }
}

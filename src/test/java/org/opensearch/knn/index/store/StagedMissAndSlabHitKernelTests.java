/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import com.carrotsearch.randomizedtesting.ThreadFilter;
import com.carrotsearch.randomizedtesting.annotations.ThreadLeakFilters;
import lombok.SneakyThrows;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.codecs.lucene95.HasIndexSlice;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.hnsw.RandomVectorScorerSupplier;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.codec.scorer.PrefetchableFlatVectorScorer;
import org.opensearch.knn.index.memory.SlabVectorCache;
import org.opensearch.knn.index.memory.VectorCacheScope;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Random;

/**
 * I5, proof (c), on the real read path: a vector scored from a slab and the same vector scored from a
 * <b>staged Direct I/O read</b> must give bit-identical scores.
 *
 * <p>The values object here reads through a real {@link DirectIOVectorIndexInput}: the scorer's prefetch
 * declares the misses on it, which stages them through {@code O_DIRECT}, and {@code vectorValue} consumes
 * them from the staging buffer. So the comparison is the one the backlog asks for rather than a
 * reconstruction of it: the first pass is a staged miss and the second is a slab hit, with the same bytes
 * having travelled two completely different ways to reach the kernel.
 *
 * <p>It holds by construction, because both passes score {@code similarityFunction.compare(target, vector)}
 * on a {@code float[]}, and the slab is little-endian for the same reason a {@code .vec} file is. This test
 * is what stops that from quietly stopping being true.
 */
@ThreadLeakFilters(defaultFilters = true, filters = { StagedMissAndSlabHitKernelTests.ReadPoolThreadFilter.class })
public class StagedMissAndSlabHitKernelTests extends KNNTestCase {

    /** The read pool is never shut down by design; its daemon threads outlive any one suite. */
    public static final class ReadPoolThreadFilter implements ThreadFilter {
        @Override
        public boolean reject(final Thread thread) {
            return thread.getName().startsWith(DirectIOReadPool.THREAD_NAME_PREFIX);
        }
    }

    private static final int SLAB_BYTES = 1 << 20;
    private static final int FIELD = 0;

    private SlabVectorCache cache;

    @Override
    public void setUp() throws Exception {
        super.setUp();
        cache = new SlabVectorCache(SlabVectorCache.DIRECT_SOURCE, SLAB_BYTES);
        cache.configure(4L * SLAB_BYTES);
        VectorCacheScope.setNodeCacheForTesting(cache);
    }

    @Override
    public void tearDown() throws Exception {
        VectorCacheScope.setNodeCacheForTesting(null);
        if (cache != null) {
            cache.configure(0L);
            cache = null;
        }
        DirectIOReadPool.resetForTesting();
        DirectIOVectorIndexInput.STATS.reset();
        super.tearDown();
    }

    /** Skips rather than fails where {@code O_DIRECT} is unavailable, as every other test here does. */
    private void assumeDirectIOWorksHere() throws Exception {
        assumeTrue("this JDK has no ExtendedOpenOption.DIRECT", DirectIOVectorIndexInput.directOpenOption() != null);
        final Path probe = createTempDir().resolve("direct-io-probe");
        Files.write(probe, new byte[8192]);
        try (FileChannel channel = FileChannel.open(probe, StandardOpenOption.READ, DirectIOVectorIndexInput.directOpenOption())) {
            channel.read(ByteBuffer.allocateDirect(8192).alignedSlice(4096));
        } catch (Exception e) {
            assumeNoException("this filesystem refuses O_DIRECT", e);
        }
    }

    /** Vectors and nothing else, so that ordinal {@code i} is at {@code i * dimension * 4} as Lucene lays them out. */
    @SneakyThrows
    private Path writeVectorFile(final float[][] vectors) {
        final int dimension = vectors[0].length;
        final Path path = createTempDir().resolve("staged.vec");
        final ByteBuffer buffer = ByteBuffer.allocate(vectors.length * dimension * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (final float[] vector : vectors) {
            for (final float value : vector) {
                buffer.putFloat(value);
            }
        }
        Files.write(path, buffer.array());
        return path;
    }

    /** Values that read one vector at a time through the Direct I/O input, and expose it for the prefetch. */
    private static final class DirectIOFloatVectorValues extends FloatVectorValues implements HasIndexSlice {
        private final DirectIOVectorIndexInput input;
        private final int dimension;
        private final int size;
        private final float[] scratch;
        private int reads;

        private DirectIOFloatVectorValues(final DirectIOVectorIndexInput input, final int dimension, final int size) {
            this.input = input;
            this.dimension = dimension;
            this.size = size;
            this.scratch = new float[dimension];
        }

        @Override
        public float[] vectorValue(final int ord) throws IOException {
            reads++;
            input.seek((long) ord * dimension * Float.BYTES);
            input.readFloats(scratch, 0, dimension);
            return scratch;
        }

        @Override
        public int dimension() {
            return dimension;
        }

        @Override
        public int size() {
            return size;
        }

        @Override
        public FloatVectorValues copy() {
            return this;
        }

        @Override
        public IndexInput getSlice() {
            return input;
        }
    }

    /** Lucene's plain float kernel, which is what the delegate on this path is. */
    private static FlatVectorsScorer plainScorer() {
        return new FlatVectorsScorer() {
            @Override
            public RandomVectorScorerSupplier getRandomVectorScorerSupplier(final VectorSimilarityFunction f, final KnnVectorValues v) {
                throw new UnsupportedOperationException();
            }

            @Override
            public RandomVectorScorer getRandomVectorScorer(
                final VectorSimilarityFunction similarity,
                final KnnVectorValues values,
                final float[] target
            ) {
                return new RandomVectorScorer.AbstractRandomVectorScorer(values) {
                    @Override
                    public float score(final int node) throws IOException {
                        return similarity.compare(target, ((FloatVectorValues) values).vectorValue(node));
                    }
                };
            }

            @Override
            public RandomVectorScorer getRandomVectorScorer(
                final VectorSimilarityFunction similarity,
                final KnnVectorValues values,
                final byte[] target
            ) {
                throw new UnsupportedOperationException();
            }
        };
    }

    @SneakyThrows
    public void testAStagedMissAndASlabHitScoreTheSameBits() {
        assumeDirectIOWorksHere();
        for (final int dimension : new int[] { 128, 768 }) {
            for (final VectorSimilarityFunction similarity : VectorSimilarityFunction.values()) {
                final Random random = new Random(dimension * 31L + similarity.ordinal());
                final float[][] vectors = new float[256][dimension];
                for (final float[] vector : vectors) {
                    for (int i = 0; i < dimension; i++) {
                        vector[i] = random.nextFloat();
                    }
                }
                final float[] target = new float[dimension];
                for (int i = 0; i < dimension; i++) {
                    target[i] = random.nextFloat();
                }
                final Path path = writeVectorFile(vectors);

                try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path)) {
                    assertNotNull("the fixture must be readable with O_DIRECT", input);
                    final DirectIOFloatVectorValues values = new DirectIOFloatVectorValues(input, dimension, vectors.length);
                    final Object core = new Object();
                    VectorCacheScope.open(core, FIELD);
                    final RandomVectorScorer scorer;
                    try {
                        scorer = new PrefetchableFlatVectorScorer(plainScorer()).getRandomVectorScorer(similarity, values, target);
                    } finally {
                        VectorCacheScope.clear();
                    }

                    final int[] nodes = new int[64];
                    for (int i = 0; i < nodes.length; i++) {
                        nodes[i] = i;
                    }
                    final float[] staged = new float[nodes.length];
                    final float[] fromSlab = new float[nodes.length];

                    final long hitsBefore = cache.hits();
                    scorer.bulkScore(nodes, staged, nodes.length);
                    assertEquals("the first pass must be all misses, i.e. staged reads", hitsBefore, cache.hits());
                    final int readsAfterStaged = values.reads;

                    scorer.bulkScore(nodes, fromSlab, nodes.length);
                    assertEquals("the second pass must be all hits", hitsBefore + nodes.length, cache.hits());
                    assertEquals("and must not have read the file again", readsAfterStaged, values.reads);

                    for (int i = 0; i < nodes.length; i++) {
                        assertEquals(
                            dimension + " dims, " + similarity + ", ordinal " + nodes[i],
                            Float.floatToIntBits(staged[i]),
                            Float.floatToIntBits(fromSlab[i])
                        );
                        assertEquals(
                            "and both must be the score of the vector that is really in the file",
                            similarity.compare(target, vectors[nodes[i]]),
                            fromSlab[i],
                            0.0f
                        );
                    }
                }
                cache.releaseSegment(null);
                cache.configure(0L);
                cache.configure(4L * SLAB_BYTES);
            }
        }
        assertTrue("the staged channel has to have been used, or this proved nothing", DirectIOVectorIndexInput.STATS.stagedReads() > 0);
    }
}

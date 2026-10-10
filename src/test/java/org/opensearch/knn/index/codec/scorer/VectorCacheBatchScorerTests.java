/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.scorer;

import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.hnsw.RandomVectorScorerSupplier;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.memory.SlabVectorCache;
import org.opensearch.knn.index.memory.VectorCacheScope;

import java.io.IOException;
import java.util.Random;

/**
 * The scorer-level lookup (I4, D1's core): a candidate is looked up before it is read, a hit is served from
 * a slab, a miss is read through the values and cached, and a score never depends on which of the two
 * happened.
 *
 * <p>The values object here is a plain array rather than a real segment, which is deliberate: what is under
 * test is the lookup, the fill, the release and the fallbacks, and an array makes "was this vector read from
 * the values or served from the cache" a counter rather than an inference. That the whole path works against
 * a real segment is what the I8 re-proof measures, and it runs at this commit.
 */
public class VectorCacheBatchScorerTests extends KNNTestCase {

    private static final int DIMENSION = 128;
    private static final int VECTOR_BYTES = DIMENSION * Float.BYTES;
    private static final int SLAB_BYTES = 1 << 20;
    private static final int FIELD = 3;

    private SlabVectorCache cache;
    private Object core;
    private float[][] vectors;
    private float[] target;

    @Override
    public void setUp() throws Exception {
        super.setUp();
        cache = new SlabVectorCache(SlabVectorCache.DIRECT_SOURCE, SLAB_BYTES);
        cache.configure(4L * SLAB_BYTES);
        VectorCacheScope.setNodeCacheForTesting(cache);
        core = new Object();
        final Random random = new Random(42L);
        vectors = new float[512][DIMENSION];
        for (final float[] vector : vectors) {
            for (int i = 0; i < DIMENSION; i++) {
                vector[i] = random.nextFloat();
            }
        }
        target = new float[DIMENSION];
        for (int i = 0; i < DIMENSION; i++) {
            target[i] = random.nextFloat();
        }
    }

    @Override
    public void tearDown() throws Exception {
        VectorCacheScope.setNodeCacheForTesting(null);
        if (cache != null) {
            cache.configure(0L);
        }
        super.tearDown();
    }

    // -----------------------------------------------------------------------------------------------
    // The test doubles: array-backed values, and a delegate that scores them the plain way.
    // -----------------------------------------------------------------------------------------------

    /** Array-backed values that count how many vectors have been read out of them. */
    private static final class CountingFloatVectorValues extends FloatVectorValues {
        private final float[][] vectors;
        private int reads;

        private CountingFloatVectorValues(final float[][] vectors) {
            this.vectors = vectors;
        }

        @Override
        public float[] vectorValue(final int ord) {
            reads++;
            return vectors[ord];
        }

        @Override
        public int dimension() {
            return vectors[0].length;
        }

        @Override
        public int size() {
            return vectors.length;
        }

        @Override
        public FloatVectorValues copy() {
            return this;
        }
    }

    /** What Lucene's default float scorer does, which is what the cache path has to reproduce exactly. */
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

    /** A delegate that scores something else entirely, which the kernel probe has to catch. */
    private static FlatVectorsScorer quantizedLookingScorer() {
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
                    public float score(final int node) {
                        // Whatever this is, it is not the full-precision compare the cache would do.
                        return 0.125f * (node + 1);
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

    private RandomVectorScorer scorerInScope(final FlatVectorsScorer delegate, final CountingFloatVectorValues values) throws IOException {
        VectorCacheScope.open(core, FIELD);
        try {
            return new PrefetchableFlatVectorScorer(delegate).getRandomVectorScorer(VectorSimilarityFunction.EUCLIDEAN, values, target);
        } finally {
            VectorCacheScope.clear();
        }
    }

    private static int[] ordinals(final int from, final int count) {
        final int[] nodes = new int[count];
        for (int i = 0; i < count; i++) {
            nodes[i] = from + i;
        }
        return nodes;
    }

    private float[] expectedScores(final int[] nodes, final float[][] vectors, final float[] target) {
        final float[] expected = new float[nodes.length];
        for (int i = 0; i < nodes.length; i++) {
            expected[i] = VectorSimilarityFunction.EUCLIDEAN.compare(target, vectors[nodes[i]]);
        }
        return expected;
    }

    // -----------------------------------------------------------------------------------------------

    /** A cold batch misses, reads every candidate, scores it right, and fills the cache. */
    public void testAColdBatchMissesReadsAndFills() throws IOException {
        final CountingFloatVectorValues values = new CountingFloatVectorValues(vectors);
        final RandomVectorScorer scorer = scorerInScope(plainScorer(), values);
        final int[] nodes = ordinals(0, 64);
        final float[] scores = new float[64];

        final float max = scorer.bulkScore(nodes, scores, 64);

        final float[] expected = expectedScores(nodes, vectors, target);
        assertArrayEquals("a miss must score exactly what the delegate would", expected, scores, 0.0f);
        float expectedMax = Float.NEGATIVE_INFINITY;
        for (final float score : expected) {
            expectedMax = Math.max(expectedMax, score);
        }
        assertEquals(expectedMax, max, 0.0f);
        assertEquals("every candidate missed", 64L, cache.misses());
        assertEquals(0L, cache.hits());
        assertEquals("and every one was cached", 64L, cache.entries());
    }

    /** A warm batch hits, serves from the slab, and reads nothing out of the values. */
    public void testAWarmBatchHitsAndReadsNothing() throws IOException {
        final CountingFloatVectorValues values = new CountingFloatVectorValues(vectors);
        final int[] nodes = ordinals(0, 64);
        final float[] cold = new float[64];
        final float[] warm = new float[64];

        final RandomVectorScorer first = scorerInScope(plainScorer(), values);
        first.bulkScore(nodes, cold, 64);
        final int readsAfterCold = values.reads;

        final RandomVectorScorer second = scorerInScope(plainScorer(), values);
        final long hitsBefore = cache.hits();
        second.bulkScore(nodes, warm, 64);

        assertEquals("every candidate hit", hitsBefore + 64, cache.hits());
        // The probe reads one vector twice, which is the one read a warm batch still makes. See
        // VectorCacheBatchScorer#kernelMatchesDelegate.
        assertEquals("a hit must not read the vector again", readsAfterCold + 2, values.reads);
        assertArrayEquals("I5: a slab hit and a staged miss must score bit-identically", cold, warm, 0.0f);
    }

    /** Hits and misses in one batch, which is the steady state, and every score still exact. */
    public void testAMixedBatchServesBothAndScoresBoth() throws IOException {
        final CountingFloatVectorValues values = new CountingFloatVectorValues(vectors);
        final int[] warmed = ordinals(0, 32);
        final RandomVectorScorer first = scorerInScope(plainScorer(), values);
        first.bulkScore(warmed, new float[32], 32);

        final int[] mixed = ordinals(0, 64);
        final float[] scores = new float[64];
        final RandomVectorScorer second = scorerInScope(plainScorer(), values);
        final long hitsBefore = cache.hits();
        final long missesBefore = cache.misses();
        second.bulkScore(mixed, scores, 64);

        assertEquals("the first half was cached", hitsBefore + 32, cache.hits());
        assertEquals("the second half was not", missesBefore + 32, cache.misses());
        assertArrayEquals(expectedScores(mixed, vectors, target), scores, 0.0f);
        assertEquals("and the misses were filled", 64L, cache.entries());
    }

    /** The nested path's method: one ordinal at a time, cached the same way. */
    public void testTheSingleOrdinalPathAlsoHitsAndFills() throws IOException {
        final CountingFloatVectorValues values = new CountingFloatVectorValues(vectors);
        final RandomVectorScorer scorer = scorerInScope(plainScorer(), values);

        final float cold = scorer.score(7);
        assertEquals("and it filled", 1L, cache.entries());
        final int readsAfterCold = values.reads;
        final float warm = scorer.score(7);

        assertEquals("the second call must not read the vector", readsAfterCold, values.reads);
        assertEquals("the same score either way", Float.floatToIntBits(cold), Float.floatToIntBits(warm));
        assertEquals(VectorSimilarityFunction.EUCLIDEAN.compare(target, vectors[7]), warm, 0.0f);
    }

    /** A segment close releases that segment's vectors, so the next batch reads them again. */
    public void testASegmentCloseSendsTheBatchBackToTheValues() throws IOException {
        final CountingFloatVectorValues values = new CountingFloatVectorValues(vectors);
        final int[] nodes = ordinals(0, 16);
        scorerInScope(plainScorer(), values).bulkScore(nodes, new float[16], 16);
        assertEquals(16L, cache.entries());

        cache.releaseSegment(core);
        assertEquals("the segment's entries are gone", 0L, cache.entries());

        final int readsBefore = values.reads;
        final float[] scores = new float[16];
        final long missesBefore = cache.misses();
        scorerInScope(plainScorer(), values).bulkScore(nodes, scores, 16);

        assertEquals("every candidate had to be read again", missesBefore + 16, cache.misses());
        assertTrue(values.reads >= readsBefore + 16);
        assertArrayEquals("and the answers are unchanged", expectedScores(nodes, vectors, target), scores, 0.0f);
    }

    /** With the cache off the scorer is the phase 15 scorer: prefetch everything, delegate everything. */
    public void testCacheOffIsTheOldPath() throws IOException {
        cache.configure(0L);
        final CountingFloatVectorValues values = new CountingFloatVectorValues(vectors);
        final int[] nodes = ordinals(0, 16);
        final float[] scores = new float[16];

        scorerInScope(plainScorer(), values).bulkScore(nodes, scores, 16);

        assertArrayEquals(expectedScores(nodes, vectors, target), scores, 0.0f);
        assertEquals("off means not even a lookup", 0L, cache.hits());
        assertEquals(0L, cache.misses());
        assertEquals(0L, cache.entries());
    }

    /** No scope, no cache: a scorer built outside {@code ExactSearcher} must behave exactly as before. */
    public void testNoScopeMeansNoCache() throws IOException {
        final CountingFloatVectorValues values = new CountingFloatVectorValues(vectors);
        final RandomVectorScorer scorer = new PrefetchableFlatVectorScorer(plainScorer()).getRandomVectorScorer(
            VectorSimilarityFunction.EUCLIDEAN,
            values,
            target
        );
        final int[] nodes = ordinals(0, 16);
        final float[] scores = new float[16];

        scorer.bulkScore(nodes, scores, 16);

        assertArrayEquals(expectedScores(nodes, vectors, target), scores, 0.0f);
        assertEquals("the cache must not have been consulted at all", 0L, cache.hits() + cache.misses());
        assertEquals(0L, cache.entries());
    }

    /**
     * A delegate whose kernel is not a full-precision compare must keep its own answers and never be served
     * from the cache. This is the SQ 1040 case: it wraps every scorer it returns.
     */
    public void testADifferentKernelDeclinesAndKeepsItsOwnScores() throws IOException {
        final CountingFloatVectorValues values = new CountingFloatVectorValues(vectors);
        final RandomVectorScorer scorer = scorerInScope(quantizedLookingScorer(), values);
        final int[] nodes = ordinals(0, 8);
        final float[] scores = new float[8];

        scorer.bulkScore(nodes, scores, 8);

        for (int i = 0; i < 8; i++) {
            assertEquals("the delegate's score, not ours", 0.125f * (nodes[i] + 1), scores[i], 0.0f);
        }
        assertEquals("and nothing was cached from it", 0L, cache.entries());
        assertEquals(0L, cache.hits() + cache.misses());
    }

    /** Two segments with the same ordinals must not see each other's vectors. */
    public void testTwoSegmentsDoNotCollide() throws IOException {
        final float[][] other = new float[vectors.length][DIMENSION];
        for (int ord = 0; ord < vectors.length; ord++) {
            for (int i = 0; i < DIMENSION; i++) {
                other[ord][i] = vectors[ord][i] + 1.0f;
            }
        }
        final int[] nodes = ordinals(0, 8);

        final CountingFloatVectorValues first = new CountingFloatVectorValues(vectors);
        final float[] firstScores = new float[8];
        scorerInScope(plainScorer(), first).bulkScore(nodes, firstScores, 8);

        final Object otherCore = new Object();
        core = otherCore;
        final CountingFloatVectorValues second = new CountingFloatVectorValues(other);
        final float[] secondScores = new float[8];
        scorerInScope(plainScorer(), second).bulkScore(nodes, secondScores, 8);

        assertArrayEquals(expectedScores(nodes, vectors, target), firstScores, 0.0f);
        assertArrayEquals("the other segment's bytes must not be served", expectedScores(nodes, other, target), secondScores, 0.0f);
        assertEquals("each segment got its own entries", 16L, cache.entries());
    }

    /** Every similarity function the float path can carry, served from a slab and scored the same. */
    public void testEverySimilarityFunctionRoundTrips() throws IOException {
        for (final VectorSimilarityFunction similarity : VectorSimilarityFunction.values()) {
            cache.releaseSegment(core);
            core = new Object();
            final CountingFloatVectorValues values = new CountingFloatVectorValues(vectors);
            VectorCacheScope.open(core, FIELD);
            final RandomVectorScorer scorer;
            try {
                scorer = new PrefetchableFlatVectorScorer(plainScorer()).getRandomVectorScorer(similarity, values, target);
            } finally {
                VectorCacheScope.clear();
            }
            final int[] nodes = ordinals(0, 8);
            final float[] cold = new float[8];
            final float[] warm = new float[8];
            scorer.bulkScore(nodes, cold, 8);
            scorer.bulkScore(nodes, warm, 8);
            for (int i = 0; i < 8; i++) {
                assertEquals(
                    similarity + ": a hit and a miss must agree to the bit",
                    Float.floatToIntBits(cold[i]),
                    Float.floatToIntBits(warm[i])
                );
                assertEquals(similarity.toString(), similarity.compare(target, vectors[nodes[i]]), warm[i], 0.0f);
            }
        }
    }
}

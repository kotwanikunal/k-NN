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
 * G1: the kernel probe is decided once per segment field and then shared, rather than once per query.
 *
 * <p><b>What this is testing against (MEASURED on the EBS node at {@code d5649548})</b>: with the cache on,
 * every query made about one synchronous 8 KiB {@code .vec} read that no prefetch covered, 1.01 per query in
 * all four dio-10m warm-ups and exactly 5 for 5 queries right after a deploy, because the probe verdict was a
 * field of a scorer and a scorer is built per query per segment. So the thing to count here is the delegate's
 * {@code score} calls and {@code direct_io_stats.cache.probes} across many scorers on one field.
 *
 * <p><b>And in both directions</b>: a shared verdict is a decision for the life of a segment, so a false fail
 * turns the cache off for that segment and a false pass lets this kernel answer for a delegate that scores
 * differently. Both are tested, with a delegate that is plain on some queries and quantized-looking on
 * others, because the rule is "two probes in a row have to agree" and not "the first probe wins".
 *
 * <p>One delegate class throughout, switched by a flag, because the class is part of the verdict key: two
 * anonymous classes would be two keys and would share nothing, which would make these tests pass for the
 * wrong reason.
 */
public class SharedKernelVerdictTests extends KNNTestCase {

    private static final int DIMENSION = 64;
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
        final Random random = new Random(7L);
        vectors = new float[256][DIMENSION];
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
    // The test doubles. One scorer class, two kernels, and a count of the delegate's score calls.
    // -----------------------------------------------------------------------------------------------

    private static final class ArrayValues extends FloatVectorValues {
        private final float[][] vectors;

        private ArrayValues(final float[][] vectors) {
            this.vectors = vectors;
        }

        @Override
        public float[] vectorValue(final int ord) {
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

    /**
     * One delegate scorer class, so every scorer it builds shares the verdict key, with a flag that decides
     * whether it answers the full-precision compare the cache would or something else entirely.
     */
    private static final class SwitchableScorer extends RandomVectorScorer.AbstractRandomVectorScorer {
        private final VectorSimilarityFunction similarity;
        private final float[] target;
        private final boolean plain;
        private final int[] scoreCalls;

        private SwitchableScorer(
            final KnnVectorValues values,
            final VectorSimilarityFunction similarity,
            final float[] target,
            final boolean plain,
            final int[] scoreCalls
        ) {
            super(values);
            this.similarity = similarity;
            this.target = target;
            this.plain = plain;
            this.scoreCalls = scoreCalls;
        }

        @Override
        public float score(final int node) throws IOException {
            scoreCalls[0]++;
            if (plain) {
                return similarity.compare(target, ((FloatVectorValues) values()).vectorValue(node));
            }
            return 0.125f * (node + 1);
        }
    }

    /** The delegate factory: {@code plain} picks the kernel, {@code scoreCalls[0]} counts its probes. */
    private FlatVectorsScorer switchable(final boolean plain, final int[] scoreCalls) {
        return new FlatVectorsScorer() {
            @Override
            public RandomVectorScorerSupplier getRandomVectorScorerSupplier(final VectorSimilarityFunction f, final KnnVectorValues v) {
                throw new UnsupportedOperationException();
            }

            @Override
            public RandomVectorScorer getRandomVectorScorer(
                final VectorSimilarityFunction similarity,
                final KnnVectorValues values,
                final float[] queryTarget
            ) {
                return new SwitchableScorer(values, similarity, queryTarget, plain, scoreCalls);
            }

            @Override
            public RandomVectorScorer getRandomVectorScorer(
                final VectorSimilarityFunction similarity,
                final KnnVectorValues values,
                final byte[] queryTarget
            ) {
                throw new UnsupportedOperationException();
            }
        };
    }

    /** One query's scorer on one segment field, built the way {@code ExactSearcher} builds one. */
    private RandomVectorScorer query(final Object segmentCore, final int field, final boolean plain, final int[] scoreCalls)
        throws IOException {
        return query(segmentCore, field, plain, scoreCalls, target);
    }

    private RandomVectorScorer query(
        final Object segmentCore,
        final int field,
        final boolean plain,
        final int[] scoreCalls,
        final float[] queryTarget
    ) throws IOException {
        VectorCacheScope.open(segmentCore, field);
        try {
            return new PrefetchableFlatVectorScorer(switchable(plain, scoreCalls)).getRandomVectorScorer(
                VectorSimilarityFunction.EUCLIDEAN,
                new ArrayValues(vectors),
                queryTarget
            );
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

    /** One query: a batch of 8 candidates scored by one freshly built scorer. */
    private float[] runQuery(final Object segmentCore, final int field, final boolean plain, final int[] scoreCalls) throws IOException {
        final int[] nodes = ordinals(0, 8);
        final float[] scores = new float[8];
        query(segmentCore, field, plain, scoreCalls).bulkScore(nodes, scores, 8);
        return scores;
    }

    private float[] expectedOurScores() {
        final int[] nodes = ordinals(0, 8);
        final float[] expected = new float[8];
        for (int i = 0; i < 8; i++) {
            expected[i] = VectorSimilarityFunction.EUCLIDEAN.compare(target, vectors[nodes[i]]);
        }
        return expected;
    }

    // -----------------------------------------------------------------------------------------------

    /**
     * The measurement G1 exists for: many queries on one segment field cost a bounded number of probes, not
     * one each, and every one of them is still served from the cache.
     */
    public void testManyQueriesOnOneFieldProbeABoundedNumberOfTimes() throws IOException {
        final int[] scoreCalls = new int[1];
        for (int q = 0; q < 20; q++) {
            assertArrayEquals("query " + q, expectedOurScores(), runQuery(core, FIELD, true, scoreCalls), 0.0f);
        }

        assertEquals("the delegate must be asked twice in all, not once per query", 2, scoreCalls[0]);
        assertEquals("and the counter must say the same", 2L, cache.kernelProbes());
        assertEquals("every query after the first was served from the cache", 8L, cache.entries());
        assertTrue("and the hits must be real", cache.hits() >= 8L * 19);
    }

    /** A verdict is about one segment core: another core's queries probe for themselves. */
    public void testTwoCoresProbeSeparately() throws IOException {
        final int[] first = new int[1];
        for (int q = 0; q < 4; q++) {
            runQuery(core, FIELD, true, first);
        }
        assertEquals(2, first[0]);

        final Object otherCore = new Object();
        final int[] second = new int[1];
        for (int q = 0; q < 4; q++) {
            runQuery(otherCore, FIELD, true, second);
        }

        assertEquals("the other core had to reach its own verdict", 2, second[0]);
        assertEquals("and the first core's is still shared", 2, first[0]);
        assertEquals("four probes in all", 4L, cache.kernelProbes());
    }

    /** Two fields of one segment are two kernels, and a decline on one must not touch the other. */
    public void testTwoFieldsOfOneSegmentKeepSeparateVerdicts() throws IOException {
        final int[] good = new int[1];
        for (int q = 0; q < 4; q++) {
            assertArrayEquals(expectedOurScores(), runQuery(core, FIELD, true, good), 0.0f);
        }
        final long entriesFromTheGoodField = cache.entries();
        assertEquals(8L, entriesFromTheGoodField);

        final int[] bad = new int[1];
        final int otherField = FIELD + 1;
        for (int q = 0; q < 4; q++) {
            final float[] scores = runQuery(core, otherField, false, bad);
            for (int i = 0; i < 8; i++) {
                assertEquals("the other field keeps its delegate's scores", 0.125f * (i + 1), scores[i], 0.0f);
            }
        }

        assertEquals("the declining field cached nothing", entriesFromTheGoodField, cache.entries());
        final long probesSoFar = cache.kernelProbes();
        assertArrayEquals("and the passing field still serves", expectedOurScores(), runQuery(core, FIELD, true, good), 0.0f);
        assertEquals("without probing again", probesSoFar, cache.kernelProbes());
        assertEquals(2, good[0]);
    }

    /** A core's close takes its verdict with it, so the segment it is reopened as probes again. */
    public void testAClosedCoreLosesItsVerdict() throws IOException {
        final int[] scoreCalls = new int[1];
        for (int q = 0; q < 4; q++) {
            runQuery(core, FIELD, true, scoreCalls);
        }
        assertEquals(2, scoreCalls[0]);

        cache.releaseSegment(core);

        // The same core object, so this can only pass if the verdict itself was dropped rather than keyed
        // around.
        runQuery(core, FIELD, true, scoreCalls);
        assertEquals("a released core has to probe again", 3, scoreCalls[0]);

        // And a reopened segment, which is a new core, is a new verdict either way.
        final Object reopened = new Object();
        final int[] afterReopen = new int[1];
        for (int q = 0; q < 4; q++) {
            runQuery(reopened, FIELD, true, afterReopen);
        }
        assertEquals(2, afterReopen[0]);
    }

    /** The decline the probe is there for still holds when the verdict is shared. */
    public void testAMismatchingDelegateStillDeclines() throws IOException {
        final int[] scoreCalls = new int[1];
        for (int q = 0; q < 10; q++) {
            final float[] scores = runQuery(core, FIELD, false, scoreCalls);
            for (int i = 0; i < 8; i++) {
                assertEquals("query " + q + " must keep the delegate's score", 0.125f * (i + 1), scores[i], 0.0f);
            }
        }

        assertEquals("the field is declined after two agreeing probes", 2L, cache.kernelProbes());
        assertEquals("and nothing was ever cached from it", 0L, cache.entries());
        assertEquals(0L, cache.hits() + cache.misses());
    }

    /**
     * One failing probe among passing ones must not turn the cache off for the field, because a shared fail
     * would last the segment's life. With inner product two correct kernels can differ by more than the
     * probe's tolerance on a score near zero, which is how a false fail would arise in production (INFERRED).
     */
    public void testOneFailingProbeDoesNotTurnTheFieldOff() throws IOException {
        final int[] scoreCalls = new int[1];
        runQuery(core, FIELD, true, scoreCalls);
        runQuery(core, FIELD, false, scoreCalls);
        runQuery(core, FIELD, true, scoreCalls);
        runQuery(core, FIELD, true, scoreCalls);

        assertEquals("four probes, because the fail broke the streak", 4L, cache.kernelProbes());
        final long probesSoFar = cache.kernelProbes();
        assertArrayEquals("the field is serving", expectedOurScores(), runQuery(core, FIELD, true, scoreCalls), 0.0f);
        assertEquals("and has stopped probing", probesSoFar, cache.kernelProbes());
        assertTrue("the fifth query hit the cache", cache.hits() > 0);
    }

    /** And one passing probe on a delegate that scores something else must not turn the field on. */
    public void testOnePassingProbeDoesNotTurnAMismatchingFieldOn() throws IOException {
        final int[] scoreCalls = new int[1];
        runQuery(core, FIELD, true, scoreCalls);
        runQuery(core, FIELD, false, scoreCalls);
        runQuery(core, FIELD, false, scoreCalls);

        assertEquals("three probes: the stray pass, then two agreeing fails", 3L, cache.kernelProbes());
        final long probesSoFar = cache.kernelProbes();
        final long entriesBefore = cache.entries();
        // The flag says plain, but the field is declined, so this must go to the delegate and cache nothing.
        final float[] scores = runQuery(core, FIELD, true, scoreCalls);

        assertEquals("the shared fail holds", probesSoFar, cache.kernelProbes());
        assertEquals("so nothing was cached", entriesBefore, cache.entries());
        assertArrayEquals("and the delegate answered, which here is the same number", expectedOurScores(), scores, 0.0f);
    }

    /**
     * A target that cannot tell two kernels apart decides the scorer that ran it and nothing beyond it. An
     * all-zero target scores zero under any kernel, so believing it would be the false pass this rule exists
     * to prevent (INFERRED).
     */
    public void testAnAllZeroTargetIsNeverShared() throws IOException {
        final int[] scoreCalls = new int[1];
        final float[] zeros = new float[DIMENSION];
        for (int q = 0; q < 5; q++) {
            query(core, FIELD, true, scoreCalls, zeros).bulkScore(ordinals(0, 8), new float[8], 8);
        }

        assertEquals("every one of those queries had to probe for itself", 5, scoreCalls[0]);
        assertEquals(5L, cache.kernelProbes());

        // And the field is still undecided, so a query with a real target probes and can be believed.
        for (int q = 0; q < 2; q++) {
            runQuery(core, FIELD, true, scoreCalls);
        }
        assertEquals(7, scoreCalls[0]);
        runQuery(core, FIELD, true, scoreCalls);
        assertEquals("two real probes agreed, so this one did not probe", 7, scoreCalls[0]);
    }
}

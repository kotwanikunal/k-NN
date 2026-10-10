/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.memory;

import org.apache.lucene.index.VectorSimilarityFunction;
import org.opensearch.knn.KNNTestCase;

/**
 * G1's rule for believing a probe, on its own: what it takes to share a verdict, what it takes to lose one,
 * and that a probe nobody may believe still counts in {@code direct_io_stats.cache.probes}.
 *
 * <p>{@code SharedKernelVerdictTests} drives the same rule through the scorer, which is where it matters;
 * this one reaches the cases a scorer cannot easily be made to produce, in particular a kernel that never
 * agrees with itself.
 */
public class KernelVerdictsTests extends KNNTestCase {

    private static final int FIELD = 2;
    private static final VectorSimilarityFunction SIMILARITY = VectorSimilarityFunction.EUCLIDEAN;

    private KernelVerdicts verdicts;
    private Object core;

    @Override
    public void setUp() throws Exception {
        super.setUp();
        verdicts = new KernelVerdicts();
        core = new Object();
    }

    private KernelVerdicts.Verdict verdict() {
        return verdicts.verdict(core, FIELD, String.class, SIMILARITY);
    }

    private void probe(final boolean passed) {
        verdicts.recordProbe(core, FIELD, String.class, SIMILARITY, passed, true);
    }

    public void testOneProbeDecidesNothing() {
        probe(true);
        assertEquals(KernelVerdicts.Verdict.UNDECIDED, verdict());
        assertEquals("but it is counted", 1L, verdicts.probes());
    }

    public void testTwoAgreeingProbesShareTheirAnswer() {
        probe(true);
        probe(true);
        assertEquals(KernelVerdicts.Verdict.PASSED, verdict());

        // A verdict is final: a later probe cannot move it, and the scorer that ran it keeps its own answer.
        probe(false);
        assertEquals(KernelVerdicts.Verdict.PASSED, verdict());
        assertEquals(3L, verdicts.probes());
    }

    public void testTwoAgreeingFailsDeclineTheField() {
        probe(false);
        probe(false);
        assertEquals(KernelVerdicts.Verdict.FAILED, verdict());
    }

    public void testAStrayProbeOnlyBreaksTheStreak() {
        probe(true);
        probe(false);
        assertEquals("neither answer has been seen twice in a row", KernelVerdicts.Verdict.UNDECIDED, verdict());
        probe(true);
        assertEquals(KernelVerdicts.Verdict.UNDECIDED, verdict());
        probe(true);
        assertEquals(KernelVerdicts.Verdict.PASSED, verdict());
    }

    /** A kernel that never agrees with itself is not one this cache may serve from, and it has to stop. */
    public void testAKernelThatNeverAgreesIsDeclinedAndStopsCostingProbes() {
        for (int i = 0; i < 8; i++) {
            assertEquals("still undecided after " + i + " probes", KernelVerdicts.Verdict.UNDECIDED, verdict());
            probe(i % 2 == 0);
        }
        assertEquals("the safe direction", KernelVerdicts.Verdict.FAILED, verdict());
    }

    public void testAnUnshareableProbeIsCountedAndNothingElse() {
        verdicts.recordProbe(core, FIELD, String.class, SIMILARITY, true, false);
        verdicts.recordProbe(core, FIELD, String.class, SIMILARITY, true, false);
        assertEquals(KernelVerdicts.Verdict.UNDECIDED, verdict());
        assertEquals("no state was even created for them", 0, verdicts.size());
        assertEquals("but both ran, so both are counted", 2L, verdicts.probes());
    }

    public void testTheKeyIsTheCoreTheFieldTheDelegateClassAndTheSimilarity() {
        probe(true);
        probe(true);
        assertEquals(KernelVerdicts.Verdict.PASSED, verdict());

        assertEquals("another core", KernelVerdicts.Verdict.UNDECIDED, verdicts.verdict(new Object(), FIELD, String.class, SIMILARITY));
        assertEquals("another field", KernelVerdicts.Verdict.UNDECIDED, verdicts.verdict(core, FIELD + 1, String.class, SIMILARITY));
        assertEquals("another delegate class", KernelVerdicts.Verdict.UNDECIDED, verdicts.verdict(core, FIELD, Integer.class, SIMILARITY));
        assertEquals(
            "another similarity function is another kernel",
            KernelVerdicts.Verdict.UNDECIDED,
            verdicts.verdict(core, FIELD, String.class, VectorSimilarityFunction.DOT_PRODUCT)
        );
    }

    public void testReleasingACoreForgetsEveryVerdictItHad() {
        probe(true);
        probe(true);
        verdicts.recordProbe(core, FIELD + 1, String.class, SIMILARITY, false, true);
        final Object otherCore = new Object();
        verdicts.recordProbe(otherCore, FIELD, String.class, SIMILARITY, true, true);
        verdicts.recordProbe(otherCore, FIELD, String.class, SIMILARITY, true, true);
        assertEquals(3, verdicts.size());

        verdicts.releaseCore(core);

        assertEquals("both of that core's fields are gone", 1, verdicts.size());
        assertEquals(KernelVerdicts.Verdict.UNDECIDED, verdict());
        assertEquals(
            "and the other core's verdict is untouched",
            KernelVerdicts.Verdict.PASSED,
            verdicts.verdict(otherCore, FIELD, String.class, SIMILARITY)
        );
    }

    /** The cache's release path is the one that calls it, and it has to run even with the cache off. */
    public void testTheCacheReleasesVerdictsEvenWhenItIsOff() {
        final SlabVectorCache cache = new SlabVectorCache(SlabVectorCache.DIRECT_SOURCE, 1 << 20);
        cache.kernelVerdicts().recordProbe(core, FIELD, String.class, SIMILARITY, true, true);
        cache.kernelVerdicts().recordProbe(core, FIELD, String.class, SIMILARITY, true, true);
        assertEquals(KernelVerdicts.Verdict.PASSED, cache.kernelVerdicts().verdict(core, FIELD, String.class, SIMILARITY));
        assertEquals("the cache is off, which is what makes this the case worth testing", 0L, cache.maxBytes());

        cache.releaseSegment(core);

        assertEquals(0, cache.kernelVerdicts().size());
        assertEquals(2L, cache.kernelProbes());
    }
}

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.memory;

import org.apache.lucene.index.VectorSimilarityFunction;

import java.util.Iterator;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.LongAdder;

/**
 * Remembers, per segment field, whether the vector cache's scoring kernel answers what that field's delegate
 * scorer answers, so the probe that decides it runs a handful of times per segment field rather than once per
 * query.
 *
 * <h2>The measurement this exists for (G1, MEASURED on the EBS node at {@code d5649548})</h2>
 * {@code VectorCacheBatchScorer} keeps its probe verdict in an instance field, and
 * {@code PrefetchableFlatVectorScorer#getRandomVectorScorer} builds one scorer per query per segment, so
 * every query probed again. A probe reads one vector through the delegate, which no prefetch covers and no
 * lookup precedes, so with the cache on every query made about one synchronous 8 KiB {@code .vec} read: 1.01
 * per query in all four dio-10m warm-ups at 25 to 27% hits, and exactly 5 for 5 queries at 2% hits right
 * after a deploy. On dio-1m with a 4 GB cache that was about 1 of the 4.25 device reads per query, and it sat
 * on the critical path, before the batch lookup and before any prefetch.
 *
 * <h2>What a verdict is keyed by, and why all four parts are needed</h2>
 * {@code (segment core identity, field number, the delegate scorer's class, the similarity function)}. The
 * core and the field are the cache's own key, and the kernel question is a property of the bytes a field's
 * values decode to. The delegate's class is in the key because one segment field can be scored by more than
 * one scorer class over its life, and the SQ 1040 codec wraps every scorer it returns, so the class is what
 * decides whether the delegate answers from full-precision floats. The similarity function is in the key
 * because it <em>is</em> the kernel on this side: a verdict reached with Euclidean says nothing about inner
 * product.
 *
 * <h2>Two agreeing probes, and a probe that cannot be shared is not counted</h2>
 * A verdict shared across queries must not be decided by one unlucky probe in either direction. A false fail
 * would turn the cache off for the rest of the segment's life; a false pass would let this kernel serve a
 * field whose delegate scores differently, which is the one thing the probe exists to prevent. Two things can
 * make a single probe misleading (INFERRED, from the kernels rather than from a measurement): with inner
 * product two correct kernels can differ by more than the probe's tolerance on a score near zero, which is a
 * false fail, and an all-zero target makes any two kernels agree at zero, which is a false pass.
 *
 * <p>So a probe whose target is all zero, or not finite, is counted in the probe counter and then discarded:
 * it decides the scorer that ran it and nothing beyond it. And a verdict is shared only when
 * {@link #AGREEING_PROBES} probes <em>in a row</em>, from different scorers and so from different queries,
 * agree. One failing probe among passing ones therefore cannot turn the cache off for the field, and one
 * passing probe on a mismatching delegate cannot turn it on.
 *
 * <p>A delegate that keeps disagreeing with itself would never reach a streak, so after
 * {@link #MAX_PROBES} probes with no verdict the field is declined for good. That is the safe direction: a
 * kernel whose answer is not reproducible is not a kernel this cache may serve from, and it bounds the
 * probes a field can ever cost.
 *
 * <h2>Lifetime</h2>
 * A verdict is about one segment core, so it lives exactly as long as that core: {@link #releaseCore} is
 * called from {@link SlabVectorCache#releaseSegment}, which the core-closed listener
 * {@link VectorCacheScope#registerRelease} installs already calls. A verdict therefore cannot outlive its
 * segment, and a reopened segment is a new core and probes again.
 *
 * <p>An instance of this belongs to one {@link SlabVectorCache}, not to the JVM, so that a cache that is
 * replaced takes its verdicts with it.
 */
public final class KernelVerdicts {

    /** Probes in a row that have to agree before their answer is shared across queries. */
    private static final int AGREEING_PROBES = 2;

    /** Probes a field may cost before it is declined for not reaching a stable answer. */
    private static final int MAX_PROBES = 8;

    /** What is known about one segment field's kernel. */
    public enum Verdict {
        /** Not decided yet: the scorer asking has to probe, and its probe may be shared. */
        UNDECIDED,
        /** This kernel is the delegate's, so the cache may serve the field. */
        PASSED,
        /** It is not, so the field keeps the delegate's own scores. */
        FAILED
    }

    private record Key(Object core, int field, Class<?> delegate, VectorSimilarityFunction similarity) {
    }

    /** One field's running state. Mutated under its own monitor; {@link #verdict} is read without one. */
    private static final class State {
        private volatile Verdict verdict = Verdict.UNDECIDED;
        private boolean lastPassed;
        private int streak;
        private int probes;
    }

    private final Map<Key, State> states = new ConcurrentHashMap<>();

    private final LongAdder probes = new LongAdder();

    /**
     * What is already known about this segment field's kernel.
     *
     * @param core the segment core identity
     * @param field the field number
     * @param delegate the delegate scorer's class
     * @param similarity the similarity function the kernel would use
     * @return the shared verdict, or {@link Verdict#UNDECIDED} when the caller has to probe
     */
    public Verdict verdict(final Object core, final int field, final Class<?> delegate, final VectorSimilarityFunction similarity) {
        final State state = states.get(new Key(core, field, delegate, similarity));
        return state == null ? Verdict.UNDECIDED : state.verdict;
    }

    /**
     * Counts a probe that was run and folds its answer into the field's shared verdict.
     *
     * @param core the segment core identity
     * @param field the field number
     * @param delegate the delegate scorer's class
     * @param similarity the similarity function the probe used
     * @param passed whether the kernel matched the delegate
     * @param shareable whether this probe may decide anything beyond the scorer that ran it: false for an
     *                  ordinal the delegate would not score and for a target that cannot distinguish two
     *                  kernels, which are the two ways a probe can be misleading
     */
    public void recordProbe(
        final Object core,
        final int field,
        final Class<?> delegate,
        final VectorSimilarityFunction similarity,
        final boolean passed,
        final boolean shareable
    ) {
        probes.increment();
        if (shareable == false) {
            return;
        }
        final State state = states.computeIfAbsent(new Key(core, field, delegate, similarity), key -> new State());
        synchronized (state) {
            if (state.verdict != Verdict.UNDECIDED) {
                return;
            }
            state.probes++;
            state.streak = (state.probes > 1 && state.lastPassed == passed) ? state.streak + 1 : 1;
            state.lastPassed = passed;
            if (state.streak >= AGREEING_PROBES) {
                state.verdict = passed ? Verdict.PASSED : Verdict.FAILED;
            } else if (state.probes >= MAX_PROBES) {
                state.verdict = Verdict.FAILED;
            }
        }
    }

    /**
     * Forgets every verdict about a segment core, which is what its close means. Called from
     * {@link SlabVectorCache#releaseSegment}.
     *
     * @param core the segment core identity that has closed
     */
    public void releaseCore(final Object core) {
        final Iterator<Map.Entry<Key, State>> entries = states.entrySet().iterator();
        while (entries.hasNext()) {
            if (entries.next().getKey().core() == core) {
                entries.remove();
            }
        }
    }

    /** Kernel probes run since this cache was built, which on a node is since node start. */
    public long probes() {
        return probes.sum();
    }

    /** Verdicts held right now, for the tests that count them. */
    int size() {
        return states.size();
    }

    /** Test seam: forgets every verdict and zeroes the counter. */
    void reset() {
        states.clear();
        probes.reset();
    }
}

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.scorer;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.memory.KernelVerdicts;
import org.opensearch.knn.index.memory.SlabVectorCache;
import org.opensearch.knn.index.memory.VectorCacheScope;
import org.opensearch.knn.index.store.DirectIOVectorIndexInput;

import java.io.IOException;

/**
 * Scores a batch of candidates against the node's full-precision vector cache: looks every candidate up
 * first, prefetches only the misses, reads those through the segment's own {@code IndexInput} and caches
 * them, and scores every candidate, hit or miss, through one kernel.
 *
 * <h2>This is D1's core, and the order of the four steps is the point</h2>
 * Phase 15 cached vectors underneath the {@code IndexInput}, so a hit still went through the whole read
 * path to find out that no read was needed. Here the lookup happens above it, in the scorer, so a hit costs
 * a copy out of a slab and a compare, and the prefetch burst is built from the misses alone.
 *
 * <h2>One kernel for a hit and for a miss (I5)</h2>
 * Every candidate is scored by {@code similarityFunction.compare(target, vector)} where {@code vector} is
 * the same {@code float[]} either way: for a hit the cache copies the slab's bytes into it, for a miss the
 * segment's values decode into one. A score therefore cannot depend on whether the vector was cached, which
 * is what proof (c) asks for, and it holds by construction rather than by a tolerance.
 *
 * <h2>The kernel is checked against the delegate before any of this is used</h2>
 * The one risk in moving scoring above the {@code IndexInput} is that the scorer this wraps might not score
 * the way this does: the SQ 1040 codec wraps <em>every</em> scorer it returns
 * ({@code KNN1040ScalarQuantizedVectorScorer:107}), and a quantized or ADC delegate answers from different
 * bytes entirely. Rather than enumerate delegate classes, which would be a guess that a Lucene upgrade could
 * invalidate silently, this asks: it scores one candidate both ways and uses the cache only if the two agree
 * to within {@link #PROBE_TOLERANCE_ULPS}. A delegate that scores anything else declines here and keeps
 * today's behaviour exactly.
 *
 * <p><b>The answer is shared per segment field (G1)</b>, in {@link KernelVerdicts}, because a scorer is built
 * per query per segment and a probe reads a vector nothing prefetched: on the node that was about one
 * synchronous 8 KiB read per query on the critical path, in every regime and at every hit rate. A scorer asks
 * for the field's verdict first and probes only while there is none; see {@link KernelVerdicts} for the rule
 * that decides when two probes may be believed.
 *
 * <h2>Locking (the I4 criterion)</h2>
 * The lookup takes the cache's read lock once per batch, not once per vector, and scores each hit while it
 * holds it, which is why {@link SlabVectorCache#lookupBatch} takes a visitor. Lucene's batch is 64 ordinals
 * (FINDINGS I0.2), so this is 64 times fewer acquisitions than I2 measured at 2.1x scaling on eight threads.
 * Nothing under that lock reads a file: the misses are read after it is dropped. A fill takes the write lock
 * for one vector at a time, which is affordable because a fill only ever follows a device read.
 */
@Log4j2
final class VectorCacheBatchScorer {

    /**
     * How far the probe lets the delegate's score and this one differ and still count as the same kernel.
     * Not zero, because Lucene's own float scorers reach the same similarity through different accumulation
     * orders (a {@code MemorySegment} kernel against {@code VectorUtil}'s), and a few ulp of accumulation
     * difference is the same answer. A quantized or ADC delegate is not within a few ulp of a
     * full-precision compare; it is a different number, which is what this is here to detect. Phase 15's
     * re-proof measured a worst case of 3 ulp between arms for the same reason.
     */
    private static final int PROBE_TOLERANCE_ULPS = 16;

    private static final byte PROBE_PENDING = 0;
    private static final byte PROBE_PASSED = 1;
    private static final byte PROBE_FAILED = 2;

    private final SlabVectorCache cache;
    private final Object core;
    private final int field;
    private final VectorSimilarityFunction similarity;
    private final float[] target;
    private final FloatVectorValues values;
    private final RandomVectorScorer.AbstractRandomVectorScorer delegate;
    private final int vectorBytes;

    /** The one vector buffer a hit is copied into and a miss is scored from. See the class javadoc. */
    private final float[] buffer;

    private boolean[] hit = new boolean[0];
    private int[] missNodes = new int[0];

    private byte probe = PROBE_PENDING;

    /**
     * Whether the vector cache's target has been recomputed for the batch about to be scored.
     *
     * <p>Recomputing is where a change to {@code knn.direct_io.rescore.cache.max_bytes}, to
     * {@code knn.direct_io.rescore.enabled} or to the native budget reaches the cache, and {@link #engages}
     * is the hook because it is the one call that runs whether or not the cache is on and whether or not
     * anything missed. {@link #bulkScore} arms the next one, so the cadence is once per scored batch there
     * and once per scorer on the nested path, which never reaches it. Two cluster-settings reads per batch,
     * and {@code SlabVectorCache#configure} returns on a volatile read when the target has not moved.
     */
    private boolean targetRefreshed;

    /**
     * Whether span fill is on, re-read with the target on the same cadence and for the same reason: it is a
     * dynamic node setting, so a batch has to be able to see a change without a reopen. Read in
     * {@link #engages} rather than in {@link #spanFill} so that it is one settings read per batch and not
     * one per span. Starts false, so a scorer that never reaches {@link #engages} never span fills.
     */
    private boolean spanFillEnabled;

    private VectorCacheBatchScorer(
        final SlabVectorCache cache,
        final VectorCacheScope.Key key,
        final VectorSimilarityFunction similarity,
        final float[] target,
        final FloatVectorValues values,
        final RandomVectorScorer.AbstractRandomVectorScorer delegate
    ) {
        this.cache = cache;
        this.core = key.core();
        this.field = key.field();
        this.similarity = similarity;
        this.target = target;
        this.values = values;
        this.delegate = delegate;
        this.vectorBytes = values.dimension() * Float.BYTES;
        this.buffer = new float[values.dimension()];
    }

    /**
     * Builds one for this scorer, or returns {@code null} when the cache cannot serve this scorer at all:
     * no key in scope (the scorer was built somewhere that does not know its segment), values that are not
     * full-precision floats, a target that is not a float vector, or no cache on this node. Every one of
     * those is a decline rather than a failure, and leaves the read path exactly as it was.
     */
    static VectorCacheBatchScorer createOrNull(
        final VectorSimilarityFunction similarity,
        final KnnVectorValues values,
        final float[] target,
        final RandomVectorScorer.AbstractRandomVectorScorer delegate
    ) {
        if (similarity == null || target == null || delegate == null) {
            return null;
        }
        final VectorCacheScope.Key key = VectorCacheScope.current();
        if (key == null) {
            return null;
        }
        if (values instanceof FloatVectorValues == false) {
            return null;
        }
        final FloatVectorValues floatValues = (FloatVectorValues) values;
        final int dimension = floatValues.dimension();
        if (dimension <= 0 || target.length != dimension || floatValues.getVectorByteLength() != dimension * Float.BYTES) {
            // A byte length that is not the float length means these values are a view over something else,
            // so the ordinals here are not the ordinals the full-precision vectors are keyed by.
            return null;
        }
        final SlabVectorCache cache = VectorCacheScope.nodeCache();
        if (cache == null) {
            return null;
        }
        return new VectorCacheBatchScorer(cache, key, similarity, target, floatValues, delegate);
    }

    /**
     * Whether the cache should serve this batch: it has to be on, and this scorer's kernel has to be the
     * delegate's. Asked once per batch, because the cache can be turned on and off under an open segment
     * (D2) and the probe can only be run where there is a candidate to probe with.
     *
     * @param probeNode a candidate ordinal to price the kernel against the delegate with
     * @return true if {@link #bulkScore} and {@link #score} may be used
     * @throws IOException if the probe's read fails
     */
    boolean engages(final int probeNode) throws IOException {
        if (targetRefreshed == false) {
            // Before isOn(), and that order is the whole of I6's cache half. This is the only place on the
            // read path that always runs: a prefetch burst happens only when there are misses to declare,
            // and bulkScore runs only when the cache is already on, so a refresh at either of those
            // cadences cannot turn a cache that is off back on. Arm D of the re-proof measured both
            // failures in turn.
            targetRefreshed = true;
            VectorCacheScope.refreshNodeCacheTarget();
            spanFillEnabled = readSpanFillSetting();
        }
        if (cache.isOn() == false) {
            return false;
        }
        if (probe == PROBE_PENDING) {
            probe = resolveKernel(probeNode);
        }
        return probe == PROBE_PASSED;
    }

    /**
     * This scorer's verdict on its delegate's kernel: the segment field's shared one where there is one, and
     * otherwise a probe of its own, which is offered to the field (G1).
     *
     * @param node a candidate ordinal to probe with
     * @return {@link #PROBE_PASSED} or {@link #PROBE_FAILED}
     * @throws IOException if the probe's read fails
     */
    private byte resolveKernel(final int node) throws IOException {
        final KernelVerdicts verdicts = cache.kernelVerdicts();
        final Class<?> delegateClass = delegate.getClass();
        final KernelVerdicts.Verdict shared = verdicts.verdict(core, field, delegateClass, similarity);
        if (shared == KernelVerdicts.Verdict.PASSED) {
            return PROBE_PASSED;
        }
        if (shared == KernelVerdicts.Verdict.FAILED) {
            return PROBE_FAILED;
        }
        final ProbeResult result = kernelMatchesDelegate(node);
        // An ordinal the delegate would not score, or a target that cannot tell two kernels apart, decides
        // this scorer and must not decide the field. See KernelVerdicts.
        final boolean shareable = result != ProbeResult.UNSCORABLE && targetDistinguishesKernels();
        verdicts.recordProbe(core, field, delegateClass, similarity, result == ProbeResult.MATCHED, shareable);
        return result == ProbeResult.MATCHED ? PROBE_PASSED : PROBE_FAILED;
    }

    /**
     * Whether a probe with this target says anything about two kernels. An all-zero target scores zero under
     * any kernel, so it makes a mismatching delegate look like a match, and a target with a NaN or an
     * infinity in it makes both sides NaN, which compares unequal to itself. Either way the probe decides the
     * scorer that ran it and nothing beyond it.
     */
    private boolean targetDistinguishesKernels() {
        boolean nonZero = false;
        for (final float value : target) {
            if (Float.isFinite(value) == false) {
                return false;
            }
            if (value != 0.0f) {
                nonZero = true;
            }
        }
        return nonZero;
    }

    /**
     * Scores {@code numNodes} candidates, serving from the cache what it holds and filling it with what it
     * does not.
     *
     * @return the highest score, which is what {@code RandomVectorScorer#bulkScore} returns
     */
    float bulkScore(final int[] nodes, final float[] scores, final int numNodes) throws IOException {
        // Arms the next batch's refresh. The cadence is one recompute per scored batch for the bulk path
        // and one per scorer for the nested path, which reaches score(int) and never this method.
        targetRefreshed = false;
        if (hit.length < numNodes) {
            hit = new boolean[numNodes];
            missNodes = new int[numNodes];
        }
        final BatchHits visitor = new BatchHits(scores);
        cache.lookupBatch(core, field, nodes, numNodes, vectorBytes, buffer, hit, visitor);
        int misses = 0;
        for (int i = 0; i < numNodes; i++) {
            if (hit[i] == false) {
                missNodes[misses++] = nodes[i];
            }
        }
        if (misses == 0) {
            return visitor.max;
        }
        // Only the misses are declared, which is the whole point of looking up first: a burst that would
        // have staged 64 ranges stages the ones the device actually has to serve.
        PrefetchableVectorValuesHelper.doPrefetch(values, missNodes, misses);
        float max = visitor.max;
        for (int i = 0; i < numNodes; i++) {
            if (hit[i]) {
                continue;
            }
            final int node = nodes[i];
            final float[] vector = values.vectorValue(node);
            final float score = similarity.compare(target, vector);
            scores[i] = score;
            max = Math.max(max, score);
            cache.put(core, field, node, vector, vectorBytes);
        }
        spanFill();
        return max;
    }

    /**
     * F5: while the cache is below its slab target, also admits the whole vectors that happen to lie inside
     * the spans this batch staged.
     *
     * <p><b>The measurement this exists for</b> (ref/PILOT-REPORT.md section 4, MEASURED): on dio-1m with a
     * 4 GB cache, room for every vector in the file, 16% of candidates still missed after a 10,000-query
     * warm-up, and in that warm-up Direct I/O read 2.01 MB per query in about 40 KB spans, 6.5 times the
     * whole file, while caching only the requested vectors. mmap's page cache kept every page it read and
     * ended holding 95% of the file. The cache was paying for the reads and keeping a fraction of what they
     * delivered.
     *
     * <p><b>After the misses have been read and filled, not before</b>, for two reasons. The requested
     * vectors get their slots first, so the candidates this query actually asked for are the ones that
     * survive if the target is reached half way through; and the staged bytes are still valid here, because
     * the burst ends at the next {@code prefetch} on this thread and nothing between the read loop and this
     * call prefetches.
     *
     * <p>Silent about everything it cannot do: values read through mmap have no staged spans, a batch whose
     * prefetch was declined has none either, and a cache at its target admits nothing. All three are a
     * future hit forgone and nothing else.
     */
    private void spanFill() throws IOException {
        if (spanFillEnabled == false || cache.belowSlabTarget() == false) {
            return;
        }
        final DirectIOVectorIndexInput input = PrefetchableVectorValuesHelper.directIOSliceOrNull(values);
        if (input == null) {
            return;
        }
        // Outside the cache's lock, because this is the call that may wait for a staged read.
        if (input.prepareStagedSpans() == 0) {
            return;
        }
        input.visitStagedSpanVectors(vectorBytes, (buffer, position, firstOrdinal, count) -> {
            cache.putRun(core, field, firstOrdinal, count, buffer, position, vectorBytes);
            // Carry on only while there is still room the target allows. A run that admitted fewer than it
            // offered may simply have found them cached already, so the target and not the count is what
            // says whether the next span is worth walking.
            return cache.belowSlabTarget();
        });
    }

    /**
     * Scores one candidate. The nested path reaches only this method
     * ({@code NestedBestChildVectorScorer:159} scores child by child), so without it the vectors of a
     * nested field would never be cached or served; see FINDINGS I0.2.
     */
    float score(final int node) throws IOException {
        if (cache.get(core, field, node, vectorBytes, buffer)) {
            return similarity.compare(target, buffer);
        }
        final float[] vector = values.vectorValue(node);
        final float score = similarity.compare(target, vector);
        cache.put(core, field, node, vector, vectorBytes);
        return score;
    }

    /**
     * {@code knn.direct_io.rescore.cache.span_fill}, or false if it cannot be read. A setting that cannot be
     * read leaves span fill off rather than guessing it on, which is the same direction every other
     * unreadable setting on this path takes: the behaviour without it is the behaviour every earlier
     * measurement was taken against.
     */
    private static boolean readSpanFillSetting() {
        try {
            return KNNSettings.isDirectIORescoreCacheSpanFillEnabled();
        } catch (RuntimeException e) {
            log.debug("Could not read knn.direct_io.rescore.cache.span_fill; span fill stays off", e);
            return false;
        }
    }

    /** Collects the hits' scores while the cache holds its read lock. See {@link SlabVectorCache.HitVisitor}. */
    private final class BatchHits implements SlabVectorCache.HitVisitor {
        private final float[] scores;
        private float max = Float.NEGATIVE_INFINITY;

        private BatchHits(final float[] scores) {
            this.scores = scores;
        }

        @Override
        public void onHit(final int index, final float[] vector) {
            final float score = similarity.compare(target, vector);
            scores[index] = score;
            max = Math.max(max, score);
        }
    }

    /** What one probe found. {@code UNSCORABLE} declines the scorer and decides nothing for the field. */
    private enum ProbeResult {
        MATCHED,
        MISMATCHED,
        UNSCORABLE
    }

    /**
     * Scores one candidate both ways and reports whether the two agree. See the class javadoc for why this
     * is a measurement rather than a list of delegate classes.
     */
    private ProbeResult kernelMatchesDelegate(final int node) throws IOException {
        final float theirs;
        final float ours;
        try {
            theirs = delegate.score(node);
            ours = similarity.compare(target, values.vectorValue(node));
        } catch (IndexOutOfBoundsException e) {
            // An ordinal the delegate will not score is not a verdict on the kernel, so decline this scorer
            // rather than the whole feature.
            return ProbeResult.UNSCORABLE;
        }
        if (Float.floatToIntBits(theirs) == Float.floatToIntBits(ours)) {
            return ProbeResult.MATCHED;
        }
        final float allowed = PROBE_TOLERANCE_ULPS * Math.ulp(Math.max(Math.abs(theirs), Math.abs(ours)));
        if (Float.isFinite(theirs) && Float.isFinite(ours) && Math.abs(theirs - ours) <= allowed) {
            return ProbeResult.MATCHED;
        }
        log.debug(
            "The vector cache declines {}: it scores ordinal {} as {} where this kernel gets {}",
            delegate.getClass().getSimpleName(),
            node,
            theirs,
            ours
        );
        return ProbeResult.MISMATCHED;
    }
}

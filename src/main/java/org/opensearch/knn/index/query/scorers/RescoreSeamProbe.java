/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.scorers;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.hnsw.FlatVectorScorerUtil;
import org.apache.lucene.codecs.lucene95.HasIndexSlice;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.VectorScorer;
import org.apache.lucene.store.FilterIndexInput;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.MemorySegmentAccessInput;
import org.apache.lucene.util.Bits;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.opensearch.knn.index.codec.scorer.HasFullPrecisionVectorValues;

import java.io.IOException;
import java.lang.reflect.Method;
import java.util.Locale;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * TEMPORARY Phase-0 gate instrumentation for the Direct I/O rescore seam. This class exists only to
 * answer two empirical questions on the target index and is deleted in Phase 1:
 *
 * <ol>
 *   <li>Which concrete {@code RandomVectorScorer} binds on the fp32 {@code .vec} rescore path? If it is
 *       Lucene's {@code Lucene99MemorySegmentFloatVectorScorer}, scoring reads straight out of a
 *       {@code MemorySegment} and {@link FloatVectorValues#vectorValue(int)} is never called, so a seam
 *       hook that does not displace it intercepts nothing.</li>
 *   <li>Can the seam displace that scorer by handing down its own {@link FloatVectorValues}? Mode
 *       {@code hide_slice} hands down a delegate that does not implement {@link HasIndexSlice}, which is
 *       the first term of Lucene's binding predicate in
 *       {@code Lucene99MemorySegmentFlatVectorsScorer#getRandomVectorScorer}, and counts how many
 *       {@code vectorValue(int)} calls become visible to the hook.</li>
 * </ol>
 *
 * Selected with the JVM system property {@code knn.probe.rescore_seam}:
 * {@code off} (default), {@code observe}, {@code hide_slice}. Never a node setting, so it cannot be
 * turned on by accident on a running cluster.
 */
@Log4j2
public final class RescoreSeamProbe {

    private static final String MODE = System.getProperty("knn.probe.rescore_seam", "off").toLowerCase(Locale.ROOT);
    private static final boolean HIDE_SLICE = "hide_slice".equals(MODE);
    private static final boolean OBSERVE = HIDE_SLICE || "observe".equals(MODE);

    /** One log line per (field, values class, slice) so a 200-query run does not flood the log. */
    private static final Set<String> LOGGED_ONCE = ConcurrentHashMap.newKeySet();

    /** Counts {@code vectorValue(int)} calls that reached hook-owned code. Zero means no interception. */
    private static final AtomicLong VECTOR_VALUE_CALLS = new AtomicLong();

    /** Counts scorers built through the hook, so the call count can be read per scorer. */
    private static final AtomicLong SCORERS_BUILT = new AtomicLong();

    /** Counter cadence, chosen so a 200-query run yields a readable handful of lines rather than a flood. */
    private static final int COUNTER_LOG_EVERY = 25;

    private RescoreSeamProbe() {}

    /**
     * The seam. Observes the binding predicate and, in {@code hide_slice} mode, returns a hook-owned
     * delegate in place of the codec's values. A no-op unless the probe is on and the mode is RESCORE.
     */
    static FloatVectorValues intercept(
        final FloatVectorValues values,
        final float[] target,
        final VectorScorerMode vectorScorerMode,
        final FieldInfo fieldInfo
    ) {
        if (OBSERVE == false || vectorScorerMode != VectorScorerMode.RESCORE) {
            return values;
        }
        observe(values, target, fieldInfo);
        if (HIDE_SLICE == false) {
            return values;
        }
        final long scorersBuilt = SCORERS_BUILT.incrementAndGet();
        if (scorersBuilt % COUNTER_LOG_EVERY == 0) {
            log.info(
                "RESCORE-SEAM-PROBE-COUNTERS mode={} scorersBuilt={} vectorValueCallsThroughHook={}",
                MODE,
                scorersBuilt,
                VECTOR_VALUE_CALLS.get()
            );
        }
        return new SliceHidingFloatVectorValues(values, fieldInfo.getVectorSimilarityFunction());
    }

    /**
     * Logs, once per segment, every term of Lucene's SIMD binding predicate plus the concrete scorer
     * class an identical {@code getRandomVectorScorer} call selects.
     */
    private static void observe(final FloatVectorValues values, final float[] target, final FieldInfo fieldInfo) {
        try {
            final KnnVectorValues fullPrecision = values instanceof HasFullPrecisionVectorValues hasFullPrecision
                ? hasFullPrecision.getFullPrecisionVectorValues()
                : values;
            final IndexInput slice = fullPrecision instanceof HasIndexSlice hasIndexSlice ? hasIndexSlice.getSlice() : null;

            final String key = fieldInfo.name + "|" + values.getClass().getName() + "|" + slice;
            if (LOGGED_ONCE.add(key) == false) {
                return;
            }

            final IndexInput unwrapped = slice == null ? null : FilterIndexInput.unwrapOnlyTest(slice);
            final boolean isMemorySegmentAccessInput = unwrapped instanceof MemorySegmentAccessInput;
            // segmentSliceOrNull returns a MemorySegment, which is a preview API at this build's source
            // level 21, so it is reached reflectively through the (public) interface.
            String wholeSliceSegment = "n/a";
            if (isMemorySegmentAccessInput) {
                final Method segmentSliceOrNull = MemorySegmentAccessInput.class.getMethod("segmentSliceOrNull", long.class, long.class);
                final Object segment = segmentSliceOrNull.invoke(unwrapped, 0L, unwrapped.length());
                wholeSliceSegment = segment == null ? "null" : "non-null";
            }

            // The real path calls exactly this method with exactly these arguments (via
            // PrefetchableFlatVectorScorer, which only wraps the result and does not affect selection),
            // so the class named here is the class that binds.
            String boundScorerClass = "not-attempted";
            if (fullPrecision instanceof FloatVectorValues fullPrecisionFloats) {
                final RandomVectorScorer probeScorer = FlatVectorScorerUtil.getLucene99FlatVectorsScorer()
                    .getRandomVectorScorer(fieldInfo.getVectorSimilarityFunction(), fullPrecisionFloats, target);
                boundScorerClass = probeScorer.getClass().getName();
            }

            log.info(
                "RESCORE-SEAM-PROBE mode={} field={} valuesClass={} fullPrecisionClass={} sliceClass={} unwrappedClass={} "
                    + "isMemorySegmentAccessInput={} segmentSliceOrNull(0,length)={} sliceLength={} size={} vectorByteLength={} "
                    + "expectedBytes={} similarity={} luceneFlatScorer={} boundScorerClass={}",
                MODE,
                fieldInfo.name,
                values.getClass().getName(),
                fullPrecision == null ? "null" : fullPrecision.getClass().getName(),
                slice == null ? "null" : slice.getClass().getName(),
                unwrapped == null ? "null" : unwrapped.getClass().getName(),
                isMemorySegmentAccessInput,
                wholeSliceSegment,
                slice == null ? -1L : slice.length(),
                values.size(),
                values.getVectorByteLength(),
                (long) values.size() * values.getVectorByteLength(),
                fieldInfo.getVectorSimilarityFunction(),
                FlatVectorScorerUtil.getLucene99FlatVectorsScorer().getClass().getName(),
                boundScorerClass
            );
        } catch (Exception e) {
            log.warn("RESCORE-SEAM-PROBE failed to observe the seam", e);
        }
    }

    /**
     * A hook-owned {@link FloatVectorValues} that deliberately does <em>not</em> implement
     * {@link HasIndexSlice}. That alone fails the first term of Lucene's SIMD binding predicate, so
     * {@code Lucene99MemorySegmentFlatVectorsScorer} falls through to its scalar delegate, which reads
     * every vector through {@link #vectorValue(int)} — i.e. through code the hook owns. This is the
     * shape a Direct I/O loader will take in Phase 2; here it only counts the reads it would serve.
     */
    private static final class SliceHidingFloatVectorValues extends FloatVectorValues {

        private final FloatVectorValues delegate;
        private final VectorSimilarityFunction similarityFunction;

        private SliceHidingFloatVectorValues(final FloatVectorValues delegate, final VectorSimilarityFunction similarityFunction) {
            this.delegate = delegate;
            this.similarityFunction = similarityFunction;
        }

        @Override
        public float[] vectorValue(final int ord) throws IOException {
            VECTOR_VALUE_CALLS.incrementAndGet();
            return delegate.vectorValue(ord);
        }

        @Override
        public FloatVectorValues copy() throws IOException {
            return new SliceHidingFloatVectorValues(delegate.copy(), similarityFunction);
        }

        @Override
        public int dimension() {
            return delegate.dimension();
        }

        @Override
        public int size() {
            return delegate.size();
        }

        @Override
        public int getVectorByteLength() {
            return delegate.getVectorByteLength();
        }

        @Override
        public VectorEncoding getEncoding() {
            return delegate.getEncoding();
        }

        @Override
        public int ordToDoc(final int ord) {
            return delegate.ordToDoc(ord);
        }

        @Override
        public Bits getAcceptOrds(final Bits acceptDocs) {
            return delegate.getAcceptOrds(acceptDocs);
        }

        @Override
        public DocIndexIterator iterator() {
            return delegate.iterator();
        }

        /**
         * Mirrors {@code OffHeapFloatVectorValues.DenseOffHeapVectorValues#scorer}: score over a private
         * copy so the iterator this scorer advances is not shared. {@code rescorer(float[])} inherits
         * this, since {@link FloatVectorValues#rescorer(float[])} delegates to {@code scorer}.
         */
        @Override
        public VectorScorer scorer(final float[] target) throws IOException {
            final FloatVectorValues scoringCopy = copy();
            final DocIndexIterator iterator = scoringCopy.iterator();
            final RandomVectorScorer randomVectorScorer = FlatVectorScorerUtil.getLucene99FlatVectorsScorer()
                .getRandomVectorScorer(similarityFunction, scoringCopy, target);
            return new VectorScorer() {
                @Override
                public float score() throws IOException {
                    return randomVectorScorer.score(iterator.index());
                }

                @Override
                public DocIdSetIterator iterator() {
                    return iterator;
                }

                @Override
                public Bulk bulk(final DocIdSetIterator matchingDocs) {
                    return Bulk.fromRandomScorerSparse(randomVectorScorer, iterator, matchingDocs);
                }
            };
        }
    }
}

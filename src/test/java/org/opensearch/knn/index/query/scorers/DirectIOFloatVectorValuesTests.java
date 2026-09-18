/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.scorers;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.lucene95.HasIndexSlice;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocAndFloatFeatureBuffer;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.VectorScorer;
import org.apache.lucene.util.Bits;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.store.DirectIOVectorSource;
import org.opensearch.knn.index.store.VectorLoaderSource;
import org.opensearch.knn.index.store.VectorStagingArea;
import org.opensearch.knn.index.vectorvalues.TestVectorValues;

import java.util.List;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * The values wrapper the rescore path scores through, tested against a stubbed source so that what is
 * asserted is where the bytes come from rather than whether this filesystem allows {@code O_DIRECT}.
 * {@link org.opensearch.knn.index.store.DirectIOVectorSourceTests} covers the real reads.
 * <p>
 * The delegate's vectors are deliberately <b>different</b> from the source's throughout: every test that
 * claims a read came off the device would still pass with an mmap read if they agreed.
 */
public class DirectIOFloatVectorValuesTests extends KNNTestCase {

    private static final int DIMENSION = 4;

    private static final List<float[]> DELEGATE_VECTORS = List.of(
        new float[] { 0.0f, 0.0f, 0.0f, 0.0f },
        new float[] { 0.0f, 0.0f, 0.0f, 0.0f },
        new float[] { 0.0f, 0.0f, 0.0f, 0.0f }
    );

    private static final float[][] SOURCE_VECTORS = {
        { 1.0f, 2.0f, 3.0f, 4.0f },
        { 5.0f, 6.0f, 7.0f, 8.0f },
        { 9.0f, 10.0f, 11.0f, 12.0f } };

    private FloatVectorValues delegate() {
        return new TestVectorValues.PreDefinedFloatVectorValues(DELEGATE_VECTORS);
    }

    /**
     * A source of the same shape as {@link #delegate()} whose loaders answer {@link #SOURCE_VECTORS}. Each
     * {@code newLoader} call yields a distinct loader, matching the seam's contract that every loader owns
     * its own per-read state. The loaders are {@link DirectIOVectorSource.Reader} mocks, so they implement
     * the staging seam as well as the loader seam.
     */
    @SneakyThrows
    private VectorLoaderSource stubSource() {
        return stubSource(new java.util.ArrayList<>());
    }

    /** As {@link #stubSource()}, collecting each loader it hands out in {@code loaders}. */
    @SneakyThrows
    private VectorLoaderSource stubSource(final List<DirectIOVectorSource.Reader> loaders) {
        final VectorLoaderSource source = mock(VectorLoaderSource.class);
        when(source.size()).thenReturn(DELEGATE_VECTORS.size());
        when(source.dimension()).thenReturn(DIMENSION);
        when(source.vectorByteLength()).thenReturn(DIMENSION * Float.BYTES);
        when(source.newLoader(org.mockito.ArgumentMatchers.any())).thenAnswer(invocation -> {
            final DirectIOVectorSource.Reader reader = mock(DirectIOVectorSource.Reader.class);
            when(reader.read(org.mockito.ArgumentMatchers.anyInt())).thenAnswer(read -> SOURCE_VECTORS[(int) read.getArgument(0)]);
            loaders.add(reader);
            return reader;
        });
        return source;
    }

    private DirectIOFloatVectorValues values() {
        return new DirectIOFloatVectorValues(delegate(), stubSource(), VectorSimilarityFunction.EUCLIDEAN, VectorScorerMode.RESCORE);
    }

    @SneakyThrows
    public void testVectorValueComesFromTheSourceAndNotFromTheDelegate() {
        final FloatVectorValues delegate = mock(FloatVectorValues.class);
        final VectorLoaderSource source = stubSource();
        final DirectIOFloatVectorValues values = new DirectIOFloatVectorValues(
            delegate,
            source,
            VectorSimilarityFunction.EUCLIDEAN,
            VectorScorerMode.RESCORE
        );

        for (int ord = 0; ord < SOURCE_VECTORS.length; ord++) {
            assertArrayEquals("ordinal " + ord, SOURCE_VECTORS[ord], values.vectorValue(ord), 0.0f);
        }
        // The whole point of the class: the mmap-backed delegate is never asked for a vector.
        verify(delegate, never()).vectorValue(org.mockito.ArgumentMatchers.anyInt());
    }

    @SneakyThrows
    public void testEverythingExceptVectorValueDelegates() {
        final FloatVectorValues delegate = mock(FloatVectorValues.class);
        final Bits acceptDocs = mock(Bits.class);
        final Bits acceptOrds = mock(Bits.class);
        final FloatVectorValues.DocIndexIterator iterator = mock(FloatVectorValues.DocIndexIterator.class);
        when(delegate.dimension()).thenReturn(768);
        when(delegate.size()).thenReturn(1_000_000);
        when(delegate.getVectorByteLength()).thenReturn(3072);
        when(delegate.getEncoding()).thenReturn(VectorEncoding.FLOAT32);
        when(delegate.ordToDoc(17)).thenReturn(4242);
        when(delegate.getAcceptOrds(acceptDocs)).thenReturn(acceptOrds);
        when(delegate.iterator()).thenReturn(iterator);

        final DirectIOFloatVectorValues values = new DirectIOFloatVectorValues(
            delegate,
            stubSource(),
            VectorSimilarityFunction.EUCLIDEAN,
            VectorScorerMode.RESCORE
        );

        assertEquals(768, values.dimension());
        assertEquals(1_000_000, values.size());
        assertEquals(3072, values.getVectorByteLength());
        assertEquals(VectorEncoding.FLOAT32, values.getEncoding());
        assertEquals(4242, values.ordToDoc(17));
        assertSame(acceptOrds, values.getAcceptOrds(acceptDocs));
        assertSame(iterator, values.iterator());
    }

    /**
     * Lucene hands each per-leaf scoring task its own copy, and those tasks run concurrently, so a copy
     * that shared a reader would share the reader's buffer. Every copy must take a fresh reader.
     */
    @SneakyThrows
    public void testCopyTakesAFreshLoaderAndAFreshDelegateCopy() {
        final FloatVectorValues delegate = mock(FloatVectorValues.class);
        final FloatVectorValues delegateCopy = mock(FloatVectorValues.class);
        when(delegate.copy()).thenReturn(delegateCopy);
        final VectorLoaderSource source = stubSource();

        final DirectIOFloatVectorValues values = new DirectIOFloatVectorValues(
            delegate,
            source,
            VectorSimilarityFunction.EUCLIDEAN,
            VectorScorerMode.RESCORE
        );
        final FloatVectorValues copy = values.copy();

        assertNotSame(values, copy);
        assertTrue(copy instanceof DirectIOFloatVectorValues);
        verify(delegate).copy();
        // one loader for the original, one for the copy, both carrying the reuse hint the values were built with
        verify(source, times(2)).newLoader(VectorScorerMode.RESCORE);
        // and the copy still reads through the source
        assertArrayEquals(SOURCE_VECTORS[1], copy.vectorValue(1), 0.0f);
    }

    /**
     * Not implementing {@code HasIndexSlice} is the mechanism that displaces Lucene's MemorySegment
     * scorer. If it were ever implemented, the scorer would bind again, the reads would go back to the
     * mapping, and nothing else would fail - so this assertion is the guard. It is written reflectively
     * because {@code instanceof HasIndexSlice} on a final class that does not implement it is a compile
     * error, which is a stronger version of the same guarantee but not one a test can assert.
     */
    public void testDoesNotImplementHasIndexSlice() {
        assertFalse(
            "Re-exposing the index slice restores the mmap reads this path exists to remove",
            HasIndexSlice.class.isAssignableFrom(DirectIOFloatVectorValues.class)
        );
    }

    /**
     * The scorer must score the bytes the source returns. With the delegate holding zeros and the source
     * holding known vectors, the expected Euclidean score is computable, and an mmap read would give the
     * score against zeros instead.
     */
    @SneakyThrows
    public void testScorerScoresTheBytesTheSourceReturns() {
        final float[] target = { 1.0f, 2.0f, 3.0f, 4.0f };
        final VectorScorer scorer = values().scorer(target);

        final DocIdSetIterator iterator = scorer.iterator();
        for (int ord = 0; ord < SOURCE_VECTORS.length; ord++) {
            assertEquals(ord, iterator.nextDoc());
            assertEquals("ordinal " + ord, VectorSimilarityFunction.EUCLIDEAN.compare(target, SOURCE_VECTORS[ord]), scorer.score(), 1e-6f);
        }
        assertEquals(DocIdSetIterator.NO_MORE_DOCS, iterator.nextDoc());
    }

    /**
     * {@code FloatVectorValues.rescorer} delegates to {@code scorer}, and the base {@code scorer} throws
     * {@link UnsupportedOperationException}. The rescore path calls {@code rescorer}, so an override that
     * only covered {@code scorer} would still be enough — but only because of that delegation, which this
     * pins.
     */
    @SneakyThrows
    public void testRescorerIsServedByTheScorerOverride() {
        final float[] target = { 1.0f, 2.0f, 3.0f, 4.0f };
        final VectorScorer rescorer = values().rescorer(target);
        assertNotNull(rescorer);
        assertEquals(0, rescorer.iterator().nextDoc());
        assertEquals(VectorSimilarityFunction.EUCLIDEAN.compare(target, SOURCE_VECTORS[0]), rescorer.score(), 1e-6f);
    }

    /**
     * The scorer must stage each batch before scoring it, because {@code bulkScore} is the only point on the
     * rescore path where more than one upcoming ordinal is known. If this wiring were dropped, every number
     * would still be right and the path would silently cost one blocking read per candidate again — which is
     * the 123 ms Phase 2 measured.
     */
    @SneakyThrows
    public void testBulkScoringStagesTheBatchOnTheReaderItScoresThrough() {
        final List<DirectIOVectorSource.Reader> readers = new java.util.ArrayList<>();
        final VectorLoaderSource source = stubSource(readers);

        final float[] target = { 1.0f, 2.0f, 3.0f, 4.0f };
        final DirectIOFloatVectorValues values = new DirectIOFloatVectorValues(
            delegate(),
            source,
            VectorSimilarityFunction.EUCLIDEAN,
            VectorScorerMode.RESCORE
        );
        final VectorScorer scorer = values.scorer(target);

        final DocAndFloatFeatureBuffer buffer = new DocAndFloatFeatureBuffer();
        final float max = scorer.bulk(null).nextDocsAndScores(DocIdSetIterator.NO_MORE_DOCS, null, buffer);

        assertEquals(SOURCE_VECTORS.length, buffer.size);
        float expectedMax = Float.NEGATIVE_INFINITY;
        for (int ord = 0; ord < SOURCE_VECTORS.length; ord++) {
            final float expected = VectorSimilarityFunction.EUCLIDEAN.compare(target, SOURCE_VECTORS[ord]);
            assertEquals("ordinal " + ord, expected, buffer.features[ord], 1e-6f);
            expectedMax = Math.max(expectedMax, expected);
        }
        assertEquals(expectedMax, max, 1e-6f);

        // The scoring copy's reader is the second one, since the values object itself took the first.
        assertEquals(2, readers.size());
        verify(readers.get(1)).stage(org.mockito.ArgumentMatchers.any(), org.mockito.ArgumentMatchers.eq(SOURCE_VECTORS.length));
        verify(readers.get(0), never()).stage(org.mockito.ArgumentMatchers.any(), org.mockito.ArgumentMatchers.anyInt());
    }

    /**
     * The loader seam and the staging seam are separate interfaces, and this is what that separation buys: a
     * loader that offers no read ahead - which is what a cache at the loader seam would most likely be - is
     * scored without it rather than refused. If {@code scorer} ever required the staging seam, introducing
     * such a loader would mean rewriting read ahead instead of implementing an interface beside it.
     */
    @SneakyThrows
    public void testAStagelessLoaderIsScoredWithoutStagingRatherThanRefused() {
        final VectorLoaderSource source = mock(VectorLoaderSource.class);
        when(source.size()).thenReturn(DELEGATE_VECTORS.size());
        when(source.dimension()).thenReturn(DIMENSION);
        when(source.vectorByteLength()).thenReturn(DIMENSION * Float.BYTES);
        // mock of the seam interface alone, so it is NOT a VectorStagingArea
        when(source.newLoader(org.mockito.ArgumentMatchers.any())).thenAnswer(invocation -> {
            final VectorLoaderSource.Loader loader = mock(VectorLoaderSource.Loader.class);
            when(loader.read(org.mockito.ArgumentMatchers.anyInt())).thenAnswer(read -> SOURCE_VECTORS[(int) read.getArgument(0)]);
            return loader;
        });
        assertFalse(VectorStagingArea.class.isAssignableFrom(VectorLoaderSource.Loader.class));

        final float[] target = { 1.0f, 2.0f, 3.0f, 4.0f };
        final DirectIOFloatVectorValues values = new DirectIOFloatVectorValues(
            delegate(),
            source,
            VectorSimilarityFunction.EUCLIDEAN,
            VectorScorerMode.RESCORE
        );

        final DocAndFloatFeatureBuffer buffer = new DocAndFloatFeatureBuffer();
        values.scorer(target).bulk(null).nextDocsAndScores(DocIdSetIterator.NO_MORE_DOCS, null, buffer);

        assertEquals(SOURCE_VECTORS.length, buffer.size);
        for (int ord = 0; ord < SOURCE_VECTORS.length; ord++) {
            assertEquals(
                "ordinal " + ord,
                VectorSimilarityFunction.EUCLIDEAN.compare(target, SOURCE_VECTORS[ord]),
                buffer.features[ord],
                1e-6f
            );
        }
    }

    @SneakyThrows
    public void testIsCompatibleWhenShapesAgree() {
        assertTrue(DirectIOFloatVectorValues.isCompatible(delegate(), stubSource()));
    }

    /**
     * A shape mismatch means the source was verified against a different field or a different segment
     * generation. Reading through it would score the wrong vectors, so the seam has to fall back.
     */
    @SneakyThrows
    public void testIsIncompatibleOnEveryShapeMismatch() {
        final VectorLoaderSource wrongSize = stubSource();
        when(wrongSize.size()).thenReturn(DELEGATE_VECTORS.size() + 1);
        assertFalse("size", DirectIOFloatVectorValues.isCompatible(delegate(), wrongSize));

        final VectorLoaderSource wrongDimension = stubSource();
        when(wrongDimension.dimension()).thenReturn(DIMENSION + 1);
        assertFalse("dimension", DirectIOFloatVectorValues.isCompatible(delegate(), wrongDimension));

        final VectorLoaderSource wrongByteLength = stubSource();
        when(wrongByteLength.vectorByteLength()).thenReturn(DIMENSION * Float.BYTES + 4);
        assertFalse("vectorByteLength", DirectIOFloatVectorValues.isCompatible(delegate(), wrongByteLength));
    }
}

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.scorers;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.lucene95.HasIndexSlice;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.VectorScorer;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.BitSet;
import org.apache.lucene.util.BitSetIterator;
import org.apache.lucene.util.FixedBitSet;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.mockito.MockedStatic;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.featureflags.KNNFeatureFlags;
import org.opensearch.knn.index.codec.scorer.PrefetchableFlatVectorScorer.PrefetchableRandomVectorScorer;
import org.opensearch.knn.index.vectorvalues.TestVectorValues;

import java.io.Closeable;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Random;

import static org.mockito.Mockito.mockStatic;

/**
 * Tests for {@link NestedBestChildVectorScorer}.
 * <p>
 * Document layout used across sections 1 and 2:
 *   children [0, 1, 2, 3, 4] → parent 5
 *   children [6, 7, 8]       → parent 9
 *   child    [10]             → parent 11
 * <p>
 * Parent bits at positions 5, 9, 11 → long value = 2592.
 */
public class NestedBestChildVectorScorerTests extends KNNTestCase {

    private static final int NUM_DOCS = 12;
    // Parent bits: positions 5, 9, 11 → 2^5 + 2^9 + 2^11 = 32 + 512 + 2048 = 2592
    private static final BitSet PARENT_BIT_SET = new FixedBitSet(new long[] { 2592 }, NUM_DOCS);

    // Scores per child doc id
    private static final Map<Integer, Float> SCORES = Map.of(
        0,
        0.3f,
        1,
        0.7f,
        2,
        0.95f,
        3,
        0.4f,
        4,
        0.1f,
        6,
        0.6f,
        7,
        0.85f,
        8,
        0.2f,
        10,
        0.5f
    );

    // The layouts of sections 3 to 5 generate their own vectors instead, in groups of known shapes.
    private static final int DIMENSION = 128;
    private static final int VECTOR_BYTES = DIMENSION * Float.BYTES;
    /** Not a multiple of the vector length and not a multiple of a block, which is the awkward real case. */
    private static final int HEADER_LENGTH = 44;
    private static final int FOOTER_LENGTH = 16;
    private static final int BATCH = VectorScorer.DEFAULT_BULK_BATCH_SIZE;

    private float[][] vectors;
    private final List<Closeable> toClose = new ArrayList<>();
    private int fixtureCount;

    @Override
    public void tearDown() throws Exception {
        for (int i = toClose.size() - 1; i >= 0; i--) {
            toClose.get(i).close();
        }
        toClose.clear();
        super.tearDown();
    }

    // ──────────────────────────────────────────────
    // Section 1: Without filter
    // ──────────────────────────────────────────────

    @SneakyThrows
    public void testWithoutFilter_iteratorReturnsSameInstance() {
        NestedBestChildVectorScorer scorer = createScorer(null);
        assertSame(scorer.iterator(), scorer.iterator());
    }

    @SneakyThrows
    public void testWithoutFilter_nextDocAndScore() {
        NestedBestChildVectorScorer scorer = createScorer(null);
        DocIdSetIterator iterator = scorer.iterator();

        // Parent 5: best child is 2 (score 0.95)
        assertEquals(2, iterator.nextDoc());
        assertEquals(0.95f, scorer.score(), 0.001f);

        // Parent 9: best child is 7 (score 0.85)
        assertEquals(7, iterator.nextDoc());
        assertEquals(0.85f, scorer.score(), 0.001f);

        // Parent 11: only child 10 (score 0.5)
        assertEquals(10, iterator.nextDoc());
        assertEquals(0.5f, scorer.score(), 0.001f);

        assertEquals(DocIdSetIterator.NO_MORE_DOCS, iterator.nextDoc());
    }

    @SneakyThrows
    public void testWithoutFilter_cost() {
        FixedBitSet vectorBits = vectorBitSet();
        long expectedCost = new BitSetIterator(vectorBits, vectorBits.length()).cost();

        NestedBestChildVectorScorer scorer = createScorer(null);
        assertEquals(expectedCost, scorer.iterator().cost());
    }

    // ──────────────────────────────────────────────
    // Section 2: With filter
    // ──────────────────────────────────────────────

    @SneakyThrows
    public void testWithFilter_iteratorReturnsSameInstance() {
        NestedBestChildVectorScorer scorer = createScorer(filterBitSet());
        assertSame(scorer.iterator(), scorer.iterator());
    }

    @SneakyThrows
    public void testWithFilter_nextDocAndScore() {
        // Filter excludes children 2 and 7 (the unfiltered best children for parents 5 and 9)
        NestedBestChildVectorScorer scorer = createScorer(filterBitSet());
        DocIdSetIterator iterator = scorer.iterator();

        // Parent 5: best accepted child is 1 (score 0.7), child 2 filtered out
        assertEquals(1, iterator.nextDoc());
        assertEquals(0.7f, scorer.score(), 0.001f);

        // Parent 9: best accepted child is 6 (score 0.6), child 7 filtered out
        assertEquals(6, iterator.nextDoc());
        assertEquals(0.6f, scorer.score(), 0.001f);

        // Parent 11: only child 10 (score 0.5)
        assertEquals(10, iterator.nextDoc());
        assertEquals(0.5f, scorer.score(), 0.001f);

        assertEquals(DocIdSetIterator.NO_MORE_DOCS, iterator.nextDoc());
    }

    @SneakyThrows
    public void testWithFilter_cost() {
        FixedBitSet filter = filterBitSet();
        FixedBitSet vectors = vectorBitSet();
        // Conjunction cost is the minimum of the two iterator costs
        long expectedCost = Math.min(
            new BitSetIterator(filter, filter.length()).cost(),
            new BitSetIterator(vectors, vectors.length()).cost()
        );

        NestedBestChildVectorScorer scorer = createScorer(filter);
        assertEquals(expectedCost, scorer.iterator().cost());
    }

    /**
     * Filter accepts doc ids that have no vector. The conjunction iterator should skip
     * those automatically, and if an entire parent group has no children with vectors,
     * that parent group should be skipped entirely.
     *
     * Layout (same parents at 5, 9, 11):
     *   Vectors exist for:  {0, 4, 8, 10}
     *   Filter accepts:     {0, 1, 2, 3, 4, 6, 7, 8, 10}  (children 1,2,3,6,7 have no vector)
     *   Effective children:  {0, 4} → parent 5,  {8} → parent 9,  {10} → parent 11
     */
    @SneakyThrows
    public void testWithFilter_filterIncludesChildrenWithoutVectors() {
        Map<Integer, Float> sparseScores = Map.of(0, 0.3f, 4, 0.9f, 8, 0.6f, 10, 0.5f);

        FixedBitSet vectorBits = new FixedBitSet(NUM_DOCS);
        sparseScores.keySet().forEach(vectorBits::set);
        DocIdSetIterator vectorIter = new BitSetIterator(vectorBits, vectorBits.length());

        VectorScorer baseScorer = new VectorScorer() {
            @Override
            public float score() throws IOException {
                int doc = vectorIter.docID();
                Float s = sparseScores.get(doc);
                if (s == null) {
                    throw new IllegalStateException("No score for doc " + doc);
                }
                return s;
            }

            @Override
            public DocIdSetIterator iterator() {
                return vectorIter;
            }
        };

        // Filter accepts children 0-4, 6-8, 10 (all children except 5/9/11 which are parents)
        FixedBitSet filterBits = new FixedBitSet(NUM_DOCS);
        for (int i : new int[] { 0, 1, 2, 3, 4, 6, 7, 8, 10 }) {
            filterBits.set(i);
        }
        DocIdSetIterator filterIter = new BitSetIterator(filterBits, filterBits.length());

        NestedBestChildVectorScorer scorer = new NestedBestChildVectorScorer(filterIter, PARENT_BIT_SET, baseScorer);
        DocIdSetIterator iterator = scorer.iterator();

        // Parent 5: children 1,2,3 have no vector and are skipped; best of {0,4} is child 4 (0.9)
        assertEquals(4, iterator.nextDoc());
        assertEquals(0.9f, scorer.score(), 0.001f);

        // Parent 9: children 6,7 have no vector and are skipped; only child 8 (0.6)
        assertEquals(8, iterator.nextDoc());
        assertEquals(0.6f, scorer.score(), 0.001f);

        // Parent 11: child 10 (0.5)
        assertEquals(10, iterator.nextDoc());
        assertEquals(0.5f, scorer.score(), 0.001f);

        assertEquals(DocIdSetIterator.NO_MORE_DOCS, iterator.nextDoc());
    }

    /**
     * Filter accepts children that have no vectors, and for one parent group none of the
     * accepted children have vectors. That parent group should be skipped entirely.
     *
     * Layout (same parents at 5, 9, 11):
     *   Vectors exist for:  {0, 10}
     *   Filter accepts:     {0, 6, 7, 8, 10}  (children 6,7,8 have no vector)
     *   Effective children:  {0} → parent 5,  {} → parent 9 (skipped),  {10} → parent 11
     */
    @SneakyThrows
    public void testWithFilter_entireParentGroupSkippedWhenNoChildrenHaveVectors() {
        Map<Integer, Float> sparseScores = Map.of(0, 0.4f, 10, 0.7f);

        FixedBitSet vectorBits = new FixedBitSet(NUM_DOCS);
        sparseScores.keySet().forEach(vectorBits::set);
        DocIdSetIterator vectorIter = new BitSetIterator(vectorBits, vectorBits.length());

        VectorScorer baseScorer = new VectorScorer() {
            @Override
            public float score() throws IOException {
                int doc = vectorIter.docID();
                Float s = sparseScores.get(doc);
                if (s == null) {
                    throw new IllegalStateException("No score for doc " + doc);
                }
                return s;
            }

            @Override
            public DocIdSetIterator iterator() {
                return vectorIter;
            }
        };

        // Filter accepts 0, 6, 7, 8, 10 — but only 0 and 10 have vectors
        FixedBitSet filterBits = new FixedBitSet(NUM_DOCS);
        for (int i : new int[] { 0, 6, 7, 8, 10 }) {
            filterBits.set(i);
        }
        DocIdSetIterator filterIter = new BitSetIterator(filterBits, filterBits.length());

        NestedBestChildVectorScorer scorer = new NestedBestChildVectorScorer(filterIter, PARENT_BIT_SET, baseScorer);
        DocIdSetIterator iterator = scorer.iterator();

        // Parent 5: only child 0 (0.4)
        assertEquals(0, iterator.nextDoc());
        assertEquals(0.4f, scorer.score(), 0.001f);

        // Parent 9: skipped entirely — children 6,7,8 accepted by filter but none have vectors
        // Parent 11: child 10 (0.7)
        assertEquals(10, iterator.nextDoc());
        assertEquals(0.7f, scorer.score(), 0.001f);

        assertEquals(DocIdSetIterator.NO_MORE_DOCS, iterator.nextDoc());
    }

    // ──────────────────────────────────────────────
    // Section 3: The batch bounds
    // ──────────────────────────────────────────────

    @SneakyThrows
    public void testBulk_whenGroupsAreScored_thenTheBatchBoundsAreTheSameSequence() {
        final int unbounded = DocIdSetIterator.NO_MORE_DOCS;

        // Three groups of two, all inside one batch: one unbounded batch covers them, the last group's refill is
        // bounded by its parent at doc 8, and the call that ends the iteration is unbounded again.
        assertEquals("three groups inside one batch", List.of(unbounded, 8, unbounded), boundsOf(Layout.groupsOf(2, 3), null, null, -1));

        // One group of 200: unbounded, then refills bounded by the parent until one comes back empty.
        assertEquals(
            "one group longer than three batches",
            List.of(unbounded, 200, 200, 200, 200, unbounded),
            boundsOf(Layout.groupsOf(200, 1), null, null, -1)
        );

        // Six groups of twenty: a batch that cuts through group 3 is refilled bounded by that group's parent.
        assertEquals(
            "six groups of twenty",
            List.of(unbounded, 83, 83, unbounded, 125, unbounded),
            boundsOf(Layout.groupsOf(20, 6), null, null, -1)
        );

        // A parent bit on doc 0, which is also a child: its group is itself, and no bound of 0 is asked.
        final Layout twoGroupsOfThree = Layout.groupsOf(3, 2);
        final FixedBitSet withBitOnZero = new FixedBitSet(twoGroupsOfThree.docCount());
        withBitOnZero.set(0);
        withBitOnZero.set(3);
        withBitOnZero.set(7);
        assertEquals("a parent bit on doc 0", List.of(unbounded, 7, unbounded), boundsOf(twoGroupsOfThree, withBitOnZero, null, -1));

        // The same, with doc 0 the only entry of its batch.
        final Layout oneChild = Layout.groupsOf(1, 1);
        final FixedBitSet aloneOnZero = new FixedBitSet(oneChild.docCount());
        aloneOnZero.set(0);
        aloneOnZero.set(1);
        assertEquals("doc 0 alone in its batch", List.of(unbounded, unbounded), boundsOf(oneChild, aloneOnZero, null, -1));

        // A filter that drops the second child of every group asks for the same bounds as no filter at all.
        final Layout threeGroupsOfTwo = Layout.groupsOf(2, 3);
        final FixedBitSet accepted = new FixedBitSet(threeGroupsOfTwo.docCount());
        for (int ordinal = 0; ordinal < threeGroupsOfTwo.childCount(); ordinal += 2) {
            accepted.set(threeGroupsOfTwo.docOf(ordinal));
        }
        assertEquals(
            "a filter does not change the bounds",
            List.of(unbounded, 8, unbounded),
            boundsOf(threeGroupsOfTwo, null, accepted, -1)
        );

        // A batch view that fails on its second call is not asked again.
        assertEquals("a failure ends the sequence", List.of(unbounded, 8), boundsOf(Layout.groupsOf(2, 3), null, null, 2));
    }

    // ──────────────────────────────────────────────
    // Section 4: What a group declares to the storage layer
    // ──────────────────────────────────────────────

    public void testPrefetch_whenGroupScoredThroughBulk_thenGroupsVectorsAreDeclaredOnce() throws IOException {
        final Layout layout = Layout.groupsOf(20, 1);
        final SliceBackedValues values = sliceValues(1L, layout);
        final float[] target = vectors[0].clone();

        final NestedBestChildVectorScorer scorer = new NestedBestChildVectorScorer(null, layout.parentBitSet(), bulkScorer(values, target));
        final DocIdSetIterator iterator = scorer.iterator();
        assertEquals("the group's best child", layout.docOf(expectedBest(target, 0, 20)), iterator.nextDoc());
        assertEquals(DocIdSetIterator.NO_MORE_DOCS, iterator.nextDoc());

        final List<TrackingIndexInput.PrefetchCall> calls = values.tracking.prefetchCalls;
        assertEquals("the group was declared once", 1, calls.size());
        assertEquals("from the first child's vector", 0L, calls.get(0).offset());
        assertEquals("spanning all twenty", 20L * VECTOR_BYTES, calls.get(0).length());
    }

    public void testPrefetch_whenSeveralSmallGroups_thenOneBatchDeclaresThemTogether() throws IOException {
        final Layout layout = Layout.groupsOf(20, 6);
        final SliceBackedValues values = sliceValues(2L, layout);
        final float[] target = vectors[0].clone();

        final List<Integer> docs = new ArrayList<>();
        final List<Float> scores = new ArrayList<>();
        collect(bulkScorer(values, target), layout.parentBitSet(), null, docs, scores);

        assertEquals(6, docs.size());
        for (int group = 0; group < 6; group++) {
            assertEquals(
                "best child of group " + group,
                (Integer) layout.docOf(expectedBest(target, group * 20, (group + 1) * 20)),
                docs.get(group)
            );
        }
        // 120 children: an unbounded batch of 64, one refill bounded by the group that batch cut through, then an
        // unbounded batch for the rest.
        final List<TrackingIndexInput.PrefetchCall> calls = values.tracking.prefetchCalls;
        assertTrue("one declared span covers several groups: " + calls.size(), calls.size() < 6);
        assertEquals("three spans for six groups", 3, calls.size());
    }

    public void testPrefetch_whenGroupLongerThanOneBatch_thenRefillsBoundedByTheParent() throws IOException {
        final Layout layout = Layout.groupsOf(200, 2);
        final SliceBackedValues values = sliceValues(3L, layout);
        final float[] target = vectors[137].clone();

        final List<Integer> docs = new ArrayList<>();
        final List<Float> scores = new ArrayList<>();
        collect(bulkScorer(values, target), layout.parentBitSet(), null, docs, scores);

        assertEquals("two groups, not one per batch", 2, docs.size());
        assertEquals((Integer) layout.docOf(expectedBest(target, 0, 200)), docs.get(0));
        assertEquals((Integer) layout.docOf(expectedBest(target, 200, 400)), docs.get(1));

        final List<TrackingIndexInput.PrefetchCall> calls = values.tracking.prefetchCalls;
        // Each group of 200 is read as 64, 64, 64 then the 8 left below its parent.
        assertEquals("four spans a group", 8, calls.size());
        final long sliceLength = (long) layout.childCount() * VECTOR_BYTES;
        for (final TrackingIndexInput.PrefetchCall call : calls) {
            assertTrue("a span starts inside the slice: " + call, call.offset() >= 0L && call.offset() < sliceLength);
            assertTrue("and ends inside it: " + call, call.offset() + call.length() <= sliceLength);
            assertTrue("and is bounded by one batch of vectors: " + call, call.length() <= (long) BATCH * VECTOR_BYTES);
        }
    }

    public void testPrefetch_whenFlagDisabled_thenNoPrefetchAndSameScores() throws IOException {
        final Layout layout = Layout.groupsOf(20, 3);

        final List<Integer> enabledDocs = new ArrayList<>();
        final List<Float> enabledScores = new ArrayList<>();
        final int enabledCalls;
        try (MockedStatic<KNNFeatureFlags> flags = mockStatic(KNNFeatureFlags.class)) {
            flags.when(KNNFeatureFlags::isPrefetchEnabled).thenReturn(true);
            final SliceBackedValues values = sliceValues(4L, layout);
            collect(bulkScorer(values, vectors[0].clone()), layout.parentBitSet(), null, enabledDocs, enabledScores);
            enabledCalls = values.tracking.prefetchCalls.size();
        }
        assertTrue("with the flag on the group is declared", enabledCalls > 0);

        final List<Integer> disabledDocs = new ArrayList<>();
        final List<Float> disabledScores = new ArrayList<>();
        try (MockedStatic<KNNFeatureFlags> flags = mockStatic(KNNFeatureFlags.class)) {
            flags.when(KNNFeatureFlags::isPrefetchEnabled).thenReturn(false);
            final SliceBackedValues values = sliceValues(4L, layout);
            collect(bulkScorer(values, vectors[0].clone()), layout.parentBitSet(), null, disabledDocs, disabledScores);
            assertEquals("with the flag off nothing is declared", 0, values.tracking.prefetchCalls.size());
        }

        assertEquals(3, disabledDocs.size());
        assertEquals(enabledDocs, disabledDocs);
        assertEquals(enabledScores, disabledScores);
    }

    // ──────────────────────────────────────────────
    // Section 5: That the grouping and the scores did not move
    // ──────────────────────────────────────────────

    public void testBulk_whenGroupsAreScoredInBatches_thenSameAsTheOneAtATimeLoop() throws IOException {
        final Layout layout = Layout.groupsOf(7, 40);
        seedVectors(5L, layout.childCount());
        assertBothArmsAgree(layout, vectors[3].clone(), null, 40);
    }

    public void testBulk_whenAGroupTies_thenTheFirstMaximumWins() throws IOException {
        final Layout layout = Layout.groupsOf(4, 1);
        seedVectors(6L, layout.childCount());
        // Children 1 and 2 carry the same vector, so they score the same against it.
        vectors[2] = vectors[1].clone();
        final float[] target = vectors[1].clone();

        final List<Integer> docs = new ArrayList<>();
        final List<Float> scores = new ArrayList<>();
        collect(bulkScorer(heapValues(layout), target), layout.parentBitSet(), null, docs, scores);

        assertEquals(1, docs.size());
        assertEquals("the earlier of the two tied children", (Integer) layout.docOf(1), docs.get(0));
    }

    public void testBulk_whenAGroupIsLargerThanOneBatch_thenItIsStillOneGroup() throws IOException {
        final Layout layout = Layout.groupsOf(200, 2);
        seedVectors(7L, layout.childCount());
        final List<Integer> bulkDocs = assertBothArmsAgree(layout, vectors[137].clone(), null, 2);
        assertEquals("the maximum sits in the third batch of its group", (Integer) layout.docOf(137), bulkDocs.get(0));
    }

    public void testBulk_whenAGroupBoundaryFallsInsideABatch_thenLaterGroupsKeepTheirChildren() throws IOException {
        // Groups of 13 put a boundary at every awkward place inside a batch of 64.
        final Layout layout = Layout.groupsOf(13, 30);
        seedVectors(8L, layout.childCount());
        assertBothArmsAgree(layout, vectors[5].clone(), null, 30);
    }

    public void testBulk_whenADocCarriesAParentBit_thenItIsAGroupOfOne() throws IOException {
        final Layout layout = Layout.groupsOf(3, 2);
        seedVectors(9L, layout.childCount());
        // Parent bits on doc 0, which is also child ordinal 0, and on the two real parents.
        final FixedBitSet parents = new FixedBitSet(layout.docCount());
        parents.set(0);
        for (int group = 0; group < 2; group++) {
            parents.set(group * 4 + 3);
        }
        final float[] target = vectors[1].clone();

        final NestedBestChildVectorScorer scorer = new NestedBestChildVectorScorer(null, parents, bulkScorer(heapValues(layout), target));
        final DocIdSetIterator iterator = scorer.iterator();
        assertEquals("doc 0 is its own group", 0, iterator.nextDoc());
        assertEquals("then children 1 and 2 under parent 3", layout.docOf(expectedBest(target, 1, 3)), iterator.nextDoc());
        assertEquals("then the second group", layout.docOf(expectedBest(target, 3, 6)), iterator.nextDoc());
        assertEquals(DocIdSetIterator.NO_MORE_DOCS, iterator.nextDoc());
    }

    public void testWithFilter_whenScoredThroughBulk_thenSameAsTheOneAtATimeLoop() throws IOException {
        final Layout layout = Layout.groupsOf(11, 20);
        seedVectors(10L, layout.childCount());

        final FixedBitSet accepted = new FixedBitSet(layout.docCount());
        for (int ordinal = 0; ordinal < layout.childCount(); ordinal++) {
            if (ordinal % 3 != 0) {
                accepted.set(layout.docOf(ordinal));
            }
        }

        assertBothArmsAgree(layout, vectors[9].clone(), accepted, 20);
    }

    public void testWithFilter_whenAGroupHasNoAcceptedChildren_thenThatParentIsSkipped() throws IOException {
        final Layout layout = Layout.groupsOf(11, 20);
        seedVectors(11L, layout.childCount());
        final float[] target = vectors[9].clone();

        final int rejected = 7;
        final FixedBitSet accepted = new FixedBitSet(layout.docCount());
        for (int ordinal = 0; ordinal < layout.childCount(); ordinal++) {
            if (ordinal / layout.childrenPerGroup() != rejected) {
                accepted.set(layout.docOf(ordinal));
            }
        }

        final List<Integer> docs = new ArrayList<>();
        final List<Float> scores = new ArrayList<>();
        collect(bulkScorer(heapValues(layout), target), layout.parentBitSet(), accepted, docs, scores);

        assertEquals("nineteen groups, the rejected one skipped", 19, docs.size());
        final int firstDocOfRejected = layout.docOf(rejected * layout.childrenPerGroup());
        final int parentOfRejected = firstDocOfRejected + layout.childrenPerGroup();
        for (final int doc : docs) {
            assertTrue("no child of the rejected group was returned", doc < firstDocOfRejected || doc > parentOfRejected);
        }
        assertEquals((Integer) layout.docOf(expectedBest(target, (rejected - 1) * 11, rejected * 11)), docs.get(rejected - 1));
        assertEquals((Integer) layout.docOf(expectedBest(target, (rejected + 1) * 11, (rejected + 2) * 11)), docs.get(rejected));
    }

    public void testBulk_whenABatchFails_thenTheScorerIsExhausted() throws IOException {
        final Layout layout = Layout.groupsOf(5, 3);
        seedVectors(12L, layout.childCount());
        final FloatVectorValues values = heapValues(layout);

        final VectorScorer failing = new VectorScorer() {
            private final KnnVectorValues.DocIndexIterator iterator = values.iterator();

            @Override
            public float score() {
                return 0.0f;
            }

            @Override
            public DocIdSetIterator iterator() {
                return iterator;
            }

            @Override
            public VectorScorer.Bulk bulk(final DocIdSetIterator matchingDocs) {
                return (upTo, acceptDocs, buffer) -> {
                    // What Lucene does, in order: ordinals into the buffer, the size published, then the score.
                    buffer.growNoCopy(4);
                    for (int i = 0; i < 4; i++) {
                        buffer.docs[i] = i;
                        buffer.features[i] = 0.0f;
                    }
                    buffer.size = 4;
                    throw new IOException("the read failed");
                };
            }
        };

        final NestedBestChildVectorScorer scorer = new NestedBestChildVectorScorer(null, layout.parentBitSet(), failing);
        final DocIdSetIterator iterator = scorer.iterator();
        expectThrows(IOException.class, iterator::nextDoc);
        assertEquals(
            "a swallowed failure must not resume into the ordinals the batch left",
            DocIdSetIterator.NO_MORE_DOCS,
            iterator.nextDoc()
        );
    }

    public void testBulk_whenExhausted_thenTheBatchViewIsNotAskedAgain() throws IOException {
        final Layout layout = Layout.groupsOf(4, 2);
        seedVectors(17L, layout.childCount());
        final VectorScorer delegate = bulkScorer(heapValues(layout), vectors[0].clone());
        final int[] calls = new int[1];

        final VectorScorer counting = new VectorScorer() {
            @Override
            public float score() throws IOException {
                return delegate.score();
            }

            @Override
            public DocIdSetIterator iterator() {
                return delegate.iterator();
            }

            @Override
            public Bulk bulk(final DocIdSetIterator matchingDocs) throws IOException {
                final Bulk inner = delegate.bulk(matchingDocs);
                return (upTo, liveDocs, buffer) -> {
                    calls[0]++;
                    return inner.nextDocsAndScores(upTo, liveDocs, buffer);
                };
            }
        };

        final NestedBestChildVectorScorer scorer = new NestedBestChildVectorScorer(null, layout.parentBitSet(), counting);
        final DocIdSetIterator iterator = scorer.iterator();
        final float[] target = vectors[0].clone();
        assertEquals(layout.docOf(expectedBest(target, 0, 4)), iterator.nextDoc());
        assertEquals(layout.docOf(expectedBest(target, 4, 8)), iterator.nextDoc());
        assertEquals(DocIdSetIterator.NO_MORE_DOCS, iterator.nextDoc());
        final int whenExhausted = calls[0];
        assertEquals(DocIdSetIterator.NO_MORE_DOCS, iterator.nextDoc());
        assertEquals(DocIdSetIterator.NO_MORE_DOCS, iterator.nextDoc());
        assertEquals("a spent batch view was asked again", whenExhausted, calls[0]);
        assertEquals(DocIdSetIterator.NO_MORE_DOCS, iterator.docID());
    }

    public void testWithFilter_whenTheFilterCostsMoreThanTheVectors_thenCostIsTheSmaller() throws IOException {
        final Layout layout = Layout.groupsOf(8, 5);
        seedVectors(15L, layout.childCount());

        final FixedBitSet accepted = new FixedBitSet(layout.docCount());
        for (int ordinal = 0; ordinal < layout.childCount(); ordinal += 2) {
            accepted.set(layout.docOf(ordinal));
        }
        // A filter that claims to cost ten times the vector iterator, which is what the two cost cases above
        // cannot tell apart from the conjunction's own cost.
        final BitSetIterator expensive = new BitSetIterator(accepted, 10L * layout.childCount());
        final NestedBestChildVectorScorer scorer = new NestedBestChildVectorScorer(
            expensive,
            layout.parentBitSet(),
            bulkScorer(heapValues(layout), vectors[0].clone())
        );
        assertEquals("the smaller of the two costs", layout.childCount(), scorer.iterator().cost());
    }

    // ──────────────────────────────────────────────
    // Helpers
    // ──────────────────────────────────────────────

    /**
     * The {@code upTo} of every {@link VectorScorer.Bulk#nextDocsAndScores} call a full drain of the layout makes,
     * in order. A positive {@code failOnCall} makes that call throw instead of scoring.
     */
    private static List<Integer> boundsOf(final Layout layout, final BitSet parentBits, final FixedBitSet accepted, final int failOnCall)
        throws IOException {
        final List<Integer> bounds = new ArrayList<>();
        final BitSet parents = parentBits != null ? parentBits : layout.parentBitSet();
        final DocIdSetIterator filter = accepted == null ? null : new BitSetIterator(accepted, accepted.cardinality());
        final NestedBestChildVectorScorer scorer = new NestedBestChildVectorScorer(
            filter,
            parents,
            recordingScorer(layout, bounds, failOnCall)
        );
        final DocIdSetIterator iterator = scorer.iterator();
        try {
            while (iterator.nextDoc() != DocIdSetIterator.NO_MORE_DOCS) {
                // drain
            }
        } catch (IOException failure) {
            // The failure layout. What it records is the bounds up to the failure, and the two calls below assert
            // that an exhausted scorer asks for nothing more.
        }
        assertEquals(DocIdSetIterator.NO_MORE_DOCS, iterator.nextDoc());
        assertEquals(DocIdSetIterator.NO_MORE_DOCS, iterator.nextDoc());
        return bounds;
    }

    private static VectorScorer recordingScorer(final Layout layout, final List<Integer> bounds, final int failOnCall) {
        final List<float[]> ordinalVectors = new ArrayList<>(layout.childCount());
        for (int ordinal = 0; ordinal < layout.childCount(); ordinal++) {
            ordinalVectors.add(new float[] { ordinal });
        }
        final FloatVectorValues values = new TestVectorValues.PreDefinedSparseFloatVectorValues(ordinalVectors, ordToDocOf(layout));
        final KnnVectorValues.DocIndexIterator valuesIterator = values.iterator();
        final RandomVectorScorer random = new RandomVectorScorer.AbstractRandomVectorScorer(values) {
            @Override
            public float score(final int ordinal) {
                return 1.0f / (1 + ordinal);
            }
        };
        return new VectorScorer() {
            @Override
            public float score() throws IOException {
                return random.score(valuesIterator.index());
            }

            @Override
            public DocIdSetIterator iterator() {
                return valuesIterator;
            }

            @Override
            public Bulk bulk(final DocIdSetIterator matchingDocs) {
                final Bulk inner = Bulk.fromRandomScorerSparse(random, valuesIterator, matchingDocs);
                return (upTo, liveDocs, buffer) -> {
                    bounds.add(upTo);
                    if (bounds.size() == failOnCall) {
                        throw new IOException("the read failed");
                    }
                    return inner.nextDocsAndScores(upTo, liveDocs, buffer);
                };
            }
        };
    }

    /**
     * Scores the layout through the bulk path and through the per-child loop and asserts the two agree on every
     * group's best child and its score, to the bit. Returns the bulk arm's best children.
     */
    private List<Integer> assertBothArmsAgree(final Layout layout, final float[] target, final FixedBitSet accepted, final int groups)
        throws IOException {
        final List<Integer> bulkDocs = new ArrayList<>();
        final List<Float> bulkScores = new ArrayList<>();
        collect(bulkScorer(heapValues(layout), target), layout.parentBitSet(), accepted, bulkDocs, bulkScores);

        final List<Integer> loopDocs = new ArrayList<>();
        final List<Float> loopScores = new ArrayList<>();
        collect(loopScorer(heapValues(layout), target), layout.parentBitSet(), accepted, loopDocs, loopScores);

        assertEquals(groups, bulkDocs.size());
        assertEquals(loopDocs.size(), bulkDocs.size());
        for (int group = 0; group < bulkDocs.size(); group++) {
            assertEquals("best child of group " + group, loopDocs.get(group), bulkDocs.get(group));
            assertEquals("score of group " + group, loopScores.get(group), bulkScores.get(group), 0.0f);
        }
        return bulkDocs;
    }

    /** Drains a nested scorer over the given base scorer into the doc ids and scores it yields. */
    private void collect(
        final VectorScorer base,
        final BitSet parents,
        final FixedBitSet accepted,
        final List<Integer> docs,
        final List<Float> scores
    ) throws IOException {
        final DocIdSetIterator filter = accepted == null ? null : new BitSetIterator(accepted, accepted.length());
        final NestedBestChildVectorScorer scorer = new NestedBestChildVectorScorer(filter, parents, base);
        final DocIdSetIterator iterator = scorer.iterator();
        int doc;
        while ((doc = iterator.nextDoc()) != DocIdSetIterator.NO_MORE_DOCS) {
            docs.add(doc);
            scores.add(scorer.score());
        }
    }

    /** A base scorer whose {@code bulk()} hands the batch to a prefetching random scorer, as k-NN's do. */
    private VectorScorer bulkScorer(final FloatVectorValues values, final float[] target) {
        final RandomVectorScorer random = new PrefetchableRandomVectorScorer(randomScorer(values, target));
        final KnnVectorValues.DocIndexIterator iterator = values.iterator();
        return new VectorScorer() {
            @Override
            public float score() throws IOException {
                return random.score(iterator.index());
            }

            @Override
            public DocIdSetIterator iterator() {
                return iterator;
            }

            @Override
            public Bulk bulk(final DocIdSetIterator matchingDocs) {
                return Bulk.fromRandomScorerSparse(random, iterator, matchingDocs);
            }
        };
    }

    /** A base scorer with no {@code bulk()} override, so Lucene's interface default scores child by child. */
    private VectorScorer loopScorer(final FloatVectorValues values, final float[] target) {
        final RandomVectorScorer random = randomScorer(values, target);
        final KnnVectorValues.DocIndexIterator iterator = values.iterator();
        return new VectorScorer() {
            @Override
            public float score() throws IOException {
                return random.score(iterator.index());
            }

            @Override
            public DocIdSetIterator iterator() {
                return iterator;
            }
        };
    }

    private RandomVectorScorer.AbstractRandomVectorScorer randomScorer(final FloatVectorValues values, final float[] target) {
        return new RandomVectorScorer.AbstractRandomVectorScorer(values) {
            @Override
            public float score(final int ordinal) throws IOException {
                return dot(target, values.vectorValue(ordinal));
            }
        };
    }

    private static float dot(final float[] a, final float[] b) {
        float sum = 0.0f;
        for (int i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    /** The ordinal of the best-scoring child in {@code [from, to)} against the target, first maximum winning. */
    private int expectedBest(final float[] target, final int from, final int to) {
        int best = from;
        float bestScore = dot(target, vectors[from]);
        for (int ordinal = from + 1; ordinal < to; ordinal++) {
            final float score = dot(target, vectors[ordinal]);
            if (score > bestScore) {
                bestScore = score;
                best = ordinal;
            }
        }
        return best;
    }

    private void seedVectors(final long seed, final int vectorCount) {
        final Random random = new Random(seed);
        vectors = new float[vectorCount][DIMENSION];
        for (final float[] vector : vectors) {
            for (int i = 0; i < DIMENSION; i++) {
                vector[i] = (random.nextFloat() - 0.5f) * 8.0f;
            }
        }
    }

    /** The seeded vectors on the heap, laid out so only a child doc id carries one. */
    private FloatVectorValues heapValues(final Layout layout) {
        return new TestVectorValues.PreDefinedSparseFloatVectorValues(Arrays.asList(vectors), ordToDocOf(layout));
    }

    private static int[] ordToDocOf(final Layout layout) {
        final int[] ordToDoc = new int[layout.childCount()];
        for (int ordinal = 0; ordinal < layout.childCount(); ordinal++) {
            ordToDoc[ordinal] = layout.docOf(ordinal);
        }
        return ordToDoc;
    }

    /**
     * Writes the seeded vectors into a directory behind a header and a footer, slices out just the vectors and
     * wraps that slice in a tracking input, so the slice's base offset is not zero and the prefetch offsets the
     * bulk path issues are relative to the slice, as they are on a node.
     */
    private SliceBackedValues sliceValues(final long seed, final Layout layout) throws IOException {
        seedVectors(seed, layout.childCount());
        final String name = "vectors-" + (fixtureCount++);
        final Directory directory = new ByteBuffersDirectory();
        toClose.add(directory);
        try (IndexOutput out = directory.createOutput(name, IOContext.DEFAULT)) {
            out.writeBytes(filler(HEADER_LENGTH, 0xA0), HEADER_LENGTH);
            final byte[] one = new byte[VECTOR_BYTES];
            final ByteBuffer buffer = ByteBuffer.wrap(one).order(ByteOrder.LITTLE_ENDIAN);
            for (final float[] vector : vectors) {
                buffer.clear();
                for (final float value : vector) {
                    buffer.putFloat(value);
                }
                out.writeBytes(one, VECTOR_BYTES);
            }
            out.writeBytes(filler(FOOTER_LENGTH, 0xF0), FOOTER_LENGTH);
        }
        final IndexInput root = directory.openInput(name, IOContext.DEFAULT);
        toClose.add(root);
        final TrackingIndexInput tracking = new TrackingIndexInput(
            root.slice("vectors", HEADER_LENGTH, (long) layout.childCount() * VECTOR_BYTES)
        );
        toClose.add(tracking);
        return new SliceBackedValues(tracking, layout);
    }

    private static byte[] filler(final int length, final int base) {
        final byte[] bytes = new byte[length];
        for (int i = 0; i < length; i++) {
            bytes[i] = (byte) (base + (i & 0x0F));
        }
        return bytes;
    }

    /** Children in runs of {@code childrenPerGroup}, each run followed by its parent doc. */
    private record Layout(int childrenPerGroup, int groupCount) {
        static Layout groupsOf(final int childrenPerGroup, final int groupCount) {
            return new Layout(childrenPerGroup, groupCount);
        }

        int childCount() {
            return childrenPerGroup * groupCount;
        }

        int docCount() {
            return groupCount * (childrenPerGroup + 1);
        }

        int docOf(final int ordinal) {
            final int group = ordinal / childrenPerGroup;
            return group * (childrenPerGroup + 1) + (ordinal % childrenPerGroup);
        }

        BitSet parentBitSet() {
            final FixedBitSet bits = new FixedBitSet(docCount());
            for (int group = 0; group < groupCount; group++) {
                bits.set(group * (childrenPerGroup + 1) + childrenPerGroup);
            }
            return bits;
        }
    }

    /** All child doc ids that have vectors: {0,1,2,3,4,6,7,8,10}. */
    private static FixedBitSet vectorBitSet() {
        FixedBitSet bits = new FixedBitSet(NUM_DOCS);
        SCORES.keySet().forEach(bits::set);
        return bits;
    }

    /** Accepted children filter: all vector children except 2 and 7. */
    private static FixedBitSet filterBitSet() {
        FixedBitSet bits = vectorBitSet();
        bits.clear(2);
        bits.clear(7);
        return bits;
    }

    /**
     * Creates a {@link NestedBestChildVectorScorer} using real {@link BitSetIterator}s.
     *
     * @param filterBits if non-null, used as the accepted children filter
     */
    private static NestedBestChildVectorScorer createScorer(FixedBitSet filterBits) throws IOException {
        FixedBitSet vectorBits = vectorBitSet();
        DocIdSetIterator vectorIter = new BitSetIterator(vectorBits, vectorBits.length());

        VectorScorer baseScorer = new VectorScorer() {
            @Override
            public float score() throws IOException {
                int doc = vectorIter.docID();
                Float s = SCORES.get(doc);
                if (s == null) {
                    throw new IllegalStateException("No score for doc " + doc);
                }
                return s;
            }

            @Override
            public DocIdSetIterator iterator() {
                return vectorIter;
            }
        };

        DocIdSetIterator filterIter = filterBits != null ? new BitSetIterator(filterBits, filterBits.length()) : null;
        return new NestedBestChildVectorScorer(filterIter, PARENT_BIT_SET, baseScorer);
    }

    /**
     * The seeded vectors read from an index slice, naming that slice through {@link HasIndexSlice} so the prefetch
     * the bulk path issues reaches the input the way it does on a node.
     */
    private final class SliceBackedValues extends FloatVectorValues implements HasIndexSlice {
        private final TrackingIndexInput tracking;
        private final Layout layout;
        private final float[] scratch = new float[DIMENSION];

        private SliceBackedValues(final TrackingIndexInput tracking, final Layout layout) {
            this.tracking = tracking;
            this.layout = layout;
        }

        @Override
        public IndexInput getSlice() {
            return tracking;
        }

        @Override
        public float[] vectorValue(final int ordinal) throws IOException {
            tracking.seek((long) ordinal * VECTOR_BYTES);
            tracking.readFloats(scratch, 0, DIMENSION);
            return scratch;
        }

        @Override
        public FloatVectorValues copy() {
            return this;
        }

        @Override
        public int dimension() {
            return DIMENSION;
        }

        @Override
        public int size() {
            return layout.childCount();
        }

        @Override
        public int getVectorByteLength() {
            return VECTOR_BYTES;
        }

        @Override
        public int ordToDoc(final int ordinal) {
            return layout.docOf(ordinal);
        }

        @Override
        public DocIndexIterator iterator() {
            return createSparseIterator();
        }

        @Override
        public VectorEncoding getEncoding() {
            return VectorEncoding.FLOAT32;
        }
    }
}

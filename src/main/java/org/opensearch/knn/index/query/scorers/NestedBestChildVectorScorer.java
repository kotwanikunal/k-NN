/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.query.scorers;

import org.apache.lucene.search.DocAndFloatFeatureBuffer;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.VectorScorer;
import org.apache.lucene.util.BitSet;
import org.opensearch.common.Nullable;

import java.io.IOException;

/**
 * A {@link VectorScorer} for nested documents that groups children by their parent and yields the best scoring
 * child of each parent. It follows Lucene's {@code DiversifyingChildrenVectorScorer}, as a standalone scorer for
 * OpenSearch's exact search path that scores a group through the child scorer's {@link VectorScorer.Bulk} view, so
 * that the child scorer declares the group's vectors to the storage layer before it reads them. Unlike Lucene's,
 * it keeps a group's first child whatever it scores, so every group yields a child of its own.
 *
 * <p>Of that view it needs two things, which every one Lucene builds provides: a non-empty batch ascends in doc id
 * and starts above the last doc consumed, and an empty batch means nothing is left below the bound. Group
 * boundaries are read off the batch, so a view that broke either would mis-group the children.
 *
 * @see org.apache.lucene.search.VectorScorer
 * @see org.apache.lucene.search.join.DiversifyingChildrenFloatKnnVectorQuery
 */
class NestedBestChildVectorScorer implements VectorScorer {
    private final DocAndFloatFeatureBuffer buffer = new DocAndFloatFeatureBuffer();
    private final VectorScorer.Bulk bulkScorer;
    private final BitSet parentBitSet;
    private final DocIdSetIterator iterator;
    private final long cost;

    private int currentBatchIdx = 0;
    private int bestChild = -1;
    private float currentScore = Float.NEGATIVE_INFINITY;
    private float childScore = Float.NEGATIVE_INFINITY;

    /**
     * Creates a scorer over the children of the parents in {@code parentBitSet}, optionally restricted to the
     * children a filter accepts. The child scorer's {@link VectorScorer.Bulk} view is built here and owns both
     * iterators from then on, so {@code filterIdsIterator} must not have been positioned by anyone else.
     *
     * @param filterIdsIterator    the accepted child doc ids, or {@code null} to score every child with a vector
     * @param parentBitSet         a {@link BitSet} with a bit set at every parent doc id
     * @param childrenVectorScorer the scorer that scores one child against the query vector
     * @throws IOException if the child scorer cannot build its bulk view
     */
    NestedBestChildVectorScorer(
        @Nullable final DocIdSetIterator filterIdsIterator,
        final BitSet parentBitSet,
        final VectorScorer childrenVectorScorer
    ) throws IOException {
        this.parentBitSet = parentBitSet;
        // The cost of the conjunction the bulk view builds, taken before that view is built because the iterator it
        // builds is private to it. A conjunction costs the least of its inputs.
        final long vectorCost = childrenVectorScorer.iterator().cost();
        this.cost = filterIdsIterator == null ? vectorCost : Math.min(filterIdsIterator.cost(), vectorCost);
        this.bulkScorer = childrenVectorScorer.bulk(filterIdsIterator);
        this.iterator = createIterator();
    }

    /**
     * Returns the score of the best-scoring child for the current parent group.
     * Only valid after a successful call to {@code iterator().nextDoc()}.
     */
    @Override
    public float score() throws IOException {
        return currentScore;
    }

    /**
     * Returns a {@link DocIdSetIterator} whose {@code nextDoc()} yields the doc id of the
     * best-scoring child for each successive parent. The same instance is returned on every call.
     */
    @Override
    public DocIdSetIterator iterator() {
        return iterator;
    }

    /**
     * Hands out the next scored child below {@code upTo}, exclusive, refilling the batch when it is spent, and
     * returns its doc id with its score left in {@link #childScore}. Returns -1 when there is no such child.
     */
    private int nextChildBelow(final int upTo) throws IOException {
        if (currentBatchIdx >= buffer.size) {
            fillBuffer(upTo);
            if (buffer.size == 0) {
                return -1;
            }
        }
        if (buffer.docs[currentBatchIdx] >= upTo) {
            // It belongs to a later group, so it stays in the buffer for the next call.
            return -1;
        }
        childScore = buffer.features[currentBatchIdx];
        return buffer.docs[currentBatchIdx++];
    }

    /**
     * Fills {@link #buffer} with the next batch of scored children below {@code upTo}, exclusive, and resets the
     * read position into it. An empty batch means there are no children left below {@code upTo}. {@code upTo} must
     * be positive, which Lucene asserts.
     */
    private void fillBuffer(final int upTo) throws IOException {
        currentBatchIdx = 0;
        try {
            // The filter is already inside the bulk view's iterator, so there is no accept-docs set to apply.
            bulkScorer.nextDocsAndScores(upTo, null, buffer);
        } catch (Throwable failure) {
            // A batch publishes its size before it scores, so a failure leaves doc ids that are still ordinals.
            // Emptying the buffer leaves the scorer exhausted instead of resumable into them.
            buffer.size = 0;
            bestChild = DocIdSetIterator.NO_MORE_DOCS;
            currentScore = Float.NEGATIVE_INFINITY;
            throw failure;
        }
    }

    /**
     * Creates a {@link DocIdSetIterator} that groups children by parent and yields the
     * best-scoring child per parent. Each {@code nextDoc()} call advances through one
     * parent group and returns the doc id of the highest-scoring child within that group.
     */
    private DocIdSetIterator createIterator() {
        return new DocIdSetIterator() {
            @Override
            public int docID() {
                return bestChild;
            }

            @Override
            public int nextDoc() throws IOException {
                if (bestChild == NO_MORE_DOCS) {
                    return NO_MORE_DOCS;
                }
                // Unbounded, so one batch can cover several groups and the child scorer declares them together.
                int child = nextChildBelow(NO_MORE_DOCS);
                if (child == -1) {
                    return bestChild = NO_MORE_DOCS;
                }
                // The first child of the group is kept whatever it scores, so every group yields a document.
                bestChild = child;
                currentScore = childScore;
                final int currentParent = parentBitSet.nextSetBit(child);
                if (currentParent == child) {
                    // The doc carries a parent bit, so its group is itself. It is also what keeps doc 0 from asking
                    // for a bound of 0.
                    return bestChild;
                }
                while ((child = nextChildBelow(currentParent)) != -1) {
                    // Strictly greater, so the first child holding the maximum wins the tie.
                    if (childScore > currentScore) {
                        bestChild = child;
                        currentScore = childScore;
                    }
                }
                return bestChild;
            }

            /**
             * Not supported. This iterator returns the best-scoring child per parent group,
             * which requires evaluating <em>all</em> children within a group. Advancing to an
             * arbitrary target could land in the middle of a parent group, making it impossible
             * to consider earlier (potentially higher-scoring) children without backtracking
             * — violating the forward-only iterator contract.
             */
            @Override
            public int advance(int target) {
                throw new UnsupportedOperationException();
            }

            @Override
            public long cost() {
                return cost;
            }
        };
    }
}

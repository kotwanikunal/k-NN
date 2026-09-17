/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.scorer;

import com.google.common.annotations.VisibleForTesting;
import lombok.AccessLevel;
import lombok.NoArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.ScoreDoc;
import org.opensearch.knn.common.featureflags.KNNFeatureFlags;

import java.io.IOException;
import java.util.Arrays;

/**
 * Issues a single prefetch over a whole leaf's rescore candidate set, before the exact search over that
 * leaf begins.
 * <p>
 * This widens, rather than introduces, the prefetch that is already on the rescore path.
 * {@link PrefetchableFlatVectorScorer} already prefetches inside every bulk-scoring batch, but a batch
 * is {@code VectorScorer.Bulk.DEFAULT_BULK_BATCH_SIZE} = 64 ordinals wide and there is a hard
 * serialisation point between batches. The rescore candidate set for one leaf is {@code firstPassK}
 * entries - typically 200-500 - and it is fully materialised before the exact search runs, so the whole
 * set can be handed to the storage layer at once. That raises the achievable queue depth from 64 to the
 * whole candidate set.
 * <p>
 * Prefetching is purely advisory: it never changes which vectors are read, which scores are computed, or
 * which documents are returned. Every failure mode - an unknown field, a vector values implementation
 * that exposes no slice, or an {@link IOException} from the hint itself - degrades to doing nothing.
 * <p>
 * Controlled by {@link KNNFeatureFlags#isWholeLeafPrefetchEnabled()} (off by default) and, so that the
 * existing prefetch kill switch still turns off everything, by
 * {@link KNNFeatureFlags#isPrefetchEnabled()}.
 * <p>
 * Known gap: a codec whose {@code getFloatVectorValues} returns a two-slice wrapper that deliberately
 * does not expose {@code HasIndexSlice} - the scalar-quantized wrapper fronting both {@code .veq} and
 * {@code .vec} is the one in this plugin - cannot be prefetched from here, and
 * {@link PrefetchableVectorValuesHelper#doPrefetch} logs that it declined. The in-scorer prefetch still
 * covers those fields, because it is handed the inner full-precision values rather than the wrapper.
 */
@Log4j2
@NoArgsConstructor(access = AccessLevel.PRIVATE)
public class WholeLeafPrefetcher {

    /**
     * Prefetches the full-precision vectors of a leaf's rescore candidates.
     * <p>
     * Never throws. The number of ordinals actually handed to {@link PrefetchHelper} is returned so
     * callers and tests can tell a real prefetch from a no-op; in particular
     * {@link PrefetchHelper#prefetch} itself is a no-op below two ordinals, so a return value of
     * {@code 0} or {@code 1} means nothing was hinted.
     *
     * @param leafReader the leaf being rescored
     * @param field      the {@code knn_vector} field name
     * @param docIds     the candidate document ids, leaf-local, in any order; may contain duplicates
     * @return the number of vector ordinals passed to the prefetch, or 0 if nothing was prefetched
     */
    public static int prefetchCandidates(final LeafReader leafReader, final String field, final int[] docIds) {
        if (docIds == null || isEnabled(docIds.length) == false) {
            return 0;
        }
        try {
            final KnnVectorValues vectorValues = vectorValues(leafReader, field);
            if (vectorValues == null) {
                log.debug("Skipping whole-leaf prefetch, no vector values for field [{}]", field);
                return 0;
            }
            final int[] ords = toOrdinals(vectorValues, docIds);
            if (ords.length <= 1) {
                return 0;
            }
            PrefetchableVectorValuesHelper.doPrefetch(vectorValues, ords, ords.length);
            log.trace("Whole-leaf prefetch issued for [{}] ordinals of field [{}]", ords.length, field);
            return ords.length;
        } catch (IOException | RuntimeException e) {
            // Prefetch is a hint. Losing it must never fail a query.
            log.debug("Whole-leaf prefetch skipped for field [{}]", field, e);
            return 0;
        }
    }

    /**
     * Prefetches the full-precision vectors behind a leaf's rescore hits.
     * <p>
     * Convenience overload for the rescore call site, which holds the candidates as the
     * {@code scoreDocs} of the leaf's first-pass {@code TopDocs}. The document ids are extracted only
     * once the switches have been checked, so nothing is allocated when the feature is off.
     *
     * @param leafReader the leaf being rescored
     * @param field      the {@code knn_vector} field name
     * @param scoreDocs  the leaf's first-pass hits, holding leaf-local document ids
     * @return the number of vector ordinals passed to the prefetch, or 0 if nothing was prefetched
     */
    public static int prefetchCandidates(final LeafReader leafReader, final String field, final ScoreDoc[] scoreDocs) {
        if (scoreDocs == null || isEnabled(scoreDocs.length) == false) {
            return 0;
        }
        final int[] docIds = new int[scoreDocs.length];
        for (int i = 0; i < scoreDocs.length; i++) {
            docIds[i] = scoreDocs[i].doc;
        }
        return prefetchCandidates(leafReader, field, docIds);
    }

    /**
     * True when there are at least two candidates to hint and both the whole-leaf switch and the general
     * prefetch kill switch are on. Below two candidates {@link PrefetchHelper#prefetch} is a no-op
     * anyway, so this avoids opening vector values for nothing.
     */
    private static boolean isEnabled(final int numCandidates) {
        return numCandidates > 1 && KNNFeatureFlags.isWholeLeafPrefetchEnabled() && KNNFeatureFlags.isPrefetchEnabled();
    }

    /**
     * Returns the float vector values for the field, or the byte vector values when the field is byte or
     * binary encoded, or null when the leaf has neither.
     */
    private static KnnVectorValues vectorValues(final LeafReader leafReader, final String field) throws IOException {
        final KnnVectorValues floatValues = leafReader.getFloatVectorValues(field);
        return floatValues != null ? floatValues : leafReader.getByteVectorValues(field);
    }

    /**
     * Translates leaf-local document ids into vector ordinals.
     * <p>
     * The translation goes through the values' own {@link KnnVectorValues.DocIndexIterator}, never
     * {@code ord == docId}. Those two coincide only for a dense field with no gaps; on a sparse field, or
     * a field a document simply does not have, assuming they are equal reads a different document's
     * vector - silently, with no exception. Candidates without a vector are dropped.
     *
     * @param vectorValues the values whose ordinal space is being translated into
     * @param docIds       candidate document ids in any order, possibly with duplicates
     * @return the ordinals of those candidates that have a vector, ascending
     */
    @VisibleForTesting
    static int[] toOrdinals(final KnnVectorValues vectorValues, final int[] docIds) throws IOException {
        final int[] sortedDocIds = docIds.clone();
        Arrays.sort(sortedDocIds);

        final KnnVectorValues.DocIndexIterator iterator = vectorValues.iterator();
        final int[] ords = new int[sortedDocIds.length];
        int numOrds = 0;
        int previousDocId = -1;

        for (final int docId : sortedDocIds) {
            if (docId == previousDocId) {
                continue;
            }
            previousDocId = docId;
            // advance() requires a target strictly greater than the current position.
            int current = iterator.docID() < docId ? iterator.advance(docId) : iterator.docID();
            if (current == DocIdSetIterator.NO_MORE_DOCS) {
                break;
            }
            if (current == docId) {
                ords[numOrds++] = iterator.index();
            }
        }

        return numOrds == ords.length ? ords : Arrays.copyOf(ords, numOrds);
    }
}

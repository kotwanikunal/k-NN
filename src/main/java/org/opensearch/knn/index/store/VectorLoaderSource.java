/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import org.opensearch.knn.index.query.scorers.VectorScorerMode;

import java.io.Closeable;
import java.io.IOException;

/**
 * <b>The loader seam.</b> A pluggable byte source for one field's full-precision vectors in one segment:
 * given an ordinal, it yields that vector's floats, by whatever means it likes.
 *
 * <p>This is the extension point a future full-precision vector cache belongs at. Everything above it —
 * the rescore gate, the values wrapper, Lucene's scorer — is written against this interface and
 * {@link Loader}, so a cache can be introduced by implementing them and choosing a different
 * implementation at {@link org.opensearch.knn.index.codec.scorer.HasVectorLoaderSource}, without touching
 * the query path. {@link DirectIOVectorSource} is the only implementation today and it holds no bytes
 * between reads.
 *
 * <h2>Two levels, because they have different lifetimes</h2>
 * <ul>
 *   <li>The <b>source</b> is segment scoped: one per field per segment, opened lazily, shared by every
 *       query, closed by the codec reader that made it. Anything expensive to establish — a file handle, a
 *       cache region — belongs here.</li>
 *   <li>A <b>{@link Loader}</b> is scorer scoped: one per {@code FloatVectorValues} copy, single threaded,
 *       owning whatever per-read state it needs. Lucene hands every per-leaf scoring task its own copy of
 *       the values and runs those tasks concurrently, so per-read state shared between loaders would be a
 *       data race.</li>
 * </ul>
 *
 * <h2>Why {@link #newLoader} takes a {@link VectorScorerMode}</h2>
 * The mode is a <b>reuse hint</b>, and it is the one piece of query context this seam carries, because it
 * is the piece a caching implementation cannot do its job without:
 * <ul>
 *   <li>{@link VectorScorerMode#RESCORE} reads have approximately no reuse. A rescore pass reads each
 *       candidate's full-precision vector once and never asks again, so caching those bytes is worse than
 *       not caching them: it re-retains exactly the single-use data whose eviction pressure on the graph
 *       and the quantized vectors motivates reading them with {@code O_DIRECT} in the first place.</li>
 *   <li>{@link VectorScorerMode#SCORE} reads — traversal and the exact-search fallback — were measured at
 *       6.6x reuse, which is the profile a cache is for.</li>
 * </ul>
 * A hint is not a gate. An implementation may ignore it entirely, as {@link DirectIOVectorSource} does
 * because it retains nothing either way; what it may not do is decide caching policy without it. Note that
 * whether a given read <em>reaches</em> this seam at all is a separate decision made above it, by
 * {@code DirectIORescoreSeam}, and this hint neither widens nor narrows that gate.
 *
 * <h2>Separate from the staging seam, deliberately</h2>
 * Read ahead is {@link VectorStagingArea}, a different interface that a {@link Loader} may also implement.
 * They are not merged because their lifecycles are opposites — see {@link VectorStagingArea}'s javadoc,
 * which explains what fusing them would cost a future cache.
 */
public interface VectorLoaderSource extends Closeable {

    /**
     * A single-threaded reader over this source: "give me the floats for ordinal N".
     *
     * <p>This is the whole loader contract. It says nothing about where the bytes come from, so an
     * implementation is free to read them from a device, from a mapping, or from a cache.
     */
    interface Loader {

        /**
         * The vector at {@code ord}.
         *
         * <p>The returned array may be owned by this loader and overwritten by the next call, which is the
         * contract {@code FloatVectorValues#vectorValue(int)} already has. A caller that needs to keep it
         * copies it.
         *
         * @param ord the ordinal to read, in this source's ordinal space
         * @return the vector's floats
         * @throws IOException              if the read fails
         * @throws IllegalArgumentException if {@code ord} is not in {@code [0, size())}
         */
        float[] read(int ord) throws IOException;

        /**
         * The reuse hint this loader was created with, as given to {@link #newLoader}. Reported rather than
         * consumed so that the hint reaching the loader is observable — a seam whose one piece of context is
         * silently dropped somewhere in the middle is a seam a future cache cannot rely on.
         */
        VectorScorerMode reuseHint();
    }

    /**
     * A new loader over this source.
     *
     * <p>Must be cheap: the query path creates one for the values it returns and another for the private
     * copy the scorer actually reads through, so anything costly should be deferred to the first read.
     *
     * @param reuseHint how much reuse the reads through this loader are expected to have, as a
     *                  {@link VectorScorerMode}; see the class javadoc
     * @return a loader, never {@code null}
     */
    Loader newLoader(VectorScorerMode reuseHint);

    /** Number of vectors in the region this source serves. */
    int size();

    /** Dimension of the vectors this source serves. */
    int dimension();

    /**
     * On-disk size of one vector, in bytes. Together with {@link #size()} and {@link #dimension()} this is
     * what lets the query layer check that a source really describes the same vectors as the codec's values
     * before reading through it.
     */
    int vectorByteLength();
}

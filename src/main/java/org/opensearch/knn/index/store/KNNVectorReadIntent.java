/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import org.apache.lucene.store.DataAccessHint;
import org.apache.lucene.store.FileDataHint;
import org.apache.lucene.store.FileTypeHint;
import org.apache.lucene.store.IOContext;

/**
 * Why the plugin is opening a vector data file, expressed as a Lucene {@link IOContext.FileOpenHint}
 * so that it travels down the directory chain on the one channel that can still change how the bytes
 * are fetched: {@code Directory.openInput}.
 *
 * <p>This is the <em>what</em> signal of the directory design. A caller that knows its purpose names
 * it here; the {@code Directory} at the bottom of the chain decides the <em>how</em> (mmap, Direct
 * I/O, a cache) without the caller naming a mechanism. {@link org.apache.lucene.store.FileDataHint}
 * cannot carry it because it is a closed Lucene enum with only {@code POSTINGS} and
 * {@code KNN_VECTORS}; {@code FileOpenHint} is an open interface, so a plugin-defined hint is the
 * supported extension point.
 *
 * <p>Three properties of the Lucene API make this safe to attach:
 *
 * <ul>
 *   <li>An unrecognised hint is inert to everything that already reads hints.
 *       {@code MMapDirectory.ADVISE_BY_CONTEXT} only tests {@code hints().contains(...)} for the two
 *       {@link DataAccessHint} constants, so a context carrying this hint madvises exactly as the
 *       same context without it would.
 *   <li>{@code DefaultIOContext} rejects two hints of the same class, and an enum constant without a
 *       class body reports the enum as its class, so at most one intent can ever be attached.
 *   <li>A merge or flush context cannot carry it at all: {@code IOContext.merge(..).withHints(..)}
 *       and {@code IOContext.flush(..).withHints(..)} return {@code this} with an empty hint set. So
 *       "a merge read must never be routed to Direct I/O" is enforced by Lucene's own API rather
 *       than by a check of ours.
 * </ul>
 *
 * <p>The hint is only ever <em>added</em> to a context the plugin itself builds for an open it
 * itself issues. Lucene's own {@code .vec} open carries a hint set the plugin does not author
 * ({@code FileTypeHint.DATA}, {@code FileDataHint.KNN_VECTORS}, {@code DataAccessHint.RANDOM}) and
 * that set is identical for every reader of the file, which is why an intent has to ride on a
 * plugin-issued open rather than be read off Lucene's.
 */
public enum KNNVectorReadIntent implements IOContext.FileOpenHint {
    /**
     * A full-precision re-score read: a sparse, random gather of a few hundred vectors out of a file
     * whose working set does not fit the page cache. This is the read Direct I/O exists for.
     */
    RESCORE;

    /**
     * An {@link IOContext} for opening a full-precision vector data file with this intent.
     *
     * <p>It reproduces the hint triple Lucene attaches to its own {@code .vec} open
     * ({@code FileTypeHint.DATA}, {@code FileDataHint.KNN_VECTORS}, {@code DataAccessHint.RANDOM})
     * and adds this intent. Reproducing rather than extending is forced by the API:
     * {@code IOContext.withHints} replaces the hint set instead of merging into it, so a caller that
     * starts from a context it did not build must restate every hint it wants to keep.
     */
    public IOContext vectorDataContext() {
        return IOContext.DEFAULT.withHints(FileTypeHint.DATA, FileDataHint.KNN_VECTORS, DataAccessHint.RANDOM, this);
    }

    /**
     * The intent carried by {@code context}, or {@code null} when it carries none — which is the case
     * for every open Lucene issues, and for every merge and flush context.
     */
    public static KNNVectorReadIntent of(final IOContext context) {
        return context.hints(KNNVectorReadIntent.class).findFirst().orElse(null);
    }
}

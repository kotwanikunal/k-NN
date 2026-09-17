/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import org.apache.lucene.misc.store.DirectIODirectory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.IOContext;

import java.io.IOException;
import java.util.OptionalLong;

/**
 * A {@link DirectIODirectory} whose routing gate is narrowed to the one file the k-NN rescore path
 * reads in bulk: the flat full-precision vector file, {@code .vec}.
 * <p>
 * Lucene's default gate is merge-only — {@code context() == MERGE && estimatedMergeBytes >=
 * minBytesDirect} — so wrapping a directory in the unmodified {@link DirectIODirectory} changes
 * nothing for search, because a query's {@link IOContext} is never
 * {@link IOContext.Context#MERGE}. Overriding {@link #useDirectIO} is the extension point that
 * makes it apply to reads, and it is the only behaviour this class changes; everything else,
 * including the delegation of every non-{@code .vec} file back to the wrapped directory, is
 * inherited from {@link DirectIODirectory} and {@code FilterDirectory}.
 * <p>
 * The gate is an explicit include list rather than an exclude list, so any file type we have not
 * reasoned about keeps its current behaviour by default.
 */
public final class KNNDirectIODirectory extends DirectIODirectory {

    /**
     * The suffix of Lucene's flat vector data file. This is the file
     * {@code OffHeapFloatVectorValues} reads one vector at a time during full-precision rescoring,
     * and the only file this directory serves with Direct I/O.
     * <p>
     * Matched with a plain suffix test rather than through
     * {@link org.apache.lucene.store.FileSwitchDirectory#getExtension}, because that method
     * deliberately looks *through* a trailing {@code .tmp} and reports the inner extension — so
     * {@code _0_NativeEngines990.vec.tmp} comes back as {@code vec} there. A temporary file is a
     * write target, and Direct I/O must not follow it.
     */
    static final String VECTOR_DATA_SUFFIX = ".vec";

    /**
     * {@link DirectIODirectory} keeps its own copy of this privately, so we hold a second one for
     * the gate.
     */
    private final long minBytesDirect;

    /**
     * @param delegate      the directory that serves every file this one does not route, and the
     *                      reference for the filesystem path. Must be an {@link FSDirectory}; the
     *                      superclass casts it.
     * @param readBufferSize size of the per-{@code IndexInput} read buffer, in bytes. Named
     *                      {@code mergeBufferSize} upstream because upstream only uses it for
     *                      merges; on the read path it is the granularity of every refill, so it
     *                      is the single most important tuning knob. Callers supply it; a later
     *                      change derives it per index from the mapping's vector dimensions.
     * @param minBytesDirect files shorter than this are served by the delegate. Small segments are
     *                      likely fully page-resident already, so bypassing the page cache for
     *                      them costs syscalls and buys no cache hygiene.
     * @throws IOException if the superclass cannot read the filesystem block size
     */
    public KNNDirectIODirectory(final FSDirectory delegate, final int readBufferSize, final long minBytesDirect) throws IOException {
        super(delegate, readBufferSize, minBytesDirect);
        this.minBytesDirect = minBytesDirect;
    }

    /**
     * Routes only {@code .vec} reads above the size floor through Direct I/O.
     * <p>
     * The conditions are ordered deliberately and each is load bearing:
     * <ol>
     *   <li>An empty {@code fileLength} means the caller wants an {@link org.apache.lucene.store.IndexOutput},
     *       not an input. Returning {@code true} there would hand back Lucene's
     *       {@code DirectIOIndexOutput}, which opens with {@code CREATE_NEW} and so fails on any
     *       rewrite, and would make us write vectors through Direct I/O — neither of which this
     *       change is about. Writes always go to the delegate.</li>
     *   <li>Merges keep the delegate too. Merge reads of {@code .vec} are sequential and benefit
     *       from the page cache and read-ahead that Direct I/O throws away, and Lucene already
     *       offers its own merge-only gate for operators who want the opposite. Note this
     *       deliberately inverts the superclass' default rather than extending it.</li>
     *   <li>The include list. Anything that is not {@code .vec} — the FAISS index file, the doc-id
     *       map, {@code segments_N}, temporary files whose final extension is {@code .tmp} — is
     *       untouched.</li>
     *   <li>The size floor.</li>
     * </ol>
     * Unlike the superclass' implementation this never calls {@code context.mergeInfo()}, so it
     * cannot NPE on a non-merge context; do not "restore" that call.
     *
     * @param name       the file name, without any directory component
     * @param context    the open context
     * @param fileLength the file length when known, empty when an output is being requested
     * @return true if this file should be served with Direct I/O
     */
    @Override
    protected boolean useDirectIO(final String name, final IOContext context, final OptionalLong fileLength) {
        return shouldUseDirectIO(name, context, fileLength, minBytesDirect);
    }

    /**
     * The gate as a pure function, so it can be tested without a filesystem. See
     * {@link #useDirectIO} for why each condition is there.
     */
    static boolean shouldUseDirectIO(final String name, final IOContext context, final OptionalLong fileLength, final long minBytesDirect) {
        if (fileLength.isEmpty()) {
            return false;
        }
        if (context.context() == IOContext.Context.MERGE) {
            return false;
        }
        if (name.endsWith(VECTOR_DATA_SUFFIX) == false) {
            return false;
        }
        return fileLength.getAsLong() >= minBytesDirect;
    }
}

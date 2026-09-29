/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FilterDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;

import java.io.IOException;
import java.util.stream.Stream;

/**
 * A {@link Directory} view whose only effect is to add {@link KNNVectorReadIntent#RESCORE} to the
 * {@link IOContext} of every full-precision vector data open that passes through it, and to delegate
 * everything else untouched.
 *
 * <h2>Why a Directory view, and not a context</h2>
 * The intent has to be on the {@code IOContext} of the {@code .vec} {@code openInput} itself, because
 * that open (and the four-argument {@code slice} a compound segment substitutes for it) is the only
 * channel that still carries both the file name and the caller's hints. The obvious way to arrange
 * that — hand the codec a {@link org.apache.lucene.index.SegmentReadState} whose context already
 * carries the intent — does not work: {@code IOContext.withHints} <em>replaces</em> the hint set
 * rather than merging into it ({@code DefaultIOContext}), and Lucene's flat vectors reader calls it on
 * the state's context before opening the data file
 * ({@code Lucene99FlatVectorsReader}: {@code dataContext = state.context.withHints(hints)}). An intent
 * placed above the codec is therefore erased, silently, before it can reach a directory.
 *
 * <p>So the intent is added <em>below</em> the codec instead. A reader constructed over this view
 * issues its own opens as usual, with its own contexts, and this class adds the intent to the ones
 * that name a full-precision vector file. Nothing above needs to know, and the reader needs no
 * modification at all — which is what lets one mechanism serve every encoding whose full-precision
 * vectors live in a {@code .vec}, rather than one wrapper class per encoding.
 *
 * <h2>What it does not do</h2>
 * It names a purpose; it does not choose a mechanism. Whether an intent-carrying open becomes Direct
 * I/O, a cache lookup, or the same memory mapping it would have been is decided further down, by the
 * directory that the store factory installed — and on a node with no such directory in the chain this
 * class is inert, because an unrecognised {@link IOContext.FileOpenHint} is ignored by everything that
 * already reads hints. That is the point of the split: this class is the <em>what</em>, and the
 * storage layer keeps the <em>how</em>.
 *
 * <p>Three consequences of Lucene's API are worth stating, because they are load-bearing and free:
 *
 * <ul>
 *   <li><b>A merge or flush read can never be tagged.</b> {@code IOContext.merge(..).withHints(..)}
 *       and {@code IOContext.flush(..).withHints(..)} return the context unchanged with an empty hint
 *       set, so the exclusion of the one read pattern that must stay sequential and buffered is
 *       enforced by Lucene rather than by a check here.
 *   <li><b>The metadata opens are untouched.</b> A flat vectors reader reads its {@code .vemf}
 *       sidecar through {@code openChecksumInput}, which this class does not override, and the name
 *       test excludes it in any case.
 *   <li><b>At most one intent can ever be attached.</b> {@code DefaultIOContext} rejects two hints of
 *       the same class, so a context that already carries an intent is passed through unchanged rather
 *       than having a second added to it.
 * </ul>
 */
public final class KNNRescoreIntentDirectory extends FilterDirectory {

    /** Extension of Lucene's flat full-precision vector file — the only file this view tags. */
    static final String VECTOR_DATA_EXTENSION = ".vec";

    public KNNRescoreIntentDirectory(final Directory delegate) {
        super(delegate);
    }

    @Override
    public IndexInput openInput(final String name, final IOContext context) throws IOException {
        return in.openInput(name, withRescoreIntent(name, context));
    }

    /**
     * {@code context} with {@link KNNVectorReadIntent#RESCORE} added when {@code name} is a
     * full-precision vector data file, and {@code context} itself otherwise.
     *
     * <p>The existing hints are restated rather than dropped because {@code withHints} replaces the
     * set: the access advice and file-type hints Lucene put there decide how the mapping is advised,
     * and losing them would change the behaviour of the default path, which this view must not do.
     */
    static IOContext withRescoreIntent(final String name, final IOContext context) {
        if (name.endsWith(VECTOR_DATA_EXTENSION) == false) {
            return context;
        }
        if (KNNVectorReadIntent.of(context) != null) {
            return context;
        }
        final IOContext.FileOpenHint[] hints = Stream.concat(context.hints().stream(), Stream.of(KNNVectorReadIntent.RESCORE))
            .toArray(IOContext.FileOpenHint[]::new);
        return context.withHints(hints);
    }
}

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.store.Directory;
import org.apache.lucene.util.IOUtils;
import org.opensearch.common.Nullable;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.store.KNNRescoreIntentDirectory;

import java.io.Closeable;
import java.io.IOException;

/**
 * A second, lazily opened view of a segment's full-precision vectors whose reads carry
 * {@link org.opensearch.knn.index.store.KNNVectorReadIntent#RESCORE}, so that the storage layer can
 * serve them differently from the same segment's traversal, merge, warmup and derived-source reads of
 * the same file.
 *
 * <h2>The one idea</h2>
 * It does not decode anything. It hands the segment's own flat vectors format a
 * {@link SegmentReadState} that differs from the real one in exactly one respect — the {@link Directory}
 * is wrapped in a {@link KNNRescoreIntentDirectory} — and lets <em>Lucene</em> build the reader and the
 * {@link FloatVectorValues}. Metadata parsing, the base offset, the {@code ordToDoc} mapping, slicing
 * and the values class are therefore Lucene's own code, unmodified, and the plugin writes no per-encoding
 * decoding, no offset arithmetic and no entry sizing. That is what makes one mechanism cover every
 * encoding whose full-precision vectors live in a {@code .vec}: faiss scalar quantization at any code
 * width, the native engines' unquantized fp32, and the Lucene-engine quantized format alike, because for
 * all of them the {@code .vec} is the same fp32 flat file and the format that reads it is a plugin class
 * that already holds the per-field read state.
 *
 * <h2>Laziness is a correctness property, not an optimisation</h2>
 * A second handle on {@code .vec} is observable. It costs a file descriptor and a mapping, and anything
 * that keys open inputs by file name cannot tell the two handles apart. So nothing is opened until a
 * caller actually asks for rescore values <em>and</em> {@link KNNSettings#isDirectIORescoreEnabled()} is
 * on; on a default node this class is one field on the reader and no I/O at all.
 *
 * <p>The setting is read before the reader is built rather than on every request. Once the second reader
 * exists, turning the setting off stops new readers from being built but does not close this one — the
 * same "read once at open" shape the per-source cache size already has, and the reason is the same: a
 * live reader's handles cannot be withdrawn from under the values objects already handed out.
 *
 * <h2>Failure is always a fallback, never an error</h2>
 * Every way this can fail — a directory with no rescore route, a format that does not recognise the
 * segment, an {@code .vec} that cannot be opened twice — ends in {@code null} and a log line, and the
 * caller reads the vectors the way it would have without this class. There is no configuration in which
 * a failure here can change a result, because the bytes behind both views are the same bytes.
 */
@Log4j2
public final class KNNRescoreVectorsReader implements Closeable {

    /** The segment's own flat vectors format — the thing that knows how to read this {@code .vec}. */
    private final FlatVectorsFormat flatFormat;

    /**
     * The real read state with its {@link Directory} replaced by an intent-adding view. Built eagerly
     * because it allocates nothing but a wrapper; the reader it will be handed to is not.
     */
    private final SegmentReadState rescoreState;

    /** The segment name, for log lines only. */
    private final String segmentName;

    /** Built on first use under {@code this}, then read without synchronisation. */
    @Nullable
    private volatile FlatVectorsReader reader;

    /**
     * Set when the reader could not be built, so that a segment which cannot offer a rescore view is
     * not retried once per query for the life of the reader.
     */
    private volatile boolean unavailable;

    private volatile boolean closed;

    private KNNRescoreVectorsReader(final FlatVectorsFormat flatFormat, final SegmentReadState state) {
        this.flatFormat = flatFormat;
        this.segmentName = state.segmentInfo.name;
        this.rescoreState = new SegmentReadState(
            new KNNRescoreIntentDirectory(state.directory),
            state.segmentInfo,
            state.fieldInfos,
            state.context,
            state.segmentSuffix
        );
    }

    /**
     * A rescore view over {@code state}'s segment, or {@code null} when one cannot be described —
     * which is the case for a reader built without a read state, as the plugin's own tests and its
     * merge-instance paths do.
     *
     * <p>Creating one opens nothing. It is safe to call from a format's {@code fieldsReader} on every
     * segment of every index.
     *
     * @param flatFormat the format that reads this segment's full-precision {@code .vec}. For a
     *                   quantized format this is the <em>raw</em> flat format nested inside it, not the
     *                   quantized format itself: the rescore path wants the fp32 vectors, and going
     *                   through the quantized reader would open its {@code .veq} as well for nothing.
     * @param state      the read state the segment is being opened with, whose {@code segmentSuffix}
     *                   names this field's files
     */
    @Nullable
    public static KNNRescoreVectorsReader create(@Nullable final FlatVectorsFormat flatFormat, @Nullable final SegmentReadState state) {
        if (flatFormat == null || state == null) {
            return null;
        }
        return new KNNRescoreVectorsReader(flatFormat, state);
    }

    /**
     * Full-precision vectors for {@code field} whose reads carry the rescore intent, or {@code null}
     * when this segment cannot offer them — including when the Direct I/O rescore setting is off, which
     * is the default and is not a failure.
     *
     * <p>The returned values are a Lucene {@link FloatVectorValues} over the same bytes as the reader's
     * own, so they are interchangeable with them for every purpose except which {@code IndexInput} the
     * bytes come through.
     */
    @Nullable
    public FloatVectorValues floatVectorValues(final String field) {
        final FlatVectorsReader rescoreReader = readerFor(field);
        if (rescoreReader == null) {
            return null;
        }
        try {
            return rescoreReader.getFloatVectorValues(field);
        } catch (IOException | RuntimeException e) {
            // The values are the same bytes either way, so the caller reading them the ordinary way is
            // a complete answer to this.
            log.warn("k-NN rescore view: segment [{}] field [{}] could not produce values: {}", segmentName, field, e.toString());
            return null;
        }
    }

    /**
     * The lazily built reader, or {@code null}. Synchronised only on the transition: the common case
     * after the first rescore query on a segment is a volatile read.
     */
    @Nullable
    private FlatVectorsReader readerFor(final String field) {
        final FlatVectorsReader existing = reader;
        if (existing != null) {
            return closed ? null : existing;
        }
        if (unavailable || closed) {
            return null;
        }
        if (KNNSettings.isDirectIORescoreEnabled() == false) {
            // Deliberately not remembered as unavailable: the setting is dynamic, and a node that turns
            // it on should get the view on the next query without reopening the index.
            return null;
        }
        synchronized (this) {
            if (reader != null) {
                return closed ? null : reader;
            }
            if (unavailable || closed) {
                return null;
            }
            try {
                reader = flatFormat.fieldsReader(rescoreState);
                log.debug(
                    "k-NN rescore view: segment [{}] field [{}] opened a rescore view of the full-precision vectors",
                    segmentName,
                    field
                );
                return reader;
            } catch (IOException | RuntimeException e) {
                unavailable = true;
                log.warn("k-NN rescore view: segment [{}] field [{}] has no rescore view: {}", segmentName, field, e.toString());
                return null;
            }
        }
    }

    /** Whether a second handle on the full-precision vectors is currently open. */
    public boolean isOpen() {
        return reader != null && closed == false;
    }

    @Override
    public synchronized void close() throws IOException {
        closed = true;
        final FlatVectorsReader toClose = reader;
        reader = null;
        IOUtils.close(toClose);
    }
}

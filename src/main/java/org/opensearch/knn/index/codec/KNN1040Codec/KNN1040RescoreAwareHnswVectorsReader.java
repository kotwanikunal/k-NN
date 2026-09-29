/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.hnsw.HnswGraphProvider;
import org.apache.lucene.codecs.lucene99.Lucene99HnswVectorsReader;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.hnsw.CloseableRandomVectorScorerSupplier;
import org.apache.lucene.util.hnsw.HnswGraph;
import org.apache.lucene.util.quantization.BaseQuantizedByteVectorValues;
import org.apache.lucene.util.quantization.QuantizedVectorsReader;
import org.apache.lucene.util.quantization.ScalarQuantizer;
import org.opensearch.common.Nullable;
import org.opensearch.knn.index.codec.KNNRescoreVectorsReader;
import org.opensearch.knn.index.codec.scorer.HasRescoreVectorsReader;

import java.io.IOException;
import java.util.Map;

/**
 * A {@link Lucene99HnswVectorsReader} that can also hand out a rescore view of the field's
 * full-precision vectors.
 *
 * <h2>Why a delegate and not a subclass</h2>
 * {@link Lucene99HnswVectorsReader} is {@code final}, and the flat reader it holds is private with no
 * accessor, so the Lucene-engine quantized row has no plugin class anywhere on the path from the segment
 * to the {@code .vec} — {@code KNN1040HnswScalarQuantizedVectorsFormat} builds a stock HNSW reader over a
 * stock {@code Lucene104ScalarQuantizedVectorsReader}. This class is the smallest thing that gives the row
 * a plugin-owned reader to carry the capability, and it is deliberately a <em>reader</em> delegate: every
 * {@link FloatVectorValues} and {@link ByteVectorValues} it returns is the object Lucene built, unwrapped,
 * so no SIMD binding, bulk accessor or slice exposure is disturbed on the traversal, merge, warmup or
 * fetch paths. The rescore view is an <em>additional</em> object, reached only by a caller that asks for it
 * by name.
 *
 * <h2>The two interfaces that are not optional</h2>
 * {@link Lucene99HnswVectorsReader} implements {@link QuantizedVectorsReader} and
 * {@link HnswGraphProvider}, and Lucene tests for both with {@code instanceof} on the reader it is handed:
 * {@code IncrementalHnswGraphMerger#addReader} requires {@link HnswGraphProvider} to reuse an existing
 * graph instead of rebuilding it, and {@code Lucene99HnswVectorsWriter#mergeOneField} requires
 * {@link QuantizedVectorsReader} to reuse the quantized scorer supplier. A delegate that hid either would
 * silently turn incremental merges into full rebuilds with <em>identical resulting bytes</em> — a pure
 * performance regression no correctness test can see. Both are therefore implemented here by delegation,
 * and belt-and-braces, {@link #getMergeInstance()} returns the <em>delegate's</em> merge instance, so the
 * object the merger actually receives is the stock reader and not this one at all.
 *
 * @see HasRescoreVectorsReader
 */
public final class KNN1040RescoreAwareHnswVectorsReader extends KnnVectorsReader
    implements
        HasRescoreVectorsReader,
        QuantizedVectorsReader,
        HnswGraphProvider {

    private final Lucene99HnswVectorsReader delegate;

    /** The lazily opened second view of the full-precision vectors, or {@code null} when there is none. */
    @Nullable
    private final KNNRescoreVectorsReader rescoreVectorsReader;

    KNN1040RescoreAwareHnswVectorsReader(
        final Lucene99HnswVectorsReader delegate,
        @Nullable final KNNRescoreVectorsReader rescoreVectorsReader
    ) {
        this.delegate = delegate;
        this.rescoreVectorsReader = rescoreVectorsReader;
    }

    /**
     * A second view of {@code field}'s full-precision vectors whose reads carry the rescore intent, or
     * {@code null} when this segment offers none — which is the default, because the Direct I/O rescore
     * setting is off by default and the view is what checks it.
     */
    @Override
    @Nullable
    public FloatVectorValues rescoreVectorValues(final String field) {
        return rescoreVectorsReader == null ? null : rescoreVectorsReader.floatVectorValues(field);
    }

    /**
     * The <em>delegate's</em> merge instance, deliberately not a wrapper around it. Merge has no use for a
     * rescore view — its reads are sequential and must stay on the page cache — and handing the merger the
     * stock reader keeps the {@code instanceof HnswGraphProvider} and {@code instanceof
     * QuantizedVectorsReader} tests that decide whether a merge can reuse work rather than redo it.
     */
    @Override
    public KnnVectorsReader getMergeInstance() throws IOException {
        return delegate.getMergeInstance();
    }

    @Override
    public void finishMerge() throws IOException {
        delegate.finishMerge();
    }

    @Override
    public void checkIntegrity() throws IOException {
        delegate.checkIntegrity();
    }

    @Override
    public FloatVectorValues getFloatVectorValues(final String field) throws IOException {
        return delegate.getFloatVectorValues(field);
    }

    @Override
    public ByteVectorValues getByteVectorValues(final String field) throws IOException {
        return delegate.getByteVectorValues(field);
    }

    @Override
    public void search(final String field, final float[] target, final KnnCollector knnCollector, final AcceptDocs acceptDocs)
        throws IOException {
        delegate.search(field, target, knnCollector, acceptDocs);
    }

    @Override
    public void search(final String field, final byte[] target, final KnnCollector knnCollector, final AcceptDocs acceptDocs)
        throws IOException {
        delegate.search(field, target, knnCollector, acceptDocs);
    }

    @Override
    public Map<String, Long> getOffHeapByteSize(final FieldInfo fieldInfo) {
        return delegate.getOffHeapByteSize(fieldInfo);
    }

    @Override
    public HnswGraph getGraph(final String field) throws IOException {
        return delegate.getGraph(field);
    }

    @Override
    public BaseQuantizedByteVectorValues getQuantizedVectorValues(final String fieldName) throws IOException {
        return delegate.getQuantizedVectorValues(fieldName);
    }

    @Override
    public ScalarQuantizer getQuantizationState(final String fieldName) {
        return delegate.getQuantizationState(fieldName);
    }

    @Override
    public CloseableRandomVectorScorerSupplier getRandomVectorScorerSupplierForMerge(
        final FieldInfo fieldInfo,
        final SegmentWriteState segmentWriteState
    ) throws IOException {
        return delegate.getRandomVectorScorerSupplierForMerge(fieldInfo, segmentWriteState);
    }

    @Override
    public long ramBytesUsed() {
        return delegate.ramBytesUsed();
    }

    /**
     * Closes the rescore view and the delegate. The delegate is closed even if the view fails to close,
     * since it owns every file the default path reads.
     */
    @Override
    public void close() throws IOException {
        try {
            IOUtils.close(rescoreVectorsReader);
        } finally {
            delegate.close();
        }
    }

    @Override
    public String toString() {
        return getClass().getSimpleName() + "(delegate=" + delegate + ", rescoreView=" + rescoreVectorsReader + ")";
    }
}

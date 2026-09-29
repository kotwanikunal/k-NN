/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.scorer;

import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.perfield.PerFieldKnnVectorsFormat;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.SegmentReader;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.KnnCollector;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.vectorvalues.TestVectorValues;

import java.io.IOException;
import java.util.List;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * The bridge from the query layer's <em>what</em> signal to the storage layer's <em>how</em>: given the
 * segment reader the rescore path already holds, find the field's rescore view if its codec reader offers one.
 *
 * <p>Every case that is not "a rescore-aware reader answered" has to be a quiet {@code null}, because this
 * runs on the query path of every rescore query on every node — including the overwhelming majority where no
 * reader offers anything at all. A throw here would fail a search that has a perfectly good default path.
 */
public class RescoreVectorValuesSelectorTests extends KNNTestCase {

    private static final String FIELD = "vector";

    private static FloatVectorValues someValues() {
        return new TestVectorValues.PreDefinedFloatVectorValues(List.of(new float[] { 1.0f, 2.0f }));
    }

    /** A segment reader whose vector reader is the given one. */
    private static SegmentReader segmentReaderWith(final KnnVectorsReader vectorReader) {
        final SegmentReader segmentReader = mock(SegmentReader.class);
        when(segmentReader.getVectorReader()).thenReturn(vectorReader);
        return segmentReader;
    }

    /** A per-field reader that resolves {@link #FIELD} to the given reader. */
    private static PerFieldKnnVectorsFormat.FieldsReader perFieldResolving(final KnnVectorsReader fieldReader) {
        final PerFieldKnnVectorsFormat.FieldsReader perField = mock(PerFieldKnnVectorsFormat.FieldsReader.class);
        when(perField.getFieldReader(FIELD)).thenReturn(fieldReader);
        return perField;
    }

    /** The case the whole design turns on: a plugin reader behind the per-field reader answers. */
    public void testSelect_whenThePerFieldReaderIsRescoreAware_thenReturnsItsView() {
        final FloatVectorValues view = someValues();
        final SegmentReader segmentReader = segmentReaderWith(perFieldResolving(new RescoreAwareReader(view)));

        assertSame(view, RescoreVectorValuesSelector.select(segmentReader, FIELD));
    }

    /**
     * A reader that is already the field's own reader, not wrapped in a per-field reader. Lucene always uses
     * the per-field form today; not depending on that is one virtual call and removes a way for a future
     * arrangement to silently turn the route off.
     */
    public void testSelect_whenTheVectorReaderIsItselfRescoreAware_thenReturnsItsView() {
        final FloatVectorValues view = someValues();

        assertSame(view, RescoreVectorValuesSelector.select(segmentReaderWith(new RescoreAwareReader(view)), FIELD));
    }

    /** The ordinary answer on every encoding that is not an fp32 {@code .vec}: no capability, no view. */
    public void testSelect_whenTheReaderIsNotRescoreAware_thenReturnsNull() {
        assertNull(RescoreVectorValuesSelector.select(segmentReaderWith(perFieldResolving(new PlainReader())), FIELD));
    }

    /** A rescore-aware reader that declines — the setting off, a directory with no rescore route. */
    public void testSelect_whenTheReaderDeclines_thenReturnsNull() {
        assertNull(RescoreVectorValuesSelector.select(segmentReaderWith(perFieldResolving(new RescoreAwareReader(null))), FIELD));
    }

    /** A field the per-field reader does not know. */
    public void testSelect_whenTheFieldHasNoReader_thenReturnsNull() {
        assertNull(RescoreVectorValuesSelector.select(segmentReaderWith(perFieldResolving(null)), FIELD));
    }

    /** A segment with no vectors at all. */
    public void testSelect_whenThereIsNoVectorReader_thenReturnsNull() {
        assertNull(RescoreVectorValuesSelector.select(segmentReaderWith(null), FIELD));
    }

    /** Defensive on both arguments, because this is called from a lambda on the query path. */
    public void testSelect_whenGivenNulls_thenReturnsNull() {
        assertNull(RescoreVectorValuesSelector.select(null, FIELD));
        assertNull(RescoreVectorValuesSelector.select(segmentReaderWith(perFieldResolving(new RescoreAwareReader(someValues()))), null));
    }

    /**
     * A reader whose capability throws must not fail the query. The default path reads the same bytes, so the
     * only thing a failure here can cost is the isolation.
     */
    public void testSelect_whenTheReaderThrows_thenReturnsNull() {
        assertNull(RescoreVectorValuesSelector.select(segmentReaderWith(perFieldResolving(new ThrowingRescoreAwareReader())), FIELD));
    }

    /** A reader with no rescore capability at all. */
    private static class PlainReader extends KnnVectorsReader {

        @Override
        public void checkIntegrity() {}

        @Override
        public FloatVectorValues getFloatVectorValues(final String field) {
            return null;
        }

        @Override
        public ByteVectorValues getByteVectorValues(final String field) {
            return null;
        }

        @Override
        public void search(final String field, final float[] target, final KnnCollector knnCollector, final AcceptDocs acceptDocs) {}

        @Override
        public void search(final String field, final byte[] target, final KnnCollector knnCollector, final AcceptDocs acceptDocs) {}

        @Override
        public void close() throws IOException {}
    }

    /** A reader that offers the capability and answers with a fixed view (possibly {@code null}). */
    private static final class RescoreAwareReader extends PlainReader implements HasRescoreVectorsReader {

        private final FloatVectorValues view;

        private RescoreAwareReader(final FloatVectorValues view) {
            this.view = view;
        }

        @Override
        public FloatVectorValues rescoreVectorValues(final String field) {
            return view;
        }
    }

    private static final class ThrowingRescoreAwareReader extends PlainReader implements HasRescoreVectorsReader {

        @Override
        public FloatVectorValues rescoreVectorValues(final String field) {
            throw new IllegalStateException("boom");
        }
    }
}

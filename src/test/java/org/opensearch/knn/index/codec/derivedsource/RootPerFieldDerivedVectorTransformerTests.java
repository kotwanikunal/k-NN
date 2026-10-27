/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.derivedsource;

import org.apache.lucene.codecs.DocValuesProducer;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.index.BinaryDocValues;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.util.BytesRef;
import org.mockito.Mockito;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.KNNCodecTestUtil;
import org.opensearch.knn.index.codec.util.KNNVectorAsCollectionOfFloatsSerializer;
import org.opensearch.knn.index.vectorvalues.TestVectorValues;
import org.opensearch.test.OpenSearchTestCase;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * How many times one fetch builds a {@link org.opensearch.knn.index.vectorvalues.KNNVectorValues} for the same
 * field, which is what one document's reconstruction costs: every values object is a fresh slice of the
 * {@code .vec} and, on the Direct I/O route, a fresh read-ahead ring that has seen nothing.
 *
 * <p>Counted at the reader rather than at the factory, because what a values object costs is the
 * {@code getFloatVectorValues} call that opens the slice.
 */
public class RootPerFieldDerivedVectorTransformerTests extends OpenSearchTestCase {

    private static final String FIELD = "test_vector";
    private static final int DIMENSION = 4;

    public void testOneValuesObjectServesAForwardScan() throws Exception {
        final Harness harness = new Harness(8);
        final RootPerFieldDerivedVectorTransformer transformer = harness.transformer();

        for (final int docId : List.of(0, 3, 7)) {
            transformer.setCurrentDoc(0, docId);
            assertArrayEquals(harness.vector(docId), (float[]) transformer.apply(new Object()), 0.0f);
        }

        assertEquals("an ascending scan must build one values object", 1, harness.built());
    }

    public void testABackwardTargetRebuilds() throws Exception {
        final Harness harness = new Harness(8);
        final RootPerFieldDerivedVectorTransformer transformer = harness.transformer();

        transformer.setCurrentDoc(0, 7);
        assertArrayEquals(harness.vector(7), (float[]) transformer.apply(new Object()), 0.0f);
        transformer.setCurrentDoc(0, 3);
        assertArrayEquals(
            "the rebuilt object must answer for the earlier document",
            harness.vector(3),
            (float[]) transformer.apply(new Object()),
            0.0f
        );

        assertEquals("a backward target must build a second values object", 2, harness.built());
    }

    public void testAnEqualTargetDoesNotRebuild() throws Exception {
        final Harness harness = new Harness(8);
        final RootPerFieldDerivedVectorTransformer transformer = harness.transformer();

        transformer.setCurrentDoc(0, 7);
        assertArrayEquals(harness.vector(7), (float[]) transformer.apply(new Object()), 0.0f);
        transformer.setCurrentDoc(0, 7);
        assertArrayEquals(
            "the same document must read the same vector",
            harness.vector(7),
            (float[]) transformer.apply(new Object()),
            0.0f
        );

        assertEquals("a repeated document must not build a second values object", 1, harness.built());
    }

    /**
     * A target past the last document that has a vector exhausts the iterator, and the next scan that starts
     * behind it gets a new values object rather than an exhausted one.
     */
    public void testATargetPastTheLastVectorExhaustsAndTheNextScanRebuilds() throws Exception {
        final Harness harness = new Harness(4);
        final RootPerFieldDerivedVectorTransformer transformer = harness.transformer();

        transformer.setCurrentDoc(0, 9);
        assertEquals("the first scan must build one values object", 1, harness.built());
        transformer.setCurrentDoc(0, 1);
        assertArrayEquals(
            "the rebuilt object must answer for the earlier document",
            harness.vector(1),
            (float[]) transformer.apply(new Object()),
            0.0f
        );
        assertEquals("a scan that starts behind an exhausted iterator must rebuild", 2, harness.built());
    }

    /**
     * A failed advance must not leave an iterator in use whose position nobody knows: the next document is
     * served from a new values object.
     */
    public void testAFailedAdvanceDropsTheValuesObject() throws Exception {
        final SparseHarness harness = new SparseHarness(List.of(0, 2, 5));
        final RootPerFieldDerivedVectorTransformer transformer = harness.transformer();

        transformer.setCurrentDoc(0, 0);
        assertEquals(1, harness.built());
        harness.failAdvancesFrom(2);
        expectThrows(IOException.class, () -> transformer.setCurrentDoc(0, 2));
        harness.failAdvancesFrom(Integer.MAX_VALUE);
        transformer.setCurrentDoc(0, 2);
        assertArrayEquals(harness.vector(2), (float[]) transformer.apply(new Object()), 0.0f);
        assertEquals("the failed advance must have cost the values object", 2, harness.built());
    }

    /**
     * A sparse field whose gaps are one document wide keeps its values object. {@code setCurrentDoc} is called
     * for every document whether or not it has a vector
     * ({@code DerivedSourceVectorTransformer#injectVectors}: "the per field injectors are responsible for
     * skipping if the field is not present"), so the iterator is asked for documents that are not in it and
     * lands on the next one that is. The document after a one-wide gap is then the document the iterator is
     * already on, which a forward-only iterator must not be advanced to, and this is the case that holds the
     * second guard.
     */
    public void testASparseFieldWithOneWideGapsKeepsOneValuesObject() throws Exception {
        final SparseHarness harness = new SparseHarness(List.of(0, 2, 4, 6));
        final RootPerFieldDerivedVectorTransformer transformer = harness.transformer();

        for (int docId = 0; docId <= 6; docId++) {
            transformer.setCurrentDoc(0, docId);
            if (docId % 2 == 0) {
                assertArrayEquals(harness.vector(docId), (float[]) transformer.apply(new Object()), 0.0f);
            }
        }

        assertEquals("one-wide gaps must not cost a values object", 1, harness.built());
    }

    /**
     * A gap wider than one document costs one values object per document inside it past the first, and this
     * records that rather than hiding it: the iterator lands past those documents, so each of them is a target
     * behind the iterator and rebuilds. It is still never worse than a values object per document, which is
     * what the method did before, and on a field that is dense or has one-wide gaps it is one object for the
     * whole scan.
     */
    public void testAGapWiderThanOneDocumentRebuildsOncePerDocumentInside() throws Exception {
        final SparseHarness harness = new SparseHarness(List.of(0, 2, 5));
        final RootPerFieldDerivedVectorTransformer transformer = harness.transformer();

        for (final int docId : List.of(0, 1, 2, 3, 4, 5)) {
            transformer.setCurrentDoc(0, docId);
            if (List.of(0, 2, 5).contains(docId)) {
                assertArrayEquals(harness.vector(docId), (float[]) transformer.apply(new Object()), 0.0f);
            }
        }

        // Documents 3 and 4 are the gap: advancing to 3 lands on 5, and 4 is then behind the iterator.
        assertEquals("the one gap of two documents must cost exactly one rebuild", 2, harness.built());
    }

    /**
     * One field of {@code docs} float vectors behind a reader that counts how many times its values were
     * opened. Every open returns a fresh iterator, the way a real reader does, so a transformer that rebuilds
     * starts again from the beginning.
     */
    private static final class Harness {

        private final List<float[]> vectors = new ArrayList<>();
        private final FieldInfo fieldInfo;
        private final DerivedSourceReaders readers;
        private int built;

        Harness(final int docs) throws Exception {
            for (int doc = 0; doc < docs; doc++) {
                final float[] vector = new float[DIMENSION];
                Arrays.fill(vector, doc + 1);
                vectors.add(vector);
            }
            fieldInfo = KNNCodecTestUtil.FieldInfoBuilder.builder(FIELD)
                .vectorDimension(DIMENSION)
                .vectorEncoding(VectorEncoding.FLOAT32)
                .build();

            final KnnVectorsReader reader = Mockito.mock(KnnVectorsReader.class);
            Mockito.when(reader.getFloatVectorValues(FIELD)).thenAnswer(invocation -> {
                built++;
                return new TestVectorValues.PreDefinedFloatVectorValues(vectors);
            });
            readers = Mockito.mock(DerivedSourceReaders.class);
            Mockito.when(readers.getKnnVectorsReader()).thenReturn(reader);
        }

        RootPerFieldDerivedVectorTransformer transformer() {
            return new RootPerFieldDerivedVectorTransformer(fieldInfo, readers);
        }

        float[] vector(final int docId) {
            return vectors.get(docId);
        }

        int built() {
            return built;
        }
    }

    /**
     * The same field read through doc values instead of a vector reader, carrying vectors for
     * {@code docsWithVectors} only, so that the iterator can land past a document that was asked for.
     */
    private static final class SparseHarness {

        private final List<Integer> docsWithVectors;
        private final FieldInfo fieldInfo;
        private final DerivedSourceReaders readers;
        private SparseFloatVectorDocValues live;
        private int failFrom = Integer.MAX_VALUE;
        private int built;

        SparseHarness(final List<Integer> docsWithVectors) throws IOException {
            this.docsWithVectors = docsWithVectors;
            // No vector dimension, so the factory takes its doc-values branch; the data type then has to be
            // declared, because it cannot be read off a vector encoding that is not there.
            fieldInfo = KNNCodecTestUtil.FieldInfoBuilder.builder(FIELD)
                .docValuesType(DocValuesType.BINARY)
                .addAttribute(KNNConstants.VECTOR_DATA_TYPE_FIELD, VectorDataType.FLOAT.getValue())
                .build();

            final DocValuesProducer producer = Mockito.mock(DocValuesProducer.class);
            Mockito.when(producer.getBinary(fieldInfo)).thenAnswer(invocation -> {
                built++;
                live = new SparseFloatVectorDocValues(docsWithVectors);
                live.failFrom = failFrom;
                return live;
            });
            readers = Mockito.mock(DerivedSourceReaders.class);
            Mockito.when(readers.getDocValuesProducer()).thenReturn(producer);
            Mockito.when(readers.getKnnVectorsReader()).thenReturn(null);
        }

        RootPerFieldDerivedVectorTransformer transformer() {
            return new RootPerFieldDerivedVectorTransformer(fieldInfo, readers);
        }

        /** Makes every advance to {@code target} or beyond fail, on the values object now and the next one. */
        void failAdvancesFrom(final int target) {
            failFrom = target;
            if (live != null) {
                live.failFrom = target;
            }
        }

        float[] vector(final int docId) {
            final float[] vector = new float[DIMENSION];
            Arrays.fill(vector, docsWithVectors.indexOf(docId) + 1);
            return vector;
        }

        int built() {
            return built;
        }
    }

    /**
     * Binary doc values over an ascending list of document ids, which is the shape a sparse field has: an
     * {@code advance} past the last id answers {@code NO_MORE_DOCS}, and one to a document with no value
     * answers the next document that has one.
     */
    private static final class SparseFloatVectorDocValues extends BinaryDocValues {

        private final List<Integer> docs;
        private int position = -1;
        /** Advancing to this document or beyond throws, so that a failed advance can be tested. */
        private int failFrom = Integer.MAX_VALUE;

        SparseFloatVectorDocValues(final List<Integer> docs) {
            this.docs = docs;
        }

        @Override
        public BytesRef binaryValue() {
            final float[] vector = new float[DIMENSION];
            Arrays.fill(vector, position + 1);
            return new BytesRef(KNNVectorAsCollectionOfFloatsSerializer.INSTANCE.floatToByteArray(vector));
        }

        @Override
        public boolean advanceExact(final int target) {
            return false;
        }

        @Override
        public int docID() {
            if (position < 0) {
                return -1;
            }
            return position >= docs.size() ? NO_MORE_DOCS : docs.get(position);
        }

        @Override
        public int nextDoc() {
            position++;
            return docID();
        }

        @Override
        public int advance(final int target) throws IOException {
            if (target >= failFrom) {
                throw new IOException("advance to " + target + " was told to fail");
            }
            while (position + 1 < docs.size() && docs.get(position + 1) < target) {
                position++;
            }
            return nextDoc();
        }

        @Override
        public long cost() {
            return docs.size();
        }
    }
}

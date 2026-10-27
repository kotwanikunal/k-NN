/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.derivedsource;

import org.apache.lucene.index.FieldInfo;
import org.opensearch.common.CheckedSupplier;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory;

import java.io.IOException;

public class RootPerFieldDerivedVectorTransformer extends AbstractPerFieldDerivedVectorTransformer {

    private final FieldInfo fieldInfo;
    private final CheckedSupplier<KNNVectorValues<?>, IOException> vectorValuesSupplier;
    private KNNVectorValues<?> vectorValues;

    /**
     * Constructor for RootPerFieldDerivedVectorTransformer.
     *
     * @param fieldInfo FieldInfo for the field to create the injector for
     * @param derivedSourceReaders {@link DerivedSourceReaders} instance
     */
    public RootPerFieldDerivedVectorTransformer(FieldInfo fieldInfo, DerivedSourceReaders derivedSourceReaders) {
        this.fieldInfo = fieldInfo;
        this.vectorValuesSupplier = () -> KNNVectorValuesFactory.getVectorValues(
            fieldInfo,
            derivedSourceReaders.getDocValuesProducer(),
            derivedSourceReaders.getKnnVectorsReader()
        );
    }

    @Override
    public void setCurrentDoc(int offset, int docId) throws IOException {
        // One values object serves a whole scan, because the only thing it cannot do is go backwards: the
        // iterator advances forward only, so a target behind the document it is on is the one case that needs
        // a new one. Keeping it matters because every values object is a fresh slice of the vector file, and a
        // slice that is asked for one vector reads that vector on its own.
        if (vectorValues == null || docId < vectorValues.docId()) {
            vectorValues = vectorValuesSupplier.get();
        }
        // Advancing to the document the iterator is already on is outside DocIdSetIterator's contract, and a
        // sparse field reaches that case whenever the document after a gap is the one the iterator landed on;
        // the vector already in hand is that document's.
        if (docId > vectorValues.docId()) {
            try {
                vectorValues.advance(docId);
            } catch (IOException e) {
                // A failed advance leaves the iterator's position undefined, so it must not be reused: the
                // next document would be served from a place no one knows.
                vectorValues = null;
                throw e;
            }
        }
    }

    @Override
    public Object apply(Object object) {
        if (object == null) {
            return object;
        }

        try {
            return formatVector(fieldInfo, vectorValues::getVector, vectorValues::conditionalCloneVector);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}

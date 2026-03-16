/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN990Codec;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.store.DataAccessHint;
import org.apache.lucene.store.FileDataHint;
import org.apache.lucene.store.FileTypeHint;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.util.IOSupplier;
import org.apache.lucene.util.IOUtils;
import org.opensearch.knn.common.FieldInfoExtractor;
import org.opensearch.knn.index.codec.util.KNNCodecUtil;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.memoryoptsearch.VectorSearcher;
import org.opensearch.knn.memoryoptsearch.VectorSearcherFactory;
import org.opensearch.knn.memoryoptsearch.faiss.FaissMemoryOptimizedSearcher;

import java.io.Closeable;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.opensearch.knn.index.mapper.KNNVectorFieldMapper.KNN_FIELD;

/**
 * Vectors reader for Faiss BBQ fields. Delegates flat vector reads to Lucene's
 * {@link org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorsReader}
 * and search to Faiss memory-optimized searcher.
 *
 * <p>The {@link #getFloatVectorValues(String)} method returns a {@link FloatVectorValues} from the
 * underlying Lucene104 reader which wraps both quantized and raw vector values. This provides:
 * <ul>
 *   <li>{@code scorer()} — uses quantized (BBQ) scoring for fast approximate distance</li>
 *   <li>{@code rescorer()} — delegates to the full-precision raw vectors for highest fidelity rescoring</li>
 * </ul>
 */
@Log4j2
public class FaissBBQ990KnnVectorsReader extends KnnVectorsReader {

    private final FlatVectorsReader flatVectorsReader;
    private final SegmentReadState segmentReadState;
    private volatile NativeEngines990KnnVectorsReader.VectorSearcherHolder vectorSearcherHolder;
    private final Object vectorSearcherHolderLockObject;
    private final IOContext ioContext;

    FaissBBQ990KnnVectorsReader(SegmentReadState state, FlatVectorsReader flatVectorsReader) {
        this.flatVectorsReader = flatVectorsReader;
        this.segmentReadState = state;
        this.ioContext = state.context.withHints(FileTypeHint.DATA, FileDataHint.KNN_VECTORS, DataAccessHint.RANDOM);
        this.vectorSearcherHolder = new NativeEngines990KnnVectorsReader.VectorSearcherHolder();
        this.vectorSearcherHolderLockObject = new Object();
    }

    @Override
    public void checkIntegrity() throws IOException {
        flatVectorsReader.checkIntegrity();
    }

    /**
     * Returns {@link FloatVectorValues} from the underlying Lucene104ScalarQuantizedVectorsReader.
     * The returned values expose both {@code scorer()} (quantized BBQ scoring) and
     * {@code rescorer()} (full-precision scoring) methods.
     */
    @Override
    public FloatVectorValues getFloatVectorValues(String field) throws IOException {
        return flatVectorsReader.getFloatVectorValues(field);
    }

    @Override
    public ByteVectorValues getByteVectorValues(String field) throws IOException {
        throw new UnsupportedOperationException("Byte vector search is not supported for Faiss BBQ");
    }

    @Override
    public void search(String field, float[] target, KnnCollector knnCollector, AcceptDocs acceptDocs) throws IOException {
        final FieldInfo fieldInfo = segmentReadState.fieldInfos.fieldInfo(field);
        final VectorSearcher memoryOptimizedSearcher = loadMemoryOptimizedSearcherIfRequired(fieldInfo);

        if (target == null) {
            throw new FaissMemoryOptimizedSearcher.WarmupInitializationException("Null vector supplied for warmup");
        }

        if (memoryOptimizedSearcher != null) {
            memoryOptimizedSearcher.search(target, knnCollector, acceptDocs);
            return;
        }

        throw new UnsupportedOperationException("Search functionality using codec is not supported with Faiss BBQ Reader");
    }

    @Override
    public void search(String field, byte[] target, KnnCollector knnCollector, AcceptDocs acceptDocs) throws IOException {
        throw new UnsupportedOperationException("Byte vector search is not supported for Faiss BBQ");
    }

    @Override
    public void close() throws IOException {
        final List<Closeable> closeables = new ArrayList<>();
        closeables.add(flatVectorsReader);
        if (vectorSearcherHolder != null) {
            closeables.add(vectorSearcherHolder.getVectorSearcher());
        }
        IOUtils.close(closeables);
    }

    private VectorSearcher loadMemoryOptimizedSearcherIfRequired(FieldInfo fieldInfo) {
        if (vectorSearcherHolder.isSet()) {
            return vectorSearcherHolder.getVectorSearcher();
        }

        synchronized (vectorSearcherHolderLockObject) {
            if (vectorSearcherHolder.isSet()) {
                return vectorSearcherHolder.getVectorSearcher();
            }
            final IOSupplier<VectorSearcher> searcherSupplier = getVectorSearcherSupplier(fieldInfo);
            if (searcherSupplier != null) {
                try {
                    vectorSearcherHolder.setVectorSearcher(searcherSupplier.get());
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            } else {
                log.error("Failed to load memory optimized searcher for field [{}]", fieldInfo.getName());
            }
            return vectorSearcherHolder.getVectorSearcher();
        }
    }

    private IOSupplier<VectorSearcher> getVectorSearcherSupplier(FieldInfo fieldInfo) {
        final Map<String, String> attributes = fieldInfo.attributes();
        if (attributes == null || attributes.containsKey(KNN_FIELD) == false) {
            return null;
        }
        final KNNEngine knnEngine = FieldInfoExtractor.extractKNNEngine(fieldInfo);
        if (knnEngine == null) {
            return null;
        }
        final VectorSearcherFactory searcherFactory = knnEngine.getVectorSearcherFactory();
        if (searcherFactory == null) {
            return null;
        }
        final String fileName = KNNCodecUtil.getNativeEngineFileFromFieldInfo(fieldInfo, segmentReadState.segmentInfo);
        if (fileName != null) {
            return () -> searcherFactory.createVectorSearcher(segmentReadState.directory, fileName, fieldInfo, ioContext);
        }
        return null;
    }
}

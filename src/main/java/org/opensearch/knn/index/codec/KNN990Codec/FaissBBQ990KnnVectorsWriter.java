/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN990Codec;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.KnnFieldVectorsWriter;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorScorerUtil;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorScorer;
import org.apache.lucene.codecs.lucene104.Lucene104ScalarQuantizedVectorsReader;
import org.apache.lucene.codecs.lucene104.QuantizedByteVectorValues;
import org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsFormat;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.Sorter;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.RamUsageEstimator;
import org.opensearch.common.StopWatch;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexWriter;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.plugin.stats.KNNGraphValue;

import java.io.Closeable;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

import static org.opensearch.knn.common.FieldInfoExtractor.extractVectorDataType;
import static org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory.getKNNVectorValuesSupplierForMerge;
import static org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory.getVectorValuesSupplier;

/**
 * KNNVectorsWriter for Faiss BBQ fields. Handles a single field per format instance.
 * <ol>
 *   <li>Delegates flat vector writing to Lucene's scalar quantized format (.vec + .veq/.vemq)</li>
 *   <li>Reads back quantized vectors from Lucene's on-disk format</li>
 *   <li>Passes them to {@code NativeIndexWriter} → {@code MemOptimizedBBQIndexBuildStrategy}
 *       via {@code BuildIndexParams.quantizedVectorValues} for Faiss HNSW graph construction</li>
 * </ol>
 */
@Log4j2
class FaissBBQ990KnnVectorsWriter extends KnnVectorsWriter {
    private static final long SHALLOW_SIZE = RamUsageEstimator.shallowSizeOfInstance(FaissBBQ990KnnVectorsWriter.class);

    private final SegmentWriteState segmentWriteState;
    private final FlatVectorsWriter flatVectorsWriter;
    private final int approximateThreshold;
    private NativeEngineFieldVectorsWriter<?> field;
    private boolean finished;
    // Readers opened by readQuantizedVectors that must be closed when the writer closes.
    private final List<Closeable> quantizedVectorReaders = new ArrayList<>();

    FaissBBQ990KnnVectorsWriter(
        SegmentWriteState segmentWriteState,
        FlatVectorsWriter flatVectorsWriter,
        int approximateThreshold
    ) {
        this.segmentWriteState = segmentWriteState;
        this.flatVectorsWriter = flatVectorsWriter;
        this.approximateThreshold = approximateThreshold;
    }

    @Override
    public KnnFieldVectorsWriter<?> addField(FieldInfo fieldInfo) throws IOException {
        if (this.field != null) {
            throw new IllegalStateException(
                "FaissBBQ990KnnVectorsWriter supports only a single field, but addField was called for ["
                    + fieldInfo.name + "] after [" + this.field.getFieldInfo().name + "]"
            );
        }
        this.field = NativeEngineFieldVectorsWriter.create(fieldInfo, flatVectorsWriter.addField(fieldInfo), segmentWriteState.infoStream);
        return field;
    }

    @Override
    public void flush(int maxDoc, Sorter.DocMap sortMap) throws IOException {
        flatVectorsWriter.flush(maxDoc, sortMap);

        if (field == null) {
            return;
        }

        final FieldInfo fieldInfo = field.getFieldInfo();
        int totalLiveDocs = field.getVectors().size();
        if (totalLiveDocs == 0) {
            log.debug("[Flush] No live docs for field {}", fieldInfo.getName());
            return;
        }

        if (shouldSkipBuildingVectorDataStructure(totalLiveDocs)) {
            log.debug(
                "Skip building vector data structure for field: {}, as liveDoc: {} is less than the threshold {} during flush",
                fieldInfo.name, totalLiveDocs, approximateThreshold
            );
            return;
        }

        final VectorDataType vectorDataType = extractVectorDataType(fieldInfo);
        final Supplier<KNNVectorValues<?>> knnVectorValuesSupplier = getVectorValuesSupplier(
            vectorDataType, field.getFlatFieldVectorsWriter().getDocsWithFieldSet(), field.getVectors()
        );

        final NativeIndexWriter writer = NativeIndexWriter.getWriter(
            fieldInfo, segmentWriteState, null, () -> {
                try {
                    return readQuantizedVectors(fieldInfo);
                } catch (IOException e) {
                    throw new UncheckedIOException(e);
                }
            }
        );

        StopWatch stopWatch = new StopWatch().start();
        writer.flushIndex(knnVectorValuesSupplier, totalLiveDocs);
        long time_in_millis = stopWatch.stop().totalTime().millis();
        KNNGraphValue.REFRESH_TOTAL_TIME_IN_MILLIS.incrementBy(time_in_millis);
        log.debug("Flush took {} ms for vector field [{}]", time_in_millis, fieldInfo.getName());
    }

    @Override
    public void mergeOneField(FieldInfo fieldInfo, MergeState mergeState) throws IOException {
        flatVectorsWriter.mergeOneField(fieldInfo, mergeState);

        final VectorDataType vectorDataType = extractVectorDataType(fieldInfo);
        final Supplier<KNNVectorValues<?>> knnVectorValuesSupplier = getKNNVectorValuesSupplierForMerge(
            vectorDataType, fieldInfo, mergeState
        );
        int totalLiveDocs = getLiveDocs(knnVectorValuesSupplier.get());
        if (totalLiveDocs == 0) {
            log.debug("[Merge] No live docs for field {}", fieldInfo.getName());
            return;
        }

        if (shouldSkipBuildingVectorDataStructure(totalLiveDocs)) {
            log.debug(
                "Skip building vector data structure for field: {}, as liveDoc: {} is less than the threshold {} during merge",
                fieldInfo.name, totalLiveDocs, approximateThreshold
            );
            return;
        }

        final NativeIndexWriter writer = NativeIndexWriter.getWriter(
            fieldInfo, segmentWriteState, null, () -> {
                try {
                    return readQuantizedVectors(fieldInfo);
                } catch (IOException e) {
                    throw new UncheckedIOException(e);
                }
            }
        );

        StopWatch stopWatch = new StopWatch().start();
        writer.mergeIndex(knnVectorValuesSupplier, totalLiveDocs);
        long time_in_millis = stopWatch.stop().totalTime().millis();
        KNNGraphValue.MERGE_TOTAL_TIME_IN_MILLIS.incrementBy(time_in_millis);
        log.debug("Merge took {} ms for vector field [{}]", time_in_millis, fieldInfo.getName());
    }

    /**
     * Reads back quantized vectors from the .veq file that was just flushed/merged by the flat writer.
     * The opened readers are tracked in {@link #quantizedVectorReaders} and closed in {@link #close()}.
     */
    private QuantizedByteVectorValues readQuantizedVectors(FieldInfo fieldInfo) throws IOException {
        final SegmentReadState readState = new SegmentReadState(
            segmentWriteState.directory, segmentWriteState.segmentInfo,
            new FieldInfos(new FieldInfo[] { fieldInfo }),
            segmentWriteState.context, fieldInfo.getName()
        );

        final FlatVectorsReader fullPrecisionReader = new Lucene99FlatVectorsFormat(
            FlatVectorScorerUtil.getLucene99FlatVectorsScorer()
        ).fieldsReader(readState);

        final Lucene104ScalarQuantizedVectorsReader bbqReader;
        try {
            bbqReader = new Lucene104ScalarQuantizedVectorsReader(
                readState, fullPrecisionReader,
                new Lucene104ScalarQuantizedVectorScorer(fullPrecisionReader.getFlatVectorScorer())
            );
        } catch (final Exception e) {
            fullPrecisionReader.close();
            throw e;
        }

        try {
            // Lucene104ScalarQuantizedVectorsReader.getFloatVectorValues returns a ScalarQuantizedVectorValues
            // that wraps both raw and quantized values. The quantized values are stored in a package-private
            // field "quantizedVectorValues" — reflection is required since the accessor is not public.
            // Target: Lucene 10.4 ScalarQuantizedVectorValues.quantizedVectorValues
            final FloatVectorValues fvv = bbqReader.getFloatVectorValues(fieldInfo.getName());
            final Field f = fvv.getClass().getDeclaredField("quantizedVectorValues");
            f.setAccessible(true);
            final QuantizedByteVectorValues quantizedValues = (QuantizedByteVectorValues) f.get(fvv);
            // Track readers so they stay alive while quantizedValues is in use and are closed in close()
            quantizedVectorReaders.add(bbqReader);
            return quantizedValues;
        } catch (NoSuchFieldException | IllegalAccessException e) {
            bbqReader.close();
            throw new RuntimeException("Failed to extract quantized vector values", e);
        } catch (final Exception e) {
            bbqReader.close();
            throw e;
        }
    }

    @Override
    public void finish() throws IOException {
        if (finished) {
            throw new IllegalStateException("FaissBBQ990KnnVectorsWriter is already finished");
        }
        finished = true;
        flatVectorsWriter.finish();
    }

    @Override
    public void close() throws IOException {
        IOUtils.close(flatVectorsWriter);
        IOUtils.close(quantizedVectorReaders);
    }

    @Override
    public long ramBytesUsed() {
        return SHALLOW_SIZE + flatVectorsWriter.ramBytesUsed() + (field != null ? field.ramBytesUsed() : 0);
    }

    private int getLiveDocs(KNNVectorValues<?> vectorValues) throws IOException {
        int liveDocs = 0;
        while (vectorValues.nextDoc() != DocIdSetIterator.NO_MORE_DOCS) {
            liveDocs++;
        }
        return liveDocs;
    }

    private boolean shouldSkipBuildingVectorDataStructure(long docCount) {
        return approximateThreshold < 0 || docCount < approximateThreshold;
    }
}

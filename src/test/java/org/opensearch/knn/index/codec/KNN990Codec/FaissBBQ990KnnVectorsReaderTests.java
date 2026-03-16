/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN990Codec;

import com.google.common.collect.ImmutableSet;
import lombok.SneakyThrows;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.junit.Assert;
import org.mockito.MockedStatic;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.codec.KNNCodecTestUtil;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;
import org.opensearch.knn.memoryoptsearch.VectorSearcher;
import org.opensearch.knn.memoryoptsearch.VectorSearcherFactory;

import java.io.IOException;
import java.lang.reflect.Field;
import java.util.Collections;
import java.util.Set;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.mockStatic;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

public class FaissBBQ990KnnVectorsReaderTests extends KNNTestCase {

    // --- checkIntegrity ---

    @SneakyThrows
    public void testCheckIntegrity_thenDelegatesToFlatReader() {
        final FlatVectorsReader flatVectorsReader = mock(FlatVectorsReader.class);
        final FaissBBQ990KnnVectorsReader reader = createReader(new FieldInfos(new FieldInfo[0]), Collections.emptySet(), flatVectorsReader);
        reader.checkIntegrity();
        verify(flatVectorsReader).checkIntegrity();
    }

    // --- getFloatVectorValues ---

    @SneakyThrows
    public void testGetFloatVectorValues_thenDelegatesToFlatReader() {
        final FlatVectorsReader flatVectorsReader = mock(FlatVectorsReader.class);
        final FloatVectorValues mockValues = mock(FloatVectorValues.class);
        when(flatVectorsReader.getFloatVectorValues("test_field")).thenReturn(mockValues);

        final FaissBBQ990KnnVectorsReader reader = createReader(new FieldInfos(new FieldInfo[0]), Collections.emptySet(), flatVectorsReader);
        final FloatVectorValues result = reader.getFloatVectorValues("test_field");

        assertSame(mockValues, result);
        verify(flatVectorsReader).getFloatVectorValues("test_field");
    }

    // --- getByteVectorValues ---

    @SneakyThrows
    public void testGetByteVectorValues_thenThrowsUnsupported() {
        final FlatVectorsReader flatVectorsReader = mock(FlatVectorsReader.class);
        final FaissBBQ990KnnVectorsReader reader = createReader(new FieldInfos(new FieldInfo[0]), Collections.emptySet(), flatVectorsReader);

        expectThrows(UnsupportedOperationException.class, () -> reader.getByteVectorValues("test_field"));
    }

    // --- search(float[]) ---

    @SneakyThrows
    public void testSearchFloat_whenNullTarget_thenThrowsWarmupException() {
        final FieldInfo fieldInfo = createFieldInfo("field1", KNNEngine.FAISS, 0);
        final FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { fieldInfo });

        KNNEngine mockFaiss = spy(KNNEngine.FAISS);
        VectorSearcherFactory mockFactory = mock(VectorSearcherFactory.class);
        VectorSearcher mockSearcher = mock(VectorSearcher.class);
        when(mockFaiss.getVectorSearcherFactory()).thenReturn(mockFactory);
        when(mockFactory.createVectorSearcher(any(), any(), any(), any())).thenReturn(mockSearcher);

        try (MockedStatic<KNNEngine> mockedStatic = mockStatic(KNNEngine.class)) {
            mockedStatic.when(() -> KNNEngine.getEngine(any())).thenReturn(mockFaiss);
            mockedStatic.when(KNNEngine::getEnginesThatCreateCustomSegmentFiles).thenReturn(ImmutableSet.of(mockFaiss));

            final FaissBBQ990KnnVectorsReader reader = createReader(fieldInfos, Set.of("_0_165_field1.faiss"), mock(FlatVectorsReader.class));

            // null target should throw warmup exception
            expectThrows(Exception.class, () -> reader.search("field1", (float[]) null, null, null));
        }
    }

    @SneakyThrows
    public void testSearchFloat_whenSearcherAvailable_thenDelegatesToSearcher() {
        final FieldInfo fieldInfo = createFieldInfo("field1", KNNEngine.FAISS, 0);
        final FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { fieldInfo });

        KNNEngine mockFaiss = spy(KNNEngine.FAISS);
        VectorSearcherFactory mockFactory = mock(VectorSearcherFactory.class);
        VectorSearcher mockSearcher = mock(VectorSearcher.class);
        when(mockFaiss.getVectorSearcherFactory()).thenReturn(mockFactory);
        when(mockFactory.createVectorSearcher(any(), any(), any(), any())).thenReturn(mockSearcher);

        try (MockedStatic<KNNEngine> mockedStatic = mockStatic(KNNEngine.class)) {
            mockedStatic.when(() -> KNNEngine.getEngine(any())).thenReturn(mockFaiss);
            mockedStatic.when(KNNEngine::getEnginesThatCreateCustomSegmentFiles).thenReturn(ImmutableSet.of(mockFaiss));

            final FaissBBQ990KnnVectorsReader reader = createReader(fieldInfos, Set.of("_0_165_field1.faiss"), mock(FlatVectorsReader.class));
            float[] target = new float[] { 1, 2, 3 };
            reader.search("field1", target, null, null);

            verify(mockSearcher).search(target, null, null);
        }
    }

    @SneakyThrows
    public void testSearchFloat_whenNoSearcherAvailable_thenThrowsUnsupported() {
        final FieldInfo fieldInfo = createFieldInfo("field1", KNNEngine.FAISS, 0);
        final FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { fieldInfo });

        KNNEngine mockFaiss = spy(KNNEngine.FAISS);
        // No searcher factory
        when(mockFaiss.getVectorSearcherFactory()).thenReturn(null);

        try (MockedStatic<KNNEngine> mockedStatic = mockStatic(KNNEngine.class)) {
            mockedStatic.when(() -> KNNEngine.getEngine(any())).thenReturn(mockFaiss);
            mockedStatic.when(KNNEngine::getEnginesThatCreateCustomSegmentFiles).thenReturn(ImmutableSet.of(mockFaiss));

            final FaissBBQ990KnnVectorsReader reader = createReader(fieldInfos, Collections.emptySet(), mock(FlatVectorsReader.class));

            expectThrows(
                UnsupportedOperationException.class,
                () -> reader.search("field1", new float[] { 1, 2, 3 }, null, null)
            );
        }
    }

    // --- search(byte[]) ---

    @SneakyThrows
    public void testSearchByte_thenThrowsUnsupported() {
        final FlatVectorsReader flatVectorsReader = mock(FlatVectorsReader.class);
        final FaissBBQ990KnnVectorsReader reader = createReader(new FieldInfos(new FieldInfo[0]), Collections.emptySet(), flatVectorsReader);

        expectThrows(
            UnsupportedOperationException.class,
            () -> reader.search("test_field", new byte[] { 1, 2, 3 }, null, null)
        );
    }

    // --- close ---

    @SneakyThrows
    public void testClose_thenClosesFlatReaderAndSearcher() {
        final FieldInfo fieldInfo = createFieldInfo("field1", KNNEngine.FAISS, 0);
        final FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { fieldInfo });

        KNNEngine mockFaiss = spy(KNNEngine.FAISS);
        VectorSearcherFactory mockFactory = mock(VectorSearcherFactory.class);
        VectorSearcher mockSearcher = mock(VectorSearcher.class);
        when(mockFaiss.getVectorSearcherFactory()).thenReturn(mockFactory);
        when(mockFactory.createVectorSearcher(any(), any(), any(), any())).thenReturn(mockSearcher);

        final FlatVectorsReader flatVectorsReader = mock(FlatVectorsReader.class);

        try (MockedStatic<KNNEngine> mockedStatic = mockStatic(KNNEngine.class)) {
            mockedStatic.when(() -> KNNEngine.getEngine(any())).thenReturn(mockFaiss);
            mockedStatic.when(KNNEngine::getEnginesThatCreateCustomSegmentFiles).thenReturn(ImmutableSet.of(mockFaiss));

            final FaissBBQ990KnnVectorsReader reader = createReader(fieldInfos, Set.of("_0_165_field1.faiss"), flatVectorsReader);
            // Trigger searcher initialization
            reader.search("field1", new float[] { 1, 2, 3 }, null, null);

            reader.close();

            verify(flatVectorsReader).close();
            verify(mockSearcher).close();
        }
    }

    @SneakyThrows
    public void testClose_whenNoSearcher_thenClosesFlatReaderOnly() {
        final FlatVectorsReader flatVectorsReader = mock(FlatVectorsReader.class);
        final FaissBBQ990KnnVectorsReader reader = createReader(new FieldInfos(new FieldInfo[0]), Collections.emptySet(), flatVectorsReader);

        reader.close();
        verify(flatVectorsReader).close();
    }

    // --- vectorSearcherHolder lazy init ---

    @SneakyThrows
    public void testVectorSearcherHolder_initiallyNotSet() {
        final FlatVectorsReader flatVectorsReader = mock(FlatVectorsReader.class);
        final FaissBBQ990KnnVectorsReader reader = createReader(new FieldInfos(new FieldInfo[0]), Collections.emptySet(), flatVectorsReader);

        final NativeEngines990KnnVectorsReader.VectorSearcherHolder holder = getVectorSearcherHolder(reader);
        assertFalse(holder.isSet());
    }

    // --- helpers ---

    private static FieldInfo createFieldInfo(final String fieldName, final KNNEngine engine, final int fieldNo) {
        final KNNCodecTestUtil.FieldInfoBuilder builder = KNNCodecTestUtil.FieldInfoBuilder.builder(fieldName);
        builder.fieldNumber(fieldNo);
        if (engine != null) {
            builder.addAttribute(KNNVectorFieldMapper.KNN_FIELD, "true");
            builder.addAttribute(KNNConstants.KNN_ENGINE, engine.getName());
        }
        return builder.build();
    }

    @SneakyThrows
    private static FaissBBQ990KnnVectorsReader createReader(
        final FieldInfos fieldInfos,
        final Set<String> filesInSegment,
        final FlatVectorsReader flatVectorsReader
    ) {
        final IndexInput mockIndexInput = mock(IndexInput.class);
        final Directory mockDirectory = mock(Directory.class);
        when(mockDirectory.openInput(any(), any())).thenReturn(mockIndexInput);
        final SegmentInfo segmentInfo = mock(SegmentInfo.class);
        when(segmentInfo.files()).thenReturn(filesInSegment);
        when(segmentInfo.getId()).thenReturn((segmentInfo.hashCode() + "").getBytes());
        final SegmentReadState readState = new SegmentReadState(mockDirectory, segmentInfo, fieldInfos, IOContext.DEFAULT);

        return new FaissBBQ990KnnVectorsReader(readState, flatVectorsReader);
    }

    @SneakyThrows
    private static NativeEngines990KnnVectorsReader.VectorSearcherHolder getVectorSearcherHolder(
        final FaissBBQ990KnnVectorsReader reader
    ) {
        final Field tableField = FaissBBQ990KnnVectorsReader.class.getDeclaredField("vectorSearcherHolder");
        tableField.setAccessible(true);
        return (NativeEngines990KnnVectorsReader.VectorSearcherHolder) tableField.get(reader);
    }
}

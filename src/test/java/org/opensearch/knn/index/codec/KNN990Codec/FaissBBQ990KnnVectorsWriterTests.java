/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN990Codec;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.hnsw.FlatFieldVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.index.DocsWithFieldSet;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.MergeState;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.util.Version;
import org.mockito.Mock;
import org.mockito.MockedStatic;
import org.mockito.Mockito;
import org.mockito.MockitoAnnotations;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexWriter;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory;
import org.opensearch.knn.index.vectorvalues.TestVectorValues;
import org.opensearch.knn.plugin.stats.KNNGraphValue;

import java.io.Closeable;
import java.io.IOException;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.mockStatic;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

public class FaissBBQ990KnnVectorsWriterTests extends KNNTestCase {

    @Mock
    private FlatVectorsWriter flatVectorsWriter;
    @Mock
    private SegmentWriteState segmentWriteState;
    @Mock
    private NativeIndexWriter nativeIndexWriter;
    @Mock
    private SegmentInfo segmentInfo;

    private FlatFieldVectorsWriter mockedFlatFieldVectorsWriter;
    private FaissBBQ990KnnVectorsWriter objectUnderTest;

    private static final int BUILD_GRAPH_ALWAYS_THRESHOLD = 0;
    private static final int BUILD_GRAPH_NEVER_THRESHOLD = -1;

    @Override
    public void setUp() throws Exception {
        super.setUp();
        MockitoAnnotations.openMocks(this);
        objectUnderTest = new FaissBBQ990KnnVectorsWriter(segmentWriteState, flatVectorsWriter, BUILD_GRAPH_ALWAYS_THRESHOLD);
        mockedFlatFieldVectorsWriter = mock(FlatFieldVectorsWriter.class);
        Mockito.doNothing().when(mockedFlatFieldVectorsWriter).addValue(Mockito.anyInt(), Mockito.any());
        when(flatVectorsWriter.addField(any())).thenReturn(mockedFlatFieldVectorsWriter);

        try {
            Field infoField = SegmentWriteState.class.getDeclaredField("segmentInfo");
            infoField.setAccessible(true);
            infoField.set(segmentWriteState, segmentInfo);
        } catch (Exception ignored) {}
        when(segmentInfo.getVersion()).thenReturn(Version.LATEST);
    }

    // --- addField tests ---

    @SneakyThrows
    public void testAddField_thenSuccess() {
        final FieldInfo fieldInfo = fieldInfo(0, VectorEncoding.FLOAT32, Map.of(KNNConstants.VECTOR_DATA_TYPE_FIELD, "float"));
        try (
            MockedStatic<NativeEngineFieldVectorsWriter> fieldWriterMockedStatic = mockStatic(NativeEngineFieldVectorsWriter.class)
        ) {
            NativeEngineFieldVectorsWriter field = nativeEngineFieldVectorsWriter(fieldInfo, Map.of(0, new float[] { 1, 2, 3 }));
            fieldWriterMockedStatic.when(
                () -> NativeEngineFieldVectorsWriter.create(fieldInfo, mockedFlatFieldVectorsWriter, segmentWriteState.infoStream)
            ).thenReturn(field);

            objectUnderTest.addField(fieldInfo);
            verify(flatVectorsWriter).addField(fieldInfo);
        }
    }

    @SneakyThrows
    public void testAddField_calledTwice_thenThrowsException() {
        final FieldInfo fieldInfo1 = fieldInfo(0, VectorEncoding.FLOAT32, Map.of(KNNConstants.VECTOR_DATA_TYPE_FIELD, "float"));
        final FieldInfo fieldInfo2 = fieldInfo(1, VectorEncoding.FLOAT32, Map.of(KNNConstants.VECTOR_DATA_TYPE_FIELD, "float"));
        when(fieldInfo1.getName()).thenReturn("field1");
        when(fieldInfo2.getName()).thenReturn("field2");

        try (
            MockedStatic<NativeEngineFieldVectorsWriter> fieldWriterMockedStatic = mockStatic(NativeEngineFieldVectorsWriter.class)
        ) {
            NativeEngineFieldVectorsWriter field = nativeEngineFieldVectorsWriter(fieldInfo1, Map.of(0, new float[] { 1, 2, 3 }));
            fieldWriterMockedStatic.when(
                () -> NativeEngineFieldVectorsWriter.create(fieldInfo1, mockedFlatFieldVectorsWriter, segmentWriteState.infoStream)
            ).thenReturn(field);

            objectUnderTest.addField(fieldInfo1);

            expectThrows(IllegalStateException.class, () -> objectUnderTest.addField(fieldInfo2));
        }
    }

    // --- flush tests ---

    @SneakyThrows
    public void testFlush_whenNoField_thenOnlyDelegates() {
        objectUnderTest.flush(5, null);
        verify(flatVectorsWriter).flush(5, null);
    }

    @SneakyThrows
    public void testFlush_whenNoLiveDocs_thenSkipsBuild() {
        final FieldInfo fieldInfo = fieldInfo(0, VectorEncoding.FLOAT32, Map.of(KNNConstants.VECTOR_DATA_TYPE_FIELD, "float"));

        try (
            MockedStatic<NativeEngineFieldVectorsWriter> fieldWriterMockedStatic = mockStatic(NativeEngineFieldVectorsWriter.class);
            MockedStatic<NativeIndexWriter> nativeIndexWriterMockedStatic = mockStatic(NativeIndexWriter.class)
        ) {
            NativeEngineFieldVectorsWriter field = nativeEngineFieldVectorsWriter(fieldInfo, Map.of());
            fieldWriterMockedStatic.when(
                () -> NativeEngineFieldVectorsWriter.create(fieldInfo, mockedFlatFieldVectorsWriter, segmentWriteState.infoStream)
            ).thenReturn(field);

            objectUnderTest.addField(fieldInfo);
            objectUnderTest.flush(5, null);

            verify(flatVectorsWriter).flush(5, null);
            nativeIndexWriterMockedStatic.verifyNoInteractions();
        }
    }

    @SneakyThrows
    public void testFlush_whenBelowThreshold_thenSkipsBuild() {
        // Use a high threshold so docs are below it
        FaissBBQ990KnnVectorsWriter writerWithHighThreshold = new FaissBBQ990KnnVectorsWriter(
            segmentWriteState, flatVectorsWriter, 100
        );

        final FieldInfo fieldInfo = fieldInfo(0, VectorEncoding.FLOAT32, Map.of(KNNConstants.VECTOR_DATA_TYPE_FIELD, "float"));
        Map<Integer, float[]> vectors = Map.of(0, new float[] { 1, 2, 3 });

        try (
            MockedStatic<NativeEngineFieldVectorsWriter> fieldWriterMockedStatic = mockStatic(NativeEngineFieldVectorsWriter.class);
            MockedStatic<NativeIndexWriter> nativeIndexWriterMockedStatic = mockStatic(NativeIndexWriter.class)
        ) {
            NativeEngineFieldVectorsWriter field = nativeEngineFieldVectorsWriter(fieldInfo, vectors);
            fieldWriterMockedStatic.when(
                () -> NativeEngineFieldVectorsWriter.create(fieldInfo, mockedFlatFieldVectorsWriter, segmentWriteState.infoStream)
            ).thenReturn(field);

            writerWithHighThreshold.addField(fieldInfo);
            writerWithHighThreshold.flush(5, null);

            verify(flatVectorsWriter).flush(5, null);
            nativeIndexWriterMockedStatic.verifyNoInteractions();
        }
    }

    @SneakyThrows
    public void testFlush_whenNegativeThreshold_thenSkipsBuild() {
        FaissBBQ990KnnVectorsWriter writerNeverBuild = new FaissBBQ990KnnVectorsWriter(
            segmentWriteState, flatVectorsWriter, BUILD_GRAPH_NEVER_THRESHOLD
        );

        final FieldInfo fieldInfo = fieldInfo(0, VectorEncoding.FLOAT32, Map.of(KNNConstants.VECTOR_DATA_TYPE_FIELD, "float"));
        Map<Integer, float[]> vectors = Map.of(0, new float[] { 1, 2, 3 });

        try (
            MockedStatic<NativeEngineFieldVectorsWriter> fieldWriterMockedStatic = mockStatic(NativeEngineFieldVectorsWriter.class);
            MockedStatic<NativeIndexWriter> nativeIndexWriterMockedStatic = mockStatic(NativeIndexWriter.class)
        ) {
            NativeEngineFieldVectorsWriter field = nativeEngineFieldVectorsWriter(fieldInfo, vectors);
            fieldWriterMockedStatic.when(
                () -> NativeEngineFieldVectorsWriter.create(fieldInfo, mockedFlatFieldVectorsWriter, segmentWriteState.infoStream)
            ).thenReturn(field);

            writerNeverBuild.addField(fieldInfo);
            writerNeverBuild.flush(5, null);

            verify(flatVectorsWriter).flush(5, null);
            nativeIndexWriterMockedStatic.verifyNoInteractions();
        }
    }

    @SneakyThrows
    public void testFlush_whenAboveThreshold_thenBuildsIndex() {
        final FieldInfo fieldInfo = fieldInfo(
            0,
            VectorEncoding.FLOAT32,
            Map.of(KNNConstants.VECTOR_DATA_TYPE_FIELD, "float", KNNConstants.KNN_ENGINE, "faiss")
        );
        Map<Integer, float[]> vectors = Map.of(0, new float[] { 1, 2, 3 }, 1, new float[] { 4, 5, 6 });

        final TestVectorValues.PreDefinedFloatVectorValues preDefinedVectors = new TestVectorValues.PreDefinedFloatVectorValues(
            new ArrayList<>(vectors.values())
        );
        final Supplier<KNNVectorValues<?>> expectedSupplier = KNNVectorValuesFactory.getVectorValuesSupplier(
            VectorDataType.FLOAT, preDefinedVectors
        );

        try (
            MockedStatic<NativeEngineFieldVectorsWriter> fieldWriterMockedStatic = mockStatic(NativeEngineFieldVectorsWriter.class);
            MockedStatic<KNNVectorValuesFactory> knnVectorValuesFactoryMockedStatic = mockStatic(KNNVectorValuesFactory.class);
            MockedStatic<NativeIndexWriter> nativeIndexWriterMockedStatic = mockStatic(NativeIndexWriter.class)
        ) {
            NativeEngineFieldVectorsWriter field = nativeEngineFieldVectorsWriter(fieldInfo, vectors);
            fieldWriterMockedStatic.when(
                () -> NativeEngineFieldVectorsWriter.create(fieldInfo, mockedFlatFieldVectorsWriter, segmentWriteState.infoStream)
            ).thenReturn(field);

            objectUnderTest.addField(fieldInfo);

            DocsWithFieldSet docsWithFieldSet = field.getFlatFieldVectorsWriter().getDocsWithFieldSet();
            knnVectorValuesFactoryMockedStatic.when(
                () -> KNNVectorValuesFactory.getVectorValuesSupplier(VectorDataType.FLOAT, docsWithFieldSet, vectors)
            ).thenReturn(expectedSupplier);

            nativeIndexWriterMockedStatic.when(
                () -> NativeIndexWriter.getWriter(any(FieldInfo.class), any(SegmentWriteState.class), any(), any(Supplier.class))
            ).thenReturn(nativeIndexWriter);

            doAnswer(answer -> {
                Thread.sleep(2);
                return null;
            }).when(nativeIndexWriter).flushIndex(any(), anyInt());

            objectUnderTest.flush(5, null);

            verify(flatVectorsWriter).flush(5, null);
            verify(nativeIndexWriter).flushIndex(expectedSupplier, vectors.size());
            assertNotEquals(0L, (long) KNNGraphValue.REFRESH_TOTAL_TIME_IN_MILLIS.getValue());
        }
    }

    // --- mergeOneField tests ---

    @SneakyThrows
    public void testMergeOneField_whenNoLiveDocs_thenSkipsBuild() {
        final FieldInfo fieldInfo = fieldInfo(
            0,
            VectorEncoding.FLOAT32,
            Map.of(KNNConstants.VECTOR_DATA_TYPE_FIELD, "float", KNNConstants.KNN_ENGINE, "faiss")
        );
        final MergeState mergeState = mock(MergeState.class);

        final KNNVectorValues<float[]> knnVectorValues = KNNVectorValuesFactory.getVectorValues(
            VectorDataType.FLOAT,
            new TestVectorValues.PreDefinedFloatVectorValues(new ArrayList<>())
        );
        final Supplier<KNNVectorValues<?>> supplier = () -> knnVectorValues;

        try (
            MockedStatic<KNNVectorValuesFactory> knnVectorValuesFactoryMockedStatic = mockStatic(KNNVectorValuesFactory.class);
            MockedStatic<NativeIndexWriter> nativeIndexWriterMockedStatic = mockStatic(NativeIndexWriter.class)
        ) {
            knnVectorValuesFactoryMockedStatic.when(
                () -> KNNVectorValuesFactory.getKNNVectorValuesSupplierForMerge(any(), any(), any())
            ).thenReturn(supplier);

            objectUnderTest.mergeOneField(fieldInfo, mergeState);

            verify(flatVectorsWriter).mergeOneField(fieldInfo, mergeState);
            nativeIndexWriterMockedStatic.verifyNoInteractions();
        }
    }

    @SneakyThrows
    public void testMergeOneField_whenBelowThreshold_thenSkipsBuild() {
        FaissBBQ990KnnVectorsWriter writerWithHighThreshold = new FaissBBQ990KnnVectorsWriter(
            segmentWriteState, flatVectorsWriter, 100
        );

        final FieldInfo fieldInfo = fieldInfo(
            0,
            VectorEncoding.FLOAT32,
            Map.of(KNNConstants.VECTOR_DATA_TYPE_FIELD, "float", KNNConstants.KNN_ENGINE, "faiss")
        );
        final MergeState mergeState = mock(MergeState.class);

        final KNNVectorValues<float[]> knnVectorValues = KNNVectorValuesFactory.getVectorValues(
            VectorDataType.FLOAT,
            new TestVectorValues.PreDefinedFloatVectorValues(List.of(new float[] { 1, 2, 3 }))
        );
        final Supplier<KNNVectorValues<?>> supplier = () -> knnVectorValues;

        try (
            MockedStatic<KNNVectorValuesFactory> knnVectorValuesFactoryMockedStatic = mockStatic(KNNVectorValuesFactory.class);
            MockedStatic<NativeIndexWriter> nativeIndexWriterMockedStatic = mockStatic(NativeIndexWriter.class)
        ) {
            knnVectorValuesFactoryMockedStatic.when(
                () -> KNNVectorValuesFactory.getKNNVectorValuesSupplierForMerge(any(), any(), any())
            ).thenReturn(supplier);

            writerWithHighThreshold.mergeOneField(fieldInfo, mergeState);

            verify(flatVectorsWriter).mergeOneField(fieldInfo, mergeState);
            nativeIndexWriterMockedStatic.verifyNoInteractions();
        }
    }

    @SneakyThrows
    public void testMergeOneField_whenAboveThreshold_thenBuildsIndex() {
        final FieldInfo fieldInfo = fieldInfo(
            0,
            VectorEncoding.FLOAT32,
            Map.of(KNNConstants.VECTOR_DATA_TYPE_FIELD, "float", KNNConstants.KNN_ENGINE, "faiss")
        );
        final MergeState mergeState = mock(MergeState.class);

        final KNNVectorValues<float[]> knnVectorValues = KNNVectorValuesFactory.getVectorValues(
            VectorDataType.FLOAT,
            new TestVectorValues.PreDefinedFloatVectorValues(List.of(new float[] { 1, 2, 3 }, new float[] { 4, 5, 6 }))
        );
        final Supplier<KNNVectorValues<?>> supplier = () -> knnVectorValues;

        try (
            MockedStatic<KNNVectorValuesFactory> knnVectorValuesFactoryMockedStatic = mockStatic(KNNVectorValuesFactory.class);
            MockedStatic<NativeIndexWriter> nativeIndexWriterMockedStatic = mockStatic(NativeIndexWriter.class)
        ) {
            knnVectorValuesFactoryMockedStatic.when(
                () -> KNNVectorValuesFactory.getKNNVectorValuesSupplierForMerge(any(), any(), any())
            ).thenReturn(supplier);

            nativeIndexWriterMockedStatic.when(
                () -> NativeIndexWriter.getWriter(any(FieldInfo.class), any(SegmentWriteState.class), any(), any(Supplier.class))
            ).thenReturn(nativeIndexWriter);

            doAnswer(answer -> {
                Thread.sleep(2);
                return null;
            }).when(nativeIndexWriter).mergeIndex(any(), anyInt());

            objectUnderTest.mergeOneField(fieldInfo, mergeState);

            verify(flatVectorsWriter).mergeOneField(fieldInfo, mergeState);
            verify(nativeIndexWriter).mergeIndex(supplier, 2);
            assertNotEquals(0L, (long) KNNGraphValue.MERGE_TOTAL_TIME_IN_MILLIS.getValue());
        }
    }

    // --- finish tests ---

    @SneakyThrows
    public void testFinish_thenDelegatesToFlatWriter() {
        objectUnderTest.finish();
        verify(flatVectorsWriter).finish();
    }

    @SneakyThrows
    public void testFinish_calledTwice_thenThrowsException() {
        objectUnderTest.finish();
        expectThrows(IllegalStateException.class, () -> objectUnderTest.finish());
    }

    // --- close tests ---

    @SneakyThrows
    public void testClose_thenClosesFlatWriter() {
        objectUnderTest.close();
        verify(flatVectorsWriter).close();
    }

    @SneakyThrows
    public void testClose_thenClosesQuantizedVectorReaders() {
        // Access the quantizedVectorReaders field via reflection and add a mock closeable
        Field readersField = FaissBBQ990KnnVectorsWriter.class.getDeclaredField("quantizedVectorReaders");
        readersField.setAccessible(true);
        @SuppressWarnings("unchecked")
        List<Closeable> readers = (List<Closeable>) readersField.get(objectUnderTest);

        Closeable mockReader1 = mock(Closeable.class);
        Closeable mockReader2 = mock(Closeable.class);
        readers.add(mockReader1);
        readers.add(mockReader2);

        objectUnderTest.close();

        verify(flatVectorsWriter).close();
        verify(mockReader1).close();
        verify(mockReader2).close();
    }

    // --- ramBytesUsed tests ---

    public void testRamBytesUsed_whenNoField_thenReturnsShallowSize() {
        when(flatVectorsWriter.ramBytesUsed()).thenReturn(100L);
        long result = objectUnderTest.ramBytesUsed();
        assertTrue(result > 100L); // shallow size + flat writer
    }

    // --- helpers ---

    private FieldInfo fieldInfo(int fieldNumber, VectorEncoding vectorEncoding, Map<String, String> attributes) {
        FieldInfo fieldInfo = mock(FieldInfo.class);
        when(fieldInfo.getFieldNumber()).thenReturn(fieldNumber);
        when(fieldInfo.getVectorEncoding()).thenReturn(vectorEncoding);
        when(fieldInfo.attributes()).thenReturn(attributes);
        when(fieldInfo.getName()).thenReturn("field" + fieldNumber);
        attributes.forEach((key, value) -> when(fieldInfo.getAttribute(key)).thenReturn(value));
        return fieldInfo;
    }

    private <T> NativeEngineFieldVectorsWriter nativeEngineFieldVectorsWriter(FieldInfo fieldInfo, Map<Integer, T> vectors) {
        NativeEngineFieldVectorsWriter fieldVectorsWriter = mock(NativeEngineFieldVectorsWriter.class);
        FlatFieldVectorsWriter flatFieldVectorsWriter = mock(FlatFieldVectorsWriter.class);
        DocsWithFieldSet docsWithFieldSet = new DocsWithFieldSet();
        vectors.keySet().stream().sorted().forEach(docsWithFieldSet::add);
        when(fieldVectorsWriter.getFieldInfo()).thenReturn(fieldInfo);
        when(fieldVectorsWriter.getVectors()).thenReturn(vectors);
        when(fieldVectorsWriter.getFlatFieldVectorsWriter()).thenReturn(flatFieldVectorsWriter);
        when(flatFieldVectorsWriter.getDocsWithFieldSet()).thenReturn(docsWithFieldSet);
        return fieldVectorsWriter;
    }
}

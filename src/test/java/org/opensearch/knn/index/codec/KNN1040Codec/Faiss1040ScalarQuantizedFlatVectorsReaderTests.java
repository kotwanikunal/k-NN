/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FilterDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.util.StringHelper;
import org.apache.lucene.util.Version;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.quantization.QuantizedByteVectorValues;
import org.mockito.MockedStatic;
import org.opensearch.knn.KNNTestCase;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Collections;
import java.util.HashMap;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.mockStatic;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

public class Faiss1040ScalarQuantizedFlatVectorsReaderTests extends KNNTestCase {

    @SneakyThrows
    public void testConstructor_thenUsesScorerFromDelegate() {
        FlatVectorsReader delegate = mock(FlatVectorsReader.class);
        FlatVectorsScorer expectedScorer = mock(FlatVectorsScorer.class);
        when(delegate.getFlatVectorScorer("test_field")).thenReturn(expectedScorer);

        Faiss1040ScalarQuantizedFlatVectorsReader reader = new Faiss1040ScalarQuantizedFlatVectorsReader(delegate);
        assertSame(expectedScorer, reader.getFlatVectorScorer("test_field"));
    }

    @SneakyThrows
    public void testGetRandomVectorScorerFloat_thenDelegatesToReader() {
        FlatVectorsReader delegate = mock(FlatVectorsReader.class);
        RandomVectorScorer expectedScorer = mock(RandomVectorScorer.class);
        float[] target = { 1.0f, 2.0f };
        when(delegate.getRandomVectorScorer("field", target)).thenReturn(expectedScorer);

        Faiss1040ScalarQuantizedFlatVectorsReader reader = new Faiss1040ScalarQuantizedFlatVectorsReader(delegate);
        assertSame(expectedScorer, reader.getRandomVectorScorer("field", target));
        verify(delegate).getRandomVectorScorer("field", target);
    }

    @SneakyThrows
    public void testGetRandomVectorScorerByte_thenDelegatesToReader() {
        FlatVectorsReader delegate = mock(FlatVectorsReader.class);
        RandomVectorScorer expectedScorer = mock(RandomVectorScorer.class);
        byte[] target = { 1, 2 };
        when(delegate.getRandomVectorScorer("field", target)).thenReturn(expectedScorer);

        Faiss1040ScalarQuantizedFlatVectorsReader reader = new Faiss1040ScalarQuantizedFlatVectorsReader(delegate);
        assertSame(expectedScorer, reader.getRandomVectorScorer("field", target));
        verify(delegate).getRandomVectorScorer("field", target);
    }

    @SneakyThrows
    public void testCheckIntegrity_thenDelegatesToReader() {
        FlatVectorsReader delegate = mock(FlatVectorsReader.class);
        Faiss1040ScalarQuantizedFlatVectorsReader reader = new Faiss1040ScalarQuantizedFlatVectorsReader(delegate);
        reader.checkIntegrity();
        verify(delegate).checkIntegrity();
    }

    @SneakyThrows
    public void testGetFloatVectorValues_thenReturnsWrappedValuesWithBothDelegates() {
        FlatVectorsReader delegate = mock(FlatVectorsReader.class);
        FloatVectorValues mockFvv = mock(FloatVectorValues.class);
        QuantizedByteVectorValues mockQbvv = mock(QuantizedByteVectorValues.class);
        when(delegate.getFloatVectorValues("field")).thenReturn(mockFvv);
        when(mockFvv.size()).thenReturn(1);

        try (MockedStatic<KNN1040ScalarQuantizedUtils> mockedUtils = mockStatic(KNN1040ScalarQuantizedUtils.class)) {
            mockedUtils.when(() -> KNN1040ScalarQuantizedUtils.extractQuantizedByteVectorValues(any())).thenReturn(mockQbvv);

            Faiss1040ScalarQuantizedFlatVectorsReader reader = new Faiss1040ScalarQuantizedFlatVectorsReader(delegate);
            FloatVectorValues result = reader.getFloatVectorValues("field");

            assertTrue("Result should be ScalarQuantizedFloatVectorValues", result instanceof ScalarQuantizedFloatVectorValues);
            ScalarQuantizedFloatVectorValues wrapper = (ScalarQuantizedFloatVectorValues) result;
            assertSame(mockFvv, wrapper.getFloatVectorValues());
            assertSame(mockQbvv, wrapper.getQuantizedVectorValues());
            verify(delegate).getFloatVectorValues("field");
            mockedUtils.verify(() -> KNN1040ScalarQuantizedUtils.extractQuantizedByteVectorValues(mockFvv));
        }
    }

    @SneakyThrows
    public void testGetFloatVectorValues_whenEmpty_thenReturnsWrappedEmptyValuesWithoutExtraction() {
        FlatVectorsReader delegate = mock(FlatVectorsReader.class);
        FloatVectorValues emptyValues = mock(FloatVectorValues.class);
        when(delegate.getFloatVectorValues("field")).thenReturn(emptyValues);
        when(emptyValues.size()).thenReturn(0);

        try (MockedStatic<KNN1040ScalarQuantizedUtils> mockedUtils = mockStatic(KNN1040ScalarQuantizedUtils.class)) {
            Faiss1040ScalarQuantizedFlatVectorsReader reader = new Faiss1040ScalarQuantizedFlatVectorsReader(delegate);
            FloatVectorValues result = reader.getFloatVectorValues("field");

            assertTrue(result instanceof ScalarQuantizedFloatVectorValues);
            ScalarQuantizedFloatVectorValues wrapper = (ScalarQuantizedFloatVectorValues) result;
            assertSame(emptyValues, wrapper.getFloatVectorValues());
            assertNull(wrapper.getQuantizedVectorValues());
            assertEquals(0, result.size());
            verify(delegate).getFloatVectorValues("field");
            // The assertion is that the *quantized* extraction is skipped for an empty segment. The wrapper
            // also unwraps the raw full-precision values, which is a different call on the same utility and
            // does happen, so this cannot be verifyNoInteractions.
            mockedUtils.verify(() -> KNN1040ScalarQuantizedUtils.extractQuantizedByteVectorValues(any()), never());
        }
    }

    @SneakyThrows
    public void testGetFloatVectorValues_whenDelegateReturnsNull_thenReturnsNull() {
        FlatVectorsReader delegate = mock(FlatVectorsReader.class);
        when(delegate.getFloatVectorValues("field")).thenReturn(null);

        Faiss1040ScalarQuantizedFlatVectorsReader reader = new Faiss1040ScalarQuantizedFlatVectorsReader(delegate);

        assertNull(reader.getFloatVectorValues("field"));
        verify(delegate).getFloatVectorValues("field");
    }

    @SneakyThrows
    public void testGetByteVectorValues_thenDelegatesToReader() {
        FlatVectorsReader delegate = mock(FlatVectorsReader.class);
        ByteVectorValues expectedValues = mock(ByteVectorValues.class);
        when(delegate.getByteVectorValues("field")).thenReturn(expectedValues);

        Faiss1040ScalarQuantizedFlatVectorsReader reader = new Faiss1040ScalarQuantizedFlatVectorsReader(delegate);
        assertSame(expectedValues, reader.getByteVectorValues("field"));
        verify(delegate).getByteVectorValues("field");
    }

    @SneakyThrows
    public void testClose_thenDelegatesToReader() {
        FlatVectorsReader delegate = mock(FlatVectorsReader.class);
        Faiss1040ScalarQuantizedFlatVectorsReader reader = new Faiss1040ScalarQuantizedFlatVectorsReader(delegate);
        reader.close();
        verify(delegate).close();
    }

    @SneakyThrows
    public void testRamBytesUsed_thenDelegatesToReader() {
        FlatVectorsReader delegate = mock(FlatVectorsReader.class);
        when(delegate.ramBytesUsed()).thenReturn(12345L);

        Faiss1040ScalarQuantizedFlatVectorsReader reader = new Faiss1040ScalarQuantizedFlatVectorsReader(delegate);
        assertEquals(12345L, reader.ramBytesUsed());
        verify(delegate).ramBytesUsed();
    }

    /**
     * A reader built without a read state - every existing caller of the one-argument constructor - offers
     * no Direct I/O source, so it behaves exactly as it did before that path existed.
     */
    @SneakyThrows
    public void testVectorLoaderSource_whenNoReadState_thenNullAndNothingIsOpened() {
        FlatVectorsReader delegate = mock(FlatVectorsReader.class);
        Faiss1040ScalarQuantizedFlatVectorsReader reader = new Faiss1040ScalarQuantizedFlatVectorsReader(delegate);

        assertNull(reader.vectorLoaderSource("field"));
        // Not even the reference values the source would verify against are requested.
        verify(delegate, never()).getFloatVectorValues(any());
    }

    /**
     * The values a query holds must not open anything by themselves: with the flag off, the supplier is
     * never called, and that is what keeps a disabled node on exactly today's file handles.
     */
    @SneakyThrows
    public void testGetFloatVectorValues_thenOpensNoDirectIOHandle() {
        FlatVectorsReader delegate = mock(FlatVectorsReader.class);
        FloatVectorValues mockFvv = mock(FloatVectorValues.class);
        when(delegate.getFloatVectorValues("field")).thenReturn(mockFvv);
        when(mockFvv.size()).thenReturn(1);

        try (
            MockedStatic<KNN1040ScalarQuantizedUtils> mockedUtils = mockStatic(KNN1040ScalarQuantizedUtils.class);
            Directory dir = newFSDirectory(createTempDir())
        ) {
            mockedUtils.when(() -> KNN1040ScalarQuantizedUtils.extractQuantizedByteVectorValues(any()))
                .thenReturn(mock(QuantizedByteVectorValues.class));

            Faiss1040ScalarQuantizedFlatVectorsReader reader = new Faiss1040ScalarQuantizedFlatVectorsReader(delegate, readState(dir, "0"));
            FloatVectorValues values = reader.getFloatVectorValues("field");

            assertTrue(values instanceof ScalarQuantizedFloatVectorValues);
            // getFloatVectorValues was called once - for the values themselves. A source opened eagerly
            // would have called it a second time to fetch its verification reference.
            verify(delegate).getFloatVectorValues("field");
        }
    }

    /**
     * The values do name a source once asked, and a directory with no {@code .vec} in it makes the answer
     * {@code null} rather than an exception. The "there is none" answer is remembered, so a field that
     * cannot be served is not retried once per query.
     */
    @SneakyThrows
    public void testVectorLoaderSource_whenTheFileIsAbsent_thenNullAndNotRetried() {
        FlatVectorsReader delegate = mock(FlatVectorsReader.class);
        FloatVectorValues mockFvv = mock(FloatVectorValues.class);
        when(delegate.getFloatVectorValues("field")).thenReturn(mockFvv);
        when(mockFvv.size()).thenReturn(4);
        when(mockFvv.dimension()).thenReturn(8);
        when(mockFvv.getVectorByteLength()).thenReturn(32);

        try (Directory dir = newFSDirectory(createTempDir())) {
            Faiss1040ScalarQuantizedFlatVectorsReader reader = new Faiss1040ScalarQuantizedFlatVectorsReader(delegate, readState(dir, "0"));

            assertNull(reader.vectorLoaderSource("field"));
            assertNull(reader.vectorLoaderSource("field"));
            verify(delegate, times(1)).getFloatVectorValues("field");
        }
    }

    /** A failure while looking for a source is a fallback, not a query failure. */
    @SneakyThrows
    public void testVectorLoaderSource_whenTheReferenceValuesThrow_thenNull() {
        FlatVectorsReader delegate = mock(FlatVectorsReader.class);
        when(delegate.getFloatVectorValues("field")).thenThrow(new IOException("corrupt"));

        try (Directory dir = newFSDirectory(createTempDir())) {
            Faiss1040ScalarQuantizedFlatVectorsReader reader = new Faiss1040ScalarQuantizedFlatVectorsReader(delegate, readState(dir, "0"));
            assertNull(reader.vectorLoaderSource("field"));
        }
    }

    @SneakyThrows
    public void testResolveVectorDataPath_whenNoState_thenNull() {
        assertNull(Faiss1040ScalarQuantizedFlatVectorsReader.resolveVectorDataPath(null));
    }

    /**
     * The name carries the per-field format's {@code segmentSuffix}, because that is what distinguishes one
     * vector format's {@code .vec} from another's in the same segment.
     */
    @SneakyThrows
    public void testResolveVectorDataPath_whenFilesystemBacked_thenNamesTheSuffixedVecFile() {
        final Path directory = createTempDir();
        try (Directory dir = newFSDirectory(directory)) {
            final Path resolved = Faiss1040ScalarQuantizedFlatVectorsReader.resolveVectorDataPath(readState(dir, "7"));
            assertNotNull(resolved);
            assertEquals(directory.toRealPath(), resolved.getParent().toRealPath());
            assertEquals(SEGMENT_NAME + "_7.vec", resolved.getFileName().toString());
        }
    }

    /**
     * A compound segment's files are regions of a {@code .cfs} and its directory is not filesystem backed,
     * so it resolves to {@code null} and falls back. That is the compound-file fallback the design requires,
     * and it comes from this one type check.
     */
    @SneakyThrows
    public void testResolveVectorDataPath_whenNotFilesystemBacked_thenNull() {
        try (Directory dir = new ByteBuffersDirectory()) {
            assertNull(Faiss1040ScalarQuantizedFlatVectorsReader.resolveVectorDataPath(readState(dir, "0")));
        }
    }

    /** Wrapping the directory - which OpenSearch does - must not hide the filesystem underneath it. */
    @SneakyThrows
    public void testResolveVectorDataPath_whenTheDirectoryIsWrapped_thenStillResolves() {
        final Path directory = createTempDir();
        try (Directory dir = newFSDirectory(directory)) {
            final Directory wrapped = new FilterDirectory(dir) {
            };
            final Path resolved = Faiss1040ScalarQuantizedFlatVectorsReader.resolveVectorDataPath(readState(wrapped, "0"));
            assertNotNull(resolved);
            assertEquals(SEGMENT_NAME + "_0.vec", resolved.getFileName().toString());
        }
    }

    private static final String SEGMENT_NAME = "_9";

    /** A real read state, because {@code SegmentReadState}'s fields are final and a mock leaves them null. */
    private static SegmentReadState readState(final Directory directory, final String segmentSuffix) {
        final SegmentInfo segmentInfo = new SegmentInfo(
            directory,
            Version.LATEST,
            Version.LATEST,
            SEGMENT_NAME,
            4,
            false,
            false,
            null,
            Collections.emptyMap(),
            StringHelper.randomId(),
            new HashMap<>(),
            null
        );
        return new SegmentReadState(directory, segmentInfo, new FieldInfos(new FieldInfo[0]), IOContext.DEFAULT, segmentSuffix);
    }
}

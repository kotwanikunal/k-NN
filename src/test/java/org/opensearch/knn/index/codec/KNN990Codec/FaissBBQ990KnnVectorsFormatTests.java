/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN990Codec;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.KnnVectorsWriter;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.InfoStream;
import org.apache.lucene.util.Version;
import org.junit.Assert;
import org.mockito.MockedStatic;
import org.mockito.Mockito;
import org.mockito.stubbing.Answer;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.engine.KNNEngine;

import java.util.Map;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;

public class FaissBBQ990KnnVectorsFormatTests extends KNNTestCase {

    public void testFormatName_thenSuccess() {
        final FaissBBQ990KnnVectorsFormat format = new FaissBBQ990KnnVectorsFormat();
        assertEquals("FaissBBQ990KnnVectorsFormat", format.getName());
    }

    public void testDefaultConstructor_thenUsesDefaultThreshold() {
        final FaissBBQ990KnnVectorsFormat format = new FaissBBQ990KnnVectorsFormat();
        assertTrue(format.toString().contains("approximateThreshold=" + KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD_DEFAULT_VALUE));
    }

    public void testParameterizedConstructor_thenUsesProvidedThreshold() {
        final FaissBBQ990KnnVectorsFormat format = new FaissBBQ990KnnVectorsFormat(42);
        assertTrue(format.toString().contains("approximateThreshold=42"));
    }

    public void testGetMaxDimensions_thenUsesLuceneEngine() {
        try (MockedStatic<KNNEngine> mockedKNNEngine = Mockito.mockStatic(KNNEngine.class)) {
            mockedKNNEngine.when(() -> KNNEngine.getMaxDimensionByEngine(KNNEngine.LUCENE)).thenReturn(16000);

            final FaissBBQ990KnnVectorsFormat format = new FaissBBQ990KnnVectorsFormat();
            int result = format.getMaxDimensions("test-field");

            assertEquals(16000, result);
            mockedKNNEngine.verify(() -> KNNEngine.getMaxDimensionByEngine(KNNEngine.LUCENE));
        }
    }

    @SneakyThrows
    public void testFieldsWriter_thenReturnsFaissBBQWriter() {
        final SegmentInfo segmentInfo = new SegmentInfo(
            mock(Directory.class),
            mock(Version.class),
            mock(Version.class),
            "test-segment",
            0,
            false,
            false,
            mock(org.apache.lucene.codecs.Codec.class),
            mock(Map.class),
            new byte[16],
            mock(Map.class),
            null
        );

        final Directory directory = mock(Directory.class);
        final IndexOutput indexOutput = mock(IndexOutput.class);
        Mockito.when(directory.createOutput(anyString(), any())).thenReturn(indexOutput);

        final SegmentWriteState writeState = new SegmentWriteState(
            mock(InfoStream.class),
            directory,
            segmentInfo,
            mock(FieldInfos.class),
            null,
            mock(IOContext.class)
        );

        final FaissBBQ990KnnVectorsFormat format = new FaissBBQ990KnnVectorsFormat(0);
        try (MockedStatic<CodecUtil> mockedStaticCodecUtil = Mockito.mockStatic(CodecUtil.class)) {
            mockedStaticCodecUtil.when(
                () -> CodecUtil.writeIndexHeader(any(IndexOutput.class), anyString(), anyInt(), any(byte[].class), anyString())
            ).thenAnswer((Answer<Void>) invocation -> null);

            final KnnVectorsWriter writer = format.fieldsWriter(writeState);
            Assert.assertTrue(writer instanceof FaissBBQ990KnnVectorsWriter);
            writer.close();
        }
    }

    public void testApproximateThreshold_whenMultipleInstances_thenIndependent() {
        final FaissBBQ990KnnVectorsFormat format1 = new FaissBBQ990KnnVectorsFormat(100);
        final FaissBBQ990KnnVectorsFormat format2 = new FaissBBQ990KnnVectorsFormat(200);

        assertTrue(format1.toString().contains("approximateThreshold=100"));
        assertTrue(format2.toString().contains("approximateThreshold=200"));
    }

    public void testToString_thenContainsFormatInfo() {
        final FaissBBQ990KnnVectorsFormat format = new FaissBBQ990KnnVectorsFormat(50);
        final String str = format.toString();
        assertTrue(str.contains("FaissBBQ990KnnVectorsFormat"));
        assertTrue(str.contains("approximateThreshold=50"));
    }
}

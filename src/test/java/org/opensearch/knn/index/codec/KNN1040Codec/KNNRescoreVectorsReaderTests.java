/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import com.carrotsearch.randomizedtesting.annotations.ThreadLeakFilters;
import lombok.SneakyThrows;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsFormat;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FilterDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.MMapDirectory;
import org.mockito.Mock;
import org.opensearch.common.settings.ClusterSettings;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.codec.KNNRescoreVectorsReader;
import org.opensearch.knn.index.store.DirectIOReadPoolTests;
import org.opensearch.knn.index.store.DirectIOVectorIndexInput;
import org.opensearch.knn.index.store.KNNVectorReadIntent;
import org.opensearch.knn.index.store.KNNVectorStorageDirectory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static org.mockito.Mockito.when;
import static org.opensearch.knn.index.KNNSettings.KNN_DIRECT_IO_RESCORE_ENABLED_SETTING;
import static org.opensearch.knn.index.codec.KNN1040Codec.KNN1040ScalarQuantizedTestUtils.DIMENSION;
import static org.opensearch.knn.index.codec.KNN1040Codec.KNN1040ScalarQuantizedTestUtils.FIELD_NAME;
import static org.opensearch.knn.index.codec.KNN1040Codec.KNN1040ScalarQuantizedTestUtils.NUM_VECTORS;

/**
 * The joint the directory design rests on, compiled and run: handing a segment's own flat vectors
 * format a {@link SegmentReadState} whose {@link Directory} adds a read intent, and getting back a
 * working, byte-identical second view of the same full-precision vectors.
 *
 * <p>Everything else in the design is a consequence of this working. If Lucene's reader cannot be built
 * over a substituted directory, or builds but reads different bytes, the design has no mechanism and the
 * per-encoding coverage gap cannot be closed generically. So the assertions here are deliberately about
 * the two things that would sink it — <em>it composes</em> and <em>the bytes are the same</em> — plus the
 * one property that makes it safe to create on every segment of every index: it opens nothing until the
 * setting is on and a caller asks.
 *
 * <p>The segment is a real one, written by the real quantized format
 * ({@link KNN1040ScalarQuantizedTestUtils#writeQuantizedVectors}), so the {@code .vec}, its {@code .vemf}
 * sidecar and the {@code .veq} codes beside them are laid out exactly as they are in an index.
 */
@ThreadLeakFilters(defaultFilters = true, filters = { DirectIOReadPoolTests.ReadPoolThreadFilter.class })
public class KNNRescoreVectorsReaderTests extends KNNTestCase {

    @Mock
    ClusterSettings clusterSettings;

    @Override
    public void setUp() throws Exception {
        super.setUp();
        when(clusterService.getClusterSettings()).thenReturn(clusterSettings);
        KNNSettings.state().setClusterService(clusterService);
        setRescoreEnabled(true);
    }

    private void setRescoreEnabled(final boolean enabled) {
        when(clusterSettings.get(KNN_DIRECT_IO_RESCORE_ENABLED_SETTING)).thenReturn(enabled);
    }

    /** Records the name and intent of every {@code openInput} that reaches it, then delegates unchanged. */
    private static final class RecordingDirectory extends FilterDirectory {
        private record Open(String name, KNNVectorReadIntent intent) {
        }

        private final List<Open> opens = new ArrayList<>();

        RecordingDirectory(final Directory delegate) {
            super(delegate);
        }

        @Override
        public IndexInput openInput(final String name, final IOContext context) throws IOException {
            opens.add(new Open(name, KNNVectorReadIntent.of(context)));
            return in.openInput(name, context);
        }

        List<Open> opensOf(final String extension) {
            return opens.stream().filter(open -> open.name().endsWith(extension)).toList();
        }
    }

    private static List<float[]> readAll(final FloatVectorValues values) throws IOException {
        final List<float[]> vectors = new ArrayList<>(values.size());
        for (int ord = 0; ord < values.size(); ord++) {
            vectors.add(values.vectorValue(ord).clone());
        }
        return vectors;
    }

    /**
     * A read state over {@code directory} that is otherwise {@code state} — the same substitution
     * {@link KNNRescoreVectorsReader} performs, done here so the test can put a recorder in between.
     */
    private static SegmentReadState stateOver(final SegmentReadState state, final Directory directory) {
        return new SegmentReadState(directory, state.segmentInfo, state.fieldInfos, state.context, state.segmentSuffix);
    }

    /**
     * The joint itself. A second reader builds over a substituted directory, and the vectors it returns
     * are bit-for-bit the ones the ordinary reader returns.
     */
    @SneakyThrows
    public void testTheSecondReaderComposesAndReadsTheSameBytes() {
        final KNN1040ScalarQuantizedVectorsFormat format = new KNN1040ScalarQuantizedVectorsFormat();
        try (MMapDirectory dir = new MMapDirectory(createTempDir())) {
            final SegmentReadState state = KNN1040ScalarQuantizedTestUtils.writeQuantizedVectors(dir, format, random());
            final Lucene99FlatVectorsFormat rawFormat = format.rawVectorsFormat();

            final List<float[]> expected;
            try (FlatVectorsReader stock = rawFormat.fieldsReader(state)) {
                expected = readAll(stock.getFloatVectorValues(FIELD_NAME));
            }

            final RecordingDirectory recording = new RecordingDirectory(dir);
            try (KNNRescoreVectorsReader rescoreView = KNNRescoreVectorsReader.create(rawFormat, stateOver(state, recording))) {
                assertNotNull(rescoreView);
                final FloatVectorValues rescoreValues = rescoreView.floatVectorValues(FIELD_NAME);
                assertNotNull("the second reader did not compose over a substituted Directory", rescoreValues);
                assertEquals(NUM_VECTORS, rescoreValues.size());
                assertEquals(DIMENSION, rescoreValues.dimension());

                final List<float[]> actual = readAll(rescoreValues);
                assertEquals(expected.size(), actual.size());
                for (int ord = 0; ord < expected.size(); ord++) {
                    assertArrayEquals(
                        "ordinal " + ord + " differs between the mmap reader and the rescore view",
                        expected.get(ord),
                        actual.get(ord),
                        0.0f
                    );
                }
            }
        }
    }

    /** The intent reaches the {@code .vec} open, which is the only reason the substitution is worth doing. */
    @SneakyThrows
    public void testTheSecondReadersVectorDataOpenCarriesTheIntent() {
        final KNN1040ScalarQuantizedVectorsFormat format = new KNN1040ScalarQuantizedVectorsFormat();
        try (MMapDirectory dir = new MMapDirectory(createTempDir())) {
            final SegmentReadState state = KNN1040ScalarQuantizedTestUtils.writeQuantizedVectors(dir, format, random());
            final RecordingDirectory recording = new RecordingDirectory(dir);
            try (
                KNNRescoreVectorsReader rescoreView = KNNRescoreVectorsReader.create(format.rawVectorsFormat(), stateOver(state, recording))
            ) {
                assertNotNull(rescoreView.floatVectorValues(FIELD_NAME));

                final List<RecordingDirectory.Open> vectorDataOpens = recording.opensOf(".vec");
                assertEquals("the second reader should open .vec exactly once, saw " + vectorDataOpens, 1, vectorDataOpens.size());
                assertEquals(KNNVectorReadIntent.RESCORE, vectorDataOpens.get(0).intent());
            }
        }
    }

    /**
     * The quantized codes are the traversal path's file and the metadata sidecar is read sequentially at
     * open; neither may be tagged, and in this shape neither is even opened by the second reader.
     */
    @SneakyThrows
    public void testTheSecondReaderTouchesNothingButTheFullPrecisionVectors() {
        final KNN1040ScalarQuantizedVectorsFormat format = new KNN1040ScalarQuantizedVectorsFormat();
        try (MMapDirectory dir = new MMapDirectory(createTempDir())) {
            final SegmentReadState state = KNN1040ScalarQuantizedTestUtils.writeQuantizedVectors(dir, format, random());
            final RecordingDirectory recording = new RecordingDirectory(dir);
            try (
                KNNRescoreVectorsReader rescoreView = KNNRescoreVectorsReader.create(format.rawVectorsFormat(), stateOver(state, recording))
            ) {
                assertNotNull(rescoreView.floatVectorValues(FIELD_NAME));
                assertTrue("the quantized codes must not be opened by the rescore view", recording.opensOf(".veq").isEmpty());
                for (final RecordingDirectory.Open open : recording.opens) {
                    if (open.name().endsWith(".vec") == false) {
                        assertNull("only .vec may carry the intent, but " + open.name() + " did", open.intent());
                    }
                }
            }
        }
    }

    /**
     * Creating the view opens nothing, and with the setting off nothing is ever opened. This is the
     * property that lets a format construct one on every segment of every index: on a default node the
     * cost is a field and a wrapper object.
     */
    @SneakyThrows
    public void testNothingIsOpenedUntilTheSettingIsOnAndACallerAsks() {
        final KNN1040ScalarQuantizedVectorsFormat format = new KNN1040ScalarQuantizedVectorsFormat();
        try (MMapDirectory dir = new MMapDirectory(createTempDir())) {
            final SegmentReadState state = KNN1040ScalarQuantizedTestUtils.writeQuantizedVectors(dir, format, random());
            final RecordingDirectory recording = new RecordingDirectory(dir);
            setRescoreEnabled(false);
            try (
                KNNRescoreVectorsReader rescoreView = KNNRescoreVectorsReader.create(format.rawVectorsFormat(), stateOver(state, recording))
            ) {
                assertFalse("creating the view must not open anything", rescoreView.isOpen());
                assertTrue(recording.opens.isEmpty());

                assertNull("with the setting off there is no rescore view", rescoreView.floatVectorValues(FIELD_NAME));
                assertFalse(rescoreView.isOpen());
                assertTrue("with the setting off no second handle may be opened", recording.opens.isEmpty());

                // The setting is dynamic, so declining once must not be remembered as a failure.
                setRescoreEnabled(true);
                assertNotNull(rescoreView.floatVectorValues(FIELD_NAME));
                assertTrue(rescoreView.isOpen());
            }
        }
    }

    /** One reader for the life of the view, however many fields and queries ask for values. */
    @SneakyThrows
    public void testTheSecondHandleIsOpenedAtMostOnce() {
        final KNN1040ScalarQuantizedVectorsFormat format = new KNN1040ScalarQuantizedVectorsFormat();
        try (MMapDirectory dir = new MMapDirectory(createTempDir())) {
            final SegmentReadState state = KNN1040ScalarQuantizedTestUtils.writeQuantizedVectors(dir, format, random());
            final RecordingDirectory recording = new RecordingDirectory(dir);
            try (
                KNNRescoreVectorsReader rescoreView = KNNRescoreVectorsReader.create(format.rawVectorsFormat(), stateOver(state, recording))
            ) {
                for (int i = 0; i < 5; i++) {
                    assertNotNull(rescoreView.floatVectorValues(FIELD_NAME));
                }
                assertEquals(1, recording.opensOf(".vec").size());
            }
        }
    }

    /** Closing releases the second handle and the view stops offering values. */
    @SneakyThrows
    public void testCloseReleasesTheSecondHandle() {
        final KNN1040ScalarQuantizedVectorsFormat format = new KNN1040ScalarQuantizedVectorsFormat();
        try (MMapDirectory dir = new MMapDirectory(createTempDir())) {
            final SegmentReadState state = KNN1040ScalarQuantizedTestUtils.writeQuantizedVectors(dir, format, random());
            final KNNRescoreVectorsReader rescoreView = KNNRescoreVectorsReader.create(format.rawVectorsFormat(), state);
            assertNotNull(rescoreView.floatVectorValues(FIELD_NAME));
            assertTrue(rescoreView.isOpen());

            rescoreView.close();
            assertFalse(rescoreView.isOpen());
            assertNull("a closed view must not hand out values over a closed handle", rescoreView.floatVectorValues(FIELD_NAME));
            rescoreView.close();
        }
    }

    /**
     * A field the segment does not have is a {@code null}, not an exception: the caller's answer is to
     * read the vectors the ordinary way, and there is no configuration in which that is wrong.
     */
    @SneakyThrows
    public void testAnUnknownFieldYieldsNoValuesRatherThanFailing() {
        final KNN1040ScalarQuantizedVectorsFormat format = new KNN1040ScalarQuantizedVectorsFormat();
        try (MMapDirectory dir = new MMapDirectory(createTempDir())) {
            final SegmentReadState state = KNN1040ScalarQuantizedTestUtils.writeQuantizedVectors(dir, format, random());
            try (KNNRescoreVectorsReader rescoreView = KNNRescoreVectorsReader.create(format.rawVectorsFormat(), state)) {
                assertNull(rescoreView.floatVectorValues("no_such_field"));
            }
        }
    }

    /** A reader built without a read state — the plugin's own test and merge-instance paths — offers no view. */
    public void testCreateWithoutAReadStateOrFormatYieldsNothing() {
        assertNull(KNNRescoreVectorsReader.create(null, null));
        assertNull(KNNRescoreVectorsReader.create(new KNN1040ScalarQuantizedVectorsFormat().rawVectorsFormat(), null));
    }

    // -------------------------------------------------------------------------------------------------
    // The whole chain, with the storage layer's half installed: what/how end to end
    // -------------------------------------------------------------------------------------------------

    /**
     * Skipped rather than failed where {@code O_DIRECT} is unavailable — no
     * {@code ExtendedOpenOption.DIRECT} on this JDK, or a filesystem that answers {@code EINVAL}. Asked
     * through the production entry point, so the assumption is exactly the condition the directory's
     * fourth dispatch condition tests.
     */
    @SneakyThrows
    private void assumeDirectIOWorksHere() {
        final Path probe = createTempDir().resolve("probe");
        Files.write(probe, new byte[8192]);
        try (IndexInput input = DirectIOVectorIndexInput.open(probe)) {
            assumeTrue("O_DIRECT is not available here", input != null);
        }
    }

    /**
     * The other half of the invariant, and the one a regression would hide: with the storage directory
     * installed, the segment's <em>ordinary</em> reader — the object traversal, merge, warmup and
     * derived-source reads all share — opens the same {@code .vec} and is not routed. Nothing about
     * installing the directory changes how that file is read; only the intent does, and only the rescore
     * view attaches it.
     */
    @SneakyThrows
    public void testTheOrdinaryReaderOverTheSameStorageDirectoryIsNotRouted() {
        assumeDirectIOWorksHere();
        final KNN1040ScalarQuantizedVectorsFormat format = new KNN1040ScalarQuantizedVectorsFormat();
        try (MMapDirectory dir = new MMapDirectory(createTempDir())) {
            final SegmentReadState state = KNN1040ScalarQuantizedTestUtils.writeQuantizedVectors(dir, format, random());
            try (KNNVectorStorageDirectory storage = new KNNVectorStorageDirectory(dir, "test-index")) {
                try (FlatVectorsReader stock = format.rawVectorsFormat().fieldsReader(stateOver(state, storage))) {
                    assertNotNull(stock.getFloatVectorValues(FIELD_NAME));
                    assertEquals("an untagged .vec open must never be routed", 0, storage.routedOpens());
                    assertEquals(0, storage.declinedOpens());
                }
            }
        }
    }

    /**
     * With the setting off the storage directory is inert on this path for two independent reasons — the
     * view is never built, so no intent-bearing open is ever issued, and the directory checks the setting
     * itself. Asserted because "off means byte-identical to a node without the feature" is the property the
     * default configuration rests on.
     */
    @SneakyThrows
    public void testWithTheSettingOffTheStorageDirectoryRoutesNothing() {
        final KNN1040ScalarQuantizedVectorsFormat format = new KNN1040ScalarQuantizedVectorsFormat();
        try (MMapDirectory dir = new MMapDirectory(createTempDir())) {
            final SegmentReadState state = KNN1040ScalarQuantizedTestUtils.writeQuantizedVectors(dir, format, random());
            setRescoreEnabled(false);
            try (KNNVectorStorageDirectory storage = new KNNVectorStorageDirectory(dir, "test-index")) {
                try (
                    KNNRescoreVectorsReader rescoreView = KNNRescoreVectorsReader.create(
                        format.rawVectorsFormat(),
                        stateOver(state, storage)
                    )
                ) {
                    assertNull(rescoreView.floatVectorValues(FIELD_NAME));
                    assertEquals(0, storage.routedOpens());
                    assertEquals(0, storage.declinedOpens());
                    assertEquals(0, storage.wrappedContainers());
                }
            }
        }
    }
}

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN990Codec;

import lombok.SneakyThrows;
import org.apache.lucene.codecs.KnnVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatFieldVectorsWriter;
import org.apache.lucene.codecs.hnsw.FlatVectorScorerUtil;
import org.apache.lucene.codecs.hnsw.FlatVectorsWriter;
import org.apache.lucene.codecs.lucene99.Lucene99FlatVectorsFormat;
import org.apache.lucene.index.DocValuesSkipIndexType;
import org.apache.lucene.index.DocValuesType;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.AcceptDocs;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.KnnCollector;
import org.apache.lucene.search.VectorScorer;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.util.InfoStream;
import org.apache.lucene.util.StringHelper;
import org.apache.lucene.util.Version;
import org.opensearch.common.settings.ClusterSettings;
import org.opensearch.common.settings.Setting;
import org.opensearch.common.settings.Settings;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.codec.KNNRescoreVectorsReader;
import org.opensearch.knn.index.codec.nativeindex.AbstractNativeEnginesKnnVectorsReader;
import org.opensearch.knn.index.codec.scorer.HasRescoreVectorsReader;
import org.opensearch.knn.index.query.scorers.VectorScorerMode;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import static org.mockito.Mockito.when;

/**
 * Closes the <b>native-engines fp32 row</b> of the coverage matrix — the one the query-layer loader seam could
 * never reach, because its values are Lucene's own {@code OffHeapFloatVectorValues} and there is no plugin
 * values class anywhere on the path.
 *
 * <p>The row is closed with <b>no encoding-specific code at all</b>: {@code NativeEngines990KnnVectorsFormat}
 * hands {@code KNNRescoreVectorsReader} the very {@code Lucene99FlatVectorsFormat} it already reads the
 * {@code .vec} with, and the reader offers the result through {@link HasRescoreVectorsReader}. That the view's
 * values are Lucene's, built by Lucene, over the same file, is exactly what makes the identity assertions here
 * cheap to believe — and the reason a failure anywhere on this path is safe to treat as a silent fallback.
 *
 * <p>The segment is written by a plain {@code Lucene99FlatVectorsFormat} rather than by
 * {@code NativeEngines990KnnVectorsWriter}, because the fp32 {@code .vec} is the same file either way — the
 * native writer adds a {@code .faiss} graph this row's rescore reads never touch — and the reader under test is
 * the one the format really builds.
 */
public class NativeEnginesRescoreViewTests extends KNNTestCase {

    private static final String FIELD_NAME = "vector";
    private static final int DIMENSION = 16;
    private static final int NUM_VECTORS = 40;

    @Override
    public void setUp() throws Exception {
        super.setUp();
        setRescoreEnabled(true);
    }

    /**
     * Sets the Direct I/O rescore setting on a <em>real</em> {@link ClusterSettings} carrying every node-scope
     * k-NN default, rather than on a bare mock. The readers under test are the production ones, and closing a
     * {@code NativeEngines990KnnVectorsReader} initialises the node-wide native-memory and quantization-state
     * caches, which read several unrelated settings; a mock that answers {@code null} to those turns an
     * ordinary close into an NPE that has nothing to do with the rescore path.
     */
    private void setRescoreEnabled(final boolean enabled) {
        final Set<Setting<?>> nodeSettings = new HashSet<>(ClusterSettings.BUILT_IN_CLUSTER_SETTINGS);
        nodeSettings.addAll(
            KNNSettings.state()
                .getSettings()
                .stream()
                .filter(setting -> setting.getProperties().contains(Setting.Property.NodeScope))
                .collect(Collectors.toList())
        );
        when(clusterService.getClusterSettings()).thenReturn(
            new ClusterSettings(Settings.builder().put(KNNSettings.KNN_DIRECT_IO_RESCORE_ENABLED, enabled).build(), nodeSettings)
        );
        KNNSettings.state().setClusterService(clusterService);
    }

    /** An fp32 flat segment, as the native-engines format's own flat format writes one. */
    private static SegmentReadState writeFlatVectors(final MMapDirectory dir) throws Exception {
        final FieldInfo fieldInfo = new FieldInfo(
            FIELD_NAME,
            0,
            false,
            false,
            false,
            IndexOptions.NONE,
            DocValuesType.NONE,
            DocValuesSkipIndexType.NONE,
            -1,
            Map.of(),
            0,
            0,
            0,
            DIMENSION,
            VectorEncoding.FLOAT32,
            VectorSimilarityFunction.EUCLIDEAN,
            false,
            false
        );
        final FieldInfos fieldInfos = new FieldInfos(new FieldInfo[] { fieldInfo });
        final SegmentInfo segmentInfo = new SegmentInfo(
            dir,
            Version.LATEST,
            Version.LATEST,
            "_0",
            NUM_VECTORS,
            false,
            false,
            null,
            Collections.emptyMap(),
            StringHelper.randomId(),
            new HashMap<>(),
            null
        );
        final SegmentWriteState writeState = new SegmentWriteState(
            InfoStream.NO_OUTPUT,
            dir,
            segmentInfo,
            fieldInfos,
            null,
            IOContext.DEFAULT
        );

        final Lucene99FlatVectorsFormat format = new Lucene99FlatVectorsFormat(FlatVectorScorerUtil.getLucene99FlatVectorsScorer());
        try (FlatVectorsWriter writer = format.fieldsWriter(writeState)) {
            @SuppressWarnings("unchecked")
            final FlatFieldVectorsWriter<float[]> fieldWriter = (FlatFieldVectorsWriter<float[]>) writer.addField(fieldInfo);
            for (int i = 0; i < NUM_VECTORS; i++) {
                fieldWriter.addValue(i, randomVector());
            }
            writer.flush(NUM_VECTORS, null);
            writer.finish();
        }
        return new SegmentReadState(dir, segmentInfo, fieldInfos, IOContext.DEFAULT);
    }

    private static float[] randomVector() {
        final float[] v = new float[DIMENSION];
        for (int i = 0; i < DIMENSION; i++) {
            v[i] = random().nextFloat() * 2 - 1;
        }
        return v;
    }

    private static List<float[]> readAll(final FloatVectorValues values) throws IOException {
        final List<float[]> vectors = new ArrayList<>(values.size());
        for (int ord = 0; ord < values.size(); ord++) {
            vectors.add(values.vectorValue(ord).clone());
        }
        return vectors;
    }

    private static FloatVectorValues viewFrom(final KnnVectorsReader reader) {
        return ((HasRescoreVectorsReader) reader).rescoreVectorValues(FIELD_NAME);
    }

    /**
     * The row closes, and the vectors are the same ones. This is the assertion that says the coverage gap is
     * gone: the format the codec selects for a native-engine field produces a reader that can hand the rescore
     * path a second view of the same {@code .vec}.
     */
    @SneakyThrows
    public void testTheNativeEngineReaderOffersARescoreViewOfTheSameVectors() {
        try (MMapDirectory dir = new MMapDirectory(createTempDir())) {
            final SegmentReadState state = writeFlatVectors(dir);
            try (KnnVectorsReader reader = new NativeEngines990KnnVectorsFormat().fieldsReader(state)) {
                assertTrue(reader.getClass().getName(), reader instanceof HasRescoreVectorsReader);

                final FloatVectorValues values = reader.getFloatVectorValues(FIELD_NAME);
                final FloatVectorValues view = viewFrom(reader);
                assertNotNull("the native-engines fp32 row must offer a rescore view when the setting is on", view);

                assertEquals(NUM_VECTORS, view.size());
                assertEquals(DIMENSION, view.dimension());
                assertEquals(values.getVectorByteLength(), view.getVectorByteLength());
                assertEquals(values.getEncoding(), view.getEncoding());

                final List<float[]> expected = readAll(values);
                final List<float[]> actual = readAll(view);
                for (int ord = 0; ord < expected.size(); ord++) {
                    assertArrayEquals("ordinal " + ord, expected.get(ord), actual.get(ord), 0.0f);
                }
            }
        }
    }

    /**
     * The identity oracle at unit scale for this row: substituting the view changes no score, exactly. That is
     * the property the whole design rests on, and it is per-row because it is the encoding's {@code .vec} layout
     * that could make it false.
     */
    @SneakyThrows
    public void testScoresThroughTheViewAreIdenticalForTheNativeEngineRow() {
        try (MMapDirectory dir = new MMapDirectory(createTempDir())) {
            final SegmentReadState state = writeFlatVectors(dir);
            try (KnnVectorsReader reader = new NativeEngines990KnnVectorsFormat().fieldsReader(state)) {
                final FloatVectorValues values = reader.getFloatVectorValues(FIELD_NAME);
                final FloatVectorValues view = viewFrom(reader);
                assertNotNull(view);

                final float[] target = randomVector();
                final VectorScorer expected = VectorScorerMode.RESCORE.createScorer(values, target);
                final VectorScorer actual = VectorScorerMode.RESCORE.createScorer(view, target);
                assertNotNull(expected);
                assertNotNull(actual);

                final DocIdSetIterator expectedDocs = expected.iterator();
                final DocIdSetIterator actualDocs = actual.iterator();
                int scored = 0;
                for (int doc = expectedDocs.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = expectedDocs.nextDoc()) {
                    assertEquals(doc, actualDocs.nextDoc());
                    assertEquals("doc " + doc, expected.score(), actual.score(), 0.0f);
                    scored++;
                }
                assertEquals(NUM_VECTORS, scored);
                assertEquals(DocIdSetIterator.NO_MORE_DOCS, actualDocs.nextDoc());
            }
        }
    }

    /** With the setting off there is no view, so a default node never acquires a second handle on {@code .vec}. */
    @SneakyThrows
    public void testWithTheSettingOffTheNativeEngineRowOffersNoView() {
        setRescoreEnabled(false);
        try (MMapDirectory dir = new MMapDirectory(createTempDir())) {
            final SegmentReadState state = writeFlatVectors(dir);
            try (KnnVectorsReader reader = new NativeEngines990KnnVectorsFormat().fieldsReader(state)) {
                assertTrue(reader instanceof HasRescoreVectorsReader);
                assertNull(viewFrom(reader));
                setRescoreEnabled(true);
                assertNotNull("the setting is dynamic and declining is not remembered as failure", viewFrom(reader));
            }
        }
    }

    /**
     * Closing the reader closes the view with it, and the view was really opened first.
     *
     * <p>Asserted on {@code AbstractNativeEnginesKnnVectorsReader} directly rather than on
     * {@code NativeEngines990KnnVectorsReader}, because that is where the new code is — this row holds the view
     * on the base class rather than on a flat reader below it, which is a different close path from the faiss
     * scalar-quantized row's — and because the 990 reader's own {@code close()} additionally reaches into the
     * node-wide native-memory and quantization-state caches, which have nothing to do with this and need a
     * whole node's settings to initialise.
     */
    @SneakyThrows
    public void testClosingTheReaderClosesTheView() {
        try (MMapDirectory dir = new MMapDirectory(createTempDir())) {
            final SegmentReadState state = writeFlatVectors(dir);
            final Lucene99FlatVectorsFormat flatFormat = new Lucene99FlatVectorsFormat(FlatVectorScorerUtil.getLucene99FlatVectorsScorer());
            final KNNRescoreVectorsReader view = KNNRescoreVectorsReader.create(flatFormat, state);
            assertNotNull(view);
            final ProbeReader reader = new ProbeReader(state, flatFormat.fieldsReader(state), view);

            assertFalse("nothing may be opened until the view is asked for", view.isOpen());
            assertNotNull(viewFrom(reader));
            assertTrue("asking for the view opens the second handle", view.isOpen());

            reader.close();

            assertFalse(view.isOpen());
            assertNull(viewFrom(reader));
        }
    }

    /** The smallest concrete {@code AbstractNativeEnginesKnnVectorsReader}: nothing but the parts under test. */
    private static final class ProbeReader extends AbstractNativeEnginesKnnVectorsReader {

        private ProbeReader(
            final SegmentReadState state,
            final org.apache.lucene.codecs.hnsw.FlatVectorsReader flatVectorsReader,
            final KNNRescoreVectorsReader rescoreVectorsReader
        ) {
            super(state, flatVectorsReader, rescoreVectorsReader);
        }

        @Override
        public ByteVectorValues getByteVectorValues(final String field) throws IOException {
            return flatVectorsReader.getByteVectorValues(field);
        }

        @Override
        public void search(final String field, final float[] target, final KnnCollector collector, final AcceptDocs acceptDocs) {
            throw new UnsupportedOperationException();
        }

        @Override
        public void search(final String field, final byte[] target, final KnnCollector collector, final AcceptDocs acceptDocs) {
            throw new UnsupportedOperationException();
        }

        @Override
        public void warmUp(final String fieldName) {
            throw new UnsupportedOperationException();
        }
    }

    /**
     * The base class falls back to the flat reader's capability when it holds no view of its own. That fallback is
     * how the faiss scalar-quantized row, whose view lives on a plugin flat reader, reaches the same accessor.
     */
    @SneakyThrows
    public void testTheBaseReaderFallsBackToTheFlatReadersCapability() {
        try (MMapDirectory dir = new MMapDirectory(createTempDir())) {
            final SegmentReadState state = writeFlatVectors(dir);
            final Lucene99FlatVectorsFormat flatFormat = new Lucene99FlatVectorsFormat(FlatVectorScorerUtil.getLucene99FlatVectorsScorer());
            final FloatVectorValues offered = new org.opensearch.knn.index.vectorvalues.TestVectorValues.PreDefinedFloatVectorValues(
                List.of()
            );
            try (ProbeReader reader = new ProbeReader(state, new RescoreAwareFlatReader(flatFormat.fieldsReader(state), offered), null)) {
                assertSame(offered, viewFrom(reader));
            }
        }
    }

    /** A flat reader that offers the capability, as {@code Faiss1040ScalarQuantizedFlatVectorsReader} does. */
    private static final class RescoreAwareFlatReader extends org.apache.lucene.codecs.hnsw.FlatVectorsReader
        implements
            HasRescoreVectorsReader {

        private final org.apache.lucene.codecs.hnsw.FlatVectorsReader delegate;
        private final FloatVectorValues view;

        private RescoreAwareFlatReader(final org.apache.lucene.codecs.hnsw.FlatVectorsReader delegate, final FloatVectorValues view) {
            this.delegate = delegate;
            this.view = view;
        }

        @Override
        public org.apache.lucene.codecs.hnsw.FlatVectorsScorer getFlatVectorScorer(final String field) throws IOException {
            return delegate.getFlatVectorScorer(field);
        }

        @Override
        public FloatVectorValues rescoreVectorValues(final String field) {
            return view;
        }

        @Override
        public org.apache.lucene.util.hnsw.RandomVectorScorer getRandomVectorScorer(final String field, final float[] target)
            throws IOException {
            return delegate.getRandomVectorScorer(field, target);
        }

        @Override
        public org.apache.lucene.util.hnsw.RandomVectorScorer getRandomVectorScorer(final String field, final byte[] target)
            throws IOException {
            return delegate.getRandomVectorScorer(field, target);
        }

        @Override
        public void checkIntegrity() throws IOException {
            delegate.checkIntegrity();
        }

        @Override
        public FloatVectorValues getFloatVectorValues(final String field) throws IOException {
            return delegate.getFloatVectorValues(field);
        }

        @Override
        public ByteVectorValues getByteVectorValues(final String field) throws IOException {
            return delegate.getByteVectorValues(field);
        }

        @Override
        public long ramBytesUsed() {
            return delegate.ramBytesUsed();
        }

        @Override
        public void close() throws IOException {
            delegate.close();
        }
    }

    /** The values themselves stay Lucene's own and carry no capability — no per-encoding values wrapper. */
    @SneakyThrows
    public void testTheNativeEngineValuesAreLucenesOwnAndCarryNoCapability() {
        try (MMapDirectory dir = new MMapDirectory(createTempDir())) {
            final SegmentReadState state = writeFlatVectors(dir);
            try (KnnVectorsReader reader = new NativeEngines990KnnVectorsFormat().fieldsReader(state)) {
                final FloatVectorValues values = reader.getFloatVectorValues(FIELD_NAME);
                assertFalse(values.getClass().getName(), values instanceof HasRescoreVectorsReader);
                assertTrue(values.getClass().getName(), values.getClass().getName().startsWith("org.apache.lucene."));
            }
        }
    }

    /** An unknown field is an ordinary decline, not a throw. */
    @SneakyThrows
    public void testAnUnknownFieldOffersNoView() {
        try (MMapDirectory dir = new MMapDirectory(createTempDir())) {
            final SegmentReadState state = writeFlatVectors(dir);
            try (KnnVectorsReader reader = new NativeEngines990KnnVectorsFormat().fieldsReader(state)) {
                assertNull(((HasRescoreVectorsReader) reader).rescoreVectorValues("no_such_field"));
            }
        }
    }
}

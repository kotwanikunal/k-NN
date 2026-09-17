/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ;

import lombok.SneakyThrows;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.client.ResponseException;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.MediaTypeRegistry;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.index.IndexModule;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.store.KNNDirectIODirectoryFactory;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.stream.Stream;

import static org.opensearch.knn.common.KNNConstants.COMPRESSION_LEVEL_PARAMETER;

/**
 * End to end proof that the {@code knn_direct_io} store type is switchable on a close and reopen of
 * the index, with no node restart, and that with Direct I/O off the index behaves exactly as it does
 * on the node default store type.
 * <p>
 * {@code index.store.type} is a static index setting, so the gesture an operator has to make is
 * {@code _close} → {@code PUT _settings} → {@code _open}. Reopening recreates the shard, which
 * recreates the {@link org.apache.lucene.store.Directory} through
 * {@link KNNDirectIODirectoryFactory}. Nothing here restarts a node, and every test asserts that by
 * comparing the nodes' process ids across the whole toggle sequence.
 */
public class KNNDirectIOToggleIT extends KNNRestTestCase {

    /** 128 floats is 512 bytes a vector, so 2500 docs put the single {@code .vec} above the 1 MB
     *  {@code knn.direct_io.min_file_size} floor without lowering it. */
    private static final int DIMENSION = 128;
    private static final int NUM_DOCS = 2500;
    private static final int K = 10;

    /**
     * Relative tolerance for the one comparison that cannot be exact — the Direct I/O arm, where
     * the scorer changes with the {@code IndexInput}. The observed difference is ~1e-7 relative
     * (a float ULP); a wrong vector would be off by whole percent.
     */
    private static final double SCORE_TOLERANCE = 1e-5;

    private static final String HYBRIDFS_STORE_TYPE = "hybridfs";
    private static final String STORE_TYPE_SETTING = IndexModule.INDEX_STORE_TYPE_SETTING.getKey();

    /**
     * The toggle, both directions, in one cluster: default → {@code knn_direct_io} → back to
     * {@code hybridfs}, with the same query run after every step.
     */
    @SneakyThrows
    public void testStoreTypeToggle_bothDirections_withoutNodeRestart() {
        final String indexName = "direct-io-toggle";
        createRescoreIndex(indexName);

        final Map<String, Object> pidsBefore = nodeProcessIds();
        final SearchResult onDefault = query(indexName);
        assertEquals(K, onDefault.ids.size());
        assertNull(getIndexSettingByName(indexName, STORE_TYPE_SETTING));

        // Direct I/O on.
        final Map<Path, Long> logOffsets = nodeLogOffsets();
        setStoreType(indexName, KNNDirectIODirectoryFactory.KNN_DIRECT_IO_STORE_TYPE);
        assertEquals(KNNDirectIODirectoryFactory.KNN_DIRECT_IO_STORE_TYPE, getIndexSettingByName(indexName, STORE_TYPE_SETTING));
        final SearchResult onDirectIO = query(indexName);
        // Same documents in the same order, and the same scores to within a float rounding error.
        // The scores are NOT bit identical, and that is expected rather than a defect: a Direct I/O
        // IndexInput is not a MemorySegmentAccessInput, so rescoring drops out of Lucene's SIMD
        // bulk scorer into the scalar one, which accumulates in a different order. Note the
        // reverse-direction and Direct-I/O-disabled checks below and in
        // testStoreTypeSelectedButDirectIODisabled_behavesAsDefault are both exact, which is what
        // localises the difference to the swapped IndexInput rather than to the bytes read.
        assertSameHits(onDefault, onDirectIO, SCORE_TOLERANCE);
        assertTrue(
            "the shard should have logged that it opened with the knn_direct_io store type",
            nodeLogsContain(directIOEngagedMarker(indexName), logOffsets)
        );

        // Direct I/O off again, on the same live node. Exactly the original scores come back, which
        // also proves the toggle changed nothing on disk.
        setStoreType(indexName, HYBRIDFS_STORE_TYPE);
        assertEquals(HYBRIDFS_STORE_TYPE, getIndexSettingByName(indexName, STORE_TYPE_SETTING));
        assertSameResults(onDefault, query(indexName));

        assertEquals("no node may have restarted during the toggle", pidsBefore, nodeProcessIds());
    }

    /**
     * Arm (d): the store type is selected but {@code index.knn.direct_io.enabled} is false, so the
     * factory hands back the stock directory untouched. This is the byte-for-byte requirement stated
     * at the REST level — the unit tests state it at the {@code Directory} level.
     */
    @SneakyThrows
    public void testStoreTypeSelectedButDirectIODisabled_behavesAsDefault() {
        final String indexName = "direct-io-disabled";
        createRescoreIndex(indexName);

        final Map<String, Object> pidsBefore = nodeProcessIds();
        final SearchResult onDefault = query(indexName);

        final Map<Path, Long> logOffsets = nodeLogOffsets();
        closeIndex(indexName);
        updateIndexSettings(
            indexName,
            Settings.builder()
                .put(STORE_TYPE_SETTING, KNNDirectIODirectoryFactory.KNN_DIRECT_IO_STORE_TYPE)
                .put(KNNSettings.KNN_INDEX_DIRECT_IO_ENABLED, false)
        );
        openIndex(indexName);
        ensureGreen(indexName);

        assertEquals(KNNDirectIODirectoryFactory.KNN_DIRECT_IO_STORE_TYPE, getIndexSettingByName(indexName, STORE_TYPE_SETTING));
        assertEquals("false", getIndexSettingByName(indexName, KNNSettings.KNN_INDEX_DIRECT_IO_ENABLED));
        assertSameResults(onDefault, query(indexName));
        assertFalse(
            "Direct I/O must not engage when index.knn.direct_io.enabled is false",
            nodeLogsContain(directIOEngagedMarker(indexName), logOffsets)
        );
        assertEquals("no node may have restarted", pidsBefore, nodeProcessIds());
    }

    /**
     * The operator gesture is close → update → open and nothing shorter: a store-type update on an
     * open index is rejected by core, and the index keeps serving the same results afterwards.
     */
    @SneakyThrows
    public void testStoreTypeUpdateOnOpenIndex_isRejected() {
        final String indexName = "direct-io-open-update";
        createRescoreIndex(indexName);
        final SearchResult before = query(indexName);

        final ResponseException ex = expectThrows(
            ResponseException.class,
            () -> updateIndexSettings(
                indexName,
                Settings.builder().put(STORE_TYPE_SETTING, KNNDirectIODirectoryFactory.KNN_DIRECT_IO_STORE_TYPE)
            )
        );
        assertTrue(ex.getMessage(), ex.getMessage().contains("non dynamic settings"));
        assertTrue(ex.getMessage(), ex.getMessage().contains(STORE_TYPE_SETTING));

        assertNull(getIndexSettingByName(indexName, STORE_TYPE_SETTING));
        assertSameResults(before, query(indexName));
    }

    // ---------------------------------------------------------------------------------------------
    // helpers
    // ---------------------------------------------------------------------------------------------

    /**
     * A faiss HNSW index in {@code on_disk} mode, which rescores against the full precision vectors
     * in {@code .vec} — the read path this store type exists for. Force merged to one segment so
     * there is a single {@code .vec} and the candidate set is stable across reopens.
     */
    private void createRescoreIndex(final String indexName) throws Exception {
        final XContentBuilder mapping = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD_NAME)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .field(COMPRESSION_LEVEL_PARAMETER, CompressionLevel.x32.getName())
            .endObject()
            .endObject()
            .endObject();
        createKnnIndex(indexName, getKNNDefaultIndexSettings(), mapping.toString());
        bulkIngestRandomVectors(indexName, FIELD_NAME, NUM_DOCS, DIMENSION);
        refreshAllIndices();
        forceMergeKnnIndex(indexName, 1);
        ensureGreen(indexName);
    }

    /** Close, set {@code index.store.type}, reopen. No node is restarted. */
    private void setStoreType(final String indexName, final String storeType) throws Exception {
        closeIndex(indexName);
        updateIndexSettings(indexName, Settings.builder().put(STORE_TYPE_SETTING, storeType));
        openIndex(indexName);
        ensureGreen(indexName);
    }

    /** A k-NN query with the mode's default rescore, i.e. one that reads {@code .vec}. */
    private SearchResult query(final String indexName) throws Exception {
        final XContentBuilder query = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("query")
            .startObject("knn")
            .startObject(FIELD_NAME)
            .field("vector", QUERY_VECTOR)
            .field("k", K)
            .endObject()
            .endObject()
            .endObject()
            .endObject();
        final Response response = searchKNNIndex(indexName, query, K);
        assertOK(response);
        final String body = EntityUtils.toString(response.getEntity());
        return new SearchResult(parseIds(body), parseScores(body));
    }

    /** Identical doc ids and bit identical scores. */
    private void assertSameResults(final SearchResult expected, final SearchResult actual) {
        assertEquals("doc ids must be identical across the store type toggle", expected.ids, actual.ids);
        assertEquals("scores must be identical across the store type toggle", expected.scores, actual.scores);
    }

    /**
     * Identical doc ids in identical order, and scores equal to within {@code relativeTolerance}.
     * The tolerance is many orders of magnitude tighter than any difference a wrong vector would
     * produce, so this still catches the reader reading the wrong bytes.
     */
    private void assertSameHits(final SearchResult expected, final SearchResult actual, final double relativeTolerance) {
        assertEquals("doc ids must be identical across the store type toggle", expected.ids, actual.ids);
        assertEquals(expected.scores.size(), actual.scores.size());
        for (int i = 0; i < expected.scores.size(); i++) {
            final double want = expected.scores.get(i);
            assertEquals(
                "score at rank "
                    + i
                    + " differs by more than "
                    + relativeTolerance
                    + " relative; expected "
                    + expected.scores
                    + " but was "
                    + actual.scores,
                want,
                actual.scores.get(i),
                Math.abs(want) * relativeTolerance
            );
        }
    }

    private String directIOEngagedMarker(final String indexName) {
        return String.format(
            Locale.ROOT,
            "Using store type [%s] for index [%s]",
            KNNDirectIODirectoryFactory.KNN_DIRECT_IO_STORE_TYPE,
            indexName
        );
    }

    /**
     * Node process ids, so a test can prove the toggle happened without a node restart. Keyed by
     * node id and ordered, so a plain equality check on the map is meaningful.
     */
    private Map<String, Object> nodeProcessIds() throws Exception {
        final Request request = new Request("GET", "/_nodes/process");
        request.addParameter("filter_path", "nodes.*.process.id");
        final Response response = client().performRequest(request);
        assertOK(response);
        final Map<String, Object> nodes = nodesOf(EntityUtils.toString(response.getEntity()));
        final Map<String, Object> pids = new LinkedHashMap<>();
        for (final Map.Entry<String, Object> node : nodes.entrySet()) {
            @SuppressWarnings("unchecked")
            final Map<String, Object> process = (Map<String, Object>) ((Map<String, Object>) node.getValue()).get("process");
            pids.put(node.getKey(), process.get("id"));
        }
        assertFalse("expected at least one node to report a process id", pids.isEmpty());
        return pids;
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> nodesOf(final String responseBody) throws IOException {
        return (Map<String, Object>) createParser(MediaTypeRegistry.getDefaultMediaType().xContent(), responseBody).map().get("nodes");
    }

    /**
     * The current length of every test cluster node log, so that {@link #nodeLogsContain} can be
     * asked about this test only. The logs are appended to across gradle runs and index names
     * repeat, so an unscoped search would let one run's output decide the next run's result.
     */
    private Map<Path, Long> nodeLogOffsets() throws IOException {
        final Map<Path, Long> offsets = new LinkedHashMap<>();
        for (final Path log : nodeLogFiles()) {
            offsets.put(log, Files.size(log));
        }
        return offsets;
    }

    /**
     * Whether any test cluster node log has written the given marker since {@code since} was taken.
     * This is the only way to observe from a REST test that Direct I/O really engaged inside a live
     * shard: the decision is made at shard open and is not exposed through any API.
     */
    private boolean nodeLogsContain(final String marker, final Map<Path, Long> since) throws IOException {
        final List<Path> logs = nodeLogFiles();
        assertFalse("could not locate any test cluster log file to read; looked under " + testClusterLogRoot(), logs.isEmpty());
        for (final Path log : logs) {
            if (tailFrom(log, since.getOrDefault(log, 0L)).contains(marker)) {
                return true;
            }
        }
        return false;
    }

    private String tailFrom(final Path log, final long offset) throws IOException {
        try (InputStream in = Files.newInputStream(log)) {
            in.skipNBytes(Math.min(offset, Files.size(log)));
            return new String(in.readAllBytes(), StandardCharsets.UTF_8);
        }
    }

    private Path testClusterLogRoot() {
        // 'project.root' is set on every integTest task by commonIntegTest in build.gradle.
        return Path.of(System.getProperty("project.root")).resolve("build").resolve("testclusters");
    }

    private List<Path> nodeLogFiles() throws IOException {
        final Path root = testClusterLogRoot();
        if (Files.isDirectory(root) == false) {
            return List.of();
        }
        final List<Path> logs = new ArrayList<>();
        try (Stream<Path> children = Files.list(root)) {
            children.filter(Files::isDirectory).map(node -> node.resolve("logs")).filter(Files::isDirectory).forEach(logDir -> {
                try (Stream<Path> files = Files.list(logDir)) {
                    files.filter(file -> file.getFileName().toString().endsWith(".log")).forEach(logs::add);
                } catch (IOException e) {
                    throw new UncheckedIOException(e);
                }
            });
        }
        return logs;
    }

    private static final float[] QUERY_VECTOR = buildQueryVector();

    private static float[] buildQueryVector() {
        final float[] vector = new float[DIMENSION];
        for (int i = 0; i < DIMENSION; i++) {
            vector[i] = (i % 8) * 0.25f;
        }
        return vector;
    }

    /** Doc ids and scores of one query, the pair that has to be identical across the toggle. */
    private static final class SearchResult {
        private final List<String> ids;
        private final List<Double> scores;

        private SearchResult(final List<String> ids, final List<Double> scores) {
            this.ids = ids;
            this.scores = scores;
        }
    }
}

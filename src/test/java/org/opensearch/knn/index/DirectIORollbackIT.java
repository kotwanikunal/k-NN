/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import lombok.SneakyThrows;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.client.Response;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.common.xcontent.support.XContentMapValues;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.knn.KNNRestTestCase;
import org.opensearch.knn.KNNResult;
import org.opensearch.knn.index.query.KNNQueryBuilder;
import org.opensearch.knn.plugin.stats.StatNames;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;

import static org.opensearch.knn.common.KNNConstants.COMPRESSION_LEVEL_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.MODE_PARAMETER;

/**
 * O1 on a running cluster: {@code knn.direct_io.rescore.enabled} turned off and back on <b>under queries</b>,
 * with no reopen, read off the counters an operator would read.
 *
 * <p>The unit tests in {@code DirectIORouteSwitchTests} prove the mechanism against one input. This proves
 * the thing an operator actually does: a {@code PUT _cluster/settings} against a node that is serving
 * queries, with the index left alone. Three phases over one set of readers --
 *
 * <ul>
 *   <li><b>on</b>: the {@code .vec} is routed, reads are staged on the Direct I/O pool, the store fills;</li>
 *   <li><b>off</b>: the same readers serve the same queries through the delegate input, no device read
 *       happens, and the store gives its slabs back;</li>
 *   <li><b>on again</b>: the same readers go back to {@code O_DIRECT} and the store grows again.</li>
 * </ul>
 *
 * <p>and {@code dispatch.routed_opens} frozen across both flips is what makes it a proof rather than a
 * counter: if it moved, a reader had been rebuilt and the test would be measuring reopened readers.
 *
 * <p><b>Why {@code switched_to_mmap} is reported and not asserted (MEASURED here, 0 in every phase).</b>
 * What serves a query's reads is not the root input the dispatch rule produced, it is a per-query
 * <em>slice</em> of it: Lucene's flat vectors reader builds one values object, and so one slice, per query
 * per segment. A slice is constructed after the flip, so it is <b>born</b> on the route the switch names at
 * that moment rather than changing to it, and a route that was never changed is correctly not counted as a
 * change. The counters that move are {@code fallback_reads}, which rises while {@code staged_reads} and
 * {@code blocking_reads} stand still, and that pair is the rollback. {@code switched_to_mmap} moves for an
 * input that outlives a flip and prefetches again, which {@code DirectIORouteSwitchTests} exercises
 * directly and which on a node is the long-lived exact-search inputs.
 *
 * <p><b>When this skips.</b> Direct I/O needs {@code ExtendedOpenOption.DIRECT} and a filesystem that
 * accepts {@code O_DIRECT} for the data directory. If the cluster routes nothing the test assumes out
 * rather than failing, because the thing that is missing is the substrate and not the code. The skip is
 * visible: it names {@code routed_opens} and the decline counters.
 */
public class DirectIORollbackIT extends KNNRestTestCase {

    private static final String INDEX = "direct-io-rollback";
    private static final String FIELD = "v";
    private static final int DIMENSION = 128;
    private static final int DOCS = 2_000;
    private static final int QUERIES = 20;
    private static final int K = 10;

    private static final String ENABLED = KNNSettings.KNN_DIRECT_IO_RESCORE_ENABLED;
    private static final String MIN_FILE_SIZE = KNNSettings.KNN_DIRECT_IO_MIN_FILE_SIZE;
    private static final String CACHE_MAX_BYTES = KNNSettings.KNN_DIRECT_IO_RESCORE_CACHE_MAX_BYTES;

    @Override
    public void tearDown() throws Exception {
        // Explicit values in both directions and never null: removing a persisted key wedges
        // _cluster/settings, which is the rule the whole phase works under.
        try {
            updateClusterSettings(ENABLED, false);
            updateClusterSettings(CACHE_MAX_BYTES, "0b");
            updateClusterSettings(MIN_FILE_SIZE, "10mb");
            try {
                deleteKNNIndex(INDEX);
            } catch (Exception e) {
                // The base class wipes the cluster's indices too, so a 404 here means it got there first.
                logger.info("could not delete {}; the cluster wipe had already removed it", INDEX, e);
            }
        } finally {
            super.tearDown();
        }
    }

    @SneakyThrows
    public void testTheSwitchReachesOpenSegmentsWithNoReopen() {
        updateClusterSettings(MIN_FILE_SIZE, "0b");
        updateClusterSettings(CACHE_MAX_BYTES, "64mb");
        updateClusterSettings(ENABLED, true);

        createOnDiskIndex();
        ingest();
        // One segment, left alone from here on: refresh_interval is -1 and nothing is indexed after this,
        // so no reader can be rebuilt and routed_opens is free to be an assertion.
        forceMergeKnnIndex(INDEX, 1);

        final float[][] queries = randomVectors(4242L, QUERIES);

        // -- phase on --
        final List<Integer> idsOn = queryAll(queries);
        final Counters on = counters();
        // Asserted, not assumed. A run that routed nothing proves nothing, and a skip here is how a chain
        // that stopped routing -- a new layer, a content decline -- would have passed as green.
        assertTrue("this cluster routed nothing, so there is no switch to test: " + on, on.routedOpens > 0);
        assertTrue("and nothing was staged, so the route was not exercised: " + on, on.stagedReads > 0);
        assertTrue("the store must have filled, " + on, on.slabBytes > 0);
        assertEquals("nothing may have gone through the delegate yet, " + on, 0L, on.fallbackReads);

        // -- phase off: one PUT, no reopen, the same readers --
        updateClusterSettings(ENABLED, false);
        final List<Integer> idsOff = queryAll(queries);
        final Counters off = counters();
        assertEquals("the answers must not change with the route, on against off", idsOn, idsOff);
        assertTrue("the off phase must have read through the delegate, " + off, off.fallbackReads > on.fallbackReads);
        assertEquals("and must not have touched the device, " + off, on.deviceReads(), off.deviceReads());
        assertEquals("the store must have given its slabs back, " + off, 0L, off.slabBytes);
        assertEquals("and no reader may have been reopened, " + off, on.routedOpens, off.routedOpens);
        logger.info("O1 off phase: {}", off);

        // -- phase on again: the same readers go back --
        updateClusterSettings(ENABLED, true);
        final List<Integer> idsOnAgain = queryAll(queries);
        final Counters onAgain = counters();
        assertEquals("the answers must not change with the route, off against on again", idsOn, idsOnAgain);
        assertTrue("the device must be read again, " + onAgain, onAgain.deviceReads() > off.deviceReads());
        assertEquals("and the delegate must stop serving reads, " + onAgain, off.fallbackReads, onAgain.fallbackReads);
        assertTrue("the store must grow again, " + onAgain, onAgain.slabBytes > 0);
        assertEquals("still with no reopen, " + onAgain, on.routedOpens, onAgain.routedOpens);
        logger.info("O1 on-again phase: {}", onAgain);
    }

    /**
     * An {@code on_disk} 32x faiss index: the quantized graph answers approximately and the rescore pass
     * reads the full-precision {@code .vec}, which is the file the dispatch rule routes and the only read
     * shape this feature exists for. {@code index.use_compound_file} false because a compound segment has
     * no {@code .vec} of its own and is declined by condition 3.
     */
    @SneakyThrows
    private void createOnDiskIndex() {
        final XContentBuilder mapping = XContentFactory.jsonBuilder()
            .startObject()
            .startObject("properties")
            .startObject(FIELD)
            .field("type", "knn_vector")
            .field("dimension", DIMENSION)
            .field(MODE_PARAMETER, "on_disk")
            .field(COMPRESSION_LEVEL_PARAMETER, "32x")
            .endObject()
            .endObject()
            .endObject();
        final Settings settings = Settings.builder()
            .put("index.knn", true)
            .put("number_of_shards", 1)
            .put("number_of_replicas", 0)
            .put("index.refresh_interval", "-1")
            .put("index.use_compound_file", false)
            .build();
        createKnnIndex(INDEX, settings, mapping.toString());
    }

    @SneakyThrows
    private void ingest() {
        bulkAddKnnDocs(INDEX, FIELD, randomVectors(20261014L, DOCS), DOCS);
        refreshIndex(INDEX);
    }

    /** Every query, in order, so the whole run is one list of ids to compare phase against phase. */
    @SneakyThrows
    private List<Integer> queryAll(final float[][] queries) {
        final List<Integer> ids = new ArrayList<>(queries.length * K);
        for (final float[] query : queries) {
            final Response response = searchKNNIndex(INDEX, new KNNQueryBuilder(FIELD, query, K), K);
            final String body = EntityUtils.toString(response.getEntity());
            for (final KNNResult result : parseSearchResponse(body, FIELD)) {
                ids.add(Integer.parseInt(result.getDocId()));
            }
        }
        return ids;
    }

    private static float[][] randomVectors(final long seed, final int count) {
        final Random random = new Random(seed);
        final float[][] vectors = new float[count][DIMENSION];
        for (final float[] vector : vectors) {
            for (int i = 0; i < DIMENSION; i++) {
                vector[i] = (random.nextFloat() - 0.5f) * 20.0f;
            }
        }
        return vectors;
    }

    /**
     * The {@code direct_io_stats} an operator reads, summed over the cluster's nodes. Node-wide and
     * cumulative, so every assertion above is on a difference between two of these and never on an
     * absolute.
     */
    @SneakyThrows
    private Counters counters() {
        final Response response = getKnnStats(Collections.emptyList(), Collections.emptyList());
        final List<Map<String, Object>> nodes = parseNodeStatsResponse(EntityUtils.toString(response.getEntity()));
        final Counters total = new Counters();
        for (final Map<String, Object> node : nodes) {
            total.routedOpens += value(node, "dispatch.routed_opens");
            total.stagedReads += value(node, "reads.staged_reads");
            total.blockingReads += value(node, "reads.blocking_reads");
            total.fallbackReads += value(node, "reads.fallback_reads");
            total.switchedToMmap += value(node, "reads.switched_to_mmap");
            total.switchedToDirect += value(node, "reads.switched_to_direct");
            total.slabBytes += value(node, "cache.slab_bytes");
        }
        return total;
    }

    private static long value(final Map<String, Object> node, final String path) {
        final Object found = XContentMapValues.extractValue(StatNames.DIRECT_IO_STATS.getName() + "." + path, node);
        return found instanceof Number number ? number.longValue() : 0L;
    }

    /** One snapshot of the counters, with a {@code toString} every assertion message carries. */
    private static final class Counters {
        private long routedOpens;
        private long stagedReads;
        private long blockingReads;
        private long fallbackReads;
        private long switchedToMmap;
        private long switchedToDirect;
        private long slabBytes;

        private long deviceReads() {
            return stagedReads + blockingReads;
        }

        @Override
        public String toString() {
            return String.format(
                Locale.ROOT,
                "routed_opens=%d staged_reads=%d blocking_reads=%d fallback_reads=%d "
                    + "switched_to_mmap=%d switched_to_direct=%d slab_bytes=%d",
                routedOpens,
                stagedReads,
                blockingReads,
                fallbackReads,
                switchedToMmap,
                switchedToDirect,
                slabBytes
            );
        }
    }
}

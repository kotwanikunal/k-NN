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
 * W3 on a cluster of more than one node: one {@code PUT _cluster/settings} frees <b>every</b> node's slabs,
 * with no traffic after the PUT and no reopen.
 *
 * <p><b>Why this is not the rollback IT again.</b> {@code DirectIORollbackIT} proves the route switch on a
 * node that is serving queries, and it reads the counters summed over the cluster. The finding W3 exists
 * for is a different one and is per node: on the campaign cluster, turning Direct I/O off left 4.5 GiB of
 * slabs held on a node that then served no Direct I/O read at all (ref/kb/oss/output/STATE.md lines
 * 731-742, ref/kb/CHALLENGES.md 8 and 19). Nothing read the setting again on that node because nothing read
 * anything. So the property is "the applier reached every node and the memory went back there", and the
 * only way to see it is per node, with no query in between.
 *
 * <p>Both keys an operator would reach for are covered, because they are different mechanisms:
 * {@code cache.max_bytes: 0b} is the cache's own consumer resizing the store to nothing, and
 * {@code rescore.enabled: false} publishes into the directory, which releases the store with the route
 * because slabs serve routed reads only. Neither case queries after the PUT, which is the whole point.
 *
 * <p><b>When this skips.</b> Below two nodes there is nothing multi-node to prove, and the skip names the
 * node count: the acceptance is the {@code -PnumNodes=3} run. It also assumes out where the substrate has
 * no {@code O_DIRECT}, like every other Direct I/O IT, and that skip names {@code routed_opens}.
 */
public class DirectIOSettingsMultiNodeIT extends KNNRestTestCase {

    private static final String INDEX = "direct-io-settings-multinode";
    private static final String FIELD = "v";
    private static final int DIMENSION = 128;
    private static final int DOCS = 3_000;
    private static final int QUERIES = 12;
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
                logger.info("could not delete {}; the cluster wipe had already removed it", INDEX, e);
            }
        } finally {
            super.tearDown();
        }
    }

    @SneakyThrows
    public void testCacheMaxBytesZeroFreesEverySlabOnEveryNode() {
        final List<Node> loaded = loadEveryNode();

        // One PUT, and nothing else: no query, no refresh, no reopen.
        updateClusterSettings(CACHE_MAX_BYTES, "0b");

        for (final Node after : nodes()) {
            assertEquals("node " + after.name() + " must hold no slab, " + after, 0L, after.slabs());
            assertEquals("and no cached byte, " + after, 0L, after.bytes());
            assertEquals("and must report the target it was given, " + after, 0L, after.configuredMaxBytes());
        }
        assertNoReaderWasReopened(loaded);
    }

    @SneakyThrows
    public void testRescoreDisabledFreesEverySlabOnEveryNode() {
        final List<Node> loaded = loadEveryNode();

        // The other key an operator reaches for, and the one the 4.5 GiB finding was about: a node that
        // stops routing must stop holding, and on that node nothing queries afterwards to notice. So this
        // is one PUT and nothing else, exactly like the case above.
        updateClusterSettings(ENABLED, false);

        for (final Node after : nodes()) {
            assertEquals("node " + after.name() + " must hold no slab, " + after, 0L, after.slabs());
            assertEquals("and no cached byte, " + after, 0L, after.bytes());
        }
        assertNoReaderWasReopened(loaded);
    }

    // ---------------------------------------------------------------------------------------------------
    // The load every case starts from
    // ---------------------------------------------------------------------------------------------------

    /**
     * Routes and fills on every node that holds a shard, and returns the per-node snapshot the assertions
     * are taken against. Asserted rather than assumed at both ends: a run in which no node routed, or in
     * which only one node held slabs, proves nothing about an applier reaching a cluster.
     */
    @SneakyThrows
    private List<Node> loadEveryNode() {
        updateClusterSettings(MIN_FILE_SIZE, "0b");
        updateClusterSettings(CACHE_MAX_BYTES, "64mb");
        updateClusterSettings(ENABLED, true);

        createOnDiskIndex();
        bulkAddKnnDocs(INDEX, FIELD, randomVectors(20261027L, DOCS), DOCS);
        refreshIndex(INDEX);
        forceMergeKnnIndex(INDEX, 1);
        queryAll(randomVectors(4242L, QUERIES));

        final List<Node> loaded = nodes();
        assumeTrue("this cluster has " + loaded.size() + " node(s); the W3 acceptance is the -PnumNodes=3 run", loaded.size() >= 2);
        final List<Node> routing = loaded.stream().filter(node -> node.routedOpens() > 0).toList();
        assumeTrue("this cluster routed nothing, so there is no per-node release to see: " + loaded, routing.isEmpty() == false);
        assertTrue(
            "only " + routing.size() + " of " + loaded.size() + " nodes routed, so this is not a multi-node proof: " + loaded,
            routing.size() >= 2
        );
        for (final Node node : routing) {
            assertTrue("node " + node.name() + " routed but holds no slab, " + node, node.slabs() > 0);
            assertTrue("and no cached byte, " + node, node.bytes() > 0);
        }
        logger.info("W3 multi-node load: {} of {} nodes routing, {}", routing.size(), loaded.size(), loaded);
        return loaded;
    }

    /** Every node's {@code routed_opens} frozen: the release happened under the readers, not after a reopen. */
    private void assertNoReaderWasReopened(final List<Node> before) {
        final List<Node> after = nodes();
        assertEquals("the cluster must still have the same nodes", before.size(), after.size());
        for (int i = 0; i < before.size(); i++) {
            assertEquals(
                "node " + before.get(i).name() + " reopened a reader: " + before.get(i) + " -> " + after.get(i),
                before.get(i).routedOpens(),
                after.get(i).routedOpens()
            );
        }
    }

    /**
     * A 3-shard, 0-replica {@code on_disk} 32x faiss index, so that a balanced 3-node cluster puts one
     * shard and so one routed {@code .vec} on each node. {@code index.use_compound_file} false because a
     * compound segment has no {@code .vec} of its own and is declined by condition 3.
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
            .put("number_of_shards", 3)
            .put("number_of_replicas", 0)
            .put("index.refresh_interval", "-1")
            .put("index.use_compound_file", false)
            .build();
        createKnnIndex(INDEX, settings, mapping.toString());
    }

    @SneakyThrows
    private void queryAll(final float[][] queries) {
        for (final float[] query : queries) {
            final Response response = searchKNNIndex(INDEX, new KNNQueryBuilder(FIELD, query, K), K);
            EntityUtils.toString(response.getEntity());
        }
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

    /** One snapshot per node, in the order the stats API returns them, which is stable within a run. */
    @SneakyThrows
    private List<Node> nodes() {
        final Response response = getKnnStats(Collections.emptyList(), Collections.emptyList());
        final List<Map<String, Object>> raw = parseNodeStatsResponse(EntityUtils.toString(response.getEntity()));
        final List<Node> nodes = new ArrayList<>(raw.size());
        for (int i = 0; i < raw.size(); i++) {
            final Map<String, Object> node = raw.get(i);
            nodes.add(
                new Node(
                    "n" + i,
                    value(node, "dispatch.routed_opens"),
                    value(node, "cache.slabs"),
                    value(node, "cache.bytes"),
                    value(node, "cache.configured_max_bytes")
                )
            );
        }
        return nodes;
    }

    private static long value(final Map<String, Object> node, final String path) {
        final Object found = XContentMapValues.extractValue(StatNames.DIRECT_IO_STATS.getName() + "." + path, node);
        return found instanceof Number number ? number.longValue() : 0L;
    }

    /** One node's Direct I/O state, with a {@code toString} every assertion message carries. */
    private record Node(String name, long routedOpens, long slabs, long bytes, long configuredMaxBytes) {
        @Override
        public String toString() {
            return String.format(
                Locale.ROOT,
                "%s[routed_opens=%d slabs=%d bytes=%d configured_max_bytes=%d]",
                name,
                routedOpens,
                slabs,
                bytes,
                configuredMaxBytes
            );
        }
    }
}

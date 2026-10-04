/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.stats;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.store.DirectIOVectorIndexInput;

import java.util.Map;
import java.util.Set;

/**
 * Unit tests for {@link KNNStats}: the remote vector index build stats map which surfaces the
 * {@link KNNRemoteIndexBuildValue} counters (including the merge-abort and terminal-exception counters) through
 * the {@code _stats} API, and the Direct I/O read path's node-wide counters added in phase 14 S4.
 */
public class KNNStatsTests extends KNNTestCase {

    @Override
    public void tearDown() throws Exception {
        super.tearDown();
        for (KNNRemoteIndexBuildValue value : KNNRemoteIndexBuildValue.values()) {
            value.reset();
        }
    }

    /**
     * The remote vector index build stats expose client, repository, and build sub-maps, and the client sub-map
     * surfaces every client-side counter including the benign-termination counters added for early merge abort.
     */
    @SuppressWarnings("unchecked")
    public void testRemoteIndexBuildStatsMapContainsAllCounters() {
        KNNStats knnStats = new KNNStats();

        Object remoteStatsValue = knnStats.getStats().get(StatNames.REMOTE_VECTOR_INDEX_BUILD_STATS.getName()).getValue();
        assertTrue(remoteStatsValue instanceof Map);
        Map<String, Map<String, Object>> remoteStats = (Map<String, Map<String, Object>>) remoteStatsValue;

        // The three sub-maps must all be present.
        assertTrue(remoteStats.containsKey(StatNames.CLIENT_STATS.getName()));
        assertTrue(remoteStats.containsKey(StatNames.REPOSITORY_STATS.getName()));
        assertTrue(remoteStats.containsKey(StatNames.BUILD_STATS.getName()));

        Map<String, Object> clientStats = remoteStats.get(StatNames.CLIENT_STATS.getName());
        assertTrue(clientStats.containsKey(KNNRemoteIndexBuildValue.INDEX_BUILD_SUCCESS_COUNT.getName()));
        assertTrue(clientStats.containsKey(KNNRemoteIndexBuildValue.INDEX_BUILD_FAILURE_COUNT.getName()));
        // Benign-termination counters added for the early merge-abort feature.
        assertTrue(clientStats.containsKey(KNNRemoteIndexBuildValue.INDEX_BUILD_MERGE_ABORT_EXCEPTION.getName()));
        assertTrue(clientStats.containsKey(KNNRemoteIndexBuildValue.INDEX_BUILD_TERMINAL_EXCEPTION.getName()));
    }

    /**
     * The stats map is backed by a supplier, so incrementing the underlying counters is reflected the next time the
     * stat value is read.
     */
    @SuppressWarnings("unchecked")
    public void testRemoteIndexBuildStatsReflectCounterValues() {
        KNNRemoteIndexBuildValue.INDEX_BUILD_MERGE_ABORT_EXCEPTION.increment();
        KNNRemoteIndexBuildValue.INDEX_BUILD_TERMINAL_EXCEPTION.increment();
        KNNRemoteIndexBuildValue.INDEX_BUILD_TERMINAL_EXCEPTION.increment();

        KNNStats knnStats = new KNNStats();
        Map<String, Map<String, Object>> remoteStats = (Map<String, Map<String, Object>>) knnStats.getStats()
            .get(StatNames.REMOTE_VECTOR_INDEX_BUILD_STATS.getName())
            .getValue();
        Map<String, Object> clientStats = remoteStats.get(StatNames.CLIENT_STATS.getName());

        assertEquals(1L, (long) (Long) clientStats.get(KNNRemoteIndexBuildValue.INDEX_BUILD_MERGE_ABORT_EXCEPTION.getName()));
        assertEquals(2L, (long) (Long) clientStats.get(KNNRemoteIndexBuildValue.INDEX_BUILD_TERMINAL_EXCEPTION.getName()));
        assertEquals(0L, (long) (Long) clientStats.get(KNNRemoteIndexBuildValue.INDEX_BUILD_FAILURE_COUNT.getName()));
    }

    /**
     * The Direct I/O counters an operator needs are reachable at {@code _plugins/_knn/stats}, which is the
     * answer to S4's "state exactly where". The names are asserted literally because they are the API: a
     * node bench reading {@code direct_io_stats.dispatch.routed_opens} breaks silently if one is renamed.
     *
     * <p>The counters this pins by name are the four S4 asked for — {@code staged_reads},
     * {@code stage_declined}, {@code routed_opens}, {@code declined_opens} — plus the four S2 decline
     * reasons, which are the ones that make a zero {@code routed_opens} diagnosable rather than mysterious.
     */
    @SuppressWarnings("unchecked")
    public void testDirectIOStatsAreReachableAndNamedStably() {
        DirectIOVectorIndexInput.STATS.reset();
        final KNNStats knnStats = new KNNStats();

        final KNNStat<?> stat = knnStats.getStats().get(StatNames.DIRECT_IO_STATS.getName());
        assertNotNull("direct_io_stats must be registered", stat);
        assertFalse("these are one node's reads, so the stat is node level", stat.isClusterLevel());
        assertTrue(
            "and it must therefore appear in the node stats an operator gets",
            knnStats.getNodeStats().containsKey(StatNames.DIRECT_IO_STATS.getName())
        );

        final Map<String, Map<String, Object>> directIO = (Map<String, Map<String, Object>>) stat.getValue();
        assertEquals(Set.of("dispatch", "reads"), directIO.keySet());

        assertEquals(
            Set.of("routed_opens", "declined_opens", "declined_chain", "declined_missing", "declined_too_small", "declined_open_failed"),
            directIO.get("dispatch").keySet()
        );
        for (final String required : new String[] {
            "staged_reads",
            "stage_declined",
            "blocking_reads",
            "readahead_reads",
            "served_bytes" }) {
            assertTrue(
                "reads." + required + " must be exposed, saw " + directIO.get("reads").keySet(),
                directIO.get("reads").containsKey(required)
            );
        }
        // On a node with the feature off -- the shipped default -- every counter reads zero, which is what
        // makes "off" distinguishable from "on and declining".
        directIO.get("dispatch").forEach((name, value) -> assertEquals(name + " must start at zero", 0L, (long) (Long) value));
    }

}

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin.stats;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.KNNSettings;
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
     * {@code stage_declined}, {@code routed_opens}, {@code declined_opens} — plus the {@code declined} map's
     * six reason names, which are the ones that make a zero {@code routed_opens} diagnosable rather than
     * mysterious: a compound segment is both the commonest decline and the one that means nothing is wrong,
     * and it must not be confusable with a chain the feature cannot serve or with a chain whose bytes are not
     * the file's.
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
        assertEquals(Set.of("dispatch", "reads", "memory", "cache"), directIO.keySet());
        for (final String required : new String[] {
            "live_bytes",
            "peak_bytes",
            "declined_budget",
            "budget_bytes",
            "wall_bytes",
            // O2's four, which are 16-I's I7 names: the wide-group buffers and their own ceiling.
            "wide_buffer_bytes",
            "wide_buffer_peak_bytes",
            "wide_buffer_budget_bytes",
            "declined_wide_budget" }) {
            assertTrue(
                "memory." + required + " must be exposed, saw " + directIO.get("memory").keySet(),
                directIO.get("memory").containsKey(required)
            );
        }
        assertEquals(
            Set.of(
                "hits",
                "misses",
                "evictions",
                "bytes",
                "entries",
                "max_bytes",
                // W3: the target the operator published, beside the one the store rounded it down to.
                "configured_max_bytes",
                "slab_memory",
                "slabs",
                "slab_bytes",
                "reclaimed_slabs",
                "readmissions",
                "span_fills",
                // O3: the entries a closed .vec gave back, and the files they came from.
                "closed_files",
                "closed_file_entries",
                // O4: the filter-then-read race's re-admissions, also inside readmissions.
                "race_readmissions",
                // Q (C6): the segmented store. segments is the count in force; the contended counts and the
                // read wait nanos are what say whether the segmentation is doing its job; the entry spread
                // is the skew the hit-rate cost is priced against; slab_transfers is the broker's lending;
                // the two close-hold keys are C4's bound; segment_count_changes is a re-warm an operator
                // caused.
                "segments",
                "contended_read_acquisitions",
                "contended_write_acquisitions",
                "read_lock_wait_nanos",
                "max_segment_entries",
                "min_segment_entries",
                "slab_transfers",
                "longest_close_hold_nanos",
                "longest_close_hold_entries",
                "segment_count_changes"
            ),
            directIO.get("cache").keySet()
        );
        // The cache is off by default, so every one of its counters must read zero on a node that has not
        // set knn.direct_io.rescore.cache.max_bytes -- which is what makes "off" distinguishable from
        // "on and missing". slab_memory is not a counter: it names the off-heap backend slabs come from, and
        // it reads the same whether the cache is on or off, which is why it is asserted separately.
        directIO.get("cache").forEach((name, value) -> {
            if ("slab_memory".equals(name)) {
                assertTrue("cache.slab_memory must name a backend, saw " + value, Set.of("ffm", "direct_buffer").contains(value));
            } else if ("segments".equals(name)) {
                // Not a counter either: it is the segment count in force, and with the cache off that is
                // one, which is the unsegmented store exactly (Q, C1b).
                assertEquals("cache.segments is one with the cache off", 1, (int) (Integer) value);
            } else {
                assertEquals("cache." + name + " must start at zero", 0L, (long) (Long) value);
            }
        });

        assertEquals(Set.of("routed_opens", "declined_opens", "declined"), directIO.get("dispatch").keySet());
        // F-08: the six reasons are one map keyed by the reason, each entry carrying its count and the
        // top of the chain the last decline under it came through. The reason names are the API now, so they
        // are asserted literally for the same reason the key names are, and in the same lower-case shape.
        final Map<String, Object> declined = (Map<String, Object>) directIO.get("dispatch").get("declined");
        assertEquals(Set.of("compound", "chain", "missing", "too_small", "open_failed", "content"), declined.keySet());
        declined.forEach((reason, entry) -> {
            final Map<String, Object> fields = (Map<String, Object>) entry;
            assertEquals("declined." + reason + " must carry a count and a wrapper", Set.of("count", "outermost"), fields.keySet());
            assertEquals("declined." + reason + ".count must start at zero", 0L, (long) (Long) fields.get("count"));
            assertEquals("declined." + reason + ".outermost is empty until something declines", "", fields.get("outermost"));
        });
        for (final String required : new String[] {
            "staged_reads",
            "stage_declined",
            "blocking_reads",
            "readahead_reads",
            "served_bytes",
            "queue_peak",
            "pool_rebuilds",
            // O1's four, which are 16-I's I6 names: the pair that shows a rollback reaching open segments,
            // and the route changes themselves.
            "fallback_reads",
            "fallback_bytes",
            "switched_to_mmap",
            "switched_to_direct",
            // O5's: staged miss reads widened to admit their neighbours.
            "read_around_reads",
            // W3: the thread count the operator published, beside the live pool's maximum.
            "configured_prefetch_threads",
            "active_prefetch_threads" }) {
            assertTrue(
                "reads." + required + " must be exposed, saw " + directIO.get("reads").keySet(),
                directIO.get("reads").containsKey(required)
            );
        }
        // On a node with the feature off -- the shipped default -- every counter reads zero, which is what
        // makes "off" distinguishable from "on and declining". The declined map's own zeros are asserted
        // above, entry by entry.
        directIO.get("dispatch").forEach((name, value) -> {
            if ("declined".equals(name) == false) {
                assertEquals(name + " must start at zero", 0L, (long) (Long) value);
            }
        });
    }

    /**
     * D3b. {@code memory.wall_bytes} reports the JVM-wide direct-memory wall the budget was derived from, so
     * that "is this ceiling actually below the wall" is one GET rather than an SSH to read the JVM's command
     * line. It is a real need: the bench node pins {@code -XX:MaxDirectMemorySize=2g} against a 24g heap,
     * where the pre-D3b default sat three times above the wall and nothing in the API said so.
     */
    @SuppressWarnings("unchecked")
    public void testTheDirectMemoryWallIsReportedBesideTheBudget() {
        final Map<String, Map<String, Object>> directIO = (Map<String, Map<String, Object>>) new KNNStats().getStats()
            .get(StatNames.DIRECT_IO_STATS.getName())
            .getValue();
        final Map<String, Object> memory = directIO.get("memory");

        final long wall = (Long) memory.get("wall_bytes");
        assertEquals("it is the wall the default was derived from", KNNSettings.KNN_DIRECT_IO_DIRECT_MEMORY_WALL_BYTES, wall);
        assertTrue("and a positive number on any JVM this runs on", wall > 0);
        assertEquals("the shipped default is exactly half of it", wall / 2, (long) (Long) memory.get("budget_bytes"));
    }
}

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn;

import com.google.common.collect.Multimap;
import lombok.SneakyThrows;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.support.XContentMapValues;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.plugin.stats.StatNames;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.TreeSet;

/**
 * What W4's two nested measurements share: the index, the query shapes, the three routes they are each run
 * under, and the {@code direct_io_stats} snapshot taken around a query.
 *
 * <p>The two sides of the comparison are one index and a setting flip, not two indices. Both sides then score
 * the same segment with the same scorer and the only difference is the {@code IndexInput}, so ids and scores
 * must be identical rather than merely close, and no control arm is needed to price a graph difference.
 *
 * <p>A subclass asserts ({@code DirectIONestedIdentityIT}) or reports ({@code DirectIONestedCountersIT}); the
 * cells and the routes are declared once, here, so the two cannot drift apart.
 */
public abstract class DirectIONestedTestCase extends KNNRestTestCase {

    /** Direct I/O on or off, and how big its store may be, as an operator would set them. */
    public enum Arm {
        MMAP("mmap", false, "0b"),
        DIO_0B("dio-0b", true, "0b"),
        DIO_CACHE("dio-cache", true, "64mb");

        private final String label;
        private final boolean enabled;
        private final String cacheMaxBytes;

        Arm(final String label, final boolean enabled, final String cacheMaxBytes) {
            this.label = label;
            this.enabled = enabled;
            this.cacheMaxBytes = cacheMaxBytes;
        }

        /** The arm's name in a log line and a table row. */
        public String label() {
            return label;
        }

        public boolean routes() {
            return enabled;
        }
    }

    /**
     * One query shape of the matrix.
     *
     * @param name            the cell's name, which every log line and table row is keyed by
     * @param expandNestedDocs whether the query asks for the matching children as inner hits
     * @param filter          the k-NN query's filter clause as raw json, or null for an unfiltered search
     * @param rescore         the rescore clause as raw json, or null to take the field's default
     */
    public record Cell(String name, boolean expandNestedDocs, String filter, String rescore) {
    }

    /**
     * What one cell answered, every query of the cell in order: the parent ids, the parent scores, and, for a
     * cell that asked for the children, one line per parent naming the child offsets it came back with.
     *
     * <p>The children are compared and not only the parents because a parent's score is its best child's: a
     * child that came back wrong, or did not come back, moves no parent score and would otherwise not show.
     *
     * @param ids      one parent id per hit
     * @param scores   one parent score per hit, in the same order
     * @param children one {@code parent=offsets} line per hit for an expanding cell, empty otherwise
     */
    public record Answers(List<String> ids, List<Double> scores, List<String> children) {
        /** How many hits this cell returned, which is the number of pairs a comparison walks. */
        public int pairs() {
            return ids.size();
        }
    }

    protected static final int K = 10;
    protected static final int QUERIES = 8;

    /**
     * The arms in the order they must run: the delegate first, so it is the reference the two Direct I/O arms
     * are compared against, and the sized store last, so it starts from a store the 0 B arm has released.
     */
    protected static final List<Arm> ARMS = List.of(Arm.MMAP, Arm.DIO_0B, Arm.DIO_CACHE);

    private static final String NESTED_VECTOR_PATH = NestedDirectIOFixture.NESTED_FIELD + "." + NestedDirectIOFixture.VECTOR_FIELD;
    private static final String NESTED_CHILD_FILTER_PATH = NestedDirectIOFixture.NESTED_FIELD
        + "."
        + NestedDirectIOFixture.CHILD_FILTER_FIELD;

    private final List<String> created = new ArrayList<>();

    @Override
    public void tearDown() throws Exception {
        // Explicit values in both directions and never null: removing a persisted key wedges
        // _cluster/settings. Each key is restored on its own, because a failure restoring the first must not
        // leave the next at this class's value for whatever runs next on the same cluster -- min_file_size
        // left at 0b would route a .vec that another test needs declined.
        try {
            restore(KNNSettings.KNN_DIRECT_IO_RESCORE_ENABLED, false);
            restore(KNNSettings.KNN_DIRECT_IO_RESCORE_CACHE_MAX_BYTES, "0b");
            restore(KNNSettings.KNN_DIRECT_IO_MIN_FILE_SIZE, "10mb");
            for (final String index : created) {
                try {
                    deleteKNNIndex(index);
                } catch (Exception e) {
                    // The base class wipes the cluster's indices too, so a 404 here means it got there first.
                    logger.info("could not delete {}; the cluster wipe had already removed it", index, e);
                }
            }
        } finally {
            created.clear();
            super.tearDown();
        }
    }

    /** One setting back to its shipped value, loudly but without stopping the rest of the restore. */
    private void restore(final String key, final Object value) {
        try {
            updateClusterSettings(key, value);
        } catch (Exception e) {
            logger.error("could not restore {} to {}; the next test inherits this class's value", key, value, e);
        }
    }

    /**
     * Puts the cluster on {@code arm}. {@code min_file_size} is 0 B throughout, because the generated
     * fixture's {@code .vec} is a few megabytes and the shipped 10 MB default would decline it.
     */
    protected void applyArm(final Arm arm) throws Exception {
        updateClusterSettings(KNNSettings.KNN_DIRECT_IO_MIN_FILE_SIZE, "0b");
        updateClusterSettings(KNNSettings.KNN_DIRECT_IO_RESCORE_CACHE_MAX_BYTES, arm.cacheMaxBytes);
        updateClusterSettings(KNNSettings.KNN_DIRECT_IO_RESCORE_ENABLED, arm.routes());
    }

    /** One created index: its name and the documents it holds, whose child counts are every ratio's denominator. */
    public record NestedIndex(String name, NestedDirectIOFixture.Documents documents) {
    }

    /**
     * Creates, fills and force merges one index of the fixture. One segment and {@code refresh_interval} -1,
     * so that nothing reopens a reader between the arms and {@code dispatch.routed_opens} stays a number the
     * arms can be compared on. {@code use_compound_file} false because a compound segment has no
     * {@code .vec} of its own and is declined.
     */
    @SneakyThrows
    protected NestedIndex createNestedIndex(final NestedDirectIOFixture.DataType dataType, final NestedDirectIOFixture.Shape shape) {
        final String index = String.format(
            Locale.ROOT,
            "dio-nested-%s-%s",
            dataType.name().toLowerCase(Locale.ROOT),
            shape.name().toLowerCase(Locale.ROOT)
        );
        final Settings settings = Settings.builder()
            .put("index.knn", true)
            .put("number_of_shards", 1)
            .put("number_of_replicas", 0)
            .put("index.refresh_interval", "-1")
            .put("index.use_compound_file", false)
            // Pinned at its own default, so that a change to the default cannot quietly turn these queries
            // into exact search and change what the measurement is of.
            .put(KNNSettings.INDEX_KNN_ADVANCED_APPROXIMATE_THRESHOLD, 0)
            .build();
        createKnnIndex(index, settings, NestedDirectIOFixture.mapping(dataType));
        created.add(index);
        final NestedDirectIOFixture.Documents documents = NestedDirectIOFixture.documents(dataType, shape);
        bulkIndex(index, documents.documents());
        forceMergeKnnIndex(index, 1);
        return new NestedIndex(index, documents);
    }

    /**
     * The fixture's documents in one bulk request per thousand, under ids 0, 1, 2 and so on. Each request
     * refreshes, so the force merge that follows sees every document.
     */
    @SneakyThrows
    private void bulkIndex(final String index, final List<String> documents) {
        final int batch = 1_000;
        for (int from = 0; from < documents.size(); from += batch) {
            final StringBuilder body = new StringBuilder();
            for (int i = from; i < Math.min(from + batch, documents.size()); i++) {
                body.append(String.format(Locale.ROOT, "{\"index\":{\"_index\":\"%s\",\"_id\":\"%d\"}}%n", index, i))
                    .append(documents.get(i))
                    .append(System.lineSeparator());
            }
            final Request request = new Request("POST", "/_bulk");
            request.addParameter("refresh", "true");
            request.setJsonEntity(body.toString());
            final Response response = client().performRequest(request);
            assertEquals(RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));
            final String responseBody = EntityUtils.toString(response.getEntity());
            assertFalse("a bulk request failed: " + responseBody, responseBody.contains("\"errors\":true"));
        }
    }

    /**
     * The cells of the matrix for {@code dataType}: the shapes every type has, plus the rescore cells, which
     * only a field with a rescore pass has ({@code byte} has no {@code on_disk} mode).
     */
    protected List<Cell> cells(final NestedDirectIOFixture.DataType dataType) {
        final String parentFilter = String.format(Locale.ROOT, "{\"term\":{\"%s\":true}}", NestedDirectIOFixture.PARENT_FILTER_FIELD);
        final String rareFilter = String.format(Locale.ROOT, "{\"term\":{\"%s\":true}}", NestedDirectIOFixture.PARENT_RARE_FIELD);
        final String childFilter = String.format(
            Locale.ROOT,
            "{\"nested\":{\"path\":\"%s\",\"query\":{\"term\":{\"%s\":true}}}}",
            NestedDirectIOFixture.NESTED_FIELD,
            NESTED_CHILD_FILTER_PATH
        );
        final List<Cell> cells = new ArrayList<>(
            List.of(
                new Cell("plain", false, null, null),
                new Cell("expand-nested", true, null, null),
                new Cell("parent-filter", true, parentFilter, null),
                new Cell("child-filter", true, childFilter, null),
                // One parent in a hundred passes, which is far below the threshold the approximate path needs,
                // so this cell is exact search over the surviving children.
                new Cell("exact-search", false, rareFilter, null)
            )
        );
        if (dataType.hasRescore()) {
            cells.add(new Cell("rescore-off", false, null, "false"));
            for (final String factor : List.of("2.0", "3.0", "5.0")) {
                cells.add(new Cell("oversample-" + factor, false, null, "{\"oversample_factor\":" + factor + "}"));
            }
        }
        return cells;
    }

    /**
     * Every query of {@code cell} in order, as the parents, their scores and, where the cell asked for them,
     * the children each parent came back with. A cell that returned no hit at all is a cell that proves
     * nothing, so it fails here rather than being reported as a row of zeroes.
     */
    protected Answers run(final String index, final Cell cell, final List<Object[]> queries) throws Exception {
        final List<String> ids = new ArrayList<>();
        final List<Double> scores = new ArrayList<>();
        final List<String> children = new ArrayList<>();
        for (final Object[] query : queries) {
            final String body = search(index, cell, query);
            final List<String> hits = parseIds(body);
            assertFalse(cell.name() + " returned no hit, so nothing it reports means anything", hits.isEmpty());
            ids.addAll(hits);
            scores.addAll(parseScores(body));
            if (cell.expandNestedDocs()) {
                final Multimap<String, Integer> inner = parseInnerHits(body, NestedDirectIOFixture.NESTED_FIELD);
                for (final String parent : hits) {
                    children.add(parent + "=" + new TreeSet<>(inner.get(parent)));
                }
            }
        }
        return new Answers(ids, scores, children);
    }

    /** One query of {@code cell}, as the response body. */
    protected String search(final String index, final Cell cell, final Object[] query) throws Exception {
        final String innerHits = cell.expandNestedDocs() ? ",\"inner_hits\":{\"size\":100,\"_source\":false}" : "";
        final String expand = cell.expandNestedDocs() ? ",\"expand_nested_docs\":true" : "";
        final String filter = cell.filter() == null ? "" : String.format(Locale.ROOT, ",\"filter\":%s", cell.filter());
        final String rescore = cell.rescore() == null ? "" : String.format(Locale.ROOT, ",\"rescore\":%s", cell.rescore());
        final String body = String.format(
            Locale.ROOT,
            "{\"_source\":false,\"size\":%d,\"query\":{\"nested\":{\"path\":\"%s\","
                + "\"query\":{\"knn\":{\"%s\":{\"vector\":%s,\"k\":%d%s%s%s}}}%s}}}",
            K,
            NestedDirectIOFixture.NESTED_FIELD,
            NESTED_VECTOR_PATH,
            Arrays.toString(query),
            K,
            expand,
            filter,
            rescore,
            innerHits
        );
        return post(index, body);
    }

    /** A search body straight through, for the one cell whose shape is not a {@link Cell}. */
    protected String post(final String index, final String body) throws Exception {
        final Request request = new Request("POST", String.format(Locale.ROOT, "/%s/_search", index));
        request.setJsonEntity(body);
        final Response response = client().performRequest(request);
        assertEquals(RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));
        return EntityUtils.toString(response.getEntity());
    }

    /**
     * The {@code direct_io_stats} an operator reads, summed over the cluster's nodes. Node-wide and cumulative,
     * so every number taken from it is a difference between two snapshots and never an absolute.
     *
     * <p><b>Reading this is not free of effect.</b> {@code memory.live_bytes} is served by
     * {@code DirectIOVectorIndexInput#reclaim()}, which drains the reference queue and gives the charges back
     * to the direct-memory budget on the calling thread. So a caller that snapshots often relieves budget
     * pressure that production would relieve only at the next allocation, and the decline counters a frequent
     * caller reports are a lower bound.
     */
    @SneakyThrows
    protected Counters counters() {
        final Response response = getKnnStats(Collections.emptyList(), Collections.emptyList());
        final List<Map<String, Object>> nodes = parseNodeStatsResponse(EntityUtils.toString(response.getEntity()));
        final Counters total = new Counters();
        for (final Map<String, Object> node : nodes) {
            total.routedOpens += value(node, "dispatch.routed_opens");
            total.declinedOpens += value(node, "dispatch.declined_opens");
            total.blockingReads += value(node, "reads.blocking_reads");
            total.blockingBytes += value(node, "reads.blocking_bytes");
            total.stagedReads += value(node, "reads.staged_reads");
            total.stagedBytes += value(node, "reads.staged_bytes");
            total.servedBytes += value(node, "reads.served_bytes");
            total.fallbackReads += value(node, "reads.fallback_reads");
            total.prefetchCalls += value(node, "reads.prefetch_calls");
            total.readaheadArmed += value(node, "reads.readahead_armed");
            total.readaheadReads += value(node, "reads.readahead_reads");
            total.readaheadHits += value(node, "reads.readahead_hits");
            total.readaheadBytes += value(node, "reads.readahead_bytes");
            // Why a read did not take the cheap path: the ring refused to arm, the pool refused the stage,
            // or an allocation was refused by one of the two memory budgets. Without these four a read that
            // fell back to a blocking device read looks unexplained.
            total.readaheadDeclined += value(node, "reads.readahead_declined");
            total.stageDeclined += value(node, "reads.stage_declined");
            total.switchedToMmap += value(node, "reads.switched_to_mmap");
            total.switchedToDirect += value(node, "reads.switched_to_direct");
            total.declinedBudget += value(node, "memory.declined_budget");
            total.declinedWideBudget += value(node, "memory.declined_wide_budget");
            total.wideBufferBytes += value(node, "memory.wide_buffer_bytes");
            total.liveBytes += value(node, "memory.live_bytes");
            total.cacheHits += value(node, "cache.hits");
            total.cacheMisses += value(node, "cache.misses");
            total.cacheEntries += value(node, "cache.entries");
            total.spanFills += value(node, "cache.span_fills");
        }
        return total;
    }

    private static long value(final Map<String, Object> node, final String path) {
        final Object found = XContentMapValues.extractValue(StatNames.DIRECT_IO_STATS.getName() + "." + path, node);
        return found instanceof Number number ? number.longValue() : 0L;
    }

    /** One snapshot of the counters, and the difference between two of them. */
    public static final class Counters {
        public long routedOpens;
        public long declinedOpens;
        public long blockingReads;
        public long blockingBytes;
        public long stagedReads;
        public long stagedBytes;
        public long servedBytes;
        public long fallbackReads;
        public long prefetchCalls;
        public long readaheadArmed;
        public long readaheadReads;
        public long readaheadHits;
        public long readaheadBytes;
        public long readaheadDeclined;
        public long stageDeclined;
        public long switchedToMmap;
        public long switchedToDirect;
        public long declinedBudget;
        public long declinedWideBudget;
        /** A level and not a count: the wide-group buffers alive at the snapshot. */
        public long wideBufferBytes;
        /** A level and not a count: the direct memory alive at the snapshot. */
        public long liveBytes;
        public long cacheHits;
        public long cacheMisses;
        /** A level and not a count: the entries the store holds at the snapshot. */
        public long cacheEntries;
        public long spanFills;

        /** Reads that went to the device, staged on the pool or taken blocking on the calling thread. */
        public long deviceReads() {
            return stagedReads + blockingReads;
        }

        /** {@code this} minus {@code before}, so a caller holds the cost of what happened in between. */
        public Counters since(final Counters before) {
            final Counters delta = new Counters();
            delta.routedOpens = routedOpens - before.routedOpens;
            delta.declinedOpens = declinedOpens - before.declinedOpens;
            delta.blockingReads = blockingReads - before.blockingReads;
            delta.blockingBytes = blockingBytes - before.blockingBytes;
            delta.stagedReads = stagedReads - before.stagedReads;
            delta.stagedBytes = stagedBytes - before.stagedBytes;
            delta.servedBytes = servedBytes - before.servedBytes;
            delta.fallbackReads = fallbackReads - before.fallbackReads;
            delta.prefetchCalls = prefetchCalls - before.prefetchCalls;
            delta.readaheadArmed = readaheadArmed - before.readaheadArmed;
            delta.readaheadReads = readaheadReads - before.readaheadReads;
            delta.readaheadHits = readaheadHits - before.readaheadHits;
            delta.readaheadBytes = readaheadBytes - before.readaheadBytes;
            delta.readaheadDeclined = readaheadDeclined - before.readaheadDeclined;
            delta.stageDeclined = stageDeclined - before.stageDeclined;
            delta.switchedToMmap = switchedToMmap - before.switchedToMmap;
            delta.switchedToDirect = switchedToDirect - before.switchedToDirect;
            delta.declinedBudget = declinedBudget - before.declinedBudget;
            delta.declinedWideBudget = declinedWideBudget - before.declinedWideBudget;
            // Levels, so the difference is a change and the snapshot's own value is what the arm ended at.
            delta.wideBufferBytes = wideBufferBytes - before.wideBufferBytes;
            delta.liveBytes = liveBytes - before.liveBytes;
            delta.cacheHits = cacheHits - before.cacheHits;
            delta.cacheMisses = cacheMisses - before.cacheMisses;
            // Entries is a level, so its difference is a change and may be negative: a release between the
            // two snapshots lowers it. Span fills is a count like the others.
            delta.cacheEntries = cacheEntries - before.cacheEntries;
            delta.spanFills = spanFills - before.spanFills;
            return delta;
        }

        @Override
        public String toString() {
            return String.format(
                Locale.ROOT,
                "routed_opens=%d declined_opens=%d blocking_reads=%d blocking_bytes=%d staged_reads=%d "
                    + "staged_bytes=%d served_bytes=%d fallback_reads=%d prefetch_calls=%d readahead_armed=%d "
                    + "readahead_reads=%d readahead_hits=%d readahead_bytes=%d readahead_declined=%d stage_declined=%d "
                    + "switched_to_mmap=%d switched_to_direct=%d declined_budget=%d declined_wide_budget=%d "
                    + "wide_buffer_bytes=%d live_bytes=%d cache_hits=%d cache_misses=%d cache_entries=%d span_fills=%d",
                routedOpens,
                declinedOpens,
                blockingReads,
                blockingBytes,
                stagedReads,
                stagedBytes,
                servedBytes,
                fallbackReads,
                prefetchCalls,
                readaheadArmed,
                readaheadReads,
                readaheadHits,
                readaheadBytes,
                readaheadDeclined,
                stageDeclined,
                switchedToMmap,
                switchedToDirect,
                declinedBudget,
                declinedWideBudget,
                wideBufferBytes,
                liveBytes,
                cacheHits,
                cacheMisses,
                cacheEntries,
                spanFills
            );
        }
    }
}

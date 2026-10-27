/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn;

import org.opensearch.common.xcontent.support.XContentMapValues;
import org.opensearch.knn.plugin.stats.StatNames;

import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * One snapshot of the {@code direct_io_stats} an operator reads, summed over the cluster's nodes, and the
 * difference between two of them.
 *
 * <p>Node-wide and cumulative, so every number taken from it is a difference between two snapshots and never an
 * absolute. A few of the keys are levels rather than counts, and those are marked: their difference is a change
 * and may be negative.
 *
 * <p><b>Taking a snapshot is not free of effect.</b> {@code memory.live_bytes} is served by
 * {@code DirectIOVectorIndexInput#reclaim()}, which drains the reference queue and gives the charges back to the
 * direct-memory budget on the calling thread. So a caller that snapshots often relieves budget pressure that
 * production would relieve only at the next allocation, and the decline counters a frequent caller reports are a
 * lower bound.
 */
public final class DirectIOCounters {

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

    /**
     * The counters as the cluster reports them now, summed over {@code nodeStats}, which is what
     * {@code KNNRestTestCase#parseNodeStatsResponse} returns for a {@code _plugins/_knn/stats} response.
     */
    public static DirectIOCounters of(final List<Map<String, Object>> nodeStats) {
        final DirectIOCounters total = new DirectIOCounters();
        for (final Map<String, Object> node : nodeStats) {
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
            // Why a read did not take the cheap path: the ring refused to arm, the pool refused the stage, or
            // an allocation was refused by one of the two memory budgets. Without these four a read that fell
            // back to a blocking device read looks unexplained.
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

    /** Reads that went to the device, staged on the pool or taken blocking on the calling thread. */
    public long deviceReads() {
        return stagedReads + blockingReads;
    }

    /** {@code this} minus {@code before}, so a caller holds the cost of what happened in between. */
    public DirectIOCounters since(final DirectIOCounters before) {
        final DirectIOCounters delta = new DirectIOCounters();
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
        // Entries is a level, so its difference is a change and may be negative: a release between the two
        // snapshots lowers it. Span fills is a count like the others.
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

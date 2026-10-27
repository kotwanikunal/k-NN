/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import lombok.SneakyThrows;
import org.opensearch.knn.DirectIOArm;
import org.opensearch.knn.DirectIOCounters;
import org.opensearch.knn.DirectIONestedTestCase;
import org.opensearch.knn.NestedDirectIOFixture;
import org.opensearch.knn.NestedDirectIOFixture.DataType;
import org.opensearch.knn.NestedDirectIOFixture.Shape;

import java.util.List;
import java.util.Locale;

/**
 * What a nested query costs on each route: the reads, the bytes and the store's hits per query, for every query
 * shape of the nested matrix and both child shapes.
 *
 * <p>This class reports rather than asserts a bound, because the bound is what the measurement sets. What the
 * numbers answer is whether a nested query's blocking reads are worth a fix: the children of one parent have
 * contiguous ordinals, so a group could be coalesced into one read, and whether that is worth building depends
 * on how much of the cost is child vectors rather than the sparse field's doc-to-ord metadata.
 *
 * <p>Four things are asserted, because a run that got them wrong would be reporting noise: the index routed
 * while it was built (a difference across its creation, not a node-lifetime total); no reader was rebuilt while
 * the arms ran, which is {@code routed_opens} and {@code declined_opens} both standing still; every cell
 * returned at least one hit, which {@code run} checks; and each arm read the device if and only if it was on
 * the route.
 *
 * <p>Two things about the numbers themselves, both of which the rows carry so that a reader is not misled.
 * Cells run back to back inside an arm and nothing is released between them, so a cell that exhausted a budget
 * leaves the next cell reading a degraded path: a row whose {@code readahead_declined} or
 * {@code declined_budget} moved is marked {@code carried_pressure=true} and is not that cell's cost alone. And
 * reading the counters drains the reclaim queue ({@link #counters()}), so the declines are a lower bound.
 */
public class DirectIONestedCountersIT extends DirectIONestedTestCase {

    @SneakyThrows
    public void testFloatNestedReadCounters() {
        reportCounters(DataType.FLOAT);
    }

    @SneakyThrows
    public void testHalfFloatNestedReadCounters() {
        reportCounters(DataType.HALF_FLOAT);
    }

    @SneakyThrows
    public void testByteNestedReadCounters() {
        reportCounters(DataType.BYTE);
    }

    private void reportCounters(final DataType dataType) throws Exception {
        for (final Shape shape : Shape.values()) {
            applyArm(DirectIOArm.DIO_CACHE);
            final DirectIOCounters beforeIndex = counters();
            final NestedIndex index = createNestedIndex(dataType, shape);
            final DirectIOCounters opened = counters().since(beforeIndex);
            assertTrue(
                String.format(Locale.ROOT, "%s/%s routed nothing while its index was built: %s", dataType, shape, opened),
                opened.routedOpens > 0
            );

            final List<Object[]> queries = NestedDirectIOFixture.queries(dataType, QUERIES);
            final List<Cell> cells = cells(dataType);
            logger.info(
                "W4-SHAPE {} {} parents={} children={} children_of_matching_parents={} queries={} k={}",
                dataType,
                shape,
                shape.parents(),
                index.documents().children(),
                index.documents().childrenOfMatchingParents(),
                QUERIES,
                K
            );

            final DirectIOCounters beforeArms = counters();
            for (final DirectIOArm arm : ARMS) {
                // The store is released going into the sized arm because the arm before it set max_bytes to
                // 0b, which is why ARMS runs 0b before the sized store and why the sized arm starts cold.
                applyArm(arm);
                long deviceReads = 0;
                for (final Cell cell : cells) {
                    final DirectIOCounters before = counters();
                    run(index.name(), cell, queries);
                    final DirectIOCounters delta = counters().since(before);
                    deviceReads += delta.deviceReads();
                    logger.info(
                        "W4-COUNTERS {} {} {} {} per_query blocking_reads={} staged_reads={} cache_hits={} "
                            + "cache_misses={} fallback_reads={} blocking_bytes={} staged_bytes={} served_bytes={} "
                            + "prefetch_calls={} readahead_armed={} readahead_reads={} readahead_hits={} "
                            + "readahead_bytes={} readahead_declined={} stage_declined={} declined_budget={} "
                            + "declined_wide_budget={} switched_to_mmap_total={} switched_to_direct_total={} "
                            + "wide_buffer_bytes_change={} live_bytes_change={} cache_entries_change={} span_fills={} "
                            + "carried_pressure={}",
                        dataType,
                        shape,
                        arm.label(),
                        cell.name(),
                        perQuery(delta.blockingReads),
                        perQuery(delta.stagedReads),
                        perQuery(delta.cacheHits),
                        perQuery(delta.cacheMisses),
                        perQuery(delta.fallbackReads),
                        perQuery(delta.blockingBytes),
                        perQuery(delta.stagedBytes),
                        perQuery(delta.servedBytes),
                        perQuery(delta.prefetchCalls),
                        perQuery(delta.readaheadArmed),
                        perQuery(delta.readaheadReads),
                        perQuery(delta.readaheadHits),
                        perQuery(delta.readaheadBytes),
                        perQuery(delta.readaheadDeclined),
                        perQuery(delta.stageDeclined),
                        perQuery(delta.declinedBudget),
                        perQuery(delta.declinedWideBudget),
                        delta.switchedToMmap,
                        delta.switchedToDirect,
                        delta.wideBufferBytes,
                        delta.liveBytes,
                        delta.cacheEntries,
                        perQuery(delta.spanFills),
                        delta.readaheadDeclined > 0 || delta.declinedBudget > 0 || delta.declinedWideBudget > 0
                    );
                }
                if (arm.routes()) {
                    assertTrue(arm.label() + " read nothing from the device, so it was not on the route", deviceReads > 0);
                } else {
                    assertEquals(arm.label() + " must not read the device", 0L, deviceReads);
                }
            }

            final DirectIOCounters arms = counters().since(beforeArms);
            assertEquals("no reader may have been rebuilt while the arms ran, " + arms, 0L, arms.routedOpens + arms.declinedOpens);
            deleteKNNIndex(index.name());
        }
    }

    /** A counter's delta over {@link #QUERIES} queries, as the cost of one query. */
    private static String perQuery(final long delta) {
        return String.format(Locale.ROOT, "%.1f", delta / (double) QUERIES);
    }
}

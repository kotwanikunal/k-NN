/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.integ;

import com.carrotsearch.randomizedtesting.annotations.ParametersFactory;
import lombok.SneakyThrows;
import org.opensearch.knn.CompressionTestConfig;
import org.opensearch.knn.DerivedSourceUtils;
import org.opensearch.knn.DirectIOArm;
import org.opensearch.knn.DirectIOCounters;
import org.opensearch.knn.DirectIODerivedSourceTestCase;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Locale;

/**
 * What one reconstructed vector costs on each route: the reads, the bytes and the store's hits per vector a
 * derived-source fetch returned, for a fetch of ten documents, of a hundred, a multi-get and a scroll.
 *
 * <p>One {@code float} field and one vector per document, so a counter divided by the number of documents the
 * cell fetched is the cost of one vector and nothing has to be apportioned.
 *
 * <p>This class reports rather than asserts a bound, because the bound is what the measurement sets. What the
 * numbers answer is where a derived-source fetch's reads come from: if a fetch of a hundred documents costs
 * about a hundred blocking reads then one values object per document is the cost, and if the read-ahead ring
 * already coalesces the scan then it is not.
 *
 * <p>Three things are asserted, because a run that got them wrong would be reporting noise: the index routed
 * while it was built, as a difference across its creation rather than a node-lifetime total; every cell fetched
 * the documents it asked for; and each arm read the device if and only if it was on the route.
 */
public class DirectIODerivedSourceCountersIT extends DirectIODerivedSourceTestCase {

    /** Documents per index. One vector each, so this is also the largest number of vectors a cell can fetch. */
    private static final int DOCS = 500;

    public DirectIODerivedSourceCountersIT(final CompressionTestConfig compressionConfig) {
        super(compressionConfig);
    }

    @ParametersFactory(argumentFormatting = "compression:%1$s")
    public static Collection<Object[]> compressionParameters() {
        return List.<Object[]>of(new Object[] { CompressionTestConfig.X1 });
    }

    /** One cell: a name, how it fetches, and how many vectors that fetch reconstructs. */
    private record Cell(String name, int vectors) {
    }

    @SneakyThrows
    public void testReadCountersPerReconstructedVector() {
        applyArm(DirectIOArm.DIO_CACHE);
        final List<DerivedSourceUtils.IndexConfigContext> contexts = singleFieldContexts("w6counters", DOCS);
        final String derived = contexts.get(0).indexName;

        final DirectIOCounters beforeBuild = counters();
        prepareIndexPair(contexts);
        final DirectIOCounters built = counters().since(beforeBuild);
        assertTrue("the index routed nothing while it was built: " + built, built.routedOpens > 0);
        logger.info("W6-COUNTERS-BUILD docs={} {}", DOCS, built);

        final List<Cell> cells = List.of(
            new Cell("search-10", 10),
            new Cell("search-100", 100),
            new Cell("mget-100-descending", 100),
            new Cell("scroll-500", DOCS)
        );

        for (final DirectIOArm arm : DirectIOArm.ORDER) {
            // The store is released going into the sized arm because the arm before it set max_bytes to 0b,
            // which is why the order runs 0b before the sized store and why the sized arm starts cold.
            applyArm(arm);
            long deviceReads = 0;
            for (final Cell cell : cells) {
                final DirectIOCounters before = counters();
                final int fetched = runCell(derived, cell);
                final DirectIOCounters delta = counters().since(before);
                assertEquals(cell.name() + " fetched the wrong number of documents", cell.vectors(), fetched);
                // Per cell and not only per arm: a cell that read nothing from the device is a row of
                // counters divided by a vector count it never reconstructed.
                if (arm.routes()) {
                    assertTrue(arm.label() + " " + cell.name() + " read nothing from the device", delta.deviceReads() > 0);
                }
                deviceReads += delta.deviceReads();
                logger.info(
                    "W6-COUNTERS {} {} vectors={} per_vector blocking_reads={} staged_reads={} device_reads={} "
                        + "blocking_bytes={} staged_bytes={} served_bytes={} readahead_armed={} readahead_reads={} "
                        + "readahead_hits={} readahead_bytes={} readahead_declined={} cache_hits={} cache_misses={} "
                        + "span_fills={} fallback_reads={} totals[{}]",
                    arm.label(),
                    cell.name(),
                    cell.vectors(),
                    per(delta.blockingReads, cell.vectors()),
                    per(delta.stagedReads, cell.vectors()),
                    per(delta.deviceReads(), cell.vectors()),
                    per(delta.blockingBytes, cell.vectors()),
                    per(delta.stagedBytes, cell.vectors()),
                    per(delta.servedBytes, cell.vectors()),
                    per(delta.readaheadArmed, cell.vectors()),
                    per(delta.readaheadReads, cell.vectors()),
                    per(delta.readaheadHits, cell.vectors()),
                    per(delta.readaheadBytes, cell.vectors()),
                    per(delta.readaheadDeclined, cell.vectors()),
                    per(delta.cacheHits, cell.vectors()),
                    per(delta.cacheMisses, cell.vectors()),
                    per(delta.spanFills, cell.vectors()),
                    per(delta.fallbackReads, cell.vectors()),
                    delta
                );
            }
            if (arm.routes()) {
                assertTrue(arm.label() + " read nothing from the device, so it was not on the route", deviceReads > 0);
            } else {
                assertEquals(arm.label() + " must not read the device", 0L, deviceReads);
            }
        }
    }

    /**
     * Reindex throughput on each route, two runs each into a fresh target. A reindex reads every document's
     * {@code _source} in doc id order, which is the scan the fix is predicted to help most, and it is the one
     * only number here that is a time rather than a count.
     */
    @SneakyThrows
    public void testReindexThroughputOnBothRoutes() {
        applyArm(DirectIOArm.DIO_CACHE);
        final List<DerivedSourceUtils.IndexConfigContext> contexts = singleFieldContexts("w6reindex", DOCS);
        final String derived = contexts.get(0).indexName;

        final DirectIOCounters beforeBuild = counters();
        prepareIndexPair(contexts);
        final DirectIOCounters built = counters().since(beforeBuild);
        assertTrue("the index routed nothing while it was built: " + built, built.routedOpens > 0);

        for (final DirectIOArm arm : List.of(DirectIOArm.MMAP, DirectIOArm.DIO_CACHE)) {
            applyArm(arm);
            for (int run = 1; run <= 2; run++) {
                final String target = String.format(Locale.ROOT, "w6rx-%s-%d", arm.label(), run);
                createTargetIndex(target, contexts.get(0));
                final DirectIOCounters before = counters();
                final long start = System.nanoTime();
                reindex(derived, target);
                final long millis = (System.nanoTime() - start) / 1_000_000L;
                // A reindex that copied nothing answers 200 like one that copied everything, and would report
                // the best throughput in the table, so the target is counted before the number is believed.
                refreshAllIndices();
                assertEquals("the reindex did not copy every document", DOCS, getDocCount(target));
                final DirectIOCounters delta = counters().since(before);
                logger.info(
                    "W6-REINDEX {} run={} docs={} millis={} docs_per_second={} {}",
                    arm.label(),
                    run,
                    DOCS,
                    millis,
                    String.format(Locale.ROOT, "%.1f", DOCS * 1000.0 / Math.max(1L, millis)),
                    delta
                );
            }
        }
    }

    /** One cell's fetch, as the number of documents it returned. */
    private int runCell(final String index, final Cell cell) throws Exception {
        return switch (cell.name()) {
            case "search-10" -> searchFetch(index, 10).size();
            case "search-100" -> searchFetch(index, 100).size();
            case "mget-100-descending" -> multiGet(index, descendingIds(100)).size();
            case "scroll-500" -> scroll(index, 100).size();
            default -> throw new IllegalArgumentException("no such cell: " + cell.name());
        };
    }

    /**
     * The first {@code count} ids, highest first: a multi-get whose order has nothing to do with doc id
     * order, which is the shape a client reaches for when it already knows the ids it wants.
     *
     * <p>What it costs is the point. A multi-get fetches one document per item, each through its own
     * stored-fields reader, so nothing can be shared between them whatever order they are asked for in.
     */
    private static List<String> descendingIds(final int count) {
        final List<String> ids = new ArrayList<>(count);
        for (int id = count; id >= 1; id--) {
            ids.add(String.valueOf(id));
        }
        return ids;
    }

    /** A counter's delta over the vectors a cell reconstructed, as the cost of one vector. */
    private static String per(final long delta, final int vectors) {
        return String.format(Locale.ROOT, "%.2f", delta / (double) vectors);
    }
}

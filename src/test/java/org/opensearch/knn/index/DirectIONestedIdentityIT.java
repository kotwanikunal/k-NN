/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index;

import lombok.SneakyThrows;
import org.opensearch.knn.DirectIONestedTestCase;
import org.opensearch.knn.NestedDirectIOFixture;
import org.opensearch.knn.NestedDirectIOFixture.DataType;
import org.opensearch.knn.NestedDirectIOFixture.Shape;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * A nested k-NN field answers a query the same way whether its {@code .vec} is read with {@code O_DIRECT} or
 * through the delegate, for every shape of nested query the plugin supports.
 *
 * <p>Nested is the one query shape the Direct I/O path had no coverage for. A nested field is sparse in the doc
 * space, so its {@code .vec} carries the doc-to-ord metadata Lucene keeps for a sparse field, and its children
 * are scored one at a time rather than through the prefetching bulk path. Both of those are read shapes no
 * other test produces.
 *
 * <p>One index per data type and child shape, force merged to a single segment and never reopened, read under
 * three routes: the delegate ({@code knn.direct_io.rescore.enabled} off), {@code O_DIRECT} with no store, and
 * {@code O_DIRECT} with a 64 MB store. The index is created with the route on, so the segment's inputs are
 * routed inputs and a flip changes which path a read takes without rebuilding a reader. Because all three arms
 * score the same bytes with the same scorer, the verdict is equality and not closeness: every id, every score
 * and every expanded child identical, at 0 ulp.
 *
 * <p>Three things keep that from being an assumption. The route is asserted to have engaged as a difference
 * across the index being created, not as a node-lifetime total, because the counters are cumulative and shared
 * by every test on the cluster. No reader may be rebuilt under the comparison, which is
 * {@code dispatch.routed_opens} <em>and</em> {@code dispatch.declined_opens} both standing still -- a reopen
 * while the route is off increments the second and leaves the first alone. And a cell that the scorer never
 * asked for a vector in cannot corroborate anything, so each cell's own device reads are checked against
 * whether it served any vector bytes at all, and a cell that served none is logged as such.
 */
public class DirectIONestedIdentityIT extends DirectIONestedTestCase {

    @SneakyThrows
    public void testFloatNestedMatrixIsIdenticalAcrossRoutes() {
        assertMatrixIdenticalAcrossRoutes(DataType.FLOAT);
    }

    @SneakyThrows
    public void testHalfFloatNestedMatrixIsIdenticalAcrossRoutes() {
        assertMatrixIdenticalAcrossRoutes(DataType.HALF_FLOAT);
    }

    @SneakyThrows
    public void testByteNestedMatrixIsIdenticalAcrossRoutes() {
        assertMatrixIdenticalAcrossRoutes(DataType.BYTE);
    }

    /**
     * Radial search under a nested path, which the directive asks for "where nested supports it". Run on the
     * float field only: if the query is rejected the cell is recorded as not supported rather than failed,
     * because what would be missing is the query shape and not the Direct I/O path.
     */
    @SneakyThrows
    public void testNestedRadialIsIdenticalAcrossRoutesOrRecordedAsUnsupported() {
        applyArm(Arm.DIO_CACHE);
        final Counters beforeIndex = counters();
        final NestedIndex index = createNestedIndex(DataType.FLOAT, Shape.SMALL);
        final Counters opened = counters().since(beforeIndex);
        assertTrue("this cluster routed nothing while the index was built: " + opened, opened.routedOpens > 0);

        final Object[] query = NestedDirectIOFixture.queries(DataType.FLOAT, 1).get(0);
        final String body = String.format(
            Locale.ROOT,
            "{\"_source\":false,\"size\":%d,\"query\":{\"nested\":{\"path\":\"%s\","
                + "\"query\":{\"knn\":{\"%s.%s\":{\"vector\":%s,\"min_score\":0.001}}}}}}",
            K,
            NestedDirectIOFixture.NESTED_FIELD,
            NestedDirectIOFixture.NESTED_FIELD,
            NestedDirectIOFixture.VECTOR_FIELD,
            Arrays.toString(query)
        );

        final Counters beforeDirect = counters();
        final String withDirectIO;
        try {
            withDirectIO = post(index.name(), body);
        } catch (Exception e) {
            logger.info("W4-RADIAL unsupported: {}", e.getMessage());
            return;
        }
        final Counters direct = counters().since(beforeDirect);
        assertTrue("the radial query read nothing from the device, so it proves nothing: " + direct, direct.deviceReads() > 0);

        applyArm(Arm.MMAP);
        final Counters beforeDelegate = counters();
        final String withDelegate = post(index.name(), body);
        final Counters delegate = counters().since(beforeDelegate);
        assertEquals("the delegate arm must not read the device, " + delegate, 0L, delegate.deviceReads());
        assertTrue("the delegate arm must have served the reads itself, " + delegate, delegate.fallbackReads > 0);

        assertEquals("a radial nested query must answer the same way on either route", parseIds(withDelegate), parseIds(withDirectIO));
        assertEquals("and with the same scores", parseScores(withDelegate), parseScores(withDirectIO));
        logger.info("W4-RADIAL supported: {} hits, identical on both routes", parseIds(withDirectIO).size());
    }

    /**
     * Every cell of the matrix for {@code dataType}, both child shapes, three routes. The delegate arm is the
     * reference; both Direct I/O arms are compared against it and against each other, which is the "cache
     * against no cache" clause.
     */
    private void assertMatrixIdenticalAcrossRoutes(final DataType dataType) throws Exception {
        for (final Shape shape : Shape.values()) {
            // On, before the index exists: the segment's inputs have to be routed inputs for a flip to mean
            // anything, and an input is routed at open.
            applyArm(Arm.DIO_CACHE);
            final Counters beforeIndex = counters();
            final NestedIndex index = createNestedIndex(dataType, shape);
            final Counters opened = counters().since(beforeIndex);
            assertTrue(
                String.format(Locale.ROOT, "%s/%s routed nothing while its index was built: %s", dataType, shape, opened),
                opened.routedOpens > 0
            );

            final List<Object[]> queries = NestedDirectIOFixture.queries(dataType, QUERIES);
            final List<Cell> cells = cells(dataType);

            applyArm(Arm.MMAP);
            final Map<String, Answers> reference = new LinkedHashMap<>();
            for (final Cell cell : cells) {
                final Counters before = counters();
                reference.put(cell.name(), run(index.name(), cell, queries));
                final Counters delta = counters().since(before);
                final String where = String.format(Locale.ROOT, "%s/%s/mmap/%s", dataType, shape, cell.name());
                assertEquals(where + ": the delegate arm must not read the device, " + delta, 0L, delta.deviceReads());
                assertEquals(where + ": no reader may have been reopened, " + delta, 0L, delta.routedOpens + delta.declinedOpens);
                if (delta.servedBytes > 0) {
                    assertTrue(where + ": the delegate must have served the reads itself, " + delta, delta.fallbackReads > 0);
                } else {
                    // Not a failure: a byte field's plain nested query is answered by the graph alone, and so
                    // is a query with the rescore pass turned off. Logged so the cell is not read as evidence
                    // about the route.
                    logger.info("W4-NOREAD {} {} mmap {}: the scorer asked for no vector bytes", dataType, shape, cell.name());
                }
            }

            final Map<String, Answers> noStore = assertArmMatches(index, dataType, shape, Arm.DIO_0B, cells, queries, reference);
            assertArmMatches(index, dataType, shape, Arm.DIO_CACHE, cells, queries, noStore);

            // Whether an oversample factor changes the nested answer at all. Reported and not asserted: if the
            // three factors and the plain cell agree, the factor is not reaching the nested rescore pass, which
            // is a k-NN behaviour and not a Direct I/O one, and the cells are then three more copies of plain.
            for (final String factor : List.of("2.0", "3.0", "5.0")) {
                final Answers oversampled = reference.get("oversample-" + factor);
                if (oversampled != null) {
                    logger.info(
                        "W4-OVERSAMPLE {} {} factor={}: ids {} plain's, scores {} plain's",
                        dataType,
                        shape,
                        factor,
                        oversampled.ids().equals(reference.get("plain").ids()) ? "equal" : "differ from",
                        oversampled.scores().equals(reference.get("plain").scores()) ? "equal" : "differ from"
                    );
                }
            }

            deleteKNNIndex(index.name());
        }
    }

    /**
     * Runs {@code arm} over every cell and compares it to {@code reference} pair by pair, then returns its own
     * answers so that the next arm can be compared against this one.
     */
    private Map<String, Answers> assertArmMatches(
        final NestedIndex index,
        final DataType dataType,
        final Shape shape,
        final Arm arm,
        final List<Cell> cells,
        final List<Object[]> queries,
        final Map<String, Answers> reference
    ) throws Exception {
        applyArm(arm);
        final Map<String, Answers> answers = new LinkedHashMap<>();
        int pairs = 0;
        int differing = 0;
        int readingCells = 0;
        for (final Cell cell : cells) {
            final Counters before = counters();
            final Answers got = run(index.name(), cell, queries);
            final Counters delta = counters().since(before);
            answers.put(cell.name(), got);

            final Answers want = reference.get(cell.name());
            final String where = String.format(Locale.ROOT, "%s/%s/%s/%s", dataType, shape, arm.label(), cell.name());
            assertEquals(where + ": the two routes returned a different number of hits", want.pairs(), got.pairs());
            assertEquals(where + ": the two routes expanded a different number of parents", want.children().size(), got.children().size());
            final List<String> mismatches = new ArrayList<>();
            for (int i = 0; i < want.pairs(); i++) {
                if (want.ids().get(i).equals(got.ids().get(i)) == false || want.scores().get(i).equals(got.scores().get(i)) == false) {
                    mismatches.add(
                        String.format(
                            Locale.ROOT,
                            "pair %d: %s@%s against %s@%s",
                            i,
                            want.ids().get(i),
                            want.scores().get(i),
                            got.ids().get(i),
                            got.scores().get(i)
                        )
                    );
                }
            }
            for (int i = 0; i < want.children().size(); i++) {
                if (want.children().get(i).equals(got.children().get(i)) == false) {
                    mismatches.add(
                        String.format(Locale.ROOT, "children %d: %s against %s", i, want.children().get(i), got.children().get(i))
                    );
                }
            }
            pairs += want.pairs() + want.children().size();
            differing += mismatches.size();
            assertEquals(where + ": " + mismatches, 0, mismatches.size());

            assertEquals(where + ": no reader may have been reopened, " + delta, 0L, delta.routedOpens + delta.declinedOpens);
            if (delta.servedBytes > 0) {
                readingCells++;
                assertTrue(
                    where + ": nothing was read from the device, so this cell was not on the route: " + delta,
                    delta.deviceReads() > 0
                );
            } else {
                logger.info("W4-NOREAD {} {} {} {}: the scorer asked for no vector bytes", dataType, shape, arm.label(), cell.name());
            }
        }
        assertTrue(
            String.format(Locale.ROOT, "%s/%s/%s: no cell read a vector, so the arm proves nothing", dataType, shape, arm.label()),
            readingCells > 0
        );
        logger.info(
            "W4-IDENTITY {} {} {}: {} of {} differing pairs, {} cells of {} read vectors",
            dataType,
            shape,
            arm.label(),
            differing,
            pairs,
            readingCells,
            cells.size()
        );
        return answers;
    }
}

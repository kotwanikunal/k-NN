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

import java.util.Collection;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * The identity matrix for derived source: what a client reads back is the same whether the vectors came
 * through {@code O_DIRECT} or through the page cache, and in both cases it is what the client sent.
 *
 * <p>Every data type a {@code knn_vector} holds, at the root and inside a nested object, through the single
 * document path, a search fetch, a multi-get and a scroll; and, in the third test, through a partial update, an
 * update by query and a reindex performed while the route is on. {@link DirectIODerivedSourceTestCase}'s javadoc
 * states which comparison is made on bytes and which on parsed content, and why.
 *
 * <p>Only {@code X1} is parameterised. The compression config decides how the vectors are indexed for search
 * and nothing about how {@code _source} is reconstructed, which is read from the full-precision values, so a
 * second parameterisation would double the runtime and assert the same thing.
 */
public class DirectIODerivedSourceIdentityIT extends DirectIODerivedSourceTestCase {

    public DirectIODerivedSourceIdentityIT(final CompressionTestConfig compressionConfig) {
        super(compressionConfig);
    }

    @ParametersFactory(argumentFormatting = "compression:%1$s")
    public static Collection<Object[]> compressionParameters() {
        return List.<Object[]>of(new Object[] { CompressionTestConfig.X1 });
    }

    @SneakyThrows
    public void testRootVectorsReadTheSameOnBothRoutes() {
        assertIdentity("w6root", false);
    }

    @SneakyThrows
    public void testNestedVectorsReadTheSameOnBothRoutes() {
        assertIdentity("w6nested", true);
    }

    /**
     * Writes through the route and then reads the result both ways: a partial update, an update by query and a
     * reindex all reconstruct {@code _source} to do their work, so each of them is a write that reads a vector.
     *
     * <p>The mutations run once, under {@code dio-cache}, because they cannot be replayed identically; what is
     * compared is the document they left behind, against the non-derived index that received the same mutation
     * and against the same document read back on {@code mmap}.
     */
    @SneakyThrows
    public void testMutatingPathsThroughDirectIO() {
        applyArm(DirectIOArm.DIO_CACHE);
        final List<DerivedSourceUtils.IndexConfigContext> contexts = contexts("w6mutate", false);
        final String derived = contexts.get(0).indexName;

        final DirectIOCounters beforeBuild = counters();
        prepareIndexPair(contexts);
        final DirectIOCounters built = counters().since(beforeBuild);
        assertTrue("the index routed nothing while it was built: " + built, built.routedOpens > 0);

        // Each of these asserts the derived index against the non-derived one over every document, inside the
        // helper every other derived-source test uses; what is new here is that the route is on while it runs.
        final DirectIOCounters beforeUpdate = counters();
        testUpdate(contexts);
        final DirectIOCounters update = counters().since(beforeUpdate);
        logger.info("W6-MUTATE update_and_update_by_query {}", update);
        assertTrue("the update paths read nothing from the device: " + update, update.deviceReads() > 0);

        final DirectIOCounters beforeRead = counters();
        final Map<String, String> afterUpdate = sourceBytes(derived, allIds(DOC_COUNT));
        final DirectIOCounters read = counters().since(beforeRead);
        assertTrue("the updated documents were not read through the route: " + read, read.deviceReads() > 0);
        applyArm(DirectIOArm.MMAP);
        assertEquals(
            "an updated document reads differently on mmap than it did on the route",
            afterUpdate,
            sourceBytes(derived, allIds(DOC_COUNT))
        );

        applyArm(DirectIOArm.DIO_CACHE);
        final DirectIOCounters beforeReindex = counters();
        testReindex(contexts);
        final DirectIOCounters reindex = counters().since(beforeReindex);
        logger.info("W6-MUTATE reindex {}", reindex);
        assertTrue("the reindex read nothing from the device: " + reindex, reindex.deviceReads() > 0);

        // The two derived reindex targets are read on both routes as well, because a reindex writes a fresh
        // segment whose vectors are the ones a later fetch will reconstruct from.
        for (final int reindexTarget : List.of(2, 4)) {
            final String index = contexts.get(reindexTarget).indexName;
            // A reindex target is not force merged, so whether its segments route is not a given; without
            // this the comparison below could be the delegate against itself and would prove nothing.
            final DirectIOCounters beforeTarget = counters();
            final Map<String, String> onRoute = sourceBytes(index, sampledIds());
            final DirectIOCounters onTarget = counters().since(beforeTarget);
            assertTrue(index + " was not read through the route: " + onTarget, onTarget.deviceReads() > 0);
            applyArm(DirectIOArm.MMAP);
            assertEquals(index + " reads differently on mmap than on the route", onRoute, sourceBytes(index, sampledIds()));
            applyArm(DirectIOArm.DIO_CACHE);
        }
    }

    /**
     * The read-path matrix for one index shape: the index is built on the route, read on {@code mmap} as the
     * reference, and then read again on each Direct I/O arm.
     */
    private void assertIdentity(final String prefix, final boolean nested) throws Exception {
        applyArm(DirectIOArm.DIO_CACHE);
        final List<DerivedSourceUtils.IndexConfigContext> contexts = contexts(prefix, nested);
        final String derived = contexts.get(0).indexName;
        final String reference = contexts.get(1).indexName;

        final DirectIOCounters beforeBuild = counters();
        prepareIndexPair(contexts);
        final DirectIOCounters built = counters().since(beforeBuild);
        assertTrue("the index routed nothing while it was built: " + built, built.routedOpens > 0);
        logger.info("W6-BUILD {} nested={} {}", prefix, nested, built);

        applyArm(DirectIOArm.MMAP);
        final DirectIOCounters beforeMmap = counters();
        final Map<String, Map<String, String>> mmapContent = readPaths(derived, DOC_COUNT);
        final Map<String, String> mmapBytes = sourceBytes(derived, allIds(DOC_COUNT));
        final DirectIOCounters mmap = counters().since(beforeMmap);
        logger.info("W6-ARM {} nested={} mmap {}", prefix, nested, mmap);
        assertEquals("the mmap arm read the device, so the switch did not take: " + mmap, 0L, mmap.deviceReads());

        // Stored source never opens a .vec, so the non-derived index is the same ground truth on every arm and
        // is captured once.
        assertCapturesMatch("the derived index on mmap against stored source", readPaths(reference, DOC_COUNT), mmapContent);

        for (final DirectIOArm arm : List.of(DirectIOArm.DIO_0B, DirectIOArm.DIO_CACHE)) {
            applyArm(arm);
            final DirectIOCounters before = counters();
            final Map<String, Map<String, String>> content = readPaths(derived, DOC_COUNT);
            final Map<String, String> bytes = sourceBytes(derived, allIds(DOC_COUNT));
            final DirectIOCounters delta = counters().since(before);
            logger.info("W6-ARM {} nested={} {} {}", prefix, nested, arm.label(), delta);

            assertCapturesMatch(arm.label() + " against mmap", mmapContent, content);
            assertEquals(String.format(Locale.ROOT, "%s returned different _source bytes from mmap's", arm.label()), mmapBytes, bytes);
            assertTrue(
                String.format(Locale.ROOT, "%s read nothing from the device, so it was not on the route: %s", arm.label(), delta),
                delta.deviceReads() > 0
            );
        }
    }
}

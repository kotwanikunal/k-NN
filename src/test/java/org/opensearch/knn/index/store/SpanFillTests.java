/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import com.carrotsearch.randomizedtesting.annotations.ThreadLeakFilters;
import lombok.SneakyThrows;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.codecs.lucene95.HasIndexSlice;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.apache.lucene.util.hnsw.RandomVectorScorerSupplier;
import org.opensearch.common.settings.ClusterSettings;
import org.opensearch.common.settings.Setting;
import org.opensearch.common.settings.Settings;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.codec.scorer.PrefetchableFlatVectorScorer;
import org.opensearch.knn.index.memory.SlabVectorCache;
import org.opensearch.knn.index.memory.VectorCacheScope;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

import static org.mockito.Mockito.when;

/**
 * F5, span fill: while the vector cache is below its slab target, a batch also admits the whole vectors that
 * happen to lie inside the spans it staged, not only the candidates it asked for.
 *
 * <h2>The measurement it exists for</h2>
 * MEASURED in the phase 16-P pilot (ref/PILOT-REPORT.md section 4): on dio-1m with a 4 GB cache, which has
 * room for every vector in the file, Direct I/O's p50 stayed 9.7x mmap's because 16% of candidates still
 * missed after a 10,000-query warm-up. In that warm-up Direct I/O read 2.01 MB per query in about 40 KB
 * staged spans, 20 GB in all and 6.5 times the whole file, and cached only the requested vectors; mmap's
 * page cache kept every page it read and ended holding 95% of the file. The cache was paying for the reads
 * and keeping a fraction of what they delivered.
 *
 * <h2>What is on trial here</h2>
 * Not that it fills, which is easy, but the three ways it could fill something wrong. A staged range is
 * block aligned in the whole file, so its ends routinely fall inside a vector and routinely reach outside
 * the field's own slice, into another field's vectors or into the file's header or footer. A span fill that
 * got either edge wrong would cache a vector made of the wrong bytes under a key a later lookup hits, and
 * nothing would report an error: the query would just get a different answer. So the tests below check the
 * geometry against the file, not against the arithmetic that produced it.
 */
@ThreadLeakFilters(defaultFilters = true, filters = { DirectIOReadPoolTests.ReadPoolThreadFilter.class })
public class SpanFillTests extends KNNTestCase {

    private static final int SLAB_BYTES = 1 << 20;
    private static final int FIELD = 0;
    private static final int DIMENSION = 128;
    private static final int VECTOR_BYTES = DIMENSION * Float.BYTES;
    private static final int VECTORS = 1024;
    /** Neither block nor vector aligned, so the slice start cannot accidentally make the geometry work. */
    private static final int HEADER_BYTES = 100;
    /** Bytes after the field's slice, which a span reaching past its end would admit if the clip were wrong. */
    private static final int TRAILER_BYTES = 8192;

    private SlabVectorCache cache;
    private Set<Setting<?>> nodeSettings;

    @Override
    public void setUp() throws Exception {
        super.setUp();
        cache = new SlabVectorCache(SlabVectorCache.DIRECT_SOURCE, SLAB_BYTES);
        cache.configure(8L * SLAB_BYTES);
        VectorCacheScope.setNodeCacheForTesting(cache);
        nodeSettings = new HashSet<>(ClusterSettings.BUILT_IN_CLUSTER_SETTINGS);
        nodeSettings.addAll(
            KNNSettings.state()
                .getSettings()
                .stream()
                .filter(setting -> setting.getProperties().contains(Setting.Property.NodeScope))
                .collect(Collectors.toList())
        );
    }

    @Override
    public void tearDown() throws Exception {
        VectorCacheScope.setNodeCacheForTesting(null);
        if (cache != null) {
            cache.configure(0L);
            cache = null;
        }
        DirectIOStagingLease.resetForTesting();
        DirectIOReadPool.resetForTesting();
        DirectIOVectorIndexInput.STATS.reset();
        super.tearDown();
    }

    /** Sets {@code knn.direct_io.rescore.cache.span_fill} as the scorer reads it. */
    private void setSpanFill(final boolean enabled) {
        final Settings settings = Settings.builder().put(KNNSettings.KNN_DIRECT_IO_RESCORE_CACHE_SPAN_FILL, enabled).build();
        when(clusterService.getClusterSettings()).thenReturn(new ClusterSettings(settings, nodeSettings));
        assertEquals("the test's own setting plumbing is broken", enabled, KNNSettings.isDirectIORescoreCacheSpanFillEnabled());
    }

    @SneakyThrows
    private void assumeDirectIOWorksHere() {
        assumeTrue("this JDK has no ExtendedOpenOption.DIRECT", DirectIOVectorIndexInput.directOpenOption() != null);
        final Path probe = createTempDir().resolve("span-fill-probe");
        Files.write(probe, new byte[8192]);
        try (FileChannel channel = FileChannel.open(probe, StandardOpenOption.READ, DirectIOVectorIndexInput.directOpenOption())) {
            channel.read(ByteBuffer.allocateDirect(8192).alignedSlice(4096));
        } catch (Exception e) {
            assumeNoException("this filesystem refuses O_DIRECT", e);
        }
    }

    /** Every float of a vector is derived from its ordinal, so a wrong vector cannot look right. */
    private static float[] vectorFor(final int ordinal) {
        final float[] vector = new float[DIMENSION];
        for (int component = 0; component < DIMENSION; component++) {
            vector[component] = ordinal * 4.0f + component * 0.125f;
        }
        return vector;
    }

    /**
     * A header, then {@link #VECTORS} vectors, then a trailer whose floats are recognisably not vectors.
     * The field's slice is {@code [HEADER_BYTES, HEADER_BYTES + VECTORS * VECTOR_BYTES)}.
     */
    @SneakyThrows
    private Path writeFile() {
        final Path path = createTempDir().resolve("span-fill.vec");
        final ByteBuffer buffer = ByteBuffer.allocate(HEADER_BYTES + VECTORS * VECTOR_BYTES + TRAILER_BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < HEADER_BYTES; i++) {
            buffer.put(i, (byte) 0xA5);
        }
        for (int ordinal = 0; ordinal < VECTORS; ordinal++) {
            final float[] vector = vectorFor(ordinal);
            for (int component = 0; component < DIMENSION; component++) {
                buffer.putFloat(HEADER_BYTES + ordinal * VECTOR_BYTES + component * Float.BYTES, vector[component]);
            }
        }
        for (int i = 0; i < TRAILER_BYTES; i += Float.BYTES) {
            buffer.putFloat(HEADER_BYTES + VECTORS * VECTOR_BYTES + i, -98765.0f);
        }
        Files.write(path, buffer.array());
        return path;
    }

    /** One run the walk offered: the ordinals, and the bytes it pointed at, copied out. */
    private record OfferedRun(int firstOrdinal, int count, float[][] vectors) {
    }

    /**
     * Prefetches {@code [offsetInSlice, +length)} of the field's slice and returns what the span walk then
     * offers, with the bytes it pointed at already decoded, so a caller can compare them with the file.
     */
    private List<OfferedRun> offeredRuns(final DirectIOVectorIndexInput slice, final long offsetInSlice, final long length)
        throws IOException {
        slice.prefetch(offsetInSlice, length);
        final List<OfferedRun> runs = new ArrayList<>();
        assertTrue("the prefetch staged nothing, so there is no span to walk", slice.prepareStagedSpans() > 0);
        slice.visitStagedSpanVectors(VECTOR_BYTES, (buffer, position, firstOrdinal, count) -> {
            final float[][] decoded = new float[count][DIMENSION];
            for (int i = 0; i < count; i++) {
                for (int component = 0; component < DIMENSION; component++) {
                    decoded[i][component] = buffer.getFloat(position + i * VECTOR_BYTES + component * Float.BYTES);
                }
            }
            runs.add(new OfferedRun(firstOrdinal, count, decoded));
            return true;
        });
        return runs;
    }

    private void assertRunsMatchTheFile(final List<OfferedRun> runs) {
        assertFalse("the walk offered nothing at all", runs.isEmpty());
        for (final OfferedRun run : runs) {
            assertTrue("a run must start inside the field, it started at " + run.firstOrdinal(), run.firstOrdinal() >= 0);
            assertTrue(
                "a run must end inside the field, it ended at " + (run.firstOrdinal() + run.count() - 1),
                run.firstOrdinal() + run.count() <= VECTORS
            );
            for (int i = 0; i < run.count(); i++) {
                assertArrayEquals(
                    "ordinal " + (run.firstOrdinal() + i) + " was offered bytes that are not its own",
                    vectorFor(run.firstOrdinal() + i),
                    run.vectors()[i],
                    0.0f
                );
            }
        }
    }

    // ---------------------------------------------------------------------------------------------------
    // Geometry: what a span may and may not offer
    // ---------------------------------------------------------------------------------------------------

    /**
     * A span in the middle of the field. Its block-aligned ends fall inside vectors at both edges, and the
     * walk must offer the whole ones in between and no partial one.
     */
    @SneakyThrows
    public void testASpanOffersOnlyWholeVectorsAndTheirOwnBytes() {
        assumeDirectIOWorksHere();
        final Path path = writeFile();
        try (DirectIOVectorIndexInput root = DirectIOVectorIndexInput.open(path)) {
            assertNotNull("the fixture must be readable with O_DIRECT", root);
            try (DirectIOVectorIndexInput slice = root.slice("field", HEADER_BYTES, (long) VECTORS * VECTOR_BYTES)) {
                final List<OfferedRun> runs = offeredRuns(slice, 5L * VECTOR_BYTES, 3L * VECTOR_BYTES);
                assertRunsMatchTheFile(runs);
                // More than the three that were asked for: that is the whole point of span fill.
                final int offered = runs.stream().mapToInt(OfferedRun::count).sum();
                assertTrue("a span should hold more whole vectors than the 3 requested, it held " + offered, offered > 3);
            }
        }
    }

    /**
     * A span at the very end of the field. The read is block aligned, so it delivers bytes past the slice,
     * and those bytes are another region of the file: the last ordinal offered must be the field's last.
     */
    @SneakyThrows
    public void testASpanAtTheSliceEndOffersNothingBeyondIt() {
        assumeDirectIOWorksHere();
        final Path path = writeFile();
        try (DirectIOVectorIndexInput root = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(root);
            try (DirectIOVectorIndexInput slice = root.slice("field", HEADER_BYTES, (long) VECTORS * VECTOR_BYTES)) {
                final List<OfferedRun> runs = offeredRuns(slice, (long) (VECTORS - 2) * VECTOR_BYTES, 2L * VECTOR_BYTES);
                assertRunsMatchTheFile(runs);
                final int highest = runs.stream().mapToInt(run -> run.firstOrdinal() + run.count() - 1).max().orElse(-1);
                assertEquals("the field's last vector is the last thing that may be offered", VECTORS - 1, highest);
            }
        }
    }

    /**
     * A span at the very start of the field, where the aligned read reaches back into the header. The first
     * ordinal offered must be 0 and must carry ordinal 0's bytes, not the header's.
     */
    @SneakyThrows
    public void testASpanAtTheSliceStartOffersNothingBeforeIt() {
        assumeDirectIOWorksHere();
        final Path path = writeFile();
        try (DirectIOVectorIndexInput root = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(root);
            try (DirectIOVectorIndexInput slice = root.slice("field", HEADER_BYTES, (long) VECTORS * VECTOR_BYTES)) {
                final List<OfferedRun> runs = offeredRuns(slice, 0L, 2L * VECTOR_BYTES);
                assertRunsMatchTheFile(runs);
                assertEquals("the field's first vector", 0, runs.get(0).firstOrdinal());
            }
        }
    }

    /**
     * A slice that is only part of the file's vector region, which is what a second field on the same file
     * looks like. A span may reach into the neighbouring field's bytes and must offer none of them.
     */
    @SneakyThrows
    public void testASpanMayNotReachIntoTheNeighbouringFieldsSlice() {
        assumeDirectIOWorksHere();
        final Path path = writeFile();
        final int sliceVectors = 64;
        try (DirectIOVectorIndexInput root = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(root);
            // A slice holding ordinals 0..63 of the file's vectors; everything after it belongs to someone
            // else as far as this slice is concerned.
            try (DirectIOVectorIndexInput slice = root.slice("field", HEADER_BYTES, (long) sliceVectors * VECTOR_BYTES)) {
                slice.prefetch((long) (sliceVectors - 1) * VECTOR_BYTES, VECTOR_BYTES);
                assertTrue(slice.prepareStagedSpans() > 0);
                final List<Integer> offered = new ArrayList<>();
                slice.visitStagedSpanVectors(VECTOR_BYTES, (buffer, position, firstOrdinal, count) -> {
                    for (int i = 0; i < count; i++) {
                        offered.add(firstOrdinal + i);
                    }
                    return true;
                });
                assertFalse(offered.isEmpty());
                for (final int ordinal : offered) {
                    assertTrue("ordinal " + ordinal + " is outside this slice of " + sliceVectors, ordinal < sliceVectors);
                }
                assertEquals("the slice's last vector must still be offered", sliceVectors - 1, (int) offered.get(offered.size() - 1));
            }
        }
    }

    /** No burst, nothing to walk: both calls answer zero rather than reading anything. */
    @SneakyThrows
    public void testThereIsNothingToWalkWithoutABurst() {
        assumeDirectIOWorksHere();
        final Path path = writeFile();
        try (DirectIOVectorIndexInput root = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(root);
            try (DirectIOVectorIndexInput slice = root.slice("field", HEADER_BYTES, (long) VECTORS * VECTOR_BYTES)) {
                assertEquals(0, slice.prepareStagedSpans());
                assertEquals(0, slice.visitStagedSpanVectors(VECTOR_BYTES, (b, p, f, c) -> true));
            }
        }
    }

    // ---------------------------------------------------------------------------------------------------
    // Through the scorer: the admission the read path makes
    // ---------------------------------------------------------------------------------------------------

    /**
     * One scored batch, and afterwards the cache holds vectors no candidate asked for, each with its own
     * bytes, and a later lookup of one of them is a hit.
     */
    @SneakyThrows
    public void testSpanFillCachesVectorsNobodyAskedForAndTheyHitAfterwards() {
        assumeDirectIOWorksHere();
        setSpanFill(true);
        final Path path = writeFile();
        final int[] nodes = { 4, 40, 400 };
        try (DirectIOVectorIndexInput root = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(root);
            try (DirectIOVectorIndexInput slice = root.slice("field", HEADER_BYTES, (long) VECTORS * VECTOR_BYTES)) {
                final Object core = new Object();
                final RandomVectorScorer scorer = scorerOver(slice, core);
                final float[] scores = new float[nodes.length];
                scorer.bulkScore(nodes, scores, nodes.length);

                assertTrue("span fill admitted nothing: " + cache.spanFills(), cache.spanFills() > 0);
                assertTrue(
                    "the cache should hold more than the " + nodes.length + " candidates, it holds " + cache.entries(),
                    cache.entries() > nodes.length
                );

                // Every entry the cache now holds has to be the vector its key names, requested or not.
                final float[] read = new float[DIMENSION];
                int neighbours = 0;
                for (int ordinal = 0; ordinal < VECTORS; ordinal++) {
                    if (cache.get(core, FIELD, ordinal, VECTOR_BYTES, read) == false) {
                        continue;
                    }
                    assertArrayEquals("ordinal " + ordinal, vectorFor(ordinal), read, 0.0f);
                    if (ordinal != 4 && ordinal != 40 && ordinal != 400) {
                        neighbours++;
                    }
                }
                assertTrue("no vector was cached that was not asked for", neighbours > 0);
            }
        }
    }

    /** With the setting off, admission is the requested candidates and nothing else. */
    @SneakyThrows
    public void testTheSettingOffRestoresRequestedOnlyAdmission() {
        assumeDirectIOWorksHere();
        setSpanFill(false);
        final Path path = writeFile();
        final int[] nodes = { 4, 40, 400 };
        try (DirectIOVectorIndexInput root = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(root);
            try (DirectIOVectorIndexInput slice = root.slice("field", HEADER_BYTES, (long) VECTORS * VECTOR_BYTES)) {
                final Object core = new Object();
                final RandomVectorScorer scorer = scorerOver(slice, core);
                scorer.bulkScore(nodes, new float[nodes.length], nodes.length);

                assertEquals("span fill must be off", 0L, cache.spanFills());
                assertEquals("only the candidates may be cached", nodes.length, cache.entries());
                final float[] read = new float[DIMENSION];
                for (final int node : nodes) {
                    assertTrue(cache.get(core, FIELD, node, VECTOR_BYTES, read));
                }
                assertFalse(cache.get(core, FIELD, 5, VECTOR_BYTES, read));
            }
        }
    }

    /** Staged spans only: values read through mmap have none, so their admission is unchanged. */
    @SneakyThrows
    public void testAnMmapInputSpanFillsNothing() {
        setSpanFill(true);
        final Path directory = createTempDir();
        final Path path = writeFile();
        Files.copy(path, directory.resolve("mmap.vec"));
        final int[] nodes = { 4, 40, 400 };
        try (MMapDirectory mmap = new MMapDirectory(directory)) {
            try (IndexInput input = mmap.openInput("mmap.vec", IOContext.DEFAULT)) {
                final IndexInput slice = input.slice("field", HEADER_BYTES, (long) VECTORS * VECTOR_BYTES);
                final Object core = new Object();
                final RandomVectorScorer scorer = scorerOver(slice, core);
                scorer.bulkScore(nodes, new float[nodes.length], nodes.length);

                assertEquals("an mmap input has no staged span to fill from", 0L, cache.spanFills());
                assertEquals(nodes.length, cache.entries());
            }
        }
    }

    // ---------------------------------------------------------------------------------------------------
    // Helpers: the scorer the read path builds
    // ---------------------------------------------------------------------------------------------------

    /** The scorer {@code ExactSearcher} would build over these values, with the cache key in scope. */
    private RandomVectorScorer scorerOver(final IndexInput slice, final Object core) throws IOException {
        final SliceFloatVectorValues values = new SliceFloatVectorValues(slice, DIMENSION, VECTORS);
        VectorCacheScope.open(core, FIELD);
        try {
            return new PrefetchableFlatVectorScorer(plainScorer()).getRandomVectorScorer(
                VectorSimilarityFunction.EUCLIDEAN,
                values,
                vectorFor(0)
            );
        } finally {
            VectorCacheScope.clear();
        }
    }

    /** Values that read one vector at a time through the given slice, and expose it for the prefetch. */
    private static final class SliceFloatVectorValues extends FloatVectorValues implements HasIndexSlice {
        private final IndexInput slice;
        private final int dimension;
        private final int size;
        private final float[] scratch;

        private SliceFloatVectorValues(final IndexInput slice, final int dimension, final int size) {
            this.slice = slice;
            this.dimension = dimension;
            this.size = size;
            this.scratch = new float[dimension];
        }

        @Override
        public float[] vectorValue(final int ord) throws IOException {
            slice.seek((long) ord * dimension * Float.BYTES);
            slice.readFloats(scratch, 0, dimension);
            return scratch;
        }

        @Override
        public int dimension() {
            return dimension;
        }

        @Override
        public int size() {
            return size;
        }

        @Override
        public FloatVectorValues copy() {
            return this;
        }

        @Override
        public IndexInput getSlice() {
            return slice;
        }
    }

    /** Lucene's plain float kernel, which is what the delegate on this path is. */
    private static FlatVectorsScorer plainScorer() {
        return new FlatVectorsScorer() {
            @Override
            public RandomVectorScorerSupplier getRandomVectorScorerSupplier(final VectorSimilarityFunction f, final KnnVectorValues v) {
                throw new UnsupportedOperationException();
            }

            @Override
            public RandomVectorScorer getRandomVectorScorer(
                final VectorSimilarityFunction similarity,
                final KnnVectorValues values,
                final float[] target
            ) {
                return new RandomVectorScorer.AbstractRandomVectorScorer(values) {
                    @Override
                    public float score(final int node) throws IOException {
                        return similarity.compare(target, ((FloatVectorValues) values).vectorValue(node));
                    }
                };
            }

            @Override
            public RandomVectorScorer getRandomVectorScorer(
                final VectorSimilarityFunction similarity,
                final KnnVectorValues values,
                final byte[] target
            ) {
                throw new UnsupportedOperationException();
            }
        };
    }
}

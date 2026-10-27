/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import com.carrotsearch.randomizedtesting.ThreadFilter;
import com.carrotsearch.randomizedtesting.annotations.ThreadLeakFilters;
import lombok.SneakyThrows;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.opensearch.common.settings.ClusterSettings;
import org.opensearch.common.settings.Setting;
import org.opensearch.common.settings.Settings;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.KNNSettings;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;

import static org.mockito.Mockito.when;

/**
 * M3: what the slab store under the input has to be, read through the real
 * {@link DirectIOVectorIndexInput}.
 *
 * <p>Three properties, and the first is the one the whole design rests on. <b>A hit must be bit-identical to
 * the miss path</b>, because the scorer above this input cannot tell the two apart and must not need to: it
 * prefetches a candidate's range, reads through the values object, and scores whatever floats come back. If a
 * hit decoded even one float differently, the cache would change answers, and a changed answer is not an
 * error anybody sees — it is a different search result. So the comparison here is on raw float bits, not on a
 * delta, and it is repeated for every {@link VectorSimilarityFunction} the plugin can score with, because a
 * similarity is where a one-bit difference would surface as a rank change.
 *
 * <p>The second is that <b>a hit costs no device read</b>: the lookup in {@code prefetch} drops the range
 * from the burst, so the staged read count must not move. The third is that <b>a blocking read is not
 * admitted</b>: only the burst table fills the store, because the read-ahead ring is sequential (where a
 * cache buys nothing and would evict the random path's working set) and a blocking read is the fallback the
 * cache exists to avoid.
 */
@ThreadLeakFilters(defaultFilters = true, filters = { DirectIOCacheHitIdentityTests.ReadPoolThreadFilter.class })
public class DirectIOCacheHitIdentityTests extends KNNTestCase {

    /** The read pool is never shut down by design; its daemon threads outlive any one suite. */
    public static final class ReadPoolThreadFilter implements ThreadFilter {
        @Override
        public boolean reject(final Thread thread) {
            return thread.getName().startsWith(DirectIOReadPool.THREAD_NAME_PREFIX);
        }
    }

    /** A Lucene {@code .vec}'s header length, so offsets are not multiples of the vector length. */
    private static final int HEADER_LENGTH = 44;
    private static final int VECTOR_COUNT = 64;
    private static final int TEST_SLAB_BYTES = 1024 * 1024;

    @Override
    public void tearDown() throws Exception {
        DirectIOVectorCache.resetForTesting();
        DirectIOReadPool.resetForTesting();
        DirectIOVectorIndexInput.STATS.reset();
        super.tearDown();
    }

    /**
     * A hit and a miss for the same vector must be the same bits, and must score the same under every
     * similarity function. Both dimensions the backlog names, because 128 is 512 B (so several vectors share
     * a 4 KiB block) and 768 is 3,072 B (so one vector straddles a block boundary), and the decode path
     * differs between them in whether a read is served whole from one buffer.
     */
    @SneakyThrows
    public void testAHitIsBitIdenticalToAMissAndScoresTheSame() {
        assumeDirectIOWorksHere();
        for (final int dimension : new int[] { 128, 768 }) {
            final float[][] vectors = randomVectors(11L + dimension, dimension);
            final Path path = writeVectorFile(createTempDir(), "identity-" + dimension + ".vec", vectors);

            // Pass 1: the cache off, so every read is a staged or blocking device read. This is phase 15's
            // behaviour with the feature at its shipped default, and it is the reference.
            cacheOff();
            final float[][] fromDevice = new float[VECTOR_COUNT][dimension];
            try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path)) {
                assertNotNull(input);
                for (int ord = 0; ord < VECTOR_COUNT; ord++) {
                    readVector(input, ord, dimension, fromDevice[ord]);
                }
            }

            // Pass 2: the cache on. Each vector is read twice -- the first read is a miss that admits it, the
            // second is a hit served out of a slab.
            cacheOn(2);
            final float[][] fromCache = new float[VECTOR_COUNT][dimension];
            try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path)) {
                assertNotNull(input);
                for (int ord = 0; ord < VECTOR_COUNT; ord++) {
                    readVector(input, ord, dimension, new float[dimension]);
                }
                final long hitsBefore = DirectIOVectorCache.hits();
                for (int ord = 0; ord < VECTOR_COUNT; ord++) {
                    readVector(input, ord, dimension, fromCache[ord]);
                }
                assertEquals(
                    "every vector of the second pass must have been a hit at " + dimension + " dims",
                    hitsBefore + VECTOR_COUNT,
                    DirectIOVectorCache.hits()
                );
            }

            // Bit identity, not a delta: assertArrayEquals with a 0.0f delta already treats -0.0f and 0.0f
            // as equal and both NaNs as equal, and a cache that returned a different NaN payload or a sign
            // flip on zero would pass it. Raw bits cannot be fooled.
            for (int ord = 0; ord < VECTOR_COUNT; ord++) {
                for (int i = 0; i < dimension; i++) {
                    assertEquals(
                        "dimension " + dimension + " vector " + ord + " component " + i,
                        Float.floatToRawIntBits(fromDevice[ord][i]),
                        Float.floatToRawIntBits(fromCache[ord][i])
                    );
                }
            }

            // And the thing a changed bit would actually do: move a score. Every similarity function, on
            // every vector, against a fixed query.
            final float[] query = randomVectors(99L, dimension)[0];
            for (final VectorSimilarityFunction similarity : VectorSimilarityFunction.values()) {
                for (int ord = 0; ord < VECTOR_COUNT; ord++) {
                    final float expected = similarity.compare(query, fromDevice[ord]);
                    final float actual = similarity.compare(query, fromCache[ord]);
                    assertEquals(
                        similarity + " at " + dimension + " dims, vector " + ord,
                        Float.floatToRawIntBits(expected),
                        Float.floatToRawIntBits(actual)
                    );
                }
            }
        }
    }

    /**
     * The first lookup's whole purpose: a cached vector is dropped from the burst, so the device sees nothing
     * at all for it. Asserted on the counters that attribute reads to this input, because that is the only
     * place the saving is visible from inside the process.
     */
    @SneakyThrows
    public void testAHitRemovesItsRangeFromTheBurst() {
        assumeDirectIOWorksHere();
        final int dimension = 128;
        final float[][] vectors = randomVectors(12L, dimension);
        final Path path = writeVectorFile(createTempDir(), "burst.vec", vectors);
        cacheOn(2);
        final float[] scratch = new float[dimension];

        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(input);
            for (int ord = 0; ord < 16; ord++) {
                readVector(input, ord, dimension, scratch);
            }
            final long stagedAfterCold = DirectIOVectorIndexInput.STATS.stagedReads();
            final long blockingAfterCold = DirectIOVectorIndexInput.STATS.blockingReads();
            final long prefetchesAfterCold = DirectIOVectorIndexInput.STATS.prefetchCalls();
            assertTrue("the cold pass had to reach the device", stagedAfterCold + blockingAfterCold > 0);

            for (int repeat = 0; repeat < 8; repeat++) {
                for (int ord = 0; ord < 16; ord++) {
                    readVector(input, ord, dimension, scratch);
                    assertArrayEquals(vectors[ord], scratch, 0.0f);
                }
            }

            assertEquals("staged reads must not grow", stagedAfterCold, DirectIOVectorIndexInput.STATS.stagedReads());
            assertEquals("and neither may blocking reads", blockingAfterCold, DirectIOVectorIndexInput.STATS.blockingReads());
            assertEquals("the prefetches happened and were dropped, not declined", 0L, DirectIOVectorIndexInput.STATS.stageDeclined());
            assertEquals(
                "and the prefetch calls were still made: the drop is inside prefetch",
                prefetchesAfterCold + 8 * 16,
                DirectIOVectorIndexInput.STATS.prefetchCalls()
            );
        }
    }

    /**
     * Admission is from a staged read only. A read with no {@code prefetch} in front of it is served by a
     * blocking {@code pread}, and must leave the store empty: the ring and the blocking window are not the
     * random rescore channel this cache exists for.
     */
    @SneakyThrows
    public void testABlockingReadIsNotAdmitted() {
        assumeDirectIOWorksHere();
        final int dimension = 128;
        final float[][] vectors = randomVectors(13L, dimension);
        final Path path = writeVectorFile(createTempDir(), "blocking.vec", vectors);
        cacheOn(2);
        final float[] scratch = new float[dimension];

        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(input);
            // No prefetch, so source() falls through the (empty) burst table and the disabled ring to a
            // blocking read. Backwards through the file, so no sequential run can be detected either.
            for (int ord = VECTOR_COUNT - 1; ord >= 0; ord--) {
                input.seek(HEADER_LENGTH + (long) ord * dimension * Float.BYTES);
                input.readFloats(scratch, 0, dimension);
                assertArrayEquals("a blocking read must still be correct", vectors[ord], scratch, 0.0f);
            }

            assertTrue("these were blocking reads", DirectIOVectorIndexInput.STATS.blockingReads() > 0);
            assertEquals("and none of them may have been admitted", 0L, DirectIOVectorCache.entries());
            assertEquals(0L, DirectIOVectorCache.bytes());
            assertEquals("every lookup missed", 0L, DirectIOVectorCache.hits());
            assertEquals((long) VECTOR_COUNT, DirectIOVectorCache.misses());
            // A slab is only taken by a fill, so none was.
            assertEquals(0L, DirectIOVectorCache.slabs());
        }
    }

    /**
     * A second input over the same file reads a hit the first one admitted, which is the cross-reader reuse
     * the whole mechanism exists to collect, and it has to be bit identical too.
     */
    @SneakyThrows
    public void testASecondReadersHitIsBitIdentical() {
        assumeDirectIOWorksHere();
        final int dimension = 768;
        final float[][] vectors = randomVectors(14L, dimension);
        final Path path = writeVectorFile(createTempDir(), "shared-identity.vec", vectors);
        cacheOn(2);
        final float[] first = new float[dimension];
        final float[] second = new float[dimension];

        // Both held open: the claim is that a second reader of a file another reader cached gets the same
        // bits. Since O3 the owning input's close frees that file's entries, so the writer stays open here
        // (a reopened file refills instead, which DirectIOClosedFileReleaseTests covers).
        try (
            DirectIOVectorIndexInput writer = DirectIOVectorIndexInput.open(path);
            DirectIOVectorIndexInput reader = DirectIOVectorIndexInput.open(path)
        ) {
            assertNotNull(writer);
            assertNotNull(reader);
            readVector(writer, 9, dimension, first);
            final long hitsBefore = DirectIOVectorCache.hits();
            readVector(reader, 9, dimension, second);
            assertEquals("the second reader must hit", hitsBefore + 1, DirectIOVectorCache.hits());
        }
        for (int i = 0; i < dimension; i++) {
            assertEquals("component " + i, Float.floatToRawIntBits(first[i]), Float.floatToRawIntBits(second[i]));
        }
    }

    // -----------------------------------------------------------------------------------------------

    /** Skips rather than fails where {@code O_DIRECT} is unavailable, as every other test here does. */
    private void assumeDirectIOWorksHere() throws Exception {
        assumeTrue("this JDK has no ExtendedOpenOption.DIRECT", DirectIOVectorIndexInput.directOpenOption() != null);
        final Path probe = createTempDir().resolve("direct-io-probe");
        Files.write(probe, new byte[8192]);
        try (FileChannel channel = FileChannel.open(probe, StandardOpenOption.READ, DirectIOVectorIndexInput.directOpenOption())) {
            channel.read(ByteBuffer.allocateDirect(8192).alignedSlice(4096));
        } catch (Exception e) {
            assumeNoException("this filesystem refuses O_DIRECT", e);
        }
    }

    private void withNodeSettings(final Settings settings) {
        final Set<Setting<?>> registered = new HashSet<>(ClusterSettings.BUILT_IN_CLUSTER_SETTINGS);
        registered.addAll(
            KNNSettings.state()
                .getSettings()
                .stream()
                .filter(setting -> setting.getProperties().contains(Setting.Property.NodeScope))
                .toList()
        );
        // The route is on unless the caller says otherwise. These cases read through O_DIRECT with no
        // delegate to fall back to, which is the one shape the route switch does not reach, so the node
        // they model is a node that is routing: publishing the route off would give the store back under
        // them. A case whose subject is the switch puts its own value and that wins.
        final Settings routed = Settings.builder().put(KNNSettings.KNN_DIRECT_IO_RESCORE_ENABLED, true).put(settings).build();
        when(clusterService.getClusterSettings()).thenReturn(new ClusterSettings(routed, registered));
        KNNSettings.state().setClusterService(clusterService);
        // The seeding a node does at startup: the dynamic Direct I/O settings live in volatile fields of
        // the components that own them, so publishing is how a setting reaches the read path.
        KNNSettings.state().publishDirectIOSettings();
    }

    /** The read-ahead ring off, so the burst table and the blocking window are the only two channels. */
    private void cacheOff() {
        withNodeSettings(Settings.builder().put(KNNSettings.KNN_DIRECT_IO_READAHEAD_WINDOW, 0).build());
        DirectIOVectorCache.resetForTesting();
    }

    private void cacheOn(final int slabs) {
        final long maxBytes = (long) slabs * TEST_SLAB_BYTES;
        withNodeSettings(
            Settings.builder()
                .put(KNNSettings.KNN_DIRECT_IO_RESCORE_CACHE_MAX_BYTES, maxBytes + "b")
                .put(KNNSettings.KNN_DIRECT_IO_READAHEAD_WINDOW, 0)
                .build()
        );
        DirectIOVectorCache.resetForTesting();
        DirectIOVectorCache.setSlabBytesForTesting(TEST_SLAB_BYTES);
        DirectIOVectorCache.publishMaxBytes(maxBytes);
    }

    private float[][] randomVectors(final long seed, final int dimension) {
        final Random random = new Random(seed);
        final float[][] vectors = new float[VECTOR_COUNT][dimension];
        for (final float[] vector : vectors) {
            for (int i = 0; i < dimension; i++) {
                // Scaled away from zero so that every similarity function has something to work with, and
                // signed so that a sign bit is part of what is being compared.
                vector[i] = (random.nextFloat() - 0.5f) * 8.0f;
            }
        }
        return vectors;
    }

    @SneakyThrows
    private Path writeVectorFile(final Path dir, final String name, final float[][] vectors) {
        final Path path = dir.resolve(name);
        final int vectorBytes = vectors[0].length * Float.BYTES;
        final ByteBuffer buffer = ByteBuffer.allocate(HEADER_LENGTH + vectors.length * vectorBytes + 16).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < HEADER_LENGTH; i++) {
            buffer.put((byte) i);
        }
        for (final float[] vector : vectors) {
            for (final float value : vector) {
                buffer.putFloat(value);
            }
        }
        buffer.putLong(0).putLong(0);
        Files.write(path, buffer.array());
        return path;
    }

    /** One vector through the real input: declare it, consume it. This is the shape the rescore path has. */
    @SneakyThrows
    private void readVector(final DirectIOVectorIndexInput input, final int ord, final int dimension, final float[] into) {
        final int vectorBytes = dimension * Float.BYTES;
        final long at = HEADER_LENGTH + (long) ord * vectorBytes;
        input.prefetch(at, vectorBytes);
        input.seek(at);
        input.readFloats(into, 0, dimension);
    }
}

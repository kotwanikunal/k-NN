/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import com.carrotsearch.randomizedtesting.ThreadFilter;
import com.carrotsearch.randomizedtesting.annotations.ThreadLeakFilters;
import lombok.SneakyThrows;
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
 * The node-wide vector cache on the Direct I/O rescore path (phase 15 D4).
 *
 * <p>What has to be true of it, in the order it would hurt if it were not: the accounting is honest, the
 * weigher really bounds the heap, two readers of one file share entries (which is the whole of the measured
 * reuse), two different files never collide (which would serve one file's bytes for another's), and with the
 * cache off nothing in the read path touches it.
 */
@ThreadLeakFilters(defaultFilters = true, filters = { DirectIOVectorCacheTests.ReadPoolThreadFilter.class })
public class DirectIOVectorCacheTests extends KNNTestCase {

    /** The read pool is never shut down by design; its daemon threads outlive any one suite. */
    public static final class ReadPoolThreadFilter implements ThreadFilter {
        @Override
        public boolean reject(final Thread thread) {
            return thread.getName().startsWith(DirectIOReadPool.THREAD_NAME_PREFIX);
        }
    }

    private static final int DIMENSION = 64;
    private static final int VECTOR_BYTES = DIMENSION * Float.BYTES;
    private static final int HEADER_LENGTH = 44;
    private static final int VECTOR_COUNT = 400;

    /**
     * 64 KiB rather than the shipped 32 MiB, so that a budget of one slab is 256 of this file's vectors and
     * eviction is reached inside a unit test. Everything else is the shipped path: real slabs, the real
     * index, the real input.
     */
    private static final int TEST_SLAB_BYTES = 64 * 1024;

    /** Slots a slab of {@link #TEST_SLAB_BYTES} gives this file's vectors. */
    private static final int SLOTS_PER_SLAB = TEST_SLAB_BYTES / VECTOR_BYTES;

    @Override
    public void tearDown() throws Exception {
        DirectIOVectorCache.resetForTesting();
        DirectIOReadPool.resetForTesting();
        DirectIOVectorIndexInput.STATS.reset();
        super.tearDown();
    }

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
        when(clusterService.getClusterSettings()).thenReturn(new ClusterSettings(settings, registered));
        KNNSettings.state().setClusterService(clusterService);
    }

    /**
     * The cache on at {@code slabs} test slabs and the read-ahead ring off, so the burst table is the only
     * channel. The slab size is set here rather than in {@code setUp} because {@code resetForTesting} puts it
     * back to the shipped 32 MiB, and the store refuses a change while it holds slabs.
     */
    private void cacheOn(final int slabs) {
        final long maxBytes = (long) slabs * TEST_SLAB_BYTES;
        withNodeSettings(
            Settings.builder()
                .put(KNNSettings.KNN_DIRECT_IO_RESCORE_CACHE_MAX_BYTES, maxBytes + "b")
                // Span fill off, so the counts here are the requested fills' and one read admits one vector.
                // It is on by default and has a suite of its own, DirectIOSpanFillTests, where the admitted
                // set is computed from the span's geometry; what these tests pin is the accounting, the key
                // and the resize, and a span admitting fifteen neighbours would hide all three.
                .put(KNNSettings.KNN_DIRECT_IO_RESCORE_CACHE_SPAN_FILL, false)
                .put(KNNSettings.KNN_DIRECT_IO_READAHEAD_WINDOW, 0)
                .build()
        );
        DirectIOVectorCache.resetForTesting();
        DirectIOVectorCache.setSlabBytesForTesting(TEST_SLAB_BYTES);
        DirectIOVectorCache.configure(maxBytes);
    }

    @SneakyThrows
    private float[][] randomVectors(final long seed) {
        final Random random = new Random(seed);
        final float[][] vectors = new float[VECTOR_COUNT][DIMENSION];
        for (final float[] vector : vectors) {
            for (int i = 0; i < DIMENSION; i++) {
                vector[i] = random.nextFloat();
            }
        }
        return vectors;
    }

    @SneakyThrows
    private Path writeVectorFile(final Path dir, final String name, final float[][] vectors) {
        final Path path = dir.resolve(name);
        final ByteBuffer buffer = ByteBuffer.allocate(HEADER_LENGTH + vectors.length * VECTOR_BYTES + 16).order(ByteOrder.LITTLE_ENDIAN);
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
    private void readVector(final DirectIOVectorIndexInput input, final int ord, final float[] into) {
        final long at = HEADER_LENGTH + (long) ord * VECTOR_BYTES;
        input.prefetch(at, VECTOR_BYTES);
        input.seek(at);
        input.readFloats(into, 0, DIMENSION);
    }

    // -----------------------------------------------------------------------------------------------

    /**
     * Hit, miss and admission accounting, and what {@code bytes} and {@code entries} mean.
     *
     * <p>A vector is counted <b>once</b>, at the lookup in {@code readFloats} that serves or misses it. The
     * lookup in {@code prefetch} decides whether the range joins the burst and counts nothing, which is the
     * M2 change from phase 15's two counts per vector: it makes {@code hits / (hits + misses)} a hit rate
     * whose absolute counts are also the vectors served, and so directly comparable with the node figures.
     */
    @SneakyThrows
    public void testHitAndMissAccountingAndWhatIsHeld() {
        assumeDirectIOWorksHere();
        final float[][] vectors = randomVectors(1L);
        final Path path = writeVectorFile(createTempDir(), "a.vec", vectors);

        // Two slabs, far more than this test's working set, so nothing evicts and the counts are exact.
        cacheOn(2);
        final float[] scratch = new float[DIMENSION];

        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(input);
            assertEquals(1, DirectIOVectorCache.lookupsPerVector());

            readVector(input, 0, scratch);
            assertArrayEquals(vectors[0], scratch, 0.0f);
            assertEquals("cold: the one lookup that serves it misses", 0L, DirectIOVectorCache.hits());
            assertEquals(1L, DirectIOVectorCache.misses());
            assertEquals("and the vector is now held", 1L, DirectIOVectorCache.entries());
            assertEquals("bytes is the slot the vector occupies", VECTOR_BYTES, DirectIOVectorCache.bytes());
            assertEquals("one slab was taken for it", 1L, DirectIOVectorCache.slabs());
            assertEquals((long) TEST_SLAB_BYTES, DirectIOVectorCache.slabBytes());

            readVector(input, 0, scratch);
            assertArrayEquals("a hit must return the same bytes", vectors[0], scratch, 0.0f);
            assertEquals("warm: one hit", 1L, DirectIOVectorCache.hits());
            assertEquals(1L, DirectIOVectorCache.misses());
            assertEquals("and nothing new is held", 1L, DirectIOVectorCache.entries());
            assertEquals("re-reading must not double-charge", VECTOR_BYTES, DirectIOVectorCache.bytes());

            readVector(input, 1, scratch);
            readVector(input, 2, scratch);
            assertEquals(3L, DirectIOVectorCache.entries());
            assertEquals("bytes is exactly entries x the vector length", 3L * VECTOR_BYTES, DirectIOVectorCache.bytes());
            assertEquals("and with slots to spare nothing is evicted", 0L, DirectIOVectorCache.evictions());
            assertEquals("and no slab was reclaimed", 0L, DirectIOVectorCache.reclaimedSlabs());
        }
    }

    /**
     * Eviction, and the two things it is there to guarantee: the store never holds more than its slab target,
     * and a read is correct whether or not its vector survived.
     *
     * <p>The target is whole slabs, so {@code max_bytes} bounds the <em>slabs</em> and the live vectors'
     * bytes are at most that. Nothing is read twice here, so no vector earns a second chance and
     * {@code readmissions} must stay at zero: that is what makes the FIFO's behaviour visible rather than
     * inferred.
     */
    @SneakyThrows
    public void testEvictionStaysInsideTheSlabTarget() {
        assumeDirectIOWorksHere();
        final float[][] vectors = randomVectors(7L);
        final Path path = writeVectorFile(createTempDir(), "evict.vec", vectors);

        cacheOn(1);
        final long budget = TEST_SLAB_BYTES;
        final float[] scratch = new float[DIMENSION];

        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(input);
            for (int ord = 0; ord < VECTOR_COUNT; ord++) {
                readVector(input, ord, scratch);
                assertArrayEquals("eviction must never change a byte", vectors[ord], scratch, 0.0f);
            }
            assertTrue(
                VECTOR_COUNT + " distinct vectors into a " + SLOTS_PER_SLAB + "-slot slab must evict",
                DirectIOVectorCache.evictions() > 0
            );
            assertEquals("by reclaiming the one slab it has", 1L, DirectIOVectorCache.slabs());
            assertEquals("nothing was read twice, so nothing earned a second chance", 0L, DirectIOVectorCache.readmissions());
            assertTrue(
                "the live vectors must stay inside max_bytes, which is the point: " + DirectIOVectorCache.bytes() + " > " + budget,
                DirectIOVectorCache.bytes() <= budget
            );
            assertEquals(
                "bytes is always entries x the vector length",
                DirectIOVectorCache.entries() * VECTOR_BYTES,
                DirectIOVectorCache.bytes()
            );
            assertTrue("and entries cannot exceed the slab's slots", DirectIOVectorCache.entries() <= SLOTS_PER_SLAB);

            // Re-reading an evicted vector is a miss that reaches the device and repopulates: correctness
            // does not depend on an entry still being there.
            final long missesBefore = DirectIOVectorCache.misses();
            readVector(input, 0, scratch);
            assertArrayEquals(vectors[0], scratch, 0.0f);
            assertTrue("an evicted vector must be re-read, not mis-served", DirectIOVectorCache.misses() > missesBefore);
        }
    }

    /**
     * Two inputs over the same file must share entries. This is not a nicety: the codec seam opens a
     * {@code .vec} once per reader, so if file identity were the {@code Handle}'s object identity, two
     * readers of one segment would never share and the 40.3%-at-16-MB cross-query reuse the gate measured
     * would be unreachable by construction.
     */
    @SneakyThrows
    public void testTwoInputsOverOneFileShareEntries() {
        assumeDirectIOWorksHere();
        final float[][] vectors = randomVectors(2L);
        final Path path = writeVectorFile(createTempDir(), "shared.vec", vectors);
        cacheOn(2);
        final float[] scratch = new float[DIMENSION];

        try (DirectIOVectorIndexInput first = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(first);
            readVector(first, 7, scratch);
            assertArrayEquals(vectors[7], scratch, 0.0f);
        }
        final long hitsAfterPopulate = DirectIOVectorCache.hits();
        final long missesAfterPopulate = DirectIOVectorCache.misses();
        assertEquals(1L, DirectIOVectorCache.entries());

        // A SECOND open of the same path, which is a different Handle over a different FileChannel -- which
        // is exactly why identity would not do as the key's file part.
        try (DirectIOVectorIndexInput second = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(second);
            readVector(second, 7, scratch);
            assertArrayEquals("the second reader must get the right bytes", vectors[7], scratch, 0.0f);
        }
        assertEquals("the second reader's lookup must hit", hitsAfterPopulate + 1, DirectIOVectorCache.hits());
        assertEquals("and neither may miss", missesAfterPopulate, DirectIOVectorCache.misses());
        assertEquals("sharing means one entry, not two", 1L, DirectIOVectorCache.entries());
    }

    /**
     * Two different files must never collide, even at the same offset with <b>byte-identical content</b> —
     * which is the case that would hide the bug, because the read would return the right answer for the
     * wrong reason. Asserted on the counters: the second file's lookups must <em>miss</em>.
     */
    @SneakyThrows
    public void testTwoFilesNeverCollide() {
        assumeDirectIOWorksHere();
        final Path dir = createTempDir();
        final float[][] vectors = randomVectors(3L);
        final Path a = writeVectorFile(dir, "a.vec", vectors);
        final Path b = writeVectorFile(dir, "b.vec", vectors);
        assertEquals("the two files are byte identical, which is the hard case", Files.size(a), Files.size(b));
        assertArrayEquals(Files.readAllBytes(a), Files.readAllBytes(b));

        cacheOn(2);
        final float[] scratch = new float[DIMENSION];

        try (DirectIOVectorIndexInput inputA = DirectIOVectorIndexInput.open(a)) {
            assertNotNull(inputA);
            readVector(inputA, 11, scratch);
        }
        final long missesAfterA = DirectIOVectorCache.misses();
        assertEquals(1L, DirectIOVectorCache.entries());

        try (DirectIOVectorIndexInput inputB = DirectIOVectorIndexInput.open(b)) {
            assertNotNull(inputB);
            readVector(inputB, 11, scratch);
            assertArrayEquals(vectors[11], scratch, 0.0f);
        }
        assertEquals("the other file's entry must not be served", missesAfterA + 1, DirectIOVectorCache.misses());
        assertEquals("each file gets its own entry", 2L, DirectIOVectorCache.entries());
    }

    /**
     * Cache off — the shipped default — must be a path that does not exist. No entry, no counter, and in
     * particular no key allocated: {@code isOn()} is one volatile read and both call sites are behind it.
     */
    @SneakyThrows
    public void testCacheOffIsAZeroOverheadPath() {
        assumeDirectIOWorksHere();
        final float[][] vectors = randomVectors(4L);
        final Path path = writeVectorFile(createTempDir(), "off.vec", vectors);

        withNodeSettings(Settings.builder().put(KNNSettings.KNN_DIRECT_IO_READAHEAD_WINDOW, 0).build());
        DirectIOVectorCache.resetForTesting();
        assertFalse("0b is the shipped default", DirectIOVectorCache.isOn());
        assertEquals(0L, DirectIOVectorCache.maxBytes());

        final float[] scratch = new float[DIMENSION];
        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(input);
            for (int ord = 0; ord < 20; ord++) {
                readVector(input, ord, scratch);
                assertArrayEquals("off must not change a byte", vectors[ord], scratch, 0.0f);
            }
        }
        assertFalse("and the burst must not have turned it on", DirectIOVectorCache.isOn());
        assertEquals(0L, DirectIOVectorCache.hits());
        assertEquals("not even a miss: a lookup that cannot succeed is not made", 0L, DirectIOVectorCache.misses());
        assertEquals(0L, DirectIOVectorCache.entries());
        assertEquals(0L, DirectIOVectorCache.bytes());
    }

    /**
     * A live change is applied at the next prefetch burst, and a change of size <b>resizes</b> rather than
     * rebuilding: growing keeps every vector already held and lets the next miss take a slab. That is the M2
     * change from phase 15, whose Guava cache had to be rebuilt because {@code maximumWeight} is fixed at
     * {@code build()}, so every retune cost the whole cache and a re-warm. Asserted rather than assumed,
     * because it is the difference an operator feels.
     */
    @SneakyThrows
    public void testALiveSizeChangeResizesAndKeepsTheEntries() {
        assumeDirectIOWorksHere();
        final float[][] vectors = randomVectors(5L);
        final Path path = writeVectorFile(createTempDir(), "live.vec", vectors);
        final float[] scratch = new float[DIMENSION];

        cacheOn(1);
        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(input);
            readVector(input, 1, scratch);
            assertEquals(1L, DirectIOVectorCache.entries());
            assertTrue(DirectIOVectorCache.bytes() > 0);

            // The operator raises it. applyConfiguredSize runs at the next burst boundary, which the next
            // prefetch after a read establishes.
            withNodeSettings(
                Settings.builder()
                    .put(KNNSettings.KNN_DIRECT_IO_RESCORE_CACHE_MAX_BYTES, (4 * TEST_SLAB_BYTES) + "b")
                    .put(KNNSettings.KNN_DIRECT_IO_RESCORE_CACHE_SPAN_FILL, false)
                    .put(KNNSettings.KNN_DIRECT_IO_READAHEAD_WINDOW, 0)
                    .build()
            );
            readVector(input, 2, scratch);
            assertEquals("the new target is live", 4L * TEST_SLAB_BYTES, DirectIOVectorCache.maxBytes());
            assertEquals("growing keeps what was held and adds this burst's vector", 2L, DirectIOVectorCache.entries());
            assertArrayEquals(vectors[2], scratch, 0.0f);
            final float[] stillThere = new float[DIMENSION];
            assertTrue(
                "and the vector from before the change is still a hit",
                DirectIOVectorCache.getFloats(
                    DirectIOVectorCache.fileId(path, Files.size(path)),
                    HEADER_LENGTH + (long) VECTOR_BYTES,
                    VECTOR_BYTES,
                    stillThere,
                    0
                )
            );
            assertArrayEquals(vectors[1], stillThere, 0.0f);

            // And off again, live.
            withNodeSettings(
                Settings.builder()
                    .put(KNNSettings.KNN_DIRECT_IO_RESCORE_CACHE_MAX_BYTES, "0b")
                    .put(KNNSettings.KNN_DIRECT_IO_RESCORE_CACHE_SPAN_FILL, false)
                    .put(KNNSettings.KNN_DIRECT_IO_READAHEAD_WINDOW, 0)
                    .build()
            );
            readVector(input, 3, scratch);
            assertArrayEquals("turning it off must not change a byte either", vectors[3], scratch, 0.0f);
            assertFalse(DirectIOVectorCache.isOn());
            assertEquals(0L, DirectIOVectorCache.entries());
            assertEquals(0L, DirectIOVectorCache.bytes());
        }
    }

    /**
     * A cache hit must not cost a device read, which is the entire point: on the node the expected effect is
     * {@code .vec} reads per 5,000 queries falling from 707k to ~420k. Asserted from inside the process on
     * the counters that attribute reads to this path.
     */
    @SneakyThrows
    public void testAHitCostsNoDeviceRead() {
        assumeDirectIOWorksHere();
        final float[][] vectors = randomVectors(6L);
        final Path path = writeVectorFile(createTempDir(), "iops.vec", vectors);
        cacheOn(2);
        final float[] scratch = new float[DIMENSION];

        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(input);
            readVector(input, 5, scratch);
            final long stagedAfterCold = DirectIOVectorIndexInput.STATS.stagedReads();
            final long blockingAfterCold = DirectIOVectorIndexInput.STATS.blockingReads();
            assertTrue("the cold read had to reach the device", stagedAfterCold + blockingAfterCold > 0);

            for (int repeat = 0; repeat < 10; repeat++) {
                readVector(input, 5, scratch);
                assertArrayEquals(vectors[5], scratch, 0.0f);
            }
            assertEquals("ten warm reads, no staged read", stagedAfterCold, DirectIOVectorIndexInput.STATS.stagedReads());
            assertEquals("and no blocking read", blockingAfterCold, DirectIOVectorIndexInput.STATS.blockingReads());
            assertEquals(
                "the prefetches were not even declined -- they were satisfied",
                0L,
                DirectIOVectorIndexInput.STATS.stageDeclined()
            );
        }
    }

    /** The setting itself: key, default, and that it can be changed on a live node. */
    public void testTheSettingIsOffByDefaultAndDynamic() {
        assertEquals("knn.direct_io.rescore.cache.max_bytes", KNNSettings.KNN_DIRECT_IO_RESCORE_CACHE_MAX_BYTES_SETTING.getKey());
        assertEquals(
            "off is the shipped default; the node arm decides the value",
            0L,
            KNNSettings.KNN_DIRECT_IO_RESCORE_CACHE_MAX_BYTES_DEFAULT_VALUE.getBytes()
        );
        assertTrue(KNNSettings.KNN_DIRECT_IO_RESCORE_CACHE_MAX_BYTES_SETTING.isDynamic());
    }
}

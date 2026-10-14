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

    /** The cache on at {@code maxBytes} and the read-ahead ring off, so the burst table is the only channel. */
    private void cacheOn(final long maxBytes) {
        withNodeSettings(
            Settings.builder()
                .put(KNNSettings.KNN_DIRECT_IO_RESCORE_CACHE_MAX_BYTES, maxBytes + "b")
                .put(KNNSettings.KNN_DIRECT_IO_READAHEAD_WINDOW, 0)
                .build()
        );
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
     * Hit, miss and eviction accounting, and the weigher that makes {@code max_bytes} mean something.
     *
     * <p>Two lookups happen per vector — one in {@code prefetch}, one in {@code readFloats} — so the counts
     * are asserted in pairs. That is documented in {@link DirectIOVectorCache#lookupsPerVector()} rather
     * than hidden, because an operator comparing the absolute numbers against a query count would otherwise
     * conclude the cache was being consulted twice as often as it should be. It is, and on purpose.
     */
    @SneakyThrows
    public void testHitAndMissAccountingAndTheWeigher() {
        assumeDirectIOWorksHere();
        final float[][] vectors = randomVectors(1L);
        final Path path = writeVectorFile(createTempDir(), "a.vec", vectors);

        // A budget far larger than this test's working set, so nothing evicts and the counts are exact.
        final int entryWeight = VECTOR_BYTES + DirectIOVectorCache.ENTRY_OVERHEAD_BYTES;
        cacheOn(1 << 20);
        final float[] scratch = new float[DIMENSION];

        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(input);
            assertEquals(2, DirectIOVectorCache.lookupsPerVector());

            readVector(input, 0, scratch);
            assertArrayEquals(vectors[0], scratch, 0.0f);
            assertEquals("cold: both lookups miss", 0L, DirectIOVectorCache.hits());
            assertEquals(2L, DirectIOVectorCache.misses());
            assertEquals("and the vector is now held", 1L, DirectIOVectorCache.entries());
            assertEquals("weighed as its bytes plus the entry's own cost", entryWeight, DirectIOVectorCache.bytes());

            readVector(input, 0, scratch);
            assertArrayEquals("a hit must return the same bytes", vectors[0], scratch, 0.0f);
            assertEquals("warm: both lookups hit", 2L, DirectIOVectorCache.hits());
            assertEquals(2L, DirectIOVectorCache.misses());
            assertEquals("and nothing new is held", 1L, DirectIOVectorCache.entries());
            assertEquals("re-reading must not double-charge", entryWeight, DirectIOVectorCache.bytes());

            readVector(input, 1, scratch);
            readVector(input, 2, scratch);
            assertEquals(3L, DirectIOVectorCache.entries());
            assertEquals("the weigher is exactly bytes + overhead per entry", 3L * entryWeight, DirectIOVectorCache.bytes());
            assertEquals("and with budget to spare nothing is evicted", 0L, DirectIOVectorCache.evictions());
        }
    }

    /**
     * Eviction, and the thing it is there to guarantee: the charged weight never exceeds {@code max_bytes}.
     *
     * <p><b>Not asserted as "N entries fit in N entries' worth of budget", because that is false.</b>
     * MEASURED while writing this test: a budget of three entries' weight holds <b>zero</b> entries. Guava's
     * {@code LocalCache} splits {@code maximumWeight} across {@code concurrencyLevel} segments and evicts
     * against the per-segment share, so an entry heavier than {@code max_bytes / concurrencyLevel} is
     * inserted and immediately evicted again. At the sizes the reuse gate measured (16 MB and up) that
     * division is immaterial; at a handful of entries it is the whole behaviour. So what is pinned here is
     * the invariant that actually matters to a node — the bound holds, and the reads stay correct across
     * eviction — rather than a per-entry count that depends on Guava's hashing.
     */
    @SneakyThrows
    public void testEvictionKeepsTheChargedWeightUnderTheBudget() {
        assumeDirectIOWorksHere();
        final float[][] vectors = randomVectors(7L);
        final Path path = writeVectorFile(createTempDir(), "evict.vec", vectors);

        final int entryWeight = VECTOR_BYTES + DirectIOVectorCache.ENTRY_OVERHEAD_BYTES;
        final long budget = 16L * entryWeight;
        cacheOn(budget);
        final float[] scratch = new float[DIMENSION];

        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(input);
            for (int ord = 0; ord < 120; ord++) {
                readVector(input, ord, scratch);
                assertArrayEquals("eviction must never change a byte", vectors[ord], scratch, 0.0f);
            }
            assertTrue("120 distinct vectors into a 16-entry budget must evict", DirectIOVectorCache.evictions() > 0);
            assertTrue(
                "and the charged weight must stay under max_bytes, which is the point: " + DirectIOVectorCache.bytes() + " > " + budget,
                DirectIOVectorCache.bytes() <= budget
            );
            assertEquals(
                "the weight is always entries x entryWeight, so the weigher and the listener agree",
                DirectIOVectorCache.entries() * entryWeight,
                DirectIOVectorCache.bytes()
            );

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
        cacheOn(1 << 20);
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
        assertEquals("both of the second reader's lookups must hit", hitsAfterPopulate + 2, DirectIOVectorCache.hits());
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

        cacheOn(1 << 20);
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
        assertEquals("the other file's entry must not be served", missesAfterA + 2, DirectIOVectorCache.misses());
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
     * A live change is applied at the next prefetch burst, and a change of size <b>rebuilds</b> rather than
     * resizing — Guava fixes {@code maximumWeight} at build time. So the entries go, which is a real cost of
     * retuning this knob and is why it is asserted rather than assumed.
     */
    @SneakyThrows
    public void testALiveSizeChangeRebuildsAndDropsTheEntries() {
        assumeDirectIOWorksHere();
        final float[][] vectors = randomVectors(5L);
        final Path path = writeVectorFile(createTempDir(), "live.vec", vectors);
        final float[] scratch = new float[DIMENSION];

        cacheOn(1 << 20);
        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(input);
            readVector(input, 1, scratch);
            assertEquals(1L, DirectIOVectorCache.entries());
            assertTrue(DirectIOVectorCache.bytes() > 0);

            // The operator raises it. applyConfiguredSize runs at the next burst boundary, which the next
            // prefetch after a read establishes.
            withNodeSettings(
                Settings.builder()
                    .put(KNNSettings.KNN_DIRECT_IO_RESCORE_CACHE_MAX_BYTES, "4mb")
                    .put(KNNSettings.KNN_DIRECT_IO_READAHEAD_WINDOW, 0)
                    .build()
            );
            readVector(input, 2, scratch);
            assertEquals("the new budget is live", 4L * 1024 * 1024, DirectIOVectorCache.maxBytes());
            assertEquals("a rebuild drops what was held, then re-populates from this burst", 1L, DirectIOVectorCache.entries());
            assertArrayEquals(vectors[2], scratch, 0.0f);

            // And off again, live.
            withNodeSettings(
                Settings.builder()
                    .put(KNNSettings.KNN_DIRECT_IO_RESCORE_CACHE_MAX_BYTES, "0b")
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
        cacheOn(1 << 20);
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

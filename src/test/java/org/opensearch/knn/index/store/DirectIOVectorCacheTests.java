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
 * The phase 15 node-wide vector cache (D4), tested on its own API.
 *
 * <p><b>What changed in I4, and why these tests look different.</b> D1 moved the vector lookup out of
 * {@link DirectIOVectorIndexInput} and up into the scorer, where the key is the segment rather than the
 * file, so this class is no longer on any read path: nothing in {@code prefetch} or {@code readFloats}
 * consults it. The six tests that drove it through a real input were therefore testing a path that no longer
 * exists, and they were replaced by tests of the same properties on this class's own API: honest
 * accounting, a weigher that really bounds the heap, a key that shares between two readers of one file and
 * never collides between two files, and a size change that takes effect. The one input-driven test kept is
 * the one that now asserts the removal itself: a read through the input must not touch this cache at all.
 *
 * <p>The class stays one phase, with its setting key, because a persisted
 * {@code knn.direct_io.rescore.cache.max_bytes} must keep resolving on a live node; from I4 that setting is
 * the slab cache's budget. See {@code SlabVectorCache} and {@code VectorCacheBatchScorer}.
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

    /** {@code bytes} of {@code vector}, as a staged read would hand them over. */
    private ByteBuffer vectorBytes(final float[] vector) {
        final ByteBuffer buffer = ByteBuffer.allocate(vector.length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (final float value : vector) {
            buffer.putFloat(value);
        }
        return buffer;
    }

    // -----------------------------------------------------------------------------------------------

    /** Hit, miss and eviction accounting, and the weigher that makes {@code max_bytes} mean something. */
    public void testHitAndMissAccountingAndTheWeigher() {
        final float[][] vectors = randomVectors(1L);
        final int entryWeight = VECTOR_BYTES + DirectIOVectorCache.ENTRY_OVERHEAD_BYTES;
        DirectIOVectorCache.configure(1 << 20);
        assertTrue(DirectIOVectorCache.isOn());

        assertNull("cold", DirectIOVectorCache.get("a", 0, VECTOR_BYTES));
        assertEquals(0L, DirectIOVectorCache.hits());
        assertEquals(1L, DirectIOVectorCache.misses());

        DirectIOVectorCache.put("a", 0, vectorBytes(vectors[0]), 0, VECTOR_BYTES);
        assertEquals("the vector is now held", 1L, DirectIOVectorCache.entries());
        assertEquals("weighed as its bytes plus the entry's own cost", entryWeight, DirectIOVectorCache.bytes());

        final byte[] served = DirectIOVectorCache.get("a", 0, VECTOR_BYTES);
        assertNotNull("warm", served);
        final ByteBuffer expected = vectorBytes(vectors[0]);
        assertArrayEquals("a hit must return the same bytes", expected.array(), served);
        assertEquals(1L, DirectIOVectorCache.hits());

        DirectIOVectorCache.put("a", VECTOR_BYTES, vectorBytes(vectors[1]), 0, VECTOR_BYTES);
        DirectIOVectorCache.put("a", 2L * VECTOR_BYTES, vectorBytes(vectors[2]), 0, VECTOR_BYTES);
        assertEquals(3L, DirectIOVectorCache.entries());
        assertEquals("the weigher is exactly bytes + overhead per entry", 3L * entryWeight, DirectIOVectorCache.bytes());
        assertEquals("and with budget to spare nothing is evicted", 0L, DirectIOVectorCache.evictions());
    }

    /**
     * Eviction, and the thing it is there to guarantee: the charged weight never exceeds {@code max_bytes}.
     *
     * <p><b>Not asserted as "N entries fit in N entries' worth of budget", because that is false.</b>
     * MEASURED while writing the phase 15 version of this test: a budget of three entries' weight holds
     * <b>zero</b> entries, because Guava's {@code LocalCache} splits {@code maximumWeight} across
     * {@code concurrencyLevel} segments and evicts against the per-segment share.
     */
    public void testEvictionKeepsTheChargedWeightUnderTheBudget() {
        final float[][] vectors = randomVectors(7L);
        final int entryWeight = VECTOR_BYTES + DirectIOVectorCache.ENTRY_OVERHEAD_BYTES;
        final long budget = 40L * entryWeight;
        DirectIOVectorCache.configure(budget);

        for (int i = 0; i < vectors.length; i++) {
            DirectIOVectorCache.put("a", (long) i * VECTOR_BYTES, vectorBytes(vectors[i]), 0, VECTOR_BYTES);
            assertTrue("the bound must hold at every step", DirectIOVectorCache.bytes() <= budget);
        }
        assertTrue("400 entries into 40 entries' worth of budget has to evict", DirectIOVectorCache.evictions() > 0);
        assertTrue("and something must still be held", DirectIOVectorCache.entries() > 0);
    }

    /**
     * The key: two readers of one file share an entry, and two different files never collide even at the
     * same offset with <b>byte-identical content</b>, which is the case that would hide the bug because the
     * read would return the right answer for the wrong reason.
     */
    @SneakyThrows
    public void testTheKeySharesWithinAFileAndNeverCollidesAcrossFiles() {
        final Path dir = createTempDir();
        final float[][] vectors = randomVectors(3L);
        final Path a = writeVectorFile(dir, "a.vec", vectors);
        final Path b = writeVectorFile(dir, "b.vec", vectors);
        assertEquals("the two files are byte identical, which is the hard case", Files.size(a), Files.size(b));
        assertArrayEquals(Files.readAllBytes(a), Files.readAllBytes(b));

        final String idA = DirectIOVectorCache.fileId(a, Files.size(a));
        final String idB = DirectIOVectorCache.fileId(b, Files.size(b));
        assertEquals(
            "a second open of one file must produce the same id, so the two opens share",
            idA,
            DirectIOVectorCache.fileId(a, Files.size(a))
        );
        assertNotEquals(idA, idB);

        DirectIOVectorCache.configure(1 << 20);
        DirectIOVectorCache.put(idA, 64L, vectorBytes(vectors[11]), 0, VECTOR_BYTES);
        assertNotNull("the same file at the same offset hits", DirectIOVectorCache.get(idA, 64L, VECTOR_BYTES));
        assertNull("the other file's entry must not be served", DirectIOVectorCache.get(idB, 64L, VECTOR_BYTES));
        assertNull("nor another offset of the same file", DirectIOVectorCache.get(idA, 128L, VECTOR_BYTES));
    }

    /**
     * The removal itself: a read through the input must not touch this cache, whatever its budget. That is
     * D1, and it is what the scorer-level lookup replaced.
     */
    @SneakyThrows
    public void testTheReadPathNoLongerConsultsThisCache() {
        assumeDirectIOWorksHere();
        final float[][] vectors = randomVectors(4L);
        final Path path = writeVectorFile(createTempDir(), "off.vec", vectors);

        withNodeSettings(Settings.builder().put(KNNSettings.KNN_DIRECT_IO_READAHEAD_WINDOW, 0).build());
        // On, deliberately: the point is that the read path does not consult it even when it could.
        DirectIOVectorCache.configure(1 << 20);

        final float[] scratch = new float[DIMENSION];
        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(input);
            for (int ord = 0; ord < 20; ord++) {
                readVector(input, ord, scratch);
                assertArrayEquals("every read must be exact", vectors[ord], scratch, 0.0f);
            }
            for (int ord = 0; ord < 20; ord++) {
                readVector(input, ord, scratch);
                assertArrayEquals(vectors[ord], scratch, 0.0f);
            }
        }
        assertEquals("no lookup", 0L, DirectIOVectorCache.hits());
        assertEquals("not even a miss", 0L, DirectIOVectorCache.misses());
        assertEquals("and no fill", 0L, DirectIOVectorCache.entries());
        assertEquals(0L, DirectIOVectorCache.bytes());
    }

    /** A size change rebuilds, which is the behaviour the slab cache deliberately does not copy. */
    public void testALiveSizeChangeRebuildsAndDropsTheEntries() {
        final float[][] vectors = randomVectors(5L);
        DirectIOVectorCache.configure(1 << 20);
        DirectIOVectorCache.put("a", 0L, vectorBytes(vectors[1]), 0, VECTOR_BYTES);
        assertEquals(1L, DirectIOVectorCache.entries());
        assertTrue(DirectIOVectorCache.bytes() > 0);

        DirectIOVectorCache.configure(4L * 1024 * 1024);
        assertEquals("the new budget is live", 4L * 1024 * 1024, DirectIOVectorCache.maxBytes());
        assertEquals("a rebuild drops what was held", 0L, DirectIOVectorCache.entries());

        DirectIOVectorCache.configure(0L);
        assertFalse(DirectIOVectorCache.isOn());
        assertEquals(0L, DirectIOVectorCache.entries());
        assertEquals(0L, DirectIOVectorCache.bytes());
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

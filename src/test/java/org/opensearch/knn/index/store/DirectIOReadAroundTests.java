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
 * O5: a staged miss read is widened so the vectors around the one being fetched are admitted with it, while
 * the whole file would still fit in the room the store's target leaves.
 *
 * <p><b>Why it exists (MEASURED on the EBS node, ref/RESULTS-N.md section 5).</b> N1 made each miss exactly
 * one 12 KB read, which is what took the device traffic on M4's fixture from 4.05x the file to 1.53x. What it
 * also removed was the first build's <em>accidental</em> read-around: that build staged whole 65 KB coalesced
 * groups, and the neighbours inside them were admitted for free. On dio-1m, coverage after 20,000 queries
 * fell from 97.1% to 93.4% and the tail got worse (p99 30.5 against 25.8 ms) even while the median and the
 * throughput improved. O5 buys that coverage back deliberately and with a bound.
 *
 * <p><b>Three gates, and the second is the one that makes this safe.</b> The width setting must not be
 * {@code 0b}; the whole file must still fit in the room the target leaves, which is what makes it impossible
 * for a read-around to evict a live vector, because if the file fits nothing has to be displaced to hold it;
 * and span fill must be on, because the widened bytes are admitted by span fill and a wider read nobody
 * harvests is device traffic for nothing. Each gate has a test here, because a gate that silently stopped
 * working would show up as device traffic rather than as a wrong answer.
 */
@ThreadLeakFilters(defaultFilters = true, filters = { DirectIOReadAroundTests.ReadPoolThreadFilter.class })
public class DirectIOReadAroundTests extends KNNTestCase {

    /** The read pool is never shut down by design; its daemon threads outlive any one suite. */
    public static final class ReadPoolThreadFilter implements ThreadFilter {
        @Override
        public boolean reject(final Thread thread) {
            return thread.getName().startsWith(DirectIOReadPool.THREAD_NAME_PREFIX);
        }
    }

    private static final int DIMENSION = 128;
    private static final int VECTOR_BYTES = DIMENSION * Float.BYTES;
    /** Not a multiple of the vector length and not a multiple of a block, which is the awkward real case. */
    private static final int HEADER_LENGTH = 44;
    private static final int FOOTER_LENGTH = 16;
    private static final int VECTOR_COUNT = 512;
    /** 64 KiB, so a slab holds 128 of these vectors and a handful of slabs holds the whole file. */
    private static final int TEST_SLAB_BYTES = 64 * 1024;
    /** The shipped default, and wide enough to hold 128 vectors. */
    private static final String READ_AROUND = "64kb";

    private Path path;
    private float[][] vectors;

    @Override
    public void tearDown() throws Exception {
        DirectIOVectorCache.resetForTesting();
        DirectIOReadPool.resetForTesting();
        DirectIOVectorIndexInput.STATS.reset();
        super.tearDown();
    }

    // -----------------------------------------------------------------------------------------------
    // The widened read, and what it admits
    // -----------------------------------------------------------------------------------------------

    /**
     * One miss, one read, and the neighbours come with it. The read is still <b>one</b> staged read: O5
     * changes a read's size and not how many there are.
     */
    @SneakyThrows
    public void testAWidenedReadAdmitsItsNeighbours() {
        assumeDirectIOWorksHere();
        writeFixture(20L);
        // Eight slabs of 64 KiB = 512 KiB against a 262 KiB file, so the whole file fits in the room.
        configure(8, true, READ_AROUND);

        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(input);
            final float[] scratch = new float[DIMENSION];
            readVector(input, offsetOf(200), scratch);
            assertArrayEquals(vectors[200], scratch, 0.0f);

            assertEquals("the read was widened", 1L, DirectIOVectorIndexInput.STATS.readAroundReads());
            assertEquals("and it is still one read", 1L, DirectIOVectorIndexInput.STATS.stagedReads());
            final long bytes = DirectIOVectorIndexInput.STATS.stagedBytes();
            assertEquals("of the configured width", 64 * 1024L, bytes);

            // A 64 KiB window holds 128 whole vectors, less whatever the ends clip, and the one that was
            // asked for is in the middle of it. The store must hold a run around ordinal 200.
            final long held = DirectIOVectorCache.entries();
            assertTrue("a widened read should admit a run of neighbours, it admitted " + held, held > 64);
            assertTrue("and no more than the window can hold", held <= 64 * 1024 / VECTOR_BYTES);
            logger.info("O5: one widened read of {} B admitted {} vectors around ordinal 200", bytes, held);

            // Every one of them is its own vector's bytes, which is the thing a geometry slip would break.
            int checked = 0;
            for (int ordinal = 0; ordinal < VECTOR_COUNT; ordinal++) {
                if (DirectIOVectorCache.contains(fileId(), offsetOf(ordinal), VECTOR_BYTES)) {
                    final float[] fromStore = new float[DIMENSION];
                    assertTrue(DirectIOVectorCache.getFloats(fileId(), offsetOf(ordinal), VECTOR_BYTES, fromStore, 0));
                    assertArrayEquals("admitted vector " + ordinal, vectors[ordinal], fromStore, 0.0f);
                    checked++;
                }
            }
            assertEquals("every admitted entry was checked", held, (long) checked);
            DirectIOVectorCache.checkInvariantsForTesting();
        }
    }

    /**
     * A neighbour admitted by a read-around is served from the store afterwards, byte for byte, and costs no
     * second device read. That is the whole point: coverage the next query can hit.
     */
    @SneakyThrows
    public void testANeighbourIsThenServedFromTheStoreByteExact() {
        assumeDirectIOWorksHere();
        writeFixture(21L);
        configure(8, true, READ_AROUND);

        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(input);
            final float[] scratch = new float[DIMENSION];
            readVector(input, offsetOf(200), scratch);
            final long stagedAfterFirst = DirectIOVectorIndexInput.STATS.stagedReads();
            final long blockingAfterFirst = DirectIOVectorIndexInput.STATS.blockingReads();

            // A neighbour the first read admitted, read through the input exactly as a query would.
            int neighbour = -1;
            for (int ordinal = 201; ordinal < VECTOR_COUNT; ordinal++) {
                if (DirectIOVectorCache.contains(fileId(), offsetOf(ordinal), VECTOR_BYTES)) {
                    neighbour = ordinal;
                    break;
                }
            }
            assertTrue("the read-around must have admitted a neighbour above 200", neighbour > 200);

            final long hitsBefore = DirectIOVectorCache.hits();
            readVector(input, offsetOf(neighbour), scratch);
            assertArrayEquals("the neighbour's own bytes", vectors[neighbour], scratch, 0.0f);
            assertEquals("served from the store", hitsBefore + 1, DirectIOVectorCache.hits());
            assertEquals("with no further staged read", stagedAfterFirst, DirectIOVectorIndexInput.STATS.stagedReads());
            assertEquals("and no blocking read", blockingAfterFirst, DirectIOVectorIndexInput.STATS.blockingReads());
        }
    }

    // -----------------------------------------------------------------------------------------------
    // The three gates
    // -----------------------------------------------------------------------------------------------

    /** Gate 1: at {@code 0b} the read is exactly N1's, one aligned vector span. */
    @SneakyThrows
    public void testZeroBytesIsExactlyTheNarrowRead() {
        assumeDirectIOWorksHere();
        writeFixture(22L);
        configure(8, true, "0b");

        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(input);
            readVector(input, offsetOf(200), new float[DIMENSION]);

            assertEquals("no read was widened", 0L, DirectIOVectorIndexInput.STATS.readAroundReads());
            assertEquals(1L, DirectIOVectorIndexInput.STATS.stagedReads());
            final long bytes = DirectIOVectorIndexInput.STATS.stagedBytes();
            assertTrue("a narrow read is a block or two, it was " + bytes, bytes <= 2L * input.blockSize());
            // Span fill still admits the whole vectors the narrow span holds, and no more.
            assertTrue(DirectIOVectorCache.entries() > 0);
            assertTrue("but far fewer than a widened read's", DirectIOVectorCache.entries() < 64);
        }
    }

    /**
     * Gate 2: a file the target has no room for is not widened, however small the store's current contents.
     * This is the gate that makes a read-around unable to evict: if the whole file fits, nothing has to be
     * displaced to hold it.
     */
    @SneakyThrows
    public void testAFileThatWouldNotFitIsNotWidened() {
        assumeDirectIOWorksHere();
        writeFixture(23L);
        // One slab of 64 KiB against a 262 KiB file: the file cannot fit, even though the store is empty and
        // below its target.
        configure(1, true, READ_AROUND);
        assertTrue("the store is below its target, which is NOT the gate", DirectIOVectorCache.belowSlabTarget());

        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(input);
            readVector(input, offsetOf(200), new float[DIMENSION]);

            assertEquals("the file does not fit, so nothing is widened", 0L, DirectIOVectorIndexInput.STATS.readAroundReads());
            assertTrue(DirectIOVectorIndexInput.STATS.stagedBytes() <= 2L * input.blockSize());
        }
    }

    /** Gate 3: with span fill off nothing would harvest the widened bytes, so the read is not widened. */
    @SneakyThrows
    public void testSpanFillOffMeansNoReadAround() {
        assumeDirectIOWorksHere();
        writeFixture(24L);
        configure(8, false, READ_AROUND);

        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(input);
            final float[] scratch = new float[DIMENSION];
            readVector(input, offsetOf(200), scratch);
            assertArrayEquals(vectors[200], scratch, 0.0f);

            assertEquals("span fill is off, so nothing is widened", 0L, DirectIOVectorIndexInput.STATS.readAroundReads());
            assertTrue(DirectIOVectorIndexInput.STATS.stagedBytes() <= 2L * input.blockSize());
            assertEquals("and only the requested vector is admitted", 1L, DirectIOVectorCache.entries());
        }
    }

    /** With the store off there is no target to fit inside, so there is nothing to widen for. */
    @SneakyThrows
    public void testTheStoreOffMeansNoReadAround() {
        assumeDirectIOWorksHere();
        writeFixture(25L);
        configure(0, true, READ_AROUND);
        assertFalse(DirectIOVectorCache.isOn());

        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(input);
            final float[] scratch = new float[DIMENSION];
            readVector(input, offsetOf(200), scratch);
            assertArrayEquals(vectors[200], scratch, 0.0f);
            assertEquals(0L, DirectIOVectorIndexInput.STATS.readAroundReads());
            assertEquals(0L, DirectIOVectorCache.entries());
        }
    }

    /**
     * The widened window never reaches outside the file and never admits a partial vector or a byte of the
     * header, which is the same guarantee span fill has and is why read-around needs no geometry of its own:
     * it only changes how wide the staged range is, and the span-fill walk clips to the reading slice and
     * ends at the bytes the read returned. The first and the last vector of the file are the two places a
     * slip would show.
     */
    @SneakyThrows
    public void testTheEndsOfTheFileAreSafe() {
        assumeDirectIOWorksHere();
        writeFixture(26L);
        configure(8, true, READ_AROUND);

        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(input);
            final float[] scratch = new float[DIMENSION];
            // Ordinal 0 sits right after the 44-byte header, so a window widened backwards is clipped at the
            // start of the file and the header must not be admitted as a vector.
            readVector(input, offsetOf(0), scratch);
            assertArrayEquals(vectors[0], scratch, 0.0f);
            // And the last vector, where the window runs into the footer and past the end of the file.
            readVector(input, offsetOf(VECTOR_COUNT - 1), scratch);
            assertArrayEquals(vectors[VECTOR_COUNT - 1], scratch, 0.0f);

            // Whatever was admitted, every entry is the vector its key names: a header or footer byte that
            // had been admitted as a vector would fail here.
            for (int ordinal = 0; ordinal < VECTOR_COUNT; ordinal++) {
                if (DirectIOVectorCache.contains(fileId(), offsetOf(ordinal), VECTOR_BYTES)) {
                    final float[] fromStore = new float[DIMENSION];
                    assertTrue(DirectIOVectorCache.getFloats(fileId(), offsetOf(ordinal), VECTOR_BYTES, fromStore, 0));
                    assertArrayEquals("admitted vector " + ordinal, vectors[ordinal], fromStore, 0.0f);
                }
            }
            // And nothing outside the vector grid is in the store at all.
            assertFalse(DirectIOVectorCache.contains(fileId(), 0L, VECTOR_BYTES));
            assertFalse(DirectIOVectorCache.contains(fileId(), offsetOf(VECTOR_COUNT), VECTOR_BYTES));
            DirectIOVectorCache.checkInvariantsForTesting();
        }
    }

    // -----------------------------------------------------------------------------------------------

    private static long offsetOf(final int ordinal) {
        return HEADER_LENGTH + (long) ordinal * VECTOR_BYTES;
    }

    @SneakyThrows
    private String fileId() {
        return DirectIOVectorCache.fileId(path, Files.size(path));
    }

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

    private void configure(final int slabs, final boolean spanFill, final String readAround) {
        final long maxBytes = (long) slabs * TEST_SLAB_BYTES;
        withNodeSettings(
            Settings.builder()
                .put(KNNSettings.KNN_DIRECT_IO_RESCORE_CACHE_MAX_BYTES, maxBytes + "b")
                .put(KNNSettings.KNN_DIRECT_IO_RESCORE_CACHE_SPAN_FILL, spanFill)
                .put(KNNSettings.KNN_DIRECT_IO_RESCORE_CACHE_READ_AROUND_BYTES, readAround)
                .put(KNNSettings.KNN_DIRECT_IO_READAHEAD_WINDOW, 0)
                .build()
        );
        DirectIOVectorCache.resetForTesting();
        DirectIOVectorCache.setSlabBytesForTesting(TEST_SLAB_BYTES);
        DirectIOVectorCache.configure(maxBytes);
        DirectIOVectorIndexInput.STATS.reset();
    }

    @SneakyThrows
    private void writeFixture(final long seed) {
        final Random random = new Random(seed);
        vectors = new float[VECTOR_COUNT][DIMENSION];
        for (final float[] vector : vectors) {
            for (int i = 0; i < DIMENSION; i++) {
                vector[i] = (random.nextFloat() - 0.5f) * 8.0f;
            }
        }
        path = createTempDir().resolve("read-around-" + seed + ".vec");
        final ByteBuffer buffer = ByteBuffer.allocate(HEADER_LENGTH + VECTOR_COUNT * VECTOR_BYTES + FOOTER_LENGTH)
            .order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < HEADER_LENGTH; i++) {
            // Recognisable bytes: if any of the header were ever admitted as a vector, the comparison with
            // the fixture would say so rather than passing by luck.
            buffer.put((byte) (0xA0 + (i & 0x0F)));
        }
        for (final float[] vector : vectors) {
            for (final float value : vector) {
                buffer.putFloat(value);
            }
        }
        for (int i = 0; i < FOOTER_LENGTH; i++) {
            buffer.put((byte) (0xF0 + (i & 0x0F)));
        }
        Files.write(path, buffer.array());
    }

    /** One vector the way the rescore path reads one: declare it, then consume it. */
    @SneakyThrows
    private void readVector(final DirectIOVectorIndexInput input, final long at, final float[] into) {
        input.prefetch(at, VECTOR_BYTES);
        input.seek(at);
        input.readFloats(into, 0, DIMENSION);
    }
}

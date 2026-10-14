/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import com.carrotsearch.randomizedtesting.ThreadFilter;
import com.carrotsearch.randomizedtesting.annotations.ThreadLeakFilters;
import lombok.SneakyThrows;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.MMapDirectory;
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
import java.util.concurrent.atomic.AtomicBoolean;

import static org.mockito.Mockito.when;

/**
 * O8: the FP16 {@code .vec} is routed, served and admitted, and the reads that are <b>not</b> vector reads
 * still are not.
 *
 * <h2>Why this needed code at all, when the store keys on offset and byte length</h2>
 * Routing was two names ({@code KNNDirectIODirectoryTests}): upstream's half-float formats nest
 * {@code KNN1040HalfFloatFlatVectorsFormat}, whose data extension is also {@code vec}, but their per-field
 * format names carry an inserted {@code HalfFloat} that broke the dispatch rule's substring, so the file was
 * declined at condition 1 before any counter moved. The real work was elsewhere.
 * {@code KNN1040HalfFloatFlatVectorsValues#readRawVectorBytes} reads a vector with
 * {@code slice.readBytes(dest, destOffset, 2 * dimension)} and never with {@code readFloats} — the point of
 * that format is that nothing decodes FP16 to FP32 on the read path — and the store's only serve site was
 * inside {@link DirectIOVectorIndexInput#readFloats}. Routing alone would therefore have given the FP16 path
 * prefetch and staged reads and <b>never a hit and never an admission</b>.
 *
 * <p>So O8 is a byte-level serve site ({@link DirectIOVectorCache#getBytes}) and the same two-site treatment
 * in {@link DirectIOVectorIndexInput#readBytes}. The store itself needed nothing: a slab holds raw bytes, the
 * key is {@code (fileId, offset)} and the slot carries its own byte length, so a 2-byte-per-dimension vector
 * is just a shorter slot. The one store rule that had to move is the fill paths' length check, from "a whole
 * number of floats" to "an even number of bytes", because an FP16 vector of an <em>odd</em> dimension is a
 * multiple of 2 and not of 4; {@link DirectIOVectorCache#getFloats} refuses such a length outright so that a
 * slot which is not float-aligned can never be read through the float view.
 *
 * <h2>And why the gate is the interesting part</h2>
 * {@code readBytes} is not the vector read. It is also the file's header, its footer, a
 * {@code DirectMonotonicReader}'s metadata and the whole sequential merge scan. Hooking it unconditionally
 * would admit a 44-byte header as a "vector" and break M4's whole-vectors-only geometry. The gate is the
 * vector grid {@link DirectIOVectorIndexInput#selectUncachedOrdinals} is told by the layer that knows it, and
 * the tests below for the header, for an off-grid read and for an input nobody declared a grid on are the
 * ones that would fail if it ever stopped being precise.
 */
@ThreadLeakFilters(defaultFilters = true, filters = { DirectIOHalfFloatVectorTests.ReadPoolThreadFilter.class })
public class DirectIOHalfFloatVectorTests extends KNNTestCase {

    /** The read pool is never shut down by design; its daemon threads outlive any one suite. */
    public static final class ReadPoolThreadFilter implements ThreadFilter {
        @Override
        public boolean reject(final Thread thread) {
            return thread.getName().startsWith(DirectIOReadPool.THREAD_NAME_PREFIX);
        }
    }

    /** A dimension whose FP16 vector is a multiple of four bytes, which is the ordinary case. */
    private static final int EVEN_DIMENSION = 128;
    /** And one whose is not: 65 half floats is 130 bytes, a multiple of 2 and not of 4. */
    private static final int ODD_DIMENSION = 65;
    /** Not a multiple of any vector length and not a multiple of a block, which is the awkward real case. */
    private static final int HEADER_LENGTH = 44;
    private static final int FOOTER_LENGTH = 16;
    private static final int VECTOR_COUNT = 512;
    private static final int TEST_SLAB_BYTES = 64 * 1024;

    private Path path;
    private int vectorBytes;
    /** Each vector's raw FP16 bytes, which is what the FP16 read path reads and what a hit must equal. */
    private byte[][] vectors;

    @Override
    public void tearDown() throws Exception {
        DirectIOVectorCache.resetForTesting();
        DirectIOReadPool.resetForTesting();
        DirectIOVectorIndexInput.STATS.reset();
        super.tearDown();
    }

    // -----------------------------------------------------------------------------------------------
    // The vector read: staged, admitted, then served
    // -----------------------------------------------------------------------------------------------

    /**
     * The whole of O8 in one test: a half-float vector read through {@code readBytes} is staged once, kept,
     * and the next read of it is served from the store with no device read at all.
     */
    @SneakyThrows
    public void testAHalfFloatVectorIsStagedAdmittedAndThenServed() {
        assumeDirectIOWorksHere();
        writeFixture(40L, EVEN_DIMENSION);
        configure(8, true, "0b");

        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(input);
            final byte[] scratch = new byte[vectorBytes];
            readVector(input, 200, scratch);

            assertArrayEquals("the vector's own bytes", vectors[200], scratch);
            assertEquals("one staged read", 1L, DirectIOVectorIndexInput.STATS.stagedReads());
            assertTrue("and it was admitted", DirectIOVectorCache.contains(fileId(), offsetOf(200), vectorBytes));

            final long staged = DirectIOVectorIndexInput.STATS.stagedReads();
            final long blocking = DirectIOVectorIndexInput.STATS.blockingReads();
            final long hitsBefore = DirectIOVectorCache.hits();
            final byte[] second = new byte[vectorBytes];
            readVector(input, 200, second);

            assertArrayEquals("served byte for byte", vectors[200], second);
            assertEquals("from the store", hitsBefore + 1, DirectIOVectorCache.hits());
            assertEquals("with no further staged read", staged, DirectIOVectorIndexInput.STATS.stagedReads());
            assertEquals("and no blocking read", blocking, DirectIOVectorIndexInput.STATS.blockingReads());
            DirectIOVectorCache.checkInvariantsForTesting();
        }
    }

    /**
     * A hit is the delegate's own bytes. This is the FP16 analogue of the fp32 identity proof: the scores the
     * FP16 scorer computes are a function of these bytes alone, so byte identity against the mmap read of the
     * same range is what makes a store hit unable to change an answer.
     */
    @SneakyThrows
    public void testAHitIsByteIdenticalToTheDelegatesRead() {
        assumeDirectIOWorksHere();
        writeFixture(41L, EVEN_DIMENSION);
        configure(8, true, "0b");

        final int[] ordinals = { 0, 1, 7, 63, 128, 301, VECTOR_COUNT - 1 };
        try (MMapDirectory delegate = new MMapDirectory(path.getParent())) {
            try (
                DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path);
                IndexInput mmap = delegate.openInput(path.getFileName().toString(), org.apache.lucene.store.IOContext.DEFAULT)
            ) {
                assertNotNull(input);
                final byte[] fromStore = new byte[vectorBytes];
                final byte[] fromDelegate = new byte[vectorBytes];
                for (final int ordinal : ordinals) {
                    // First read: a miss, which admits it. Second: a hit.
                    readVector(input, ordinal, fromStore);
                    readVector(input, ordinal, fromStore);
                    mmap.seek(offsetOf(ordinal));
                    mmap.readBytes(fromDelegate, 0, vectorBytes);
                    assertArrayEquals("ordinal " + ordinal, fromDelegate, fromStore);
                }
                assertTrue("the second read of each was a hit", DirectIOVectorCache.hits() >= ordinals.length);
            }
        }
    }

    /**
     * An odd dimension, which is the case the fill paths' length rule had to change for: 65 half floats is 130
     * bytes, so a slot is not float-aligned. It is admitted and served through the byte path, and the float
     * path refuses the length rather than reading the slot through a float view.
     */
    @SneakyThrows
    public void testAnOddDimensionIsAdmittedThroughTheBytePathOnly() {
        assumeDirectIOWorksHere();
        writeFixture(42L, ODD_DIMENSION);
        assertEquals("the premise: not a whole number of floats", 2, vectorBytes % 4);
        configure(8, true, "0b");

        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(input);
            final byte[] scratch = new byte[vectorBytes];
            readVector(input, 11, scratch);
            assertArrayEquals(vectors[11], scratch);
            assertTrue(DirectIOVectorCache.contains(fileId(), offsetOf(11), vectorBytes));

            final byte[] fromStore = new byte[vectorBytes];
            assertTrue(DirectIOVectorCache.getBytes(fileId(), offsetOf(11), vectorBytes, fromStore, 0));
            assertArrayEquals(vectors[11], fromStore);

            // And the float view cannot reach it: a length that is not a whole number of floats is refused
            // before the lookup, so it counts neither a hit nor a miss.
            final long hits = DirectIOVectorCache.hits();
            final long misses = DirectIOVectorCache.misses();
            assertFalse(DirectIOVectorCache.getFloats(fileId(), offsetOf(11), vectorBytes, new float[vectorBytes], 0));
            assertEquals(hits, DirectIOVectorCache.hits());
            assertEquals("and it is not a lookup either", misses, DirectIOVectorCache.misses());
            DirectIOVectorCache.checkInvariantsForTesting();
        }
    }

    // -----------------------------------------------------------------------------------------------
    // Span fill and read-around, at two bytes a dimension
    // -----------------------------------------------------------------------------------------------

    /** M4's span fill harvests the whole half-float vectors of a staged span, and nothing else. */
    @SneakyThrows
    public void testSpanFillAdmitsTheWholeHalfFloatVectorsOfASpan() {
        assumeDirectIOWorksHere();
        writeFixture(43L, EVEN_DIMENSION);
        configure(8, true, "0b");

        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(input);
            readVector(input, 200, new byte[vectorBytes]);

            final long held = DirectIOVectorCache.entries();
            assertTrue("a span holds more than the one vector asked for, it held " + held, held > 1);
            assertEquals("and span fill is what admitted the rest", held - 1, DirectIOVectorCache.spanFills());
            assertEveryEntryIsItsOwnVector();
        }
    }

    /** With span fill off, admission is the requested vector and nothing else. */
    @SneakyThrows
    public void testSpanFillOffAdmitsOnlyTheRequestedVector() {
        assumeDirectIOWorksHere();
        writeFixture(44L, EVEN_DIMENSION);
        configure(8, false, "0b");

        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(input);
            readVector(input, 200, new byte[vectorBytes]);
            assertEquals(1L, DirectIOVectorCache.entries());
            assertEquals(0L, DirectIOVectorCache.spanFills());
        }
    }

    /**
     * O5's read-around, on the FP16 path: the miss read is widened, every whole vector in it is admitted, and
     * each one is served afterwards as its own bytes. Halving the vector length doubles what a window of the
     * same width holds, which is the only thing precision changes here.
     */
    @SneakyThrows
    public void testReadAroundWidensAHalfFloatMissRead() {
        assumeDirectIOWorksHere();
        writeFixture(45L, EVEN_DIMENSION);
        // Eight 64 KiB slabs against a 131 KiB file, so the whole file fits in the room the target leaves.
        configure(8, true, "64kb");

        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(input);
            final byte[] scratch = new byte[vectorBytes];
            readVector(input, 200, scratch);
            assertArrayEquals(vectors[200], scratch);

            assertEquals("the read was widened", 1L, DirectIOVectorIndexInput.STATS.readAroundReads());
            assertEquals("and it is still one read", 1L, DirectIOVectorIndexInput.STATS.stagedReads());
            final long held = DirectIOVectorCache.entries();
            assertTrue("a 64 KiB window holds many 256-byte vectors, it admitted " + held, held > 128);
            assertTrue("and no more than the window can hold", held <= 64 * 1024 / vectorBytes);
            assertEveryEntryIsItsOwnVector();

            // A neighbour is then served from the store, with no second device read.
            int neighbour = -1;
            for (int ordinal = 201; ordinal < VECTOR_COUNT; ordinal++) {
                if (DirectIOVectorCache.contains(fileId(), offsetOf(ordinal), vectorBytes)) {
                    neighbour = ordinal;
                    break;
                }
            }
            assertTrue("the read-around must have admitted a neighbour above 200", neighbour > 200);
            final long staged = DirectIOVectorIndexInput.STATS.stagedReads();
            readVector(input, neighbour, scratch);
            assertArrayEquals(vectors[neighbour], scratch);
            assertEquals(staged, DirectIOVectorIndexInput.STATS.stagedReads());
        }
    }

    // -----------------------------------------------------------------------------------------------
    // The gate: what readBytes must NOT treat as a vector
    // -----------------------------------------------------------------------------------------------

    /**
     * The header is a {@code readBytes} too, and the merge path's scan is a long run of them. Neither is the
     * declared vector length at a declared grid offset, so neither is looked up and neither is admitted. A
     * 44-byte header admitted as a "vector" is exactly what M4's geometry tests exist to prevent, and
     * {@code readBytes} is the one read method that would have let it in.
     */
    @SneakyThrows
    public void testAHeaderOrAnOffGridReadIsNeitherServedNorAdmitted() {
        assumeDirectIOWorksHere();
        writeFixture(46L, EVEN_DIMENSION);
        configure(8, false, "0b");

        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(input);
            // Declare the grid, so the only thing standing between these reads and the store is the gate.
            input.selectUncachedOrdinals(HEADER_LENGTH, vectorBytes, new int[] { 0 }, 1, new int[1]);
            final long hits = DirectIOVectorCache.hits();
            final long misses = DirectIOVectorCache.misses();

            // The header: a shorter run, at an offset before ordinal zero.
            final byte[] header = new byte[HEADER_LENGTH];
            input.prefetch(0, HEADER_LENGTH);
            input.seek(0);
            input.readBytes(header, 0, HEADER_LENGTH);
            for (int i = 0; i < HEADER_LENGTH; i++) {
                assertEquals("the header's own bytes", (byte) (0xA0 + (i & 0x0F)), header[i]);
            }

            // A vector-length run that is two bytes off the grid.
            final byte[] skewed = new byte[vectorBytes];
            input.prefetch(offsetOf(5) + 2, vectorBytes);
            input.seek(offsetOf(5) + 2);
            input.readBytes(skewed, 0, vectorBytes);

            assertEquals("no entry of any kind", 0L, DirectIOVectorCache.entries());
            assertEquals("and not one lookup: these are not vector reads", hits, DirectIOVectorCache.hits());
            assertEquals(misses, DirectIOVectorCache.misses());
            assertFalse(DirectIOVectorCache.contains(fileId(), 0L, vectorBytes));
            assertFalse(DirectIOVectorCache.contains(fileId(), offsetOf(5) + 2, vectorBytes));
        }
    }

    /**
     * An input nobody declared a grid on admits nothing, whatever it reads. This is what keeps the merge path
     * out of the store: a merge opens its own reader and never calls the prefetch filter, so its sequential
     * {@code readBytes} run cannot be mistaken for a rescore read even when it happens to be one vector long
     * at one vector's offset.
     */
    @SneakyThrows
    public void testAnUndeclaredInputNeitherServesNorAdmits() {
        assumeDirectIOWorksHere();
        writeFixture(47L, EVEN_DIMENSION);
        configure(8, true, "0b");

        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(input);
            final long hits = DirectIOVectorCache.hits();
            final long misses = DirectIOVectorCache.misses();
            final byte[] scratch = new byte[vectorBytes];
            // Exactly one vector, at exactly its offset, and no filter call ever made on this input.
            input.prefetch(offsetOf(77), vectorBytes);
            input.seek(offsetOf(77));
            input.readBytes(scratch, 0, vectorBytes);

            assertArrayEquals("the right bytes, as always", vectors[77], scratch);
            assertEquals("nothing admitted", 0L, DirectIOVectorCache.entries());
            assertEquals("and no lookup", hits, DirectIOVectorCache.hits());
            assertEquals(misses, DirectIOVectorCache.misses());
        }
    }

    /**
     * The grid outlives the burst that declared it, and a clone inherits it. Both are needed because
     * {@code PrefetchHelper#prefetch} returns early for a batch of one, so a single-candidate batch makes no
     * filter call of its own, and {@code KNN1040HalfFloatFlatVectorsValues#copy} clones the slice.
     */
    @SneakyThrows
    public void testTheGridSurvivesTheBurstBoundaryAndAClone() {
        assumeDirectIOWorksHere();
        writeFixture(48L, EVEN_DIMENSION);
        configure(8, false, "0b");

        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(input);
            final byte[] scratch = new byte[vectorBytes];
            // One declared burst, which also ends it: the read is the burst's last act.
            readVector(input, 300, scratch);
            assertTrue(DirectIOVectorCache.contains(fileId(), offsetOf(300), vectorBytes));

            // A new burst with no filter call at all, which is what a batch of one looks like.
            final long hitsBefore = DirectIOVectorCache.hits();
            input.prefetch(offsetOf(300), vectorBytes);
            input.seek(offsetOf(300));
            input.readBytes(scratch, 0, vectorBytes);
            assertArrayEquals(vectors[300], scratch);
            assertEquals("served from the store on the grid declared earlier", hitsBefore + 1, DirectIOVectorCache.hits());

            // And a clone, which may read before its own first filter call.
            try (DirectIOVectorIndexInput copy = input.clone()) {
                final long hits = DirectIOVectorCache.hits();
                copy.seek(offsetOf(300));
                copy.readBytes(scratch, 0, vectorBytes);
                assertArrayEquals(vectors[300], scratch);
                assertEquals("a clone carries its parent's grid", hits + 1, DirectIOVectorCache.hits());
            }
        }
    }

    // -----------------------------------------------------------------------------------------------
    // O1's switch, on the byte path
    // -----------------------------------------------------------------------------------------------

    /**
     * O1's rollback reaches the FP16 read path too: with the setting off the bytes come from the delegate's
     * own input and nothing is served from or admitted to the store, and with it on again the next burst is
     * back on {@code O_DIRECT} and admitting.
     */
    @SneakyThrows
    public void testTheRouteSwitchReachesTheBytePath() {
        assumeDirectIOWorksHere();
        writeFixture(49L, EVEN_DIMENSION);
        configure(8, false, "0b");

        final AtomicBoolean enabled = new AtomicBoolean(false);
        try (MMapDirectory delegate = new MMapDirectory(path.getParent())) {
            final IndexInput fallback = delegate.openInput(path.getFileName().toString(), org.apache.lucene.store.IOContext.DEFAULT);
            try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path, fallback, enabled::get)) {
                assertNotNull(input);
                assertFalse("the switch is off, so this input starts on the delegate", input.isDirectIOActive());

                final byte[] scratch = new byte[vectorBytes];
                readVector(input, 120, scratch);
                assertArrayEquals("the delegate reads the same bytes", vectors[120], scratch);
                assertTrue("through the delegate", DirectIOVectorIndexInput.STATS.fallbackReads() > 0);
                assertEquals("and nothing is admitted while the switch is off", 0L, DirectIOVectorCache.entries());

                // Flip it on. The route changes at the next burst, with no reopen.
                enabled.set(true);
                readVector(input, 121, scratch);
                assertArrayEquals(vectors[121], scratch);
                assertTrue("back on O_DIRECT", input.isDirectIOActive());
                assertTrue("and admitting again", DirectIOVectorCache.contains(fileId(), offsetOf(121), vectorBytes));
            }
        }
    }

    // -----------------------------------------------------------------------------------------------
    // O4's race, on the byte path
    // -----------------------------------------------------------------------------------------------

    /**
     * O4 on the FP16 path: a vector the prefetch filter dropped as cached and the store then lost is read
     * synchronously, and its bytes are kept rather than thrown away.
     */
    @SneakyThrows
    public void testTheRacesSynchronousReadIsAdmittedOnTheBytePath() {
        assumeDirectIOWorksHere();
        writeFixture(50L, EVEN_DIMENSION);
        configure(8, false, "0b");

        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(input);
            final byte[] scratch = new byte[vectorBytes];
            readVector(input, 9, scratch);
            assertTrue(DirectIOVectorCache.contains(fileId(), offsetOf(9), vectorBytes));

            final int[] batch = { 9, 400 };
            assertEquals(
                "the filter drops the cached one",
                1,
                input.selectUncachedOrdinals(HEADER_LENGTH, vectorBytes, batch, batch.length, new int[batch.length])
            );
            // Lose it without touching the input, so the burst and its filter answers stand.
            DirectIOVectorCache.releaseFile(fileId());
            assertFalse(DirectIOVectorCache.contains(fileId(), offsetOf(9), vectorBytes));

            final long raceBefore = DirectIOVectorCache.raceReadmissions();
            input.seek(offsetOf(9));
            input.readBytes(scratch, 0, vectorBytes);

            assertArrayEquals("the vector's own bytes either way", vectors[9], scratch);
            assertTrue("its bytes were kept", DirectIOVectorCache.contains(fileId(), offsetOf(9), vectorBytes));
            assertEquals("and counted as a race readmission", raceBefore + 1, DirectIOVectorCache.raceReadmissions());
            DirectIOVectorCache.checkInvariantsForTesting();
        }
    }

    // -----------------------------------------------------------------------------------------------
    // Fixture and helpers
    // -----------------------------------------------------------------------------------------------

    /** Every entry the store holds is the vector its key names, byte for byte, and nothing else is in it. */
    @SneakyThrows
    private void assertEveryEntryIsItsOwnVector() {
        int checked = 0;
        final byte[] fromStore = new byte[vectorBytes];
        for (int ordinal = 0; ordinal < VECTOR_COUNT; ordinal++) {
            if (DirectIOVectorCache.contains(fileId(), offsetOf(ordinal), vectorBytes)) {
                assertTrue(DirectIOVectorCache.getBytes(fileId(), offsetOf(ordinal), vectorBytes, fromStore, 0));
                assertArrayEquals("admitted vector " + ordinal, vectors[ordinal], fromStore);
                checked++;
            }
        }
        assertEquals("every admitted entry was checked", DirectIOVectorCache.entries(), (long) checked);
        // And nothing outside the vector grid: the header before ordinal zero and the footer after the last.
        assertFalse(DirectIOVectorCache.contains(fileId(), 0L, vectorBytes));
        assertFalse(DirectIOVectorCache.contains(fileId(), offsetOf(VECTOR_COUNT), vectorBytes));
        DirectIOVectorCache.checkInvariantsForTesting();
    }

    private long offsetOf(final int ordinal) {
        return HEADER_LENGTH + (long) ordinal * vectorBytes;
    }

    @SneakyThrows
    private String fileId() {
        return DirectIOVectorCache.fileId(path, Files.size(path));
    }

    /**
     * One vector the way {@code KNN1040HalfFloatFlatVectorsValues#readRawVectorBytes} reads one, with the
     * grouping layer's two declarations in front of it: the field's grid, then the range.
     */
    @SneakyThrows
    private void readVector(final DirectIOVectorIndexInput input, final int ordinal, final byte[] into) {
        input.selectUncachedOrdinals(HEADER_LENGTH, vectorBytes, new int[] { ordinal }, 1, new int[1]);
        input.prefetch(offsetOf(ordinal), vectorBytes);
        input.seek(offsetOf(ordinal));
        input.readBytes(into, 0, vectorBytes);
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

    /**
     * A half-float {@code .vec}: a header, {@link #VECTOR_COUNT} vectors of {@code dimension} raw 2-byte
     * values, and a footer. The values are written as shorts rather than through an FP16 encoder because what
     * O8 has to be right about is bytes: the decode is the scorer's and is not on this path at all.
     */
    @SneakyThrows
    private void writeFixture(final long seed, final int dimension) {
        this.vectorBytes = dimension * Short.BYTES;
        final Random random = new Random(seed);
        vectors = new byte[VECTOR_COUNT][vectorBytes];
        final ByteBuffer buffer = ByteBuffer.allocate(HEADER_LENGTH + VECTOR_COUNT * vectorBytes + FOOTER_LENGTH)
            .order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < HEADER_LENGTH; i++) {
            // Recognisable bytes: if any of the header were ever admitted as a vector, the comparison with
            // the fixture would say so rather than passing by luck.
            buffer.put((byte) (0xA0 + (i & 0x0F)));
        }
        for (int ordinal = 0; ordinal < VECTOR_COUNT; ordinal++) {
            random.nextBytes(vectors[ordinal]);
            buffer.put(vectors[ordinal]);
        }
        for (int i = 0; i < FOOTER_LENGTH; i++) {
            buffer.put((byte) (0xF0 + (i & 0x0F)));
        }
        path = createTempDir().resolve("half-float-" + seed + ".vec");
        Files.write(path, buffer.array());
    }
}

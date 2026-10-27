/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import com.carrotsearch.randomizedtesting.annotations.ThreadLeakFilters;
import lombok.SneakyThrows;
import org.apache.lucene.store.DataAccessHint;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FileDataHint;
import org.apache.lucene.store.FileTypeHint;
import org.apache.lucene.store.FilterIndexInput;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.store.MMapDirectory;
import org.opensearch.common.settings.ClusterSettings;
import org.opensearch.common.settings.Setting;
import org.opensearch.common.settings.Settings;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.codec.KNN990Codec.NativeEngines990KnnVectorsFormat;

import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.OpenOption;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

import static org.mockito.Mockito.when;

/**
 * O1 (16-I's I6, D2): {@code knn.direct_io.rescore.enabled} reaches an input that is <b>already open</b>.
 *
 * <p>The decision this proves is the one the design review took: there is no automatic switching between
 * mmap and Direct I/O, the setting is the operator's lever, and it has to be a lever that works now. Before
 * this, the dispatch rule was taken at {@code openInput} and an input kept its {@code O_DIRECT} channel for
 * life, so turning the feature off left every open reader reading the device until the next refresh --
 * which is not a rollback. A routed input now carries the delegate's own input beside the channel and
 * chooses between them at each prefetch burst boundary.
 *
 * <p>Four properties, and the third is the one that makes the first two safe to have:
 * <ul>
 *   <li>the route changes at the next burst, with no reopen, in both directions;</li>
 *   <li>the bytes are the same bytes on either route, so a flip mid-query cannot change a score;</li>
 *   <li>the change happens only at a burst boundary, never between a prefetch and the read that consumes
 *       it, because a staged read and its reader have to be on the same channel;</li>
 *   <li>and in this phase, where the slab store sits <em>below</em> the input rather than in the scorer,
 *       the delegate route neither serves from the store nor admits to it, and the store's slabs go back.</li>
 * </ul>
 */
@ThreadLeakFilters(defaultFilters = true, filters = { DirectIOReadPoolTests.ReadPoolThreadFilter.class })
public class DirectIORouteSwitchTests extends KNNTestCase {

    /** A native-engine (faiss / MOS) fp32 flat file: the shape the dispatch rule routes. */
    private static final String VECTOR_DATA = "_0_" + NativeEngines990KnnVectorsFormat.FORMAT_NAME + "_0.vec";

    private static final int DIMENSION = 128;
    private static final int VECTORS = 512;
    private static final int VECTOR_BYTES = DIMENSION * Float.BYTES;
    private static final int TEST_SLAB_BYTES = 1024 * 1024;

    @Override
    public void tearDown() throws Exception {
        DirectIOVectorCache.resetForTesting();
        DirectIOReadPool.resetForTesting();
        DirectIOVectorIndexInput.STATS.reset();
        super.tearDown();
    }

    @SneakyThrows
    private void assumeDirectIOWorksHere() {
        final OpenOption direct = DirectIOVectorIndexInput.directOpenOption();
        assumeTrue("this JDK has no ExtendedOpenOption.DIRECT", direct != null);
        final Path probe = createTempDir().resolve("probe");
        Files.write(probe, new byte[8192]);
        try (FileChannel channel = FileChannel.open(probe, StandardOpenOption.READ, direct)) {
            assertNotNull(channel);
        } catch (IOException | UnsupportedOperationException e) {
            assumeNoException("this filesystem refuses O_DIRECT", e);
        }
    }

    /** The context Lucene's flat vectors reader builds for its own {@code .vec} open. */
    private static IOContext vectorDataContext() {
        return IOContext.DEFAULT.withHints(FileTypeHint.DATA, FileDataHint.KNN_VECTORS, DataAccessHint.RANDOM);
    }

    /** Deterministic vectors, so a comparison failure names the ordinal rather than the seed. */
    @SneakyThrows
    private static float[][] writeVectors(final Directory directory) {
        final float[][] vectors = new float[VECTORS][DIMENSION];
        try (IndexOutput output = directory.createOutput(VECTOR_DATA, IOContext.DEFAULT)) {
            for (int ordinal = 0; ordinal < VECTORS; ordinal++) {
                for (int component = 0; component < DIMENSION; component++) {
                    vectors[ordinal][component] = ordinal * 0.5f + component * 0.125f;
                    output.writeInt(Float.floatToRawIntBits(vectors[ordinal][component]));
                }
            }
        }
        return vectors;
    }

    /** One prefetch burst over {@code ordinals}, then one read of each: the rescore path's shape. */
    private static float[][] burst(final IndexInput input, final int... ordinals) throws IOException {
        for (final int ordinal : ordinals) {
            input.prefetch((long) ordinal * VECTOR_BYTES, VECTOR_BYTES);
        }
        final float[][] read = new float[ordinals.length][DIMENSION];
        for (int i = 0; i < ordinals.length; i++) {
            input.seek((long) ordinals[i] * VECTOR_BYTES);
            input.readFloats(read[i], 0, DIMENSION);
        }
        return read;
    }

    private static void assertVectorsEqual(final String message, final float[][] expected, final int[] ordinals, final float[][] actual) {
        for (int i = 0; i < ordinals.length; i++) {
            assertArrayEquals(message + " at ordinal " + ordinals[i], expected[ordinals[i]], actual[i], 0.0f);
        }
    }

    // -------------------------------------------------------------------------------------------------
    // The route changes under an open input, in both directions
    // -------------------------------------------------------------------------------------------------

    /**
     * The whole of O1 in one test: one input, opened once, serving three bursts on two different channels
     * as the switch moves, with no reopen and the same bytes throughout.
     */
    @SneakyThrows
    public void testTheSwitchChangesAnOpenInputsRouteAtTheNextBurst() {
        assumeDirectIOWorksHere();
        final AtomicBoolean enabled = new AtomicBoolean(true);
        final int[] ordinals = { 3, 11, 64, 200, 511 };
        try (MMapDirectory delegate = new MMapDirectory(createTempDir())) {
            final float[][] expected = writeVectors(delegate);
            try (KNNDirectIODirectory storage = new KNNDirectIODirectory(delegate, "_0", enabled::get, () -> 0L)) {
                DirectIOVectorIndexInput.STATS.reset();
                try (IndexInput opened = storage.openInput(VECTOR_DATA, vectorDataContext())) {
                    assertTrue("the file must route, or there is nothing to switch", opened instanceof DirectIOVectorIndexInput);
                    final DirectIOVectorIndexInput routed = (DirectIOVectorIndexInput) opened;
                    assertTrue("a routed input must carry the input it falls back to", routed.hasMmapFallback());

                    // Burst one: the switch is on, so this is O_DIRECT.
                    assertVectorsEqual("O_DIRECT", expected, ordinals, burst(routed, ordinals));
                    final long deviceReads = DirectIOVectorIndexInput.STATS.stagedReads() + DirectIOVectorIndexInput.STATS.blockingReads();
                    assertTrue("the first burst should have read the device, it did " + deviceReads + " reads", deviceReads > 0);
                    assertEquals("and nothing should have gone through the delegate", 0L, DirectIOVectorIndexInput.STATS.fallbackReads());
                    assertTrue(routed.isDirectIOActive());

                    // The operator turns it off. Nothing is reopened.
                    enabled.set(false);
                    assertVectorsEqual("the delegate", expected, ordinals, burst(routed, ordinals));
                    assertFalse("the input must have changed route", routed.isDirectIOActive());
                    assertEquals("exactly one input changed over", 1L, DirectIOVectorIndexInput.STATS.switchedToMmap());
                    assertEquals(
                        "every read of the second burst must be a delegate read",
                        ordinals.length,
                        DirectIOVectorIndexInput.STATS.fallbackReads()
                    );
                    assertEquals(
                        "and the device must be untouched by it",
                        deviceReads,
                        DirectIOVectorIndexInput.STATS.stagedReads() + DirectIOVectorIndexInput.STATS.blockingReads()
                    );

                    // And back on.
                    enabled.set(true);
                    assertVectorsEqual("O_DIRECT again", expected, ordinals, burst(routed, ordinals));
                    assertTrue("the input must have changed back", routed.isDirectIOActive());
                    assertEquals(1L, DirectIOVectorIndexInput.STATS.switchedToDirect());
                    assertEquals("the delegate must stop serving reads", ordinals.length, DirectIOVectorIndexInput.STATS.fallbackReads());
                    assertTrue(
                        "the device must be read again",
                        DirectIOVectorIndexInput.STATS.stagedReads() + DirectIOVectorIndexInput.STATS.blockingReads() > deviceReads
                    );
                }
                assertEquals("the switch must not have caused a second open", 1L, storage.routedOpens());
            }
        }
    }

    /**
     * The file read whole on either route, byte for byte. A flip is only safe to take mid-query because
     * this holds: both routes are reads of the same bytes of the same file, which is the same argument the
     * dispatch rule's own fallback rests on.
     */
    @SneakyThrows
    public void testTheTwoRoutesReadTheSameBytes() {
        assumeDirectIOWorksHere();
        final AtomicBoolean enabled = new AtomicBoolean(true);
        try (MMapDirectory delegate = new MMapDirectory(createTempDir())) {
            writeVectors(delegate);
            try (KNNDirectIODirectory storage = new KNNDirectIODirectory(delegate, "_0", enabled::get, () -> 0L)) {
                try (IndexInput routed = storage.openInput(VECTOR_DATA, vectorDataContext())) {
                    final byte[] direct = new byte[Math.toIntExact(routed.length())];
                    routed.seek(0);
                    routed.readBytes(direct, 0, direct.length);

                    enabled.set(false);
                    // A prefetch is what carries the switch, so the burst boundary comes first.
                    routed.prefetch(0, VECTOR_BYTES);
                    final byte[] fallback = new byte[direct.length];
                    routed.seek(0);
                    routed.readBytes(fallback, 0, fallback.length);

                    assertArrayEquals("the two routes must read the same file", direct, fallback);
                    assertFalse(((DirectIOVectorIndexInput) routed).isDirectIOActive());
                }
            }
        }
    }

    /**
     * A slice and a clone each carry their own view of the delegate's input, with this input's coordinates,
     * so the switch reaches them too. This is not a detail of the test: {@code OffHeapFloatVectorValues}
     * holds a <em>slice</em> and nothing else, so a slice that could not switch would mean the switch never
     * reached the rescore path at all.
     */
    @SneakyThrows
    public void testASliceAndACloneSwitchToo() {
        assumeDirectIOWorksHere();
        final AtomicBoolean enabled = new AtomicBoolean(true);
        final int[] ordinals = { 0, 5, 63 };
        try (MMapDirectory delegate = new MMapDirectory(createTempDir())) {
            final float[][] expected = writeVectors(delegate);
            try (KNNDirectIODirectory storage = new KNNDirectIODirectory(delegate, "_0", enabled::get, () -> 0L)) {
                try (IndexInput routed = storage.openInput(VECTOR_DATA, vectorDataContext())) {
                    final IndexInput slice = routed.slice("vectors", 0, routed.length());
                    final IndexInput clone = routed.clone();
                    assertTrue(((DirectIOVectorIndexInput) slice).hasMmapFallback());
                    assertTrue(((DirectIOVectorIndexInput) clone).hasMmapFallback());

                    enabled.set(false);
                    assertVectorsEqual("a slice on the delegate", expected, ordinals, burst(slice, ordinals));
                    assertVectorsEqual("a clone on the delegate", expected, ordinals, burst(clone, ordinals));
                    assertFalse(((DirectIOVectorIndexInput) slice).isDirectIOActive());
                    assertFalse(((DirectIOVectorIndexInput) clone).isDirectIOActive());

                    enabled.set(true);
                    assertVectorsEqual("a slice back on O_DIRECT", expected, ordinals, burst(slice, ordinals));
                    assertTrue(((DirectIOVectorIndexInput) slice).isDirectIOActive());
                }
            }
        }
    }

    /**
     * A clone taken while its parent is on the delegate starts on the delegate too, and does not read the
     * device on its first read. Without the route being copied, the clone's constructor would read the
     * switch -- which is the same answer here -- but a clone taken in the window between a flip and the
     * next burst would disagree with its parent mid burst.
     */
    @SneakyThrows
    public void testACloneInheritsItsParentsRoute() {
        assumeDirectIOWorksHere();
        final AtomicBoolean enabled = new AtomicBoolean(true);
        try (MMapDirectory delegate = new MMapDirectory(createTempDir())) {
            final float[][] expected = writeVectors(delegate);
            try (KNNDirectIODirectory storage = new KNNDirectIODirectory(delegate, "_0", enabled::get, () -> 0L)) {
                try (IndexInput routed = storage.openInput(VECTOR_DATA, vectorDataContext())) {
                    // Put the parent on the delegate, then put the switch back on WITHOUT a burst, so the
                    // parent's route and the setting disagree. A clone must follow the parent, not the
                    // setting, because the parent may be mid burst.
                    enabled.set(false);
                    burst(routed, 1);
                    assertFalse(((DirectIOVectorIndexInput) routed).isDirectIOActive());
                    enabled.set(true);

                    final DirectIOVectorIndexInput clone = (DirectIOVectorIndexInput) routed.clone();
                    assertFalse("a clone starts on its parent's route", clone.isDirectIOActive());

                    // And it picks the new setting up at its own first burst, like any other input.
                    final int[] ordinals = { 2, 9 };
                    assertVectorsEqual("the clone's first burst", expected, ordinals, burst(clone, ordinals));
                    assertTrue(clone.isDirectIOActive());
                }
            }
        }
    }

    /**
     * The route may change only at a burst boundary. Within a burst the staged ranges and the reads that
     * consume them have to be on the same channel, so a flip that lands after the prefetches is not seen
     * until the burst after it.
     */
    @SneakyThrows
    public void testTheRouteDoesNotChangeInsideABurst() {
        assumeDirectIOWorksHere();
        final AtomicBoolean enabled = new AtomicBoolean(true);
        final int[] ordinals = { 7, 21, 99 };
        try (MMapDirectory delegate = new MMapDirectory(createTempDir())) {
            final float[][] expected = writeVectors(delegate);
            try (KNNDirectIODirectory storage = new KNNDirectIODirectory(delegate, "_0", enabled::get, () -> 0L)) {
                DirectIOVectorIndexInput.STATS.reset();
                try (IndexInput routed = storage.openInput(VECTOR_DATA, vectorDataContext())) {
                    for (final int ordinal : ordinals) {
                        routed.prefetch((long) ordinal * VECTOR_BYTES, VECTOR_BYTES);
                    }
                    // The flip lands between the prefetches and the reads they were staged for.
                    enabled.set(false);
                    final float[][] read = new float[ordinals.length][DIMENSION];
                    for (int i = 0; i < ordinals.length; i++) {
                        routed.seek((long) ordinals[i] * VECTOR_BYTES);
                        routed.readFloats(read[i], 0, DIMENSION);
                    }
                    assertVectorsEqual("the burst that was already staged", expected, ordinals, read);
                    assertTrue("the route must not change mid burst", ((DirectIOVectorIndexInput) routed).isDirectIOActive());
                    assertEquals("so no read may go through the delegate yet", 0L, DirectIOVectorIndexInput.STATS.fallbackReads());

                    // The next burst picks it up.
                    assertVectorsEqual("the next burst", expected, ordinals, burst(routed, ordinals));
                    assertFalse(((DirectIOVectorIndexInput) routed).isDirectIOActive());
                    assertEquals(ordinals.length, DirectIOVectorIndexInput.STATS.fallbackReads());
                }
            }
        }
    }

    /**
     * An input opened with no delegate -- every caller outside the directory, which is the measurement and
     * test entry points -- stays on {@code O_DIRECT} whatever the switch says, because a read with nowhere
     * to go is worse than a read on a channel the operator has turned off. The phase 9 probes and every
     * existing measurement in this package open inputs this way, so this is also what keeps them measuring
     * what they were taken against.
     *
     * <p>The directory itself can no longer produce this shape: a delegate open it cannot complete declines
     * the whole open, because without the delegate's input the file's bytes cannot be compared with what the
     * chain serves.
     */
    @SneakyThrows
    public void testAnInputWithNoFallbackStaysOnDirectIO() {
        assumeDirectIOWorksHere();
        try (MMapDirectory delegate = new MMapDirectory(createTempDir())) {
            writeVectors(delegate);
            final Path file = delegate.getDirectory().resolve(VECTOR_DATA);
            DirectIOVectorIndexInput.STATS.reset();
            try (DirectIOVectorIndexInput bare = DirectIOVectorIndexInput.open(file, null, () -> false)) {
                assertFalse(bare.hasMmapFallback());
                final float[] read = new float[DIMENSION];
                bare.prefetch(0, VECTOR_BYTES);
                bare.seek(0);
                bare.readFloats(read, 0, DIMENSION);
                assertTrue("with no delegate the route cannot change", bare.isDirectIOActive());
                assertEquals(0L, DirectIOVectorIndexInput.STATS.fallbackReads());
                assertEquals(0L, DirectIOVectorIndexInput.STATS.switchedToMmap());
            }
        }
    }

    // -------------------------------------------------------------------------------------------------
    // The store below the input follows the route: no service, no admission, and the slabs go back
    // -------------------------------------------------------------------------------------------------

    /**
     * This is the part of O1 that is 16-M's and not 16-I's: here the slab store sits <b>below</b> the input,
     * so a route that bypasses the {@code O_DIRECT} channel bypasses the store with it. With the switch off
     * nothing is admitted, nothing is served from the store, and the store's slabs go back -- which is what
     * makes the rollback return the memory and not only the device traffic.
     */
    @SneakyThrows
    public void testTheDelegateRouteNeitherServesNorAdmitsAndGivesTheSlabsBack() {
        assumeDirectIOWorksHere();
        final AtomicBoolean enabled = new AtomicBoolean(true);
        final int[] ordinals = { 4, 17, 128, 300 };
        try (MMapDirectory delegate = new MMapDirectory(createTempDir())) {
            final float[][] expected = writeVectors(delegate);
            cacheOn(1);
            try (KNNDirectIODirectory storage = new KNNDirectIODirectory(delegate, "_0", enabled::get, () -> 0L)) {
                DirectIOVectorIndexInput.STATS.reset();
                try (IndexInput routed = storage.openInput(VECTOR_DATA, vectorDataContext())) {
                    // On: the store fills and then serves.
                    assertVectorsEqual("O_DIRECT, cold", expected, ordinals, burst(routed, ordinals));
                    final long filled = DirectIOVectorCache.entries();
                    assertTrue("the store must have been filled, it holds " + filled, filled > 0);
                    assertVectorsEqual("O_DIRECT, warm", expected, ordinals, burst(routed, ordinals));
                    final long hits = DirectIOVectorCache.hits();
                    assertTrue("and must have served hits, it served " + hits, hits > 0);
                    assertTrue(DirectIOVectorCache.slabBytes() > 0);

                    // Off: the same bytes, through the delegate, and the store is released.
                    enabled.set(false);
                    assertVectorsEqual("the delegate", expected, ordinals, burst(routed, ordinals));
                    assertEquals("the store must have been released", 0L, DirectIOVectorCache.slabBytes());
                    assertEquals("and hold nothing", 0L, DirectIOVectorCache.entries());
                    assertEquals("no read may have been served from it", hits, DirectIOVectorCache.hits());
                    assertEquals(
                        "every read must have gone through the delegate",
                        ordinals.length,
                        DirectIOVectorIndexInput.STATS.fallbackReads()
                    );

                    // On again: the store comes back and refills.
                    enabled.set(true);
                    assertVectorsEqual("O_DIRECT again", expected, ordinals, burst(routed, ordinals));
                    assertEquals("the target is read again at the next burst", (long) TEST_SLAB_BYTES, DirectIOVectorCache.slabBytes());
                    assertTrue("and the store refills", DirectIOVectorCache.entries() > 0);
                }
            }
        }
    }

    /**
     * The delegate's input is closed exactly once, and only by the input {@code open} returned: a clone or a
     * slice closing one would close bytes its parent is still reading. Driven through
     * {@link DirectIOVectorIndexInput#open(Path, IndexInput, java.util.function.BooleanSupplier)} rather
     * than through the directory, so that the counting wrapper is the input under test rather than one more
     * layer the dispatch rule has to resolve a path through. The directory's own side of the ownership — the
     * release taken when an open is declined after both inputs are open — is pinned by
     * {@code KNNDirectIODirectoryTests#testTheContentDeclineReleasesBothInputsAndEvictsNothing}.
     */
    @SneakyThrows
    public void testTheDelegateInputIsClosedExactlyOnceAndOnlyByItsOwner() {
        assumeDirectIOWorksHere();
        final AtomicInteger closes = new AtomicInteger();
        try (MMapDirectory delegate = new MMapDirectory(createTempDir())) {
            writeVectors(delegate);
            final Path file = delegate.getDirectory().resolve(VECTOR_DATA);
            final IndexInput counting = new CountingCloseInput(delegate.openInput(VECTOR_DATA, vectorDataContext()), closes);
            final DirectIOVectorIndexInput routed = DirectIOVectorIndexInput.open(file, counting, () -> true);
            assertNotNull(routed);
            assertTrue(routed.hasMmapFallback());
            assertEquals("opening must not close anything", 0, closes.get());

            routed.slice("vectors", 0, routed.length()).close();
            routed.clone().close();
            assertEquals("a slice and a clone own no delegate input", 0, closes.get());

            routed.close();
            assertEquals("the owner closes it, once", 1, closes.get());
            routed.close();
            assertEquals("and a second close is a no-op", 1, closes.get());
        }
    }

    /**
     * With the setting off at {@code openInput} the file is not routed at all, so no second input is opened
     * and no {@code O_DIRECT} channel exists. That is today's rule and O1 does not change it: the enable
     * direction stays gradual, reaching only segments opened after the flip, because the alternative is an
     * {@code O_DIRECT} channel for every {@code .vec} on a node that will never use one.
     */
    @SneakyThrows
    public void testARoutedOpenNeverHappensWithTheSettingOff() {
        assumeDirectIOWorksHere();
        try (MMapDirectory delegate = new MMapDirectory(createTempDir())) {
            writeVectors(delegate);
            try (KNNDirectIODirectory storage = new KNNDirectIODirectory(delegate, "_0", () -> false, () -> 0L)) {
                DirectIOVectorIndexInput.STATS.reset();
                try (IndexInput opened = storage.openInput(VECTOR_DATA, vectorDataContext())) {
                    assertFalse("the file must not route", opened instanceof DirectIOVectorIndexInput);
                    // And it is readable, which is the point: the delegate's own input, unwrapped.
                    final float[] read = new float[DIMENSION];
                    opened.seek(0);
                    opened.readFloats(read, 0, DIMENSION);
                }
                assertEquals("nothing was routed", 0L, storage.routedOpens());
                assertEquals("and no decline reason was counted either: condition 2 is before them", 0L, storage.declinedOpens());
                assertEquals("so no input ever changed route", 0L, DirectIOVectorIndexInput.STATS.switchedToMmap());
                assertEquals(0L, DirectIOVectorIndexInput.STATS.switchedToDirect());
            }
        }
    }

    /**
     * Counts {@link #close()} on the input it wraps, and on nothing else: a clone or a slice of it is the
     * delegate's own object and is not wrapped, which is exactly the distinction the test is making.
     */
    private static final class CountingCloseInput extends FilterIndexInput {
        private final AtomicInteger closes;

        CountingCloseInput(final IndexInput in, final AtomicInteger closes) {
            super("counting(" + in + ")", in);
            this.closes = closes;
        }

        @Override
        public void close() throws IOException {
            closes.incrementAndGet();
            super.close();
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
     * A store of {@code slabs} test-sized slabs, with the read-ahead ring off so the burst table and the
     * blocking window are the only two channels. {@code rescore.enabled} is left at its default here and
     * driven by the directory's own supplier, which is the one the inputs read.
     */
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
        DirectIOVectorCache.configure(maxBytes);
    }
}

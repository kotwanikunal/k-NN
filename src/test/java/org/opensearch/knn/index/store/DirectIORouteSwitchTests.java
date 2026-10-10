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
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.store.MMapDirectory;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.codec.KNN990Codec.NativeEngines990KnnVectorsFormat;

import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.OpenOption;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * I6 (D2): {@code knn.direct_io.rescore.enabled} reaches an input that is <b>already open</b>.
 *
 * <p>The decision this proves is the one the design review took: there is no automatic switching between
 * mmap and Direct I/O, the setting is the operator's lever, and it has to be a lever that works now. Before
 * this, the dispatch rule was taken at {@code openInput} and an input kept its {@code O_DIRECT} channel for
 * life, so turning the feature off left every open reader reading the device until the next refresh --
 * which is not a rollback. A routed input now carries the delegate's own input beside the channel and
 * chooses between them at each prefetch burst boundary.
 *
 * <p>Three properties, and the third is the one that makes the first two safe to have:
 * <ul>
 *   <li>the route changes at the next burst, with no reopen, in both directions;</li>
 *   <li>the bytes are the same bytes on either route, so a flip mid-query cannot change a score;</li>
 *   <li>the change happens only at a burst boundary, never between a prefetch and the read that consumes
 *       it, because a staged read and its reader have to be on the same channel.</li>
 * </ul>
 */
@ThreadLeakFilters(defaultFilters = true, filters = { DirectIOReadPoolTests.ReadPoolThreadFilter.class })
public class DirectIORouteSwitchTests extends KNNTestCase {

    /** A native-engine (faiss / MOS) fp32 flat file: the shape the dispatch rule routes. */
    private static final String VECTOR_DATA = "_0_" + NativeEngines990KnnVectorsFormat.FORMAT_NAME + "_0.vec";

    private static final int DIMENSION = 128;
    private static final int VECTORS = 512;
    private static final int VECTOR_BYTES = DIMENSION * Float.BYTES;

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
     * The whole of I6 in one test: one input, opened once, serving three bursts on two different channels
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
                    assertEquals("and nothing should have gone through the fallback", 0L, DirectIOVectorIndexInput.STATS.fallbackReads());
                    assertTrue(routed.isDirectIOActive());

                    // The operator turns it off. Nothing is reopened.
                    enabled.set(false);
                    assertVectorsEqual("the mmap fallback", expected, ordinals, burst(routed, ordinals));
                    assertFalse("the input must have changed route", routed.isDirectIOActive());
                    assertEquals("exactly one input changed over", 1L, DirectIOVectorIndexInput.STATS.switchedToMmap());
                    assertEquals(
                        "every read of the second burst must be a fallback read",
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
                    assertEquals("the fallback must stop serving reads", ordinals.length, DirectIOVectorIndexInput.STATS.fallbackReads());
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
     * A slice and a clone each carry their own view of the fallback, with this input's coordinates, so the
     * switch reaches them too. This is not a detail of the test: {@code OffHeapFloatVectorValues} holds a
     * <em>slice</em> and nothing else, so a slice that could not switch would mean the switch never reached
     * the rescore path at all.
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
                    assertVectorsEqual("a slice on the fallback", expected, ordinals, burst(slice, ordinals));
                    assertVectorsEqual("a clone on the fallback", expected, ordinals, burst(clone, ordinals));
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
                    assertEquals("so no read may go through the fallback yet", 0L, DirectIOVectorIndexInput.STATS.fallbackReads());

                    // The next burst picks it up.
                    assertVectorsEqual("the next burst", expected, ordinals, burst(routed, ordinals));
                    assertFalse(((DirectIOVectorIndexInput) routed).isDirectIOActive());
                    assertEquals(ordinals.length, DirectIOVectorIndexInput.STATS.fallbackReads());
                }
            }
        }
    }

    /**
     * An input opened with no fallback -- every caller outside the directory, and the directory itself when
     * the delegate open failed -- stays on {@code O_DIRECT} whatever the switch says, because a read with
     * nowhere to go is worse than a read on a channel the operator has turned off. The phase 9 probes and
     * every existing measurement in this package open inputs this way, so this is also what keeps them
     * measuring what they were taken against.
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
                assertTrue("with no fallback the route cannot change", bare.isDirectIOActive());
                assertEquals(0L, DirectIOVectorIndexInput.STATS.fallbackReads());
                assertEquals(0L, DirectIOVectorIndexInput.STATS.switchedToMmap());
            }
        }
    }
}

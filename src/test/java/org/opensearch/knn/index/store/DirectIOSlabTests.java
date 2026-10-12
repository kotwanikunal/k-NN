/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import org.opensearch.knn.KNNTestCase;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;

/**
 * Slab memory without JNI (M1), on whichever backend this runtime selects and on the fallback as well.
 *
 * <p>The properties that have to hold, in the order they would hurt: a slab hands back the bytes that were
 * written into it, releasing it is idempotent, and reading one that has been released cannot take the JVM
 * down. The last of those is the reason this class exists rather than a {@code ByteBuffer} field: a JNI slab
 * gives the JVM no way to reject a read after the free, and both backends here do reject it or survive it.
 */
public class DirectIOSlabTests extends KNNTestCase {

    /** Small enough to allocate a few dozen in a suite, and a whole number of floats. */
    private static final int SMALL_SLAB = 64 * 1024;

    /**
     * Both backends where FFM works, which is every JDK this is built or run on today (MEASURED on Corretto
     * 21 and Temurin 25, output/bench/m1), and the fallback alone on a runtime where it does not. Listing
     * {@code ffm} unconditionally would turn "this JDK has no FFM" into a test failure, which is the one
     * thing the fallback exists so it is not.
     */
    private static final List<String> BACKENDS = DirectIOSlab.ffmUsable()
        ? List.of(DirectIOSlab.BACKEND_FFM, DirectIOSlab.BACKEND_DIRECT_BUFFER)
        : List.of(DirectIOSlab.BACKEND_DIRECT_BUFFER);

    @Override
    public void tearDown() throws Exception {
        DirectIOSlab.resetCountersForTesting();
        super.tearDown();
    }

    /** The selected backend is one of the two names the stats publish, and FFM is selected where it works. */
    public void testBackendIsNamedAndSelected() {
        final String backend = DirectIOSlab.backend();
        assertTrue(backend, BACKENDS.contains(backend));
        assertEquals(DirectIOSlab.ffmUsable() ? DirectIOSlab.BACKEND_FFM : DirectIOSlab.BACKEND_DIRECT_BUFFER, backend);
        assertEquals(backend, DirectIOVectorCache.slabMemory());
    }

    /** Allocate, write, read back, release, on each backend. */
    public void testWriteAndReadBackOnBothBackends() {
        for (final String backend : BACKENDS) {
            final DirectIOSlab slab = DirectIOSlab.allocate(SMALL_SLAB, backend);
            assertNotNull(backend, slab);
            try {
                assertEquals(backend, SMALL_SLAB, slab.capacity());
                assertFalse(backend, slab.isClosed());
                assertEquals(backend, ByteOrder.LITTLE_ENDIAN, slab.bytes().order());
                assertTrue(backend, slab.bytes().isDirect());

                final int floats = SMALL_SLAB / Float.BYTES;
                for (int i = 0; i < floats; i++) {
                    slab.floats().put(i, i * 1.5f);
                }
                for (int i = 0; i < floats; i++) {
                    assertEquals(backend + " at " + i, i * 1.5f, slab.floats().get(i), 0.0f);
                }
                // And through the byte view, which is the one a fill writes with absolute put(position, ...).
                final float[] read = new float[8];
                slab.bytes().asFloatBuffer().get(0, read, 0, read.length);
                for (int i = 0; i < read.length; i++) {
                    assertEquals(backend + " byte view at " + i, i * 1.5f, read[i], 0.0f);
                }
            } finally {
                slab.close();
            }
            assertTrue(backend, slab.isClosed());
        }
    }

    /** A copy in and a copy out through the absolute forms the cache uses, which is the whole hot path. */
    public void testAbsoluteCopyInAndOut() {
        for (final String backend : BACKENDS) {
            final DirectIOSlab slab = DirectIOSlab.allocate(SMALL_SLAB, backend);
            assertNotNull(backend, slab);
            try {
                final int dimension = 128;
                final float[] source = new float[dimension];
                for (int i = 0; i < dimension; i++) {
                    source[i] = i + 0.25f;
                }
                final ByteBuffer staged = ByteBuffer.allocate(dimension * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
                staged.asFloatBuffer().put(0, source);

                final int slot = 3;
                final int slotBytes = dimension * Float.BYTES;
                slab.bytes().put(slot * slotBytes, staged, 0, slotBytes);

                final float[] out = new float[dimension];
                slab.floats().get((slot * slotBytes) >> 2, out, 0, dimension);
                assertArrayEquals(backend, source, out, 0.0f);
            } finally {
                slab.close();
            }
        }
    }

    /** A second release is a no-op rather than the {@code IllegalStateException} a shared arena would throw. */
    public void testSecondReleaseIsANoOp() {
        for (final String backend : BACKENDS) {
            DirectIOSlab.resetCountersForTesting();
            final DirectIOSlab slab = DirectIOSlab.allocate(SMALL_SLAB, backend);
            assertNotNull(backend, slab);
            slab.close();
            slab.close();
            slab.close();
            assertTrue(backend, slab.isClosed());
            assertEquals(backend, 1L, DirectIOSlab.allocatedCount());
            assertEquals(backend, 1L, DirectIOSlab.freedCount());
        }
    }

    /**
     * Reading a released slab never takes the JVM down. What it does instead differs by backend and both are
     * asserted here rather than hidden behind one weaker assertion: FFM rejects the read, a direct buffer
     * still returns the old bytes until a collection unmaps them. The store never relies on either, because a
     * slab is closed only under its write lock, which every reader excludes.
     */
    public void testReadingAReleasedSlabDoesNotCrash() {
        for (final String backend : BACKENDS) {
            final DirectIOSlab slab = DirectIOSlab.allocate(SMALL_SLAB, backend);
            assertNotNull(backend, slab);
            // A view kept from before the close, which is the only way to reach the memory afterwards.
            final ByteBuffer kept = slab.bytes();
            kept.putFloat(0, 7.5f);
            slab.close();

            // The slab's own accessors refuse, on both backends.
            expectThrows(IllegalStateException.class, slab::bytes);
            expectThrows(IllegalStateException.class, slab::floats);

            if (DirectIOSlab.BACKEND_FFM.equals(backend)) {
                final IllegalStateException e = expectThrows(IllegalStateException.class, () -> kept.getFloat(0));
                assertTrue(String.valueOf(e.getMessage()), e.getMessage().toLowerCase().contains("closed"));
            } else {
                assertEquals(backend, 7.5f, kept.getFloat(0), 0.0f);
            }
        }
    }

    /** A capacity of zero or less is a programming error, not a null. */
    public void testNonPositiveCapacityIsRejected() {
        expectThrows(IllegalArgumentException.class, () -> DirectIOSlab.allocate(0));
        expectThrows(IllegalArgumentException.class, () -> DirectIOSlab.allocate(-1));
    }

    /** Several slabs at once hold distinct memory: slab i's bytes are not slab j's. */
    public void testSlabsDoNotOverlap() {
        for (final String backend : BACKENDS) {
            final DirectIOSlab[] slabs = new DirectIOSlab[4];
            try {
                for (int i = 0; i < slabs.length; i++) {
                    slabs[i] = DirectIOSlab.allocate(SMALL_SLAB, backend);
                    assertNotNull(backend, slabs[i]);
                    slabs[i].floats().put(0, i + 100.0f);
                }
                for (int i = 0; i < slabs.length; i++) {
                    assertEquals(backend + " slab " + i, i + 100.0f, slabs[i].floats().get(0), 0.0f);
                }
            } finally {
                for (final DirectIOSlab slab : slabs) {
                    if (slab != null) {
                        slab.close();
                    }
                }
            }
        }
    }

    /** A full-size slab, because 32 MiB is what production allocates and the backends differ in cost there. */
    public void testFullSizeSlab() {
        final DirectIOSlab slab = DirectIOSlab.allocate(DirectIOSlab.SLAB_BYTES);
        assertNotNull(slab);
        try {
            assertEquals(DirectIOSlab.SLAB_BYTES, slab.capacity());
            slab.floats().put(0, 1.0f);
            slab.floats().put((DirectIOSlab.SLAB_BYTES >> 2) - 1, -1.0f);
            assertEquals(1.0f, slab.floats().get(0), 0.0f);
            assertEquals(-1.0f, slab.floats().get((DirectIOSlab.SLAB_BYTES >> 2) - 1), 0.0f);
        } finally {
            slab.close();
        }
    }

    /** The allocated and freed counters are what a leak test reads, so they have to move in step. */
    public void testCountersTrackTheLivePopulation() {
        DirectIOSlab.resetCountersForTesting();
        final DirectIOSlab first = DirectIOSlab.allocate(SMALL_SLAB);
        final DirectIOSlab second = DirectIOSlab.allocate(SMALL_SLAB);
        assertEquals(2L, DirectIOSlab.allocatedCount());
        assertEquals(0L, DirectIOSlab.freedCount());
        first.close();
        assertEquals(1L, DirectIOSlab.freedCount());
        second.close();
        assertEquals(DirectIOSlab.allocatedCount(), DirectIOSlab.freedCount());
    }
}

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.memory;

import org.opensearch.knn.KNNTestCase;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

/**
 * I1's acceptance tests for {@link VectorCacheSlab}: allocate, write, read back, free, double free is safe,
 * and use after free is rejected without taking the JVM with it.
 *
 * <p>The last of those is the one worth reading carefully. A direct {@link ByteBuffer} over freed
 * {@code malloc} memory is a valid Java object pointing at memory the allocator may have reused, so the JVM
 * cannot reject a read through a reference a caller already holds. What {@link VectorCacheSlab} guarantees
 * instead, and what these tests check, is that the slab will not <em>hand out</em> a view once it is closed
 * and that a close cannot begin while a reader holds the read lock.
 */
public class VectorCacheSlabTests extends KNNTestCase {

    /** Small enough to keep the test fast; the shipped size is exercised by testShippedSlabSize. */
    private static final int TEST_SLAB_BYTES = 1 << 20;

    public void testAllocateWriteReadBackFree() {
        final VectorCacheSlab slab = VectorCacheSlab.allocate(TEST_SLAB_BYTES);
        try {
            assertEquals(TEST_SLAB_BYTES, slab.capacity());
            assertFalse(slab.isClosed());
            assertNotEquals(0L, slab.address());

            slab.readLock();
            try {
                final ByteBuffer bytes = slab.bytes();
                assertTrue(bytes.isDirect());
                assertEquals(TEST_SLAB_BYTES, bytes.capacity());
                assertEquals(ByteOrder.LITTLE_ENDIAN, bytes.order());

                final int count = TEST_SLAB_BYTES / Float.BYTES;
                for (int i = 0; i < count; i++) {
                    bytes.putFloat(i * Float.BYTES, i * 0.25f);
                }
                for (int i = 0; i < count; i++) {
                    assertEquals(i * 0.25f, bytes.getFloat(i * Float.BYTES), 0.0f);
                }

                // The float view sees the same bytes, which is what makes a hit scoreable without a copy.
                final FloatBuffer floats = slab.floats();
                assertEquals(count, floats.capacity());
                final float[] read = new float[1024];
                floats.get(0, read, 0, read.length);
                for (int i = 0; i < read.length; i++) {
                    assertEquals(i * 0.25f, read[i], 0.0f);
                }
            } finally {
                slab.readUnlock();
            }
        } finally {
            slab.close();
        }
        assertTrue(slab.isClosed());
    }

    public void testDoubleFreeIsSafe() {
        final VectorCacheSlab slab = VectorCacheSlab.allocate(TEST_SLAB_BYTES);
        slab.close();
        assertTrue(slab.isClosed());
        // Three more, because a crash here would take the whole JVM and not just this test.
        slab.close();
        slab.close();
        slab.close();
        assertTrue(slab.isClosed());
    }

    public void testUseAfterFreeIsRejected() {
        final VectorCacheSlab slab = VectorCacheSlab.allocate(TEST_SLAB_BYTES);
        slab.close();

        slab.readLock();
        try {
            expectThrows(IllegalStateException.class, slab::bytes);
            expectThrows(IllegalStateException.class, slab::floats);
        } finally {
            slab.readUnlock();
        }
        // Still alive, which is the actual assertion: the rejection was an exception and not a SIGSEGV.
        assertTrue(slab.isClosed());
    }

    public void testCloseWaitsForAnInFlightRead() throws Exception {
        final VectorCacheSlab slab = VectorCacheSlab.allocate(TEST_SLAB_BYTES);
        final CountDownLatch readerHasLock = new CountDownLatch(1);
        final CountDownLatch readerMayRelease = new CountDownLatch(1);
        final AtomicReference<Throwable> readerFailure = new AtomicReference<>();

        final Thread reader = new Thread(() -> {
            slab.readLock();
            try {
                slab.bytes().putFloat(0, 42.0f);
                readerHasLock.countDown();
                assertTrue(readerMayRelease.await(30, TimeUnit.SECONDS));
                // The slab must still be alive here, because close() cannot have taken the write lock.
                assertEquals(42.0f, slab.bytes().getFloat(0), 0.0f);
            } catch (Throwable t) {
                readerFailure.set(t);
            } finally {
                slab.readUnlock();
            }
        }, "slab-reader");
        reader.start();
        assertTrue(readerHasLock.await(30, TimeUnit.SECONDS));

        final Thread closer = new Thread(slab::close, "slab-closer");
        closer.start();
        // The closer is blocked on the write lock. Give it long enough that a missing lock would show.
        Thread.sleep(200);
        assertFalse("close() must not complete while a reader holds the read lock", slab.isClosed());

        readerMayRelease.countDown();
        reader.join(30_000);
        closer.join(30_000);
        assertNull(readerFailure.get());
        assertTrue(slab.isClosed());
    }

    public void testEveryAllocationIsFreed() {
        VectorCacheSlab.resetCountersForTesting();
        final List<VectorCacheSlab> slabs = new ArrayList<>();
        for (int i = 0; i < 8; i++) {
            slabs.add(VectorCacheSlab.allocate(TEST_SLAB_BYTES));
        }
        assertEquals(8L, VectorCacheSlab.allocatedCount());
        assertEquals(0L, VectorCacheSlab.freedCount());
        for (final VectorCacheSlab slab : slabs) {
            slab.close();
        }
        assertEquals(8L, VectorCacheSlab.allocatedCount());
        assertEquals(8L, VectorCacheSlab.freedCount());
        // A double free must not double count, or the live population would go negative.
        for (final VectorCacheSlab slab : slabs) {
            slab.close();
        }
        assertEquals(8L, VectorCacheSlab.freedCount());
        VectorCacheSlab.resetCountersForTesting();
    }

    public void testShippedSlabSizeAllocatesAndFrees() {
        assertEquals(32 * 1024 * 1024, VectorCacheSlab.SLAB_BYTES);
        final VectorCacheSlab slab = VectorCacheSlab.allocate();
        try {
            assertEquals(VectorCacheSlab.SLAB_BYTES, slab.capacity());
            slab.readLock();
            try {
                // Touch the first and last page, so the test would fail if the capacity were a lie.
                final ByteBuffer bytes = slab.bytes();
                bytes.put(0, (byte) 7);
                bytes.put(VectorCacheSlab.SLAB_BYTES - 1, (byte) 9);
                assertEquals((byte) 7, bytes.get(0));
                assertEquals((byte) 9, bytes.get(VectorCacheSlab.SLAB_BYTES - 1));
            } finally {
                slab.readUnlock();
            }
        } finally {
            slab.close();
        }
    }

    public void testVectorSizedSlotsAcrossTheDimensionRange() {
        // The dimensions I2 has to hold, from the smallest this plugin supports to the largest.
        for (final int dimension : new int[] { 128, 768, 1024, 4096 }) {
            final int vectorBytes = dimension * Float.BYTES;
            final VectorCacheSlab slab = VectorCacheSlab.allocate(TEST_SLAB_BYTES);
            try {
                final int slots = TEST_SLAB_BYTES / vectorBytes;
                assertTrue("a 1 MiB slab must hold at least one " + dimension + "-dim vector", slots >= 1);
                slab.readLock();
                try {
                    final FloatBuffer floats = slab.floats();
                    for (int slot = 0; slot < slots; slot++) {
                        for (int i = 0; i < dimension; i++) {
                            floats.put(slot * dimension + i, slot + i * 0.5f);
                        }
                    }
                    final float[] read = new float[dimension];
                    for (int slot = 0; slot < slots; slot++) {
                        floats.get(slot * dimension, read, 0, dimension);
                        for (int i = 0; i < dimension; i++) {
                            assertEquals(slot + i * 0.5f, read[i], 0.0f);
                        }
                    }
                } finally {
                    slab.readUnlock();
                }
            } finally {
                slab.close();
            }
        }
    }

    public void testZeroAndNegativeCapacityAreRejected() {
        expectThrows(IllegalArgumentException.class, () -> VectorCacheSlab.allocate(0));
        expectThrows(IllegalArgumentException.class, () -> VectorCacheSlab.allocate(-1));
    }

    public void testSlabIsOutsideTheJvmDirectMemoryPool() {
        // The whole reason this is native and not ByteBuffer.allocateDirect. See the class javadoc.
        final long before = directPoolBytes();
        final VectorCacheSlab slab = VectorCacheSlab.allocate();
        try {
            slab.readLock();
            try {
                slab.bytes().put(0, (byte) 1);
            } finally {
                slab.readUnlock();
            }
            assertEquals("a slab must not be charged to -XX:MaxDirectMemorySize", before, directPoolBytes());
        } finally {
            slab.close();
        }
        assertEquals(before, directPoolBytes());
    }

    private static long directPoolBytes() {
        for (final java.lang.management.BufferPoolMXBean pool : java.lang.management.ManagementFactory.getPlatformMXBeans(
            java.lang.management.BufferPoolMXBean.class
        )) {
            if ("direct".equals(pool.getName())) {
                return pool.getMemoryUsed();
            }
        }
        return -1;
    }
}

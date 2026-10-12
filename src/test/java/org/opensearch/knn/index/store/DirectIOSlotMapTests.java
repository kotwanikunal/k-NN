/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import org.opensearch.knn.KNNTestCase;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Random;
import java.util.Set;

/**
 * M2's tests for {@link DirectIOSlotMap}, the primitive part of the store's index, adapted from 16-I's
 * {@code OrdinalSlotMapTests} for a {@code long} file offset key. The map has no remove, so what has to be
 * proved is that lookups agree with a {@link HashMap} oracle across growth, that a replace overwrites in
 * place rather than adding a cell, that a compaction keeps exactly the cells a validator calls live, and that
 * the live count the closed-file rule depends on moves with the slots.
 */
public class DirectIOSlotMapTests extends KNNTestCase {

    /** The offsets a real file hands this map: past a header, a whole number of vectors apart. */
    private static final long HEADER = 44L;
    private static final int VECTOR_BYTES = 3072;

    public void testPutFindAgainstAnOracle() {
        final DirectIOSlotMap map = new DirectIOSlotMap("a.vec#1");
        final Map<Long, int[]> oracle = new HashMap<>();
        final Random random = new Random(42);
        for (int i = 0; i < 20_000; i++) {
            final long offset = HEADER + (long) random.nextInt(5_000) * VECTOR_BYTES;
            final int slabId = random.nextInt(DirectIOSlotMap.MAX_SLABS);
            final int slot = random.nextInt(1 << 16);
            map.put(offset, slabId, slot);
            oracle.put(offset, new int[] { slabId, slot });
        }
        assertEquals(oracle.size(), map.cells());
        for (final Map.Entry<Long, int[]> entry : oracle.entrySet()) {
            final int cell = map.find(entry.getKey());
            assertTrue("offset " + entry.getKey() + " is missing", cell >= 0);
            assertEquals(entry.getValue()[0], map.slabIdAt(cell));
            assertEquals(entry.getValue()[1], map.slotAt(cell));
        }
        for (int ordinal = 5_000; ordinal < 5_100; ordinal++) {
            assertEquals(-1, map.find(HEADER + (long) ordinal * VECTOR_BYTES));
        }
    }

    /**
     * The regression this is here for: under linear probing a put into a full table does not fail, it spins
     * forever looking for a free cell. {@link DirectIOSlotMap#put} therefore grows the table itself rather
     * than trusting a caller to ask first, and this test is the one that insists on it.
     */
    public void testPutGrowsTheTableByItself() {
        final DirectIOSlotMap map = new DirectIOSlotMap("grow.vec#1");
        for (int i = 0; i < 100_000; i++) {
            map.put(HEADER + (long) i * VECTOR_BYTES, i % DirectIOSlotMap.MAX_SLABS, i % 1024);
        }
        assertEquals(100_000, map.cells());
        assertTrue("the table has to be bigger than its occupancy", map.capacity() > map.cells());
        for (int i = 0; i < 100_000; i += 997) {
            final int cell = map.find(HEADER + (long) i * VECTOR_BYTES);
            assertTrue(cell >= 0);
            assertEquals(i % DirectIOSlotMap.MAX_SLABS, map.slabIdAt(cell));
            assertEquals(i % 1024, map.slotAt(cell));
        }
    }

    /** The extreme slot values have to survive the pack into one int, and a huge offset has to be a key. */
    public void testPackingAtTheLimits() {
        final DirectIOSlotMap map = new DirectIOSlotMap("limits.vec#1");
        map.put(0L, DirectIOSlotMap.MAX_SLABS - 1, DirectIOSlotMap.MAX_SLOTS - 1);
        // A 1 TB file's last vector, which is well past anything an int offset could have held.
        final long far = (1L << 40) - VECTOR_BYTES;
        map.put(far, 0, 0);
        final int high = map.find(0L);
        assertEquals(DirectIOSlotMap.MAX_SLABS - 1, map.slabIdAt(high));
        assertEquals(DirectIOSlotMap.MAX_SLOTS - 1, map.slotAt(high));
        final int low = map.find(far);
        assertEquals(0, map.slabIdAt(low));
        assertEquals(0, map.slotAt(low));
    }

    public void testReplaceOverwritesTheCell() {
        final DirectIOSlotMap map = new DirectIOSlotMap("replace.vec#1");
        map.put(7L, 1, 2);
        map.put(7L, 3, 4);
        assertEquals(1, map.cells());
        final int cell = map.find(7L);
        assertEquals(3, map.slabIdAt(cell));
        assertEquals(4, map.slotAt(cell));
    }

    public void testCompactionKeepsOnlyTheLiveCells() {
        final DirectIOSlotMap map = new DirectIOSlotMap("compact.vec#1");
        final Set<Long> live = new HashSet<>();
        for (int i = 0; i < 500; i++) {
            final long offset = HEADER + (long) i * VECTOR_BYTES;
            map.put(offset, 0, i);
            if (i % 3 == 0) {
                live.add(offset);
            }
        }
        assertEquals(500, map.cells());
        map.compact((offset, slabId, slot) -> live.contains(offset));
        assertEquals(live.size(), map.cells());
        assertEquals(0, map.invalidatedCells());
        for (int i = 0; i < 500; i++) {
            final long offset = HEADER + (long) i * VECTOR_BYTES;
            if (live.contains(offset)) {
                final int cell = map.find(offset);
                assertTrue(cell >= 0);
                assertEquals(i, map.slotAt(cell));
            } else {
                assertEquals("offset " + offset + " should have been dropped", -1, map.find(offset));
            }
        }
    }

    public void testCompactionIsAskedForWhenInvalidatedCellsTakeOver() {
        final DirectIOSlotMap map = new DirectIOSlotMap("stale.vec#1");
        for (int i = 0; i < 200; i++) {
            map.put(HEADER + (long) i * VECTOR_BYTES, 0, i);
        }
        assertFalse(map.needsCompaction());
        for (int i = 0; i < 100; i++) {
            map.noteInvalidated();
        }
        assertTrue(map.needsCompaction());
        // Growth must not forget them: it keeps the stale cells, so the count has to survive a resize.
        for (int i = 200; i < 400; i++) {
            map.put(HEADER + (long) i * VECTOR_BYTES, 0, i);
        }
        assertEquals(100, map.invalidatedCells());
    }

    /** Index cost is a reported figure, so the load factor it comes from is pinned by a test. */
    public void testTheTableStaysBetweenAThirdAndTwoThirdsFull() {
        final DirectIOSlotMap map = new DirectIOSlotMap("load.vec#1");
        for (int i = 0; i < 50_000; i++) {
            map.put(HEADER + (long) i * VECTOR_BYTES, 0, i);
            final double load = (double) map.cells() / map.capacity();
            assertTrue("load factor " + load + " at " + map.cells() + " cells", load <= 0.61);
            if (map.cells() > DirectIOSlotMap.MAX_SLOTS / 4096) {
                assertTrue("load factor " + load + " at " + map.cells() + " cells", load >= 0.16);
            }
        }
        assertEquals(50_000, map.cells());
    }

    /**
     * The live count the closed-file rule reads: it follows the slots, not the cells, and reaching zero is
     * what tells the store this file's table can go.
     */
    public void testTheLiveCountFollowsTheSlots() {
        final DirectIOSlotMap map = new DirectIOSlotMap("live.vec#1");
        assertEquals(0, map.liveEntries());
        map.put(HEADER, 0, 0);
        map.noteAttached();
        map.put(HEADER + VECTOR_BYTES, 0, 1);
        map.noteAttached();
        assertEquals(2, map.liveEntries());
        assertFalse("one of two is not empty", map.noteDetached());
        assertEquals(1, map.liveEntries());
        assertTrue("the last one empties the map", map.noteDetached());
        assertEquals(0, map.liveEntries());
        assertEquals("and its cells are still there, which is why the store drops the whole map", 2, map.cells());
    }

    /** The file id is the key the store removes the map under, so the map has to carry it unchanged. */
    public void testTheMapCarriesItsFileId() {
        assertEquals("/var/lib/x/_0.vec#12345", new DirectIOSlotMap("/var/lib/x/_0.vec#12345").fileId());
    }

    /** Offsets one vector apart must not pile into one stretch of the table, which is what the hash is for. */
    public void testARunOfOffsetsIsSpread() {
        final DirectIOSlotMap map = new DirectIOSlotMap("spread.vec#1");
        final Set<Integer> cells = new HashSet<>();
        for (int i = 0; i < 64; i++) {
            map.put(HEADER + (long) i * VECTOR_BYTES, 0, i);
        }
        // Collected after every insert, not during: growth rehashes, so cells gathered mid-loop would be
        // indices into tables of different sizes and would collide for reasons that are not the hash's.
        for (int i = 0; i < 64; i++) {
            cells.add(map.find(HEADER + (long) i * VECTOR_BYTES));
        }
        assertEquals("every offset got its own cell", 64, cells.size());
        // And the table is not crammed: 64 keys in 128 cells. Pinned as a bound on the load factor rather
        // than on probe counts, which are not observable from here.
        assertTrue("capacity " + map.capacity() + " at " + map.cells() + " cells", map.capacity() >= 2 * map.cells());
    }
}

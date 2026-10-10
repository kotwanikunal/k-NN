/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.memory;

import org.opensearch.knn.KNNTestCase;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Random;
import java.util.Set;

/**
 * I2's tests for {@link OrdinalSlotMap}, the primitive part of the vector cache's index. The map has no
 * remove, so what has to be proved is that lookups agree with a {@link HashMap} oracle across growth, that
 * a replace overwrites in place rather than adding a cell, and that a compaction keeps exactly the cells a
 * validator calls live.
 */
public class OrdinalSlotMapTests extends KNNTestCase {

    public void testPutFindAgainstAnOracle() {
        final OrdinalSlotMap map = new OrdinalSlotMap();
        final Map<Integer, int[]> oracle = new HashMap<>();
        final Random random = new Random(42);
        for (int i = 0; i < 20_000; i++) {
            final int ordinal = random.nextInt(5_000);
            final int slabId = random.nextInt(OrdinalSlotMap.MAX_SLABS);
            final int slot = random.nextInt(1 << 16);
            map.put(ordinal, slabId, slot);
            oracle.put(ordinal, new int[] { slabId, slot });
        }
        assertEquals(oracle.size(), map.cells());
        for (final Map.Entry<Integer, int[]> entry : oracle.entrySet()) {
            final int cell = map.find(entry.getKey());
            assertTrue("ordinal " + entry.getKey() + " is missing", cell >= 0);
            assertEquals(entry.getValue()[0], map.slabIdAt(cell));
            assertEquals(entry.getValue()[1], map.slotAt(cell));
        }
        for (int ordinal = 5_000; ordinal < 5_100; ordinal++) {
            assertEquals(-1, map.find(ordinal));
        }
    }

    /**
     * The regression this is here for: under linear probing a put into a full table does not fail, it spins
     * forever looking for a free cell. {@link OrdinalSlotMap#put} therefore grows the table itself rather
     * than trusting a caller to ask first, and this test is the one that insists on it.
     */
    public void testPutGrowsTheTableByItself() {
        final OrdinalSlotMap map = new OrdinalSlotMap();
        for (int ordinal = 0; ordinal < 100_000; ordinal++) {
            map.put(ordinal, ordinal % OrdinalSlotMap.MAX_SLABS, ordinal % 1024);
        }
        assertEquals(100_000, map.cells());
        assertTrue("the table has to be bigger than its occupancy", map.capacity() > map.cells());
        for (int ordinal = 0; ordinal < 100_000; ordinal += 997) {
            final int cell = map.find(ordinal);
            assertTrue(cell >= 0);
            assertEquals(ordinal % OrdinalSlotMap.MAX_SLABS, map.slabIdAt(cell));
            assertEquals(ordinal % 1024, map.slotAt(cell));
        }
    }

    /** The extreme slot values have to survive the pack into one int. */
    public void testPackingAtTheLimits() {
        final OrdinalSlotMap map = new OrdinalSlotMap();
        map.put(0, OrdinalSlotMap.MAX_SLABS - 1, OrdinalSlotMap.MAX_SLOTS - 1);
        map.put(Integer.MAX_VALUE, 0, 0);
        final int high = map.find(0);
        assertEquals(OrdinalSlotMap.MAX_SLABS - 1, map.slabIdAt(high));
        assertEquals(OrdinalSlotMap.MAX_SLOTS - 1, map.slotAt(high));
        final int low = map.find(Integer.MAX_VALUE);
        assertEquals(0, map.slabIdAt(low));
        assertEquals(0, map.slotAt(low));
    }

    public void testReplaceOverwritesTheCell() {
        final OrdinalSlotMap map = new OrdinalSlotMap();
        map.put(7, 1, 2);
        map.put(7, 3, 4);
        assertEquals(1, map.cells());
        final int cell = map.find(7);
        assertEquals(3, map.slabIdAt(cell));
        assertEquals(4, map.slotAt(cell));
    }

    public void testCompactionKeepsOnlyTheLiveCells() {
        final OrdinalSlotMap map = new OrdinalSlotMap();
        final Set<Integer> live = new HashSet<>();
        for (int ordinal = 0; ordinal < 500; ordinal++) {
            map.put(ordinal, 0, ordinal);
            if (ordinal % 3 == 0) {
                live.add(ordinal);
            }
        }
        assertEquals(500, map.cells());
        map.compact((ordinal, slabId, slot) -> live.contains(ordinal));
        assertEquals(live.size(), map.cells());
        assertEquals(0, map.invalidatedCells());
        for (int ordinal = 0; ordinal < 500; ordinal++) {
            if (live.contains(ordinal)) {
                final int cell = map.find(ordinal);
                assertTrue(cell >= 0);
                assertEquals(ordinal, map.slotAt(cell));
            } else {
                assertEquals("ordinal " + ordinal + " should have been dropped", -1, map.find(ordinal));
            }
        }
    }

    public void testCompactionIsAskedForWhenInvalidatedCellsTakeOver() {
        final OrdinalSlotMap map = new OrdinalSlotMap();
        for (int ordinal = 0; ordinal < 200; ordinal++) {
            map.put(ordinal, 0, ordinal);
        }
        assertFalse(map.needsCompaction());
        for (int i = 0; i < 100; i++) {
            map.noteInvalidated();
        }
        assertTrue(map.needsCompaction());
        // Growth must not forget them: it keeps the stale cells, so the count has to survive a resize.
        for (int ordinal = 200; ordinal < 400; ordinal++) {
            map.put(ordinal, 0, ordinal);
        }
        assertEquals(100, map.invalidatedCells());
    }

    /** Index cost is a reported figure, so the load factor it comes from is pinned by a test. */
    public void testTheTableStaysBetweenAThirdAndTwoThirdsFull() {
        final OrdinalSlotMap map = new OrdinalSlotMap();
        for (int ordinal = 0; ordinal < 50_000; ordinal++) {
            map.put(ordinal, 0, ordinal);
            final double load = (double) map.cells() / map.capacity();
            assertTrue("load factor " + load + " at " + map.cells() + " cells", load <= 0.61);
            if (map.cells() > OrdinalSlotMap.MAX_SLOTS / 4096) {
                assertTrue("load factor " + load + " at " + map.cells() + " cells", load >= 0.16);
            }
        }
        assertEquals(50_000, map.cells());
    }
}

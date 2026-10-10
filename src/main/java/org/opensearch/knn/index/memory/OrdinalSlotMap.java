/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.memory;

import java.util.Arrays;

/**
 * One segment field's part of the vector cache's index: a primitive map from a vector's ordinal to the slot
 * that holds it, as two {@code int} arrays and nothing else.
 *
 * <h2>Why not a {@code HashMap}</h2>
 * The index competes with vectors for the same byte budget, so its per-entry cost is subtracted from the
 * cache's capacity. A {@code HashMap<Integer, Integer>} entry is a 32 B {@code Node}, a 16 B boxed key, a
 * 16 B boxed value and a 4 B table slot: about 68 B per cached vector, which is 13% on top of a 128-dim
 * vector (512 B). Open addressing over two {@code int} arrays is 8 B per cell, so about 24 B per entry at
 * this map's load factor. It also allocates nothing on a lookup, which matters because a lookup is the hot
 * path: Lucene asks for 64 ordinals per batch (see FINDINGS I0.2).
 *
 * <h2>Not thread safe, and does not need to be</h2>
 * Every mutation here happens under {@link SlabVectorCache}'s write lock and every read under its read
 * lock, and those two exclude each other, so no reader ever observes a half-finished insert or rebuild. This
 * is the same reason the cache can hold its field index in a plain {@link java.util.HashMap}.
 *
 * <h2>Entries are invalidated lazily, so there is no remove</h2>
 * When a slab is evicted the entries pointing into it become wrong, and this map is not told which. It does
 * not need to be: the slot itself records which ordinal of which field index owns it, so a lookup confirms
 * that the slot still points back at the key it came from and treats a mismatch as a miss
 * ({@code SlabVectorCache#get}). A cell whose slot has been taken is therefore harmless, merely occupied,
 * and the cache counts them through {@link #noteInvalidated()} so that {@link #needsCompaction()} can ask for a
 * {@link #compact(SlotValidator)} once they outnumber the live ones. That removes the one piece of this map
 * that would otherwise be subtle, backward-shift deletion under linear probing, and it costs an equality
 * check that the read path has to make anyway.
 *
 * <h2>Ordinals are non-negative, so -1 is the empty cell</h2>
 * A Lucene vector ordinal is an index into a segment's vector values, so it is never negative. That frees
 * {@link #EMPTY} as the marker and keeps a cell to its two ints with no third array of flags.
 */
final class OrdinalSlotMap {

    /**
     * Confirms that a cell's slot is still the one holding that ordinal, so that a compaction can drop the cells
     * whose slots have been taken. Implemented by the cache, which is the only thing that can see a slot.
     */
    interface SlotValidator {
        boolean isLive(int ordinal, int slabId, int slot);
    }

    /** Free cell. A vector ordinal is an index into a segment's values and so is never negative. */
    private static final int EMPTY = -1;

    /** Smallest table. A segment field with one cached vector costs 128 B of index and no more. */
    private static final int MIN_CAPACITY = 16;

    /**
     * How a slot is packed into one int: {@code (slabId << SLOT_BITS) | slot}. 11 bits of slab id and 20 of
     * slot index fit in a positive int, which bounds the cache at {@link #MAX_SLABS} slabs (64 GiB at the
     * 32 MiB slab size) and a slab at {@link #MAX_SLOTS} slots. Both bounds are checked where slabs are
     * registered and opened, so nothing here has to defend against an overflowing shift.
     */
    private static final int SLOT_BITS = 20;
    static final int MAX_SLABS = 1 << 11;
    static final int MAX_SLOTS = 1 << SLOT_BITS;
    private static final int SLOT_MASK = MAX_SLOTS - 1;

    /** Ordinals, {@link #EMPTY} where the cell is free. Parallel to {@link #packed}. */
    private int[] ordinals;

    /** {@code (slabId << SLOT_BITS) | slot} for the ordinal in the matching cell of {@link #ordinals}. */
    private int[] packed;

    /** Occupied cells, live and invalidated together, which is what the load factor is measured against. */
    private int cells;

    /** Cells the cache has invalidated since the last rebuild. See the class javadoc. */
    private int invalidated;

    OrdinalSlotMap() {
        allocate(MIN_CAPACITY);
    }

    private void allocate(final int capacity) {
        ordinals = new int[capacity];
        Arrays.fill(ordinals, EMPTY);
        packed = new int[capacity];
        cells = 0;
        invalidated = 0;
    }

    /**
     * The cell holding {@code ordinal}, or {@code -1}. A cell is not proof that the vector is cached: the
     * caller must still check the slot's back pointer. See the class javadoc.
     */
    int find(final int ordinal) {
        final int[] keys = ordinals;
        final int mask = keys.length - 1;
        int i = hash(ordinal) & mask;
        while (true) {
            final int key = keys[i];
            if (key == ordinal) {
                return i;
            }
            if (key == EMPTY) {
                return -1;
            }
            i = (i + 1) & mask;
        }
    }

    /** The slab id recorded in {@code cell}. */
    int slabIdAt(final int cell) {
        return packed[cell] >>> SLOT_BITS;
    }

    /** The slot index recorded in {@code cell}. */
    int slotAt(final int cell) {
        return packed[cell] & SLOT_MASK;
    }

    /**
     * Points {@code ordinal} at {@code (slabId, slot)}, replacing any slot it pointed at before, and grows
     * the table first if this insert would take it past a 0.6 load factor.
     *
     * <p>It grows itself rather than trusting the caller to ask, because under linear probing a put into a
     * full table does not misbehave quietly, it spins looking for a free cell that does not exist. Growth
     * keeps the invalidated cells; dropping those is {@link #compact(SlotValidator)}'s job, and it needs a
     * validator this method does not have.
     */
    void put(final int ordinal, final int slabId, final int slot) {
        if ((cells + 1) * 5 > ordinals.length * 3) {
            grow();
        }
        insert(ordinal, slabId, slot);
    }

    private void insert(final int ordinal, final int slabId, final int slot) {
        final int value = (slabId << SLOT_BITS) | slot;
        final int mask = ordinals.length - 1;
        int i = hash(ordinal) & mask;
        while (true) {
            final int key = ordinals[i];
            if (key == ordinal) {
                packed[i] = value;
                return;
            }
            if (key == EMPTY) {
                ordinals[i] = ordinal;
                packed[i] = value;
                cells++;
                return;
            }
            i = (i + 1) & mask;
        }
    }

    /** Rebuilds the table at about three times its occupancy, keeping every cell including the stale ones. */
    private void grow() {
        final int[] oldOrdinals = ordinals;
        final int[] oldPacked = packed;
        final int stale = invalidated;
        allocate(capacityFor(cells));
        invalidated = stale;
        for (int i = 0; i < oldOrdinals.length; i++) {
            if (oldOrdinals[i] != EMPTY) {
                insert(oldOrdinals[i], oldPacked[i] >>> SLOT_BITS, oldPacked[i] & SLOT_MASK);
            }
        }
    }

    /** A power of two giving a load factor of about a third at {@code entries}, so growth is geometric. */
    private static int capacityFor(final int entries) {
        int capacity = MIN_CAPACITY;
        while (capacity < (entries + 1) * 3) {
            capacity <<= 1;
        }
        return capacity;
    }

    /** Told by the cache that one of this map's slots has been taken, so the cell is now dead weight. */
    void noteInvalidated() {
        invalidated++;
    }

    /**
     * Whether the invalidated cells have come to outnumber the rest, so that a {@link #compact} would get
     * the table's size back. Asked on the insert path, which is the miss path, so compaction never happens
     * under a lookup.
     */
    boolean needsCompaction() {
        return invalidated > MIN_CAPACITY && invalidated * 2 >= cells;
    }

    /** Rebuilds the table at a size set by how many cells are still live, dropping the rest. */
    void compact(final SlotValidator validator) {
        final int[] oldOrdinals = ordinals;
        final int[] oldPacked = packed;
        int live = 0;
        for (int i = 0; i < oldOrdinals.length; i++) {
            if (oldOrdinals[i] != EMPTY && validator.isLive(oldOrdinals[i], oldPacked[i] >>> SLOT_BITS, oldPacked[i] & SLOT_MASK)) {
                live++;
            } else {
                // Reuse EMPTY as "do not carry this one over", so the copy below needs no second pass.
                oldOrdinals[i] = EMPTY;
            }
        }
        allocate(capacityFor(live));
        for (int i = 0; i < oldOrdinals.length; i++) {
            if (oldOrdinals[i] != EMPTY) {
                insert(oldOrdinals[i], oldPacked[i] >>> SLOT_BITS, oldPacked[i] & SLOT_MASK);
            }
        }
    }

    /** Occupied cells, live and invalidated together. */
    int cells() {
        return cells;
    }

    /** Cells invalidated since the last rebuild. */
    int invalidatedCells() {
        return invalidated;
    }

    /** The table's capacity in cells, for the index-bytes-per-entry figure I2 reports. */
    int capacity() {
        return ordinals.length;
    }

    /** Heap held by this map's two arrays, headers included. */
    long indexBytes() {
        return 2L * (16 + 4L * ordinals.length);
    }

    /**
     * Fibonacci hashing, because ordinals arrive in runs and the low bits of a run would all probe the same
     * stretch of the table under linear probing.
     */
    private static int hash(final int ordinal) {
        final int h = ordinal * 0x9E3779B1;
        return (h ^ (h >>> 16)) & 0x7FFFFFFF;
    }
}

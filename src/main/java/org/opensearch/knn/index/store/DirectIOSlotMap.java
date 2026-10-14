/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import java.util.Arrays;

/**
 * One file's part of {@link DirectIOVectorCache}'s index: a primitive map from a vector's byte offset in that
 * file to the slot that holds it, as one {@code long} array and one {@code int} array and nothing else.
 *
 * <h2>Why not a {@code HashMap}</h2>
 * The index competes with vectors for the same byte budget, so its per-entry cost comes off the cache's
 * capacity. A {@code HashMap<Long, Integer>} entry is a 32 B {@code Node}, a 24 B boxed {@code Long}, a 16 B
 * boxed {@code Integer} and a 4 B table slot: about 76 B per cached vector. Open addressing over two
 * primitive arrays is 12 B per cell, so about 36 B per entry at this map's load factor, and a lookup
 * allocates nothing at all.
 *
 * <p>This is 16-I's {@code OrdinalSlotMap} with a wider key. 16-I keyed by Lucene vector ordinal, an
 * {@code int}; below the {@code IndexInput} there is no ordinal, only the byte offset the reader asked for,
 * which is a {@code long}. That costs 4 B per cell and 8 B per slot back pointer against 16-I's, and it buys
 * the key the input already has: nothing has to be translated, grouped or inferred (see
 * {@code DirectIOVectorCache}'s "vector granularity comes for free").
 *
 * <h2>Not thread safe, and does not need to be</h2>
 * Every mutation happens under {@link DirectIOVectorCache}'s write lock and every read under its read lock,
 * and those exclude each other, so no reader observes a half-finished insert or rebuild. Same reason the
 * cache can hold its file index in a plain {@link java.util.HashMap}.
 *
 * <h2>Entries are invalidated lazily, so there is no remove</h2>
 * When a slab is reclaimed the entries pointing into it become wrong, and this map is not told which. It does
 * not need to be: the slot records which offset of which file index owns it, so a lookup confirms that the
 * slot still points back at the key it came from and treats a mismatch as a miss. A cell whose slot has been
 * taken is harmless, merely occupied, and {@link #noteInvalidated()} counts them so that
 * {@link #needsCompaction()} can ask for a {@link #compact(SlotValidator)} once they outnumber the live ones.
 * That removes the one subtle piece this map would otherwise need, backward-shift deletion under linear
 * probing, at the cost of an equality check the read path makes anyway.
 *
 * <h2>Offsets are non-negative, so -1 is the empty cell</h2>
 * An offset is a position in a file, so it is never negative. That frees {@link #EMPTY} as the marker and
 * keeps a cell to its two numbers with no third array of flags.
 */
final class DirectIOSlotMap {

    /**
     * Confirms that a cell's slot is still the one holding that offset, so a compaction can drop the cells
     * whose slots have been taken. Implemented by the cache, which is the only thing that can see a slot.
     */
    interface SlotValidator {
        boolean isLive(long offset, int slabId, int slot);
    }

    /**
     * Told every occupied cell of a table, so that the closed-file release (O3) can free this file's slots
     * without walking every slot of every slab. Implemented by the cache, which is the only thing that can
     * see a slot; it must still check each slot's back pointer, because an occupied cell is not proof that
     * the vector is still there -- see the class javadoc.
     */
    interface CellVisitor {
        void visit(long offset, int slabId, int slot);
    }

    /** Free cell. An offset is a position in a file and so is never negative. */
    private static final long EMPTY = -1L;

    /** Smallest table. A file with one cached vector costs 192 B of index and no more. */
    private static final int MIN_CAPACITY = 16;

    /**
     * How a slot is packed into one int: {@code (slabId << SLOT_BITS) | slot}. 11 bits of slab id and 20 of
     * slot index fit in a positive int, which bounds the cache at {@link #MAX_SLABS} slabs (64 GiB at the
     * 32 MiB slab size) and a slab at {@link #MAX_SLOTS} slots. Both bounds are checked where slabs are
     * registered and opened, so nothing here defends against an overflowing shift.
     */
    private static final int SLOT_BITS = 20;
    static final int MAX_SLABS = 1 << 11;
    static final int MAX_SLOTS = 1 << SLOT_BITS;
    private static final int SLOT_MASK = MAX_SLOTS - 1;

    /**
     * The file this map indexes, {@code <path>#<length>}. Carried so that the cache can drop this map from
     * its index the moment the map's last live entry goes, which is how a closed file's entries stop costing
     * anything at all; see {@code DirectIOVectorCache}'s closed-file rule.
     */
    private final String fileId;

    /** Offsets, {@link #EMPTY} where the cell is free. Parallel to {@link #packed}. */
    private long[] offsets;

    /** {@code (slabId << SLOT_BITS) | slot} for the offset in the matching cell of {@link #offsets}. */
    private int[] packed;

    /** Occupied cells, live and invalidated together, which is what the load factor is measured against. */
    private int cells;

    /** Cells the cache has invalidated since the last rebuild. See the class javadoc. */
    private int invalidated;

    /** Slots that currently point back at this map, so the cache knows when the map is dead weight. */
    private int live;

    DirectIOSlotMap(final String fileId) {
        this.fileId = fileId;
        allocate(MIN_CAPACITY);
    }

    private void allocate(final int capacity) {
        offsets = new long[capacity];
        Arrays.fill(offsets, EMPTY);
        packed = new int[capacity];
        cells = 0;
        invalidated = 0;
    }

    /** The file this map indexes. */
    String fileId() {
        return fileId;
    }

    /**
     * The cell holding {@code offset}, or {@code -1}. A cell is not proof that the vector is cached: the
     * caller must still check the slot's back pointer. See the class javadoc.
     */
    int find(final long offset) {
        final long[] keys = offsets;
        final int mask = keys.length - 1;
        int i = hash(offset) & mask;
        while (true) {
            final long key = keys[i];
            if (key == offset) {
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
     * Points {@code offset} at {@code (slabId, slot)}, replacing any slot it pointed at before, and grows the
     * table first if this insert would take it past a 0.6 load factor.
     *
     * <p>It grows itself rather than trusting the caller to ask, because under linear probing a put into a
     * full table does not misbehave quietly, it spins looking for a free cell that does not exist. Growth
     * keeps the invalidated cells; dropping those is {@link #compact(SlotValidator)}'s job, and it needs a
     * validator this method does not have.
     */
    void put(final long offset, final int slabId, final int slot) {
        if ((cells + 1) * 5 > offsets.length * 3) {
            grow();
        }
        insert(offset, slabId, slot);
    }

    private void insert(final long offset, final int slabId, final int slot) {
        final int value = (slabId << SLOT_BITS) | slot;
        final int mask = offsets.length - 1;
        int i = hash(offset) & mask;
        while (true) {
            final long key = offsets[i];
            if (key == offset) {
                packed[i] = value;
                return;
            }
            if (key == EMPTY) {
                offsets[i] = offset;
                packed[i] = value;
                cells++;
                return;
            }
            i = (i + 1) & mask;
        }
    }

    /** Rebuilds the table at about three times its occupancy, keeping every cell including the stale ones. */
    private void grow() {
        final long[] oldOffsets = offsets;
        final int[] oldPacked = packed;
        final int stale = invalidated;
        allocate(capacityFor(cells));
        invalidated = stale;
        for (int i = 0; i < oldOffsets.length; i++) {
            if (oldOffsets[i] != EMPTY) {
                insert(oldOffsets[i], oldPacked[i] >>> SLOT_BITS, oldPacked[i] & SLOT_MASK);
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

    /** Told that a slot now points back at this map. */
    void noteAttached() {
        live++;
    }

    /**
     * Told that a slot has stopped pointing back at this map.
     *
     * @return true when this map now holds nothing at all, which is the cache's signal to drop it
     */
    boolean noteDetached() {
        live--;
        return live <= 0;
    }

    /** Slots that currently point back at this map. */
    int liveEntries() {
        return live;
    }

    /**
     * Whether the invalidated cells have come to outnumber the rest, so that a {@link #compact} would get the
     * table's size back. Asked on the insert path, which is the miss path, so compaction never happens under
     * a lookup.
     */
    boolean needsCompaction() {
        return invalidated > MIN_CAPACITY && invalidated * 2 >= cells;
    }

    /** Rebuilds the table at a size set by how many cells are still live, dropping the rest. */
    void compact(final SlotValidator validator) {
        final long[] oldOffsets = offsets;
        final int[] oldPacked = packed;
        int liveCells = 0;
        for (int i = 0; i < oldOffsets.length; i++) {
            if (oldOffsets[i] != EMPTY && validator.isLive(oldOffsets[i], oldPacked[i] >>> SLOT_BITS, oldPacked[i] & SLOT_MASK)) {
                liveCells++;
            } else {
                // Reuse EMPTY as "do not carry this one over", so the copy below needs no second pass.
                oldOffsets[i] = EMPTY;
            }
        }
        allocate(capacityFor(liveCells));
        for (int i = 0; i < oldOffsets.length; i++) {
            if (oldOffsets[i] != EMPTY) {
                insert(oldOffsets[i], oldPacked[i] >>> SLOT_BITS, oldPacked[i] & SLOT_MASK);
            }
        }
    }

    /**
     * Tells {@code visitor} every occupied cell, invalidated ones included. O3's release walks the file's own
     * table rather than every slab's back pointers, so a close costs this file's cells and not the store's
     * slots: at 128 slabs of 768-dimension vectors that is 500,000 cells against 1,396,736 slots.
     */
    void forEachCell(final CellVisitor visitor) {
        final long[] keys = offsets;
        final int[] values = packed;
        for (int i = 0; i < keys.length; i++) {
            if (keys[i] != EMPTY) {
                visitor.visit(keys[i], values[i] >>> SLOT_BITS, values[i] & SLOT_MASK);
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

    /** The table's capacity in cells, for the index-bytes-per-entry figure M2 reports. */
    int capacity() {
        return offsets.length;
    }

    /** Heap held by this map's two arrays, headers included. */
    long indexBytes() {
        return 16 + 8L * offsets.length + 16 + 4L * packed.length;
    }

    /**
     * Fibonacci hashing over the whole 64 bits, because offsets arrive in runs of one vector length and both
     * the low bits (a vector length is a multiple of four, often of 512) and the high bits (one file) carry
     * no entropy, which under linear probing would put a run through the same stretch of table.
     */
    private static int hash(final long offset) {
        final long h = offset * 0x9E3779B97F4A7C15L;
        return (int) ((h ^ (h >>> 32)) & 0x7FFFFFFF);
    }
}

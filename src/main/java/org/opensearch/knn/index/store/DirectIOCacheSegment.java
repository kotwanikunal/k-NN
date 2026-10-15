/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import java.nio.ByteBuffer;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * One segment of {@link DirectIOVectorCache}: its own lock, its own file index, its own slabs, its own FIFO
 * queue with its own second chance, its own open slab per vector length and its own recycled queue (Q, C1a).
 *
 * <h2>Why the store is segmented, in one measurement</h2>
 * Before Q the whole store was one {@link ReentrantReadWriteLock}. MEASURED on the dev desktop
 * ({@code output/bench/q-baseline}), driven in the sequence {@link DirectIOVectorIndexInput} actually makes,
 * that store <b>peaked at one thread</b>, 1,529,026 serving lookups/s, and settled at 320 to 375K from four
 * threads up. From four threads up <b>every</b> read acquisition found a writer already queued, and at eight
 * threads and above more than 90% of all thread time was lock wait. The mechanism is named: the lock is
 * nonfair, and {@code NonfairSync#readerShouldBlock} makes a reader block when a <em>waiting</em> writer is
 * at the head of the queue, which at 250K fills/s is nearly always. INFERRED against
 * {@code ref/trillion-readiness.md} section 3.1, that capped a hot node at about 12,800 QPS against a 16,000
 * target.
 *
 * <h2>What a segment owns, and what it therefore never waits for</h2>
 * Everything a fill does under this lock touches only this object. A slab holds the vectors of exactly one
 * segment, so the FIFO queue, the second chance, the half-slab re-admission cap, the different-length drop
 * and the recycled queue are <b>unchanged in logic</b> and simply became per segment. The one thing that
 * cannot be per segment is the node's total slab target, and that lives in {@link DirectIOSlabBroker}.
 *
 * <p>A key's segment is a pure function of {@code (fileId, offset)}
 * ({@link DirectIOVectorCache#segmentFor}), so nothing is stored about it and a vector can only ever be
 * admitted into the segment a later lookup of that offset will consult. Slot ownership is unchanged: a slot's
 * back pointer is its {@link DirectIOSlotMap}, and with one map per (segment, file) that reference already
 * identifies both.
 *
 * <h2>The lock order</h2>
 * The broker's lock, then <b>at most one</b> segment's lock. No path holds two segment locks, and no path
 * takes the broker's lock while holding a segment's. See {@link DirectIOSlabBroker} for the proof and for why
 * the slab reservation is lock free rather than taken under the broker's lock.
 */
final class DirectIOCacheSegment {

    /** How {@link #locate} packs its answer, which is the same shape {@link DirectIOSlotMap} stores. */
    private static final int SLOT_SHIFT = 20;
    private static final int SLOT_LOW_MASK = (1 << SLOT_SHIFT) - 1;

    private final DirectIOSlabBroker broker;

    /**
     * A reader must not be shown a slot while another thread is overwriting it, and the index must not be
     * read half-rebuilt. One lock per segment solves both: a lookup takes the read lock, and insert,
     * eviction, resize and release take the write lock.
     */
    private final ReentrantReadWriteLock lock = new ReentrantReadWriteLock();

    /** File index. Mutated under the write lock, read under the read lock, so a plain map is enough. */
    private final Map<String, DirectIOSlotMap> index = new HashMap<>();

    /** This segment's slabs in service, oldest first. The head is the eviction unit. */
    private final ArrayDeque<Slab> fifo = new ArrayDeque<>();

    /** The one slab currently being appended to, per vector byte length. */
    private final Map<Integer, Slab> openSlabs = new HashMap<>();

    /**
     * Slabs that hold free slots a closed file gave back, per vector byte length, waiting to be reopened for
     * appending. O3.
     *
     * <p>Without this the slots {@link #releaseFile} frees would be unreachable until their slab came to the
     * head of {@link #fifo}, because {@link #place} only appends to {@link #openSlabs}. A fill arriving
     * straight after a close would then grow or evict while the store had room it had just been given, which
     * is the opposite of what O3 is for.
     *
     * <p>Entries are validated on the way out rather than removed on every event that could stale one (a
     * reclaim reopening the slab for another length, a {@code configure} releasing it): see
     * {@link #recycledSlab}.
     */
    private final Map<Integer, ArrayDeque<Slab>> recycled = new HashMap<>();

    /**
     * How many slabs sit in {@link #recycled}. Written under the write lock and read without it by
     * {@link DirectIOVectorCache#mayAdmitSpanFillApproximately}, which is the input's pre-check. O3.
     */
    private volatile int recycledSlabs;

    /** This segment's slabs by id, so an index entry can be one int. Null where an id is not in use. */
    private Slab[] slabsById = new Slab[0];

    /**
     * Live entries, their slot bytes, and the slabs holding them. Written under the write lock and read
     * without it, because the stats endpoint sums them across segments and has no business competing with
     * the query path for a lock (Q, section 2.6).
     */
    private volatile long liveEntries;
    private volatile long liveEntryBytes;
    private volatile int liveSlabs;

    /**
     * Reservations this segment has in flight. <b>One at a time</b>, which is N3's rule made per segment:
     * one slab holds thousands of vectors and the fills waiting for it all fit in it, so that is a
     * per-segment reason. MEASURED on the re-proof's arm E without the rule: the first burst's threads each
     * reserved a slab, a 64mb store reached its 2 slab target before a single vector had been cached, span
     * fill stopped because it only runs below the target, and the arm's hit count fell from 2,921 to 832.
     *
     * <p>Only read and written under the write lock.
     */
    private int pendingSlabs;

    /**
     * Set when a segment-count change has taken this segment out of the array, so a fill that read the old
     * array before the swap cannot register a slab into a segment no lookup will ever consult again. Without
     * it such a fill would leave a slab the broker counts and no segment holds, which is both a leak and a
     * broken invariant.
     *
     * <p>Only read and written under the write lock, which is why it is a plain field: a fill that could
     * register a slab holds that lock, so it either sees this set or was already inside when it was set and
     * its slab is released by the drain that follows.
     */
    private boolean retired;

    DirectIOCacheSegment(final DirectIOSlabBroker broker) {
        this.broker = broker;
    }

    /**
     * One slab in service: its memory, the slot layout it was opened for, and one back pointer per slot so
     * that eviction can find the index entry that owns it and a lookup can confirm it has not been taken.
     */
    private static final class Slab {
        private final int id;
        private final DirectIOSlab memory;
        private int slotBytes;
        private int slotCount;
        /** High-water mark: slots {@code [0, used)} have been written since the slab was opened. */
        private int used;
        private DirectIOSlotMap[] owner;
        private long[] offset;
        /** 1 once the slot has been read, which is what buys it a second chance. */
        private byte[] hit;
        /**
         * Slots below {@link #used} that a reclaim freed, so that a survivor can stay where it is instead of
         * being copied to the front. {@code [0, freeCount)} is the live part of the stack.
         */
        private int[] free;
        private int freeCount;
        /**
         * Whether this slab is sitting in {@link #recycled} waiting to be reopened for appending. A flag
         * rather than a containment test, so a closed file that freed slots in the same slab twice cannot
         * queue it twice. O3.
         */
        private boolean recycled;

        private Slab(final int id, final DirectIOSlab memory) {
            this.id = id;
            this.memory = memory;
        }

        /** Re-opens the slab for {@code bytes}-long vectors, discarding the previous slot layout. */
        private void openFor(final int bytes) {
            final int count = Math.min(memory.capacity() / bytes, DirectIOSlotMap.MAX_SLOTS);
            if (owner == null || count > owner.length) {
                owner = new DirectIOSlotMap[count];
                offset = new long[count];
                hit = new byte[count];
                free = new int[count];
            } else {
                Arrays.fill(owner, null);
            }
            slotBytes = bytes;
            slotCount = count;
            used = 0;
            freeCount = 0;
        }

        /** A slot to write a vector into, or {@code -1} when the slab is full. */
        private int allocSlot() {
            if (freeCount > 0) {
                return free[--freeCount];
            }
            return used < slotCount ? used++ : -1;
        }

        private int freeSlots() {
            return freeCount + (slotCount - used);
        }
    }

    /**
     * One fill's slab request, carried across a release of a segment's write lock.
     *
     * <p><b>Why the lock has to be released at all.</b> A 32 MiB slab takes about 7 to 10 ms to allocate
     * (MEASURED, M1 and output/bench/n2: FFM zero-fills it; 16-I's JNI took 7.6 us), and before N3
     * {@code growSlabs} did that with the write lock held, so every lookup on the node waited behind it.
     * A fill that needs a slab therefore reserves its headroom ({@link DirectIOSlabBroker#reserve()}), leaves
     * the lock, allocates, and comes back with the slab in hand.
     *
     * <p><b>Why the reservation.</b> Without it two fills leaving the lock at the same time would both see
     * room for one slab and both allocate one, so a 2 slab target could end up holding 3. The reservation is
     * counted against the target by every other fill, in the same word as the live count.
     *
     * <p>An allocation nobody wanted any more, because the target shrank while this fill was outside the lock,
     * is released by {@link #finish} and also outside every lock.
     */
    static final class Request {
        private final DirectIOSlabBroker broker;
        /** Allocated or borrowed outside the lock and not yet registered. */
        DirectIOSlab spare;
        /** Set by {@link DirectIOCacheSegment#growSlabs} to mean "bring me a slab and ask again". */
        boolean wanted;
        /**
         * Whether the slab is to be <b>borrowed</b> from another segment rather than allocated, which is what
         * a segment holding no slab at all asks for when the store is already at its target (Q, section 2.2
         * step 3). Set by {@link DirectIOCacheSegment#obtainSlab}.
         */
        boolean wantsTransfer;
        /** The segment whose one pending slot this request holds, or null when it holds none. */
        private DirectIOCacheSegment reservedIn;
        /** Whether this request holds one of the broker's reservations. */
        private boolean brokerPending;

        Request(final DirectIOSlabBroker broker) {
            this.broker = broker;
        }

        /** Gives back the headroom and any slab that went unused. Called with no lock held. */
        void finish() {
            if (reservedIn == null && brokerPending == false && spare == null) {
                return;
            }
            final DirectIOSlab unused = spare;
            spare = null;
            if (reservedIn != null) {
                reservedIn.cancelPending();
                reservedIn = null;
            }
            if (brokerPending) {
                brokerPending = false;
                broker.cancelReservation();
            }
            if (unused != null) {
                // A borrowed slab nobody could use goes back to the backend rather than to its donor: the
                // store is then one slab below its target and the next fill that wants one allocates it,
                // which is simpler than a hand-back path and costs one allocation on a transient.
                broker.release(unused);
            }
        }

        /** Called by the broker when a transfer has handed this request a slab and its reservation. */
        void tookTransfer(final DirectIOSlab memory) {
            spare = memory;
            brokerPending = true;
            wantsTransfer = false;
        }
    }

    // ---------------------------------------------------------------------------------------------------
    // The lock, and what it costs (C2; published by KNNStats, see DirectIOVectorCache)
    // ---------------------------------------------------------------------------------------------------

    /**
     * Takes the read lock, counting the acquisition if it had to queue behind another thread and timing it
     * if it did.
     *
     * <h4>Why the wait is timed on a contended acquisition and not sampled over all of them</h4>
     * The design run proposed sampling one acquisition in 1,024, because the instrumented baseline build
     * timed <b>every</b> acquisition with two {@code System.nanoTime} calls (31 ns each on this host) and
     * cost 26% of the plain build's one-thread throughput. Timing only the contended ones is both exact and
     * cheaper than sampling, and the measurement is why: a contended acquisition waits 20.3 us at the core
     * count and 168.3 us at 128 threads on the unsegmented store (MEASURED, output/bench/q-c1b), against
     * <b>0.045 us</b> uncontended. All of the wait is in the contended acquisitions, so an uncontended one
     * has nothing worth timing, and 62 ns against a 0.7 us wait is 9% of a figure that is 1.9% of
     * acquisitions at the core count. The uncontended path therefore costs one
     * {@code hasQueuedThreads()} volatile read and a branch, which is what the design wanted the counts to
     * cost and now also what the timing costs.
     */
    private void lockRead() {
        if (lock.hasQueuedThreads()) {
            DirectIOVectorCache.contendedReadAcquisitions.increment();
            if (DirectIOVectorCache.lockTimingEnabled) {
                final long started = System.nanoTime();
                lock.readLock().lock();
                DirectIOVectorCache.readLockWaitNanos.add(System.nanoTime() - started);
            } else {
                lock.readLock().lock();
            }
        } else {
            lock.readLock().lock();
        }
        if (DirectIOVectorCache.acquisitionCounting) {
            DirectIOVectorCache.readAcquisitions.increment();
        }
    }

    private void unlockRead() {
        lock.readLock().unlock();
    }

    /** {@link #lockRead}'s counterpart for the write lock. */
    private void lockWrite() {
        if (lock.hasQueuedThreads()) {
            DirectIOVectorCache.contendedWriteAcquisitions.increment();
            if (DirectIOVectorCache.lockTimingEnabled) {
                final long started = System.nanoTime();
                lock.writeLock().lock();
                DirectIOVectorCache.writeLockWaitNanos.add(System.nanoTime() - started);
            } else {
                lock.writeLock().lock();
            }
        } else {
            lock.writeLock().lock();
        }
        if (DirectIOVectorCache.acquisitionCounting) {
            DirectIOVectorCache.writeAcquisitions.increment();
        }
    }

    private void unlockWrite() {
        lock.writeLock().unlock();
    }

    // ---------------------------------------------------------------------------------------------------
    // The read path
    // ---------------------------------------------------------------------------------------------------

    /**
     * Walks the run of cached vectors from {@code at} towards {@code end} for as long as the offsets belong
     * to this segment, and returns the offset it reached, or {@code -1} when a vector in the run is not
     * cached. One read-lock acquisition, which is what keeps {@link DirectIOVectorCache#contains} at the
     * single acquisition it had before the store was segmented for all but a run that crosses a stripe.
     *
     * @param fileId the file's identity
     * @param at     where to start, which must belong to this segment
     * @param end    one past the last byte of the run
     * @return the offset reached, or {@code -1}
     */
    long containsUpTo(final String fileId, final long at, final long end) {
        lockRead();
        try {
            final DirectIOSlotMap map = index.get(fileId);
            if (map == null) {
                return -1;
            }
            long next = at;
            while (next < end) {
                final int cell = map.find(next);
                if (cell < 0) {
                    return -1;
                }
                final Slab slab = slabAt(map.slabIdAt(cell));
                final int slot = map.slotAt(cell);
                if (slab == null || slot >= slab.slotCount || slab.owner[slot] != map || slab.offset[slot] != next) {
                    return -1;
                }
                // The slot's own length is the stride: the caller does not know the field's vector length at
                // this point and must not have to guess it.
                next += slab.slotBytes;
                if (next < end && DirectIOVectorCache.segmentFor(fileId, next) != this) {
                    // The next vector is another segment's. Hand the offset back so the caller can take that
                    // segment's lock rather than hold two at once.
                    break;
                }
            }
            return next;
        } finally {
            unlockRead();
        }
    }

    /** {@link DirectIOVectorCache#getFloats}'s body, for the one segment the offset belongs to. */
    boolean getFloats(final String fileId, final long offset, final int length, final float[] destination, final int destinationOffset) {
        lockRead();
        try {
            final DirectIOSlotMap map = index.get(fileId);
            final int packedSlot = locate(map, offset, length);
            if (packedSlot >= 0) {
                final Slab slab = slabsById[packedSlot >>> SLOT_SHIFT];
                final int slot = packedSlot & SLOT_LOW_MASK;
                slab.memory.floats().get((slot * slab.slotBytes) >> 2, destination, destinationOffset, length >> 2);
                // A plain write, and a lost one only costs a second chance. Readers run concurrently with
                // each other but never with the reclaim that reads this, which holds the write lock.
                slab.hit[slot] = 1;
                return true;
            }
            return false;
        } finally {
            unlockRead();
        }
    }

    /** {@link DirectIOVectorCache#getBytes}'s body, for the one segment the offset belongs to. */
    boolean getBytes(final String fileId, final long offset, final int length, final byte[] destination, final int destinationOffset) {
        lockRead();
        try {
            final DirectIOSlotMap map = index.get(fileId);
            final int packedSlot = locate(map, offset, length);
            if (packedSlot >= 0) {
                final Slab slab = slabsById[packedSlot >>> SLOT_SHIFT];
                final int slot = packedSlot & SLOT_LOW_MASK;
                // An absolute get out of the shared byte view, so no position is touched and no view of slab
                // memory leaves the store -- the same discipline getFloats keeps.
                slab.memory.bytes().get(slot * slab.slotBytes, destination, destinationOffset, length);
                slab.hit[slot] = 1;
                return true;
            }
            return false;
        } finally {
            unlockRead();
        }
    }

    /**
     * Writes the members of {@code ords[from..to)} this segment does not hold whole into
     * {@code missing[found..)} and returns the new {@code found}, or {@code -1} when it holds nothing of
     * {@code fileId} at all and the caller's whole batch is therefore missing from it.
     *
     * <p>One read-lock acquisition for the range, which is the batching {@link DirectIOVectorCache} exists to
     * preserve: MEASURED, a contended read acquisition on the unsegmented lock waited 8.9 us at 8 threads and
     * 168.8 us at 128, against 0.045 us uncontended.
     */
    int selectMissing(
        final String fileId,
        final long baseOffset,
        final int vectorBytes,
        final int[] ords,
        final int from,
        final int to,
        final int[] missing,
        final int found
    ) {
        int writing = found;
        lockRead();
        try {
            final DirectIOSlotMap map = index.get(fileId);
            if (map == null) {
                return -1;
            }
            for (int i = from; i < to; i++) {
                if (locate(map, baseOffset + (long) ords[i] * vectorBytes, vectorBytes) < 0) {
                    missing[writing++] = ords[i];
                }
            }
        } finally {
            unlockRead();
        }
        return writing;
    }

    /**
     * Records in {@code miss} whether this segment holds each of the ordinals {@code ords[order[k]]} for
     * {@code k} in {@code [from, to)}, under <b>one</b> read-lock acquisition, and says whether it holds an
     * index for the file at all. C2's batched probe: {@code order} is the batch grouped by segment, so this
     * segment's members of the batch are a contiguous run of it and the caller's ordinal order is recovered
     * from {@code miss} afterwards.
     *
     * @param miss written at index {@code order[k]}, never read
     * @return whether this segment has an index table for {@code fileId}
     */
    boolean probeMissing(
        final String fileId,
        final long baseOffset,
        final int vectorBytes,
        final int[] ords,
        final int[] order,
        final int from,
        final int to,
        final boolean[] miss
    ) {
        lockRead();
        try {
            final DirectIOSlotMap map = index.get(fileId);
            if (map == null) {
                for (int k = from; k < to; k++) {
                    miss[order[k]] = true;
                }
                return false;
            }
            for (int k = from; k < to; k++) {
                final int i = order[k];
                miss[i] = locate(map, baseOffset + (long) ords[i] * vectorBytes, vectorBytes) < 0;
            }
            return true;
        } finally {
            unlockRead();
        }
    }

    /**
     * The {@code (slabId << 20) | slot} holding {@code offset} at {@code length} bytes, or {@code -1}.
     * Called under either lock.
     *
     * <p>Two checks past the index cell, and both are load bearing. The slot's back pointer is the proof that
     * the cell is not stale: a cell can outlive the slot it names, because eviction does not walk the index
     * (see {@link DirectIOSlotMap}). The slot length check is what keeps a caller asking for a different
     * length than was cached from being handed a neighbour's bytes.
     */
    private int locate(final DirectIOSlotMap map, final long offset, final int length) {
        if (map == null) {
            return -1;
        }
        final int cell = map.find(offset);
        if (cell < 0) {
            return -1;
        }
        final int slabId = map.slabIdAt(cell);
        final int slot = map.slotAt(cell);
        final Slab slab = slabAt(slabId);
        if (slab == null || slot >= slab.slotCount || slab.owner[slot] != map || slab.offset[slot] != offset) {
            return -1;
        }
        if (slab.slotBytes != length) {
            return -1;
        }
        return (slabId << SLOT_SHIFT) | slot;
    }

    // ---------------------------------------------------------------------------------------------------
    // The fill path
    // ---------------------------------------------------------------------------------------------------

    /** {@link DirectIOVectorCache#put}'s body, for the one segment the offset belongs to. */
    void put(final String fileId, final long offset, final ByteBuffer source, final int position, final int length) {
        final Request request = new Request(broker);
        try {
            // At most twice round: once with nothing in hand, and once with the slab that round asked for.
            // See Request for why the allocation cannot happen inside the lock.
            for (int round = 0; round < 2; round++) {
                lockWrite();
                try {
                    request.wanted = false;
                    final Placement placement = place(fileId, length, true, request);
                    if (placement != null) {
                        placement.slab.memory.bytes().put(placement.slot * length, source, position, length);
                        attach(placement.slab, placement.slot, placement.map, offset);
                        DirectIOVectorCache.insertions.increment();
                        return;
                    }
                } finally {
                    unlockWrite();
                }
                if (request.wanted == false || broker.allocateOutsideLock(request) == false) {
                    return;
                }
            }
        } finally {
            request.finish();
        }
    }

    /** {@link DirectIOVectorCache#readmitAfterRace}'s body, for the one segment the offset belongs to. */
    boolean readmitAfterRace(final String fileId, final long offset, final ByteBuffer source, final int position, final int length) {
        final Request request = new Request(broker);
        try {
            for (int round = 0; round < 2; round++) {
                lockWrite();
                try {
                    if (locate(index.get(fileId), offset, length) >= 0) {
                        // Another thread put it back while this read was in flight, which is the same race
                        // one turn later and needs nothing from this one.
                        return false;
                    }
                    request.wanted = false;
                    final Placement placement = place(fileId, length, false, request);
                    if (placement != null) {
                        placement.slab.memory.bytes().put(placement.slot * length, source, position, length);
                        attach(placement.slab, placement.slot, placement.map, offset);
                        DirectIOVectorCache.insertions.increment();
                        // Both counters: BACKLOG asks for this under the existing readmissions key, and
                        // race_readmissions keeps the two sources separable, because readmissions otherwise
                        // means second chances alone and the re-proof reads it arm against arm.
                        DirectIOVectorCache.readmissions.increment();
                        DirectIOVectorCache.raceReadmissions.increment();
                        return true;
                    }
                } finally {
                    unlockWrite();
                }
                if (request.wanted == false || broker.allocateOutsideLock(request) == false) {
                    return false;
                }
            }
            return false;
        } finally {
            request.finish();
        }
    }

    /**
     * {@link DirectIOVectorCache#putRun}'s body for the sub-run {@code [firstIndex, lastIndex)} of a staged
     * span, every vector of which belongs to this segment.
     *
     * <p>Returns the index it reached rather than what it admitted, because the caller has to tell "the
     * sub-run is done" from "a refusal stopped it", and the two are the same count when the run held vectors
     * that were already cached. What it admitted goes into {@code admitted}.
     *
     * @param admitted {@code {count}}, added to
     * @return one past the last index this call dealt with
     */
    int putRun(
        final String fileId,
        final long firstOffset,
        final int firstIndex,
        final int lastIndex,
        final ByteBuffer source,
        final int position,
        final int length,
        final long[] admitted
    ) {
        int next = firstIndex;
        final Request request = new Request(broker);
        try {
            while (next < lastIndex) {
                lockWrite();
                try {
                    if (mayAdmitSpanFill(length) == false) {
                        break;
                    }
                    for (; next < lastIndex; next++) {
                        final long offset = firstOffset + (long) next * length;
                        if (locate(index.get(fileId), offset, length) >= 0) {
                            continue;
                        }
                        request.wanted = false;
                        final Placement placement = place(fileId, length, false, request);
                        if (placement == null) {
                            // Either there is nowhere at all, or the store would take a slab and this thread
                            // has to leave the lock to get one. Both stop the run here; the second resumes it
                            // at the same vector on the next round.
                            break;
                        }
                        placement.slab.memory.bytes().put(placement.slot * length, source, position + next * length, length);
                        attach(placement.slab, placement.slot, placement.map, offset);
                        DirectIOVectorCache.insertions.increment();
                        admitted[0]++;
                    }
                } finally {
                    unlockWrite();
                }
                // One acquisition per span is the rule, and this is the one thing that can need a second:
                // a span that fills the open slab. The staged bytes are still the caller's and still valid
                // across the gap, and the alternative is an 8 ms allocation inside the hold (N3).
                if (next >= lastIndex || request.wanted == false || broker.allocateOutsideLock(request) == false) {
                    break;
                }
            }
        } finally {
            request.finish();
        }
        return next;
    }

    /**
     * Whether span fill may admit a vector of {@code length} bytes: the store is below its slab target, or a
     * closed file has given back slots of that length in this segment (O3). Called under the write lock.
     *
     * <h4>Why the second clause is not simply "the open slab has a free slot"</h4>
     * At the target the open slab's free slots are what a reclaim has just produced, and the requested fills
     * are entitled to them: letting span fill take them would be span fill running at the target, which is
     * the scan-resistance question (agenda item 4) this phase is not deciding. Slots a closed file vacated
     * are a different thing. Nothing is pushed out to use them, because what held them is gone, so admitting
     * into them cannot displace a working set. That is the whole of O3: after a merge the store would
     * otherwise sit at its target holding dead vectors, span fill would not re-arm, and the merged segment
     * would refill at one vector per miss.
     *
     * <p>The first clause is the <b>global</b> target, as it was before the store was segmented, so span fill
     * still turns itself off exactly when the store as a whole reaches its target, which is the policy every
     * measurement was taken under.
     *
     * <h4>What this does widen, MEASURED in {@code DirectIOClosedFileReleaseTests}</h4>
     * While a close's room is still there this says yes, and {@link #place} then uses whatever free slot it
     * finds first -- which may be a leftover in the open slab rather than one of the freed ones. In the test
     * fixture a close freeing 66 slots let span fill admit 190 vectors. That is wider than "the slots the
     * close freed" and it is deliberate: {@link #place} is called with eviction refused either way, so
     * nothing can be displaced, and refusing a free slot because of which event produced it would leave the
     * store holding room it could have warmed with. Once the free slots are gone this says no again, which
     * is what the test pins.
     */
    private boolean mayAdmitSpanFill(final int length) {
        return broker.belowTarget() || hasRecycledRoom(length);
    }

    /**
     * Whether a queued recycled slab still has a free slot of {@code length} bytes. Called under the write
     * lock. The queue holds at most one entry per slab, so this is bounded by the slab count, and it is
     * walked once per span-fill run rather than once per vector.
     *
     * <p>Validating rather than counting, because a queued slab can be reopened for another vector length by
     * a reclaim or released by a {@code configure} with no hook here, and a counter that drifted would make
     * span fill run at the target.
     */
    private boolean hasRecycledRoom(final int length) {
        final ArrayDeque<Slab> queue = recycled.get(length);
        if (queue == null) {
            return false;
        }
        for (final Slab slab : queue) {
            if (slabAt(slab.id) == slab && slab.slotBytes == length && slab.freeSlots() > 0) {
                return true;
            }
        }
        return false;
    }

    /** How many slabs sit in this segment's recycled queue. A volatile read and no lock. O3. */
    int recycledSlabs() {
        return recycledSlabs;
    }

    /** Where a fill is about to write: the file's index, a slab and a slot in it. */
    private record Placement(DirectIOSlotMap map, Slab slab, int slot) {
    }

    /**
     * Finds room for one vector of {@code length} bytes, evicting a slab if {@code allowEviction} and that is
     * what it takes, and returns where to write it or {@code null} when there is nowhere. Called under the
     * write lock; the caller writes the bytes and then {@link #attach}es the slot.
     *
     * <p>With {@code allowEviction} false the fill must displace nothing: it may use a free slot of the slab
     * already open for this length, or a slab the target still allows, and nothing else.
     *
     * <p>{@code request} is how a fill that needs a new slab is told to go and get one: this returns null with
     * {@link Request#wanted} set, and the caller allocates outside the lock and asks again.
     */
    private Placement place(final String fileId, final int length, final boolean allowEviction, final Request request) {
        if (broker.maxSlabs() == 0 || retired) {
            return null;
        }
        DirectIOSlotMap map = index.get(fileId);
        if (map == null) {
            map = new DirectIOSlotMap(fileId);
            index.put(fileId, map);
        }
        if (map.needsCompaction()) {
            final DirectIOSlotMap compacting = map;
            map.compact((offset, slabId, slot) -> {
                final Slab slab = slabAt(slabId);
                return slab != null && slot < slab.slotCount && slab.owner[slot] == compacting && slab.offset[slot] == offset;
            });
        }
        Slab slab = openSlabs.get(length);
        if (slab == null || slab.freeSlots() == 0) {
            // O3: slots a closed file gave back come first, before any growth and before any eviction. The
            // store has room it has already been given, and using it is what lets span fill re-arm for a
            // merged segment without displacing a live vector.
            slab = recycledSlab(length);
        }
        if (slab == null) {
            slab = allowEviction ? obtainSlab(length, request) : growSlabs(length, request);
        }
        if (slab == null) {
            // A file that got an index entry and no slot is left in the index with nothing in it, which costs
            // one empty table until the next fill for that file uses it. Dropping it here would need the
            // live count to distinguish "never had an entry" from "lost its last one".
            return null;
        }
        final int slot = slab.allocSlot();
        return slot < 0 ? null : new Placement(map, slab, slot);
    }

    /**
     * A slab with free slots of {@code length} bytes that a closed file gave back, reopened for appending, or
     * {@code null} when there is none. Called under the write lock. O3.
     *
     * <p>Queue entries are validated here rather than removed wherever a slab's state could change: a slab
     * that has since been reclaimed for another vector length, released by a {@code configure}, or refilled
     * is simply dropped on the way out. That keeps the release path to the slots it frees and leaves no
     * bookkeeping for {@link #obtainSlab} or {@code configure} to remember.
     */
    private Slab recycledSlab(final int length) {
        final ArrayDeque<Slab> queue = recycled.get(length);
        if (queue == null) {
            return null;
        }
        while (true) {
            // Peek and leave it queued: a slab stays in the queue until its freed slots are gone, which is
            // what keeps mayAdmitSpanFill true for as long as there is genuinely room a close gave back.
            final Slab candidate = queue.peek();
            if (candidate == null) {
                recycled.remove(length, queue);
                return null;
            }
            if (slabAt(candidate.id) == candidate && candidate.slotBytes == length && candidate.freeSlots() > 0) {
                // Appending resumes here, so the next fills for this length need no second trip through the
                // queue. The slab it displaces is full by construction: place() only asks when it is.
                openSlabs.put(length, candidate);
                return candidate;
            }
            queue.poll();
            candidate.recycled = false;
            recycledSlabs--;
        }
    }

    /** Queues {@code slab} as holding free slots, once. Called under the write lock. O3. */
    private void offerRecycled(final Slab slab) {
        if (slab.recycled || slab.freeSlots() <= 0 || openSlabs.get(slab.slotBytes) == slab) {
            return;
        }
        slab.recycled = true;
        recycled.computeIfAbsent(slab.slotBytes, length -> new ArrayDeque<>()).add(slab);
        recycledSlabs++;
    }

    /**
     * Frees this segment's entries of {@code fileId} and drops its index table, accumulating the entries and
     * the bytes into {@code freed}. One write-lock acquisition, this segment's cells only, which is what
     * bounds the hold to one {@code N}th of the unsegmented one (Q, section 2.4).
     *
     * @param fileId the file's identity
     * @param freed  {@code {entries, bytes}}, added to
     */
    void releaseFile(final String fileId, final long[] freed) {
        lockWrite();
        try {
            final DirectIOSlotMap map = index.remove(fileId);
            if (map == null) {
                return;
            }
            final long[] here = new long[2];
            final List<Slab> touched = new ArrayList<>();
            map.forEachCell((offset, slabId, slot) -> {
                final Slab slab = slabAt(slabId);
                if (slab == null || slot >= slab.slotCount) {
                    return;
                }
                if (slab.owner[slot] != map || slab.offset[slot] != offset) {
                    // An invalidated cell: the slot has been taken by another file, or by this file at
                    // another offset. Either way it is not ours to free, and nulling it would hand out a
                    // slot somebody is using.
                    return;
                }
                slab.owner[slot] = null;
                if (slab.freeCount < slab.free.length) {
                    slab.free[slab.freeCount++] = slot;
                }
                if (slab.recycled == false) {
                    touched.add(slab);
                }
                here[0]++;
                here[1] += slab.slotBytes;
            });
            liveEntries -= here[0];
            liveEntryBytes -= here[1];
            for (final Slab slab : touched) {
                offerRecycled(slab);
            }
            freed[0] += here[0];
            freed[1] += here[1];
        } finally {
            unlockWrite();
        }
    }

    // ---------------------------------------------------------------------------------------------------
    // Slab supply and eviction
    // ---------------------------------------------------------------------------------------------------

    /**
     * A slab with room for a {@code length}-byte vector, by allocating one while the store is under its
     * target and otherwise by reclaiming this segment's oldest. Returns {@code null} only when the allocation
     * failed and there was nothing to reclaim. Called under the write lock.
     */
    private Slab obtainSlab(final int length, final Request request) {
        final Slab grown = growSlabs(length, request);
        if (grown != null) {
            return grown;
        }
        if (request != null && request.wanted) {
            // The target allows another slab and this fill has to leave the lock to get one. Reclaiming
            // instead would throw away a slab's worth of vectors the target says the store may keep.
            return null;
        }
        if (pendingSlabs > 0 && broker.belowTarget()) {
            // Another fill in this segment is already fetching a slab the target allows, so this fill waits
            // for it by simply not being cached. Reclaiming now would evict vectors the store is about to
            // have room for, and a fill that does not happen costs a future hit and nothing else.
            return null;
        }
        final Slab oldest = fifo.poll();
        if (oldest == null) {
            // This segment holds no slab at all and the store is at its target, so there is nothing here to
            // reclaim and growing is refused. Ask the broker to lend one from the segment holding the most
            // (Q, section 2.2 step 3). INFERRED: with 128 segments and 1,120 slabs every segment reaches
            // about eight during the ramp, so this is a cold-start transient and not a steady-state path.
            if (request != null && pendingSlabs == 0 && retired == false && broker.maxSlabs() > 0) {
                pendingSlabs++;
                request.reservedIn = this;
                request.wantsTransfer = true;
                request.wanted = true;
            }
            return null;
        }
        DirectIOVectorCache.reclaimedSlabs.increment();
        final long started = System.nanoTime();
        if (oldest.slotBytes == length) {
            // The common case, and the only one in which a second chance can do anything: the reclaimed slab
            // is reopened for the vectors it already holds, so the ones that were read stay in their slots
            // and the rest become the free slots this call came for.
            reclaimInPlace(oldest);
            openSlabs.put(length, oldest);
        } else {
            // Reopened for a different vector length, so the slot layout changes and a survivor has nowhere
            // to stay. See the class javadoc for why they are dropped rather than copied.
            dropAll(oldest);
            openSlabs.remove(oldest.slotBytes, oldest);
            open(oldest, length);
        }
        fifo.add(oldest);
        DirectIOVectorCache.reclaimNanos.add(System.nanoTime() - started);
        return oldest;
    }

    /**
     * Registers the slab {@code request} is carrying and opens it for {@code length}-byte vectors, or returns
     * {@code null}: at the target, with growth latched, or -- the interesting case -- with
     * {@link Request#wanted} set to tell the caller to allocate one outside the lock and ask again. The
     * half of {@link #obtainSlab} that displaces nothing, which is the only half span fill is allowed. Called
     * under the write lock.
     *
     * <p>Nothing here allocates: that is N3. The target check is the broker's, and it counts the slabs other
     * fills are allocating right now as well as the live ones, so concurrent growth cannot overshoot.
     *
     * <p>One allocation attempt per latch, not one per fill: see {@link DirectIOSlabBroker} for what a failed
     * attempt costs the caller. The store then stays the size it is, keeps serving the slabs it holds, and
     * the cost is hit rate -- which is what a cache that cannot grow should cost.
     */
    private Slab growSlabs(final int length, final Request request) {
        if (request == null || retired) {
            return null;
        }
        if (request.spare == null) {
            if (pendingSlabs > 0) {
                // One pending allocation at a time per segment. See pendingSlabs for the measured reason.
                return null;
            }
            if (broker.growthLatched() || broker.reserve() == false) {
                // The latch is checked here and not on the path below, because what it says is "an
                // allocation failed" (N2) and a fill arriving with a slab in hand has had one succeed. A
                // borrowed slab is likewise unaffected: it needs no allocation at all.
                return null;
            }
            pendingSlabs++;
            request.reservedIn = this;
            request.brokerPending = true;
            request.wanted = true;
            return null;
        }
        if (request.reservedIn != this || request.brokerPending == false || broker.belowTarget() == false) {
            // The target shrank while this fill was outside the lock, so the slab it is carrying is one the
            // store may not have. Request#finish gives it back, outside every lock.
            return null;
        }
        final Slab fresh = register(request.spare);
        request.spare = null;
        request.reservedIn = null;
        request.brokerPending = false;
        pendingSlabs--;
        open(fresh, length);
        return fresh;
    }

    /** Gives one pending reservation back. Called by {@link Request#finish} with no lock held. */
    private void cancelPending() {
        lockWrite();
        try {
            pendingSlabs--;
        } finally {
            unlockWrite();
        }
    }

    /**
     * Gives {@code slab}'s second chance: the slots read since they were written keep both their vector and
     * their slot, up to half the slab, and every other slot joins the free list. No byte moves and no index
     * entry changes.
     */
    private void reclaimInPlace(final Slab slab) {
        // Half the slots, rounded down, so that a slab with a single slot keeps nothing and the caller is
        // always handed room for the vector it came to write.
        final int keep = slab.slotCount / 2;
        int kept = 0;
        int dropped = 0;
        final DirectIOSlotMap[] owners = slab.owner;
        final byte[] hitFlags = slab.hit;
        final int[] free = slab.free;
        int freeCount = 0;
        List<DirectIOSlotMap> emptied = null;
        for (int slot = 0; slot < slab.used; slot++) {
            final DirectIOSlotMap owner = owners[slot];
            if (owner == null) {
                free[freeCount++] = slot;
                continue;
            }
            if (hitFlags[slot] == 0 || kept >= keep) {
                // The bookkeeping of detach(), with the counters accumulated rather than published per slot:
                // the two LongAdder writes per slot were a measurable part of the write-lock hold on a slab
                // with ten thousand of them.
                owners[slot] = null;
                owner.noteInvalidated();
                if (owner.noteDetached()) {
                    emptied = emptied == null ? new ArrayList<>() : emptied;
                    emptied.add(owner);
                }
                dropped++;
                free[freeCount++] = slot;
                continue;
            }
            hitFlags[slot] = 0;
            kept++;
        }
        slab.freeCount = freeCount;
        liveEntries -= dropped;
        liveEntryBytes -= (long) dropped * slab.slotBytes;
        DirectIOVectorCache.droppedEntries.add(dropped);
        DirectIOVectorCache.readmissions.add(kept);
        if (emptied != null) {
            // The closed-file rule's second half, done once per reclaim rather than once per slot: a file
            // whose last cached vector has just gone loses its index table too.
            for (final DirectIOSlotMap owner : emptied) {
                if (owner.liveEntries() <= 0) {
                    index.remove(owner.fileId(), owner);
                }
            }
        }
    }

    /** Frees every live slot of {@code slab} and empties it. Called under the write lock. */
    private void dropAll(final Slab slab) {
        slab.freeCount = 0;
        for (int slot = 0; slot < slab.used; slot++) {
            detach(slab, slot, true);
        }
        slab.used = 0;
    }

    /**
     * Stops counting {@code slot} as live, and drops its file's index table when that was the file's last
     * entry. {@code leaveCellStale} says whether the index cell naming the slot is being left behind, which
     * is the case for an eviction and not for a re-cache that is about to overwrite the cell.
     */
    private void detach(final Slab slab, final int slot, final boolean leaveCellStale) {
        final DirectIOSlotMap owner = slab.owner[slot];
        if (owner == null) {
            return;
        }
        slab.owner[slot] = null;
        liveEntries--;
        liveEntryBytes -= slab.slotBytes;
        DirectIOVectorCache.droppedEntries.increment();
        if (leaveCellStale) {
            owner.noteInvalidated();
        }
        if (owner.noteDetached()) {
            index.remove(owner.fileId(), owner);
        }
    }

    /** Records that {@code slot} of {@code slab} now holds the vector at {@code offset} of {@code map}. */
    private void attach(final Slab slab, final int slot, final DirectIOSlotMap map, final long offset) {
        final int previous = map.find(offset);
        if (previous >= 0) {
            final int previousSlabId = map.slabIdAt(previous);
            final int previousSlot = map.slotAt(previous);
            final Slab previousSlab = slabAt(previousSlabId);
            if (previousSlab != null
                && previousSlot < previousSlab.slotCount
                && previousSlab.owner[previousSlot] == map
                && previousSlab.offset[previousSlot] == offset) {
                // The cell is about to be overwritten, so it is not left stale. The map cannot empty here,
                // because this very offset is about to be attached again, but detach's index removal would
                // fire if it did; re-attaching below puts it back, so the map object stays the one the slot
                // points at either way.
                detach(previousSlab, previousSlot, false);
            }
        }
        if (index.get(map.fileId()) != map) {
            index.put(map.fileId(), map);
        }
        map.put(offset, slab.id, slot);
        map.noteAttached();
        slab.owner[slot] = map;
        slab.offset[slot] = offset;
        slab.hit[slot] = 0;
        liveEntries++;
        liveEntryBytes += slab.slotBytes;
    }

    private Slab slabAt(final int id) {
        final Slab[] slabs = slabsById;
        return id >= 0 && id < slabs.length ? slabs[id] : null;
    }

    /** Opens {@code slab} for {@code length}-byte vectors and makes it the one being appended to. */
    private void open(final Slab slab, final int length) {
        slab.openFor(length);
        openSlabs.put(length, slab);
    }

    private Slab register(final DirectIOSlab memory) {
        int id = 0;
        while (id < slabsById.length && slabsById[id] != null) {
            id++;
        }
        if (id == slabsById.length) {
            final Slab[] grown = new Slab[Math.max(4, slabsById.length * 2)];
            System.arraycopy(slabsById, 0, grown, 0, slabsById.length);
            slabsById = grown;
        }
        final Slab slab = new Slab(id, memory);
        slabsById[id] = slab;
        liveSlabs++;
        fifo.add(slab);
        // Pending to live in one step, so no sum the invariant check reads is ever wrong.
        broker.commitReservation();
        return slab;
    }

    /**
     * Takes this segment's oldest slab out of service, dropping its entries, and adds its memory to
     * {@code detached} for the caller to release outside every lock (N3). One write-lock acquisition.
     *
     * <p>Clearing {@code slabsById} is what makes the release safe outside the lock: every lookup resolves a
     * slab through {@link #slabAt} under the read lock, so a reader arriving after this cannot reach the slab
     * and a reader already inside held the read lock this thread had to wait for.
     *
     * @param detached the list to add the slab's memory to
     * @return whether a slab was released, i.e. whether this segment had one
     */
    boolean releaseOldestSlab(final List<DirectIOSlab> detached) {
        lockWrite();
        try {
            final Slab oldest = fifo.poll();
            if (oldest == null) {
                return false;
            }
            dropAll(oldest);
            slabsById[oldest.id] = null;
            openSlabs.remove(oldest.slotBytes, oldest);
            liveSlabs--;
            broker.releaseLive();
            detached.add(oldest.memory);
            return true;
        } finally {
            unlockWrite();
        }
    }

    /**
     * Hands this segment's oldest slab to another segment, dropping its entries, and returns its memory, or
     * {@code null} when this segment has none to give. Called by {@link DirectIOSlabBroker} with the broker's
     * lock held and <b>no other segment's lock</b>, which is the order the whole design turns on.
     *
     * <p>A transfer has no second chance, because a survivor has nowhere to stay: the recipient reopens the
     * slab for its own vectors. That is the same reason a reopen for a different vector length drops them.
     *
     * <p>The slab stays live as far as the broker is concerned -- it becomes one of its reservations, which
     * the recipient then commits -- so the sum of the segments' slab counts plus the reservations in flight
     * is what the broker counts at every instant, including mid-transfer.
     *
     * @return the slab's memory, or null
     */
    DirectIOSlab detachOldestForTransfer() {
        lockWrite();
        try {
            final Slab oldest = fifo.poll();
            if (oldest == null) {
                return null;
            }
            dropAll(oldest);
            slabsById[oldest.id] = null;
            openSlabs.remove(oldest.slotBytes, oldest);
            liveSlabs--;
            return oldest.memory;
        } finally {
            unlockWrite();
        }
    }

    /**
     * Takes this segment out of service ahead of a segment-count change, so no later fill can put a slab in
     * it. Its slabs and its index are drained by the caller; see {@link #retired}.
     */
    void retire() {
        lockWrite();
        try {
            retired = true;
        } finally {
            unlockWrite();
        }
    }

    /** Drops this segment's index, open slabs and recycled queue. Called when the store goes off. */
    void clearIndex() {
        lockWrite();
        try {
            index.clear();
            openSlabs.clear();
            recycled.clear();
            recycledSlabs = 0;
        } finally {
            unlockWrite();
        }
    }

    // ---------------------------------------------------------------------------------------------------
    // Counters and test seams
    // ---------------------------------------------------------------------------------------------------

    long liveEntries() {
        return liveEntries;
    }

    long liveEntryBytes() {
        return liveEntryBytes;
    }

    int liveSlabs() {
        return liveSlabs;
    }

    /** Heap held by this segment's index, for the bytes-per-entry figure M2's bench reports. */
    long indexBytes() {
        lock.readLock().lock();
        try {
            long total = 48L * index.size();
            for (final DirectIOSlotMap map : index.values()) {
                total += map.indexBytes();
            }
            for (final Slab slab : fifo) {
                if (slab.owner != null) {
                    total += 48 + 4L * slab.owner.length + 16 + 8L * slab.offset.length + 16 + slab.hit.length;
                    total += 16 + 4L * slab.free.length;
                }
            }
            return total;
        } finally {
            lock.readLock().unlock();
        }
    }

    int indexedFiles() {
        lock.readLock().lock();
        try {
            return index.size();
        } finally {
            lock.readLock().unlock();
        }
    }

    boolean hasIndexFor(final String fileId) {
        lock.readLock().lock();
        try {
            return index.containsKey(fileId);
        } finally {
            lock.readLock().unlock();
        }
    }

    /** Whether the calling thread holds this segment's lock in either mode. N3's test seam. */
    boolean lockHeldByCurrentThread() {
        return lock.isWriteLockedByCurrentThread() || lock.getReadHoldCount() > 0;
    }

    void lockForInvariantCheck() {
        lock.writeLock().lock();
    }

    void unlockAfterInvariantCheck() {
        lock.writeLock().unlock();
    }

    /**
     * Re-derives this segment's live entry count and byte total from its slabs and checks them against its
     * running counters, so a test can assert that no path has lost or double counted a slot. Every slot that
     * still carries a back pointer must also be the slot its file's index names for it, and that index must
     * be <b>this segment's</b>, which is the invariant the segmentation adds.
     *
     * <p>Called with this segment's write lock held, by {@link DirectIOVectorCache#checkInvariantsForTesting}.
     *
     * @param totals {@code {entries, entryBytes, slabs}}, added to
     * @throws IllegalStateException on any disagreement, naming it
     */
    void checkInvariantsUnderLock(final long[] totals) {
        long entries = 0;
        long entryBytes = 0;
        final Map<DirectIOSlotMap, Integer> counted = new HashMap<>();
        for (final Slab slab : fifo) {
            if (slab.used > slab.slotCount) {
                throw new IllegalStateException("slab " + slab.id + " used " + slab.used + " of " + slab.slotCount);
            }
            for (int slot = 0; slot < slab.used; slot++) {
                final DirectIOSlotMap owner = slab.owner[slot];
                if (owner == null) {
                    continue;
                }
                final int cell = owner.find(slab.offset[slot]);
                if (cell < 0 || owner.slabIdAt(cell) != slab.id || owner.slotAt(cell) != slot) {
                    throw new IllegalStateException("slab " + slab.id + " slot " + slot + " is not named by its index entry");
                }
                if (index.get(owner.fileId()) != owner) {
                    throw new IllegalStateException("slab " + slab.id + " slot " + slot + " is owned by an unindexed file");
                }
                counted.merge(owner, 1, Integer::sum);
                entries++;
                entryBytes += slab.slotBytes;
            }
            for (int i = 0; i < slab.freeCount; i++) {
                // A slot on the free list and still live would be handed to a fill that overwrites a
                // vector the index still names, which is the one way the free list could corrupt a read.
                if (slab.owner[slab.free[i]] != null) {
                    throw new IllegalStateException("slab " + slab.id + " slot " + slab.free[i] + " is both free and live");
                }
            }
        }
        if (entries != liveEntries) {
            throw new IllegalStateException("counted " + entries + " live entries but the counter says " + liveEntries);
        }
        if (entryBytes != liveEntryBytes) {
            throw new IllegalStateException("counted " + entryBytes + " live bytes but the counter says " + liveEntryBytes);
        }
        if (fifo.size() != liveSlabs) {
            throw new IllegalStateException("queue holds " + fifo.size() + " slabs but the counter says " + liveSlabs);
        }
        for (final Map.Entry<DirectIOSlotMap, Integer> entry : counted.entrySet()) {
            if (entry.getKey().liveEntries() != entry.getValue()) {
                throw new IllegalStateException(
                    "file "
                        + entry.getKey().fileId()
                        + " holds "
                        + entry.getValue()
                        + " slots but its map says "
                        + entry.getKey().liveEntries()
                );
            }
        }
        totals[0] += entries;
        totals[1] += entryBytes;
        totals[2] += liveSlabs;
    }
}

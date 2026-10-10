/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.memory;

import lombok.extern.log4j.Log4j2;

import java.nio.ByteBuffer;
import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.LongAdder;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * The node's cache of full-precision vectors, held in {@link VectorCacheSlab}s and keyed by the segment that
 * owns the vector rather than by the file it was read out of.
 *
 * <h2>What this is for (D1)</h2>
 * Phase 15 cached vectors inside {@code DirectIOVectorIndexInput}, keyed by {@code (file, offset)} and held
 * on the heap. That works only where Direct I/O works: the input declines under index-level encryption and
 * under the AOSS buffer pool, so the cache went with it. D1 moves the lookup above the {@code IndexInput},
 * into the scorer, so the same cache serves whatever input a segment happens to have, and moves the memory
 * off the heap into slabs the k-NN native budget accounts for. This class is the index and the eviction
 * policy; {@link VectorCacheSlab} is the memory; {@link NativeMemoryCacheManager} grants the slabs and
 * owns the node's instance of this class; I4 calls {@link #get} and {@link #put} from the scorer.
 *
 * <h2>The key is the segment's core identity, the field and the ordinal</h2>
 * A Lucene segment's core identity ({@code LeafReader#getCoreCacheHelper().getKey()}) is stable for as long
 * as the vectors are readable and is released when the core closes, which is exactly the lifetime of the
 * cached bytes. That fixes the one thing {@code DirectIOVectorCache} documented as a deliberate gap: its
 * entries for a merged-away segment stayed until LRU pushed them out, because invalidating them would have
 * needed a per-file key index. Here the index <em>is</em> per segment field, so
 * {@link #releaseSegment(Object)} is a map removal and a scan of the slabs rather than a new structure.
 *
 * <h2>Slots are fixed size per vector byte length</h2>
 * A slab is opened for one vector byte length and divided into slots of exactly that length, so placing a
 * vector is a bump pointer and finding it is a multiply. One field's vectors are all one length, and a node
 * usually has one or two lengths in play, so a slab is never fragmented by mixed sizes. A slab opened for a
 * different length is a different slab.
 *
 * <h2>Eviction is FIFO over slabs, with one second chance (proof d)</h2>
 * Evicting one vector at a time would need a recency structure per vector, which is the per-entry cost the
 * design note rejected. Instead the slabs form a FIFO queue and the oldest slab is the eviction unit. A
 * vector that has been read since it was written carries a hit flag, and when its slab comes up for eviction
 * it is re-admitted rather than dropped: the survivors <em>stay exactly where they are</em>, the slots of the
 * vectors that were never read become free slots in that same slab, and the slab is reopened for the same
 * vector length and appended to through its free list.
 *
 * <p>A free list per slab rather than a bump pointer alone is what makes that possible, and it is the one
 * thing I4 changed about this policy. I2 compacted the survivors to the front of the slab so that appends
 * could stay a bump pointer, and MEASURED that copy at 1.7 to 4.8 ms with the write lock held throughout,
 * which is a stall of every search thread every few seconds. Nothing needed the survivors to be contiguous.
 * Leaving them in place makes a reclaim a scan of the slot table with no byte copying and no index update
 * for any survivor, since a survivor's {@code (slab, slot)} does not change.
 *
 * <p>Re-admission is capped at half a slab's slots, which is what makes an eviction always yield room and the
 * algorithm terminate in one step: a fully hot slab cannot re-admit itself whole and so cannot deadlock the
 * cache against its own hit flags. The cap is the same device S3-FIFO and Segcache use for the same reason.
 * There is no admission filter: scan resistance is undecided (agenda item 4) and exact search fills this
 * cache exactly as rescore does.
 *
 * <p>A slab reopened for a <em>different</em> vector length is the one case with no second chance: its slot
 * layout changes, so there is nowhere for a survivor to stay. Those entries are dropped. A copy into another
 * slab of the old length would be the only alternative and it would put a copy back under the write lock,
 * which is the thing this policy exists to avoid; the case needs two vector lengths competing for the same
 * slabs, which is rare, and costs a refill rather than an answer.
 *
 * <h2>One lock, and why the read path can still be concurrent</h2>
 * A reader must not be shown a slot while another thread is overwriting it, and the index must not be read
 * half-rebuilt. Both are solved by one {@link ReentrantReadWriteLock} over the whole cache: a lookup takes
 * the read lock, and everything structural (insert, eviction, release, resize) takes the write lock. Lookups
 * therefore run concurrently with each other, which is what the hot path needs, and never concurrently with
 * a mutation, which is what lets the index be plain {@link HashMap}s and {@link OrdinalSlotMap}s with no
 * per-entry synchronisation at all. Inserts serialise against each other, and that is affordable because an
 * insert only ever follows a read miss, so its rate is bounded by the device rather than by the CPU. It is
 * also why this class does not take {@link VectorCacheSlab#readLock()} on the read path: a slab is only ever
 * closed under this cache's write lock, which a reader excludes.
 *
 * <p>A lookup copies the vector out into the caller's array while it holds the read lock, and hands back no
 * view of slab memory at all. That is what makes a use after free unreachable rather than merely unlikely:
 * there is no buffer for a caller to stash and read after the lock is dropped, which is the one failure mode
 * a JNI slab gives the JVM no way to reject. {@link #lookupBatch} is the form the scorer uses, and it takes
 * the read lock once for a whole Lucene batch of 64 ordinals rather than once per vector.
 *
 * <h2>Off is off</h2>
 * With {@link #configure(long)} at zero there are no slabs, and {@link #get} and {@link #put} return on a
 * single volatile read before taking any lock or building any key.
 *
 * <h2>What it costs and what it keeps (MEASURED, output/bench/i2)</h2>
 * Dev desktop, 768-dim vectors, 256 MB of slabs:
 *
 * <table>
 *   <caption>throughput and per-operation cost</caption>
 *   <tr><th></th><th>1 thread</th><th>8 threads</th></tr>
 *   <tr><td>lookups/s, {@link #lookupBatch}, every key cached</td><td>2,200,577</td>
 *       <td><b>14,746,083</b></td></tr>
 *   <tr><td>per hit, p50 / p99</td><td>436 / 538 ns</td><td>505 / 670 ns</td></tr>
 *   <tr><td>lookups/s, one acquisition per vector (I2's form)</td><td>1,592,673</td><td>3,716,721</td></tr>
 *   <tr><td>operations/s, 76% of them fills</td><td>1,901,863</td><td>445,671</td></tr>
 *   <tr><td>write lock held per slab reclaim</td><td colspan="2">26,513 ns p50, 715,358 ns for the first
 *       one in a JVM, where the path is still interpreted</td></tr>
 *   <tr><td>index bytes per cached vector</td><td colspan="2">37.0 to 40.2 B, against the 200 B the Guava
 *       cache charged</td></tr>
 * </table>
 *
 * The read path scales 4.0x off the per-vector form because of the batch acquisition alone. The fill path
 * still takes the lock once per vector and does not scale, and that is left as it is: a fill only ever
 * follows a read miss, so a node's fill rate is its device's IOPS, around 3,000/s on the pilot volume
 * against the 190,000/s this measurement drives, and the scorer has one miss at a time to give it.
 *
 * <p>Against an exact LRU of the same byte budget on a trace fitted to the phase 15 reuse measurement, this
 * policy is within <b>0.2 points</b> of LRU's hit rate at 64, 128 and 256 MB, with the LRU charged nothing
 * for its own index. Written up with the numbers in output/bench/i2/README-i2.md and
 * output/bench/i4/README-i4.md.
 */
@Log4j2
public final class SlabVectorCache {

    /**
     * Where slabs come from and go back to. The default allocates and frees directly; I3 replaces it with
     * one that takes a grant from {@link NativeMemoryCacheManager} so the slabs are accounted under the
     * k-NN native budget and can be reclaimed when a graph needs room.
     */
    public interface SlabSource {
        /** A slab of {@code bytes}, or {@code null} when the source will not grant one now. */
        VectorCacheSlab acquire(int bytes);

        /** Returns a slab obtained from {@link #acquire(int)}. */
        void release(VectorCacheSlab slab);
    }

    /**
     * Allocates and frees without asking anyone. What the microbenchmarks use, and what a test uses when it
     * is not testing the accounting. The node's cache is owned by {@link NativeMemoryCacheManager} and takes
     * every slab as a grant from it instead; see {@code NativeMemoryCacheManager#getVectorCache}.
     */
    public static final SlabSource DIRECT_SOURCE = new SlabSource() {
        @Override
        public VectorCacheSlab acquire(final int bytes) {
            return VectorCacheSlab.allocate(bytes);
        }

        @Override
        public void release(final VectorCacheSlab slab) {
            slab.close();
        }
    };

    /** One segment field. {@code core} is a Lucene core cache key, whose equality is identity. */
    private record FieldKey(Object core, int field) {
    }

    /**
     * One slab in service: its memory, the slot layout it was opened for, and one back pointer per slot so
     * that eviction can find the index entry that owns it and a lookup can confirm it has not been taken.
     */
    private static final class Slab {
        private final int id;
        private final VectorCacheSlab memory;
        private int slotBytes;
        private int slotCount;
        /** High-water mark: slots {@code [0, used)} have been written since the slab was opened. */
        private int used;
        private OrdinalSlotMap[] owner;
        private int[] ordinal;
        /** 1 once the slot has been read, which is what buys it a second chance. */
        private byte[] hit;
        /**
         * Slots below {@link #used} that a reclaim freed, so that a survivor can stay where it is instead of
         * being copied to the front. {@code [0, freeCount)} is the live part of the stack.
         */
        private int[] free;
        private int freeCount;

        private Slab(final int id, final VectorCacheSlab memory) {
            this.id = id;
            this.memory = memory;
        }

        /** Re-opens the slab for {@code bytes}-long vectors, discarding the previous slot layout. */
        private void openFor(final int bytes) {
            final int count = Math.min(memory.capacity() / bytes, OrdinalSlotMap.MAX_SLOTS);
            if (owner == null || count > owner.length) {
                owner = new OrdinalSlotMap[count];
                ordinal = new int[count];
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

    private final SlabSource source;
    private final int slabBytes;

    private final ReentrantReadWriteLock lock = new ReentrantReadWriteLock();

    /** Field index. Mutated under the write lock, read under the read lock, so a plain map is enough. */
    private final Map<FieldKey, OrdinalSlotMap> index = new HashMap<>();

    /** Slabs in service, oldest first. The head is the eviction unit. */
    private final ArrayDeque<Slab> fifo = new ArrayDeque<>();

    /** The one slab currently being appended to, per vector byte length. */
    private final Map<Integer, Slab> openSlabs = new HashMap<>();

    /** Slabs by id, so an index entry can be one int. Null where an id is not in use. */
    private Slab[] slabsById = new Slab[0];

    /** Slabs the cache may hold, from {@link #configure(long)}. Zero means off, and is the default. */
    private volatile int maxSlabs;

    private int liveSlabs;
    private long liveEntries;
    private long liveEntryBytes;

    private final LongAdder hits = new LongAdder();
    private final LongAdder misses = new LongAdder();
    private final LongAdder insertions = new LongAdder();
    private final LongAdder reclaimedSlabs = new LongAdder();
    private final LongAdder readmissions = new LongAdder();
    private final LongAdder droppedEntries = new LongAdder();
    /** Write-lock time spent reclaiming slabs, which is the figure I4 had to bring down. */
    private final LongAdder reclaimNanos = new LongAdder();

    /**
     * @param source where slabs come from and go back to
     * @param slabBytes the fixed slab size, {@link VectorCacheSlab#SLAB_BYTES} outside tests and benchmarks
     */
    public SlabVectorCache(final SlabSource source, final int slabBytes) {
        this.source = source;
        this.slabBytes = slabBytes;
    }

    // ---------------------------------------------------------------------------------------------------
    // The read path
    // ---------------------------------------------------------------------------------------------------

    /**
     * Copies the cached vector into {@code destination} and returns {@code true}, or returns {@code false}
     * and leaves {@code destination} alone.
     *
     * <p>A hit also flags the slot, which is what earns it a second chance when its slab is evicted.
     *
     * @param core the segment core identity the vector belongs to
     * @param field the field number
     * @param ordinal the vector's ordinal in that field
     * @param vectorBytes the vector's length in bytes, checked against the slot so that a caller asking for
     *                    a different length than was cached gets a miss rather than a wrong vector
     * @param destination at least {@code vectorBytes / 4} floats
     * @return whether the vector was cached
     */
    public boolean get(final Object core, final int field, final int ordinal, final int vectorBytes, final float[] destination) {
        if (maxSlabs == 0) {
            return false;
        }
        lock.readLock().lock();
        try {
            final OrdinalSlotMap map = index.get(new FieldKey(core, field));
            final boolean hit = map != null && copyOut(map, ordinal, vectorBytes, destination, 0);
            if (hit) {
                hits.increment();
            } else {
                misses.increment();
            }
            return hit;
        } finally {
            lock.readLock().unlock();
        }
    }

    /**
     * What a batch lookup hands each hit to while the read lock is still held.
     *
     * <p>The contract is narrow on purpose, because this runs under the cache's read lock: an implementation
     * must not read a file, take a lock, or keep a reference to {@code vector}, which is one buffer reused
     * for every hit in the batch. It exists so that a hit can be scored out of the buffer the cache just
     * filled, which is what lets a whole batch share one acquisition and one buffer instead of needing an
     * array per candidate. See {@code VectorCacheBatchScorer}.
     */
    public interface HitVisitor {
        /**
         * @param index the position in the batch's {@code ordinals} array
         * @param vector the cached vector, valid only for the duration of this call
         */
        void onHit(int index, float[] vector);
    }

    /**
     * Looks up a whole batch of ordinals of one segment field under a single read-lock acquisition, copying
     * each hit into {@code buffer}, handing it to {@code visitor}, and recording in {@code hit[i]} whether
     * ordinal {@code i} was cached.
     *
     * <p>This is the form the scorer uses, and the reason it exists is MEASURED rather than tidiness. I2
     * took the lock once per vector and reached only 2.1x on eight threads, because Lucene hands a scorer 64
     * ordinals at a time (FINDINGS I0.2) and 64 acquisitions per batch put every search thread through the
     * same lock 64 times as often as it needed to be.
     *
     * @param core the segment core identity the vectors belong to
     * @param field the field number
     * @param ordinals the ordinals to look up, {@code [0, count)} of which are read
     * @param count how many ordinals to look up
     * @param vectorBytes the vector length in bytes, the same for every ordinal of one field
     * @param buffer at least {@code vectorBytes / 4} floats, overwritten once per hit
     * @param hit at least {@code count} entries; set for every ordinal, true where it was cached
     * @param visitor told about each hit, or {@code null} for a caller that only wants {@code hit}
     * @return how many of the {@code count} ordinals were cached
     */
    public int lookupBatch(
        final Object core,
        final int field,
        final int[] ordinals,
        final int count,
        final int vectorBytes,
        final float[] buffer,
        final boolean[] hit,
        final HitVisitor visitor
    ) {
        if (maxSlabs == 0 || count <= 0) {
            Arrays.fill(hit, 0, Math.max(0, count), false);
            return 0;
        }
        int found = 0;
        lock.readLock().lock();
        try {
            final OrdinalSlotMap map = index.get(new FieldKey(core, field));
            if (map == null) {
                Arrays.fill(hit, 0, count, false);
                misses.add(count);
                return 0;
            }
            for (int i = 0; i < count; i++) {
                final boolean cached = copyOut(map, ordinals[i], vectorBytes, buffer, 0);
                hit[i] = cached;
                if (cached) {
                    found++;
                    if (visitor != null) {
                        visitor.onHit(i, buffer);
                    }
                }
            }
        } finally {
            lock.readLock().unlock();
        }
        hits.add(found);
        misses.add(count - found);
        return found;
    }

    /**
     * Copies one ordinal's vector into {@code destination} at {@code destinationOffset} floats, or returns
     * false. Called with the read lock held.
     */
    private boolean copyOut(
        final OrdinalSlotMap map,
        final int ordinal,
        final int vectorBytes,
        final float[] destination,
        final int destinationOffset
    ) {
        final int cell = map.find(ordinal);
        if (cell < 0) {
            return false;
        }
        final int slabId = map.slabIdAt(cell);
        final int slot = map.slotAt(cell);
        // The slot's back pointer is the proof. A cell can outlive the slot it names, because eviction
        // does not walk the index; see OrdinalSlotMap's javadoc.
        if (slotHolds(map, ordinal, slabId, slot) == false) {
            return false;
        }
        final Slab slab = slabsById[slabId];
        if (slab.slotBytes != vectorBytes) {
            return false;
        }
        slab.memory.floats().get((slot * slab.slotBytes) >> 2, destination, destinationOffset, vectorBytes >> 2);
        // A plain write, and a lost one only costs a second chance. Readers run concurrently with each
        // other but never with the eviction that reads this, which holds the write lock.
        slab.hit[slot] = 1;
        return true;
    }

    /** Whether a lookup can succeed at all, so a caller can skip the work of asking. */
    public boolean isOn() {
        return maxSlabs != 0;
    }

    // ---------------------------------------------------------------------------------------------------
    // The fill path
    // ---------------------------------------------------------------------------------------------------

    /**
     * Caches {@code length} bytes at {@code position} of {@code source} as the vector at {@code ordinal}.
     * A copy and not a view, because {@code source} is a staging buffer the next read reuses.
     *
     * <p>Silently does nothing when the cache is off, when the length is not a whole number of floats, or
     * when no slab can be had. A fill that does not happen costs a future hit and nothing else.
     */
    public void put(final Object core, final int field, final int ordinal, final ByteBuffer source, final int position, final int length) {
        if (maxSlabs == 0 || length <= 0 || (length & 3) != 0 || ordinal < 0 || length > slabBytes) {
            return;
        }
        lock.writeLock().lock();
        try {
            final Placement placement = place(core, field, ordinal, length);
            if (placement == null) {
                return;
            }
            placement.slab.memory.bytes().put(placement.slot * length, source, position, length);
            commit(placement, ordinal);
        } finally {
            lock.writeLock().unlock();
        }
    }

    /**
     * Caches {@code length} bytes worth of {@code vector} as the vector at {@code ordinal}. The form the
     * scorer fills from, because what it has after a miss is the {@code float[]} the segment's
     * {@code IndexInput} decoded into, not the bytes behind it.
     *
     * <p>Silently does nothing on the same terms as {@link #put(Object, int, int, ByteBuffer, int, int)}.
     */
    public void put(final Object core, final int field, final int ordinal, final float[] vector, final int length) {
        if (maxSlabs == 0 || length <= 0 || (length & 3) != 0 || ordinal < 0 || length > slabBytes || vector.length < (length >> 2)) {
            return;
        }
        lock.writeLock().lock();
        try {
            final Placement placement = place(core, field, ordinal, length);
            if (placement == null) {
                return;
            }
            placement.slab.memory.floats().put((placement.slot * length) >> 2, vector, 0, length >> 2);
            commit(placement, ordinal);
        } finally {
            lock.writeLock().unlock();
        }
    }

    /** Where a fill is about to write: the field's index, a slab and a slot in it. */
    private record Placement(OrdinalSlotMap map, Slab slab, int slot) {
    }

    /**
     * Finds room for one vector of {@code length} bytes, evicting a slab if that is what it takes, and
     * returns where to write it or {@code null} when there is nowhere. Called under the write lock; the
     * caller writes the bytes and then calls {@link #commit}.
     */
    private Placement place(final Object core, final int field, final int ordinal, final int length) {
        if (maxSlabs == 0) {
            return null;
        }
        final FieldKey key = new FieldKey(core, field);
        OrdinalSlotMap map = index.get(key);
        if (map == null) {
            map = new OrdinalSlotMap();
            index.put(key, map);
        }
        if (map.needsCompaction()) {
            final OrdinalSlotMap compacting = map;
            map.compact((o, slabId, slot) -> slotHolds(compacting, o, slabId, slot));
        }
        Slab slab = openSlabs.get(length);
        if (slab == null || slab.freeSlots() == 0) {
            slab = obtainSlab(length);
        }
        if (slab == null) {
            return null;
        }
        final int slot = slab.allocSlot();
        return slot < 0 ? null : new Placement(map, slab, slot);
    }

    /** Records a written slot in the index. Called under the write lock, after the bytes are in place. */
    private void commit(final Placement placement, final int ordinal) {
        attach(placement.slab, placement.slot, placement.map, ordinal);
        insertions.increment();
    }

    /**
     * Drops every entry belonging to {@code core}, for every field, and frees the slots they held.
     *
     * <p>Called when a segment's core closes. The slots become free for the FIFO to reuse but the slab is
     * not compacted here: a closed segment's bytes are dead weight only until its slab reaches the head of
     * the queue, and compacting on close would put a scan of the whole cache on the merge path.
     */
    public void releaseSegment(final Object core) {
        if (maxSlabs == 0) {
            return;
        }
        lock.writeLock().lock();
        try {
            final Set<OrdinalSlotMap> released = Collections.newSetFromMap(new IdentityHashMap<>());
            final Iterator<Map.Entry<FieldKey, OrdinalSlotMap>> fields = index.entrySet().iterator();
            while (fields.hasNext()) {
                final Map.Entry<FieldKey, OrdinalSlotMap> entry = fields.next();
                if (entry.getKey().core() == core) {
                    released.add(entry.getValue());
                    fields.remove();
                }
            }
            if (released.isEmpty()) {
                return;
            }
            // The index entries are gone, so the slots they named have to stop being counted as live and
            // must not be re-admitted on a second chance. Clearing the back pointer does both.
            for (final Slab slab : fifo) {
                for (int slot = 0; slot < slab.used; slot++) {
                    if (released.contains(slab.owner[slot])) {
                        slab.owner[slot] = null;
                        liveEntries--;
                        liveEntryBytes -= slab.slotBytes;
                        droppedEntries.increment();
                    }
                }
            }
        } finally {
            lock.writeLock().unlock();
        }
    }

    // ---------------------------------------------------------------------------------------------------
    // Sizing
    // ---------------------------------------------------------------------------------------------------

    /**
     * Sets the cache's byte target, adding or releasing slabs to match. Zero empties the cache and returns
     * every slab to the source.
     *
     * <p>The target is rounded down to whole slabs, because a slab is the unit the cache allocates and
     * reclaims in. A target under one slab therefore holds nothing, which is logged rather than silently
     * rounded up past the operator's number.
     *
     * <p>This resizes; it does not rebuild. Growing keeps every cached vector and lets the next miss
     * allocate, and shrinking drops the oldest slabs. That is the difference the design note asks for
     * against {@code DirectIOVectorCache#configure}, which rebuilds and so loses the whole cache on a
     * change, and against {@code NativeMemoryCacheManager#rebuildCache}, which does the same to graphs.
     */
    public void configure(final long maxBytes) {
        final int wanted = (int) Math.min(OrdinalSlotMap.MAX_SLABS, Math.max(0, maxBytes) / slabBytes);
        if (wanted == maxSlabs) {
            // Before the lock and not after it, because this is called once per prefetch burst from the
            // read path to pick up a setting change, and in the steady state the answer is always no change.
            // Taking the write lock to find that out would put every burst through the lock a fill uses.
            return;
        }
        lock.writeLock().lock();
        try {
            if (wanted == maxSlabs) {
                return;
            }
            if (maxBytes > 0 && wanted == 0) {
                log.warn(
                    "A vector cache target of {} bytes is under one {} byte slab, so the cache will hold nothing",
                    maxBytes,
                    slabBytes
                );
            }
            maxSlabs = wanted;
            while (liveSlabs > wanted) {
                final Slab oldest = fifo.poll();
                if (oldest == null) {
                    break;
                }
                dropAll(oldest);
                unregister(oldest);
            }
            if (wanted == 0) {
                index.clear();
                openSlabs.clear();
            }
            log.info("The k-NN vector cache holds up to {} slabs of {} bytes", wanted, slabBytes);
        } finally {
            lock.writeLock().unlock();
        }
    }

    // ---------------------------------------------------------------------------------------------------
    // Slab supply and eviction
    // ---------------------------------------------------------------------------------------------------

    /**
     * A slab with room for a {@code length}-byte vector, by allocating one while the cache is under its
     * target and otherwise by evicting the oldest. Returns {@code null} only when the source declined and
     * there was nothing to evict. Called under the write lock.
     */
    private Slab obtainSlab(final int length) {
        if (liveSlabs < maxSlabs) {
            final VectorCacheSlab memory = source.acquire(slabBytes);
            if (memory != null) {
                final Slab fresh = register(memory);
                open(fresh, length);
                return fresh;
            }
        }
        final Slab oldest = fifo.poll();
        if (oldest == null) {
            return null;
        }
        reclaimedSlabs.increment();
        final long started = System.nanoTime();
        if (oldest.slotBytes == length) {
            // The common case, and the only one in which a second chance can do anything: the evicted slab
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
        reclaimNanos.add(System.nanoTime() - started);
        return oldest;
    }

    /**
     * Gives {@code slab}'s second chance: the slots that have been read since they were written keep both
     * their vector and their slot, up to half the slab, and every other slot joins the free list. No byte
     * moves and no index entry changes, which is what keeps the write lock hold off the millisecond scale.
     */
    private void reclaimInPlace(final Slab slab) {
        // Half the slots, rounded down, so that a slab with a single slot keeps nothing and the caller is
        // always handed room for the vector it came to write.
        final int keep = slab.slotCount / 2;
        int kept = 0;
        int dropped = 0;
        slab.freeCount = 0;
        final OrdinalSlotMap[] owners = slab.owner;
        final byte[] hits = slab.hit;
        final int[] free = slab.free;
        int freeCount = 0;
        for (int slot = 0; slot < slab.used; slot++) {
            final OrdinalSlotMap owner = owners[slot];
            if (owner == null) {
                free[freeCount++] = slot;
                continue;
            }
            if (hits[slot] == 0 || kept >= keep) {
                // The bookkeeping of detach(), with the counters accumulated rather than published per
                // slot: the two LongAdder writes per slot were a measurable part of the write-lock hold on
                // a slab with ten thousand of them.
                owners[slot] = null;
                owner.noteInvalidated();
                dropped++;
                free[freeCount++] = slot;
                continue;
            }
            hits[slot] = 0;
            kept++;
        }
        slab.freeCount = freeCount;
        liveEntries -= dropped;
        liveEntryBytes -= (long) dropped * slab.slotBytes;
        droppedEntries.add(dropped);
        readmissions.add(kept);
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
     * Stops counting {@code slot} as live. {@code leaveCellStale} says whether the index cell that names it
     * is being left behind, which is the case for an eviction and not for a re-cache that is about to
     * overwrite the cell.
     */
    private void detach(final Slab slab, final int slot, final boolean leaveCellStale) {
        final OrdinalSlotMap owner = slab.owner[slot];
        if (owner == null) {
            return;
        }
        slab.owner[slot] = null;
        liveEntries--;
        liveEntryBytes -= slab.slotBytes;
        droppedEntries.increment();
        if (leaveCellStale) {
            owner.noteInvalidated();
        }
    }

    /** Records that {@code slot} of {@code slab} now holds {@code ordinal} of {@code map}. */
    private void attach(final Slab slab, final int slot, final OrdinalSlotMap map, final int ordinal) {
        final int previous = map.find(ordinal);
        if (previous >= 0) {
            final int previousSlabId = map.slabIdAt(previous);
            final int previousSlot = map.slotAt(previous);
            if (slotHolds(map, ordinal, previousSlabId, previousSlot)) {
                // The cell is about to be overwritten, so it is not left stale.
                detach(slabsById[previousSlabId], previousSlot, false);
            }
        }
        map.put(ordinal, slab.id, slot);
        slab.owner[slot] = map;
        slab.ordinal[slot] = ordinal;
        slab.hit[slot] = 0;
        liveEntries++;
        liveEntryBytes += slab.slotBytes;
    }

    /**
     * Whether {@code map}'s index entry for {@code ordinal} is exactly {@code (slabId, slot)}: the index
     * direction, which starts from a slot and asks whether the vector in it is still the live copy.
     *
     * <p>Only {@link #checkInvariantsForTesting} asks any more, and that is the point of it. I2's eviction
     * asked it per slot, which cost one hash probe per slot of the slab being reclaimed and MEASURED at
     * 1.23 ms of write lock for a 10,922-slot slab, nearly all of it random access into the field's index.
     * It was redundant: {@link #attach} clears the previous slot's owner pointer the moment an ordinal is
     * re-cached elsewhere, and nothing else overwrites an index cell, so {@code owner != null} already means
     * "this slot is the live copy". Rather than trust that in a comment, the invariant check re-derives it
     * for every slot of every slab and every test that mutates the cache calls it.
     */
    private boolean pointsBack(final OrdinalSlotMap map, final int ordinal, final int slabId, final int slot) {
        final int cell = map.find(ordinal);
        return cell >= 0 && map.slabIdAt(cell) == slabId && map.slotAt(cell) == slot;
    }

    /**
     * Whether the slot really holds {@code map}'s {@code ordinal}: the slot direction, which is what a
     * lookup and a compaction ask, because both start from an index cell and a cell can name a slot that has
     * since been evicted and refilled by someone else.
     */
    private boolean slotHolds(final OrdinalSlotMap map, final int ordinal, final int slabId, final int slot) {
        final Slab slab = slabAt(slabId);
        return slab != null && slot < slab.slotCount && slab.owner[slot] == map && slab.ordinal[slot] == ordinal;
    }

    private Slab slabAt(final int id) {
        final Slab[] slabs = slabsById;
        return id < slabs.length ? slabs[id] : null;
    }

    /** Opens {@code slab} for {@code length}-byte vectors and makes it the one being appended to. */
    private void open(final Slab slab, final int length) {
        slab.openFor(length);
        openSlabs.put(length, slab);
    }

    private Slab register(final VectorCacheSlab memory) {
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
        return slab;
    }

    private void unregister(final Slab slab) {
        slabsById[slab.id] = null;
        openSlabs.remove(slab.slotBytes, slab);
        liveSlabs--;
        source.release(slab.memory);
    }

    // ---------------------------------------------------------------------------------------------------
    // Counters. I4 publishes these through direct_io_stats.
    // ---------------------------------------------------------------------------------------------------

    /** Lookups that returned a vector. */
    public long hits() {
        return hits.sum();
    }

    /** Lookups that did not. */
    public long misses() {
        return misses.sum();
    }

    /** Vectors written into a slot. */
    public long insertions() {
        return insertions.sum();
    }

    /** Vectors held right now. */
    public long entries() {
        lock.readLock().lock();
        try {
            return liveEntries;
        } finally {
            lock.readLock().unlock();
        }
    }

    /** Slot bytes held by live vectors right now, which is the payload and not the slab total. */
    public long bytes() {
        lock.readLock().lock();
        try {
            return liveEntryBytes;
        } finally {
            lock.readLock().unlock();
        }
    }

    /** Slabs held right now. */
    public int slabCount() {
        lock.readLock().lock();
        try {
            return liveSlabs;
        } finally {
            lock.readLock().unlock();
        }
    }

    /** Bytes of slab held right now, which is what the native budget is charged for. */
    public long slabBytes() {
        return (long) slabCount() * slabBytes;
    }

    /** The cache's byte target, rounded to whole slabs. */
    public long maxBytes() {
        return (long) maxSlabs * slabBytes;
    }

    /** Slabs taken from the head of the FIFO queue to make room. */
    public long reclaimedSlabs() {
        return reclaimedSlabs.sum();
    }

    /** Vectors kept by a second chance when their slab was evicted. */
    public long readmissions() {
        return readmissions.sum();
    }

    /** Nanoseconds spent inside a slab reclaim, which is write-lock time. */
    public long reclaimNanos() {
        return reclaimNanos.sum();
    }

    /** Vectors dropped: evicted without a second chance, or released with their segment. */
    public long droppedEntries() {
        return droppedEntries.sum();
    }

    /** Heap held by the index itself, for the bytes-per-entry figure I2 reports. */
    public long indexBytes() {
        lock.readLock().lock();
        try {
            long total = 48L * index.size();
            for (final OrdinalSlotMap map : index.values()) {
                total += map.indexBytes();
            }
            for (final Slab slab : fifo) {
                if (slab.owner != null) {
                    total += 48 + 4L * slab.owner.length + 16 + 4L * slab.ordinal.length + 16 + slab.hit.length;
                    total += 16 + 4L * slab.free.length;
                }
            }
            return total;
        } finally {
            lock.readLock().unlock();
        }
    }

    /** Test seam: empties the cache, returns every slab, and zeroes every counter. */
    void resetForTesting() {
        configure(0);
        hits.reset();
        misses.reset();
        insertions.reset();
        reclaimedSlabs.reset();
        readmissions.reset();
        droppedEntries.reset();
        reclaimNanos.reset();
    }

    /**
     * Test seam: re-derives the live entry count and byte total from the slabs themselves and checks them
     * against the running counters, so a test can assert that no path has lost or double counted a slot.
     * Every slot that still carries a back pointer must also be the slot the index names for it.
     *
     * @throws IllegalStateException on any disagreement, naming it
     */
    void checkInvariantsForTesting() {
        lock.writeLock().lock();
        try {
            long entries = 0;
            long entryBytes = 0;
            for (final Slab slab : fifo) {
                if (slab.used > slab.slotCount) {
                    throw new IllegalStateException("slab " + slab.id + " used " + slab.used + " of " + slab.slotCount);
                }
                for (int slot = 0; slot < slab.used; slot++) {
                    final OrdinalSlotMap owner = slab.owner[slot];
                    if (owner == null) {
                        continue;
                    }
                    if (pointsBack(owner, slab.ordinal[slot], slab.id, slot) == false) {
                        throw new IllegalStateException("slab " + slab.id + " slot " + slot + " is not named by its index entry");
                    }
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
        } finally {
            lock.writeLock().unlock();
        }
    }

    /** Test seam: the number of slots {@code length}-byte vectors get in one slab. */
    int slotsPerSlab(final int length) {
        return Math.min(slabBytes / length, OrdinalSlotMap.MAX_SLOTS);
    }
}

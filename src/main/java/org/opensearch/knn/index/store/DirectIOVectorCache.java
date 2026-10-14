/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import lombok.extern.log4j.Log4j2;
import org.opensearch.core.common.unit.ByteSizeValue;
import org.opensearch.knn.index.KNNSettings;

import java.nio.ByteBuffer;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.LongAdder;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * The one cache of individual vectors on the Direct I/O random rescore path, for the whole node: a slab store
 * below the {@link DirectIOVectorIndexInput}, keyed by the file and the byte offset the reader asks for.
 *
 * <h2>Why it exists, in one measurement</h2>
 * Every phase 12-14 arm, and the store-factory arm it was compared against, ran with <b>no vector cache at
 * all</b>. An LRU one existed in phase 7 and was deleted in phase 12 as unreachable code after phase 10
 * removed the channel that reached it — on reachability, never on performance evidence. So the question
 * "would a cache help" had never been asked of the design that ships.
 *
 * <p>It was asked, at scale, before a line of this was written. MEASURED 2026-10-05 on {@code dio-10m}:
 * an exact LRU stack-distance replay at vector granularity of 5,000 real OSB query vectors against the
 * exact rescore candidate set (k=200 {@code rescore:false}, which is what a k=100 rescore asks for at 32x
 * compression and 768 dims), 1,000,000 references over 544,060 unique vectors — a 1.70 GB working set:
 *
 * <table>
 *   <caption>steady-state hit rate by budget; ceiling 45.6%</caption>
 *   <tr><th>budget</th><th>entries</th><th>hit rate</th></tr>
 *   <tr><td><b>16 MB</b></td><td>5,127</td><td><b>40.3%</b></td></tr>
 *   <tr><td>64 MB</td><td>20,510</td><td>40.9%</td></tr>
 *   <tr><td>128 MB</td><td>41,020</td><td>41.4%</td></tr>
 *   <tr><td>256 MB</td><td>82,040</td><td>42.3%</td></tr>
 *   <tr><td>1024 MB</td><td>328,160</td><td>48.1%</td></tr>
 * </table>
 *
 * 16 MB captures 84% of the achievable reuse, and replaying the same 5,000 queries three times does not
 * inflate the figure (128 MB: 39.8% on every pass), so it is inner-product HNSW hub reuse rather than an
 * artefact of the query set. On the EBS node a slab store of this shape then MEASURED 6-9 more hit points
 * than the phase 15 Guava cache at equal bytes (16-S: 61.3 against 54.9% at 4 GB), with device reads per
 * query falling in step with the misses, which is what moved latency on an IOPS-capped volume.
 *
 * <h2>Vector granularity comes for free</h2>
 * {@code Lucene99FlatVectorsReader}'s values object asks this input for exactly one vector's range — a
 * {@code prefetch(offset, vectorBytes)} and then a {@code readFloats} of the same run — so a key of
 * {@code (file, offset)} <em>is</em> one vector. Nothing has to be grouped, aligned or split, and the value
 * is the vector's bytes rather than the 8 KiB block they sit in: caching the block would cache 2.7 vectors'
 * worth of neighbours that the measured access pattern does not ask for, and would charge the budget 2.7x
 * for the 40% it buys.
 *
 * <h2>File identity: path and length, not the handle</h2>
 * The key's file part is {@code <absolute path>#<file length>} ({@link #fileId}), built once per
 * {@code .vec} open, and <b>not</b> the {@code Handle}'s object identity. Identity would be wrong in the
 * direction that matters: every {@code DirectIOVectorIndexInput#open} creates its own {@code Handle}, and
 * the codec seam opens a {@code .vec} once per reader, so two readers of the same file would never share an
 * entry — which is exactly the reuse the gate above measured and this cache exists to collect.
 *
 * <p>Path and length cannot collide across different content either, and the reason is structural rather
 * than hopeful. A shard's files live under {@code .../indices/<index-uuid>/<shard>/index/}, and Lucene's
 * segment names come from a counter that only increases within an index, so a name is never reused for
 * different bytes within one index and a different index has a different UUID in its path. The file length
 * is a second discriminator on top of that. A Lucene segment file is also immutable once written, so the
 * bytes at a given offset of a given file never change.
 *
 * <h2>Slabs, slots and the index (M1, M2)</h2>
 * <ul>
 *   <li>A <b>slab</b> is {@link DirectIOSlab#SLAB_BYTES} of off-heap memory, allocated through
 *       {@code java.lang.foreign} by reflection where that works and {@code ByteBuffer.allocateDirect}
 *       otherwise ({@link DirectIOSlab}). It is opened for <b>one vector byte length</b> and cut into slots
 *       of exactly that length, so placing a vector is a bump pointer and finding it is a multiply: 10,922
 *       slots at 768 dimensions, 128 slabs at a 4 GB cache, at most {@link DirectIOSlotMap#MAX_SLABS}.</li>
 *   <li>Per slot, in plain arrays per slab: the owning file index, the byte offset (a back pointer) and a
 *       one-byte hit flag, plus a free list per slab.</li>
 *   <li>The <b>index</b> is {@code HashMap<fileId, DirectIOSlotMap>}, where {@link DirectIOSlotMap} is a
 *       primitive open-addressed table from offset to {@code (slab, slot)} with no per-entry object.</li>
 * </ul>
 *
 * <h2>Eviction is FIFO over slabs, with one second chance</h2>
 * Evicting one vector at a time would need a recency structure per vector, which is the per-entry cost this
 * design exists to avoid. Instead the slabs form a FIFO queue and the oldest slab is the eviction unit. A
 * vector read since it was written carries a hit flag, and when its slab comes up for eviction it is
 * re-admitted rather than dropped: the survivors <b>stay exactly where they are</b>, the slots of the vectors
 * that were never read become free slots in that same slab, and the slab is reopened for the same vector
 * length and appended to through its free list. No byte moves and no index entry changes, which is what keeps
 * the write-lock hold off the millisecond scale (16-I MEASURED 26.5 us p50 per reclaim this way, against
 * 1.7-4.8 ms when survivors were compacted to the front).
 *
 * <p>Re-admission is capped at half a slab's slots, which is what makes an eviction always yield room and the
 * algorithm terminate in one step: a fully hot slab cannot re-admit itself whole and so cannot deadlock the
 * cache against its own hit flags. The cap is the same device S3-FIFO and Segcache use for the same reason.
 * There is no admission filter: scan resistance is undecided (agenda item 4).
 *
 * <p>A slab reopened for a <em>different</em> vector length is the one case with no second chance: its slot
 * layout changes, so there is nowhere for a survivor to stay, and those entries are dropped. The case needs
 * two vector lengths competing for the same slabs, which is rare, and costs a refill rather than an answer.
 *
 * <h2>One lock, and no view of slab memory leaves the store</h2>
 * A reader must not be shown a slot while another thread is overwriting it, and the index must not be read
 * half-rebuilt. One {@link ReentrantReadWriteLock} over the whole store solves both: a lookup takes the read
 * lock, and insert, eviction, resize and release take the write lock. Lookups therefore run concurrently with
 * each other and never with a mutation, which is what lets the index be a plain {@link HashMap} of
 * {@link DirectIOSlotMap}s with no per-entry synchronisation. Inserts serialise against each other, which is
 * affordable because an insert only ever follows a read miss, so its rate is bounded by the device.
 *
 * <p>A lookup copies the vector into the caller's {@code float[]} while it holds the read lock and hands back
 * no view of slab memory at all. That is what makes a use after free unreachable rather than merely unlikely:
 * there is no buffer for a caller to stash and read after the lock is dropped. It is also why a slab needs no
 * lock of its own — it is closed only under this store's write lock.
 *
 * <h2>A size change resizes; it does not rebuild</h2>
 * {@code knn.direct_io.rescore.cache.max_bytes} is read once per prefetch burst
 * ({@link #applyConfiguredSize()}). Growing lets the next miss allocate a slab and keeps every cached vector;
 * shrinking releases the oldest slabs and keeps every vector that still fits; {@code 0b} releases every slab.
 * Phase 15's Guava cache could not do that — {@code maximumWeight} is fixed at {@code CacheBuilder#build()},
 * so a change dropped the whole cache and warmed it again.
 *
 * <h2>Entries of a file that has been closed: they age out, and the index cell goes with the last one</h2>
 * Nothing invalidates a merged-away segment's entries at close time, and they are <b>never wrong</b> — the
 * key says which file's bytes they are and that file's bytes are immutable. They are only <em>cold</em>, so
 * the FIFO reaches them without a second chance (nothing reads them, so no hit flag is ever set) and they
 * become free slots. What is new against phase 15 is the second half: the index is per file, so when the last
 * slot belonging to a file is freed, that file's whole {@link DirectIOSlotMap} is dropped from the index
 * ({@link #detach}). A dead file therefore costs bounded eviction pressure and then nothing, with no per-file
 * reference counting of open handles and no work on the close path.
 *
 * <h2>Off is off</h2>
 * {@code knn.direct_io.rescore.cache.max_bytes} defaults to {@code 0b}. When it is zero there are no slabs
 * and every lookup is one volatile read and a compare — no key built, no lock taken, no counter touched.
 * The shipped default is the node arm's to decide; this phase ships the mechanism and the correctness proof.
 */
@Log4j2
public final class DirectIOVectorCache {

    /** Set once per node, and only by a test, so a suite can exercise eviction without 32 MiB of slabs. */
    private static volatile int slabBytes = DirectIOSlab.SLAB_BYTES;

    private static final ReentrantReadWriteLock LOCK = new ReentrantReadWriteLock();

    /** File index. Mutated under the write lock, read under the read lock, so a plain map is enough. */
    private static final Map<String, DirectIOSlotMap> INDEX = new HashMap<>();

    /** Slabs in service, oldest first. The head is the eviction unit. */
    private static final ArrayDeque<Slab> FIFO = new ArrayDeque<>();

    /** The one slab currently being appended to, per vector byte length. */
    private static final Map<Integer, Slab> OPEN_SLABS = new HashMap<>();

    /** Slabs by id, so an index entry can be one int. Null where an id is not in use. */
    private static Slab[] slabsById = new Slab[0];

    /** Slabs the store may hold, from {@link #configure(long)}. Zero means off, and is the default. */
    private static volatile int maxSlabs;

    /**
     * Whether span fill is on, refreshed once per prefetch burst by {@link #applyConfiguredSize()}. Starts at
     * the setting's default so that a store filled before the first refresh behaves as the setting says.
     */
    private static volatile boolean spanFill = KNNSettings.KNN_DIRECT_IO_RESCORE_CACHE_SPAN_FILL_DEFAULT_VALUE;

    private static int liveSlabs;
    private static long liveEntries;
    private static long liveEntryBytes;

    /**
     * Set when an allocation failed, so the store stops asking for a slab it cannot have (N2). Cleared when
     * the target changes or a slab is released, which are the two events that can make the answer different.
     *
     * <p>MEASURED by the orchestrator (2026-10-12, desktop, {@code -Xmx256m} with
     * {@code -XX:MaxDirectMemorySize=96m}): a failed reservation blocks the caller for 515 to 525 ms and runs
     * one {@code Pause Full (System.gc())}, for {@code allocateDirect} on JDK 21 and 25 and for FFM on JDK 21,
     * because {@code java.nio.Bits#reserveMemory} calls {@code System.gc()} and then sleeps nine times with
     * backoff. Retrying it on every fill below the target costs that once per fill, under this class's write
     * lock, and the answer is the same every time.
     *
     * <p>Only read and written under the write lock, like {@link #liveSlabs}.
     */
    private static boolean growthLatched;

    /**
     * Slabs other fills have allocated and not yet registered (N3). Counted so that fills growing the store
     * concurrently cannot overshoot the target between them: each reserves its headroom under the write lock
     * before leaving it to allocate.
     *
     * <p>Only read and written under the write lock.
     */
    private static int pendingSlabs;

    /**
     * Where a slab's memory comes from and goes back to. A seam, and the only reason it is one is that N3's
     * property is about <b>where</b> these two calls run: neither may run while the caller holds this store's
     * lock, and a test can only see that if it can observe the calling thread at the moment of the call.
     * Production has exactly one implementation.
     */
    interface SlabMemory {
        /**
         * A slab of {@code bytes} bytes, or null when the backend refused.
         *
         * @param bytes the slab size
         * @return the slab, or null
         */
        DirectIOSlab allocate(int bytes);

        /**
         * Gives a slab's memory back.
         *
         * @param slab the slab to release
         */
        void release(DirectIOSlab slab);
    }

    private static final SlabMemory DIRECT_SLABS = new SlabMemory() {
        @Override
        public DirectIOSlab allocate(final int bytes) {
            return DirectIOSlab.allocate(bytes);
        }

        @Override
        public void release(final DirectIOSlab slab) {
            slab.close();
        }
    };

    private static volatile SlabMemory slabMemory = DIRECT_SLABS;

    /** Release threads still giving memory back, so a test can wait for it. */
    private static final AtomicInteger releasesRunning = new AtomicInteger();

    private static final LongAdder hits = new LongAdder();
    private static final LongAdder misses = new LongAdder();
    private static final LongAdder insertions = new LongAdder();
    private static final LongAdder reclaimedSlabs = new LongAdder();
    private static final LongAdder readmissions = new LongAdder();
    private static final LongAdder droppedEntries = new LongAdder();
    private static final LongAdder spanFills = new LongAdder();
    /** Allocations that failed, which is one per latch and so also how often growth was given up on. */
    private static final LongAdder slabAllocFailures = new LongAdder();
    /** Write-lock time spent reclaiming slabs, which is the figure a node has to be able to read. */
    private static final LongAdder reclaimNanos = new LongAdder();

    private DirectIOVectorCache() {}

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

    /** The identity {@link DirectIOSlotMap} uses for a file: two opens of one {@code .vec} give one string. */
    static String fileId(final java.nio.file.Path path, final long fileLength) {
        return path.toString() + "#" + fileLength;
    }

    /** Whether a lookup can succeed at all, so a caller can skip the work of asking. */
    static boolean isOn() {
        return maxSlabs != 0;
    }

    // ---------------------------------------------------------------------------------------------------
    // The read path
    // ---------------------------------------------------------------------------------------------------

    /**
     * Whether <b>every</b> vector in {@code [offset, offset + length)} of {@code fileId} is cached, walking
     * the run by the lengths the slots themselves record, and <b>without counting a hit or a miss</b>.
     *
     * <p>This is the input's first lookup, in {@code prefetch}, where the answer decides whether the range
     * joins the burst — so a yes here is what saves the device read, but it is not what serves the vectors.
     * Counting it would double every count against the vectors actually served, which is what phase 15 did
     * and had to document ({@link #lookupsPerVector()}); one count per vector at the lookup that serves it
     * makes {@code hits / (hits + misses)} a hit rate directly comparable with the 16-S node figures.
     *
     * <h4>Why it is a run and not one vector (MEASURED)</h4>
     * The range {@code prefetch} is given is not a vector, it is a <em>coalesced group</em>:
     * {@code PrefetchHelper#prefetchExactVectorSize} sorts the batch's ordinals and asks for the span of each
     * group it can fit in 128 KB (CODE: {@code PrefetchHelper.java:97-108}), so both ends are vector
     * boundaries on one grid but the span can hold dozens of them. A one-vector lookup therefore answers
     * "no" for every multi-vector group, however much of it is cached, and the group is staged in full.
     *
     * <p>MEASURED on M4's fixture (output/bench/m4), 4,000 batches of 64 random candidates over a 64 MB
     * {@code .vec}, span fill on, with the one-vector form: a 97.1% hit rate and <b>1.87 GB of device reads,
     * 27.9x the whole file</b>, because 10.7 groups per batch averaging 45 KB were staged even once the store
     * held every vector in the file. Walking the run takes the same arm to <b>81 MB, 1.21x the file</b>, and
     * 16-I's scorer-level cache measured 1.53x. The cost is up to {@code max_span_bytes / vectorBytes} index
     * probes (43 at the defaults) under one read-lock acquisition, against the 45 KB device read it avoids.
     *
     * <p>A run of one vector is the single lookup this method used to be, so the key's meaning is unchanged
     * and a caller asking about one vector asks exactly as before.
     *
     * @param fileId the file's identity, from {@link #fileId}
     * @param offset the first vector's byte offset in that file
     * @param length the run's length in bytes, one or more whole vectors
     * @return whether a {@link #getFloats} would hit for every vector in the run
     */
    static boolean contains(final String fileId, final long offset, final int length) {
        if (maxSlabs == 0 || length <= 0) {
            return false;
        }
        LOCK.readLock().lock();
        try {
            final DirectIOSlotMap map = INDEX.get(fileId);
            if (map == null) {
                return false;
            }
            final long end = offset + length;
            long at = offset;
            while (at < end) {
                final int cell = map.find(at);
                if (cell < 0) {
                    return false;
                }
                final Slab slab = slabAt(map.slabIdAt(cell));
                final int slot = map.slotAt(cell);
                if (slab == null || slot >= slab.slotCount || slab.owner[slot] != map || slab.offset[slot] != at) {
                    return false;
                }
                // The slot's own length is the stride: the caller does not know the field's vector length at
                // this point and must not have to guess it.
                at += slab.slotBytes;
            }
            // Exactly tiled, or not cached: a run whose last vector overshoots the range is a different
            // layout than the one being asked about, and the honest answer for one is no.
            return at == end;
        } finally {
            LOCK.readLock().unlock();
        }
    }

    /**
     * Writes into {@code missing} the members of {@code ords[0..numOrds)} whose vector this store does not
     * hold whole and returns how many, or returns {@code -1} when it holds nothing of {@code fileId} at all.
     *
     * <p>One read-lock acquisition for the batch, and like {@link #contains} it <b>counts nothing</b>: the
     * caller ({@code DirectIOVectorIndexInput#selectUncachedOrdinals}, N1) is deciding what to prefetch, not
     * serving bytes. {@code -1} rather than a copy of every ordinal is what lets a cold file take the
     * caller's unfiltered path with no array to walk.
     *
     * <p>A vector held at a different length than {@code vectorBytes} counts as missing, for the same reason
     * {@link #getFloats} answers a length mismatch with a miss: a caller asking about a layout this store did
     * not cache is a caller it was not designed for.
     *
     * @param fileId      the file's identity, from {@link #fileId}
     * @param baseOffset  the byte offset of ordinal zero in that file
     * @param vectorBytes one vector's length in bytes, which is also the ordinal stride
     * @param ords        the ordinals to ask about, read and not written
     * @param numOrds     how many of {@code ords} are valid
     * @param missing     at least {@code numOrds} ints
     * @return how many ordinals were written to {@code missing}, or {@code -1}
     */
    static int selectMissing(
        final String fileId,
        final long baseOffset,
        final int vectorBytes,
        final int[] ords,
        final int numOrds,
        final int[] missing
    ) {
        if (maxSlabs == 0) {
            return -1;
        }
        int found = 0;
        LOCK.readLock().lock();
        try {
            final DirectIOSlotMap map = INDEX.get(fileId);
            if (map == null) {
                return -1;
            }
            for (int i = 0; i < numOrds; i++) {
                if (locate(map, baseOffset + (long) ords[i] * vectorBytes, vectorBytes) < 0) {
                    missing[found++] = ords[i];
                }
            }
        } finally {
            LOCK.readLock().unlock();
        }
        return found;
    }

    /**
     * Copies the cached vector into {@code destination} and returns true, or returns false and leaves
     * {@code destination} alone. A hit also flags the slot, which is what earns it a second chance when its
     * slab is reclaimed.
     *
     * <p>{@code length} is checked against the slot rather than being part of the key, because the reader
     * always asks for one whole vector and so always asks for the same length at a given offset; a mismatch
     * therefore means a caller this cache was not designed for, and the honest answer for one is a miss.
     *
     * @param fileId the file's identity, from {@link #fileId}
     * @param offset the vector's byte offset in that file
     * @param length the vector's length in bytes
     * @param destination at least {@code destinationOffset + length / 4} floats
     * @param destinationOffset where in {@code destination} to write the first float
     * @return whether the vector was cached
     */
    static boolean getFloats(
        final String fileId,
        final long offset,
        final int length,
        final float[] destination,
        final int destinationOffset
    ) {
        if (maxSlabs == 0) {
            return false;
        }
        boolean hit = false;
        LOCK.readLock().lock();
        try {
            final DirectIOSlotMap map = INDEX.get(fileId);
            final int packedSlot = locate(map, offset, length);
            if (packedSlot >= 0) {
                final Slab slab = slabsById[packedSlot >>> SLOT_SHIFT];
                final int slot = packedSlot & SLOT_LOW_MASK;
                slab.memory.floats().get((slot * slab.slotBytes) >> 2, destination, destinationOffset, length >> 2);
                // A plain write, and a lost one only costs a second chance. Readers run concurrently with
                // each other but never with the reclaim that reads this, which holds the write lock.
                slab.hit[slot] = 1;
                hit = true;
            }
        } finally {
            LOCK.readLock().unlock();
        }
        if (hit) {
            hits.increment();
        } else {
            misses.increment();
        }
        return hit;
    }

    /** How {@link #locate} packs its answer, which is the same shape {@link DirectIOSlotMap} stores. */
    private static final int SLOT_SHIFT = 20;
    private static final int SLOT_LOW_MASK = (1 << SLOT_SHIFT) - 1;

    /**
     * The {@code (slabId << 20) | slot} holding {@code offset} at {@code length} bytes, or {@code -1}.
     * Called under either lock.
     *
     * <p>Two checks past the index cell, and both are load bearing. The slot's back pointer is the proof that
     * the cell is not stale: a cell can outlive the slot it names, because eviction does not walk the index
     * (see {@link DirectIOSlotMap}). The slot length check is what keeps a caller asking for a different
     * length than was cached from being handed a neighbour's bytes.
     */
    private static int locate(final DirectIOSlotMap map, final long offset, final int length) {
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

    /**
     * Caches {@code length} bytes at {@code position} of {@code source} as the vector at
     * {@code (fileId, offset)}. A copy and not a view, because {@code source} is a staging buffer that the
     * next burst reuses.
     *
     * <p>Silently does nothing when the cache is off, when the length is not a whole number of floats or is
     * larger than a slab, or when no slab can be had. A fill that does not happen costs a future hit and
     * nothing else.
     *
     * @param fileId the file's identity, from {@link #fileId}
     * @param offset the vector's byte offset in that file
     * @param source the staged bytes
     * @param position where in {@code source} the vector starts
     * @param length the vector's length in bytes
     */
    static void put(final String fileId, final long offset, final ByteBuffer source, final int position, final int length) {
        if (maxSlabs == 0 || length <= 0 || (length & 3) != 0 || offset < 0 || length > slabBytes) {
            return;
        }
        final SlabRequest request = new SlabRequest();
        try {
            // At most twice round: once with nothing in hand, and once with the slab that round asked for.
            // See SlabRequest for why the allocation cannot happen inside the lock.
            for (int round = 0; round < 2; round++) {
                LOCK.writeLock().lock();
                try {
                    request.wanted = false;
                    final Placement placement = place(fileId, length, true, request);
                    if (placement != null) {
                        placement.slab.memory.bytes().put(placement.slot * length, source, position, length);
                        attach(placement.slab, placement.slot, placement.map, offset);
                        insertions.increment();
                        return;
                    }
                } finally {
                    LOCK.writeLock().unlock();
                }
                if (request.wanted == false || allocateOutsideLock(request) == false) {
                    return;
                }
            }
        } finally {
            request.finish();
        }
    }

    /**
     * Whether the store would take another slab if a fill asked for one, i.e. whether it is still below its
     * slab target. This is the condition span fill admits under; see {@link #putRun}.
     *
     * <p>A volatile read of each of two ints and no lock, because the caller is the read path asking once per
     * staged span whether to bother, and the answer is advisory: {@link #putRun} re-checks it under the write
     * lock, which is where it has to hold.
     *
     * @return whether a fill may still allocate a slab
     */
    static boolean belowSlabTarget() {
        return maxSlabs != 0 && liveSlabs < maxSlabs;
    }

    /**
     * Admits the {@code count} consecutive vectors starting at {@code (fileId, firstOffset)} from
     * {@code source}, under <b>one</b> write-lock acquisition, and only while the store is below its slab
     * target. M4's span fill, ported from 16-I F5.
     *
     * <p><b>Three rules, and each is a thing this must not do rather than a thing it does.</b>
     * <ul>
     *   <li><b>It never evicts, and never takes a slab the target does not allow.</b> It places with eviction
     *       refused, so the moment the store is at its slab target this admits nothing and the admission
     *       policy is back to the requested vectors alone. That is what keeps a span fill from being a scan
     *       that pushes out the working set, which is the open agenda item (item 4) this phase is not
     *       deciding.</li>
     *   <li><b>It skips an offset that is already cached.</b> The requested fill has just put this read's
     *       vector in, and the span it was staged for contains it; admitting it again would spend a slot on a
     *       vector the store already holds.</li>
     *   <li><b>It stops at the first refusal.</b> Once {@link #place} has nowhere to put a vector it has
     *       nowhere to put the next one either, so walking the rest of the run would be an index scan under
     *       the write lock for nothing.</li>
     * </ul>
     *
     * <p>One acquisition per run rather than per vector is what bounds the hold to one span's worth of
     * copying: a run is the whole vectors of one staged span, about two at the measured mean aligned span of
     * 7,310 B for a 3,072 B vector and at most {@code max_span_bytes / length} (43 at the defaults).
     *
     * @param fileId the file's identity, from {@link #fileId}
     * @param firstOffset the byte offset of the first vector in the run
     * @param count how many consecutive vectors the run holds
     * @param source the staged bytes; copied out, never retained
     * @param position where in {@code source} the first vector starts
     * @param length one vector's length in bytes
     * @return how many vectors were admitted
     */
    static int putRun(
        final String fileId,
        final long firstOffset,
        final int count,
        final ByteBuffer source,
        final int position,
        final int length
    ) {
        if (maxSlabs == 0 || count <= 0 || length <= 0 || (length & 3) != 0 || firstOffset < 0 || length > slabBytes) {
            return 0;
        }
        int admitted = 0;
        int next = 0;
        final SlabRequest request = new SlabRequest();
        try {
            while (next < count) {
                LOCK.writeLock().lock();
                try {
                    if (belowSlabTarget() == false) {
                        break;
                    }
                    for (; next < count; next++) {
                        final long offset = firstOffset + (long) next * length;
                        if (locate(INDEX.get(fileId), offset, length) >= 0) {
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
                        insertions.increment();
                        admitted++;
                    }
                } finally {
                    LOCK.writeLock().unlock();
                }
                // One acquisition per span is the rule, and this is the one thing that can need a second:
                // a span that fills the open slab. The staged bytes are still the caller's and still valid
                // across the gap, and the alternative is an 8 ms allocation inside the hold (N3).
                if (next >= count || request.wanted == false || allocateOutsideLock(request) == false) {
                    break;
                }
            }
        } finally {
            request.finish();
        }
        spanFills.add(admitted);
        return admitted;
    }

    /**
     * One fill's slab request, carried across a release of the write lock.
     *
     * <p><b>Why the lock has to be released at all.</b> A 32 MiB slab takes about 7 to 10 ms to allocate
     * (MEASURED, M1 and output/bench/n2: FFM zero-fills it; 16-I's JNI took 7.6 us), and before N3
     * {@code growSlabs} did that with the write lock held, so every lookup on the node waited behind it.
     * A fill that needs a slab therefore reserves its headroom under the lock ({@link #pendingSlabs}), leaves
     * the lock, allocates, and comes back with the slab in hand.
     *
     * <p><b>Why the reservation.</b> Without it two fills leaving the lock at the same time would both see
     * room for one slab and both allocate one, so a 2 slab target could end up holding 3. The reservation is
     * taken under the lock and counted against the target by every other fill.
     *
     * <p>An allocation nobody wanted any more, because the target shrank while this fill was outside the lock,
     * is released by {@link #finish} and also outside the lock.
     */
    private static final class SlabRequest {
        /** Allocated outside the lock and not yet registered. */
        private DirectIOSlab spare;
        /** Set by {@link #growSlabs} to mean "bring me a slab and ask again". */
        private boolean wanted;
        /** Whether this request holds one of {@link #pendingSlabs}. */
        private boolean reserved;

        /** Gives back the headroom and any slab that went unused. Called with no lock held. */
        private void finish() {
            if (reserved == false && spare == null) {
                return;
            }
            final DirectIOSlab unused;
            LOCK.writeLock().lock();
            try {
                if (reserved) {
                    pendingSlabs--;
                    reserved = false;
                }
                unused = spare;
                spare = null;
            } finally {
                LOCK.writeLock().unlock();
            }
            if (unused != null) {
                slabMemory.release(unused);
            }
        }
    }

    /**
     * Allocates the slab {@code request} asked for, with no lock held, and says whether to ask the store
     * again. A refusal latches growth (N2) so that no later fill pays for the same answer.
     */
    private static boolean allocateOutsideLock(final SlabRequest request) {
        request.spare = slabMemory.allocate(slabBytes);
        if (request.spare != null) {
            return true;
        }
        LOCK.writeLock().lock();
        try {
            if (growthLatched == false) {
                growthLatched = true;
                slabAllocFailures.increment();
                log.warn(
                    "A k-NN Direct I/O vector cache slab of {} bytes could not be allocated from {}; the cache will hold the {} "
                        + "slabs it has until knn.direct_io.rescore.cache.max_bytes changes or a slab is released",
                    slabBytes,
                    DirectIOSlab.backend(),
                    liveSlabs
                );
            }
        } finally {
            LOCK.writeLock().unlock();
        }
        return false;
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
     * {@link SlabRequest#wanted} set, and the caller allocates outside the lock and asks again.
     */
    private static Placement place(final String fileId, final int length, final boolean allowEviction, final SlabRequest request) {
        if (maxSlabs == 0) {
            return null;
        }
        DirectIOSlotMap map = INDEX.get(fileId);
        if (map == null) {
            map = new DirectIOSlotMap(fileId);
            INDEX.put(fileId, map);
        }
        if (map.needsCompaction()) {
            final DirectIOSlotMap compacting = map;
            map.compact((offset, slabId, slot) -> {
                final Slab slab = slabAt(slabId);
                return slab != null && slot < slab.slotCount && slab.owner[slot] == compacting && slab.offset[slot] == offset;
            });
        }
        Slab slab = OPEN_SLABS.get(length);
        if (slab == null || slab.freeSlots() == 0) {
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

    // ---------------------------------------------------------------------------------------------------
    // Sizing
    // ---------------------------------------------------------------------------------------------------

    /**
     * Reads {@code knn.direct_io.rescore.cache.max_bytes} and resizes the store if it has changed.
     *
     * <p>Called once per prefetch burst, from {@code DirectIOVectorIndexInput#prefetch} at the burst
     * boundary — the same cadence and the same contract as {@code DirectIOReadPool#executor()}, so an
     * operator's change takes effect on the next burst with no restart and there is no startup gap for a
     * value that was already persisted. A setting that cannot be read leaves the store exactly as it is.
     *
     * @param routeActive whether the calling input is reading through {@code O_DIRECT} right now. {@code false}
     *                    means {@code knn.direct_io.rescore.enabled} has been turned off under an open routed
     *                    input (O1), and the store is released rather than resized.
     */
    static void applyConfiguredSize(final boolean routeActive) {
        try {
            spanFill = KNNSettings.isDirectIORescoreCacheSpanFillEnabled();
        } catch (RuntimeException e) {
            log.debug("Could not read knn.direct_io.rescore.cache.span_fill; leaving span fill as it is", e);
        }
        if (routeActive == false) {
            // The caller is reading through its delegate input because knn.direct_io.rescore.enabled is off
            // (O1), so nothing will be admitted and nothing will be looked up: slabs held here serve
            // nothing. Releasing them is what makes the rollback give the memory back as well as the device
            // traffic, which is the same choice 16-I's NativeMemoryCacheManager#updateVectorCacheTarget
            // made. Going back on re-reads max_bytes below and the store grows again on its next fill.
            //
            // It is the caller's route and not a second read of the setting, because an input with no
            // delegate to fall back to stays on O_DIRECT whatever the setting says (see
            // DirectIOVectorIndexInput#mmapFallback) and its store must stay with it. That is every probe
            // and bench driver in this package, which open a path directly.
            configure(0L);
            return;
        }
        final long wanted;
        try {
            final ByteSizeValue configured = KNNSettings.getDirectIORescoreCacheMaxBytes();
            wanted = configured == null ? 0L : configured.getBytes();
        } catch (RuntimeException e) {
            log.debug("Could not read knn.direct_io.rescore.cache.max_bytes; leaving the vector cache as it is", e);
            return;
        }
        configure(wanted);
    }

    /**
     * Whether span fill is on, from {@code knn.direct_io.rescore.cache.span_fill} as of the last burst
     * boundary (M4).
     *
     * <p>A volatile read of a field refreshed by {@link #applyConfiguredSize()}, not a settings read, because
     * the caller asks once per staged range consumed and a settings read walks the cluster state. The burst
     * boundary is the right cadence for the same reason it is for the size: a span fill can only follow a
     * staged read, and a staged read can only follow a prefetch.
     *
     * @return whether a staged span may admit the whole vectors inside it
     */
    static boolean isSpanFillOn() {
        return spanFill && maxSlabs != 0;
    }

    /**
     * Sets the store's byte target, adding or releasing slabs to match. Zero empties it and releases every
     * slab.
     *
     * <p>The target is rounded down to whole slabs, because a slab is the unit the store allocates and
     * reclaims in. A target under one slab therefore holds nothing, which is logged rather than silently
     * rounded up past the operator's number.
     *
     * <p><b>This resizes; it does not rebuild.</b> Growing keeps every cached vector and lets the next miss
     * allocate; shrinking drops the oldest slabs and keeps every vector that still fits. That is the
     * difference from phase 15, whose Guava cache had to be rebuilt because {@code maximumWeight} is fixed at
     * {@code build()}, so every change cost the whole cache and a re-warm.
     *
     * @param maxBytes the wanted ceiling in bytes; {@code 0} or less turns the cache off
     */
    static void configure(final long maxBytes) {
        final int wanted = (int) Math.min(DirectIOSlotMap.MAX_SLABS, Math.max(0, maxBytes) / slabBytes);
        if (wanted == maxSlabs) {
            // Before the lock and not after it, because this is called once per prefetch burst from the read
            // path to pick up a setting change, and in the steady state the answer is always no change.
            // Taking the write lock to find that out would put every burst through the lock a fill uses.
            return;
        }
        final List<DirectIOSlab> detached = new ArrayList<>();
        LOCK.writeLock().lock();
        try {
            if (wanted != maxSlabs) {
                if (maxBytes > 0 && wanted == 0) {
                    log.warn(
                        "A vector cache target of {} bytes is under one {} byte slab, so the cache will hold nothing",
                        maxBytes,
                        slabBytes
                    );
                }
                maxSlabs = wanted;
                // A new target is a new question, so an earlier allocation failure stops standing (N2).
                growthLatched = false;
                while (liveSlabs > wanted) {
                    final Slab oldest = FIFO.poll();
                    if (oldest == null) {
                        break;
                    }
                    dropAll(oldest);
                    detached.add(detach(oldest));
                }
                if (wanted == 0) {
                    INDEX.clear();
                    OPEN_SLABS.clear();
                    log.info("The k-NN Direct I/O vector cache is off (knn.direct_io.rescore.cache.max_bytes=0)");
                } else {
                    log.info("The k-NN Direct I/O vector cache holds up to {} slabs of {} bytes", wanted, slabBytes);
                }
            }
        } finally {
            LOCK.writeLock().unlock();
        }
        releaseDetached(detached);
    }

    /**
     * Gives back the memory of slabs a release has detached, with no lock held and not on the calling thread.
     *
     * <p><b>Why not under the lock.</b> An FFM arena takes about 2.2 ms to close (MEASURED, M1), so before N3
     * a {@code configure(0)} at a 4 GB target held the write lock for about 0.26 s and a 12 GB one for about
     * 0.8 s, and every lookup on the node waited behind it. Detaching is what makes the close safe to do
     * outside the lock: it clears the slab's {@code slabsById} entry under the write lock, so its index cells
     * go stale and no reader that arrives afterwards can reach it, while a reader already inside holds the
     * read lock the detaching thread had to wait for.
     *
     * <p><b>Why not on the calling thread.</b> Every production call to {@link #configure} arrives from
     * {@code DirectIOVectorIndexInput#beginBurst}, which is to say on a thread serving a query, so an
     * operator turning the cache off would spend that query's latency closing arenas. A thread per release
     * rather than a pool, because a release happens when an operator changes a setting: there is no steady
     * state to amortise a pool over, and a thread that does not exist cannot be left running.
     */
    private static void releaseDetached(final List<DirectIOSlab> detached) {
        if (detached.isEmpty()) {
            return;
        }
        final SlabMemory backend = slabMemory;
        releasesRunning.incrementAndGet();
        final Thread releaser = new Thread(() -> {
            try {
                for (final DirectIOSlab memory : detached) {
                    backend.release(memory);
                }
            } finally {
                releasesRunning.decrementAndGet();
            }
        }, "knn-dio-slab-release");
        releaser.setDaemon(true);
        releaser.start();
    }

    // ---------------------------------------------------------------------------------------------------
    // Slab supply and eviction
    // ---------------------------------------------------------------------------------------------------

    /**
     * A slab with room for a {@code length}-byte vector, by allocating one while the store is under its
     * target and otherwise by reclaiming the oldest. Returns {@code null} only when the allocation failed and
     * there was nothing to reclaim. Called under the write lock.
     */
    private static Slab obtainSlab(final int length, final SlabRequest request) {
        final Slab grown = growSlabs(length, request);
        if (grown != null) {
            return grown;
        }
        if (request != null && request.wanted) {
            // The target allows another slab and this fill has to leave the lock to get one. Reclaiming
            // instead would throw away a slab's worth of vectors the target says the store may keep.
            return null;
        }
        if (pendingSlabs > 0 && liveSlabs < maxSlabs) {
            // Another fill is already fetching a slab the target allows, so this fill waits for it by
            // simply not being cached. Reclaiming now would evict vectors the store is about to have room
            // for, and a fill that does not happen costs a future hit and nothing else.
            return null;
        }
        final Slab oldest = FIFO.poll();
        if (oldest == null) {
            return null;
        }
        reclaimedSlabs.increment();
        final long started = System.nanoTime();
        if (oldest.slotBytes == length) {
            // The common case, and the only one in which a second chance can do anything: the reclaimed slab
            // is reopened for the vectors it already holds, so the ones that were read stay in their slots
            // and the rest become the free slots this call came for.
            reclaimInPlace(oldest);
            OPEN_SLABS.put(length, oldest);
        } else {
            // Reopened for a different vector length, so the slot layout changes and a survivor has nowhere
            // to stay. See the class javadoc for why they are dropped rather than copied.
            dropAll(oldest);
            OPEN_SLABS.remove(oldest.slotBytes, oldest);
            open(oldest, length);
        }
        FIFO.add(oldest);
        reclaimNanos.add(System.nanoTime() - started);
        return oldest;
    }

    /**
     * Registers the slab {@code request} is carrying and opens it for {@code length}-byte vectors, or returns
     * {@code null}: at the target, with growth latched, or -- the interesting case -- with
     * {@link SlabRequest#wanted} set to tell the caller to allocate one outside the lock and ask again. The
     * half of {@link #obtainSlab} that displaces nothing, which is the only half span fill is allowed. Called
     * under the write lock.
     *
     * <p>Nothing here allocates: that is N3. The target check counts the slabs other fills are allocating
     * right now ({@link #pendingSlabs}) as well as the live ones, so concurrent growth cannot overshoot.
     *
     * <p>One allocation attempt per latch, not one per fill: see {@link #growthLatched} for what a failed
     * attempt costs the caller. The store then stays the size it is, keeps serving the slabs it holds, and
     * the cost is hit rate -- which is what a cache that cannot grow should cost.
     */
    private static Slab growSlabs(final int length, final SlabRequest request) {
        if (growthLatched || request == null) {
            return null;
        }
        // This request's own reservation does not count against it; every other fill's does.
        final int others = pendingSlabs - (request.reserved ? 1 : 0);
        if (liveSlabs + others >= maxSlabs) {
            return null;
        }
        if (request.spare == null) {
            if (others > 0) {
                // One pending allocation at a time, because one slab holds thousands of vectors and the
                // fills waiting for it will all fit in it. Letting every concurrent fill fetch its own was
                // MEASURED on the re-proof's arm E: the first burst's threads each reserved a slab, a 64mb
                // store reached its 2 slab target before a single vector had been cached, span fill stopped
                // (it only runs below the target) and the arm's hit count fell from 2,921 to 832. Before N3
                // the write lock enforced this for free, because only one fill could be allocating.
                return null;
            }
            pendingSlabs++;
            request.reserved = true;
            request.wanted = true;
            return null;
        }
        final Slab fresh = register(request.spare);
        request.spare = null;
        if (request.reserved) {
            pendingSlabs--;
            request.reserved = false;
        }
        open(fresh, length);
        return fresh;
    }

    /**
     * Gives {@code slab}'s second chance: the slots read since they were written keep both their vector and
     * their slot, up to half the slab, and every other slot joins the free list. No byte moves and no index
     * entry changes.
     */
    private static void reclaimInPlace(final Slab slab) {
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
        droppedEntries.add(dropped);
        readmissions.add(kept);
        if (emptied != null) {
            // The closed-file rule's second half, done once per reclaim rather than once per slot: a file
            // whose last cached vector has just gone loses its index table too.
            for (final DirectIOSlotMap owner : emptied) {
                if (owner.liveEntries() <= 0) {
                    INDEX.remove(owner.fileId(), owner);
                }
            }
        }
    }

    /** Frees every live slot of {@code slab} and empties it. Called under the write lock. */
    private static void dropAll(final Slab slab) {
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
    private static void detach(final Slab slab, final int slot, final boolean leaveCellStale) {
        final DirectIOSlotMap owner = slab.owner[slot];
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
        if (owner.noteDetached()) {
            INDEX.remove(owner.fileId(), owner);
        }
    }

    /** Records that {@code slot} of {@code slab} now holds the vector at {@code offset} of {@code map}. */
    private static void attach(final Slab slab, final int slot, final DirectIOSlotMap map, final long offset) {
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
                // because this very offset is about to be attached again, but detach's INDEX.remove would
                // fire if it did; re-attaching below puts it back, so the map object stays the one the slot
                // points at either way.
                detach(previousSlab, previousSlot, false);
            }
        }
        if (INDEX.get(map.fileId()) != map) {
            INDEX.put(map.fileId(), map);
        }
        map.put(offset, slab.id, slot);
        map.noteAttached();
        slab.owner[slot] = map;
        slab.offset[slot] = offset;
        slab.hit[slot] = 0;
        liveEntries++;
        liveEntryBytes += slab.slotBytes;
    }

    private static Slab slabAt(final int id) {
        final Slab[] slabs = slabsById;
        return id >= 0 && id < slabs.length ? slabs[id] : null;
    }

    /** Opens {@code slab} for {@code length}-byte vectors and makes it the one being appended to. */
    private static void open(final Slab slab, final int length) {
        slab.openFor(length);
        OPEN_SLABS.put(length, slab);
    }

    private static Slab register(final DirectIOSlab memory) {
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
        FIFO.add(slab);
        return slab;
    }

    /**
     * Takes a slab out of service under the write lock and hands its memory back to the caller to release
     * outside the lock (N3). Clearing {@code slabsById} is what makes that safe: every lookup resolves a slab
     * through {@link #slabAt} under the read lock, so a reader arriving after this cannot reach the slab and a
     * reader already inside held the read lock this thread had to wait for.
     *
     * @param slab the slab to detach
     * @return its memory, for {@link #releaseDetached}
     */
    private static DirectIOSlab detach(final Slab slab) {
        slabsById[slab.id] = null;
        OPEN_SLABS.remove(slab.slotBytes, slab);
        liveSlabs--;
        // Memory is coming back, so an earlier allocation failure may not stand any more (N2).
        growthLatched = false;
        return slab.memory;
    }

    // ---------------------------------------------------------------------------------------------------
    // Counters. KNNStats publishes these through direct_io_stats.cache.
    // ---------------------------------------------------------------------------------------------------

    /** Lookups that returned a vector. See {@link #lookupsPerVector()} for how to read this against misses. */
    public static long hits() {
        return hits.sum();
    }

    /** Lookups that did not. */
    public static long misses() {
        return misses.sum();
    }

    /**
     * <b>One.</b> A vector is counted once, at the lookup that serves or misses it, which is the one in
     * {@code readFloats}. The lookup in {@code prefetch} decides whether the range joins the burst and
     * deliberately counts nothing, so {@code hits / (hits + misses)} is the hit rate <em>and</em> the counts
     * are the vectors served. Phase 15 counted both sites, so its totals were twice the vectors.
     *
     * @return 1
     */
    public static int lookupsPerVector() {
        return 1;
    }

    /**
     * Vectors dropped: reclaimed without a second chance, or dropped when their slab was released or reopened
     * for another vector length. The {@code evictions} counter, under the name it has always had.
     */
    public static long evictions() {
        return droppedEntries.sum();
    }

    /** Vectors written into a slot, requested fills and span fills together. */
    public static long insertions() {
        return insertions.sum();
    }

    /**
     * Slot bytes held by live vectors right now. Payload, not the slab total: see {@link #slabBytes()} for
     * what the node is actually holding, which is the figure a memory budget cares about.
     */
    public static long bytes() {
        LOCK.readLock().lock();
        try {
            return liveEntryBytes;
        } finally {
            LOCK.readLock().unlock();
        }
    }

    /** Vectors held right now. */
    public static long entries() {
        LOCK.readLock().lock();
        try {
            return liveEntries;
        } finally {
            LOCK.readLock().unlock();
        }
    }

    /** The store's byte target, rounded down to whole slabs; {@code 0} when the cache is off. */
    public static long maxBytes() {
        return (long) maxSlabs * slabBytes;
    }

    /** Slabs held right now. */
    public static long slabs() {
        LOCK.readLock().lock();
        try {
            return liveSlabs;
        } finally {
            LOCK.readLock().unlock();
        }
    }

    /** Bytes of off-heap slab held right now, which is what the node is paying for. */
    public static long slabBytes() {
        return slabs() * slabBytes;
    }

    /** Slabs taken from the head of the FIFO queue to make room. */
    public static long reclaimedSlabs() {
        return reclaimedSlabs.sum();
    }

    /** Vectors kept by a second chance when their slab was reclaimed. */
    public static long readmissions() {
        return readmissions.sum();
    }

    /**
     * Vectors admitted because they lay inside a span a burst had already staged, rather than because a read
     * asked for them (M4). Zero while {@code knn.direct_io.rescore.cache.span_fill} is false, and zero once
     * the store reaches its slab target.
     */
    public static long spanFills() {
        return spanFills.sum();
    }

    /**
     * Slab allocations the backend refused (N2). One per latch: after a failure the store stops growing until
     * its target changes or it releases a slab, so this is also how many times growth was given up on.
     * Non-zero means the node's {@code knn.direct_io.rescore.cache.max_bytes} is above what the JVM would
     * hand out, and the store is smaller than the operator asked for.
     */
    public static long slabAllocFailures() {
        return slabAllocFailures.sum();
    }

    /** Nanoseconds spent inside a slab reclaim, which is write-lock time every fill waits behind. */
    public static long reclaimNanos() {
        return reclaimNanos.sum();
    }

    /**
     * How this node allocates off-heap slabs for cached vectors: {@code "ffm"} when
     * {@code java.lang.foreign.Arena} resolved and proved itself on this runtime, else
     * {@code "direct_buffer"}. Reported as {@code direct_io_stats.cache.slab_memory} because the two differ
     * in what releasing memory means and in whether the bytes are inside {@code MaxDirectMemorySize}, which
     * an operator reading a cache size needs to know; see {@link DirectIOSlab}.
     *
     * @return the backend's name, never null
     */
    public static String slabMemory() {
        return DirectIOSlab.backend();
    }

    /** Heap held by the index itself, for the bytes-per-entry figure M2's bench reports. */
    static long indexBytes() {
        LOCK.readLock().lock();
        try {
            long total = 48L * INDEX.size();
            for (final DirectIOSlotMap map : INDEX.values()) {
                total += map.indexBytes();
            }
            for (final Slab slab : FIFO) {
                if (slab.owner != null) {
                    total += 48 + 4L * slab.owner.length + 16 + 8L * slab.offset.length + 16 + slab.hit.length;
                    total += 16 + 4L * slab.free.length;
                }
            }
            return total;
        } finally {
            LOCK.readLock().unlock();
        }
    }

    /** Test seam: empties the store, releases every slab, and zeroes every counter. */
    static void resetForTesting() {
        slabMemory = DIRECT_SLABS;
        configure(0);
        awaitReleasesForTesting();
        slabBytes = DirectIOSlab.SLAB_BYTES;
        spanFill = KNNSettings.KNN_DIRECT_IO_RESCORE_CACHE_SPAN_FILL_DEFAULT_VALUE;
        hits.reset();
        misses.reset();
        insertions.reset();
        reclaimedSlabs.reset();
        readmissions.reset();
        droppedEntries.reset();
        reclaimNanos.reset();
        spanFills.reset();
        slabAllocFailures.reset();
    }

    /**
     * Test seam: where slab memory comes from and goes back to, so a suite can exercise a backend that will
     * not allocate (N2) and can see which thread, holding which lock, each call runs on (N3). Takes effect on
     * the next growth attempt or release and is restored by {@link #resetForTesting()}.
     *
     * @param memory the allocate and release pair to use until the next reset
     */
    static void setSlabMemoryForTesting(final SlabMemory memory) {
        LOCK.writeLock().lock();
        try {
            slabMemory = memory;
            growthLatched = false;
        } finally {
            LOCK.writeLock().unlock();
        }
    }

    /**
     * Test seam: waits for every release in flight to have given its memory back, because
     * {@link #releaseDetached} does that on a thread of its own.
     */
    static void awaitReleasesForTesting() {
        final long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(30);
        while (releasesRunning.get() > 0 && System.nanoTime() < deadline) {
            Thread.onSpinWait();
        }
    }

    /**
     * Test seam: whether the calling thread is inside this store's lock in either mode. N3's property is that
     * neither a slab allocation nor a slab release ever runs while it is.
     *
     * @return whether this thread holds the read or the write lock
     */
    static boolean lockHeldByCurrentThreadForTesting() {
        return LOCK.isWriteLockedByCurrentThread() || LOCK.getReadHoldCount() > 0;
    }

    /**
     * Test seam: the slab size, so a suite can exercise eviction and the second chance without allocating
     * 32 MiB at a time. Only valid with the cache off, which {@link #resetForTesting()} guarantees.
     *
     * @param bytes the slab size to use until the next reset
     */
    static void setSlabBytesForTesting(final int bytes) {
        if (maxSlabs != 0) {
            throw new IllegalStateException("The slab size can only be changed while the cache is off");
        }
        slabBytes = bytes;
    }

    /** Test seam: the number of slots {@code length}-byte vectors get in one slab. */
    static int slotsPerSlab(final int length) {
        return Math.min(slabBytes / length, DirectIOSlotMap.MAX_SLOTS);
    }

    /** Test seam: files with an index table right now, so a test can assert a dead one's table is gone. */
    static int indexedFiles() {
        LOCK.readLock().lock();
        try {
            return INDEX.size();
        } finally {
            LOCK.readLock().unlock();
        }
    }

    /** Test seam: whether {@code fileId} has an index table right now. */
    static boolean hasIndexFor(final String fileId) {
        LOCK.readLock().lock();
        try {
            return INDEX.containsKey(fileId);
        } finally {
            LOCK.readLock().unlock();
        }
    }

    /**
     * Test seam: re-derives the live entry count and byte total from the slabs themselves and checks them
     * against the running counters, so a test can assert that no path has lost or double counted a slot.
     * Every slot that still carries a back pointer must also be the slot its file's index names for it.
     *
     * @throws IllegalStateException on any disagreement, naming it
     */
    static void checkInvariantsForTesting() {
        LOCK.writeLock().lock();
        try {
            long entries = 0;
            long entryBytes = 0;
            final Map<DirectIOSlotMap, Integer> counted = new HashMap<>();
            for (final Slab slab : FIFO) {
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
                    if (INDEX.get(owner.fileId()) != owner) {
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
            if (FIFO.size() != liveSlabs) {
                throw new IllegalStateException("queue holds " + FIFO.size() + " slabs but the counter says " + liveSlabs);
            }
            if (pendingSlabs < 0 || liveSlabs + pendingSlabs > Math.max(maxSlabs, liveSlabs)) {
                throw new IllegalStateException(liveSlabs + " live and " + pendingSlabs + " pending slabs against a target of " + maxSlabs);
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
        } finally {
            LOCK.writeLock().unlock();
        }
    }
}

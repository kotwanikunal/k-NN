/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import lombok.extern.log4j.Log4j2;

import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantLock;

/**
 * The global slab target and the slab supply for {@link DirectIOVectorCache}, which is everything about the
 * store that cannot be a per-segment decision: how many slabs the node may hold in total, where their memory
 * comes from and goes back to, and whether an allocation that failed should be retried (Q, section C1a).
 *
 * <h2>Why this is a separate object from a segment</h2>
 * A {@link DirectIOCacheSegment} owns its slabs, its index, its FIFO queue and its lock, and everything it
 * does under that lock touches only itself. But {@code knn.direct_io.rescore.cache.max_bytes} is one number
 * for the node, so somebody has to hold it and has to stop two segments growing past it between them. This
 * is that somebody, and it is deliberately the smallest thing that can be: a target, a supply counter, an
 * allocation latch and the backend seam.
 *
 * <h2>The lock order, which is the whole deadlock proof</h2>
 * <b>This broker's lock, then at most one segment's lock.</b> No path holds two segment locks, and no path
 * takes this lock while holding a segment lock. The only production path that takes this lock at all is
 * {@link DirectIOVectorCache#configure(long)}, which serialises resizes; the test-only invariant check takes
 * it first and then every segment's write lock in index order, which is the same order and so cannot cycle.
 *
 * <h2>The supply counter, and why it is one word</h2>
 * {@link #liveSlabs()} and {@link #pendingSlabs()} are packed into one {@link AtomicLong} (live in the high
 * 32 bits, pending in the low 32) so that the three transitions a growing fill makes are each a single
 * compare and set:
 * <ul>
 *   <li>{@link #reserve()}: {@code pending + 1}, refused when {@code live + pending} has reached the
 *       target. This is what makes it impossible for two fills to both see room for the last slab;</li>
 *   <li>{@link #commitReservation()}: {@code pending - 1} and {@code live + 1} <b>together</b>, so the sum
 *       the reservation is counted against never dips and never jumps. Two counters would have a window in
 *       which either an overshoot or an undercount was visible, and the invariant check reads both;</li>
 *   <li>{@link #cancelReservation()}: {@code pending - 1}, for a fill whose allocation failed or whose slab
 *       the target no longer allows.</li>
 * </ul>
 *
 * <p>No lock is taken for any of them, which is what lets a segment reserve while it holds its own write
 * lock without ever violating the order above. Before Q the same arithmetic ran under the store's one write
 * lock, which is where the reservation was invented (N3).
 */
@Log4j2
final class DirectIOSlabBroker {

    /** Serialises resizes. See the lock order in the class javadoc: this one comes first, or not at all. */
    private final ReentrantLock lock = new ReentrantLock();

    /**
     * Slabs the store may hold in total, from {@link DirectIOVectorCache#configure(long)}. Zero means off,
     * and is the default: it is the one volatile read every entry point to the store makes before it does
     * anything else at all.
     */
    private volatile int maxSlabs;

    /** Set once per node, and only by a test, so a suite can exercise eviction without 32 MiB of slabs. */
    private volatile int slabBytes = DirectIOSlab.SLAB_BYTES;

    /** Live slabs in the high 32 bits, reservations in flight in the low 32. See the class javadoc. */
    private final AtomicLong supply = new AtomicLong();

    /**
     * Set when an allocation failed, so the store stops asking for a slab it cannot have (N2). Cleared when
     * the target changes or a slab is released, which are the two events that can make the answer different.
     *
     * <p>MEASURED by the orchestrator (2026-10-12, desktop, {@code -Xmx256m} with
     * {@code -XX:MaxDirectMemorySize=96m}): a failed reservation blocks the caller for 515 to 525 ms and runs
     * one {@code Pause Full (System.gc())}, for {@code allocateDirect} on JDK 21 and 25 and for FFM on JDK 21,
     * because {@code java.nio.Bits#reserveMemory} calls {@code System.gc()} and then sleeps nine times with
     * backoff. Retrying it on every fill below the target costs that once per fill, and the answer is the same
     * every time.
     *
     * <p>An {@link AtomicBoolean} rather than a field under a lock, because the one thing that must happen
     * exactly once per latch is the warning, and {@link #latchGrowth()} is the compare and set that decides
     * which caller logs it.
     */
    private final AtomicBoolean growthLatched = new AtomicBoolean();

    private volatile DirectIOVectorCache.SlabMemory slabMemory = DIRECT_SLABS;

    private static final DirectIOVectorCache.SlabMemory DIRECT_SLABS = new DirectIOVectorCache.SlabMemory() {
        @Override
        public DirectIOSlab allocate(final int bytes) {
            return DirectIOSlab.allocate(bytes);
        }

        @Override
        public void release(final DirectIOSlab slab) {
            slab.close();
        }
    };

    /** Release threads still giving memory back, so a test can wait for it. */
    private final AtomicInteger releasesRunning = new AtomicInteger();

    private static final long PENDING_MASK = 0xFFFF_FFFFL;

    private static int liveOf(final long packed) {
        return (int) (packed >>> 32);
    }

    private static int pendingOf(final long packed) {
        return (int) (packed & PENDING_MASK);
    }

    private static long packed(final int live, final int pending) {
        return ((long) live << 32) | (pending & PENDING_MASK);
    }

    // ---------------------------------------------------------------------------------------------------
    // The target
    // ---------------------------------------------------------------------------------------------------

    /** Slabs the store may hold; zero means off. One volatile read, which is the store's off switch. */
    int maxSlabs() {
        return maxSlabs;
    }

    /** Called under {@link #lock()} by {@link DirectIOVectorCache#configure(long)} and nowhere else. */
    void setMaxSlabs(final int wanted) {
        maxSlabs = wanted;
    }

    int slabBytes() {
        return slabBytes;
    }

    /** Test seam, valid only with the store off; {@link DirectIOVectorCache#setSlabBytesForTesting} checks. */
    void setSlabBytes(final int bytes) {
        slabBytes = bytes;
    }

    /** Slabs in service across every segment. */
    int liveSlabs() {
        return liveOf(supply.get());
    }

    /** Reservations in flight: slabs a fill is allocating right now and has not registered yet. */
    int pendingSlabs() {
        return pendingOf(supply.get());
    }

    /** Whether the store would take another slab if a fill asked for one. Advisory, and lock free. */
    boolean belowTarget() {
        final int target = maxSlabs;
        return target != 0 && liveSlabs() < target;
    }

    /**
     * The resize lock. Taken by {@link DirectIOVectorCache#configure(long)} and by {@link #lend}, and by
     * nothing else; see the class javadoc for the order it comes in. Counted while a test has asked for
     * acquisition counting, because C5's criterion is that a {@code configure} with an unchanged target
     * takes <b>no lock at all</b>, and that is the once-per-prefetch-burst path.
     */
    ReentrantLock lock() {
        if (DirectIOVectorCache.acquisitionCounting) {
            DirectIOVectorCache.brokerLockAcquisitions.increment();
        }
        return lock;
    }

    boolean lockHeldByCurrentThread() {
        return lock.isHeldByCurrentThread();
    }

    // ---------------------------------------------------------------------------------------------------
    // The supply
    // ---------------------------------------------------------------------------------------------------

    /**
     * Takes one of the target's slabs for a fill that is about to leave its segment's lock and allocate, or
     * returns false because the target has no room. Lock free; see the class javadoc for why that is safe and
     * why it has to be.
     *
     * @return whether the reservation was taken
     */
    boolean reserve() {
        while (true) {
            final int target = maxSlabs;
            if (target == 0) {
                return false;
            }
            final long current = supply.get();
            final int live = liveOf(current);
            final int pending = pendingOf(current);
            if (live + pending >= target) {
                return false;
            }
            if (supply.compareAndSet(current, packed(live, pending + 1))) {
                return true;
            }
        }
    }

    /** Turns this caller's reservation into a live slab, in one step, so no sum is ever wrong. */
    void commitReservation() {
        while (true) {
            final long current = supply.get();
            final int pending = pendingOf(current);
            if (pending <= 0) {
                throw new IllegalStateException("committed a slab reservation that was not held");
            }
            if (supply.compareAndSet(current, packed(liveOf(current) + 1, pending - 1))) {
                return;
            }
        }
    }

    /** Gives a reservation back unused: the allocation failed, or the target shrank under it. */
    void cancelReservation() {
        while (true) {
            final long current = supply.get();
            final int pending = pendingOf(current);
            if (pending <= 0) {
                throw new IllegalStateException("cancelled a slab reservation that was not held");
            }
            if (supply.compareAndSet(current, packed(liveOf(current), pending - 1))) {
                return;
            }
        }
    }

    /**
     * Turns a live slab a donor segment has just given up into a reservation the recipient will commit, in
     * one step, so that "the segments' slab counts sum to the broker's live count" holds at every instant
     * including mid-transfer.
     */
    private void transferOut() {
        while (true) {
            final long current = supply.get();
            final int live = liveOf(current);
            if (live <= 0) {
                throw new IllegalStateException("transferred a slab the broker was not counting");
            }
            if (supply.compareAndSet(current, packed(live - 1, pendingOf(current) + 1))) {
                return;
            }
        }
    }

    /** Stops counting a slab a segment has just detached. */
    void releaseLive() {
        while (true) {
            final long current = supply.get();
            final int live = liveOf(current);
            if (live <= 0) {
                throw new IllegalStateException("released a slab the broker was not counting");
            }
            if (supply.compareAndSet(current, packed(live - 1, pendingOf(current)))) {
                // Memory is coming back, so an earlier allocation failure may not stand any more (N2).
                growthLatched.set(false);
                return;
            }
        }
    }

    // ---------------------------------------------------------------------------------------------------
    // The backend
    // ---------------------------------------------------------------------------------------------------

    boolean growthLatched() {
        return growthLatched.get();
    }

    /**
     * Latches growth after a refusal, and says whether this caller is the one that has to say so.
     *
     * @return whether growth was not already latched
     */
    boolean latchGrowth() {
        return growthLatched.compareAndSet(false, true);
    }

    /** A new target, or memory coming back, is a new question, so an earlier failure stops standing (N2). */
    void unlatchGrowth() {
        growthLatched.set(false);
    }

    /**
     * Allocates one slab, with no lock held, and says whether to ask the store again. A refusal latches
     * growth (N2) so that no later fill pays for the same answer.
     *
     * @param request the fill's request, which carries the slab back to it
     * @return whether a slab was allocated
     */
    boolean allocateOutsideLock(final DirectIOCacheSegment.Request request) {
        if (request.wantsTransfer) {
            return lend(request);
        }
        final int bytes = slabBytes;
        request.spare = slabMemory.allocate(bytes);
        if (request.spare != null) {
            return true;
        }
        if (latchGrowth()) {
            DirectIOVectorCache.slabAllocFailures.increment();
            log.warn(
                "A k-NN Direct I/O vector cache slab of {} bytes could not be allocated from {}; the cache will hold the {} "
                    + "slabs it has until knn.direct_io.rescore.cache.max_bytes changes or a slab is released",
                bytes,
                DirectIOSlab.backend(),
                liveSlabs()
            );
        }
        return false;
    }

    void release(final DirectIOSlab slab) {
        slabMemory.release(slab);
    }

    /**
     * Lends {@code request}'s segment a slab from the segment holding the most, because the requesting
     * segment holds none and the store is already at its target (Q, section 2.2 step 3).
     *
     * <p><b>The donor keeps at least one slab</b>, which is what bounds this: a segment that has just been
     * lent one holds exactly one and so can never be the next donor, and with the segment count capped at
     * {@code maxSlabs / 8} the average segment holds eight, so a donor always exists while any segment holds
     * none. The design says "above its fair share"; the fair share is not the test used here, because every
     * segment can hold exactly its share while one holds none, and then nothing would ever be lent.
     *
     * <p>This lock, then the donor's write lock, and no other segment lock: the requesting segment's lock is
     * released before this is called, which is what keeps the one-segment-lock rule.
     *
     * @param request the fill's request, which carries the slab back to it
     * @return whether a slab was lent
     */
    private boolean lend(final DirectIOCacheSegment.Request request) {
        lock.lock();
        try {
            DirectIOCacheSegment donor = null;
            int most = 1;
            for (final DirectIOCacheSegment candidate : DirectIOVectorCache.segmentsSnapshot()) {
                final int held = candidate.liveSlabs();
                if (held > most) {
                    most = held;
                    donor = candidate;
                }
            }
            if (donor == null) {
                request.wantsTransfer = false;
                return false;
            }
            final DirectIOSlab memory = donor.detachOldestForTransfer();
            if (memory == null) {
                request.wantsTransfer = false;
                return false;
            }
            transferOut();
            DirectIOVectorCache.slabTransfers.increment();
            request.tookTransfer(memory);
            return true;
        } finally {
            lock.unlock();
        }
    }

    /**
     * Gives back the memory of slabs a release has detached, with no lock held and not on the calling thread.
     *
     * <p><b>Why not under the lock.</b> An FFM arena takes about 2.2 ms to close (MEASURED, M1), so before N3
     * a {@code configure(0)} at a 4 GB target held the store's write lock for about 0.26 s and a 12 GB one for
     * about 0.8 s, and every lookup on the node waited behind it. Detaching is what makes the close safe to do
     * outside the lock: it clears the slab's entry in its segment's {@code slabsById} under that segment's
     * write lock, so its index cells go stale and no reader that arrives afterwards can reach it, while a
     * reader already inside holds the read lock the detaching thread had to wait for.
     *
     * <p><b>Why not on the calling thread.</b> Every production call to {@link DirectIOVectorCache#configure}
     * arrives from {@code DirectIOVectorIndexInput#beginBurst}, which is to say on a thread serving a query,
     * so an operator turning the cache off would spend that query's latency closing arenas. A thread per
     * release rather than a pool, because a release happens when an operator changes a setting: there is no
     * steady state to amortise a pool over, and a thread that does not exist cannot be left running.
     *
     * @param detached the memory to give back
     */
    void releaseDetached(final List<DirectIOSlab> detached) {
        if (detached.isEmpty()) {
            return;
        }
        final DirectIOVectorCache.SlabMemory backend = slabMemory;
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

    /** Test seam: waits for every release in flight to have given its memory back. */
    void awaitReleases() {
        final long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(30);
        while (releasesRunning.get() > 0 && System.nanoTime() < deadline) {
            Thread.onSpinWait();
        }
    }

    /** Test seam: the allocate and release pair to use until the next reset. */
    void setSlabMemory(final DirectIOVectorCache.SlabMemory memory) {
        slabMemory = memory;
        growthLatched.set(false);
    }

    void resetSlabMemory() {
        setSlabMemory(DIRECT_SLABS);
    }
}

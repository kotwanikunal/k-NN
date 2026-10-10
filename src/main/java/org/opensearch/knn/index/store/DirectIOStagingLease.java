/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import lombok.extern.log4j.Log4j2;

import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.concurrent.atomic.AtomicLong;

/**
 * One staging arena per <b>thread</b>, lent to whichever {@link DirectIOVectorIndexInput} is running a
 * prefetch burst on that thread. The direct memory the random rescore path holds is then bounded by the
 * number of threads that can be inside a burst at once, not by when the garbage collector next runs.
 *
 * <h2>The measurement this exists for</h2>
 * Phase 15's budget ({@code knn.direct_io.direct_memory_budget}) bounded the wrong quantity, and the node
 * showed it twice in one night (MEASURED 2026-10-06, jar {@code 4a8cf9e9}, r7g.xlarge, {@code -Xmx24g}
 * {@code -XX:MaxDirectMemorySize=2g}, 16 clients):
 * <ul>
 *   <li>At the default budget of 1 GB, {@code live_bytes} reached <b>1,068-1,087 MB</b> within two minutes
 *       of load, {@code staged_reads} froze at 159,137, {@code stage_declined} and {@code blocking_reads}
 *       rose ~700/s and {@code declined_budget} ran to 373k-614k. Twenty seconds later {@code live_bytes}
 *       fell to <b>0 in one step</b> on a young collection. The budget did not bound the feature, it
 *       <em>wedged</em> it between garbage collections.</li>
 *   <li>At {@code 0b} the gauge ratcheted to <b>7.46 GB, 9.0 GB and 28 GB</b> against a 2 GB wall, because
 *       {@code fitsBudget} short-circuits when the budget is off and that was the only place the phantom
 *       queue was drained outside a stats read. Meanwhile the JDK's own {@code Bits.reserveMemory} forced
 *       <b>23 full collections (2.1 s)</b> in 65 minutes on a 0.6 GB-occupied 24 GB heap.</li>
 * </ul>
 *
 * <p>Neither number was a measurement of memory in use. Every {@code slice()} and {@code clone()} is a new
 * input with its own buffers, so a per-query {@code OffHeapFloatVectorValues} slice carved a fresh
 * {@code staged_ranges x arenaSlotBytes} arena (~516 KiB at the shipped defaults) on its first prefetch and
 * held it for the input's life — which, since that class is not {@code Closeable}, ends only when a
 * collection finds it. Arenas therefore churned at roughly {@code QPS x segments x 516 KiB/s} (9-13 MiB/s on
 * the node) while the memory actually being read through was about {@code 16 concurrent bursts x 516 KiB},
 * i.e. <b>8 MiB</b>. The accounting measured the churn awaiting collection and called it "live".
 *
 * <h2>Why the lease is per thread and not per input</h2>
 * The obvious fix — return the arena at the next prefetch on the same input — cannot work, because a
 * per-query input usually never sees another prefetch: it is built for one batch and abandoned. The reliable
 * boundary is the next burst <b>on the same thread</b>, and that boundary is reliable because the rescore
 * path is thread confined:
 * {@code PrefetchableFlatVectorScorer.PrefetchableRandomVectorScorer#bulkScore} calls
 * {@code PrefetchableVectorValuesHelper#doPrefetch} and then {@code delegate.bulkScore} as two statements of
 * one synchronous method, and under concurrent segment search each leaf is a
 * {@code TaskExecutor$Task extends FutureTask} whose {@code run()} is a one-shot, so a leaf's whole
 * prefetch-then-read sequence happens on exactly one thread. See FINDINGS.md D3c(a) for the citations.
 *
 * <h2>{@link ThreadLocal} rather than a map keyed by thread</h2>
 * A {@code ThreadLocal}'s values live in a map that is a <em>field of the {@link Thread}</em>, so when a
 * reader thread dies its lease becomes garbage with it and the arena's charge comes back through the phantom
 * reference with no sweeper to write. A {@code ConcurrentHashMap<Thread, Lease>} would need weak keys plus a
 * cleanup pass to get the same property, and would put a hash lookup on every burst where this puts a field
 * read. The one cost of a {@code ThreadLocal} — that it keeps this class, and so the plugin classloader,
 * reachable from a long-lived pool thread — is already paid by {@link DirectIOReadPool}'s static singleton.
 *
 * <h2>Takeback, and the two rules that make it safe without a lock</h2>
 * A burst's first {@code prefetch} calls {@link #take}, which quiesces every range the previous holder left
 * staged using the same claim logic a burst boundary always used. If they all quiesce the arena is reused;
 * if any was abandoned or is still running past the bounded wait the whole arena is retired to the collector
 * — with its charge still outstanding, which the phantom reference returns later — and a fresh one carved.
 * <b>No buffer is ever handed to a new burst while a pool read may still write into it.</b>
 *
 * <p>The previous holder's table is then <em>void</em>, and two rules keep it that way without a seqlock:
 * <ul>
 *   <li><b>Generation.</b> Every {@link #take} stamps the lease with a value from one node-wide counter, so
 *       a generation identifies a burst uniquely across every thread for the life of the JVM. An input
 *       remembers the generation it was given and may consult the table only while the lease still carries
 *       it; a revoked input falls back to the blocking path, which is the fallback every other decline
 *       already takes.</li>
 *   <li><b>Owner confinement.</b> Only the owner thread both revokes and reads. A read arriving from any
 *       other thread does not touch the table at all — it takes the blocking path and counts
 *       {@code foreign_thread_reads}. That is what makes the generation check race free: the one thread that
 *       could observe a stale generation is the one thread that cannot change it concurrently with itself.
 *       </li>
 * </ul>
 *
 * <h2>What the lease does <em>not</em> own</h2>
 * The sequential read-ahead ring stays per input. Its churn scales with the number of concurrent sequential
 * readers — merges, clones, splits — rather than with QPS, and a scan holds its ring for the length of the
 * scan, so there is no burst boundary to lend it at. Only its accounting changed.
 */
@Log4j2
final class DirectIOStagingLease {

    /**
     * Generations are handed out from one node-wide counter rather than per lease, so that a generation
     * value names one burst on one thread for the life of the JVM. That is what makes an input's remembered
     * generation safe to compare against <em>any</em> lease it might meet: a thread id can be reused after a
     * thread dies, but the generation the dead thread held can never be current again.
     */
    private static final AtomicLong GENERATIONS = new AtomicLong();

    private static final ThreadLocal<DirectIOStagingLease> LEASES = ThreadLocal.withInitial(DirectIOStagingLease::new);

    /** This thread's lease, created on first use. */
    static DirectIOStagingLease current() {
        return LEASES.get();
    }

    // -- the burst table. Held by one input at a time, named by #generation. --

    /** The current holder's generation, or {@code -1} when nobody holds the lease. */
    private long generation = -1;
    /** Geometry {@link #arena} was carved at; a burst wanting a different shape retires it. */
    private int slots;
    private int slotBytes;
    private int blockSize;
    private ByteBuffer arena;
    /** Fixed views of {@link #arena}, one per slot. */
    private ByteBuffer[] slotBuffers;
    private FloatBuffer[] slotFloats;
    /**
     * Per-slot buffers for a coalesced group wider than a slot, grown on demand and reused across bursts,
     * and dropped with the arena when it is retired, because a pool read can be writing into one of these
     * too.
     *
     * <p>{@code staged_ranges} of them at {@code max_span_bytes} each is 8.25 MiB per thread at the
     * defaults, with no bound on the number of leases, so D3d gives them a node-wide sub-budget of the
     * direct-memory budget: see {@code DirectIOVectorIndexInput#allocateWideBuffer}, which
     * {@link #bufferFor} asks before it grows one.
     */
    private ByteBuffer[] wideBuffers;
    private FloatBuffer[] wideFloats;
    private DirectIOVectorIndexInput.StagedRange[] table;
    private int count;

    // -- the blocking window, also one per thread rather than one per input. --

    private ByteBuffer window;
    private FloatBuffer windowFloats;
    private int windowBytes;
    private int windowBlockSize;
    /**
     * Which input last filled {@link #window}, so that the cached {@code [windowStart, +windowValid)} range
     * is only ever served back to the input that read it.
     *
     * <p><b>The generation is not enough here, and that is worth stating rather than discovering.</b> The
     * generation only moves when a burst is taken; two inputs doing nothing but blocking reads on one thread
     * — the merge path, the faiss first pass, an exact search with prefetch off — never move it, so a window
     * check keyed on the generation alone would let the second input's bytes be served to the first. The
     * owner token moves on every fill, which is the event that actually invalidates the cached range.
     */
    private long windowOwner = -1;
    private long windowStart = -1;
    private int windowValid;

    private DirectIOStagingLease() {}

    /** The generation this lease currently carries, or {@code -1} when it is unheld. */
    long generation() {
        return generation;
    }

    /**
     * Takes this thread's lease for a new burst at the given geometry.
     *
     * @return the new generation the caller must remember, or {@code -1} when the direct-memory budget
     *         refused a fresh arena, in which case the caller's reads must block
     */
    long take(final int slots, final int slotBytes, final int blockSize) {
        final boolean quiesced = quiesceHeld();
        generation = -1;
        if (arena != null && (quiesced == false || this.slots != slots || this.slotBytes != slotBytes || this.blockSize != blockSize)) {
            // Two reasons, one counter: a read that would not quiesce may still write into these buffers, and
            // a burst of a different shape cannot use them. Both are "an arena was dropped rather than
            // reused", which is what an operator reads arenas_retired for.
            retire();
        }
        if (arena == null) {
            // The budget is checked here and only here for the arena, because this is now the only place one
            // is carved: a burst that reuses its thread's arena cannot push the node's footprint up, so it
            // has nothing to be declined for.
            if (DirectIOVectorIndexInput.fitsBudget((long) slots * slotBytes, blockSize) == false) {
                return -1;
            }
            this.slots = slots;
            this.slotBytes = slotBytes;
            this.blockSize = blockSize;
            slotBuffers = new ByteBuffer[slots];
            slotFloats = new FloatBuffer[slots];
            wideBuffers = new ByteBuffer[slots];
            wideFloats = new FloatBuffer[slots];
            table = new DirectIOVectorIndexInput.StagedRange[slots];
            arena = DirectIOVectorIndexInput.carveArena(slots, slotBytes, blockSize, slotBuffers, slotFloats);
            DirectIOVectorIndexInput.STATS.recordArenaAllocated();
        } else {
            DirectIOVectorIndexInput.STATS.recordArenaReused();
        }
        generation = GENERATIONS.incrementAndGet();
        return generation;
    }

    /**
     * Quiesces every range the current holder still has staged and empties the table, keeping the buffers.
     *
     * @return true when no read can still be writing into any of them
     */
    private boolean quiesceHeld() {
        boolean safe = true;
        for (int i = 0; i < count; i++) {
            final DirectIOVectorIndexInput.StagedRange range = table[i];
            table[i] = null;
            if (range != null && range.quiesce() == false) {
                safe = false;
            }
        }
        count = 0;
        return safe;
    }

    /**
     * Drops the arena and every buffer carved from or beside it. The charge stays outstanding on purpose:
     * these buffers may still be written by a read that would not quiesce, so they are genuinely still live
     * and the phantom reference returns their bytes once the collector proves otherwise.
     */
    private void retire() {
        arena = null;
        slotBuffers = null;
        slotFloats = null;
        wideBuffers = null;
        wideFloats = null;
        table = null;
        slots = 0;
        slotBytes = 0;
        blockSize = 0;
        DirectIOVectorIndexInput.STATS.recordArenaRetired();
    }

    /**
     * Gives the lease up on behalf of the holder named by {@code heldGeneration}, quiescing whatever it left
     * staged. Called from {@link DirectIOVectorIndexInput#close()}, where the channel is about to go: a read
     * still in flight on a closed channel fails rather than writing, and {@code quiesce} already treats a
     * failed read as safe, but quiescing first is what keeps that off the normal path.
     */
    void releaseIfHeldBy(final long heldGeneration) {
        if (heldGeneration < 0 || generation != heldGeneration) {
            return;
        }
        if (quiesceHeld() == false && arena != null) {
            retire();
        }
        generation = -1;
    }

    /** The range of the current burst covering {@code [absolute, absolute + needed)}, or null. */
    DirectIOVectorIndexInput.StagedRange covering(final long absolute, final int needed) {
        for (int i = 0; i < count; i++) {
            final DirectIOVectorIndexInput.StagedRange range = table[i];
            if (range != null && range.covers(absolute, needed)) {
                return range;
            }
        }
        return null;
    }

    /** Ranges the current burst has staged. The ring stands down while this is non-zero. */
    int count() {
        return count;
    }

    /**
     * Range {@code i} of the current burst's table, or null where the slot is empty. For span fill (F5),
     * which walks the burst's spans rather than looking one range up by the bytes it wants.
     */
    DirectIOVectorIndexInput.StagedRange rangeAt(final int i) {
        return table == null || i < 0 || i >= count ? null : table[i];
    }

    /** Whether the burst table is full, i.e. this burst may stage nothing more. */
    boolean isFull() {
        return count == slots;
    }

    /**
     * The buffer the next range of {@code alignedLength} bytes must be read into: this burst's next arena
     * slot when the range fits in one, else that slot's wide buffer, grown on demand.
     *
     * @return null when the direct-memory budget refused a wide buffer
     */
    ByteBuffer bufferFor(final int alignedLength) {
        if (alignedLength <= slotBytes) {
            return slotBuffers[count];
        }
        if (wideBuffers[count] == null || wideBuffers[count].capacity() < alignedLength) {
            if (DirectIOVectorIndexInput.fitsBudget(alignedLength, blockSize) == false) {
                return null;
            }
            // D3d: and the wide buffers' own sub-budget, because these are the allocations a lease keeps for
            // the life of its arena. Declining costs this range its staging and nothing else -- the bytes
            // are then served by a blocking read, the same fallback the overall budget's decline takes.
            final ByteBuffer wide = DirectIOVectorIndexInput.allocateWideBuffer(alignedLength, blockSize);
            if (wide == null) {
                return null;
            }
            // Dropping the old buffer here is what returns its bytes: both charges hang off the one
            // ByteBuffer, so the reclamation that returns it to the overall budget returns it to this one.
            wideBuffers[count] = wide;
            wideFloats[count] = wideBuffers[count].asFloatBuffer();
        }
        return wideBuffers[count];
    }

    /** The float view of {@link #bufferFor}'s buffer, made once with it so a read decodes allocation free. */
    FloatBuffer floatsFor(final int alignedLength) {
        return alignedLength <= slotBytes ? slotFloats[count] : wideFloats[count];
    }

    /** Records a submitted range in the burst table, in the slot {@link #bufferFor} just answered for. */
    void record(final DirectIOVectorIndexInput.StagedRange range) {
        table[count] = range;
        count++;
    }

    /**
     * This thread's blocking-read window, of {@code bufferSize} bytes aligned to {@code blockSize}.
     *
     * <p><b>Never declined by the budget.</b> This is the path every other decline falls back <em>to</em>;
     * refusing it would leave a read with nothing to read with, which is a failed query rather than a
     * bounded one. It is one buffer per thread — 8 KiB at the shipped default — where before D3c it was one
     * per input, so the blocking channel's footprint stopped scaling with lingering per-query slices too.
     */
    ByteBuffer window(final int bufferSize, final int blockSize) {
        if (window == null || windowBytes != bufferSize || windowBlockSize != blockSize) {
            window = DirectIOVectorIndexInput.trackedAllocate(bufferSize, blockSize);
            windowFloats = window.asFloatBuffer();
            windowBytes = bufferSize;
            windowBlockSize = blockSize;
            windowOwner = -1;
            windowStart = -1;
            windowValid = 0;
        }
        return window;
    }

    /** The float view of {@link #window}. Only valid after a {@code window(..)} call in the same read. */
    FloatBuffer windowFloats() {
        return windowFloats;
    }

    /**
     * The position of {@code [absolute, absolute + needed)} inside the window, or {@code -1} when the window
     * does not hold those bytes <em>for this input</em>. See {@link #windowOwner} for why the owner and not
     * the generation.
     */
    int windowPosition(final long ownerId, final long absolute, final int needed) {
        if (windowOwner != ownerId || windowStart < 0 || absolute < windowStart || absolute + needed > windowStart + windowValid) {
            return -1;
        }
        return Math.toIntExact(absolute - windowStart);
    }

    /** Bytes of the window that are valid, after {@link #windowPosition} answered non-negative. */
    int windowValid() {
        return windowValid;
    }

    /** Records what a blocking read just put in the window, and who read it. */
    void windowFilled(final long ownerId, final long alignedStart, final int valid) {
        windowOwner = ownerId;
        windowStart = alignedStart;
        windowValid = valid;
    }

    /** The arena backing the burst table, or null when none is carved. Test and stats seam. */
    ByteBuffer arena() {
        return arena;
    }

    /**
     * Test seam: gives up this thread's lease and forgets its buffers, so that one test cannot inherit
     * another's arena. Quiesces first, because a test that left a read in flight must not have its buffer
     * collected under it.
     */
    static void resetForTesting() {
        final DirectIOStagingLease lease = LEASES.get();
        lease.quiesceHeld();
        lease.generation = -1;
        lease.arena = null;
        lease.slotBuffers = null;
        lease.slotFloats = null;
        lease.wideBuffers = null;
        lease.wideFloats = null;
        lease.table = null;
        lease.slots = 0;
        lease.slotBytes = 0;
        lease.blockSize = 0;
        lease.window = null;
        lease.windowFloats = null;
        lease.windowBytes = 0;
        lease.windowBlockSize = 0;
        lease.windowOwner = -1;
        lease.windowStart = -1;
        lease.windowValid = 0;
    }
}

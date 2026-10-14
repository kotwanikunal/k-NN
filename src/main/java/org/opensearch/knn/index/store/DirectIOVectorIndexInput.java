/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.store.IndexInput;
import org.opensearch.common.Nullable;
import org.opensearch.core.common.unit.ByteSizeValue;
import org.opensearch.knn.index.KNNSettings;

import java.io.EOFException;
import java.io.IOException;
import java.lang.ref.PhantomReference;
import java.lang.ref.Reference;
import java.lang.ref.ReferenceQueue;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.OpenOption;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.Locale;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.BooleanSupplier;

/**
 * An {@link IndexInput} that fetches with {@code O_DIRECT} and implements {@link #prefetch} by putting
 * block-aligned reads in flight on {@link DirectIOReadPool}.
 *
 * <p><b>This class exists to answer one question</b> — Phase 9b gate 2. The rescore seam reaches its queue
 * depth through a seam of its own (a staging interface driven by a query-layer scorer wrapper, removed in
 * phase 12 H5), because Lucene's {@code DirectIOIndexInput} has no {@code prefetch} member at all and
 * {@link IndexInput#prefetch} defaults to a no-op. A {@code Directory}-based design has no query-layer
 * wrapper to lean on, so it can only match the seam's numbers if an {@code IndexInput} can carry read-ahead
 * by itself. That is what this class tests, and it is deliberately not wired into the codec: gate 2 is about
 * the mechanism, and a failure of wiring and a failure of the mechanism must not be indistinguishable.
 *
 * <h2>The batch of ordinals arrives here already, through shipped code</h2>
 * Read ahead needs to know what is coming. On the mmap path it does, and not by accident:
 * {@code PrefetchableFlatVectorScorer.PrefetchableRandomVectorScorer#bulkScore} hands the whole batch to
 * {@code PrefetchableVectorValuesHelper#doPrefetch}, which finds the values' {@code HasIndexSlice#getSlice}
 * and calls {@code PrefetchHelper#prefetch}, which issues a burst of {@link #prefetch} calls <em>before</em>
 * the first {@code vectorValue} read of that batch. So the lookahead this class needs is delivered by the
 * production, default-on prefetch path ({@code knn.feature.prefetch.enabled}, default true) rather than by
 * anything a directory design would have to add.
 *
 * <p>Two properties of that burst shape this class, and neither is a detail:
 * <ul>
 *   <li>{@code PrefetchHelper} <b>sorts the ordinals ascending in place</b> and then <b>groups</b> them into
 *       ranges of up to 128 KB. So prefetch arrives in <em>file order</em> as byte <em>ranges</em>, while
 *       {@code vectorValue(ord)} consumes in the batch's original order, and one range can cover many
 *       vectors. The seam's consume-once FIFO ring, indexed by device-read count, cannot express that: the
 *       staging table here is keyed by <b>file range</b> and an entry survives being read.</li>
 *   <li>A burst has no explicit boundary. It is recovered instead: the first {@link #prefetch} to arrive
 *       after any read ends the previous burst and quiesces its table, which is exactly what
 *       the rescore seam's staging reader did with {@code abandonBatch()} on entry.</li>
 * </ul>
 *
 * <h2>Two things this class must not be, both load bearing</h2>
 * <ul>
 *   <li><b>Not a {@code MemorySegmentAccessInput}.</b> {@code Lucene99MemorySegmentFloatVectorScorer.create}
 *       tests exactly that interface and returns an empty {@code Optional} otherwise, at which point
 *       {@code Lucene99MemorySegmentFlatVectorsScorer} falls back to the scalar
 *       {@code DefaultFlatVectorScorer}. That fallback is what makes {@code vectorValue(ord)} — and so
 *       {@link #seek} plus {@link #readFloats} on this input — actually run. A SIMD scorer bound straight to
 *       a mapping would never call us.</li>
 *   <li><b>Not a {@code FilterIndexInput}.</b> Not for the reason first recorded here: the same
 *       {@code create} does call {@code FilterIndexInput.unwrapOnlyTest} first, but that method only
 *       unwraps classes registered through {@code TestSecrets}, whose setter Lucene restricts to its own
 *       test framework, so a <em>production</em> {@code FilterIndexInput} subclass is <b>not</b> seen
 *       through and would displace the SIMD scorer just as well (gate 3). The reason to extend
 *       {@link IndexInput} directly is the other one: {@code FilterIndexInput} delegates only
 *       {@code readByte} and {@code readBytes}, so it inherits {@code IndexInput}'s no-op
 *       {@link #prefetch} and {@code DataInput}'s per-{@code float} {@code readFloats} loop — a wrapper
 *       would have to override the very methods this class exists to implement.</li>
 * </ul>
 * The values object, by contrast, <em>keeps</em> {@code HasIndexSlice}, which is how the prefetch burst
 * above reaches us. That is strictly better placed than the seam, which had to hide {@code HasIndexSlice}
 * to keep the SIMD scorer off and therefore had to rebuild read-ahead in the query layer.
 *
 * <h2>Direct I/O is a property of this object, inherited</h2>
 * An {@code IOContext} does not survive the {@link #slice} Lucene uses: {@code OffHeapFloatVectorValues}
 * calls the three-argument overload, which has no context parameter. So "these bytes are fetched with
 * {@code O_DIRECT}" cannot be re-decided per slice and must be carried by the object.
 * {@link #slice} and {@link #clone} therefore return instances over the same
 * {@link Handle} by construction, and there is no code path by which a slice of this input becomes an mmap
 * read. Only the input that opened the handle closes it.
 */
@Log4j2
public final class DirectIOVectorIndexInput extends IndexInput {

    /** The shared file handle. One per {@link #open}; slices and clones borrow it and never close it. */
    private static final class Handle {
        private final Path path;
        private final FileChannel channel;
        private final int blockSize;
        private final long fileLength;
        /**
         * This file's identity in {@link DirectIOVectorCache}, built once per open rather than per lookup.
         * Deliberately a value and not this {@code Handle}'s identity — see that class's javadoc for why
         * two opens of one {@code .vec} have to share entries.
         */
        private final String fileId;

        private Handle(final Path path, final FileChannel channel, final int blockSize, final long fileLength) {
            this.path = path;
            this.channel = channel;
            this.blockSize = blockSize;
            this.fileLength = fileLength;
            this.fileId = DirectIOVectorCache.fileId(path, fileLength);
        }
    }

    /**
     * One block-aligned range staged by {@link #prefetch}, keyed by the range it holds rather than by a
     * position in a ring.
     *
     * <p>The claim is what makes a buffer reusable: {@code Future#cancel} completes the future while a
     * started read runs on, so a read abandoned before it began is skipped by the claim and one that has
     * begun is waited out.
     */
    static final class StagedRange implements Callable<Void> {
        private final FileChannel channel;
        private final Path path;
        private final ByteBuffer buffer;
        /** {@link #buffer}'s float view, made once with the buffer so that a read decodes allocation free. */
        private final FloatBuffer floats;
        private final long alignedStart;
        private final int alignedLength;
        /**
         * Where this read's completion is tallied. The burst path passes {@link Stats#stagedReads} /
         * {@link Stats#stagedBytes}; the read-ahead ring passes its own counters. The queue-depth counters
         * ({@link Stats#inFlightNow}) stay shared, because a read in flight is a read in flight whichever
         * channel issued it, and the R3 arms must be comparable.
         */
        private final AtomicLong readCounter;
        private final AtomicLong byteCounter;
        private final AtomicBoolean started = new AtomicBoolean();
        private Future<?> future;
        /** Bytes the read actually delivered; only meaningful once {@link #future} has completed. */
        private int valid;
        /**
         * Set when a bounded wait for this read gave up. From that point the range serves nothing — see
         * {@link #covers} — and its buffer is retired rather than reused, because the read it gave up on
         * may still be writing into it. Written by the one thread that owns this input and read by it;
         * {@code volatile} only so that a buffer handed back is not reused on a stale read of this flag.
         */
        private volatile boolean abandoned;
        /**
         * Whether this range has already been offered to span fill (M4). Set the first time a read consumes
         * the range, whatever the geometry then decides to admit, so the walk happens once per range rather
         * than once per vector read out of it. Written and read by the one thread that owns the burst.
         */
        private boolean spanFilled;

        private StagedRange(
            final FileChannel channel,
            final Path path,
            final ByteBuffer buffer,
            final FloatBuffer floats,
            final long alignedStart,
            final int alignedLength,
            final AtomicLong readCounter,
            final AtomicLong byteCounter
        ) {
            this.channel = channel;
            this.path = path;
            this.buffer = buffer;
            this.floats = floats;
            this.alignedStart = alignedStart;
            this.alignedLength = alignedLength;
            this.readCounter = readCounter;
            this.byteCounter = byteCounter;
        }

        @Override
        public Void call() throws IOException {
            if (started.compareAndSet(false, true) == false) {
                return null;
            }
            STATS.inFlightNow.incrementAndGet();
            STATS.recordInFlightPeak();
            try {
                buffer.clear().limit(alignedLength);
                valid = channel.read(buffer, alignedStart);
                readCounter.incrementAndGet();
                byteCounter.addAndGet(Math.max(valid, 0));
            } finally {
                STATS.inFlightNow.decrementAndGet();
            }
            return null;
        }

        /** @return true if this read will now never run, false if it is running or has already run */
        private boolean skipIfNotStarted() {
            return started.compareAndSet(false, true);
        }

        /**
         * Whether this range fully contains {@code [absolute, absolute + needed)} <em>and can still serve
         * it</em>. An abandoned range covers nothing: the check lives here rather than at the three call
         * sites so that the read path, the read-ahead ring and {@link #prefetch}'s dedupe all agree.
         */
        boolean covers(final long absolute, final int needed) {
            return abandoned == false && absolute >= alignedStart && absolute + needed <= alignedStart + alignedLength;
        }

        /**
         * Quiesces this range within {@link #STAGED_READ_WAIT_MILLIS} and reports whether its buffer is safe
         * to reuse. False means a read may still be writing into it, so the buffer — and, since D3c, the
         * whole arena it was carved from — must be retired rather than handed to the next burst.
         *
         * <p>Lives here rather than on the input because the thread taking a lease back is quiescing ranges
         * the <em>previous holder</em> staged, and that holder may be a different input that nothing holds a
         * reference to any more.
         */
        boolean quiesce() {
            if (abandoned) {
                return false;
            }
            if (skipIfNotStarted()) {
                // It will now never run, so nothing can write into the buffer.
                return true;
            }
            try {
                future.get(STAGED_READ_WAIT_MILLIS, TimeUnit.MILLISECONDS);
                return true;
            } catch (TimeoutException e) {
                abandoned = true;
                STATS.stagedWaitTimeouts.incrementAndGet();
                log.debug("Gave up after {} ms quiescing a Direct I/O read of {}", STAGED_READ_WAIT_MILLIS, path);
                return false;
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return false;
            } catch (ExecutionException e) {
                // The read failed, so it is not going to write anything either -- a closed channel is the
                // common cause, and the buffer is safe.
                log.debug("Abandoned a staged Direct I/O read of {}", path);
                return true;
            }
        }

        @Override
        public String toString() {
            return "StagedRange[" + path.getFileName() + "@" + alignedStart + "+" + alignedLength + "]";
        }
    }

    /**
     * Node-wide counters, for the gate-2 measurement and for any later wiring.
     *
     * <p>{@link #inFlightPeak} is the number gate 2 turns on: it is the measured queue depth this input
     * offers the device, so a prefetch that stages nothing shows up here as 1 and not as a latency riddle.
     *
     * <h2>The byte counters, and why they are not a duplicate of {@code /proc/diskstats}</h2>
     * 9e-2 priced the design's read amplification from the device side: {@code sectors_read} over the
     * queries of a block, divided by the vectors those queries rescored. That number is a sum of every
     * read the node made, so attributing it to this class was an inference (R60) rather than a
     * measurement — the graph, the quantized codes and the doc values are on the same device.
     * {@link #stagedBytes} + {@link #blockingBytes} over {@link #servedBytes} is the same ratio taken
     * inside the one object that issues the reads, so it can neither include another reader's traffic nor
     * miss any of this one's. Both byte figures are what the device <em>returned</em>, which equals the
     * block-aligned request except at end of file.
     *
     * <h2>The dispatch counters, and why they live here</h2>
     * {@link #routedOpens} and the four decline reasons are {@link KNNDirectIODirectory}'s, not this
     * class's. They are aggregated here because the directory is created <b>per segment per reader</b> by
     * the codec seam and is discarded with that reader, so its own instance counters are unreachable from
     * a REST API by construction — which phase 13's re-proof notes recorded as a gap. This object is the
     * one node-wide, cumulative place Direct I/O statistics already live, so one read of it answers both
     * "is the seam engaging at all" and "what is it doing once engaged". The directory still keeps its own
     * instance counters: they are what a test asserts on, and they are per segment, which the node-wide
     * sums cannot be.
     */
    public static final class Stats {
        private final AtomicLong routedOpens = new AtomicLong();
        private final AtomicLong declinedCompound = new AtomicLong();
        private final AtomicLong declinedChain = new AtomicLong();
        private final AtomicLong declinedMissing = new AtomicLong();
        private final AtomicLong declinedTooSmall = new AtomicLong();
        private final AtomicLong declinedOpenFailed = new AtomicLong();
        private final AtomicLong blockingReads = new AtomicLong();
        private final AtomicLong stagedReads = new AtomicLong();
        private final AtomicLong stagedHits = new AtomicLong();
        private final AtomicLong stageDeclined = new AtomicLong();
        private final AtomicLong prefetchCalls = new AtomicLong();
        private final AtomicLong inFlightNow = new AtomicLong();
        private final AtomicLong inFlightPeak = new AtomicLong();
        private final AtomicLong stagedBytes = new AtomicLong();
        private final AtomicLong blockingBytes = new AtomicLong();
        private final AtomicLong servedBytes = new AtomicLong();
        private final AtomicLong declinedSpanBytes = new AtomicLong();
        private final AtomicLong readaheadReads = new AtomicLong();
        private final AtomicLong readaheadBytes = new AtomicLong();
        private final AtomicLong readaheadHits = new AtomicLong();
        private final AtomicLong readaheadDeclined = new AtomicLong();
        private final AtomicLong readaheadArmed = new AtomicLong();
        private final AtomicLong stagedWaitTimeouts = new AtomicLong();
        private final AtomicLong queuePeak = new AtomicLong();
        private final AtomicLong poolRebuilds = new AtomicLong();
        private final AtomicLong directMemoryLive = new AtomicLong();
        private final AtomicLong directMemoryPeak = new AtomicLong();
        private final AtomicLong declinedBudget = new AtomicLong();
        private final AtomicLong arenasAllocated = new AtomicLong();
        private final AtomicLong arenasReused = new AtomicLong();
        private final AtomicLong arenasRetired = new AtomicLong();
        private final AtomicLong foreignThreadReads = new AtomicLong();
        private final AtomicLong fallbackReads = new AtomicLong();
        private final AtomicLong fallbackBytes = new AtomicLong();
        private final AtomicLong switchedToMmap = new AtomicLong();
        private final AtomicLong switchedToDirect = new AtomicLong();
        private final AtomicLong wideBufferLive = new AtomicLong();
        private final AtomicLong wideBufferPeak = new AtomicLong();
        private final AtomicLong declinedWideBudget = new AtomicLong();
        private final AtomicLong readAroundReads = new AtomicLong();

        private void recordInFlightPeak() {
            final long now = inFlightNow.get();
            long peak = inFlightPeak.get();
            while (now > peak && inFlightPeak.compareAndSet(peak, now) == false) {
                peak = inFlightPeak.get();
            }
        }

        /**
         * Raises the queue-occupancy high-water mark. Sampled by {@link DirectIOReadPool#sampleQueueDepth}
         * immediately after a submit, so that the derived default capacity can be validated against what a
         * real burst offers rather than against its own formula.
         */
        void recordQueuePeak(final int occupancy) {
            long peak = queuePeak.get();
            while (occupancy > peak && queuePeak.compareAndSet(peak, occupancy) == false) {
                peak = queuePeak.get();
            }
        }

        /** Counts one pool swap, i.e. one capacity change applied without a restart. */
        void recordPoolRebuild() {
            poolRebuilds.incrementAndGet();
        }

        /** Charges {@code bytes} of direct buffer to the budget and raises the peak. */
        void recordDirectAllocation(final long bytes) {
            final long now = directMemoryLive.addAndGet(bytes);
            long peak = directMemoryPeak.get();
            while (now > peak && directMemoryPeak.compareAndSet(peak, now) == false) {
                peak = directMemoryPeak.get();
            }
        }

        /** Returns {@code bytes} to the budget, when the buffer holding them has been reclaimed. */
        void recordDirectFree(final long bytes) {
            directMemoryLive.addAndGet(-bytes);
        }

        /** Charges {@code bytes} of wide-group buffer to its sub-budget and raises that peak. D3d, O2. */
        void recordWideAllocation(final long bytes) {
            raiseWidePeak(wideBufferLive.addAndGet(bytes));
        }

        /**
         * Charges {@code bytes} of wide-group buffer only if the total stays at or under {@code cap}.
         *
         * <p>A compare-and-set loop rather than a read followed by an add, because a read-then-add is a
         * check-then-act that every thread passes at once: MEASURED by 16-I at 64 threads against a 16 MiB
         * cap, the advisory form reached 18,788,213 B, a 12% overshoot. It is the same protocol
         * {@link DirectIOVectorCache}'s pending-allocation count uses to stop concurrent fills overshooting
         * the slab target (N3), and for the same reason.
         *
         * @return true when the charge was taken, false when it would have exceeded {@code cap}
         */
        boolean reserveWide(final long bytes, final long cap) {
            while (true) {
                final long live = wideBufferLive.get();
                if (live + bytes > cap) {
                    return false;
                }
                if (wideBufferLive.compareAndSet(live, live + bytes)) {
                    raiseWidePeak(live + bytes);
                    return true;
                }
            }
        }

        private void raiseWidePeak(final long now) {
            long peak = wideBufferPeak.get();
            while (now > peak && wideBufferPeak.compareAndSet(peak, now) == false) {
                peak = wideBufferPeak.get();
            }
        }

        /**
         * Returns {@code bytes} of wide-group buffer, when the buffer holding them has been reclaimed.
         *
         * <p>Floored at zero. A gauge that a reservation can bound must not be driveable negative by a
         * return that is counted twice, because a negative gauge is room the cap would then hand out.
         */
        void recordWideFree(final long bytes) {
            wideBufferLive.updateAndGet(live -> Math.max(0L, live - bytes));
        }

        /**
         * Test seam: zeroes the wide-group gauge. The counter is node wide and its charges are returned by
         * reachability, so without this one test's buffers are still charged while the next test measures --
         * and the next test's claim is about an absolute figure against a cap. Safe because
         * {@link #recordWideFree} is floored: a reclamation that arrives after a reset leaves it at zero.
         */
        void resetWideGaugeForTesting() {
            wideBufferLive.set(0L);
            wideBufferPeak.set(0L);
        }

        /** Counts one range not staged because the wide-group sub-budget was reached. */
        void recordDeclinedWideBudget() {
            declinedWideBudget.incrementAndGet();
        }

        /** Counts one staged miss read widened by O5's read-around. */
        void recordReadAround() {
            readAroundReads.incrementAndGet();
        }

        /**
         * Staged miss reads widened by {@code knn.direct_io.rescore.cache.read_around_bytes} (O5). Against
         * {@code staged_reads} this says how much of the staging was widened; it is 0 at the setting's
         * {@code 0b}, 0 once the store's target no longer leaves room for the whole file, and 0 with span
         * fill off, which are the three gates.
         */
        public long readAroundReads() {
            return readAroundReads.get();
        }

        /** Wide-group staging buffers held right now, in bytes. */
        public long wideBufferLive() {
            return wideBufferLive.get();
        }

        /** The high-water mark of {@link #wideBufferLive()}. */
        public long wideBufferPeak() {
            return wideBufferPeak.get();
        }

        /** Ranges not staged because the wide-group sub-budget was reached. */
        public long declinedWideBudget() {
            return declinedWideBudget.get();
        }

        /** Counts one prefetch declined because the direct-memory budget was reached. */
        void recordDeclinedBudget() {
            declinedBudget.incrementAndGet();
        }

        /** Counts one staging arena carved by a {@link DirectIOStagingLease}. */
        void recordArenaAllocated() {
            arenasAllocated.incrementAndGet();
        }

        /** Counts one burst that took its thread's existing arena back instead of carving one. */
        void recordArenaReused() {
            arenasReused.incrementAndGet();
        }

        /** Counts one arena dropped rather than reused — see {@link #arenasRetired()} for the two reasons. */
        void recordArenaRetired() {
            arenasRetired.incrementAndGet();
        }

        /** Counts one read that skipped the staged table because it was not on the lease's owner thread. */
        void recordForeignThreadRead() {
            foreignThreadReads.incrementAndGet();
        }

        /** Counts one read an already-open routed input served through its delegate, with the switch off. O1, D2. */
        void recordFallbackRead(final long bytes) {
            fallbackReads.incrementAndGet();
            fallbackBytes.addAndGet(bytes);
        }

        /** Counts one already-open input changing read path at a prefetch burst boundary. O1, D2. */
        void recordSwitch(final boolean toDirectIO) {
            (toDirectIO ? switchedToDirect : switchedToMmap).incrementAndGet();
        }

        /**
         * Direct-buffer bytes the Direct I/O read path holds right now: staging arenas, read-ahead arenas,
         * the per-slot buffers a wide prefetch group needs, and one blocking window per input. Charged on
         * allocation and returned when the buffer is reclaimed — see
         * {@link DirectIOVectorIndexInput#reclaim()} for why that is reachability and not
         * {@link DirectIOVectorIndexInput#close()}.
         */
        public long directMemoryLive() {
            return clampToWall(directMemoryLive.get());
        }

        /** The high-water mark of {@link #directMemoryLive()} since node start. */
        public long directMemoryPeak() {
            return clampToWall(directMemoryPeak.get());
        }

        /**
         * Caps a reported figure at the JVM's direct-memory wall, because a number above it is not a fact
         * about this node.
         *
         * <p>MEASURED on the node 2026-10-06 with the budget off: {@code peak_bytes} read <b>7,457,394,721</b>
         * — and later 9.0 GB and 28 GB — against {@code -XX:MaxDirectMemorySize=2g}. The raw counter is
         * "charged minus proven reclaimed", and with the budget off nothing drained the phantom queue outside
         * a stats read, so it ratcheted past the physical limit while the JDK's own
         * {@code java.nio.Bits.reserveMemory} kept freeing buffers behind it. D3c fixes the drain (it now runs
         * on every {@link DirectIOVectorIndexInput#trackedAllocate}) and clamps the report, which are two
         * different fixes for the same defect: the drain makes the number smaller, the clamp makes it
         * impossible for it to be absurd.
         */
        private static long clampToWall(final long bytes) {
            final long wall = KNNSettings.KNN_DIRECT_IO_DIRECT_MEMORY_WALL_BYTES;
            return wall > 0 ? Math.min(bytes, wall) : bytes;
        }

        /**
         * Staging arenas carved since node start, one per thread that has run a prefetch burst — plus one per
         * {@link #arenasRetired()}. <b>This is the number D3c exists to flatten.</b> Before it, every
         * per-query {@code OffHeapFloatVectorValues} slice carved its own, so arenas were allocated at
         * roughly {@code QPS x segments} per second (9-13 MiB/s of direct buffer on the node) and the gauge
         * measured churn awaiting a collection rather than memory in use.
         */
        public long arenasAllocated() {
            return arenasAllocated.get();
        }

        /**
         * Bursts that took their thread's existing arena back rather than carving one. On a healthy node this
         * is almost every burst: {@code arenas_reused} should be orders of magnitude above
         * {@code arenas_allocated}, and the ratio is the whole claim of the lease model expressed as one
         * division.
         */
        public long arenasReused() {
            return arenasReused.get();
        }

        /**
         * Arenas dropped rather than reused, for one of two reasons: a range the previous burst left staged
         * would not quiesce within {@link #STAGED_READ_WAIT_MILLIS} (so a pool read may still write into its
         * buffers — see {@code staged_wait_timeouts}, which moves with it), or the next burst wanted a
         * different geometry. The first is pathology and the second is a configuration change, so on a
         * settled node this stays flat.
         */
        public long arenasRetired() {
            return arenasRetired.get();
        }

        /**
         * Reads that skipped the staged table because they arrived on a thread other than the one that holds
         * the lease, and were served by a blocking read instead. <b>~0 on a healthy node</b>, because the
         * rescore path prefetches and reads a burst on one thread (FINDINGS.md D3c(a)). Non-zero means some
         * caller hands a prefetched input to another thread, which costs that read its read-ahead and
         * nothing else — the bytes are identical either way, because the owner-confinement rule is what
         * makes the lease's generation check race free rather than an optimisation.
         */
        public long foreignThreadReads() {
            return foreignThreadReads.get();
        }

        /**
         * Reads an already-open routed input served through its delegate because
         * {@code knn.direct_io.rescore.enabled} was off at the last prefetch burst boundary (O1, D2).
         *
         * <p>The counter an operator watches to see a rollback take effect <b>with no reopen</b>: it rises
         * while the switch is off and stops rising when it goes back on, and it rises while
         * {@code staged_reads}, {@code blocking_reads} and {@code dispatch.routed_opens} all stand still,
         * which is what says no reader was rebuilt to get there.
         */
        public long fallbackReads() {
            return fallbackReads.get();
        }

        /** Bytes served through the delegate input by an already-open routed input. O1. */
        public long fallbackBytes() {
            return fallbackBytes.get();
        }

        /** Times an already-open input changed over to its delegate at a burst boundary. O1. */
        public long switchedToMmap() {
            return switchedToMmap.get();
        }

        /** Times an already-open input changed back to {@code O_DIRECT} at a burst boundary. O1. */
        public long switchedToDirect() {
            return switchedToDirect.get();
        }

        /**
         * Prefetches declined because {@code knn.direct_io.direct_memory_budget} was reached. Each one costs
         * read-ahead for one burst and leaves the query correct, because the bytes are then served by
         * blocking reads — the same fallback a queue-capacity decline takes. Non-zero means the node is
         * holding as much direct memory for vector reads as it has been allowed to.
         */
        public long declinedBudget() {
            return declinedBudget.get();
        }

        /**
         * The high-water mark of reads waiting behind the pool. Against
         * {@code knn.direct_io.rescore.prefetch.queue_capacity} this says whether the queue ever came near
         * filling, which is the question the node's 0-declines-at-both-1024-and-4096 result left open.
         */
        public long queuePeak() {
            return queuePeak.get();
        }

        /**
         * Times the pool was rebuilt because {@code queue_capacity} changed. {@code 0} on a node that has
         * never retuned it; each increment is one live capacity change that actually took effect.
         */
        public long poolRebuilds() {
            return poolRebuilds.get();
        }

        /**
         * Counts one {@code .vec} open the dispatch rule routed to {@code O_DIRECT}. Called by
         * {@link KNNDirectIODirectory}, which owns the rule; see the class javadoc for why the
         * node-wide tally is here.
         */
        void recordRoutedOpen() {
            routedOpens.incrementAndGet();
        }

        /** Counts one decline, under one of the five reasons the dispatch rule distinguishes. */
        void recordDecline(final DeclineReason reason) {
            switch (reason) {
                case COMPOUND -> declinedCompound.incrementAndGet();
                case CHAIN -> declinedChain.incrementAndGet();
                case MISSING -> declinedMissing.incrementAndGet();
                case TOO_SMALL -> declinedTooSmall.incrementAndGet();
                case OPEN_FAILED -> declinedOpenFailed.incrementAndGet();
            }
        }

        /** Node-wide {@code .vec} opens served with {@code O_DIRECT}. */
        public long routedOpens() {
            return routedOpens.get();
        }

        /**
         * Declines because the segment is <b>compound</b>, so the codec was handed a
         * {@code CompoundDirectory} and the {@code .vec} is an entry in a {@code .cfs} rather than a file.
         *
         * <p>Counted apart from {@link #declinedChain()} since phase 15, and the distinction is the whole
         * point: at the shipped defaults <b>every flushed segment is compound</b>
         * ({@code index.use_compound_file} defaults to {@code true},
         * {@code EngineConfig.java:262-266}), so a healthy node accumulates this counter steadily and only
         * merged segments ever route. Folded into {@code declined_chain} it looked like an unsupported
         * substrate — on the node a 27-segment reopen added +108 and a 28-segment one +234 — and an
         * operator could not tell "the layer is unknown" from "the segment is compound" at a glance.
         */
        public long declinedCompound() {
            return declinedCompound.get();
        }

        /** Declines because no {@link java.nio.file.Path} could be resolved through the delegate chain. */
        public long declinedChain() {
            return declinedChain.get();
        }

        /** Declines because the resolved path did not exist or could not be sized. */
        public long declinedMissing() {
            return declinedMissing.get();
        }

        /** Declines because the file was below {@code knn.direct_io.min_file_size}. */
        public long declinedTooSmall() {
            return declinedTooSmall.get();
        }

        /** Declines because the {@code O_DIRECT} open itself failed. */
        public long declinedOpenFailed() {
            return declinedOpenFailed.get();
        }

        /** Every decline, whatever the reason. */
        public long declinedOpens() {
            return declinedCompound.get() + declinedChain.get() + declinedMissing.get() + declinedTooSmall.get() + declinedOpenFailed.get();
        }

        /** Device reads issued on the calling thread, i.e. with no queue depth behind them. */
        public long blockingReads() {
            return blockingReads.get();
        }

        /** Device reads issued on {@link DirectIOReadPool}, i.e. the ones that can overlap. */
        public long stagedReads() {
            return stagedReads.get();
        }

        /** Reads served out of a staged range rather than by a blocking read. */
        public long stagedHits() {
            return stagedHits.get();
        }

        /** Prefetch ranges refused, because the burst filled the table or the range was too large. */
        public long stageDeclined() {
            return stageDeclined.get();
        }

        public long prefetchCalls() {
            return prefetchCalls.get();
        }

        /** The high-water mark of concurrently in-flight device reads: the queue depth offered. */
        public long inFlightPeak() {
            return inFlightPeak.get();
        }

        /** Bytes the device returned for staged reads, i.e. the read-ahead channel's traffic. */
        public long stagedBytes() {
            return stagedBytes.get();
        }

        /** Bytes the device returned for blocking reads, i.e. the traffic of everything not staged. */
        public long blockingBytes() {
            return blockingBytes.get();
        }

        /** Bytes handed to callers out of either route — the denominator of read amplification. */
        public long servedBytes() {
            return servedBytes.get();
        }

        /** Span bytes asked for by prefetch calls that were declined, i.e. what the waste guard refused. */
        public long declinedSpanBytes() {
            return declinedSpanBytes.get();
        }

        /**
         * Device reads the sequential read-ahead ring issued. The no-waste proof's number: on a random
         * (rescore) workload detection must never arm, so this must stay 0.
         */
        public long readaheadReads() {
            return readaheadReads.get();
        }

        /** Bytes the device returned for read-ahead reads. */
        public long readaheadBytes() {
            return readaheadBytes.get();
        }

        /** Reads served out of a read-ahead ring slot rather than by a blocking read. */
        public long readaheadHits() {
            return readaheadHits.get();
        }

        /** Ring refills refused because there was no pool or the pool rejected the submit. */
        public long readaheadDeclined() {
            return readaheadDeclined.get();
        }

        /** Times detection crossed the trigger and the ring armed. The "did it ever engage" number. */
        public long readaheadArmed() {
            return readaheadArmed.get();
        }

        /**
         * Times a read gave up waiting for a staged or read-ahead read and issued it itself. <b>Zero on a
         * healthy node.</b> Non-zero means the pool could not retire a read within
         * {@link #STAGED_READ_WAIT_MILLIS}, i.e. it is saturated or the device is not answering — the
         * condition that used to hold a search thread with no bound at all. Each event costs one extra
         * device read and leaves the query correct, so this is a saturation signal rather than an error
         * count.
         */
        public long stagedWaitTimeouts() {
            return stagedWaitTimeouts.get();
        }

        public void reset() {
            routedOpens.set(0);
            declinedCompound.set(0);
            declinedChain.set(0);
            declinedMissing.set(0);
            declinedTooSmall.set(0);
            declinedOpenFailed.set(0);
            blockingReads.set(0);
            stagedReads.set(0);
            stagedHits.set(0);
            stageDeclined.set(0);
            prefetchCalls.set(0);
            inFlightPeak.set(0);
            stagedBytes.set(0);
            blockingBytes.set(0);
            servedBytes.set(0);
            declinedSpanBytes.set(0);
            readaheadReads.set(0);
            readaheadBytes.set(0);
            readaheadHits.set(0);
            readaheadDeclined.set(0);
            readaheadArmed.set(0);
            stagedWaitTimeouts.set(0);
            queuePeak.set(0);
            poolRebuilds.set(0);
            declinedBudget.set(0);
            directMemoryPeak.set(0);
            arenasAllocated.set(0);
            arenasReused.set(0);
            arenasRetired.set(0);
            foreignThreadReads.set(0);
            fallbackReads.set(0);
            fallbackBytes.set(0);
            switchedToMmap.set(0);
            switchedToDirect.set(0);
            wideBufferPeak.set(0);
            declinedWideBudget.set(0);
            readAroundReads.set(0);
            // wideBufferLive is NOT reset here, for the reason directMemoryLive is not: it is a live gauge.
            // resetWideGaugeForTesting is the seam a test that measures an absolute against the cap uses.
            // directMemoryLive is NOT reset: it is a live gauge, not a counter, and zeroing it while buffers
            // are still held would make the budget let through exactly as much again.
        }

        @Override
        public String toString() {
            // Read-ahead bytes join the numerator because they are traffic this object caused, the same as
            // the staged and blocking channels; a window fetched and never served is amplification.
            final long read = stagedBytes.get() + blockingBytes.get() + readaheadBytes.get();
            final long served = servedBytes.get();
            return String.format(
                Locale.ROOT,
                "routedOpens=%d declinedCompound=%d declinedChain=%d declinedMissing=%d declinedTooSmall=%d declinedOpenFailed=%d "
                    + "blockingReads=%d stagedReads=%d stagedHits=%d stageDeclined=%d prefetchCalls=%d inFlightPeak=%d "
                    + "stagedBytes=%d blockingBytes=%d servedBytes=%d declinedSpanBytes=%d "
                    + "readaheadReads=%d readaheadBytes=%d readaheadHits=%d readaheadDeclined=%d readaheadArmed=%d "
                    + "stagedWaitTimeouts=%d queuePeak=%d poolRebuilds=%d "
                    + "directMemoryLive=%d directMemoryPeak=%d declinedBudget=%d "
                    + "arenasAllocated=%d arenasReused=%d arenasRetired=%d foreignThreadReads=%d "
                    + "fallbackReads=%d fallbackBytes=%d switchedToMmap=%d switchedToDirect=%d "
                    + "wideBufferLive=%d wideBufferPeak=%d declinedWideBudget=%d readAroundReads=%d amplification=%.3f",
                routedOpens.get(),
                declinedCompound.get(),
                declinedChain.get(),
                declinedMissing.get(),
                declinedTooSmall.get(),
                declinedOpenFailed.get(),
                blockingReads.get(),
                stagedReads.get(),
                stagedHits.get(),
                stageDeclined.get(),
                prefetchCalls.get(),
                inFlightPeak.get(),
                stagedBytes.get(),
                blockingBytes.get(),
                served,
                declinedSpanBytes.get(),
                readaheadReads.get(),
                readaheadBytes.get(),
                readaheadHits.get(),
                readaheadDeclined.get(),
                readaheadArmed.get(),
                stagedWaitTimeouts.get(),
                queuePeak.get(),
                poolRebuilds.get(),
                directMemoryLive(),
                directMemoryPeak(),
                declinedBudget.get(),
                arenasAllocated.get(),
                arenasReused.get(),
                arenasRetired.get(),
                foreignThreadReads.get(),
                fallbackReads.get(),
                fallbackBytes.get(),
                switchedToMmap.get(),
                switchedToDirect.get(),
                wideBufferLive.get(),
                wideBufferPeak.get(),
                declinedWideBudget.get(),
                readAroundReads.get(),
                served == 0 ? 0.0 : (double) read / served
            );
        }
    }

    /** The one counter set, because the pool it counts is node wide too. */
    /** Why the dispatch rule declined a {@code .vec} it matched on name and setting. */
    public enum DeclineReason {
        /**
         * The segment is compound, so the {@code Directory} the codec was handed is a
         * {@code CompoundDirectory} and the {@code .vec} has no file of its own: condition 3, taken before
         * the chain walk because it is the one case that is <b>expected</b> rather than diagnostic.
         */
        COMPOUND,
        /** No {@link Path} resolvable through the delegate chain: condition 3. */
        CHAIN,
        /** The resolved path does not exist, or could not be sized: condition 4. */
        MISSING,
        /** Below {@code knn.direct_io.min_file_size}: condition 5. */
        TOO_SMALL,
        /** The {@code O_DIRECT} open itself failed: condition 6. */
        OPEN_FAILED
    }

    public static final Stats STATS = new Stats();

    /**
     * Where a charged buffer announces that it has been reclaimed. See {@link #reclaim()} for why the
     * accounting hangs on reachability rather than on {@link #close()}, and why it is drained on the calling
     * thread rather than by a {@link java.lang.ref.Cleaner}.
     */
    private static final ReferenceQueue<ByteBuffer> RECLAIMED = new ReferenceQueue<>();

    /**
     * A charged buffer's entry in {@link #RECLAIMED}: a phantom reference that remembers what its buffer was
     * charged, so the charge can be returned once the buffer is unreachable. Held in {@link #CHARGES} until
     * it is drained, because a {@link PhantomReference} that becomes unreachable itself is never enqueued.
     */
    private static final class Charge extends PhantomReference<ByteBuffer> {
        private final int bytes;
        /** Whether this buffer is also charged to the wide-group sub-budget. D3d, O2. */
        private final boolean wide;

        private Charge(final ByteBuffer buffer, final int bytes, final boolean wide) {
            super(buffer, RECLAIMED);
            this.bytes = bytes;
            this.wide = wide;
        }
    }

    /** One entry per charged buffer that has not yet been reclaimed — bounded by what the budget bounds. */
    private static final Set<Charge> CHARGES = ConcurrentHashMap.newKeySet();

    /**
     * Default ranges a single prefetch burst may hold in flight: the size of Lucene's bulk batch, because
     * a table smaller than the burst declines the burst's tail. Only the fallback since 9d-5 —
     * {@link #open(Path)} reads {@code knn.direct_io.rescore.prefetch.staged_ranges}, whose default this is.
     */
    static final int DEFAULT_MAX_STAGED_RANGES = 64;

    /**
     * Largest single prefetch range this input will stage, in bytes. {@code PrefetchHelper} groups up to
     * 128 KB, so this is that plus a block for the alignment slack at each end; a larger range falls back to
     * blocking reads rather than to a buffer nobody budgeted for.
     *
     * <p>Since 9d-5 this is only the <em>default</em>: {@link #open(Path)} reads
     * {@code knn.direct_io.rescore.prefetch.max_span_bytes}, whose own default is this value, so that the
     * bound can be lowered without a rebuild. Lowering it is the waste guard — see {@link #prefetch}.
     */
    static final int DEFAULT_MAX_STAGED_RANGE_BYTES = 128 * 1024 + 4096;

    /**
     * How long a read may wait for a staged read before giving up and issuing it itself.
     *
     * <p>A constant rather than a setting, because there is no value an operator should want to tune: this
     * is not a latency target, it is the point past which waiting is certainly worse than reading. The
     * derivation — a staged read on this stack costs a device read at the pool's queue depth, which is
     * tens of milliseconds at the deepest queue the bounded pool allows ({@code QUEUED_READS_PER_THREAD}
     * reads of ~30 ms), so the worst healthy wait is a few hundred milliseconds. One second is above that
     * and two orders of magnitude above the measured per-query p99 of 11 ms, so it fires only on pathology;
     * and when it does fire, the fallback is one {@code pread} of a few kilobytes.
     */
    static final long STAGED_READ_WAIT_MILLIS = 1000L;

    /**
     * What fraction of {@code knn.direct_io.direct_memory_budget} the wide-group buffers may hold when their
     * own setting is at its {@code 0b} default: one in {@value}. See {@link #allocateWideBuffer}.
     */
    static final int WIDE_BUFFER_BUDGET_SHARE = 4;

    /** Hands out {@link #inputId}. Node wide, because a lease is per thread and meets inputs from anywhere. */
    private static final AtomicLong INPUT_IDS = new AtomicLong();

    private final Handle handle;
    /**
     * The delegate's own input over the same bytes, kept open beside the {@code O_DIRECT} channel, or
     * {@code null} when the directory could not open one.
     *
     * <p><b>This is O1, D2's rollback half.</b> {@code knn.direct_io.rescore.enabled} has to reach segments
     * that are already open, because an operator who turns Direct I/O off wants the device traffic to stop
     * now, not after the next reader opens. The decision cannot be re-taken at the {@code openInput}
     * {@link KNNDirectIODirectory} describes, because that open has already happened; so the input carries
     * both routes and chooses between them at every prefetch burst boundary. Its coordinates are this
     * input's: a slice of this input holds the matching slice of the delegate's, so {@link #getFilePointer}
     * addresses both.
     *
     * <p>It is <b>only</b> opened when the switch was on at {@code openInput} time, so the cost of carrying
     * it is one extra file handle per routed {@code .vec} per reader and nothing on a node with the feature
     * off, which routes no file and so opens no second input. The asymmetry that leaves is deliberate and is
     * 16-I's: turning the switch off reaches open segments, turning it on from a cold start reaches only
     * segments opened after it, because the alternative is opening an {@code O_DIRECT} channel for every
     * {@code .vec} on a node that will never use one.
     */
    @Nullable
    private final IndexInput mmapFallback;
    /**
     * Condition 2 of the dispatch rule, re-read once per prefetch burst. It is the supplier
     * {@link KNNDirectIODirectory} dispatched on rather than a second read of the setting, so that the
     * directory and the inputs it produced can never disagree about whether the feature is on -- which they
     * would under the directory's test constructor, and which would be a silent mmap read in a test that
     * believes it is measuring {@code O_DIRECT}.
     */
    private final BooleanSupplier routeEnabled;
    /** Absolute file offset of this input's byte zero, so a slice needs no other translation. */
    private final long offset;
    private final long length;
    /** Only the input returned by {@link #open} closes the shared channel. */
    private final boolean ownsHandle;
    private final int bufferSize;
    private final int maxStagedRanges;
    private final int maxStagedRangeBytes;
    /** Windows the read-ahead ring keeps in flight. 0 ⇒ the ring is disabled and this is exact passthrough. */
    private final int readaheadWindows;
    /**
     * Bytes per read-ahead window — the ring's request size, <b>decoupled from {@link #bufferSize}</b>.
     *
     * <p>Phase 13 audit FIX 2, and the largest measured win in that audit. {@code bufferSize} was one
     * number serving two access shapes with opposite needs ({@code bench/e1b.out}): the sequential scan
     * wants a big request and the random rescore path wants a small one.
     *
     * <table>
     *   <caption>measured on a 732 MiB ascending scan of a real 1M x 768 fp32 {@code .vec}</caption>
     *   <tr><th>{@code bufferSize}</th><th>sequential MiB/s at W=8</th><th>random amplification, 16 threads</th></tr>
     *   <tr><td>8 KiB</td><td>49.7</td><td><b>2.667</b></td></tr>
     *   <tr><td>64 KiB</td><td><b>371.2</b></td><td>~21 (inferred)</td></tr>
     *   <tr><td>256 KiB</td><td>370.6</td><td><b>85.3</b></td></tr>
     * </table>
     *
     * <p>So the fix is not to raise {@code bufferSize} — at 256 KiB the rescore path's device traffic goes
     * up 32x and its qps collapses from 59.6 to 10.0. It is to give the ring its own size while
     * {@link #blockingSource} and the burst table keep {@code bufferSize} at 8 KiB. The ring already
     * allocated separately and already submitted with an explicit length, so this is one field reaching
     * three places in {@link #refillReadahead}.
     *
     * <p>Costs {@code readaheadWindows * this} of direct buffers per input, allocated lazily on the
     * <em>first arm</em> — so the default is 512 KiB rather than 64 KiB for an input that runs
     * sequentially, and still zero for one that does not. The rescore path never arms (the ring stands
     * down while a burst is live, and random access never reaches the trigger), so it pays nothing.
     */
    private final int readaheadWindowBytes;
    /** Consecutive gapless-forward blocking reads before the ring arms. */
    private final int readaheadTrigger;

    private long filePointer;
    private boolean closed;
    /**
     * Whether this input is reading through {@code O_DIRECT} right now rather than through
     * {@link #mmapFallback}. Re-decided once per burst by {@link #refreshRoute()} and never within one, so
     * the staged ranges and the read that consumes them are always on the same channel.
     */
    private boolean directIOActive;

    /**
     * This input's identity in its thread's {@link DirectIOStagingLease}, used to decide whether the shared
     * blocking window still holds <em>this</em> input's bytes. A counter and not {@code this}, so that a
     * lease never holds a reference to an input and so never pins a {@link Handle} or a channel.
     */
    private final long inputId = INPUT_IDS.incrementAndGet();

    /**
     * The lease generation this input's burst was given, or {@code -1} when it holds no lease. Equality with
     * the live generation of {@link DirectIOStagingLease#current()} is what makes the staged table readable;
     * a mismatch means the next burst on this thread took the arena back and this input must read blocking.
     */
    private long leaseGeneration = -1;
    /**
     * The thread that took {@link #leaseGeneration}. Compared before the generation so that a read arriving
     * on another thread is counted as what it is ({@code foreign_thread_reads}) rather than as a revocation,
     * and so that nothing consults a table it does not own. A thread id may be reused after a thread dies,
     * which cannot make this check wrongly pass: the generation of a dead thread's burst is never current
     * again.
     */
    private long leaseThreadId = -1;

    /**
     * The pool this burst's staged reads are submitted to, acquired by {@link #ensureStagingTable} and
     * cleared by {@link #endBurst}, so it is never held across a burst boundary. The read-ahead ring
     * does not use this field; see {@link #refillReadahead}.
     */
    private ExecutorService pool;
    /** Whether a read has happened since the last {@link #prefetch}, which is how a burst's end is found. */
    private boolean readSincePrefetch;
    /**
     * Whether this burst has already picked up a settings change, cleared by {@link #endBurst}.
     *
     * <p>The marker used to be {@code pool == null}, which is set by {@link #ensureStagingTable} part way
     * through {@link #prefetch}. It is a field of its own since N1 because the burst boundary is now also
     * reached by {@link #selectUncachedOrdinals}, which runs <em>before</em> the first {@link #prefetch} of a
     * batch and has to do the refresh itself: a batch whose every candidate is cached issues no
     * {@link #prefetch} at all, and without this an operator's {@code max_bytes} change (0b included) would
     * never be read on a node whose cache serves everything.
     */
    private boolean burstConfigured;

    /**
     * The absolute file offsets N1's prefetch filter dropped as cached in this burst, sorted ascending, and
     * how many of them are valid. Cleared by {@link #endBurst}. O4.
     *
     * <p>This is what makes the filter-then-read race recoverable. A vector the filter dropped is not staged,
     * so if the store loses it before {@link #readFloats} the read is a blocking one and M3's rule throws its
     * bytes away. Knowing that <em>this</em> blocking read is for a vector the filter believed cached is what
     * distinguishes it from a sequential scan's blocking read, which must still not be admitted. Bounded by
     * the batch the caller hands in, so one long per candidate and nothing per burst that has no filter call.
     */
    private long[] filteredOffsets;
    private int filteredCount;

    /**
     * How wide this burst may make a staged miss read so that the vectors around the one being fetched are
     * admitted with it, or 0 for "do not widen it". O5. Read once per burst by {@link #beginBurst}, because
     * the gates are a settings read and a store-wide byte comparison and the answer cannot change inside a
     * burst without the staged geometry changing under a read.
     */
    private int burstReadAroundBytes;

    // -- Sequential read-ahead ring. Distinct from the burst table above: it holds no state in common with
    // it, is scanned independently in source(), and neither mechanism resets the other. Allocated lazily
    // on the first arm, so a non-sequential input never pays for it; null until then. --
    /** The previous blocking read's aligned start, for the forward-and-gapless detection predicate. */
    private long blockingStart = -1;
    /** The previous blocking read's aligned end: the file-order frontier read-ahead runs ahead of. */
    private long blockingEnd = -1;
    /** Consecutive gapless-forward blocking reads seen so far; reset on any gap, backward jump or seek-away. */
    private int sequentialRun;
    /** The ring, size {@link #readaheadWindows}, used circularly; null until the first arm. */
    private StagedRange[] readaheadRing;
    private ByteBuffer[] readaheadBuffers;
    private FloatBuffer[] readaheadFloats;
    /** The one allocation {@link #readaheadBuffers} is carved from; held for life, like {@link #stagingArena}. */
    private ByteBuffer readaheadArena;
    /** Index of the oldest live slot. */
    private int readaheadHead;
    /** Live slots, 0..{@link #readaheadWindows}. */
    private int readaheadCount;
    /** Aligned start of the next window the ring will submit. */
    private long readaheadNext;

    private DirectIOVectorIndexInput(
        final String resourceDescription,
        final Handle handle,
        @Nullable final IndexInput mmapFallback,
        final BooleanSupplier routeEnabled,
        final long offset,
        final long length,
        final boolean ownsHandle,
        final int bufferSize,
        final int maxStagedRanges,
        final int maxStagedRangeBytes,
        final int readaheadWindows,
        final int readaheadWindowBytes,
        final int readaheadTrigger
    ) {
        super(resourceDescription);
        this.handle = handle;
        this.mmapFallback = mmapFallback;
        this.routeEnabled = routeEnabled;
        // The route a new input starts on is the one the switch names now, not the one it named when the
        // file was opened. MEASURED by 16-I: without this, arm D of its re-proof still read the device once
        // per values object with the switch off, because Lucene builds a fresh slice per query and the first
        // read of a slice arrives before its first prefetch, which is the boundary a route change is taken
        // at. One supplier call per input object, and none at all for an input with no fallback.
        this.directIOActive = mmapFallback == null || routeEnabled.getAsBoolean();
        this.offset = offset;
        this.length = length;
        this.ownsHandle = ownsHandle;
        this.bufferSize = bufferSize;
        this.maxStagedRanges = maxStagedRanges;
        this.maxStagedRangeBytes = maxStagedRangeBytes;
        this.readaheadWindows = readaheadWindows;
        this.readaheadWindowBytes = readaheadWindowBytes;
        this.readaheadTrigger = readaheadTrigger;
    }

    /**
     * Opens {@code path} with {@code O_DIRECT} and returns an input over the whole file, or {@code null}
     * when Direct I/O cannot serve it — no {@code ExtendedOpenOption.DIRECT} on this JDK, a filesystem that
     * answers {@code EINVAL}, or a path that is not a regular file. Returning {@code null} rather than
     * throwing is the fallback contract every Direct I/O open in this package has: a file this cannot serve
     * must cost a fall back to the default path, never a failed query.
     */
    public static DirectIOVectorIndexInput open(final Path path) {
        return open(path, (IndexInput) null);
    }

    /**
     * As {@link #open(Path)}, with the delegate's own input over the same file kept beside the
     * {@code O_DIRECT} channel so that {@code knn.direct_io.rescore.enabled} can be turned off under an
     * already-open input (O1). The returned input owns {@code mmapFallback} and closes it; a {@code null}
     * fallback means the input stays on {@code O_DIRECT} whatever the setting says, which is what every
     * caller that has no second input to give gets. See {@link #mmapFallback}.
     */
    public static DirectIOVectorIndexInput open(final Path path, @Nullable final IndexInput mmapFallback) {
        return open(path, mmapFallback, KNNSettings::isDirectIORescoreEnabled);
    }

    /**
     * As {@link #open(Path, IndexInput)}, with the switch the route is re-read from. Production passes
     * {@link KNNDirectIODirectory}'s own condition-2 supplier, so that the directory and its inputs read
     * one source; see {@link #routeEnabled}.
     */
    public static DirectIOVectorIndexInput open(
        final Path path,
        @Nullable final IndexInput mmapFallback,
        final BooleanSupplier routeEnabled
    ) {
        return open(
            path,
            mmapFallback,
            routeEnabled,
            0,
            configuredMaxStagedRanges(),
            configuredMaxStagedRangeBytes(),
            configuredReadaheadWindow(),
            configuredReadaheadWindowBytes(),
            configuredReadaheadTrigger()
        );
    }

    /**
     * How many ranges one burst may hold in flight, from
     * {@code knn.direct_io.rescore.prefetch.staged_ranges}.
     *
     * <p>Its own setting rather than the rescore seam's removed {@code prefetch_window}, which task-15
     * first tried: that window was a rolling ring refilled as it drained, this is a table that must
     * <b>hold a whole
     * burst</b>, and the burst's size is not ours to choose. {@code PrefetchHelper} is driven by Lucene's
     * 64-ordinal bulk batch, so a table smaller than 64 declines the tail of <em>every</em> burst and those
     * ranges become blocking reads on the calling thread. Measured on {@code dio-1m}: at 48 the route made
     * 5.4 blocking reads per query and the paced p99 was 30 ms; at 64 it makes 0.0 and the paced p99 is
     * 11 ms, at <b>identical byte volume</b> (2,581 vs 2,591 KiB/query).
     */
    private static int configuredMaxStagedRanges() {
        final int configured = KNNSettings.getDirectIORescorePrefetchStagedRanges();
        return configured > 0 ? configured : DEFAULT_MAX_STAGED_RANGES;
    }

    /**
     * The span bound from {@code knn.direct_io.rescore.prefetch.max_span_bytes}, clamped into int range.
     * Read here, once per {@code .vec} open, rather than per prefetch: the bound sizes the staging buffers,
     * and a bound that moved under a burst would mean a buffer allocated for one size holding another.
     * A setting that cannot be read falls back to its default, the same contract every other Direct I/O
     * setting has — this must never be the reason a shard fails to open.
     */
    private static int configuredMaxStagedRangeBytes() {
        final ByteSizeValue configured = KNNSettings.getDirectIORescorePrefetchMaxSpan();
        if (configured == null || configured.getBytes() <= 0) {
            return DEFAULT_MAX_STAGED_RANGE_BYTES;
        }
        return Math.toIntExact(Math.min(configured.getBytes(), Integer.MAX_VALUE / 2));
    }

    /**
     * Windows the sequential read-ahead ring keeps in flight, from {@code knn.direct_io.readahead.window}.
     * Read here, once per {@code .vec} open, like the staging bounds: the value sizes the ring's buffers,
     * and a value that moved under a running ring would mean a buffer allocated for one count holding
     * another. A setting that cannot be read falls back to its default — this must never be why a shard
     * fails to open. 0 means the ring is off.
     */
    private static int configuredReadaheadWindow() {
        try {
            return Math.max(0, KNNSettings.getDirectIOReadaheadWindow());
        } catch (RuntimeException e) {
            log.debug("Could not read knn.direct_io.readahead.window; read-ahead disabled for this open", e);
            return KNNSettings.KNN_DIRECT_IO_READAHEAD_WINDOW_DEFAULT_VALUE;
        }
    }

    /**
     * Bytes per read-ahead window, from {@code knn.direct_io.readahead.window_bytes}. Read once per
     * {@code .vec} open for the same reason the window count is: the value sizes the ring's arena, and a
     * value that moved under a running ring would mean a buffer allocated for one size holding another.
     * Clamped into int range; a setting that cannot be read falls back to its default.
     *
     * @see #readaheadWindowBytes
     */
    private static int configuredReadaheadWindowBytes() {
        try {
            final ByteSizeValue configured = KNNSettings.getDirectIOReadaheadWindowBytes();
            if (configured == null || configured.getBytes() <= 0) {
                return Math.toIntExact(KNNSettings.KNN_DIRECT_IO_READAHEAD_WINDOW_BYTES_DEFAULT_VALUE.getBytes());
            }
            return Math.toIntExact(Math.min(configured.getBytes(), Integer.MAX_VALUE / 2));
        } catch (RuntimeException e) {
            log.debug("Could not read knn.direct_io.readahead.window_bytes; using the default", e);
            return Math.toIntExact(KNNSettings.KNN_DIRECT_IO_READAHEAD_WINDOW_BYTES_DEFAULT_VALUE.getBytes());
        }
    }

    /** The arm trigger, from {@code knn.direct_io.readahead.trigger}; same read-once, fall-back contract. */
    private static int configuredReadaheadTrigger() {
        try {
            return Math.max(1, KNNSettings.getDirectIOReadaheadTrigger());
        } catch (RuntimeException e) {
            log.debug("Could not read knn.direct_io.readahead.trigger; using the default", e);
            return KNNSettings.KNN_DIRECT_IO_READAHEAD_TRIGGER_DEFAULT_VALUE;
        }
    }

    /**
     * As {@link #open(Path)}, with the sizes chosen by the caller so that a measurement can sweep them.
     *
     * @param bufferSize          bytes per blocking read, or 0 for twice the block size, which serves any
     *                            sub-block read in one syscall however it straddles a boundary
     * @param maxStagedRanges     ranges one prefetch burst may hold in flight
     * @param maxStagedRangeBytes largest single range that will be staged
     */
    public static DirectIOVectorIndexInput open(
        final Path path,
        final int bufferSize,
        final int maxStagedRanges,
        final int maxStagedRangeBytes
    ) {
        // The four-argument overload keeps its signature so the phase-9 queue-depth probe compiles against
        // it unchanged; it picks up the read-ahead sizes from settings, exactly as open(Path) does.
        return open(
            path,
            bufferSize,
            maxStagedRanges,
            maxStagedRangeBytes,
            configuredReadaheadWindow(),
            configuredReadaheadWindowBytes(),
            configuredReadaheadTrigger()
        );
    }

    /**
     * As the four-argument overload with the ring's <b>window count</b> chosen by the caller, each window
     * being one {@code bufferSize} — which is what the ring was before phase 14 decoupled the two. Kept at
     * that meaning rather than reading the new setting, so that every phase 11 read-ahead measurement and
     * the tests that pin them keep measuring the thing they were taken against.
     *
     * @param readaheadWindows windows the sequential ring keeps in flight; 0 disables it (exact passthrough)
     * @param readaheadTrigger consecutive gapless-forward blocking reads before the ring arms
     */
    public static DirectIOVectorIndexInput open(
        final Path path,
        final int bufferSize,
        final int maxStagedRanges,
        final int maxStagedRangeBytes,
        final int readaheadWindows,
        final int readaheadTrigger
    ) {
        return open(path, bufferSize, maxStagedRanges, maxStagedRangeBytes, readaheadWindows, 0, readaheadTrigger);
    }

    /**
     * As the six-argument overload, with the ring's <b>window size</b> chosen too, so that a measurement can
     * sweep audit FIX 2 without a {@link org.opensearch.cluster.service.ClusterService}.
     *
     * @param readaheadWindows     windows the sequential ring keeps in flight; 0 disables it (exact passthrough)
     * @param readaheadWindowBytes bytes per window; rounded up to a block, floored at {@code bufferSize}, and
     *                             0 means "use {@code bufferSize}", i.e. the pre-phase-14 coupled behaviour
     * @param readaheadTrigger     consecutive gapless-forward blocking reads before the ring arms
     */
    public static DirectIOVectorIndexInput open(
        final Path path,
        final int bufferSize,
        final int maxStagedRanges,
        final int maxStagedRangeBytes,
        final int readaheadWindows,
        final int readaheadWindowBytes,
        final int readaheadTrigger
    ) {
        // No fallback, so this input stays on O_DIRECT for its life: the sizing overloads are the
        // measurement and test entry points, which open a path directly and have no delegate to give. The
        // supplier is still the production one, so an input built this way reports the same route the
        // directory would have dispatched on.
        return open(
            path,
            null,
            KNNSettings::isDirectIORescoreEnabled,
            bufferSize,
            maxStagedRanges,
            maxStagedRangeBytes,
            readaheadWindows,
            readaheadWindowBytes,
            readaheadTrigger
        );
    }

    /**
     * As the seven-argument overload, with the delegate input the switch falls back to and the switch
     * itself. See {@link #mmapFallback}.
     */
    public static DirectIOVectorIndexInput open(
        final Path path,
        @Nullable final IndexInput mmapFallback,
        final BooleanSupplier routeEnabled,
        final int bufferSize,
        final int maxStagedRanges,
        final int maxStagedRangeBytes,
        final int readaheadWindows,
        final int readaheadWindowBytes,
        final int readaheadTrigger
    ) {
        final OpenOption direct = directOpenOption();
        if (direct == null) {
            log.warn("This JDK does not expose ExtendedOpenOption.DIRECT; a Direct I/O IndexInput cannot be opened");
            return null;
        }
        FileChannel channel = null;
        try {
            if (Files.isRegularFile(path) == false) {
                log.warn("Direct I/O cannot open [{}]: not a regular file", path);
                return null;
            }
            final long fileLength = Files.size(path);
            final int blockSize = Math.toIntExact(Files.getFileStore(path).getBlockSize());
            if (blockSize <= 0) {
                log.warn("Direct I/O cannot open [{}]: the filesystem reports block size {}", path, blockSize);
                return null;
            }
            final int reads = bufferSize > 0 ? alignUp(bufferSize, blockSize) : 2 * blockSize;
            channel = FileChannel.open(path, StandardOpenOption.READ, direct);
            return new DirectIOVectorIndexInput(
                "DirectIOVectorIndexInput(" + path + ")",
                new Handle(path, channel, blockSize, fileLength),
                mmapFallback,
                routeEnabled,
                0L,
                fileLength,
                true,
                reads,
                Math.max(1, maxStagedRanges),
                Math.max(blockSize, maxStagedRangeBytes),
                Math.max(0, readaheadWindows),
                // A window is a whole number of blocks (O_DIRECT needs the length aligned) and never
                // smaller than a blocking read, so decoupling can only widen the ring, never narrow it
                // below the invariant every existing read-ahead measurement was taken at.
                Math.max(reads, alignUp(Math.max(blockSize, readaheadWindowBytes), blockSize)),
                Math.max(1, readaheadTrigger)
            );
        } catch (IOException | RuntimeException e) {
            if (channel != null) {
                try {
                    channel.close();
                } catch (IOException ignored) {
                    log.debug("Failed to close a Direct I/O channel that was being abandoned", ignored);
                }
            }
            log.warn("Direct I/O could not open [{}]", path, e);
            return null;
        }
    }

    /** The filesystem block size every read of this input is aligned to. */
    public int blockSize() {
        return handle.blockSize;
    }

    /** Whether this input is reading through {@code O_DIRECT} right now rather than its delegate. O1. */
    public boolean isDirectIOActive() {
        return directIOActive;
    }

    /** Whether this input has a delegate input to fall back to. Visible for testing. */
    boolean hasMmapFallback() {
        return mmapFallback != null;
    }

    /** Bytes per blocking read. */
    public int bufferSize() {
        return bufferSize;
    }

    /** Ranges one prefetch burst may hold in flight, after the setting was read at open. */
    int maxStagedRanges() {
        return maxStagedRanges;
    }

    /** The waste guard's bound: the largest single prefetch span this input will stage, in bytes. */
    int maxStagedRangeBytes() {
        return maxStagedRangeBytes;
    }

    /** Windows the sequential read-ahead ring keeps in flight, after the setting was read at open. */
    int readaheadWindows() {
        return readaheadWindows;
    }

    /** Bytes per read-ahead window, after the setting was read, block-aligned and floored at bufferSize. */
    int readaheadWindowBytes() {
        return readaheadWindowBytes;
    }

    /** Consecutive gapless-forward blocking reads before the ring arms, after the setting was read at open. */
    int readaheadTrigger() {
        return readaheadTrigger;
    }

    /**
     * The burst table's single backing allocation while this input holds its thread's lease, or null when it
     * does not — before its first {@link #prefetch}, after the next burst on this thread took the arena
     * back, or when the budget refused one.
     */
    ByteBuffer stagingArena() {
        final DirectIOStagingLease lease = heldLease();
        return lease == null ? null : lease.arena();
    }

    /**
     * This thread's lease if this input still holds it, else null.
     *
     * <p>Two conditions, in this order: the read is on the thread that took the lease, and the lease still
     * carries the generation this input was given. The thread check comes first because a foreign thread's
     * {@link DirectIOStagingLease#current()} is a different object whose generation this input must not even
     * read as a comparison — see {@link #leaseThreadId}.
     */
    private DirectIOStagingLease heldLease() {
        if (leaseGeneration < 0 || Thread.currentThread().threadId() != leaseThreadId) {
            return null;
        }
        final DirectIOStagingLease lease = DirectIOStagingLease.current();
        return lease.generation() == leaseGeneration ? lease : null;
    }

    /** The read-ahead ring's single backing allocation, or null before the ring first arms. */
    ByteBuffer readaheadArena() {
        return readaheadArena;
    }

    // ---------------------------------------------------------------------------------------------------
    // Read ahead
    // ---------------------------------------------------------------------------------------------------

    /**
     * Writes into {@code uncached} the ordinals of {@code ords[0..numOrds)} whose vectors the store does
     * <b>not</b> hold whole, and returns how many, or returns {@code -1} to mean "ask about nothing, prefetch
     * the batch as it is".
     *
     * <p>This is N1. The caller is {@code PrefetchHelper#prefetch}, which without it hands every candidate of
     * a batch to its 128 KB coalescing and so asks this input for groups whose spans are mostly already
     * cached. The input can only accept or decline a whole group, so a group with one missing vector in it is
     * staged in full: MEASURED on the node (ref/NODE-16M.md), 73.4 staged reads per query for 77.1 missing
     * vectors at 22.9 KB a read in S2, against 16-I's 62.7 for 76.9 at 12.6 KB, because 16-I grouped only its
     * misses. Dropping the hits here before the grouping is what makes the groups the same ones 16-I built.
     *
     * <h4>What it does not do</h4>
     * It counts no hit and no miss, like {@link DirectIOVectorCache#contains}: a hit is counted once, by the
     * {@code readFloats} that serves it, so {@code hits / (hits + misses)} stays a per-vector hit rate
     * comparable with the 16-S node figures. It serves no bytes either: a vector dropped here is still read
     * through {@link #readFloats}, which looks it up again, and a vector evicted in between is a blocking read
     * of the same bytes. So no score can change.
     *
     * <p>It also opens the burst ({@link #beginBurst}), because it is the first call of a batch and a batch
     * whose every candidate is cached makes no {@link #prefetch} call at all.
     *
     * @param sliceOffset where the vectors start, relative to this input (0 for the slice Lucene hands the
     *                    scorer, which is what the helper passes)
     * @param vectorBytes one vector's length in bytes
     * @param ords        the batch's ordinals, read and not written
     * @param numOrds     how many of {@code ords} are valid
     * @param uncached    scratch of at least {@code numOrds} ints, the caller's to reuse
     * @return how many ordinals were written to {@code uncached}, or {@code -1} when the store can answer
     *         nothing useful and the caller should take its unfiltered path
     */
    public int selectUncachedOrdinals(
        final long sliceOffset,
        final long vectorBytes,
        final int[] ords,
        final int numOrds,
        final int[] uncached
    ) {
        if (closed || DirectIOVectorCache.isOn() == false) {
            return -1;
        }
        beginBurst();
        if (directIOActive == false) {
            // O1: with the switch off this input admits nothing and the store has been released, so there
            // is nothing to filter and the caller takes its unfiltered path, which prefetches through the
            // delegate. Checked after beginBurst, which is where the switch was just re-read.
            return -1;
        }
        if (vectorBytes <= 0 || vectorBytes > Integer.MAX_VALUE || numOrds <= 0 || uncached.length < numOrds) {
            return -1;
        }
        final long base = this.offset + sliceOffset;
        if (sliceOffset < 0 || base < 0) {
            return -1;
        }
        final int missing = DirectIOVectorCache.selectMissing(handle.fileId, base, (int) vectorBytes, ords, numOrds, uncached);
        if (missing >= 0 && missing < numOrds) {
            recordFilteredAsCached(base, (int) vectorBytes, ords, numOrds, uncached, missing);
        }
        return missing;
    }

    /**
     * Remembers the offsets this filter call dropped, which is {@code ords \ uncached}, so that a blocking
     * read for one of them can be admitted rather than thrown away (O4). Sorted, so the read path's check is
     * a binary search.
     *
     * <p>{@code uncached[0..missing)} is a subsequence of {@code ords[0..numOrds)} in order -- the store
     * writes the misses out as it walks the batch -- so the complement is found in one pass over both, with
     * no set and no allocation beyond growing the scratch once per input.
     */
    private void recordFilteredAsCached(
        final long base,
        final int vectorBytes,
        final int[] ords,
        final int numOrds,
        final int[] uncached,
        final int missing
    ) {
        final int dropped = numOrds - missing;
        if (filteredOffsets == null || filteredOffsets.length < dropped) {
            filteredOffsets = new long[Math.max(dropped, 64)];
        }
        int at = 0;
        int next = 0;
        for (int i = 0; i < numOrds && at < dropped; i++) {
            if (next < missing && uncached[next] == ords[i]) {
                next++;
                continue;
            }
            filteredOffsets[at++] = base + (long) ords[i] * vectorBytes;
        }
        filteredCount = at;
        // The batch's ordinals are not sorted when the filter runs -- PrefetchHelper sorts what is left of
        // them afterwards -- so this does its own sort, over at most the batch's size.
        Arrays.sort(filteredOffsets, 0, filteredCount);
    }

    /**
     * Whether N1's prefetch filter dropped the vector at {@code absoluteOffset} as cached in this burst. The
     * gate on O4's re-admission: a blocking read for one of these is the filter-then-read race and its bytes
     * are worth keeping, where any other blocking read is a sequential channel's and is not.
     */
    private boolean wasFilteredAsCached(final long absoluteOffset) {
        return filteredCount > 0 && Arrays.binarySearch(filteredOffsets, 0, filteredCount, absoluteOffset) >= 0;
    }

    /**
     * Puts the block-aligned range covering {@code [offset, offset + length)} in flight on
     * {@link DirectIOReadPool}.
     *
     * <p>Advisory in both directions: this may stage nothing, and a caller that then reads something else,
     * or reads out of order, still gets correct bytes from blocking reads. No returned value tells the
     * caller which happened, so nothing can come to depend on it.
     *
     * <h4>The span bound is the waste guard, and it is a bound on the span because it cannot be a bound
     * on the waste</h4>
     * The caller on the rescore path is {@code PrefetchHelper#prefetchExactVectorSize}, which sorts the
     * batch's ordinals and then asks for the <em>span</em> of each coalesced group — extending a group
     * while {@code (vector end) - (group start) <= 128 KB} and then calling
     * {@code prefetch(groupStart, lastEnd - groupStart)}. The gaps between the group's vectors are inside
     * that span and are not in these two arguments, so this method cannot tell a group of 42 back-to-back
     * vectors (a 128 KB span, none of it wasted) from a group of two vectors 120 KB apart (a 120 KB span,
     * 95% of it wasted). The gap structure exists one layer up and is not on this interface.
     *
     * <p>What is left is a bound on the span, {@code knn.direct_io.rescore.prefetch.max_span_bytes}. Above
     * it the range is declined, which costs read-ahead for exactly those groups and saves whatever their
     * gaps were: the bytes are then served by {@link #blockingSource}, one {@link #bufferSize} window per
     * vector touched and nothing for the gaps. That is a real trade and not a free win, which is why the
     * bound is a setting with the measured value as its default rather than a new constant.
     */
    @Override
    public void prefetch(final long offset, final long length) throws IOException {
        ensureOpen();
        STATS.prefetchCalls.incrementAndGet();
        if (length <= 0 || offset < 0 || offset + length > this.length) {
            return;
        }
        if (length > maxStagedRangeBytes) {
            // Checked before any int arithmetic, so a caller asking to prefetch gigabytes declines here
            // rather than overflowing the aligned-length computation below.
            STATS.stageDeclined.incrementAndGet();
            STATS.declinedSpanBytes.addAndGet(length);
            return;
        }
        beginBurst();
        if (directIOActive == false) {
            // O1, D2: this burst is a delegate burst, so the lookahead is the one the page-cache path has.
            // Lucene's mmap prefetch is madvise(WILLNEED) over the mapped range, and the delegate's input
            // carries this input's coordinates, so the arguments pass through unchanged.
            mmapFallback.prefetch(offset, length);
            return;
        }
        final long absolute = this.offset + offset;
        final int blockSize = handle.blockSize;
        // The cache lookup comes BEFORE the submit, so a hit costs the device nothing: the range is simply
        // not added to the burst, and readFloats will find the same entries when it consumes the vectors.
        // This is where the 707k -> ~420k device reads per 5,000 queries would come from.
        //
        // The range here is a coalesced GROUP and not one vector (PrefetchHelper sorts the batch's ordinals
        // and asks for each group's span), so the store walks the whole run and answers only if every vector
        // in it is cached. M4 MEASURED what the one-vector form costs instead: 27.9x the file in device
        // reads against 1.21x, because a group of a dozen cached vectors was staged in full.
        if (DirectIOVectorCache.isOn() && DirectIOVectorCache.contains(handle.fileId, absolute, Math.toIntExact(length))) {
            return;
        }
        long alignedStart = absolute - Math.floorMod(absolute, (long) blockSize);
        int alignedLength = alignUp(Math.toIntExact(absolute + length - alignedStart), blockSize);
        if (alignedLength > maxStagedRangeBytes) {
            STATS.stageDeclined.incrementAndGet();
            STATS.declinedSpanBytes.addAndGet(length);
            return;
        }
        // O5: past here this range is a MISS -- the store does not hold it whole -- so while the whole file
        // would still fit in the room the target leaves, widen it and let span fill harvest the neighbours.
        // One read per miss stays one read; only its size changes. The widened window is still one aligned
        // range, so nothing downstream changes: the span-fill walk clips to the reading slice and ends at
        // the bytes the read returned, exactly as it does for a narrow one.
        if (burstReadAroundBytes > alignedLength) {
            final int want = Math.min(alignUp(burstReadAroundBytes, blockSize), maxStagedRangeBytes);
            if (want > alignedLength) {
                // Half the extra behind the request and the rest in front, clipped at the start of the file.
                // Behind as well as in front because a candidate batch is scattered, so the next miss is as
                // likely to be below this one as above it. The window still covers the request: the step
                // back is at most the extra, and the window is `want` from the new start.
                final long back = Math.min(alignedStart, alignUp((want - alignedLength) / 2, blockSize));
                alignedStart -= back;
                alignedLength = want;
                STATS.recordReadAround();
            }
        }
        final DirectIOStagingLease lease = ensureStagingTable();
        if (lease == null) {
            STATS.stageDeclined.incrementAndGet();
            return;
        }
        if (lease.covering(absolute, Math.toIntExact(length)) != null) {
            // Already in flight from an earlier call in this burst. PrefetchHelper's grouping makes this
            // common rather than exceptional: one group covers every vector inside it.
            return;
        }
        if (lease.isFull()) {
            STATS.stageDeclined.incrementAndGet();
            return;
        }
        final ByteBuffer buffer = lease.bufferFor(alignedLength);
        if (buffer == null) {
            // Only reachable for a coalesced group wider than an arena slot, which needs a buffer of its own
            // (up to max_span_bytes, 132 KiB) and is therefore the one per-range allocation a burst can still
            // make. See arenaSlotBytes() for why that is the uncommon case.
            STATS.recordDeclinedBudget();
            STATS.stageDeclined.incrementAndGet();
            log.debug(
                "a direct-memory bound was reached ({} B live of the overall budget, {} B of the {} B "
                    + "wide-group sub-budget); a wide prefetch range of {} declines. declined_wide_budget "
                    + "says which of the two it was",
                STATS.directMemoryLive(),
                STATS.wideBufferLive(),
                wideBufferBudgetBytes(),
                handle.path
            );
            return;
        }
        final StagedRange range = new StagedRange(
            handle.channel,
            handle.path,
            buffer,
            lease.floatsFor(alignedLength),
            alignedStart,
            alignedLength,
            STATS.stagedReads,
            STATS.stagedBytes
        );
        try {
            range.future = pool.submit(range);
        } catch (RejectedExecutionException e) {
            STATS.stageDeclined.incrementAndGet();
            log.debug("The Direct I/O read pool refused a prefetch of {}; reads will block", handle.path);
            return;
        }
        DirectIOReadPool.sampleQueueDepth(pool);
        lease.record(range);
    }

    /**
     * Acquires the pool for this burst and allocates the burst table once.
     *
     * <p><b>Acquired per burst, which is what makes a live capacity change reach this input.</b> Before
     * phase 15 this method returned early on {@code staged != null} and so read {@link #pool} once per input
     * and then held that reference for the input's whole life — and an input lives as long as the values
     * object Lucene built it for, which on the merge path is the whole merge. A capacity change swaps the
     * pool, so a stale reference would keep submitting to the retired one until it was shut down and then
     * decline every prefetch. Re-acquiring at the burst boundary ({@link #endBurst} clears the field)
     * bounds staleness to one burst, and costs one settings read per burst rather than one per
     * {@link #prefetch} call — there are up to {@code staged_ranges} of those in a burst.
     *
     * <p><b>And it is where this input takes its thread's staging arena (D3c).</b> The first
     * {@link #prefetch} of a burst calls {@link DirectIOStagingLease#take}, which quiesces whatever the
     * previous holder left staged and then reuses that thread's arena or, if a read may still be writing into
     * it, retires it and carves one. Before D3c every input carved its own and held it until a collection
     * found the input, which on the per-query slices Lucene's flat reader creates meant arenas churned at
     * {@code QPS x segments x 516 KiB/s} — 9-13 MiB/s on the node — while the memory actually in use was
     * about {@code 16 concurrent bursts x 516 KiB}. That churn is what the direct-memory budget was
     * measuring, and why at its 1 GB default it wedged the feature between collections.
     *
     * @return this thread's lease, or null when there is no pool or the budget refused an arena, in which
     *         case reads must block
     */
    private DirectIOStagingLease ensureStagingTable() {
        final DirectIOStagingLease lease = heldLease();
        if (lease != null) {
            return lease;
        }
        // Past here this is a NEW burst, whether or not endBurst ran -- an input that prefetched, never read,
        // and was then revoked by another burst arrives here with a stale pool reference still set. Dropping
        // it is what bounds pool staleness to one burst in that case too, which is the whole point of
        // acquiring per burst (D1).
        pool = null;
        final ExecutorService acquired = DirectIOReadPool.executor();
        if (acquired == null) {
            return null;
        }
        pool = acquired;
        final DirectIOStagingLease taken = DirectIOStagingLease.current();
        final long generation = taken.take(maxStagedRanges, arenaSlotBytes(), handle.blockSize);
        if (generation < 0) {
            // D3's decline, now reached only when a lease has to carve: a burst that reuses its thread's
            // arena cannot raise the node's footprint, so there is nothing to decline it for. Past here
            // prefetch counts stageDeclined and the bytes are served by blockingSource -- the same fallback a
            // queue-capacity decline takes, already exercised on every arm.
            STATS.recordDeclinedBudget();
            log.debug(
                "knn.direct_io.direct_memory_budget reached ({} B live); prefetch of {} declines and reads will block",
                STATS.directMemoryLive(),
                handle.path
            );
            return null;
        }
        leaseGeneration = generation;
        leaseThreadId = Thread.currentThread().threadId();
        return taken;
    }

    /**
     * Bytes per arena slot: one {@link #bufferSize} window, and never less than two blocks, which is the
     * span of any single vector however it straddles a block boundary.
     *
     * <p>This is the size the <em>common</em> staged range needs, not the largest one allowed. A prefetch
     * group is {@code PrefetchHelper}'s coalescing of a sorted ordinal batch, and phase 12 S3.1 measured the
     * mean aligned span at <b>7,310 B</b> for a 3,072 B vector against an arithmetic expectation of 7,168 B
     * — i.e. nearly every group is one vector straddling one boundary and fits in two blocks. Groups that
     * hold several adjacent candidates are wider and keep their own allocation (see {@link #prefetch}), so
     * sizing the arena for the worst case ({@link #maxStagedRangeBytes}, 132 KiB × {@code staged_ranges} =
     * 8.4 MiB per input) would reserve two orders of magnitude more than the measured need.
     */
    int arenaSlotBytes() {
        return Math.max(bufferSize, 2 * handle.blockSize);
    }

    /**
     * Allocates one block-aligned direct buffer holding {@code slots} slots and fills {@code buffers} /
     * {@code floats} with a view of each.
     *
     * <p><b>Why one allocation rather than one per slot.</b> Phase 12 S1.3 measured direct-buffer capacity
     * cycling to <b>4,365 MB</b> under 16-way load, because a fresh {@link DirectIOVectorIndexInput} is
     * created per query per segment ({@code ExactSearcher#createVectorScorer} → Lucene's flat reader →
     * {@link #slice}), each one allocated its own sync window plus up to {@code staged_ranges} staged slots
     * plus a read-ahead ring, and nothing frees them — Lucene's {@code OffHeapFloatVectorValues} is not
     * {@code Closeable}, so {@link #close} never runs on a per-query slice and reclamation waits for the
     * {@code Cleaner}, i.e. for a GC. There is no end-of-life signal on that path to return a buffer at, so
     * the footprint cannot be fixed by pooling across inputs; what it can be fixed by is allocating less and
     * less often. One arena makes a per-input {@code staged_ranges}-slot table <b>one</b>
     * {@code allocateDirect} instead of up to {@code staged_ranges} of them, and drops the per-slot
     * {@code blockSize - 1} alignment slack (12,287 B → 8,192 B per slot at a 4 KiB block), which is the
     * same arena move the rescore seam's reader made for its own buffers before phase 12 H5 removed it.
     *
     * <p><b>What this does not claim.</b> It does not bound total direct memory and it does not promise the
     * pages go back to the OS: glibc raises its {@code mmap} threshold to the size of the first {@code mmap}ed
     * chunk it frees, so a repeated half-megabyte allocation can still end up served from a malloc arena that
     * never shrinks. {@code MALLOC_ARENA_MAX} remains the operational lever for that. The claim here is only
     * fewer and smaller allocations, which is measurable as peak direct-buffer capacity.
     */
    static ByteBuffer carveArena(
        final int slots,
        final int slotBytes,
        final int blockSize,
        final ByteBuffer[] buffers,
        final FloatBuffer[] floats
    ) {
        // Over-allocate by a block so a block-aligned slice exists inside the allocation, then carve at
        // multiples of a slot: the base is aligned and a slot is a whole number of blocks, so every slot is
        // aligned too and needs no slack of its own. O_DIRECT needs the buffer address, the file offset and
        // the length all block aligned.
        final ByteBuffer arena = trackedAllocate(slots * slotBytes, blockSize);
        for (int slot = 0; slot < slots; slot++) {
            buffers[slot] = arena.slice(slot * slotBytes, slotBytes).order(ByteOrder.LITTLE_ENDIAN);
            floats[slot] = buffers[slot].asFloatBuffer();
        }
        return arena;
    }

    /** Bytes {@link #trackedAllocate} actually reserves for a request of {@code usableBytes}. */
    private static long allocationBytes(final long usableBytes, final int blockSize) {
        return usableBytes + blockSize - 1;
    }

    /**
     * One block-aligned direct buffer of at least {@code usableBytes}, charged to
     * {@code knn.direct_io.direct_memory_budget}'s accounting.
     *
     * <p>The charge is the <em>allocation</em>, not the aligned slice returned: the allocation is what the
     * JVM counts against {@code MaxDirectMemorySize}, and the {@code blockSize - 1} of alignment slack is
     * as real as the rest of it.
     */
    static ByteBuffer trackedAllocate(final int usableBytes, final int blockSize) {
        return trackedAllocate(usableBytes, blockSize, false);
    }

    /**
     * As {@link #trackedAllocate(int, int)}, tagging the charge as a wide-group buffer when {@code wide} so
     * that one reclamation returns the bytes to both the overall budget and the sub-budget. D3d, O2.
     *
     * <p>The sub-budget is <b>not</b> charged here: {@link #allocateWideBuffer} has already reserved it, so
     * that the cap is a bound rather than a hint.
     */
    static ByteBuffer trackedAllocate(final int usableBytes, final int blockSize, final boolean wide) {
        // D3c: drain here, on every allocation, and not only from fitsBudget. With the budget off fitsBudget
        // short-circuits, which left the phantom queue draining on stats reads alone -- and the node then
        // reported peak_bytes of 7.46 GB, 9.0 GB and 28 GB against a 2 GB wall. The drain is a non-blocking
        // poll loop against an allocateDirect that is orders of magnitude dearer.
        reclaim();
        final int bytes = Math.toIntExact(allocationBytes(usableBytes, blockSize));
        final ByteBuffer buffer = ByteBuffer.allocateDirect(bytes).alignedSlice(blockSize).order(ByteOrder.LITTLE_ENDIAN);
        STATS.recordDirectAllocation(bytes);
        CHARGES.add(new Charge(buffer, bytes, wide));
        return buffer;
    }

    /**
     * One block-aligned direct buffer for a coalesced group too wide for an arena slot, charged to the
     * wide-group sub-budget as well as the overall one, or {@code null} when the sub-budget refuses it.
     * D3d, phase 16-I's I7, ported in O2.
     *
     * <h4>Why these buffers need a cap of their own</h4>
     * A {@link DirectIOStagingLease} keeps one wide buffer per {@code staged_ranges} slot, grown on demand
     * to at most {@code max_span_bytes} and then <b>kept for the life of the arena</b>. At the defaults that
     * is 64 x 135,168 B = 8.25 MiB per thread, and a lease is per thread with no bound on the number of
     * threads: 225 search threads on a 64-vCPU node reach 1.95 GB. {@code knn.direct_io.direct_memory_budget}
     * counts them, but it is one pot shared with the arenas, which are reused while these are held -- so wide
     * buffers accumulating to the overall budget would leave no room for an arena and then every prefetch
     * would decline. A budget that wedges the feature is not a bound on it.
     *
     * <p>{@code 0b} <b>derives</b> the cap as a quarter of the overall budget rather than disabling it, and
     * a quarter because the arenas are the common case and must always have room: phase 12 S3.1 measured the
     * mean aligned span at 7,310 B against an arena slot of 8,192, i.e. nearly every group needs no wide
     * buffer at all. With the overall budget off the derived cap is off too, which keeps the rule the budget
     * has always had: it must never be the reason a shard cannot read its vectors.
     */
    @Nullable
    static ByteBuffer allocateWideBuffer(final int usableBytes, final int blockSize) {
        final long charge = allocationBytes(usableBytes, blockSize);
        final long cap = wideBufferBudgetBytes();
        if (cap > 0) {
            // Drain first, so the reservation is taken against a figure that does not count buffers the
            // collector has already reclaimed; then reserve atomically, so the cap is a bound.
            reclaim();
            if (STATS.reserveWide(charge, cap) == false) {
                STATS.recordDeclinedWideBudget();
                return null;
            }
        } else {
            STATS.recordWideAllocation(charge);
        }
        try {
            return trackedAllocate(usableBytes, blockSize, true);
        } catch (RuntimeException | OutOfMemoryError e) {
            // The reservation outlives nothing: there is no buffer for a reclamation to hang on.
            STATS.recordWideFree(charge);
            throw e;
        }
    }

    /**
     * The wide-group sub-budget in bytes, derived from the overall budget when the setting is {@code 0b},
     * and {@code 0} when it is off. A setting that cannot be read is treated as off, for the same reason
     * {@link #fitsBudget} treats one that way.
     */
    public static long wideBufferBudgetBytes() {
        try {
            final ByteSizeValue configured = KNNSettings.getDirectIORescorePrefetchWideBufferBudget();
            final long explicit = configured == null ? 0L : configured.getBytes();
            if (explicit > 0) {
                return explicit;
            }
            final ByteSizeValue overall = KNNSettings.getDirectIODirectMemoryBudget();
            final long budget = overall == null ? 0L : overall.getBytes();
            return budget <= 0 ? 0L : budget / WIDE_BUFFER_BUDGET_SHARE;
        } catch (RuntimeException e) {
            log.debug("Could not read the wide-group buffer budget; not bounding this allocation", e);
            return 0L;
        }
    }

    /**
     * Returns the charges of every buffer reclaimed since the last call, and answers the current live total.
     *
     * <h4>Why the accounting hangs on reachability and not on {@link #close()}</h4>
     * Lucene's {@code OffHeapFloatVectorValues} is not {@code Closeable}, so {@link #close()} never runs on
     * the per-query slice that owns a staging table — which is exactly the condition phase 13's audit FIX 3
     * recorded and the node then measured as 1.69 GB of live direct buffers. A counter decremented only in
     * {@link #close()} would therefore rise monotonically and, after enough queries, decline every prefetch
     * forever: the budget would <em>wedge</em> the feature rather than bound it. The buffers are in fact
     * reclaimed — by the JDK's own cleaner, once the {@link ByteBuffer} is unreachable — so the accounting
     * is hung on that same event, which makes {@code live_bytes} direct memory that is genuinely still
     * reachable. That is the only definition of "live" that is true here.
     *
     * <h4>Why it is drained here and not on a cleaner thread</h4>
     * A {@link java.lang.ref.Cleaner} owns a thread, and a node-wide cleaner thread for this is both
     * unnecessary and visible: it leaks out of every test suite that loads this class. Draining on the
     * calling thread costs a non-blocking {@link ReferenceQueue#poll()} loop and happens at the only two
     * moments the number is used — just before an allocation is decided, and when an operator reads it.
     * Nothing waits for a GC that has not happened; an un-drained charge simply keeps the budget
     * conservative, which is the safe direction.
     */
    public static long reclaim() {
        Reference<? extends ByteBuffer> reference;
        while ((reference = RECLAIMED.poll()) != null) {
            final Charge charge = (Charge) reference;
            if (CHARGES.remove(charge)) {
                STATS.recordDirectFree(charge.bytes);
                if (charge.wide) {
                    // One reclamation, two counters: both charges hang off the one ByteBuffer, so the
                    // sub-budget is returned exactly when the overall budget is. D3d, O2.
                    STATS.recordWideFree(charge.bytes);
                }
            }
        }
        return STATS.directMemoryLive();
    }

    /**
     * Whether {@code usableBytes} more of direct buffer fits under
     * {@code knn.direct_io.direct_memory_budget}. A budget of {@code 0b} is off, and a setting that cannot
     * be read is treated as off — a budget must never be the reason a shard cannot read its vectors.
     *
     * <p>Since D3c this is consulted only where a {@link DirectIOStagingLease} must carve: a new arena, a new
     * wide-group buffer, or a read-ahead ring. A burst that takes its thread's arena back allocates nothing
     * and so cannot be declined, which is why {@code declined_budget} should now read 0 on a loaded node
     * where before it read 373k-614k at the same 1 GB default.
     */
    static boolean fitsBudget(final long usableBytes, final int blockSize) {
        final long live = reclaim();
        final long budget;
        try {
            final ByteSizeValue configured = KNNSettings.getDirectIODirectMemoryBudget();
            budget = configured == null ? 0L : configured.getBytes();
        } catch (RuntimeException e) {
            log.debug("Could not read knn.direct_io.direct_memory_budget; not bounding this allocation", e);
            return true;
        }
        if (budget <= 0) {
            return true;
        }
        return live + allocationBytes(usableBytes, blockSize) <= budget;
    }

    /**
     * Opens a burst at whichever of its two entry points arrives first: {@link #selectUncachedOrdinals}, which
     * the scorer's prefetch helper calls once per batch, or {@link #prefetch}.
     *
     * <p>A burst has no explicit boundary, so it is recovered: the first of these after any read ends the
     * previous burst. Same move the rescore seam's {@code stage()} made by calling {@code abandonBatch()} on
     * entry. The settings refresh then runs exactly once per burst, which is the same cadence and the same
     * contract as {@link DirectIOReadPool#executor()}: an operator's change to
     * {@code knn.direct_io.rescore.cache.max_bytes} or {@code .span_fill} takes effect on the next burst with
     * no restart.
     */
    private void beginBurst() {
        if (readSincePrefetch) {
            endBurst();
        }
        if (burstConfigured == false) {
            burstConfigured = true;
            // The route first, because applyConfiguredSize releases the store's slabs when this input has
            // gone over to its delegate and there is no reason for this burst to look the store up after
            // that.
            refreshRoute();
            DirectIOVectorCache.applyConfiguredSize(directIOActive);
            // O5, after the size refresh so the fit test is against the target the operator has just set.
            burstReadAroundBytes = directIOActive ? DirectIOVectorCache.readAroundBytes(handle.fileLength) : 0;
        }
    }

    /**
     * Re-reads {@code knn.direct_io.rescore.enabled} and changes this input's route if it has to. O1, D2.
     *
     * <p>Called once per prefetch burst from {@link #beginBurst()}, which is the only boundary at which a
     * route may change: within a burst the staged ranges and the read that consumes them have to be on the
     * same channel, and between bursts nothing is in flight. An input with no fallback stays on
     * {@code O_DIRECT} whatever the setting says, because the alternative would be a read with nowhere to go.
     *
     * <p>Both of this input's burst entry points reach it: {@link #selectUncachedOrdinals}, which N1's
     * prefetch filter calls once per batch, and {@link #prefetch}. That matters for the rollback, and it is
     * where this port differs from 16-I's: 16-I refreshed the route in {@code prefetch} alone, and with a
     * warm cache a batch can make no {@code prefetch} call at all.
     *
     * <p>Never throws. A setting that cannot be read leaves the route alone, for the reason every other
     * settings read on this path has: a cluster-state read must not fail a query.
     */
    private void refreshRoute() {
        if (mmapFallback == null) {
            return;
        }
        final boolean enabled;
        try {
            enabled = routeEnabled.getAsBoolean();
        } catch (RuntimeException | LinkageError e) {
            log.debug("Could not read knn.direct_io.rescore.enabled; leaving this input on its current route", e);
            return;
        }
        if (enabled == directIOActive) {
            return;
        }
        directIOActive = enabled;
        STATS.recordSwitch(enabled);
        if (enabled == false) {
            // Nothing may be left in flight on a channel this input is about to stop reading: the ring's
            // windows are quiesced and the burst's lease is given up, exactly as close() does.
            quiesceReadahead();
            DirectIOStagingLease.current().releaseIfHeldBy(leaseGeneration);
            leaseGeneration = -1;
            leaseThreadId = -1;
            pool = null;
        }
        log.debug(
            "k-NN Direct I/O: [{}] switched to {} at a prefetch burst boundary, with no reopen",
            handle.path.getFileName(),
            enabled ? "O_DIRECT" : "the delegate input"
        );
    }

    /**
     * The input this read must go through instead of the {@code O_DIRECT} channel, or {@code null} to read
     * through the channel. Positions the delegate at this input's file pointer, so a caller that seeks and
     * reads sees one cursor whichever route is live.
     */
    @Nullable
    private IndexInput fallbackFor(final int bytes) throws IOException {
        if (directIOActive) {
            return null;
        }
        ensureOpen();
        // A read ends a burst on this route too, so that the next prefetch is recognised as a boundary and
        // the switch can be read again. Without it an input that went to the delegate could never come back.
        readSincePrefetch = true;
        STATS.recordFallbackRead(bytes);
        mmapFallback.seek(filePointer);
        return mmapFallback;
    }

    /**
     * Ends this input's burst: it stops consulting the staged table and stops holding the pool.
     *
     * <p><b>It does not quiesce anything, and that is the D3c change.</b> The ranges stay staged in the
     * thread's lease until the next burst on that thread takes the arena back and quiesces them there — which
     * is the only boundary that can be relied on, because a per-query input usually never prefetches twice.
     * Letting go of the generation is enough to make the table unreadable: {@link #heldLease} will not match
     * again until this input takes a new lease.
     */
    private void endBurst() {
        leaseGeneration = -1;
        leaseThreadId = -1;
        readSincePrefetch = false;
        burstConfigured = false;
        // O4: the filter's answers belong to the burst that asked. A blocking read in the next burst for an
        // offset this one dropped is not this race.
        filteredCount = 0;
        // The burst boundary is also where the pool reference is dropped, so the next burst picks up a pool
        // rebuilt by a capacity change. See ensureStagingTable.
        pool = null;
    }

    // ---------------------------------------------------------------------------------------------------
    // Reads
    // ---------------------------------------------------------------------------------------------------

    @Override
    public byte readByte() throws IOException {
        final IndexInput fallback = fallbackFor(1);
        if (fallback != null) {
            final byte served = fallback.readByte();
            filePointer++;
            STATS.servedBytes.incrementAndGet();
            return served;
        }
        final Source source = source(1);
        final byte value = source.buffer.get(source.position);
        filePointer++;
        STATS.servedBytes.incrementAndGet();
        return value;
    }

    @Override
    public void readBytes(final byte[] destination, final int destinationOffset, final int count) throws IOException {
        final IndexInput fallback = fallbackFor(count);
        if (fallback != null) {
            fallback.readBytes(destination, destinationOffset, count);
            filePointer += count;
            STATS.servedBytes.addAndGet(count);
            return;
        }
        int written = 0;
        while (written < count) {
            final Source source = source(Math.min(count - written, 1));
            final int available = Math.min(count - written, source.available);
            source.buffer.get(source.position, destination, destinationOffset + written, available);
            filePointer += available;
            written += available;
        }
        STATS.servedBytes.addAndGet(count);
    }

    /**
     * Reads {@code count} floats, which on the {@code .vec} path is the whole of one vector and the only
     * read shape that matters for latency.
     *
     * <p>Bulk-copies through a {@code FloatBuffer} view when the vector starts four-byte aligned inside the
     * buffer holding it, and falls back to a float at a time when it does not — the same computed guard, for
     * the same reason, as the rescore seam's decode did: the alignment holds for every layout this meets,
     * and a layout where it does not must cost a slower decode rather than a wrong read.
     */
    @Override
    public void readFloats(final float[] destination, final int destinationOffset, final int count) throws IOException {
        final int requested = count * Float.BYTES;
        final long runStart = offset + filePointer;
        final IndexInput fallback = fallbackFor(requested);
        if (fallback != null) {
            // O1: the same bytes of the same file, decoded by Lucene's own reader. This is the mmap read
            // the dispatch rule would have produced had the switch been off when the file was opened, and
            // no lookup and no admission happen on this route: nothing is served from the store and nothing
            // is put into it while the switch is off.
            fallback.readFloats(destination, destinationOffset, count);
            filePointer += requested;
            STATS.servedBytes.addAndGet(requested);
            return;
        }
        if (DirectIOVectorCache.isOn()) {
            // The second lookup. The first was in prefetch, where a hit kept the range out of the burst; this
            // is the one that actually serves the bytes, and it is also the one that serves a vector nobody
            // declared (the exact-search path reads some ordinals without a prefetch). The EOF check comes
            // first, so a hit cannot serve a read the miss path would have refused.
            ensureOpen();
            if (filePointer + requested > length) {
                throw new EOFException("read past EOF: " + this);
            }
            // The store copies out of its slab into the caller's array under its own read lock, so no view of
            // slab memory reaches this method and a hit is the same little-endian decode a staged read does.
            if (DirectIOVectorCache.getFloats(handle.fileId, runStart, requested, destination, destinationOffset)) {
                filePointer += requested;
                // A cache hit is still a read, so the burst boundary is found exactly as before: without
                // this, a burst every vector of which was cached would never be quiesced.
                readSincePrefetch = true;
                STATS.servedBytes.addAndGet(requested);
                return;
            }
        }
        int written = 0;
        while (written < count) {
            // Ask for one float: whatever buffer holds it will usually hold the rest of the vector too, and
            // asking for the whole run would refuse a window that legitimately ends mid vector.
            final Source source = source(Float.BYTES);
            final int usableFloats = Math.min(count - written, source.available / Float.BYTES);
            if (source.position % Float.BYTES == 0) {
                source.floats.get(source.position / Float.BYTES, destination, destinationOffset + written, usableFloats);
            } else {
                for (int i = 0; i < usableFloats; i++) {
                    destination[destinationOffset + written + i] = source.buffer.getFloat(source.position + (i << 2));
                }
            }
            // Populate on consume, and only from a staged read that covered the whole run in one buffer.
            // Whole-run because an entry must be exactly the vector the next lookup will ask for; staged
            // because that is the random rescore channel -- the ring is sequential, where a cache buys
            // nothing and would evict the random path's working set, and a blocking read is already the
            // fallback this is trying to avoid.
            if (written == 0 && usableFloats == count && source.staged()) {
                DirectIOVectorCache.put(handle.fileId, runStart, source.buffer, source.position, requested);
                // And, while the store is below its slab target, the rest of the whole vectors this span
                // already holds (M4). Done here rather than anywhere else because this is the one point
                // where the bytes are known to be valid, the vector length is known, and the range is known.
                spanFill(source.range, runStart, requested);
            } else if (written == 0 && usableFloats == count && wasFilteredAsCached(runStart)) {
                // O4: the filter-then-read race. This vector was cached when N1's filter checked it, so it
                // was not staged, and it is not cached now -- the store lost it in between and this read went
                // to the device synchronously. Keeping its bytes is what stops it staying a miss. It is
                // admitted without evicting anything; see DirectIOVectorCache#readmitAfterRace for why this
                // is not the "a blocking read is not admitted" rule being reopened.
                DirectIOVectorCache.readmitAfterRace(handle.fileId, runStart, source.buffer, source.position, requested);
            }
            filePointer += (long) usableFloats * Float.BYTES;
            written += usableFloats;
        }
        STATS.servedBytes.addAndGet(requested);
    }

    /**
     * Admits the whole vectors of this slice that {@code range} holds besides the one just read, while the
     * store is below its slab target. M4, ported from phase 16-I's F5.
     *
     * <h4>Why it exists (MEASURED by 16-I on the EBS node, S1b)</h4>
     * On dio-1m, with a cache large enough for every vector in the file, Direct I/O's p50 stayed 9.7x mmap's
     * because 16% of candidates still missed after a 10,000-query warm-up. In that warm-up Direct I/O read
     * 2.01 MB per query in about 40 KB staged spans, 6.5x the whole file, and cached only the vectors a
     * candidate had asked for; mmap's page cache kept every page it read and ended up holding 95% of the
     * file. The cache was paying for the reads and keeping a fraction of what they delivered. With span fill
     * on, dio-1m's p50 went from 43.9 to 10.9 ms.
     *
     * <h4>Four rules, each of them a thing this must not do</h4>
     * <ul>
     *   <li><b>Below the target only.</b> {@link DirectIOVectorCache#putRun} places with eviction refused and
     *       re-checks the target under the write lock, so span fill can neither evict a requested vector nor
     *       take a slab the target does not already allow. At the target, admission is requested vectors
     *       only, which is exactly M3's policy. That is what keeps this from being a scan that pushes out
     *       the working set (agenda item 4, which this phase does not decide).</li>
     *   <li><b>Whole vectors of the reading slice only.</b> A staged range is block aligned in the whole
     *       file, so its ends routinely reach outside this slice, into another field's vectors or into the
     *       file's header or footer. The span is clipped to {@code [offset, offset + length)} <em>before</em>
     *       any boundary is computed, the first boundary is rounded up and the last down, so no partial
     *       vector and no byte outside the slice is ever admitted.</li>
     *   <li><b>Only bytes the read returned.</b> The span ends at {@code alignedStart + valid}, not at
     *       {@code alignedStart + alignedLength}, so a short read at end of file cannot offer whatever the
     *       buffer held before.</li>
     *   <li><b>Staged spans only, once per range, one lock.</b> The caller is the burst-table branch of
     *       {@link #readFloats}, so nothing changes for mmap, for the read-ahead ring or for a blocking read;
     *       {@link StagedRange#spanFilled} makes it one walk per range however many vectors are read out of
     *       it; and the whole run goes in under a single write-lock acquisition rather than one per vector.
     *       The copy happens here, while the staged bytes are still this burst's.</li>
     * </ul>
     *
     * <h4>The boundary grid is anchored on the vector just read, not on the slice start</h4>
     * 16-I's F5 computed ordinals from the slice start, which is the same grid whenever the slice begins at
     * vector zero -- which is what {@code OffHeapFloatVectorValues} gives it. Anchoring on {@code runStart}
     * instead makes the grid right for <em>any</em> caller: the keys admitted are then exactly the offsets
     * some later {@code readFloats} of this field will ask for, including the case where a slice starts part
     * way into a field. The requested vector is at {@code k = 0} and is always inside the clipped span,
     * because it was just read out of it, so the run is never empty and always contains it (the store skips
     * it, having just been given it).
     *
     * @param range the staged range the read was served from
     * @param runStart the absolute file offset of the vector just read
     * @param vectorBytes that vector's length in bytes, which is the field's
     */
    private void spanFill(final StagedRange range, final long runStart, final int vectorBytes) {
        if (range == null || range.spanFilled || vectorBytes <= 0 || DirectIOVectorCache.isSpanFillOn() == false
        // O3: or a closed file has given slots back, which is room nothing was pushed out to get.
        // Approximate on purpose, because this runs before the write lock; putRun re-asks precisely.
            || DirectIOVectorCache.mayAdmitSpanFillApproximately() == false) {
            return;
        }
        // Set before the geometry, so a range whose span offers nothing is not walked again by the next
        // vector read out of it.
        range.spanFilled = true;
        final long from = Math.max(range.alignedStart, offset);
        final long to = Math.min(range.alignedStart + range.valid, offset + length);
        if (to - from < vectorBytes) {
            return;
        }
        // First boundary at or after `from`, last vector ending at or before `to`. floorDiv because the
        // grid is anchored on runStart and the span can start before it.
        final long firstIndex = -Math.floorDiv(runStart - from, (long) vectorBytes);
        final long lastIndex = Math.floorDiv(to - runStart, (long) vectorBytes) - 1;
        if (lastIndex < firstIndex) {
            return;
        }
        final long firstOffset = runStart + firstIndex * vectorBytes;
        final long count = lastIndex - firstIndex + 1;
        if (firstOffset < from || firstOffset + count * vectorBytes > to || count > Integer.MAX_VALUE) {
            // Not reachable from the arithmetic above; a bound rather than a comment, because the next thing
            // that happens is a copy out of a staging buffer.
            return;
        }
        final int position = Math.toIntExact(firstOffset - range.alignedStart);
        DirectIOVectorCache.putRun(handle.fileId, firstOffset, (int) count, range.buffer, position, vectorBytes);
    }

    /** Where the bytes at the current file pointer live: a buffer, a position in it, and how much follows. */
    private static final class Source {
        private final ByteBuffer buffer;
        private final FloatBuffer floats;
        private final int position;
        private final int available;
        /**
         * The burst-table range these bytes came from, or null for the read-ahead ring and for a blocking
         * read. Non-null <em>is</em> "this is the random rescore channel", which is the only channel that
         * populates {@link DirectIOVectorCache}: the ring is sequential, where a cache buys nothing and
         * would evict the random path's working set, and a blocking read is the fallback the cache exists to
         * avoid rather than a thing to remember.
         *
         * <p>It is the range and not a flag because span fill (M4) needs the range's geometry -- its aligned
         * start and the bytes the read actually returned -- and needs to mark it walked.
         */
        private final StagedRange range;

        private Source(final ByteBuffer buffer, final FloatBuffer floats, final int position, final int available) {
            this(buffer, floats, position, available, null);
        }

        private Source(
            final ByteBuffer buffer,
            final FloatBuffer floats,
            final int position,
            final int available,
            final StagedRange range
        ) {
            this.buffer = buffer;
            this.floats = floats;
            this.position = position;
            this.available = available;
            this.range = range;
        }

        /** Whether these bytes came from the prefetch burst table. */
        private boolean staged() {
            return range != null;
        }
    }

    /**
     * Resolves the current file pointer to a buffer holding at least {@code needed} bytes of it. Three
     * sources are consulted in order of how specific each is: the prefetch burst table (caller-declared,
     * random), then the sequential read-ahead ring (inferred from a forward run), then one blocking
     * {@code pread}. The two staging mechanisms are independent — neither resets the other, and the burst
     * table is scanned first because a {@link #prefetch} is a stronger signal than a detected run.
     *
     * <p>Since D3c the burst table belongs to the <em>thread</em>, so it is consulted only when this input
     * still holds its thread's lease. Two ways it may not, both of which fall through to a blocking read with
     * identical bytes: another burst on this thread took the arena back (the generation moved on), or this
     * read arrived on a different thread from the one that prefetched, which is counted as
     * {@code foreign_thread_reads} because it is the premise of the lease model being violated rather than
     * the lease model working.
     */
    private Source source(final int needed) throws IOException {
        ensureOpen();
        if (filePointer + needed > length) {
            throw new EOFException("read past EOF: " + this);
        }
        final long absolute = offset + filePointer;
        readSincePrefetch = true;
        if (leaseGeneration >= 0 && Thread.currentThread().threadId() != leaseThreadId) {
            STATS.recordForeignThreadRead();
        }
        final DirectIOStagingLease lease = heldLease();
        if (lease != null) {
            final StagedRange range = lease.covering(absolute, needed);
            if (range != null && await(range)) {
                final int position = Math.toIntExact(absolute - range.alignedStart);
                // A read that came up short of these bytes can only have done so at end of file. Fall
                // through to a blocking read rather than decode whatever the buffer last held.
                if (position + needed <= range.valid) {
                    STATS.stagedHits.incrementAndGet();
                    return new Source(range.buffer, range.floats, position, range.valid - position, range);
                }
            }
        }
        if (readaheadWindows > 0 && readaheadRing != null) {
            final Source ahead = readaheadSource(absolute, needed);
            if (ahead != null) {
                return ahead;
            }
        }
        return blockingSource(absolute, needed);
    }

    /**
     * Waits for a staged read, for at most {@link #STAGED_READ_WAIT_MILLIS}.
     *
     * <p><b>No search thread may wait forever on a pool future.</b> That is the property this bound exists
     * for, and it was not previously held: this method used an unbounded {@code Future#get}, so a read that
     * never completed — because the pool was saturated, because the queue in front of it was unbounded, or
     * because the device stopped answering — held the calling search thread with no bound and no
     * cancellation check. {@code SearchBackpressure} cancellation is cooperative, so such a thread is not
     * recoverable; the Oct-2 incident's search tasks logged elapsed times of 2.1 hours.
     *
     * <p>On timeout the range is {@link StagedRange#abandoned}, which makes it {@link StagedRange#covers}
     * nothing, and the caller falls through to {@link #blockingSource} — the same fall-through the
     * short-read-at-EOF case already uses. The read may still be running and may still write into the
     * range's buffer, so that buffer is retired rather than reused; {@link #releaseAbandoned} is where
     * that happens.
     *
     * @return true if the bytes are in the range's buffer, false if the caller must read them itself
     */
    private boolean await(final StagedRange range) throws IOException {
        try {
            range.future.get(STAGED_READ_WAIT_MILLIS, TimeUnit.MILLISECONDS);
            return true;
        } catch (TimeoutException e) {
            range.abandoned = true;
            STATS.stagedWaitTimeouts.incrementAndGet();
            log.debug(
                "Gave up after {} ms waiting for a staged Direct I/O read of {} at {}; reading it on this thread",
                STAGED_READ_WAIT_MILLIS,
                handle.path,
                range.alignedStart
            );
            return false;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("Interrupted waiting for a Direct I/O read of " + handle.path, e);
        } catch (ExecutionException e) {
            final Throwable cause = e.getCause();
            // Error propagates rather than being caught: a direct-buffer OutOfMemoryError is a
            // misconfiguration to fix, not a condition to read around.
            if (cause instanceof Error) {
                throw (Error) cause;
            }
            if (cause instanceof IOException) {
                throw (IOException) cause;
            }
            throw new IOException("Direct I/O read of " + handle.path + " failed", cause);
        }
    }

    /**
     * One blocking {@code pread} of the block-aligned window holding {@code absolute}, when not already held.
     *
     * <h4>This path's own buffer is bounded, and the budget deliberately does not decline it</h4>
     * It is <b>exactly one</b> buffer of {@link #bufferSize} (8 KiB at the shipped default, two filesystem
     * blocks) per <b>thread</b> since D3c — it was one per input — allocated on that thread's first blocking
     * read and then reused: the window check below serves every later read that falls inside the window it
     * already holds, and a read outside it re-reads into the <em>same</em> buffer rather than allocating. It
     * is never grown and never allocated per read. So the blocking channel's whole direct footprint is
     * {@code bufferSize x live threads}, where before it was {@code bufferSize x live inputs} and inputs
     * outlive their queries.
     *
     * <p>Sharing it is safe because the bytes are consumed before the next read on that thread: the caller
     * copies out of the returned {@link Source} inside the same {@code readFloats} or {@code readBytes}
     * iteration. What is <em>not</em> safe to share is the belief that the window still holds a given range,
     * so the lease remembers which input filled it ({@link DirectIOStagingLease#windowPosition}) and will not
     * answer a cached position to any other.
     *
     * <p>It is charged to {@code knn.direct_io.direct_memory_budget}'s accounting (so {@code live_bytes} is
     * the truth about this class's direct memory) but is <b>never declined</b> by it, and that is forced
     * rather than chosen: this is the path every other decline falls back <em>to</em>. Declining it would
     * leave a read with nothing to read with, which is a failed query rather than a bounded one.
     */
    private Source blockingSource(final long absolute, final int needed) throws IOException {
        final DirectIOStagingLease lease = DirectIOStagingLease.current();
        // Over-allocate by a block so a block-aligned slice exists inside the allocation: O_DIRECT needs the
        // buffer address, the file offset and the length all block aligned.
        final ByteBuffer syncBuffer = lease.window(bufferSize, handle.blockSize);
        final int cached = lease.windowPosition(inputId, absolute, needed);
        if (cached >= 0) {
            return new Source(syncBuffer, lease.windowFloats(), cached, lease.windowValid() - cached);
        }
        final int blockSize = handle.blockSize;
        final long alignedStart = absolute - Math.floorMod(absolute, (long) blockSize);
        syncBuffer.clear().limit(bufferSize);
        final int read = handle.channel.read(syncBuffer, alignedStart);
        STATS.blockingReads.incrementAndGet();
        STATS.blockingBytes.addAndGet(Math.max(read, 0));
        final int syncValid = Math.max(read, 0);
        lease.windowFilled(inputId, alignedStart, syncValid);
        // Detection runs on the device read, not the sync-window hit above: a hit means no new device read
        // and so no advance of the file-order frontier. With read-ahead off this is a no-op (gated inside).
        noteBlockingRead(alignedStart);
        final int position = Math.toIntExact(absolute - alignedStart);
        if (position + needed > syncValid) {
            throw new IOException(
                String.format(
                    Locale.ROOT,
                    "Direct I/O read of %s at %d returned %d bytes, need %d",
                    handle.path,
                    alignedStart,
                    read,
                    position + needed
                )
            );
        }
        return new Source(syncBuffer, lease.windowFloats(), position, syncValid - position);
    }

    // ---------------------------------------------------------------------------------------------------
    // Sequential read-ahead ring
    //
    // A rolling ring, structurally separate from the burst table: it shares no field with it, is consulted
    // independently in source(), and neither mechanism resets the other. The burst table serves a
    // caller-declared random pattern (prefetch); this serves a forward run inferred from blocking reads,
    // which is the one shape prefetch never covers — a merge, the faiss-unquantized first pass, traversal.
    // ---------------------------------------------------------------------------------------------------

    /**
     * Records a blocking device read against the file-order frontier and, once a gapless-forward run
     * reaches {@link #readaheadTrigger}, arms the ring and keeps it refilled ahead of the frontier.
     *
     * <p>The detection predicate is forward-and-gapless: {@code alignedStart} is at or behind the previous
     * window's start (so not a backward jump) and at or before its end (so not a gap). The overlap case
     * {@code blockingStart <= alignedStart < blockingEnd} is the legitimate re-read a sequential scan makes
     * when a vector straddles a window boundary ({@link #readFloats} asks for four bytes, misses by a few,
     * and re-floors into the window it just read); it counts as sequential. A gap or backward jump resets
     * the run and quiesces whatever the ring had in flight — this is the seek-away reset, driven by file
     * order at this layer rather than by {@link #seek}, because the faiss first pass {@code seek}s per
     * ordinal <em>while remaining sequential</em> and a blanket seek reset would disarm it.
     *
     * <p>With {@link #readaheadWindows} zero this returns immediately and keeps no state, so the input is
     * exact passthrough.
     */
    private void noteBlockingRead(final long alignedStart) {
        if (readaheadWindows == 0) {
            return;
        }
        final long alignedEnd = alignedStart + bufferSize;
        final boolean sequential = blockingEnd >= 0 && alignedStart >= blockingStart && alignedStart <= blockingEnd;
        if (sequential) {
            if (sequentialRun < Integer.MAX_VALUE) {
                sequentialRun++;
            }
        } else {
            // A discontinuity: abandon the now-wrong windows and start a fresh run at this read.
            if (readaheadCount > 0) {
                quiesceReadahead();
            }
            sequentialRun = 1;
        }
        blockingStart = alignedStart;
        blockingEnd = alignedEnd;
        if (sequentialRun < readaheadTrigger) {
            return;
        }
        if (sequentialRun == readaheadTrigger) {
            STATS.readaheadArmed.incrementAndGet();
        }
        // Armed: point the ring's frontier just past this window and fill it. When the ring is empty this is
        // the arm point outright; when windows are already live the frontier is whichever is further ahead,
        // because those windows were submitted from the old frontier and re-submitting behind them would
        // fetch bytes the ring already holds.
        readaheadNext = readaheadCount == 0 ? alignedEnd : Math.max(readaheadNext, alignedEnd);
        refillReadahead();
    }

    /**
     * Serves {@code [absolute, absolute + needed)} from the ring if a live window covers it, else returns
     * null to fall through to a blocking read. Retires windows fully behind the read, keeps the ring
     * ordered, and refills after a hit so ~{@link #readaheadWindows} windows stay in flight ahead.
     */
    private Source readaheadSource(final long absolute, final int needed) throws IOException {
        // Retire head windows the read has moved past, so the ring stays ordered oldest-first.
        while (readaheadCount > 0) {
            final StagedRange head = readaheadRing[readaheadHead];
            if (head == null || head.alignedStart + head.alignedLength > absolute) {
                break;
            }
            retireHead();
        }
        for (int i = 0; i < readaheadCount; i++) {
            final int slot = (readaheadHead + i) % readaheadWindows;
            final StagedRange range = readaheadRing[slot];
            if (range == null || range.covers(absolute, needed) == false) {
                continue;
            }
            if (await(range) == false) {
                // The bounded wait gave up. Let a blocking read handle these bytes.
                return null;
            }
            final int position = Math.toIntExact(absolute - range.alignedStart);
            if (position + needed > range.valid) {
                // Came up short — only at EOF. Let a blocking read handle these bytes.
                return null;
            }
            STATS.readaheadHits.incrementAndGet();
            // Advance the detection frontier to the window just served so a later seek-away is seen as a
            // discontinuity against where we actually are, not against the stale arm-point frontier.
            blockingStart = range.alignedStart;
            blockingEnd = range.alignedStart + range.alignedLength;
            refillReadahead();
            return new Source(range.buffer, range.floats, position, range.valid - position);
        }
        return null;
    }

    /**
     * Submits windows until {@link #readaheadWindows} are in flight ahead of {@link #readaheadNext}, within
     * this inputs own region. Stands down while this input holds a live burst on this thread: the
     * rescore scorer then owns the input and its access is random, not sequential, so the ring issues no
     * speculative reads. Allocates the ring lazily on the first call that reaches here.
     *
     * <p>The pool is acquired into a <b>local</b> here rather than kept in {@link #pool}, which the burst
     * path owns and clears at each burst boundary. The ring has no burst boundary to re-acquire at — it
     * lives for the length of a scan — so re-reading it on every refill is what makes a live capacity change
     * reach the ring too. A refill happens once per window served, i.e. once per
     * {@link #readaheadWindowBytes} (64 KiB by default), so the settings read is not on a per-vector path.
     */
    private void refillReadahead() {
        final DirectIOStagingLease held = heldLease();
        if (held != null && held.count() > 0) {
            return;
        }
        final ExecutorService executor = DirectIOReadPool.executor();
        if (executor == null) {
            STATS.readaheadDeclined.incrementAndGet();
            return;
        }
        if (readaheadRing == null) {
            // The ring is direct memory too, so it is charged to the same budget. Declining it leaves the
            // scan at queue depth one, which is readaheadDeclined's existing meaning.
            if (fitsBudget((long) readaheadWindows * readaheadWindowBytes, handle.blockSize) == false) {
                STATS.recordDeclinedBudget();
                STATS.readaheadDeclined.incrementAndGet();
                log.debug(
                    "knn.direct_io.direct_memory_budget reached ({} B live); the read-ahead ring of {} is not armed",
                    STATS.directMemoryLive(),
                    handle.path
                );
                return;
            }
            readaheadRing = new StagedRange[readaheadWindows];
            readaheadBuffers = new ByteBuffer[readaheadWindows];
            readaheadFloats = new FloatBuffer[readaheadWindows];
            // Every ring window is exactly readaheadWindowBytes, so the ring carves its arena at that size
            // rather than at the burst table's arenaSlotBytes() and never needs the per-slot fall-back the
            // burst table keeps for a wide group.
            readaheadArena = carveArena(readaheadWindows, readaheadWindowBytes, handle.blockSize, readaheadBuffers, readaheadFloats);
        }
        final long regionEnd = offset + length;
        while (readaheadCount < readaheadWindows && readaheadNext < regionEnd) {
            final int slot = (readaheadHead + readaheadCount) % readaheadWindows;
            if (readaheadBuffers[slot] == null) {
                // Only reachable for a slot given up after a bounded wait timed out (see retireHead).
                if (fitsBudget(readaheadWindowBytes, handle.blockSize) == false) {
                    STATS.recordDeclinedBudget();
                    STATS.readaheadDeclined.incrementAndGet();
                    return;
                }
                readaheadBuffers[slot] = trackedAllocate(readaheadWindowBytes, handle.blockSize);
                readaheadFloats[slot] = readaheadBuffers[slot].asFloatBuffer();
            }
            final StagedRange range = new StagedRange(
                handle.channel,
                handle.path,
                readaheadBuffers[slot],
                readaheadFloats[slot],
                readaheadNext,
                readaheadWindowBytes,
                STATS.readaheadReads,
                STATS.readaheadBytes
            );
            try {
                range.future = executor.submit(range);
            } catch (RejectedExecutionException e) {
                STATS.readaheadDeclined.incrementAndGet();
                log.debug("The Direct I/O read pool refused a read-ahead of {}; reads will block", handle.path);
                return;
            }
            DirectIOReadPool.sampleQueueDepth(executor);
            readaheadRing[slot] = range;
            readaheadCount++;
            readaheadNext += readaheadWindowBytes;
        }
    }

    /**
     * Claims and forgets the oldest live window, keeping its buffer for reuse — unless the bounded quiesce
     * gave up on it, in which case the buffer is given up with it, because a read may still be writing
     * into it.
     */
    private void retireHead() {
        final StagedRange head = readaheadRing[readaheadHead];
        readaheadRing[readaheadHead] = null;
        if (head != null && head.quiesce() == false) {
            readaheadBuffers[readaheadHead] = null;
            readaheadFloats[readaheadHead] = null;
        }
        readaheadHead = (readaheadHead + 1) % readaheadWindows;
        readaheadCount--;
    }

    /**
     * Quiesces every window the ring still has in flight and forgets them, keeping the buffers. Mirrors
     * {@link #endBurst} for the burst table, and like it does not touch the other mechanisms state —
     * the run counter and frontier are left to the caller ({@link #noteBlockingRead} resets the run; a
     * close just drops everything).
     */
    private void quiesceReadahead() {
        if (readaheadRing == null) {
            return;
        }
        while (readaheadCount > 0) {
            retireHead();
        }
        readaheadHead = 0;
    }

    // ---------------------------------------------------------------------------------------------------
    // IndexInput plumbing
    // ---------------------------------------------------------------------------------------------------

    @Override
    public long getFilePointer() {
        return filePointer;
    }

    @Override
    public void seek(final long position) throws IOException {
        ensureOpen();
        if (position < 0 || position > length) {
            throw new EOFException("seek past EOF: " + this);
        }
        filePointer = position;
    }

    @Override
    public long length() {
        return length;
    }

    /**
     * A view of {@code [offset, offset + length)} of this input, over the same {@code O_DIRECT} handle.
     *
     * <p>This is where "how the bytes are fetched" is inherited rather than re-decided. Lucene calls this
     * three-argument overload with no {@code IOContext}, so there is no per-slice signal to consult and no
     * path by which a slice of this input becomes an mmap read. A slice gets its own buffers and its own
     * staging table, because {@code OffHeapFloatVectorValues} holds one slice per values object and reads
     * it from one thread.
     */
    @Override
    public DirectIOVectorIndexInput slice(final String sliceDescription, final long offset, final long length) throws IOException {
        ensureOpen();
        if (offset < 0 || length < 0 || offset + length > this.length) {
            throw new IllegalArgumentException(
                String.format(
                    Locale.ROOT,
                    "slice(%s) out of bounds: offset=%d length=%d of %d",
                    sliceDescription,
                    offset,
                    length,
                    this.length
                )
            );
        }
        return new DirectIOVectorIndexInput(
            sliceDescription,
            handle,
            // The matching slice of the delegate's input, so that the two routes share this slice's
            // coordinates and filePointer addresses both. A slice of a Lucene input is owned by its parent
            // and is not closed on its own, which is the same contract the O_DIRECT handle has here.
            mmapFallback == null ? null : mmapFallback.slice(sliceDescription, offset, length),
            routeEnabled,
            this.offset + offset,
            length,
            false,
            bufferSize,
            maxStagedRanges,
            maxStagedRangeBytes,
            readaheadWindows,
            readaheadWindowBytes,
            readaheadTrigger
        );
    }

    /**
     * An independent cursor over the same bytes, with the same {@code O_DIRECT} handle and its own buffers.
     * Clones do not close the handle, which is the contract every {@link IndexInput} clone has.
     */
    @Override
    public DirectIOVectorIndexInput clone() {
        final DirectIOVectorIndexInput copy = new DirectIOVectorIndexInput(
            toString(),
            handle,
            mmapFallback == null ? null : mmapFallback.clone(),
            routeEnabled,
            offset,
            length,
            false,
            bufferSize,
            maxStagedRanges,
            maxStagedRangeBytes,
            readaheadWindows,
            readaheadWindowBytes,
            readaheadTrigger
        );
        copy.filePointer = filePointer;
        // The route this clone starts on is the one its parent is on, so a clone taken mid burst does not
        // change path on its first read; it re-reads the switch at its own next burst like any other input.
        copy.directIOActive = directIOActive;
        return copy;
    }

    @Override
    public void close() throws IOException {
        if (closed) {
            return;
        }
        closed = true;
        // Quiesce before the channel goes, if this input still holds its thread's lease. A range left staged
        // by an input that already gave the lease up is quiesced by the next take instead, and a read that
        // finds a closed channel fails rather than writing -- which StagedRange#quiesce treats as safe.
        DirectIOStagingLease.current().releaseIfHeldBy(leaseGeneration);
        endBurst();
        quiesceReadahead();
        if (ownsHandle) {
            if (mmapFallback != null) {
                // Opened by the directory alongside the channel and owned with it, so the two routes are
                // released together and a rollback can never find a half-closed input. Exactly once: only
                // the input open() returned has ownsHandle, and a clone or a slice never reaches here.
                mmapFallback.close();
            }
            handle.channel.close();
            // O3: the file is closed for good, so its cached vectors are cold for ever and the slots they
            // hold are room the store can have back without evicting anything. Only here, in the
            // ownsHandle branch: a clone or a slice closing would otherwise free bytes its parent is still
            // reading. After the channel close rather than before it, because nothing may read the file
            // through this input again either way and the cheaper of the two orderings is the one where a
            // concurrent lookup on another input's clone can still be served until the last moment.
            DirectIOVectorCache.releaseFile(handle.fileId);
            // One snapshot per file closed, and deliberately not per burst or per clone: clones close once
            // per query thread per segment, and a line there would be per-query log traffic inside the very
            // blocks whose latency is being measured. {@link #STATS} is node wide and cumulative, so a
            // measurement reads this line before and after a block and takes the difference — which is why
            // the message says so rather than leaving a reader to assume the numbers are this file's.
            log.debug("Direct I/O vector reads (node totals, cumulative) at close of [{}]: {}", handle.path.getFileName(), STATS);
        }
    }

    private void ensureOpen() throws IOException {
        if (closed) {
            throw new IOException("this IndexInput is closed: " + this);
        }
    }

    private static int alignUp(final int value, final int blockSize) {
        return ((value + blockSize - 1) / blockSize) * blockSize;
    }

    /**
     * {@code com.sun.nio.file.ExtendedOpenOption.DIRECT}, or {@code null} on a runtime that does not have
     * it.
     *
     * <p>Looked up reflectively for the reasons Lucene's {@code DirectIODirectory} gives for doing the
     * same: it is a proprietary OpenJDK API that emits an unsuppressible warning when referenced under
     * {@code --release}, and it does not link at all on runtimes without it. Lucene's own copy of the
     * constant is private with no accessor.
     */
    static OpenOption directOpenOption() {
        try {
            final Class<? extends OpenOption> clazz = Class.forName("com.sun.nio.file.ExtendedOpenOption").asSubclass(OpenOption.class);
            for (final OpenOption option : clazz.getEnumConstants()) {
                if (option.toString().equalsIgnoreCase("DIRECT")) {
                    return option;
                }
            }
            return null;
        } catch (ClassNotFoundException | RuntimeException e) {
            return null;
        }
    }
}

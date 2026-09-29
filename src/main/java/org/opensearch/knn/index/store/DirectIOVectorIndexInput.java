/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.store.IndexInput;
import org.opensearch.core.common.unit.ByteSizeValue;
import org.opensearch.knn.index.KNNSettings;

import java.io.EOFException;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.OpenOption;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Locale;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

/**
 * An {@link IndexInput} that fetches with {@code O_DIRECT} and implements {@link #prefetch} by putting
 * block-aligned reads in flight on {@link DirectIOReadPool}.
 *
 * <p><b>This class exists to answer one question</b> — Phase 9b gate 2. The rescore seam reaches its queue
 * depth through a seam of its own ({@link VectorStagingArea#stage}, driven by a query-layer scorer wrapper),
 * because Lucene's {@code DirectIOIndexInput} has no {@code prefetch} member at all and
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
 *       {@code DirectIOVectorSource.Reader#stage} does with {@code abandonBatch()} on entry.</li>
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
 *       {@link IndexInput} directly is the other one: {@link FilterIndexInput} delegates only
 *       {@code readByte} and {@code readBytes}, so it inherits {@code IndexInput}'s no-op
 *       {@link #prefetch} and {@code DataInput}'s per-{@code float} {@code readFloats} loop — a wrapper
 *       would have to override the very methods this class exists to implement.</li>
 * </ul>
 * The values object, by contrast, <em>keeps</em> {@code HasIndexSlice}, which is how the prefetch burst
 * above reaches us. That is strictly better placed than the seam, which had to hide {@code HasIndexSlice}
 * to keep the SIMD scorer off and therefore had to rebuild read-ahead in the query layer.
 *
 * <h2>Direct I/O is a property of this object, inherited</h2>
 * An {@code IOContext} does not survive the {@link #slice} Lucene uses <em>on a non-compound segment</em>:
 * {@code OffHeapFloatVectorValues} calls the three-argument overload, which has no context parameter. The
 * context-carrying four-argument overload does have one call site — {@code Lucene90CompoundReader#openInput},
 * which is how every entry of every compound segment is opened, and which gate 3 turns into the
 * compound-segment dispatch point (see {@link KNNVectorCompoundSliceInput}). Outside a compound file,
 * though, "these bytes are fetched with {@code O_DIRECT}" cannot be re-decided per slice and must be
 * carried by the object. {@link #slice} and {@link #clone} therefore return instances over the same
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

        private Handle(final Path path, final FileChannel channel, final int blockSize, final long fileLength) {
            this.path = path;
            this.channel = channel;
            this.blockSize = blockSize;
            this.fileLength = fileLength;
        }
    }

    /**
     * One block-aligned range staged by {@link #prefetch}, keyed by the range it holds rather than by a
     * position in a ring.
     *
     * <p>The claim is what makes a buffer reusable, and it is {@code DirectIOVectorSource.StagedRead}'s
     * argument verbatim: {@code Future#cancel} completes the future while a started read runs on, so a read
     * abandoned before it began is skipped by the claim and one that has begun is waited out.
     */
    private static final class StagedRange implements Callable<Void> {
        private final FileChannel channel;
        private final Path path;
        private final ByteBuffer buffer;
        /** {@link #buffer}'s float view, made once with the buffer so that a read decodes allocation free. */
        private final FloatBuffer floats;
        private final long alignedStart;
        private final int alignedLength;
        private final AtomicBoolean started = new AtomicBoolean();
        private Future<?> future;
        /** Bytes the read actually delivered; only meaningful once {@link #future} has completed. */
        private int valid;

        private StagedRange(
            final FileChannel channel,
            final Path path,
            final ByteBuffer buffer,
            final FloatBuffer floats,
            final long alignedStart,
            final int alignedLength
        ) {
            this.channel = channel;
            this.path = path;
            this.buffer = buffer;
            this.floats = floats;
            this.alignedStart = alignedStart;
            this.alignedLength = alignedLength;
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
                STATS.stagedReads.incrementAndGet();
                STATS.stagedBytes.addAndGet(Math.max(valid, 0));
            } finally {
                STATS.inFlightNow.decrementAndGet();
            }
            return null;
        }

        /** @return true if this read will now never run, false if it is running or has already run */
        private boolean skipIfNotStarted() {
            return started.compareAndSet(false, true);
        }

        /** Whether this range fully contains {@code [absolute, absolute + needed)}. */
        private boolean covers(final long absolute, final int needed) {
            return absolute >= alignedStart && absolute + needed <= alignedStart + alignedLength;
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
     */
    public static final class Stats {
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

        private void recordInFlightPeak() {
            final long now = inFlightNow.get();
            long peak = inFlightPeak.get();
            while (now > peak && inFlightPeak.compareAndSet(peak, now) == false) {
                peak = inFlightPeak.get();
            }
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

        public void reset() {
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
        }

        @Override
        public String toString() {
            final long read = stagedBytes.get() + blockingBytes.get();
            final long served = servedBytes.get();
            return String.format(
                Locale.ROOT,
                "blockingReads=%d stagedReads=%d stagedHits=%d stageDeclined=%d prefetchCalls=%d inFlightPeak=%d "
                    + "stagedBytes=%d blockingBytes=%d servedBytes=%d declinedSpanBytes=%d amplification=%.3f",
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
                served == 0 ? 0.0 : (double) read / served
            );
        }
    }

    /** The one counter set, because the pool it counts is node wide too. */
    public static final Stats STATS = new Stats();

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

    private final Handle handle;
    /** Absolute file offset of this input's byte zero, so a slice needs no other translation. */
    private final long offset;
    private final long length;
    /** Only the input returned by {@link #open} closes the shared channel. */
    private final boolean ownsHandle;
    private final int bufferSize;
    private final int maxStagedRanges;
    private final int maxStagedRangeBytes;

    private long filePointer;
    private boolean closed;

    /** Buffer for reads not served from a staged range. Allocated on the first such read. */
    private ByteBuffer syncBuffer;
    private FloatBuffer syncFloats;
    private long syncStart = -1;
    private int syncValid;

    /** The current burst's staged ranges, or null before the first {@link #prefetch}. */
    private StagedRange[] staged;
    private ByteBuffer[] stagedBuffers;
    private FloatBuffer[] stagedFloats;
    private int stagedCount;
    private ExecutorService pool;
    /** Whether a read has happened since the last {@link #prefetch}, which is how a burst's end is found. */
    private boolean readSincePrefetch;

    private DirectIOVectorIndexInput(
        final String resourceDescription,
        final Handle handle,
        final long offset,
        final long length,
        final boolean ownsHandle,
        final int bufferSize,
        final int maxStagedRanges,
        final int maxStagedRangeBytes
    ) {
        super(resourceDescription);
        this.handle = handle;
        this.offset = offset;
        this.length = length;
        this.ownsHandle = ownsHandle;
        this.bufferSize = bufferSize;
        this.maxStagedRanges = maxStagedRanges;
        this.maxStagedRangeBytes = maxStagedRangeBytes;
    }

    /**
     * Opens {@code path} with {@code O_DIRECT} and returns an input over the whole file, or {@code null}
     * when Direct I/O cannot serve it — no {@code ExtendedOpenOption.DIRECT} on this JDK, a filesystem that
     * answers {@code EINVAL}, or a path that is not a regular file. Returning {@code null} rather than
     * throwing is the same fallback contract {@link DirectIOVectorSource#open} has: a file this cannot serve
     * must cost a fall back to the default path, never a failed query.
     */
    public static DirectIOVectorIndexInput open(final Path path) {
        return open(path, 0, configuredMaxStagedRanges(), configuredMaxStagedRangeBytes());
    }

    /**
     * How many ranges one burst may hold in flight, from
     * {@code knn.direct_io.rescore.prefetch.staged_ranges}.
     *
     * <p>Its own setting rather than the rescore seam's {@code prefetch_window}, which task-15 first tried:
     * the seam's window is a rolling ring refilled as it drains, this is a table that must <b>hold a whole
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
        final OpenOption direct = DirectIOVectorSource.directOpenOption();
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
                0L,
                fileLength,
                true,
                reads,
                Math.max(1, maxStagedRanges),
                Math.max(blockSize, maxStagedRangeBytes)
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

    // ---------------------------------------------------------------------------------------------------
    // Read ahead
    // ---------------------------------------------------------------------------------------------------

    /**
     * Puts the block-aligned range covering {@code [offset, offset + length)} in flight on
     * {@link DirectIOReadPool}.
     *
     * <p>Advisory in both directions, the same contract {@link VectorStagingArea#stage} states: this may
     * stage nothing, and a caller that then reads something else, or reads out of order, still gets correct
     * bytes from blocking reads. No returned value tells the caller which happened, so nothing can come to
     * depend on it.
     *
     * <h2>The span bound is the waste guard, and it is a bound on the span because it cannot be a bound
     * on the waste</h2>
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
        // A burst has no explicit boundary, so it is recovered: the first prefetch after any read ends the
        // previous burst. Same move as DirectIOVectorSource.Reader#stage calling abandonBatch() on entry.
        if (readSincePrefetch) {
            resetStaging();
        }
        final long absolute = this.offset + offset;
        final int blockSize = handle.blockSize;
        final long alignedStart = absolute - Math.floorMod(absolute, (long) blockSize);
        final int alignedLength = alignUp(Math.toIntExact(absolute + length - alignedStart), blockSize);
        if (alignedLength > maxStagedRangeBytes) {
            STATS.stageDeclined.incrementAndGet();
            STATS.declinedSpanBytes.addAndGet(length);
            return;
        }
        if (staged != null) {
            for (int i = 0; i < stagedCount; i++) {
                if (staged[i].covers(absolute, Math.toIntExact(length))) {
                    // Already in flight from an earlier call in this burst. PrefetchHelper's grouping makes
                    // this common rather than exceptional: one group covers every vector inside it.
                    return;
                }
            }
        }
        if (ensureStagingTable() == false || stagedCount == maxStagedRanges) {
            STATS.stageDeclined.incrementAndGet();
            return;
        }
        final int slot = stagedCount;
        if (stagedBuffers[slot] == null || stagedBuffers[slot].capacity() < alignedLength) {
            stagedBuffers[slot] = ByteBuffer.allocateDirect(alignedLength + blockSize - 1)
                .alignedSlice(blockSize)
                .order(ByteOrder.LITTLE_ENDIAN);
            stagedFloats[slot] = stagedBuffers[slot].asFloatBuffer();
        }
        final StagedRange range = new StagedRange(
            handle.channel,
            handle.path,
            stagedBuffers[slot],
            stagedFloats[slot],
            alignedStart,
            alignedLength
        );
        try {
            range.future = pool.submit(range);
        } catch (RejectedExecutionException e) {
            STATS.stageDeclined.incrementAndGet();
            log.debug("The Direct I/O read pool refused a prefetch of {}; reads will block", handle.path);
            return;
        }
        staged[slot] = range;
        stagedCount++;
    }

    /** Allocates the burst table once. @return false when there is no pool, so reads must block */
    private boolean ensureStagingTable() {
        if (staged != null) {
            return pool != null;
        }
        final ExecutorService acquired = DirectIOReadPool.executor();
        if (acquired == null) {
            return false;
        }
        staged = new StagedRange[maxStagedRanges];
        stagedBuffers = new ByteBuffer[maxStagedRanges];
        stagedFloats = new FloatBuffer[maxStagedRanges];
        pool = acquired;
        return true;
    }

    /**
     * Quiesces every range the current burst still has in flight and forgets them, keeping the buffers for
     * the next burst. A buffer is only reusable once no read can still write into it, which is what the
     * claim in {@link StagedRange#skipIfNotStarted} establishes.
     */
    private void resetStaging() {
        if (staged != null) {
            for (int i = 0; i < stagedCount; i++) {
                final StagedRange range = staged[i];
                staged[i] = null;
                if (range == null || range.skipIfNotStarted()) {
                    continue;
                }
                try {
                    range.future.get();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                } catch (ExecutionException e) {
                    log.debug("Abandoned a staged Direct I/O read of {}", handle.path);
                }
            }
        }
        stagedCount = 0;
        readSincePrefetch = false;
    }

    // ---------------------------------------------------------------------------------------------------
    // Reads
    // ---------------------------------------------------------------------------------------------------

    @Override
    public byte readByte() throws IOException {
        final Source source = source(1);
        final byte value = source.buffer.get(source.position);
        filePointer++;
        STATS.servedBytes.incrementAndGet();
        return value;
    }

    @Override
    public void readBytes(final byte[] destination, final int destinationOffset, final int count) throws IOException {
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
     * the same reason, as {@code DirectIOVectorSource.Reader#decode}: the alignment holds for every layout
     * this meets, and a layout where it does not must cost a slower decode rather than a wrong read.
     */
    @Override
    public void readFloats(final float[] destination, final int destinationOffset, final int count) throws IOException {
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
            filePointer += (long) usableFloats * Float.BYTES;
            written += usableFloats;
        }
        STATS.servedBytes.addAndGet((long) count * Float.BYTES);
    }

    /** Where the bytes at the current file pointer live: a buffer, a position in it, and how much follows. */
    private static final class Source {
        private final ByteBuffer buffer;
        private final FloatBuffer floats;
        private final int position;
        private final int available;

        private Source(final ByteBuffer buffer, final FloatBuffer floats, final int position, final int available) {
            this.buffer = buffer;
            this.floats = floats;
            this.position = position;
            this.available = available;
        }
    }

    /**
     * Resolves the current file pointer to a buffer holding at least {@code needed} bytes of it — out of a
     * staged range when this burst put one there, and otherwise by one blocking {@code pread}.
     */
    private Source source(final int needed) throws IOException {
        ensureOpen();
        if (filePointer + needed > length) {
            throw new EOFException("read past EOF: " + this);
        }
        final long absolute = offset + filePointer;
        readSincePrefetch = true;
        if (staged != null) {
            for (int i = 0; i < stagedCount; i++) {
                final StagedRange range = staged[i];
                if (range == null || range.covers(absolute, needed) == false) {
                    continue;
                }
                await(range);
                final int position = Math.toIntExact(absolute - range.alignedStart);
                if (position + needed > range.valid) {
                    // The read came up short of these bytes, which can only happen at end of file. Fall
                    // through to a blocking read rather than decode whatever the buffer last held.
                    break;
                }
                STATS.stagedHits.incrementAndGet();
                return new Source(range.buffer, range.floats, position, range.valid - position);
            }
        }
        return blockingSource(absolute, needed);
    }

    private void await(final StagedRange range) throws IOException {
        try {
            range.future.get();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("Interrupted waiting for a Direct I/O read of " + handle.path, e);
        } catch (ExecutionException e) {
            final Throwable cause = e.getCause();
            // Error propagates for the reason DirectIOVectorSource does not catch it: a direct-buffer
            // OutOfMemoryError is a misconfiguration to fix, not a condition to read around.
            if (cause instanceof Error) {
                throw (Error) cause;
            }
            if (cause instanceof IOException) {
                throw (IOException) cause;
            }
            throw new IOException("Direct I/O read of " + handle.path + " failed", cause);
        }
    }

    /** One blocking {@code pread} of the block-aligned window holding {@code absolute}, when not already held. */
    private Source blockingSource(final long absolute, final int needed) throws IOException {
        if (syncBuffer == null) {
            // Over-allocate by a block so a block-aligned slice exists inside the allocation: O_DIRECT
            // needs the buffer address, the file offset and the length all block aligned.
            syncBuffer = ByteBuffer.allocateDirect(bufferSize + handle.blockSize - 1)
                .alignedSlice(handle.blockSize)
                .order(ByteOrder.LITTLE_ENDIAN);
            syncFloats = syncBuffer.asFloatBuffer();
        }
        if (syncStart >= 0 && absolute >= syncStart && absolute + needed <= syncStart + syncValid) {
            final int position = Math.toIntExact(absolute - syncStart);
            return new Source(syncBuffer, syncFloats, position, syncValid - position);
        }
        final int blockSize = handle.blockSize;
        final long alignedStart = absolute - Math.floorMod(absolute, (long) blockSize);
        syncBuffer.clear().limit(bufferSize);
        final int read = handle.channel.read(syncBuffer, alignedStart);
        STATS.blockingReads.incrementAndGet();
        STATS.blockingBytes.addAndGet(Math.max(read, 0));
        syncStart = alignedStart;
        syncValid = Math.max(read, 0);
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
        return new Source(syncBuffer, syncFloats, position, syncValid - position);
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
            this.offset + offset,
            length,
            false,
            bufferSize,
            maxStagedRanges,
            maxStagedRangeBytes
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
            offset,
            length,
            false,
            bufferSize,
            maxStagedRanges,
            maxStagedRangeBytes
        );
        copy.filePointer = filePointer;
        return copy;
    }

    @Override
    public void close() throws IOException {
        if (closed) {
            return;
        }
        closed = true;
        resetStaging();
        if (ownsHandle) {
            handle.channel.close();
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
}

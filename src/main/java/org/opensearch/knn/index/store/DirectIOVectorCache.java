/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import lombok.extern.log4j.Log4j2;
import org.opensearch.knn.index.KNNSettings;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.LongAdder;

/**
 * The one cache of individual vectors on the Direct I/O random rescore path, for the whole node: a slab store
 * below the {@link DirectIOVectorIndexInput}, keyed by the file and the byte offset the reader asks for.
 *
 * <h2>Why it exists</h2>
 * The rescore candidate sets of successive queries overlap, so the vectors a rescore reads are reused: an
 * exact LRU stack-distance replay of a real query set against the exact candidate set reaches most of its
 * achievable hit rate at a budget of tens of megabytes, and replaying the same queries again does not
 * inflate it, so the reuse is inner-product HNSW hub reuse rather than an artefact of the query set. On an
 * IOPS-capped volume the device reads per query fall in step with the misses, which is what moves latency.
 *
 * <h2>Vector granularity comes for free</h2>
 * {@code Lucene99FlatVectorsReader}'s values object asks this input for exactly one vector's range — a
 * {@code prefetch(offset, vectorBytes)} and then a {@code readFloats} of the same run — so a key of
 * {@code (file, offset)} <em>is</em> one vector. Nothing has to be grouped, aligned or split, and the value
 * is the vector's bytes rather than the filesystem block they sit in: caching the block would also cache
 * the neighbours the measured access pattern does not ask for, and would charge the budget several times
 * over for the same hit rate.
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
 * <h2>Slabs, slots and the index</h2>
 * <ul>
 *   <li>A <b>slab</b> is {@link DirectIOSlab#SLAB_BYTES} of off-heap memory, allocated through
 *       {@code java.lang.foreign} by reflection where that works and {@code ByteBuffer.allocateDirect}
 *       otherwise ({@link DirectIOSlab}). It is opened for <b>one vector byte length</b> and cut into slots
 *       of exactly that length, so placing a vector is a bump pointer and finding it is a multiply. The
 *       store holds at most {@link DirectIOSlotMap#MAX_SLABS} of them.</li>
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
 * the write-lock hold off the millisecond scale: compacting the survivors to the front of the slab instead
 * was measured two orders of magnitude dearer per reclaim.
 *
 * <p>Re-admission is capped at half a slab's slots, which is what makes an eviction always yield room and the
 * algorithm terminate in one step: a fully hot slab cannot re-admit itself whole and so cannot deadlock the
 * cache against its own hit flags. The cap is the same device S3-FIFO and Segcache use for the same reason.
 * There is no admission filter, so the store is not scan resistant.
 *
 * <p>A slab reopened for a <em>different</em> vector length is the one case with no second chance: its slot
 * layout changes, so there is nowhere for a survivor to stay, and those entries are dropped. The case needs
 * two vector lengths competing for the same slabs, which is rare, and costs a refill rather than an answer.
 *
 * <h2>Segments, one lock each, and no view of slab memory leaves the store</h2>
 * A reader must not be shown a slot while another thread is overwriting it, and the index must not be read
 * half-rebuilt. One {@code ReentrantReadWriteLock} solves both: a lookup takes the read lock, and insert,
 * eviction, resize and release take the write lock. Lookups therefore run concurrently with each other and
 * never with a mutation, which is what lets the index be a plain {@code HashMap} of {@link DirectIOSlotMap}s
 * with no per-entry synchronisation.
 *
 * <p>One lock for the <b>whole store</b> peaks at one thread and falls to a fraction of that from four
 * threads up, with every read acquisition finding a writer already queued and most of all thread time spent
 * waiting (see {@link DirectIOCacheSegment}'s javadoc for the mechanism). So the store is split into
 * {@link DirectIOCacheSegment}s, each owning its lock, its index, its slabs and its eviction, with one
 * {@link DirectIOSlabBroker} owning the node's total slab target. A key's segment is a pure function of
 * {@code (fileId, offset)} ({@link #segmentFor}), so nothing about it is stored and a vector can only ever be
 * admitted into the segment a later lookup of that offset will consult. <b>Not one byte of the decode, the
 * admission geometry or the read path moves</b>, so a hit is still the same bytes decoded the same way and
 * scores identically by construction.
 *
 * <p>A lookup copies the vector into the caller's {@code float[]} while it holds the read lock and hands back
 * no view of slab memory at all. That is what makes a use after free unreachable rather than merely unlikely:
 * there is no buffer for a caller to stash and read after the lock is dropped. It is also why a slab needs no
 * lock of its own — it is closed only under its segment's write lock.
 *
 * <h2>The lock order, which is the whole deadlock proof</h2>
 * <b>The broker's lock, then at most one segment's lock.</b> Three rules, and every path below satisfies all
 * three:
 * <ol>
 *   <li><b>No path holds two segment locks.</b> A run that crosses a 256 KiB stripe releases one segment's
 *       lock before it takes the next ({@link #contains}, {@link #putRun}); a close walks the segments one at
 *       a time ({@link #releaseFile}); a shrink takes one segment's lock per slab it drops; and a slab lent
 *       from one segment to another is detached under the donor's lock alone, with the borrower's lock
 *       already released, and registered under the borrower's lock alone afterwards.</li>
 *   <li><b>No path takes the broker's lock while holding a segment's.</b> The only production path that
 *       takes the broker's lock is {@link #configure(long)}, which takes nothing beforehand, and the slab
 *       lending it performs. That is why the slab <em>reservation</em> is lock free: a fill has to reserve
 *       while it holds its own segment's write lock, which this rule would otherwise forbid. See
 *       {@link DirectIOSlabBroker}.</li>
 *   <li><b>Nothing that can block runs under either lock.</b> A slab allocation and a slab release both
 *       take milliseconds and both run with no lock held, which is asserted by a probe
 *       {@link SlabMemory} in the suite; so is "at most one segment lock", through
 *       {@link #segmentLocksHeldByCurrentThreadForTesting()}.</li>
 * </ol>
 * The one exception is the test-only {@link #checkInvariantsForTesting()}, which takes the broker's lock and
 * then every segment's write lock <b>in index order</b> so that it reads one consistent state. That is the
 * same order, so it cannot cycle with anything above.
 *
 * <h2>A size change resizes; it does not rebuild</h2>
 * {@code knn.direct_io.rescore.cache.max_bytes} is applied when the cluster applier accepts it
 * ({@link #publishMaxBytes(long)}); the prefetch burst only re-asserts the target already published
 * ({@link #applyConfiguredSize(boolean)}), which is how a store released under a route that was off grows
 * back. Growing lets the next miss allocate a slab and keeps every cached vector;
 * shrinking releases the oldest slabs and keeps every vector that still fits; {@code 0b} releases every
 * slab. A {@code Guava} cache cannot do that — {@code maximumWeight} is fixed at
 * {@code CacheBuilder#build()}, so a change drops the whole cache and warms it again.
 *
 * <h2>Entries of a file that has been closed are freed at the close</h2>
 * A merged-away segment's entries are <b>never wrong</b> — the key says which file's bytes they are and that
 * file's bytes are immutable — only <em>cold</em>. They would age out on their own: the FIFO reaches them
 * without a second chance (nothing reads them, so no hit flag is ever set) and they become free slots, and
 * when the last slot belonging to a file goes, that file's whole {@link DirectIOSlotMap} is dropped from the
 * index.
 *
 * <p>That is bounded but it is not free, and what it costs is this: until the FIFO cycles past them the
 * store <b>sits at its target</b>, and span fill only runs below the target, so it does not re-arm for the
 * merged segment and that segment refills at one vector per miss. So {@link #releaseFile} frees a file's
 * slots when the routed input that owned its
 * {@code O_DIRECT} handle closes — its index tables go, its slots go back to their slabs' free lists, and a
 * fill hands those slots out before it grows or evicts. The ageing-out path above stays, because it is what
 * covers an entry whose cell was already invalidated and a file nobody closes.
 *
 * <h2>Off is off</h2>
 * {@code knn.direct_io.rescore.cache.max_bytes} defaults to {@code 0b}. When it is zero there are no slabs
 * and every lookup is one volatile read and a compare — no key built, no segment selected, no lock taken, no
 * counter touched.
 */
@Log4j2
public final class DirectIOVectorCache {

    /** The node's total slab target and slab supply, which is the one thing a segment cannot own. */
    private static final DirectIOSlabBroker BROKER = new DirectIOSlabBroker();

    /**
     * Serializes the two ways the published byte target is applied, so that the last value published is the
     * one the store ends at. Held only across {@link #applyPublishedTarget}, which takes the broker's lock
     * inside it and nothing else; no path takes the broker's lock and then this.
     */
    private static final Object SIZE_APPLY = new Object();

    /**
     * The segments. Replaced only by {@link #configure(long)} under the broker's lock, which is why it is a
     * volatile reference to an array that is never mutated in place: every reader takes one consistent
     * snapshot of it and the array it sees stays the array it saw.
     */
    private static volatile DirectIOCacheSegment[] segments = { new DirectIOCacheSegment(BROKER) };

    /**
     * Whether a staged span also admits the whole vectors of the reading slice that lie inside it, while the
     * store is below its slab target.
     *
     * <p>A constant rather than a setting, because the policy has no magnitude and no arm that wants it off:
     * at or above the slab target admission is requested vectors only, exactly as before, so span fill
     * cannot cause an eviction and cannot take a slab the target does not already allow. It was measured on
     * an EBS node to take the rescore p50 down by a factor of four, because without it the store warms one
     * requested vector at a time. Turning the store off turns span fill off with it, which is the arm a node
     * compares against.
     */
    static final boolean SPAN_FILL = true;

    /**
     * How wide a staged miss read may be made so that the whole vectors around the one being fetched are
     * admitted with it, while the store is below its target and the whole file would still fit in the room
     * the target leaves. One read per miss stays one read; only its size changes.
     *
     * <p>A constant rather than a setting. A narrower read-around was measured to cost coverage and p99
     * together, so the width is coverage bought deliberately and with a bound rather than a value with a
     * better choice in it. The three gates in {@link #readAroundBytes(long)} are what keep it from evicting
     * anything.
     */
    static final long READ_AROUND_BYTES = 64L * 1024;

    /**
     * Whether span fill is on, {@link #SPAN_FILL} except where a test has moved it. Production writes it
     * once, at class load.
     *
     * @see #setSpanFillForTesting(boolean)
     */
    private static volatile boolean spanFill = SPAN_FILL;

    /**
     * The live read-around width, {@link #READ_AROUND_BYTES} except where a test has moved it. Production
     * writes it once, at class load.
     *
     * @see #setReadAroundBytesForTesting(long)
     */
    private static volatile long readAroundBytes = READ_AROUND_BYTES;

    /**
     * {@code knn.direct_io.rescore.cache.max_bytes}, as the node last published it. Published by
     * {@link KNNSettings} on a change and seeded there at startup; until then it is the setting's own
     * default, which is what a JVM with no cluster service runs on.
     *
     * @see #publishMaxBytes(long)
     */
    private static volatile long configuredMaxBytes = KNNSettings.KNN_DIRECT_IO_RESCORE_CACHE_MAX_BYTES_DEFAULT_VALUE.getBytes();

    /** A segment count pinned by a test, or 0 to derive it from the node. See {@link #setSegmentsForTesting}. */
    private static volatile int segmentsForTesting;

    /**
     * Where a slab's memory comes from and goes back to. A seam, and the only reason it is one is that the
     * property to assert is about <b>where</b> these two calls run: neither may run while the caller holds a
     * segment's
     * lock or the broker's, and a test can only see that if it can observe the calling thread at the moment
     * of the call. Production has exactly one implementation.
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

    // ---------------------------------------------------------------------------------------------------
    // Counters. KNNStats publishes these through direct_io_stats.cache.
    //
    // Package private rather than private because the work they count happens inside a segment or inside
    // the broker now, and keeping them here is what keeps every published key in the one file the stats
    // layer reads. A LongAdder is safe to increment from anywhere: it is striped and takes no lock.
    // ---------------------------------------------------------------------------------------------------

    private static final LongAdder hits = new LongAdder();
    private static final LongAdder misses = new LongAdder();
    static final LongAdder insertions = new LongAdder();
    static final LongAdder reclaimedSlabs = new LongAdder();
    static final LongAdder readmissions = new LongAdder();
    static final LongAdder droppedEntries = new LongAdder();
    private static final LongAdder spanFills = new LongAdder();
    /** Allocations that failed, which is one per latch and so also how often growth was given up on. */
    static final LongAdder slabAllocFailures = new LongAdder();
    /** Write-lock time spent reclaiming slabs, which is the figure a node has to be able to read. */
    static final LongAdder reclaimNanos = new LongAdder();
    /** Files whose entries were freed at close, and the entries that went with them. */
    private static final LongAdder closedFiles = new LongAdder();
    private static final LongAdder closedFileEntries = new LongAdder();
    /** Vectors put back after the filter-then-read race lost them. Also counted in readmissions. */
    static final LongAdder raceReadmissions = new LongAdder();
    /** Slabs the broker lent from one segment to another because the borrower held none. */
    static final LongAdder slabTransfers = new LongAdder();
    /** Resizes that crossed a segment-count boundary and so cost a re-warm. */
    static final LongAdder segmentCountChanges = new LongAdder();

    /**
     * The longest one segment's write lock has been held by a close, which is the close-hold bound made
     * readable on a node. An {@link AtomicLong} kept at its maximum rather than a {@link LongAdder}, because
     * what an operator needs is the worst one a lookup could have waited behind, not the total.
     *
     * <p>A close of a large file's entries takes tens of milliseconds unsegmented. Segmented, it takes one
     * segment's lock at a time and each holds about a {@code 1/N} share of the cells.
     */
    static final AtomicLong longestCloseHoldNanos = new AtomicLong();

    /**
     * The most entries one segment's write lock has been held for by a close. The work behind
     * {@link #longestCloseHoldNanos}, and the figure a test can assert on: it is exactly the fullest
     * segment's share of the file and so is a deterministic function of the segment count, where the nanos
     * are a wall-clock maximum that picks up a collection pause whole.
     */
    static final AtomicLong longestCloseHoldEntries = new AtomicLong();

    /** Records one segment's close hold, keeping the longest. Called with no lock held. */
    static void noteCloseHold(final long nanos, final long entries) {
        keepMax(longestCloseHoldNanos, nanos);
        keepMax(longestCloseHoldEntries, entries);
    }

    private static void keepMax(final AtomicLong at, final long value) {
        long longest = at.get();
        while (value > longest && at.compareAndSet(longest, value) == false) {
            longest = at.get();
        }
    }

    /**
     * Acquisitions that found a thread already queued on the segment's lock, and the nanoseconds they then
     * waited. Only a contended acquisition is counted and only a contended one is timed, which is why
     * both are free on the path that matters; see {@link DirectIOCacheSegment}'s {@code lockRead}.
     */
    static final LongAdder contendedReadAcquisitions = new LongAdder();
    static final LongAdder contendedWriteAcquisitions = new LongAdder();
    static final LongAdder readLockWaitNanos = new LongAdder();
    static final LongAdder writeLockWaitNanos = new LongAdder();

    /**
     * Every acquisition, contended or not, counted only while {@link #setAcquisitionCountingForTesting} has
     * turned it on. Off in production, where it would be a {@link LongAdder} increment on every lookup for a
     * figure no operator reads: the contended counts above are the ones that say something, and a test that
     * needs the denominator can ask for it.
     */
    static final LongAdder readAcquisitions = new LongAdder();
    static final LongAdder writeAcquisitions = new LongAdder();
    /** Returns of {@link DirectIOSlabBroker#lock()}, which is every intent to take the resize lock. */
    static final LongAdder brokerLockAcquisitions = new LongAdder();
    static volatile boolean acquisitionCounting;

    /**
     * Whether a contended acquisition is timed. Always true in production; a test and the overhead
     * measurement turn it off to price the timing against a build that does none.
     */
    static volatile boolean lockTimingEnabled = true;

    private DirectIOVectorCache() {}

    /** The identity {@link DirectIOSlotMap} uses for a file: two opens of one {@code .vec} give one string. */
    static String fileId(final java.nio.file.Path path, final long fileLength) {
        return path.toString() + "#" + fileLength;
    }

    /** Whether a lookup can succeed at all, so a caller can skip the work of asking. */
    static boolean isOn() {
        return BROKER.maxSlabs() != 0;
    }

    // ---------------------------------------------------------------------------------------------------
    // Segment selection
    // ---------------------------------------------------------------------------------------------------

    /**
     * The 256 KiB stripe an offset lies in, which is the unit of segmentation.
     *
     * <p>A stripe and not a vector, because a staged span is at most
     * {@link DirectIOVectorIndexInput#DEFAULT_MAX_STAGED_RANGE_BYTES}, which is below the stripe, so a span
     * lies in at most <b>two</b> stripes wherever it starts and {@link #putRun} keeps essentially the one
     * write-lock acquisition it has. A per-vector key would scatter a run across one segment per vector and
     * turn one acquisition into dozens. The two keys were measured to give the same hit rate at every segment
     * count, so the read side does not pay for the choice.
     */
    private static final int STRIPE_SHIFT = 18;

    /**
     * The segment a key belongs to: {@code mix(fileIdHash, offset >>> 18) & (segments - 1)}.
     *
     * <p><b>Its own multipliers, and the top bits.</b> Inside a segment every offset shares its stripe bits,
     * and {@link DirectIOSlotMap}'s own hash is Fibonacci over the whole 64 bits with multiplier
     * {@code 0x9E3779B97F4A7C15}. A selector that folded the slot map's bits would put a segment's offsets
     * through one stretch of its table, so this one uses different multipliers and takes the <b>top</b> bits.
     *
     * @param fileId the file's identity, from {@link #fileId}
     * @param offset the vector's byte offset in that file
     * @return the segment that owns the key
     */
    static DirectIOCacheSegment segmentFor(final String fileId, final long offset) {
        final DirectIOCacheSegment[] all = segments;
        if (all.length == 1) {
            return all[0];
        }
        return all[segmentIndex(fileId.hashCode(), offset, all.length)];
    }

    /** The segments in force, for {@link DirectIOSlabBroker}'s lending. */
    static DirectIOCacheSegment[] segmentsSnapshot() {
        return segments;
    }

    /**
     * How many segments a store of {@code maxSlabs} slabs gets on a node of {@code cores} cores:
     *
     * <pre>
     * coreSegments = clamp(nextPowerOfTwo(8 * cores), 8, 256)
     * segments     = max(1, min(coreSegments, previousPowerOfTwo(maxSlabs / 8)))
     * </pre>
     *
     * <p><b>Both terms, because the slab floor binds before the core count does.</b> The {@code 8 * cores}
     * term aims at about one thread per segment, which is where contention was measured to disappear: one
     * thread contends for nothing and two already contend for most acquisitions. The {@code maxSlabs / 8}
     * term is what the hit rate requires: an exact emulation of this policy over a real trace showed the loss
     * against an unsegmented store governed by <b>slabs per segment</b> and not by the segment count, and
     * acceptable at eight slabs per segment but not below it, because a segment with one slab evicts its
     * whole contents at once.
     *
     * <p>A small store therefore gets one segment, which is the unsegmented store exactly, so no
     * configuration gets worse than it was.
     *
     * @param cores    the node's available processors
     * @param maxSlabs the store's slab target
     * @return the segment count, a power of two, at least 1
     */
    static int derivedSegmentCount(final int cores, final int maxSlabs) {
        if (maxSlabs <= 0) {
            return 1;
        }
        final int coreSegments = Math.min(256, Math.max(8, nextPowerOfTwo(8 * Math.max(1, cores))));
        return Math.max(1, Math.min(coreSegments, previousPowerOfTwo(maxSlabs / 8)));
    }

    private static int nextPowerOfTwo(final int value) {
        return value <= 1 ? 1 : Integer.highestOneBit(value - 1) << 1;
    }

    private static int previousPowerOfTwo(final int value) {
        return value <= 0 ? 0 : Integer.highestOneBit(value);
    }

    /** The count a resize should move to, honouring {@link #setSegmentsForTesting}. */
    private static int wantedSegmentCount(final int maxSlabs) {
        final int pinned = segmentsForTesting;
        if (pinned > 0) {
            return pinned;
        }
        return derivedSegmentCount(Runtime.getRuntime().availableProcessors(), maxSlabs);
    }

    /**
     * {@link #segmentFor}'s arithmetic, separately so a test can assert it without a store. Byte for byte
     * the form {@code bench/q-baseline/SegmentedHitRateBench#segmentOf} measured the hit-rate cost of, so
     * the implementation and the design run's emulation select the same segment for the same key.
     *
     * @param fileIdHash the file identity's {@code hashCode}
     * @param offset     the vector's byte offset
     * @param count      the segment count, a power of two
     * @return the segment index
     */
    static int segmentIndex(final int fileIdHash, final long offset, final int count) {
        final long mixed = ((offset >>> STRIPE_SHIFT) ^ ((long) fileIdHash * 0x2545F4914F6CDD1DL)) * 0xBF58476D1CE4E5B9L;
        return (int) ((mixed >>> 40) & (count - 1));
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
     * Counting it would double every count against the vectors actually served, which an earlier shape of
     * this store did and had to document ({@link #lookupsPerVector()}); one count per vector at the lookup
     * that serves it makes {@code hits / (hits + misses)} a hit rate comparable with a node's own figures.
     *
     * <h4>Why it is a run and not one vector</h4>
     * The range {@code prefetch} is given is not a vector, it is a <em>coalesced group</em>:
     * {@code PrefetchHelper#prefetchExactVectorSize} sorts the batch's ordinals and asks for the span of each
     * group it can fit in its span budget, so both ends are vector boundaries on one grid but the span can
     * hold dozens of them. A one-vector lookup therefore answers "no" for every multi-vector group, however
     * much of it is cached, and the group is staged in full. The one-vector form was measured reading the
     * whole file many times over at a hit rate near its ceiling, because the groups were staged even once the store
     * held every vector in them; walking the run takes the same arm to about the file's own size. The cost is
     * up to {@code max_span_bytes / vectorBytes} index probes under one read-lock acquisition, against the
     * device read it avoids.
     *
     * <p>A run of one vector is a single lookup, so a caller asking about one vector asks exactly as before.
     *
     * <p>The run is walked segment by segment: a span lies in at most two stripes, so this takes at most
     * <b>two</b> read acquisitions and releases each before taking the next, which is the
     * no-two-segment-locks rule.
     *
     * @param fileId the file's identity, from {@link #fileId}
     * @param offset the first vector's byte offset in that file
     * @param length the run's length in bytes, one or more whole vectors
     * @return whether a {@link #getFloats} would hit for every vector in the run
     */
    static boolean contains(final String fileId, final long offset, final int length) {
        if (BROKER.maxSlabs() == 0 || length <= 0) {
            return false;
        }
        final long end = offset + length;
        long at = offset;
        while (at < end) {
            final long reached = segmentFor(fileId, at).containsUpTo(fileId, at, end);
            if (reached <= at) {
                // A miss, or a slot of zero length, which cannot happen but must not loop if it did.
                return false;
            }
            at = reached;
        }
        // Exactly tiled, or not cached: a run whose last vector overshoots the range is a different
        // layout than the one being asked about, and the honest answer for one is no.
        return at == end;
    }

    /**
     * Writes into {@code missing} the members of {@code ords[0..numOrds)} whose vector this store does not
     * hold whole and returns how many, or returns {@code -1} when it holds nothing of {@code fileId} at all.
     *
     * <p>Like {@link #contains} it <b>counts nothing</b>: the caller
     * ({@code DirectIOVectorIndexInput#selectUncachedOrdinals}) is deciding what to prefetch, not serving
     * bytes. {@code -1} rather than a copy of every ordinal is what lets a cold file take the caller's
     * unfiltered path with no array to walk.
     *
     * <p>A vector held at a different length than {@code vectorBytes} counts as missing, for the same reason
     * {@link #getFloats} answers a length mismatch with a miss: a caller asking about a layout this store did
     * not cache is a caller it was not designed for.
     *
     * <p>The misses come out <b>in the caller's ordinal order</b> whatever the segmentation does, because
     * {@code DirectIOVectorIndexInput#recordFilteredAsCached} finds the complement in one pass over both
     * arrays and so needs {@code missing[0..found)} to be a subsequence of {@code ords[0..numOrds)}.
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
        if (BROKER.maxSlabs() == 0) {
            return -1;
        }
        final DirectIOCacheSegment[] all = segments;
        if (all.length == 1) {
            // One read acquisition for the batch, which is the batching this method exists to keep.
            return all[0].selectMissing(fileId, baseOffset, vectorBytes, ords, 0, numOrds, missing, 0);
        }
        // bucket the batch by segment, walk each touched segment under ONE read acquisition, and write
        // the misses out in the caller's ordinal order. A counting sort, so the bucketing is
        // O(batch + segments) with no comparison, and the scratch is one set of arrays per thread.
        //
        // A contended read acquisition's wait rises with the thread count and an uncontended one's is three
        // orders of magnitude below it, so bucketing trades one contended acquisition per batch for one
        // uncontended acquisition per touched segment, which is the cheaper side at every thread count
        // measured.
        final SelectScratch scratch = SELECT_SCRATCH.get().sized(numOrds, all.length);
        final int[] segmentOf = scratch.segmentOf;
        final int[] bounds = scratch.bounds;
        final int[] order = scratch.order;
        final boolean[] miss = scratch.miss;
        Arrays.fill(bounds, 0, all.length, 0);
        final int hash = fileId.hashCode();
        for (int i = 0; i < numOrds; i++) {
            final int segment = segmentIndex(hash, baseOffset + (long) ords[i] * vectorBytes, all.length);
            segmentOf[i] = segment;
            bounds[segment]++;
        }
        int running = 0;
        for (int segment = 0; segment < all.length; segment++) {
            final int here = bounds[segment];
            bounds[segment] = running;
            running += here;
        }
        for (int i = 0; i < numOrds; i++) {
            order[bounds[segmentOf[i]]++] = i;
        }
        // bounds[s] is now one past segment s's group, so its group is [the previous bound, bounds[s]).
        boolean anyIndexed = false;
        int from = 0;
        for (int segment = 0; segment < all.length; segment++) {
            final int to = bounds[segment];
            if (to == from) {
                continue;
            }
            anyIndexed |= all[segment].probeMissing(fileId, baseOffset, vectorBytes, ords, order, from, to, miss);
            from = to;
        }
        int found = 0;
        for (int i = 0; i < numOrds; i++) {
            if (miss[i]) {
                missing[found++] = ords[i];
            }
        }
        return anyIndexed ? found : -1;
    }

    /**
     * {@link #selectMissing}'s working arrays, one set per thread because the batch is per thread and the
     * alternative is four allocations per prefetch filter call. Grown to the largest batch and the largest
     * segment count a thread has seen and <b>never shrunk</b>, which is the device
     * {@code PrefetchHelper#missScratch} uses for its own miss list; a search thread's batch is Lucene's bulk
     * size, so this settles at a few hundred ints.
     *
     * <p>{@code miss} needs no clearing between calls: every ordinal of the batch belongs to exactly one
     * segment's group, so every one of its {@code numOrds} entries is written before any is read.
     */
    private static final class SelectScratch {
        private int[] segmentOf = new int[0];
        private int[] order = new int[0];
        private int[] bounds = new int[0];
        private boolean[] miss = new boolean[0];

        private SelectScratch sized(final int ords, final int segments) {
            if (segmentOf.length < ords) {
                segmentOf = new int[ords];
                order = new int[ords];
                miss = new boolean[ords];
            }
            if (bounds.length < segments) {
                bounds = new int[segments];
            }
            return this;
        }
    }

    private static final ThreadLocal<SelectScratch> SELECT_SCRATCH = ThreadLocal.withInitial(SelectScratch::new);

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
        // Not a whole number of floats is a miss and not a lookup. A slab's slot length may be any
        // even number of bytes (an FP16 vector of an odd dimension is two bytes a dimension), and the
        // {@link DirectIOSlab#floats} view below indexes in floats, so a slot that is not float-aligned must
        // never be reachable through this method. The FP16 path reads through {@link #getBytes} instead.
        if (BROKER.maxSlabs() == 0 || (length & 3) != 0) {
            return false;
        }
        final boolean hit = segmentFor(fileId, offset).getFloats(fileId, offset, length, destination, destinationOffset);
        if (hit) {
            hits.increment();
        } else {
            misses.increment();
        }
        return hit;
    }

    /**
     * {@link #getFloats} for a caller that wants the vector's bytes rather than its floats: copies
     * {@code length} bytes into {@code destination} and returns true, or returns false and leaves
     * {@code destination} alone.
     *
     * <h4>Why this exists, and why the store needed nothing else for FP16</h4>
     * The FP16 flat format's values read a vector with
     * {@code KNN1040HalfFloatFlatVectorsValues#readRawVectorBytes}, which is
     * {@code slice.readBytes(dest, destOffset, 2 * dimension)} and never {@code readFloats} — the whole point
     * of that format being that nothing decodes FP16 to FP32 on the read path. The store keys on
     * {@code (fileId, offset)} and checks the slot's byte length, and a slab holds raw bytes, so a 2-byte
     * vector needs no new key, no new slot layout and no new fill path: it needs a serve site that copies
     * bytes out instead of floats. That is this method, and it counts a hit or a miss exactly as
     * {@link #getFloats} does, so {@code hits / (hits + misses)} stays one per-vector hit rate across both
     * precisions.
     *
     * <p>Even lengths only, which is what a whole number of half floats is. An odd length cannot be a vector
     * of either precision and the honest answer for one is a miss, counted as nothing.
     *
     * @param fileId the file's identity, from {@link #fileId}
     * @param offset the vector's byte offset in that file
     * @param length the vector's length in bytes
     * @param destination at least {@code destinationOffset + length} bytes
     * @param destinationOffset where in {@code destination} to write the first byte
     * @return whether the vector was cached
     */
    static boolean getBytes(
        final String fileId,
        final long offset,
        final int length,
        final byte[] destination,
        final int destinationOffset
    ) {
        if (BROKER.maxSlabs() == 0 || length <= 0 || (length & 1) != 0) {
            return false;
        }
        final boolean hit = segmentFor(fileId, offset).getBytes(fileId, offset, length, destination, destinationOffset);
        if (hit) {
            hits.increment();
        } else {
            misses.increment();
        }
        return hit;
    }

    // ---------------------------------------------------------------------------------------------------
    // The fill path
    // ---------------------------------------------------------------------------------------------------

    /**
     * Caches {@code length} bytes at {@code position} of {@code source} as the vector at
     * {@code (fileId, offset)}. A copy and not a view, because {@code source} is a staging buffer that the
     * next burst reuses.
     *
     * <p>Silently does nothing when the cache is off, when the length is odd or is larger than a slab, or
     * when no slab can be had. A fill that does not happen costs a future hit and nothing else.
     *
     * <p>The length rule is "an even number of bytes", not "a whole number of floats", because an FP16
     * vector is {@code 2 x dimension} bytes and an odd dimension makes that a multiple of 2 and not of 4. A
     * slot that is not float-aligned is unreachable through {@link #getFloats}, which refuses such a length
     * outright, and is served by {@link #getBytes}, which is what the FP16 read path uses.
     *
     * @param fileId the file's identity, from {@link #fileId}
     * @param offset the vector's byte offset in that file
     * @param source the staged bytes
     * @param position where in {@code source} the vector starts
     * @param length the vector's length in bytes
     */
    static void put(final String fileId, final long offset, final ByteBuffer source, final int position, final int length) {
        if (BROKER.maxSlabs() == 0 || length <= 0 || (length & 1) != 0 || offset < 0 || length > BROKER.slabBytes()) {
            return;
        }
        segmentFor(fileId, offset).put(fileId, offset, source, position, length);
    }

    /**
     * Admits the vector a blocking read had to fetch because the store lost it between the prefetch filter
     * and {@code readFloats}, <b>without evicting anything</b>.
     *
     * <h4>The race, and why it leaves a vector uncached without this</h4>
     * The prefetch filter asks {@link #contains} and drops the ordinals the store holds, so those vectors are
     * not staged. A vector evicted in the window between that check and the read is then a
     * <em>blocking</em> read, and the admission rule is "only from a staged read covering the vector whole",
     * so its bytes were thrown away. The next batch asking for it would stage and admit it, so this
     * self-heals; what it costs until then is one synchronous device read, and a node measured a dozen or
     * two of them per five thousand queries.
     *
     * <h4>Why this does not reopen the "a blocking read is not admitted" rule</h4>
     * That rule is about the sequential channels: the read-ahead ring buys a cache nothing and would evict
     * the random path's working set, and a blocking read is the fallback the cache exists to avoid. This is
     * neither. The caller admits only an offset <b>its own prefetch filter dropped in this burst</b>
     * ({@code DirectIOVectorIndexInput#wasFilteredAsCached}), which can only be a rescore candidate the
     * store was holding moments ago, i.e. exactly the working set. And it places with eviction refused, so
     * it can take a free slot or a slab the target allows and nothing else: a vector the store lost cannot
     * cost another vector its slot.
     *
     * @return whether the vector was admitted
     */
    static boolean readmitAfterRace(final String fileId, final long offset, final ByteBuffer source, final int position, final int length) {
        if (BROKER.maxSlabs() == 0 || length <= 0 || (length & 1) != 0 || offset < 0 || length > BROKER.slabBytes()) {
            return false;
        }
        return segmentFor(fileId, offset).readmitAfterRace(fileId, offset, source, position, length);
    }

    /**
     * Whether the store would take another slab if a fill asked for one, i.e. whether it is still below its
     * slab target. This is the condition span fill admits under; see {@link #putRun}.
     *
     * <p>A volatile read of each of two ints and no lock, because the caller is the read path asking once per
     * staged span whether to bother, and the answer is advisory: {@link #putRun} re-checks it under a segment's
     * write lock, which is where it has to hold.
     *
     * @return whether a fill may still allocate a slab
     */
    static boolean belowSlabTarget() {
        return BROKER.belowTarget();
    }

    /**
     * How wide a staged miss read of a file of {@code fileBytes} bytes may be made, or {@code 0} for "do not
     * widen it".
     *
     * <h4>The three gates, and why the second is the whole file rather than the target</h4>
     * <ul>
     *   <li>{@link #READ_AROUND_BYTES} is not zero;</li>
     *   <li><b>the whole file would still fit in the room the target leaves</b>, which is what makes it
     *       impossible for a read-around to evict a live vector: if the file fits, nothing has to be
     *       displaced to hold it. Merely being below the target would not do, because a store one vector
     *       below its target would widen a read whose neighbours then evict the working set, and scan
     *       resistance (agenda item 4) is undecided;</li>
     *   <li>span fill is on, because the widened bytes are admitted by span fill and a wider read nobody
     *       harvests is device traffic for nothing.</li>
     * </ul>
     *
     * <p>The fit test is deliberately <b>conservative</b>: it counts the bytes this file already holds
     * against the room as well, so it can turn read-around off slightly early for a file that is a large
     * fraction of the target. The alternative needs a per-file byte count, and a file can carry several
     * fields of different dimensions, so there is no one vector length to multiply by. Costing a nearly-full
     * store some coverage is the right direction for a gate whose job is to never evict.
     *
     * <p>The fit test is a lock-free sum of the segments' volatile byte counts since Q, where it took the
     * store's one read lock: it is a global question and it is asked once per prefetch burst, not once per
     * read (see {@code DirectIOVectorIndexInput#beginBurst}), so it has no business competing with the query
     * path for a lock.
     *
     * @param fileBytes the whole file's length
     * @return the read-around width in bytes, or 0
     */
    static int readAroundBytes(final long fileBytes) {
        if (isSpanFillOn() == false) {
            return 0;
        }
        final long configured = readAroundBytes;
        if (configured <= 0 || fileBytes <= 0) {
            return 0;
        }
        final long target = maxBytes();
        if (bytes() + fileBytes > target) {
            return 0;
        }
        return (int) Math.min(configured, Integer.MAX_VALUE / 2);
    }

    /**
     * The lock-free form of a segment's span-fill question, for the input's pre-check before it takes any
     * write lock. Approximate in one direction only: it can say yes where the segment then says no, which
     * costs one lock acquisition and admits nothing. It cannot say no where the precise answer is yes,
     * because a segment's recycled count is only cleared for a slab that has left its queue.
     */
    static boolean mayAdmitSpanFillApproximately() {
        if (belowSlabTarget()) {
            return true;
        }
        for (final DirectIOCacheSegment segment : segments) {
            if (segment.recycledSlabs() > 0) {
                return true;
            }
        }
        return false;
    }

    /**
     * Admits the {@code count} consecutive vectors starting at {@code (fileId, firstOffset)} from
     * {@code source}, under <b>one</b> write-lock acquisition per segment the run touches, and only while the
     * store is below its slab target. This is span fill.
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
     *   <li><b>It stops at the first refusal.</b> Once a segment has nowhere to put a vector it has nowhere
     *       to put the next one either, so walking the rest of the run would be an index scan under the write
     *       lock for nothing.</li>
     * </ul>
     *
     * <p>One acquisition per run rather than per vector is what bounds the hold to one span's worth of
     * copying: a run is the whole vectors of one staged span, which is a handful at the measured mean span
     * and at most {@code max_span_bytes / length}. The run is split at stripe boundaries into at most
     * <b>two</b> per-segment sub-runs, taken sequentially; see {@link #segmentFor}.
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
        if (BROKER.maxSlabs() == 0 || count <= 0 || length <= 0 || (length & 1) != 0 || firstOffset < 0 || length > BROKER.slabBytes()) {
            return 0;
        }
        int admitted = 0;
        int next = 0;
        while (next < count) {
            final long at = firstOffset + (long) next * length;
            final DirectIOCacheSegment segment = segmentFor(fileId, at);
            final int subRunEnd = runEndIn(fileId, firstOffset, next, count, length, segment);
            final long[] admittedHere = new long[1];
            final int reached = segment.putRun(fileId, firstOffset, next, subRunEnd, source, position, length, admittedHere);
            admitted += (int) admittedHere[0];
            if (reached <= next) {
                // The segment admitted nothing and made no progress, which is a refusal: it has nowhere to
                // put this vector and so has nowhere to put the next one either.
                break;
            }
            next = reached;
            if (reached < subRunEnd) {
                // It stopped inside its own sub-run, which is the same refusal one vector later.
                break;
            }
        }
        spanFills.add(admitted);
        return admitted;
    }

    /**
     * One past the last index of the run {@code [from, count)} whose vectors all belong to {@code segment}.
     * At most two stripes can be touched by a staged span, so this answers {@code count} or one boundary in.
     */
    private static int runEndIn(
        final String fileId,
        final long firstOffset,
        final int from,
        final int count,
        final int length,
        final DirectIOCacheSegment segment
    ) {
        if (segments.length == 1) {
            return count;
        }
        int end = from + 1;
        while (end < count && segmentFor(fileId, firstOffset + (long) end * length) == segment) {
            end++;
        }
        return end;
    }

    /**
     * Frees every entry of {@code fileId} and drops its index tables, because the input that owned the file's
     * {@code O_DIRECT} handle has closed and nothing will read those vectors again.
     *
     * <h4>What it is for</h4>
     * Without it, a merged-away segment's entries stay in their slots until the FIFO cycles past them, and
     * they are never <em>wrong</em> -- the key says which file's bytes they are and that file's bytes are
     * immutable -- only cold. But the store then sits at its target after a merge, and span fill runs
     * only below the target, so it did not re-arm for the merged segment and that segment refilled at one
     * vector per miss. Freeing them at close puts the store below its target again, which is what lets span
     * fill warm the new segment. A reopened file simply refills.
     *
     * <h4>Why no reader can see a torn or a foreign vector</h4>
     * This holds each segment's <b>write</b> lock in turn, and {@link #getFloats} copies the vector out of the
     * slab into the caller's array under that segment's <b>read</b> lock (the copy is one statement inside
     * that lock). So a reader either completes before this begins on its segment or starts after it ends;
     * there is no window in which a reader holds a resolved slot across the free. A reader that arrives
     * afterwards misses, because the index table is gone, and reads the device.
     *
     * <h4>Why one segment at a time</h4>
     * Nothing waits behind a close that is not in the segment being drained, and the segment being drained
     * waits for one {@code N}th of the work; a close of a large file's entries was measured at tens of
     * milliseconds unsegmented.
     *
     * <h4>Why the slots are freed and not evicted</h4>
     * {@code evictions} counts vectors the store had to displace to make room, which is the figure an
     * operator sizes the cache by. These were not displaced, so they are counted under their own two keys
     * and {@code evictions} does not move.
     *
     * @param fileId the file's identity, from {@link #fileId}
     * @return how many entries were freed
     */
    static long releaseFile(final String fileId) {
        if (BROKER.maxSlabs() == 0) {
            // The cheap pre-check every entry point to this class has: with the store off there is no index
            // and no lock worth taking, and a close is on a path that may be serving a query.
            return 0;
        }
        final long started = System.nanoTime();
        final long[] freed = new long[2];
        for (final DirectIOCacheSegment segment : segments) {
            segment.releaseFile(fileId, freed);
        }
        if (freed[0] > 0) {
            closedFiles.increment();
            closedFileEntries.add(freed[0]);
            log.debug(
                "Freed {} cached vectors ({} B) of [{}] at close, in {} us of write lock",
                freed[0],
                freed[1],
                fileId,
                (System.nanoTime() - started) / 1_000
            );
        }
        return freed[0];
    }

    // ---------------------------------------------------------------------------------------------------
    // Sizing
    // ---------------------------------------------------------------------------------------------------

    /**
     * Resizes the store to the published {@code knn.direct_io.rescore.cache.max_bytes} if it has changed.
     *
     * <p>Called once per prefetch burst, from {@code DirectIOVectorIndexInput#prefetch} at the burst
     * boundary. The burst is not how a change arrives — {@link #publishMaxBytes} applies it the moment the
     * cluster applier accepts it — but it is how a store released under a route that was off grows back
     * when the route comes on again, and in the steady state it costs one field read and one comparison.
     *
     * @param routeActive whether the calling input is reading through {@code O_DIRECT} right now. {@code false}
     *                    means {@code knn.direct_io.rescore.enabled} has been turned off under an open routed
     *                    input, and the store is released rather than resized.
     */
    static void applyConfiguredSize(final boolean routeActive) {
        if (routeActive == false) {
            // The caller is reading through its delegate input because knn.direct_io.rescore.enabled is off,
            // so nothing will be admitted and nothing will be looked up: slabs held here serve nothing.
            // Releasing them is what makes the rollback give the memory back as well as the device traffic.
            // Going back on re-reads the published target and the store grows again on its next fill.
            //
            // It is the caller's route and not a second read of the setting, because an input with no
            // delegate to fall back to stays on O_DIRECT whatever the setting says (see
            // DirectIOVectorIndexInput#mmapFallback) and its store must stay with it. That is every probe
            // and bench driver in this package, which open a path directly.
            applyPublishedTarget(0L);
            return;
        }
        applyPublishedTarget(null);
    }

    /** The byte target the node last published, which {@link #maxBytes()} is the rounded-down form of. */
    public static long configuredMaxBytes() {
        return configuredMaxBytes;
    }

    /**
     * Publishes {@code knn.direct_io.rescore.cache.max_bytes} and resizes the store to it. Called by
     * {@link KNNSettings} from the cluster applier thread, and it must not wait: a shrink from a large cache
     * is hundreds of slab frees. {@link #configure} is what guarantees that — it detaches the slabs under
     * the broker's lock and hands them to the release thread — so this method returns as soon as the slabs
     * have left the index, not as soon as the memory has gone back.
     *
     * <p>Not clamped: the setting is a byte size with a floor of {@code 0b}, and {@link #configure} floors
     * at zero in any case.
     */
    public static void publishMaxBytes(final long maxBytes) {
        configuredMaxBytes = maxBytes;
        applyPublishedTarget(null);
    }

    /**
     * Resizes the store to {@code explicitTarget}, or to the published target when it is {@code null}.
     *
     * <p><b>Why the published target is re-read inside this monitor.</b> Two threads reach here: the cluster
     * applier, which has just written {@link #configuredMaxBytes}, and a reading thread at a burst boundary,
     * which may have read that field before the applier wrote it. Without the monitor the reading thread
     * could call {@link #configure} with the old target <em>after</em> the applier had already applied the
     * new one, putting the store back to a size the operator has just asked it to leave — an operator who
     * sets {@code 0b} under load would watch the slabs come back. Taking the monitor first and reading the
     * field second makes the last write win whichever order the two threads arrive in.
     *
     * <p>The steady-state check is deliberately <b>outside</b> the monitor and on a stale read, because this
     * runs once per prefetch burst and the answer is almost always "no change": a stale read can only be
     * wrong when a change is in flight, and then it says "changed" and the monitor settles it.
     */
    private static void applyPublishedTarget(final Long explicitTarget) {
        final long seen = explicitTarget != null ? explicitTarget : configuredMaxBytes;
        if (wantedSlabs(seen) == BROKER.maxSlabs()) {
            return;
        }
        synchronized (SIZE_APPLY) {
            configure(explicitTarget != null ? explicitTarget : configuredMaxBytes);
        }
    }

    /** Whole slabs a byte target buys, which is what {@link #configure} actually moves the store to. */
    private static int wantedSlabs(final long maxBytes) {
        return (int) Math.min(DirectIOSlotMap.MAX_SLABS, Math.max(0, maxBytes) / BROKER.slabBytes());
    }

    /**
     * Whether span fill is on: {@link #SPAN_FILL}, and the store not being off.
     *
     * <p>The store being off turns span fill off with it: the widened bytes are admitted by span fill, and a
     * wider read nobody harvests is device traffic for nothing.
     *
     * @return whether a staged span may admit the whole vectors inside it
     */
    static boolean isSpanFillOn() {
        return spanFill && BROKER.maxSlabs() != 0;
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
     * difference from a {@code Guava} cache, which has to be rebuilt because {@code maximumWeight} is fixed at
     * {@code build()}, so every change cost the whole cache and a re-warm.
     *
     * <p>Shrinking drops the oldest slabs <b>of the fullest segments</b> rather than the oldest globally,
     * which is the same "keeps what fits" guarantee with a different tie-break and is what each segment
     * owning its slabs implies.
     *
     * @param maxBytes the wanted ceiling in bytes; {@code 0} or less turns the cache off
     */
    static void configure(final long maxBytes) {
        final int slabBytes = BROKER.slabBytes();
        final int wanted = wantedSlabs(maxBytes);
        if (wanted == BROKER.maxSlabs()) {
            // Before the lock and not after it, because this is called once per prefetch burst from the read
            // path to pick up a setting change, and in the steady state the answer is always no change.
            // Taking any lock to find that out would put every burst through a lock a fill uses.
            return;
        }
        final List<DirectIOSlab> detached = new ArrayList<>();
        BROKER.lock().lock();
        try {
            if (wanted != BROKER.maxSlabs()) {
                if (maxBytes > 0 && wanted == 0) {
                    log.warn(
                        "A vector cache target of {} bytes is under one {} byte slab, so the cache will hold nothing",
                        maxBytes,
                        slabBytes
                    );
                }
                final int wantedSegments = wantedSegmentCount(wanted);
                if (wantedSegments != segments.length) {
                    resegment(wantedSegments, detached);
                }
                BROKER.setMaxSlabs(wanted);
                // A new target is a new question, so an earlier allocation failure stops standing.
                BROKER.unlatchGrowth();
                while (BROKER.liveSlabs() > wanted) {
                    final DirectIOCacheSegment fullest = fullestSegment();
                    if (fullest == null || fullest.releaseOldestSlab(detached) == false) {
                        break;
                    }
                }
                if (wanted == 0) {
                    for (final DirectIOCacheSegment segment : segments) {
                        segment.clearIndex();
                    }
                    log.info("The k-NN Direct I/O vector cache is off (knn.direct_io.rescore.cache.max_bytes=0)");
                } else {
                    log.info("The k-NN Direct I/O vector cache holds up to {} slabs of {} bytes", wanted, slabBytes);
                }
            }
        } finally {
            BROKER.lock().unlock();
        }
        BROKER.releaseDetached(detached);
    }

    /**
     * Moves the store to {@code wantedSegments} segments, which <b>releases every slab and every index and
     * costs a re-warm</b>. Called under the broker's lock, from {@link #configure(long)} only.
     *
     * <h4>Why a re-warm, and why that is the right choice</h4>
     * A key's segment is a function of the segment count, so after a change every cached vector would be in
     * the wrong segment. The stripe key makes a cheap rehash <em>possible</em> for powers of two
     * ({@code segment' = segment & (N'-1)}, so halving merges segments pairwise and doubling splits each in
     * two), but it is an O(entries) walk under locks and this happens only when an operator moves
     * {@code max_bytes} across a factor-of-eight-slab boundary. The fork ships one value per tier, so on a
     * node it happens once, at startup, from an empty store.
     *
     * <h4>Why the target goes to zero first</h4>
     * It is the one volatile read every entry point makes, so zeroing it stops every fill that has not yet
     * taken a segment lock. A fill already inside finishes and its slab is released by the drain below; a
     * fill that read the old array and arrives late is stopped by {@link DirectIOCacheSegment#retire()},
     * which is what keeps it from leaving a slab the broker counts and no live segment holds.
     */
    private static void resegment(final int wantedSegments, final List<DirectIOSlab> detached) {
        final DirectIOCacheSegment[] old = segments;
        // Whether this change costs anything. Going from the off state to a count is free -- there is
        // nothing to lose -- and counting it would make segment_count_changes non-zero on every node at
        // startup, which is the opposite of what the key is for.
        final boolean costly = BROKER.liveSlabs() > 0;
        BROKER.setMaxSlabs(0);
        for (final DirectIOCacheSegment segment : old) {
            segment.retire();
            while (segment.releaseOldestSlab(detached)) {
                // Every slab, one write-lock acquisition at a time.
            }
            segment.clearIndex();
        }
        final DirectIOCacheSegment[] fresh = new DirectIOCacheSegment[wantedSegments];
        for (int i = 0; i < wantedSegments; i++) {
            fresh[i] = new DirectIOCacheSegment(BROKER);
        }
        segments = fresh;
        if (costly) {
            segmentCountChanges.increment();
            log.info("The k-NN Direct I/O vector cache is now {} segments, from {}; it refills", wantedSegments, old.length);
        } else {
            log.debug("The k-NN Direct I/O vector cache is now {} segments, from {}; it held nothing", wantedSegments, old.length);
        }
    }

    /**
     * The segment holding the most slabs, which is the one a shrink takes from. Called under the broker's
     * lock, and it takes no segment lock: the counts are volatile and a stale one only costs the shrink a
     * second pass round the loop.
     */
    private static DirectIOCacheSegment fullestSegment() {
        DirectIOCacheSegment fullest = null;
        int most = 0;
        for (final DirectIOCacheSegment segment : segments) {
            final int held = segment.liveSlabs();
            if (held > most) {
                most = held;
                fullest = segment;
            }
        }
        return fullest;
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
     * are the vectors served. An earlier shape counted both sites, so its totals were twice the vectors.
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
     *
     * <p>A lock-free sum of the segments' volatile counters since Q, where it took the store's one read lock.
     */
    public static long bytes() {
        long total = 0;
        for (final DirectIOCacheSegment segment : segments) {
            total += segment.liveEntryBytes();
        }
        return total;
    }

    /** Vectors held right now. A lock-free sum of the segments' volatile counters. */
    public static long entries() {
        long total = 0;
        for (final DirectIOCacheSegment segment : segments) {
            total += segment.liveEntries();
        }
        return total;
    }

    /** The store's byte target, rounded down to whole slabs; {@code 0} when the cache is off. */
    public static long maxBytes() {
        return (long) BROKER.maxSlabs() * BROKER.slabBytes();
    }

    /** Slabs held right now, which the broker counts for the node. */
    public static long slabs() {
        return BROKER.liveSlabs();
    }

    /** Bytes of off-heap slab held right now, which is what the node is paying for. */
    public static long slabBytes() {
        return slabs() * BROKER.slabBytes();
    }

    /** Slabs taken from the head of a FIFO queue to make room. */
    public static long reclaimedSlabs() {
        return reclaimedSlabs.sum();
    }

    /** Vectors kept by a second chance when their slab was reclaimed. */
    public static long readmissions() {
        return readmissions.sum();
    }

    /**
     * Vectors admitted because they lay inside a span a burst had already staged, rather than because a read
     * asked for them. Zero while the store is off, and zero once it reaches its slab target.
     */
    public static long spanFills() {
        return spanFills.sum();
    }

    /**
     * Slab allocations the backend refused. One per latch: after a failure the store stops growing until
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
     * Files whose cached vectors were freed because the input owning their {@code O_DIRECT} handle closed.
     * Counts a close that freed at least one entry, so a {@code .vec} the store never held does not
     * move it. On a merging node this rises with the merges, and {@code evictions} standing still while it
     * does is what says the merged-away segments cost the live ones nothing.
     */
    public static long closedFiles() {
        return closedFiles.sum();
    }

    /** Cached vectors freed at a close, which is room the store got back without evicting anything. */
    public static long closedFileEntries() {
        return closedFileEntries.sum();
    }

    /**
     * Vectors a blocking read put back after the filter-then-read race lost them. These are also
     * counted in {@link #readmissions()}, which BACKLOG asks for; this key is what keeps the two sources of
     * a readmission separable, because the other one is the second chance.
     *
     * <p>What it says: the race happened this often and cost a synchronous device read each time, and the
     * vector is cached again rather than staying a miss. Non-zero and rising with a <em>stable</em> hit rate
     * is the race doing what it does; non-zero and rising while the hit rate falls means the store is
     * thrashing and the eviction, not the race, is the thing to look at.
     */
    public static long raceReadmissions() {
        return raceReadmissions.sum();
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
        long total = 0;
        for (final DirectIOCacheSegment segment : segments) {
            total += segment.indexBytes();
        }
        return total;
    }

    // ---------------------------------------------------------------------------------------------------
    // Test seams
    // ---------------------------------------------------------------------------------------------------

    /**
     * Test seam: publishes whether span fill is on. A node never calls this; span fill is
     * {@link #SPAN_FILL}, and the arms that measure the store with it off have no other way to ask for it.
     */
    static void setSpanFillForTesting(final boolean on) {
        spanFill = on;
    }

    /**
     * Test seam: publishes the read-around width, {@code 0} turning read-around off. A node never calls
     * this; the width is {@link #READ_AROUND_BYTES}.
     */
    static void setReadAroundBytesForTesting(final long bytes) {
        readAroundBytes = Math.max(0L, bytes);
    }

    /** Test seam: empties the store, releases every slab, and zeroes every counter. */
    static void resetForTesting() {
        BROKER.resetSlabMemory();
        configuredMaxBytes = 0L;
        configure(0);
        awaitReleasesForTesting();
        BROKER.setSlabBytes(DirectIOSlab.SLAB_BYTES);
        spanFill = SPAN_FILL;
        readAroundBytes = READ_AROUND_BYTES;
        segmentsForTesting = 0;
        segments = new DirectIOCacheSegment[] { new DirectIOCacheSegment(BROKER) };
        hits.reset();
        misses.reset();
        insertions.reset();
        reclaimedSlabs.reset();
        readmissions.reset();
        droppedEntries.reset();
        reclaimNanos.reset();
        spanFills.reset();
        slabAllocFailures.reset();
        closedFiles.reset();
        closedFileEntries.reset();
        raceReadmissions.reset();
        slabTransfers.reset();
        segmentCountChanges.reset();
        contendedReadAcquisitions.reset();
        contendedWriteAcquisitions.reset();
        readLockWaitNanos.reset();
        writeLockWaitNanos.reset();
        readAcquisitions.reset();
        writeAcquisitions.reset();
        brokerLockAcquisitions.reset();
        acquisitionCounting = false;
        lockTimingEnabled = true;
        longestCloseHoldNanos.set(0);
        longestCloseHoldEntries.set(0);
    }

    /**
     * Test seam: where slab memory comes from and goes back to, so a suite can exercise a backend that will
     * not allocate and can see which thread, holding which lock, each call runs on. Takes effect on
     * the next growth attempt or release and is restored by {@link #resetForTesting()}.
     *
     * @param memory the allocate and release pair to use until the next reset
     */
    static void setSlabMemoryForTesting(final SlabMemory memory) {
        BROKER.setSlabMemory(memory);
    }

    /**
     * Test seam: waits for every release in flight to have given its memory back, because the broker does
     * that on a thread of its own.
     */
    static void awaitReleasesForTesting() {
        BROKER.awaitReleases();
    }

    /**
     * Test seam: whether the calling thread is inside this store's locks at all, which since Q means the
     * broker's lock or any segment's lock in either mode. the property is that neither a slab allocation
     * nor a slab release ever runs while it is.
     *
     * @return whether this thread holds any of the store's locks
     */
    static boolean lockHeldByCurrentThreadForTesting() {
        if (BROKER.lockHeldByCurrentThread()) {
            return true;
        }
        for (final DirectIOCacheSegment segment : segments) {
            if (segment.lockHeldByCurrentThread()) {
                return true;
            }
        }
        return false;
    }

    /**
     * Test seam: the slab size, so a suite can exercise eviction and the second chance without allocating
     * 32 MiB at a time. Only valid with the cache off, which {@link #resetForTesting()} guarantees.
     *
     * @param bytes the slab size to use until the next reset
     */
    static void setSlabBytesForTesting(final int bytes) {
        if (BROKER.maxSlabs() != 0) {
            throw new IllegalStateException("The slab size can only be changed while the cache is off");
        }
        BROKER.setSlabBytes(bytes);
    }

    /** Test seam: the number of slots {@code length}-byte vectors get in one slab. */
    static int slotsPerSlab(final int length) {
        return Math.min(BROKER.slabBytes() / length, DirectIOSlotMap.MAX_SLOTS);
    }

    /** Test seam: index tables right now, which is one per (segment, file), so a dead one's going shows. */
    static int indexedFiles() {
        int total = 0;
        for (final DirectIOCacheSegment segment : segments) {
            total += segment.indexedFiles();
        }
        return total;
    }

    /** Test seam: whether {@code fileId} has an index table in any segment right now. */
    static boolean hasIndexFor(final String fileId) {
        for (final DirectIOCacheSegment segment : segments) {
            if (segment.hasIndexFor(fileId)) {
                return true;
            }
        }
        return false;
    }

    /** The segment count in force, which {@code direct_io_stats.cache.segments} reports. */
    public static int segmentCount() {
        return segments.length;
    }

    /** Slabs the broker lent from one segment to another because the borrower held none. */
    public static long slabTransfers() {
        return slabTransfers.sum();
    }

    /**
     * The most and the fewest vectors any one segment holds right now, which is the entry skew the hit-rate
     * cost of segmenting the eviction is priced against. At the chosen operating point of eight slabs per
     * segment an emulation of this policy measured the spread within a few per cent of the mean, and the
     * hit-rate loss that skew carries inside the bar it was designed to.
     *
     * <p>A lock-free sum of volatile counters, like {@link #entries()}, so a wide spread between the two is
     * readable on a node without the stats endpoint competing with the query path.
     */
    public static long maxSegmentEntries() {
        long most = 0;
        for (final DirectIOCacheSegment segment : segments) {
            most = Math.max(most, segment.liveEntries());
        }
        return most;
    }

    /** The fewest vectors any one segment holds right now. See {@link #maxSegmentEntries()}. */
    public static long minSegmentEntries() {
        long fewest = Long.MAX_VALUE;
        for (final DirectIOCacheSegment segment : segments) {
            fewest = Math.min(fewest, segment.liveEntries());
        }
        return fewest == Long.MAX_VALUE ? 0 : fewest;
    }

    /** Read acquisitions that found a thread already queued on a segment's lock. */
    public static long contendedReadAcquisitions() {
        return contendedReadAcquisitions.sum();
    }

    /** Write acquisitions that found a thread already queued on a segment's lock. */
    public static long contendedWriteAcquisitions() {
        return contendedWriteAcquisitions.sum();
    }

    /**
     * Nanoseconds spent waiting for a segment's read lock, over the <b>contended</b> acquisitions only: an
     * uncontended one's wait was measured below the cost of the two {@code System.nanoTime} calls it would
     * take to measure it, so it is not timed.
     */
    public static long readLockWaitNanos() {
        return readLockWaitNanos.sum();
    }

    /** {@link #readLockWaitNanos()} for the write lock. */
    public static long writeLockWaitNanos() {
        return writeLockWaitNanos.sum();
    }

    /**
     * Test seam: counts every acquisition, not only the contended ones, so a test can assert the criterion
     * the batching exists for -- write acquisitions per burst no more than the segments the burst touched.
     *
     * @param on whether to count
     */
    static void setAcquisitionCountingForTesting(final boolean on) {
        acquisitionCounting = on;
        readAcquisitions.reset();
        writeAcquisitions.reset();
        brokerLockAcquisitions.reset();
    }

    static long readAcquisitionsForTesting() {
        return readAcquisitions.sum();
    }

    static long writeAcquisitionsForTesting() {
        return writeAcquisitions.sum();
    }

    static long brokerLockAcquisitionsForTesting() {
        return brokerLockAcquisitions.sum();
    }

    /** Test seam: whether a contended acquisition is timed, so the timing can be priced against no timing. */
    static void setLockTimingForTesting(final boolean on) {
        lockTimingEnabled = on;
    }

    /** Resizes that crossed a segment-count boundary, each of which cost the store a re-warm. */
    public static long segmentCountChanges() {
        return segmentCountChanges.sum();
    }

    /**
     * The longest a close has held one segment's write lock, which is the longest a lookup in that segment
     * could have waited behind one. Lookups in the other {@code N - 1} segments wait for nothing at all.
     */
    public static long longestCloseHoldNanos() {
        return longestCloseHoldNanos.get();
    }

    /** The most entries one segment's write lock was held for by a close. See the field's javadoc. */
    public static long longestCloseHoldEntries() {
        return longestCloseHoldEntries.get();
    }

    /**
     * Test seam: pins the segment count, so the suite and the stress can exercise a count this desktop would
     * not derive. Only valid with the cache off, which {@link #resetForTesting()} guarantees, and only a
     * power of two, because the selector masks rather than divides.
     *
     * @param count the count to pin, or 0 to go back to deriving it from the node
     */
    static void setSegmentsForTesting(final int count) {
        if (BROKER.maxSlabs() != 0) {
            throw new IllegalStateException("The segment count can only be changed while the cache is off");
        }
        if (count < 0 || (count > 0 && Integer.bitCount(count) != 1)) {
            throw new IllegalArgumentException("The segment count must be 0 or a power of two, not " + count);
        }
        segmentsForTesting = count;
    }

    /**
     * Test seam: how many of the store's segment locks the calling thread holds. <b>The design's one
     * invariant is that this is never more than one</b> on any production path, which is the whole of the
     * deadlock proof, and a probe {@link SlabMemory} asserting it at every allocate and release is what pins
     * it. The test-only {@link #checkInvariantsForTesting()} is the one exception and takes them all, in
     * index order.
     *
     * @return the number of segment locks held, in either mode
     */
    static int segmentLocksHeldByCurrentThreadForTesting() {
        int held = 0;
        for (final DirectIOCacheSegment segment : segments) {
            if (segment.lockHeldByCurrentThread()) {
                held++;
            }
        }
        return held;
    }

    /**
     * Test seam: re-derives the live entry count and byte total from the slabs themselves and checks them
     * against the running counters, so a test can assert that no path has lost or double counted a slot.
     * Every slot that still carries a back pointer must also be the slot its file's index names for it, and
     * that index must be its own segment's.
     *
     * <p>Takes the broker's lock and then every segment's write lock, <b>in index order</b>, so the check is
     * atomic against the store. That is the one place more than one segment lock is held at once, and it is
     * deadlock free for the same reason the production order is: the broker's lock comes first and the
     * segments are always taken in the same order.
     *
     * @throws IllegalStateException on any disagreement, naming it
     */
    static void checkInvariantsForTesting() {
        final DirectIOCacheSegment[] all = segments;
        final long[] totals = new long[3];
        BROKER.lock().lock();
        try {
            int locked = 0;
            try {
                for (; locked < all.length; locked++) {
                    all[locked].lockForInvariantCheck();
                }
                for (final DirectIOCacheSegment segment : all) {
                    segment.checkInvariantsUnderLock(totals);
                }
                // Both global checks inside the locks, because outside them a fill in flight makes either
                // one false for a moment and a stress test would read that as a corruption.
                if (totals[2] != BROKER.liveSlabs()) {
                    throw new IllegalStateException(
                        "the segments hold " + totals[2] + " slabs but the broker counts " + BROKER.liveSlabs()
                    );
                }
                final int pending = BROKER.pendingSlabs();
                final int live = BROKER.liveSlabs();
                if (pending < 0 || live + pending > Math.max(BROKER.maxSlabs(), live)) {
                    throw new IllegalStateException(
                        live + " live and " + pending + " pending slabs against a target of " + BROKER.maxSlabs()
                    );
                }
            } finally {
                for (int i = locked - 1; i >= 0; i--) {
                    all[i].unlockAfterInvariantCheck();
                }
            }
        } finally {
            BROKER.lock().unlock();
        }
    }
}

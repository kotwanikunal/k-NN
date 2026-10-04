/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.FilterDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.opensearch.common.Nullable;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.codec.KNN1040Codec.Faiss1040ScalarQuantizedKnnVectorsFormat;
import org.opensearch.knn.index.codec.KNN990Codec.NativeEngines990KnnVectorsFormat;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.util.Set;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.BooleanSupplier;
import java.util.function.LongSupplier;

/**
 * The storage layer's half of the directory design: the {@link Directory} that decides <em>how</em> a
 * vector data read is served, from the file's name alone.
 *
 * <h2>The dispatch rule, in one place</h2>
 * A read is served with {@code O_DIRECT} if and only if <b>all six</b> of these hold:
 *
 * <ol>
 *   <li>the file is a <b>faiss or memory-optimized-search</b> full-precision flat vector file — see
 *       {@link #isFaissVectorData};
 *   <li>{@link KNNSettings#isDirectIORescoreEnabled()} is on — the per-substrate enablement rule, which
 *       stays a setting because Direct I/O is a win only where the {@code .vec} working set cannot stay
 *       in the page cache;
 *   <li>the delegate chain reaches an {@link FSDirectory} and <b>every layer above it is a known
 *       byte-passthrough</b> — see {@link #PASSTHROUGH_FILTER_DIRECTORIES}, and
 *       {@link #resolvePath(Directory, String, Set)} for why an unknown layer must decline rather than
 *       be walked past;
 *   <li>the resolved path exists;
 *   <li>the file is at least {@code knn.direct_io.min_file_size} — the size floor, phase 13 audit FIX 1;
 *   <li>the {@code O_DIRECT} open itself succeeds.
 * </ol>
 *
 * Everything else is the delegate's own {@link IndexInput}, returned <b>unwrapped</b>, and each of (3)
 * to (6) is counted under its own reason so an operator can tell them apart — see {@link #declinedChain},
 * {@link #declinedMissing}, {@link #declinedTooSmall}, {@link #declinedOpenFailed}.
 *
 * <h2>Why the name is the whole signal</h2>
 * An earlier shape of this class also required the caller's {@link IOContext} to carry a plugin-defined
 * "this is a re-score read" hint, so that one {@code .vec} file could be served two ways. That bought
 * selectivity at the cost of a channel that had to be threaded through the codec and the query layer: a
 * second reader per segment, a {@code Directory} view to attach the hint below the codec, a capability
 * interface on three readers, and an argument on the exact-search scorer factory. Under the design this
 * class now implements — <em>all</em> faiss/MOS full-precision vectors are read with {@code O_DIRECT}
 * when the feature is on — none of that is needed, because the file name already says everything the
 * rule asks. The whole channel is gone and the query layer names nothing.
 *
 * <h2>One engagement point, and it is the codec</h2>
 * This directory is installed by {@link #routeVectorData(SegmentReadState)} alone — the codec seam, which
 * needs no index setting. Phase 14 deleted the alternative, an
 * {@code IndexStorePlugin.StoreFactory} selected per index with {@code index.store.factory}, because that
 * slot is single valued and {@code @ExperimentalApi}: one factory per index, already claimed by
 * {@code store-subdirectory}, and in the way of any encryption-at-rest wrapper. The codec seam is the
 * only engagement a plugin can take that composes with everything else.
 *
 * <h2>Compound segments delegate</h2>
 * The {@code Directory} a codec is handed for a <b>compound</b> segment is the {@code CompoundDirectory}
 * ({@code Lucene90CompoundReader}), which is not a {@link FilterDirectory} and has no filesystem path, so
 * {@link #resolvePath} answers {@code null} and the entry is served by the compound reader's own mmap
 * slice. That is the right answer rather than a gap: a compound segment is a freshly flushed one, and
 * phase 13's E4 measured a flush-sized {@code .vec} as <b>8,016x</b> better served from the page cache
 * than from {@code O_DIRECT}. Phase 14 therefore also deleted {@code KNNVectorCompoundSliceInput}, the
 * 423-line {@code .cfs}-slice interceptor the store-factory design needed to reach that same entry: the
 * code existed to route reads that measurement says must not be routed.
 *
 * <h2>Dispatch granularity is the open, never the file</h2>
 * The decision is taken once per {@code openInput} and never re-taken per read, and that is forced rather
 * than chosen. Lucene's SIMD float scorer binds by a type test on the {@code IndexInput} and then reads
 * the memory segment directly, so it never calls {@code readBytes} — an input transparent enough to keep
 * SIMD is transparent enough that a re-score read would bypass the Direct I/O staging entirely, and an
 * input opaque enough to route a re-score read also declines SIMD. One {@code .vec} object cannot serve
 * both, and under a name-based rule there is only one object. So a routed file is routed for every reader
 * of it — traversal, merge and warmup included — which is why condition 1 is narrowed to the faiss/MOS
 * files rather than to {@code .vec} as such.
 *
 * <h2>Failure is a fallback</h2>
 * Every way Direct I/O can be unavailable — no {@code ExtendedOpenOption.DIRECT} on this JDK, a
 * filesystem that answers {@code EINVAL}, a remote-store or in-memory directory with no path — ends in
 * the delegate's own input and a log line. There is no configuration in which a failure here changes a
 * result, because both routes read the same bytes of the same file.
 *
 * <h2>What this class deliberately does not have</h2>
 * The spike this design grew out of recorded every {@code openInput} in an unbounded list and set a
 * node-wide static flag so a test could tell "never installed" from "installed but unreachable".
 * Neither belongs on a production read path — the list grows for the life of the shard and the static
 * outlives every index that set it. This class keeps bounded counters and no static state at all.
 */
@Log4j2
public final class KNNVectorStorageDirectory extends FilterDirectory {

    /** Lucene's flat full-precision vector data — the only file this routes. */
    static final String VECTOR_DATA_EXTENSION = ".vec";

    /**
     * The {@link FilterDirectory} layers condition 3 is willing to walk <b>past</b>, by fully qualified
     * class name, because both are package private in OpenSearch and cannot be named as types here.
     *
     * <p><b>The allowlist is the guard, and it is deliberately short.</b> The chain the codec seam is
     * handed for a non-compound segment is
     * {@code Store$StoreDirectory -> ByteSizeCachingDirectory -> HybridDirectory}, and resolving a name
     * against the bottom {@link FSDirectory}'s path is only correct if no layer above it changes which
     * bytes a reader sees or which file a name means. These two do not: <b>neither overrides
     * {@code openInput}</b> (verified against {@code ~/workspace/OpenSearch},
     * {@code Store.java:1079-1135} and {@code index/store/ByteSizeCachingDirectory.java:55}), so a read
     * through them is the bottom directory's read.
     *
     * <p>An <b>unknown</b> {@link FilterDirectory} declines, and that is the whole point rather than
     * caution for its own sake. Two shapes that exist and must never be bypassed:
     * <ul>
     *   <li>a <b>byte-transforming</b> layer — encryption at rest. Walking past it would open the raw
     *       on-disk file and serve ciphertext as vectors, silently and only for the routed file;</li>
     *   <li>a <b>path-relocating or composing</b> layer — {@code CompositeDirectory} (remote-backed
     *       storage), {@code NRTCachingDirectory}. {@code name} may not be on local disk at all, or may be
     *       live in a RAM buffer, so the path resolved from the bottom directory is wrong or absent.</li>
     * </ul>
     * Both read correctly through the delegate, so declining costs read-ahead and nothing else. Adding a
     * name here is a claim that that class passes bytes and names through unchanged, and should be made
     * the same way these two were: by reading its {@code openInput}.
     */
    static final Set<String> PASSTHROUGH_FILTER_DIRECTORIES = Set.of(
        "org.opensearch.index.store.Store$StoreDirectory",
        "org.opensearch.index.store.ByteSizeCachingDirectory"
    );

    /** How deep the chain walk will go. A guard must never be the thing that hangs a shard. */
    private static final int MAX_CHAIN_DEPTH = 16;

    /** The segment this directory serves, so a log line can be attributed to one. */
    private final String segmentName;

    /**
     * Condition 2 of the dispatch rule, read at every dispatch rather than captured once, because the
     * setting is dynamic and a segment's {@code .vec} is opened once per reader.
     */
    private final BooleanSupplier directIORescoreEnabled;

    /** Condition 5, read at every dispatch for the same reason condition 2 is. */
    private final LongSupplier minFileSizeBytes;

    private final AtomicLong routedOpens = new AtomicLong();
    private final AtomicLong declinedChain = new AtomicLong();
    private final AtomicLong declinedMissing = new AtomicLong();
    private final AtomicLong declinedTooSmall = new AtomicLong();
    private final AtomicLong declinedOpenFailed = new AtomicLong();

    public KNNVectorStorageDirectory(final Directory delegate, final String segmentName) {
        this(delegate, segmentName, KNNSettings::isDirectIORescoreEnabled, () -> KNNSettings.getDirectIOMinFileSize().getBytes());
    }

    /** Visible for testing, so a test can drive the rule without a cluster service. */
    KNNVectorStorageDirectory(
        final Directory delegate,
        final String segmentName,
        final BooleanSupplier directIORescoreEnabled,
        final LongSupplier minFileSizeBytes
    ) {
        super(delegate);
        this.segmentName = segmentName;
        this.directIORescoreEnabled = directIORescoreEnabled;
        this.minFileSizeBytes = minFileSizeBytes;
    }

    public String segmentName() {
        return segmentName;
    }

    // -------------------------------------------------------------------------------------------------
    // The dispatch point: the open Lucene's flat vectors reader makes for its own .vec
    // -------------------------------------------------------------------------------------------------

    @Override
    public IndexInput openInput(final String name, final IOContext context) throws IOException {
        // Conditions 1 and 2 first, and in this order: the name test is a string comparison and the
        // setting read is a map lookup, so a file this never routes costs no filesystem call at all.
        if (isFaissVectorData(name) && directIORescoreEnabled.getAsBoolean()) {
            final IndexInput direct = directIOInput(name);
            if (direct != null) {
                routedOpens.incrementAndGet();
                DirectIOVectorIndexInput.STATS.recordRoutedOpen();
                log.debug("k-NN vector storage [{}]: serving [{}] with O_DIRECT", segmentName, name);
                return direct;
            }
        }
        return in.openInput(name, context);
    }

    /**
     * Conditions 3 to 6, or {@code null} when any of them declines — in which case exactly one reason
     * counter has been incremented. The returned input owns its own file handle and is closed by whoever
     * closes the input, which is the same ownership {@link Directory#openInput} always has.
     */
    @Nullable
    private IndexInput directIOInput(final String name) {
        // (3) the chain.
        final Path path = resolvePath(in, name, PASSTHROUGH_FILTER_DIRECTORIES);
        if (path == null) {
            decline(declinedChain, DirectIOVectorIndexInput.DeclineReason.CHAIN);
            log.debug(
                "k-NN vector storage [{}]: [{}] has no resolvable filesystem path below [{}], so Direct I/O cannot serve it",
                segmentName,
                name,
                in.getClass().getName()
            );
            return null;
        }
        // (4) and (5) in one syscall: Files.size throws NoSuchFileException for a missing file, which is a
        // different reason with a different cause and is counted apart from the floor.
        final long minBytes = minFileSizeBytes.getAsLong();
        try {
            final long bytes = Files.size(path);
            if (bytes < minBytes) {
                decline(declinedTooSmall, DirectIOVectorIndexInput.DeclineReason.TOO_SMALL);
                log.debug(
                    "k-NN vector storage [{}]: [{}] is {} B, below knn.direct_io.min_file_size of {} B; serving it from the page cache",
                    segmentName,
                    name,
                    bytes,
                    minBytes
                );
                return null;
            }
        } catch (NoSuchFileException e) {
            decline(declinedMissing, DirectIOVectorIndexInput.DeclineReason.MISSING);
            log.debug("k-NN vector storage [{}]: [{}] does not exist at [{}]", segmentName, name, path);
            return null;
        } catch (IOException | RuntimeException e) {
            decline(declinedMissing, DirectIOVectorIndexInput.DeclineReason.MISSING);
            log.warn("k-NN vector storage [{}]: could not size [{}]", segmentName, path, e);
            return null;
        }
        // (6) the open, which is the last thing that can decline and the only one that touches the device.
        try {
            final DirectIOVectorIndexInput direct = DirectIOVectorIndexInput.open(path);
            if (direct == null) {
                decline(declinedOpenFailed, DirectIOVectorIndexInput.DeclineReason.OPEN_FAILED);
                log.warn("k-NN vector storage [{}]: Direct I/O is not available for [{}]", segmentName, path);
            }
            return direct;
        } catch (RuntimeException e) {
            decline(declinedOpenFailed, DirectIOVectorIndexInput.DeclineReason.OPEN_FAILED);
            log.warn("k-NN vector storage [{}]: could not open [{}] with Direct I/O", segmentName, path, e);
            return null;
        }
    }

    /**
     * The file behind {@code name} when every layer of {@code directory}'s chain down to an
     * {@link FSDirectory} is a known byte-passthrough, and {@code null} otherwise.
     *
     * <p><b>Why this is not {@link FilterDirectory#unwrap}.</b> {@code unwrap} walks past every
     * {@link FilterDirectory} whatever it does, and the previous shape of this method used it. A name
     * resolved that way is the <em>bottom</em> directory's file, so any layer that was transforming bytes
     * or relocating names got bypassed for the one file this routes and for no other — which is both a
     * correctness bug and the hardest kind to see, because every other read in the system stays correct.
     * So the walk is an allowlist: an unknown layer stops it and the read goes through the delegate, where
     * that layer gets to do its job.
     *
     * <p>Package private and parameterised on the allowlist so a test can exercise the walk with its own
     * stand-in layers; production always passes {@link #PASSTHROUGH_FILTER_DIRECTORIES}.
     */
    @Nullable
    static Path resolvePath(final Directory directory, final String name, final Set<String> passthrough) {
        Directory current = directory;
        for (int depth = 0; current != null && depth < MAX_CHAIN_DEPTH; depth++) {
            // An FSDirectory is the terminal case and may be a subclass of one: OpenSearch's
            // HybridDirectory extends NIOFSDirectory and chooses mmap or nio per file, but both reads are
            // of getDirectory().resolve(name), so the path is the same either way.
            if (current instanceof FSDirectory fsDirectory) {
                return fsDirectory.getDirectory().resolve(name);
            }
            if (current instanceof FilterDirectory filterDirectory && passthrough.contains(current.getClass().getName())) {
                current = filterDirectory.getDelegate();
                continue;
            }
            return null;
        }
        return null;
    }

    /**
     * Whether {@code name} is the full-precision flat vector data of a <b>faiss or MOS</b> field — the
     * only file this routes.
     *
     * <p>The extension alone is not enough, and the difference is the whole of condition 1. Every flat
     * vectors format writes a {@code .vec}, the Lucene-engine ones included, and a routed file is routed
     * for all of its readers (see the class javadoc). Routing a Lucene-engine {@code .vec} would
     * therefore take Lucene's unquantized HNSW traversal off its memory-segment SIMD scorer, which is
     * outside what this feature is for.
     *
     * <p>The discriminator is already in the name. {@code PerFieldKnnVectorsFormat} builds each field's
     * segment suffix as {@code <formatName>_<n>} from {@code KnnVectorsFormat#getName()}, and
     * {@code Lucene99FlatVectorsReader} names the data file
     * {@code segmentFileName(segmentName, segmentSuffix, "vec")}. So a faiss/MOS fp32 file is
     * {@code _0_NativeEngines990KnnVectorsFormat_0.vec} or
     * {@code _0_Faiss1040ScalarQuantizedKnnVectorsFormat_0.vec}, while a Lucene-engine one carries
     * Lucene's own format name.
     */
    static boolean isFaissVectorData(final String name) {
        if (name == null || name.endsWith(VECTOR_DATA_EXTENSION) == false) {
            return false;
        }
        return name.contains(NativeEngines990KnnVectorsFormat.FORMAT_NAME)
            || name.contains(Faiss1040ScalarQuantizedKnnVectorsFormat.FORMAT_NAME);
    }

    // -------------------------------------------------------------------------------------------------
    // Observability: bounded counters, no node-wide state
    // -------------------------------------------------------------------------------------------------

    /**
     * Records one decline twice: on this directory's own counter, which is per segment and is what a test
     * asserts on, and on {@link DirectIOVectorIndexInput#STATS}, which is node wide and is what an operator
     * reads at {@code _plugins/_knn/stats}. Both are needed because this object is created per segment per
     * reader by the codec seam and is discarded with that reader, so its instance counters can never be
     * reached from a REST API.
     */
    private static void decline(final AtomicLong counter, final DirectIOVectorIndexInput.DeclineReason reason) {
        counter.incrementAndGet();
        DirectIOVectorIndexInput.STATS.recordDecline(reason);
    }

    /** Full-precision vector data opens this directory served with {@code O_DIRECT}. */
    public long routedOpens() {
        return routedOpens.get();
    }

    /**
     * Opens that satisfied conditions 1 and 2 but were declined by one of 3 to 6 — the sum of the four
     * reason counters below. An operator watching only this number sees that routing is not happening; the
     * reasons say why, and they want different answers: a chain decline is a substrate the feature cannot
     * serve, a size decline is working as designed, a missing file is a bug, and an open failure is a
     * filesystem that does not support {@code O_DIRECT}.
     */
    public long declinedOpens() {
        return declinedChain.get() + declinedMissing.get() + declinedTooSmall.get() + declinedOpenFailed.get();
    }

    /** Condition 3: no {@link FSDirectory} below, or an unknown {@link FilterDirectory} in the way. */
    public long declinedChain() {
        return declinedChain.get();
    }

    /** Condition 4: the resolved path does not exist, or could not be sized. */
    public long declinedMissing() {
        return declinedMissing.get();
    }

    /** Condition 5: the file is below {@code knn.direct_io.min_file_size}. */
    public long declinedTooSmall() {
        return declinedTooSmall.get();
    }

    /** Condition 6: the {@code O_DIRECT} open itself failed, and the delegate served the read. */
    public long declinedOpenFailed() {
        return declinedOpenFailed.get();
    }

    // -------------------------------------------------------------------------------------------------
    // The engagement point: the codec seam, which needs no index setting
    // -------------------------------------------------------------------------------------------------

    /**
     * {@code state} with its {@link Directory} wrapped in this class, so that the Lucene flat vectors
     * reader the faiss formats delegate to opens its {@code .vec} through the dispatch rule above —
     * <b>with no {@code index.store.factory} and no store factory in the chain</b>. This is the plugin's
     * only Direct I/O engagement point.
     *
     * <p>A {@code SegmentReadState} is a plain carrier of five public final fields with a public
     * copy-shaped constructor, and the two faiss formats construct the flat reader themselves
     * ({@code NativeEngines990KnnVectorsFormat#fieldsReader},
     * {@code Faiss1040ScalarQuantizedKnnVectorsFormat#fieldsReader}), so the one directory the flat reader
     * will use is ours to choose. The scope is as narrow as the feature: this directory is only ever asked
     * about the files of <em>one segment of one k-NN field</em>, never about the shard's translog, doc
     * values or segment infos — which is what the store factory could not say.
     *
     * <h4>Two cases, and what each gets</h4>
     * <ul>
     *   <li><b>Non-compound segment</b> — {@code state.directory} is the shard's directory, which bottoms
     *       out at an {@link FSDirectory}, so {@link #resolvePath} answers and the {@code .vec} is routed.
     *       This is the case the feature is for: every merged segment.</li>
     *   <li><b>Compound segment</b> — {@code state.directory} is a {@code CompoundDirectory}, which is
     *       <em>not</em> a {@link FilterDirectory}, so {@link #resolvePath} returns {@code null} and the
     *       open is <b>declined</b>; see the class javadoc for why that is the measured right answer.</li>
     * </ul>
     *
     * <p>Inert when the feature is off: the caller's own state object is returned, not a wrapper of it.
     *
     * <p>The wrapper is never closed, and must not be: {@link FilterDirectory#close} closes its delegate,
     * and the delegate here is the shard's (or the compound segment's) directory, owned by the reader that
     * built {@code state}. Nothing closes a {@code SegmentReadState}'s directory, so this object simply
     * becomes garbage with the reader — which is why it holds no resource of its own.
     */
    public static SegmentReadState routeVectorData(final SegmentReadState state) {
        if (KNNSettings.isDirectIORescoreEnabled() == false) {
            return state;
        }
        return new SegmentReadState(
            new KNNVectorStorageDirectory(state.directory, state.segmentInfo.name),
            state.segmentInfo,
            state.fieldInfos,
            state.context,
            state.segmentSuffix
        );
    }
}

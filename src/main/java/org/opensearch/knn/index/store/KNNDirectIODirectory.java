/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.CompoundDirectory;
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
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.BooleanSupplier;
import java.util.function.LongSupplier;

/**
 * The storage layer's half of the directory design: the {@link Directory} that decides <em>how</em> a
 * vector data read is served, from the file's name alone.
 *
 * <h2>The dispatch rule, in one place</h2>
 * A read is served with {@code O_DIRECT} if and only if <b>all eight</b> of these hold:
 *
 * <ol>
 *   <li>the file is a <b>faiss or memory-optimized-search</b> flat vector file, of either precision
 *       fp32 or FP16 — see {@link #isFaissVectorData};
 *   <li>{@code knn.direct_io.rescore.enabled} is on — the per-substrate enablement rule, which
 *       stays a setting because Direct I/O is a win only where the {@code .vec} working set cannot stay
 *       in the page cache;
 *   <li>the segment is <b>not compound</b> — a {@code CompoundDirectory} entry has no file of its own.
 *       Checked first among the structural conditions because it is the only one that is <em>expected</em>
 *       rather than diagnostic: see {@link #declinedCompound};
 *   <li>the delegate chain bottoms out at an {@link FSDirectory} — see {@link #resolvePath};
 *   <li>the resolved path exists;
 *   <li>the file is at least {@code knn.direct_io.min_file_size} — the size floor;
 *   <li><b>both</b> opens succeed: the delegate's own input, which the routed input keeps beside its
 *       channel, and the {@code O_DIRECT} open itself;
 *   <li>the file's bytes are the bytes the delegate chain serves — the length, the first 4 KiB and the
 *       codec footer, compared through both routes before either is handed out. See
 *       {@link DirectIOVectorIndexInput#contentMatches} for what the comparison covers and what it
 *       cannot.
 * </ol>
 *
 * Everything else is the delegate's own {@link IndexInput}, returned <b>unwrapped</b>, and each of (3)
 * to (8) is counted under its own reason so an operator can tell them apart — see
 * {@link #declinedCompound}, {@link #declinedChain}, {@link #declinedMissing}, {@link #declinedTooSmall},
 * {@link #declinedOpenFailed}, {@link #declinedContent}.
 *
 * <h2>Why the name is the whole signal</h2>
 * An earlier shape of this class also required the caller's {@link IOContext} to carry a plugin-defined
 * "this is a re-score read" hint, so that one {@code .vec} file could be served two ways. That bought
 * selectivity at the cost of a channel that had to be threaded through the codec and the query layer: a
 * second reader per segment, a {@code Directory} view to attach the hint below the codec, a capability
 * interface on three readers, and an argument on the exact-search scorer factory. Under the design this
 * class now implements — <em>all</em> faiss/MOS flat vector files are read with {@code O_DIRECT}
 * when the feature is on — none of that is needed, because the file name already says everything the
 * rule asks. The whole channel is gone and the query layer names nothing.
 *
 * <h2>One engagement point, and it is the codec</h2>
 * This directory is installed by {@link #routeVectorData(SegmentReadState)} alone — the codec seam, which
 * needs no index setting. The alternative, an {@code IndexStorePlugin.StoreFactory} selected per index with
 * {@code index.store.factory}, does not compose: that slot is single valued and {@code @ExperimentalApi},
 * so it is one factory per index, already claimed by {@code store-subdirectory} and in the way of any
 * encryption-at-rest wrapper. The codec seam is the only engagement a plugin can take that composes with
 * everything else.
 *
 * <h2>Compound segments delegate, and that is most segments</h2>
 * The {@code Directory} a codec is handed for a <b>compound</b> segment is the {@link CompoundDirectory}
 * ({@code Lucene90CompoundReader}), which has no filesystem path, so condition 3 declines and the entry is
 * served by the compound reader's own mmap slice. That is the right answer rather than a gap: a compound
 * segment is a freshly flushed one, and a flush-sized {@code .vec} was measured orders of magnitude better
 * served from the page cache than from {@code O_DIRECT}. So there is deliberately no {@code .cfs}-slice
 * interceptor to reach that entry: it would exist to route reads the measurement says must not be routed.
 *
 * <p><b>How often that happens is not a corner case.</b> At the shipped defaults <em>every</em> flushed
 * segment is compound, because {@code index.use_compound_file} defaults to {@code true}
 * ({@code EngineConfig.java:262-266}) and that value is what reaches
 * {@code IndexWriterConfig#setUseCompoundFile} ({@code NativeLuceneIndexWriterFactory.java:208}). So only
 * <b>merged</b> segments ever route, which is why {@link #declinedCompound} is counted apart from
 * {@link #declinedChain}: a healthy node accumulates it steadily and it means nothing is wrong.
 * {@code index.compound_format} is a <em>different</em> setting that only feeds the merge policy's
 * {@code noCFSRatio} ({@code TieredMergePolicyProvider.java:155-157}, {@code :232}), i.e. merged segments.
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
 * filesystem that answers {@code EINVAL}, a remote-store or in-memory directory with no path, a layer of
 * the chain that transforms the bytes a reader sees — ends in the delegate's own input and a log line. A
 * failure here cannot change a result, and that rests on condition 8 rather than on an assumption: the two
 * routes are compared before either is handed out, so a file is routed only where they have been shown to
 * read the same bytes.
 *
 * <h2>What this class deliberately does not have</h2>
 * The spike this design grew out of recorded every {@code openInput} in an unbounded list and set a
 * node-wide static flag so a test could tell "never installed" from "installed but unreachable".
 * Neither belongs on a production read path — the list grows for the life of the shard and the static
 * outlives every index that set it. This class keeps bounded counters and only the two node-wide
 * settings the dispatch rule asks about.
 *
 * <h2>Why those two are static</h2>
 * {@link #rescoreEnabled} and {@link #minFileSizeBytes} are node settings: one value for the node, not
 * one per directory. {@link KNNSettings} publishes them here when the cluster applier accepts a change,
 * so the dispatch rule reads a field rather than a settings map. A per-instance copy taken at
 * construction would make a change reach only the readers opened after it, and a directory lives as long
 * as its segment reader does.
 */
@Log4j2
public final class KNNDirectIODirectory extends FilterDirectory {

    /** Lucene's flat vector data — the only extension this routes, at either precision. */
    static final String VECTOR_DATA_EXTENSION = ".vec";

    /** How deep the chain description will go. A log line must never be the thing that hangs a shard. */
    private static final int MAX_CHAIN_DEPTH = 16;

    /**
     * Condition 2 of the dispatch rule, {@code knn.direct_io.rescore.enabled}. Published by
     * {@link KNNSettings} on a change and seeded there at startup; until then it is the setting's own
     * default, which is what a JVM with no cluster service runs on.
     */
    private static volatile boolean rescoreEnabled = KNNSettings.KNN_DIRECT_IO_RESCORE_ENABLED_DEFAULT_VALUE;

    /** Condition 6, {@code knn.direct_io.min_file_size}, published the same way. */
    private static volatile long minFileSizeBytes = KNNSettings.KNN_DIRECT_IO_MIN_FILE_SIZE_DEFAULT_VALUE.getBytes();

    /** The segment this directory serves, so a log line can be attributed to one. */
    private final String segmentName;

    /**
     * Where condition 2 is read from, at every dispatch rather than captured once, because the setting is
     * dynamic and a segment's {@code .vec} is opened once per reader. Production reads
     * {@link #rescoreEnabled}; a test substitutes its own answer through the four-argument constructor.
     */
    private final BooleanSupplier directIORescoreEnabled;

    /** Where condition 6 is read from, at every dispatch for the same reason condition 2 is. */
    private final LongSupplier minFileSize;

    private final AtomicLong routedOpens = new AtomicLong();
    private final AtomicLong declinedCompound = new AtomicLong();
    private final AtomicLong declinedChain = new AtomicLong();
    private final AtomicLong declinedMissing = new AtomicLong();
    private final AtomicLong declinedTooSmall = new AtomicLong();
    private final AtomicLong declinedOpenFailed = new AtomicLong();
    private final AtomicLong declinedContent = new AtomicLong();

    public KNNDirectIODirectory(final Directory delegate, final String segmentName) {
        this(delegate, segmentName, KNNDirectIODirectory::isRescoreEnabled, KNNDirectIODirectory::minFileSizeBytes);
    }

    /** Visible for testing, so a test can drive the rule without a cluster service. */
    KNNDirectIODirectory(
        final Directory delegate,
        final String segmentName,
        final BooleanSupplier directIORescoreEnabled,
        final LongSupplier minFileSize
    ) {
        super(delegate);
        this.segmentName = segmentName;
        this.directIORescoreEnabled = directIORescoreEnabled;
        this.minFileSize = minFileSize;
    }

    // -------------------------------------------------------------------------------------------------
    // The two node settings the dispatch rule asks about
    // -------------------------------------------------------------------------------------------------

    /** Condition 2, as the node last published it. */
    static boolean isRescoreEnabled() {
        return rescoreEnabled;
    }

    /** Condition 6, as the node last published it. */
    static long minFileSizeBytes() {
        return minFileSizeBytes;
    }

    /**
     * Publishes {@code knn.direct_io.rescore.enabled} <b>and gives the vector store back with it</b>, which
     * is the whole point of the key being dynamic: the slabs serve routed reads only, so a node that has
     * stopped routing is holding memory for nothing, and waiting for a burst to notice would never free it
     * on a node that has stopped reading as well. Turning the route back on restores the target, so the
     * store grows again at the next fill.
     *
     * <p>Called by {@link KNNSettings} from the cluster applier thread, and it does not wait:
     * {@code DirectIOVectorCache} detaches the slabs under the broker's lock and frees them on its own
     * release thread.
     *
     * <p>An input with no delegate to fall back to stays on {@code O_DIRECT} whatever this says, and its
     * store goes with it (see {@code DirectIOVectorIndexInput#mmapFallback}). That is every probe and bench
     * driver in this package, and none of them has a cluster service, so none of them reaches this method.
     */
    public static void publishRescoreEnabled(final boolean enabled) {
        rescoreEnabled = enabled;
        DirectIOVectorCache.applyConfiguredSize(enabled);
    }

    /**
     * Publishes {@code knn.direct_io.min_file_size}. A change reaches the next open of a {@code .vec},
     * which is the only cadence the condition is asked at. Not clamped: the setting is a byte size with a
     * floor of {@code 0b}, so there is no negative value to defend against.
     */
    public static void publishMinFileSizeBytes(final long bytes) {
        minFileSizeBytes = bytes;
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
            final IndexInput direct = directIOInput(name, context);
            if (direct != null) {
                routedOpens.incrementAndGet();
                DirectIOVectorIndexInput.STATS.recordRoutedOpen();
                log.debug("k-NN Direct I/O [{}]: serving [{}] with O_DIRECT", segmentName, name);
                return direct;
            }
        }
        return in.openInput(name, context);
    }

    /**
     * Conditions 3 to 8, or {@code null} when any of them declines — in which case exactly one reason
     * counter has been incremented. The returned input owns its own file handle <b>and the delegate's input
     * beside it</b>, and both are closed by whoever closes the input, which is the same ownership
     * {@link Directory#openInput} always has.
     */
    @Nullable
    private IndexInput directIOInput(final String name, final IOContext context) {
        // (3) the compound case, taken BEFORE the path resolution. A CompoundDirectory is not an FSDirectory,
        // so condition 4 would decline it anyway and count declinedChain -- but that conflates the one
        // decline reason that is EXPECTED with the one that means "this substrate has no file to route". At
        // the shipped defaults every flushed segment is compound (index.use_compound_file defaults to true,
        // EngineConfig.java:262-266), so this counter moves on every reader open of every flushed segment,
        // and an operator who cannot tell it from a chain decline cannot tell a healthy node from a
        // misconfigured one.
        if (in instanceof CompoundDirectory) {
            decline(declinedCompound, DirectIOVectorIndexInput.DeclineReason.COMPOUND);
            log.debug(
                "k-NN Direct I/O [{}]: [{}] is an entry in a compound segment, which has no file of its "
                    + "own; the compound reader's mmap slice serves it",
                segmentName,
                name
            );
            return null;
        }
        // (4) the chain.
        final Path path = resolvePath(in, name);
        if (path == null) {
            decline(declinedChain, DirectIOVectorIndexInput.DeclineReason.CHAIN);
            // The whole chain, not just its top: a chain decline is the one reason an operator cannot
            // diagnose from the counter alone, because the answer is always "nothing below this chain has a
            // filesystem path" and the only useful question is what the chain was. The last entry is the
            // layer the description stopped at.
            log.debug(
                "k-NN Direct I/O [{}]: [{}] has no resolvable filesystem path, because the delegate chain {} "
                    + "does not bottom out at an FSDirectory",
                segmentName,
                name,
                describeChain(in)
            );
            return null;
        }
        // (5) and (6) in one syscall: Files.size throws NoSuchFileException for a missing file, which is a
        // different reason with a different cause and is counted apart from the floor.
        final long minBytes = minFileSize.getAsLong();
        try {
            final long bytes = Files.size(path);
            if (bytes < minBytes) {
                decline(declinedTooSmall, DirectIOVectorIndexInput.DeclineReason.TOO_SMALL);
                log.debug(
                    "k-NN Direct I/O [{}]: [{}] is {} B, below knn.direct_io.min_file_size of {} B; serving it from the page cache",
                    segmentName,
                    name,
                    bytes,
                    minBytes
                );
                return null;
            }
        } catch (NoSuchFileException e) {
            decline(declinedMissing, DirectIOVectorIndexInput.DeclineReason.MISSING);
            log.debug("k-NN Direct I/O [{}]: [{}] does not exist at [{}]", segmentName, name, path);
            return null;
        } catch (IOException | RuntimeException e) {
            decline(declinedMissing, DirectIOVectorIndexInput.DeclineReason.MISSING);
            log.warn("k-NN Direct I/O [{}]: could not size [{}]", segmentName, path, e);
            return null;
        }
        // (7a) the delegate's own input. Two inputs are opened rather than one, and the second is not a
        // spare: it is kept beside the O_DIRECT channel so that knn.direct_io.rescore.enabled can be turned off under a
        // reader that is already open and the read path changes at that reader's next prefetch burst, with
        // no reopen. Its cost is one file handle per routed .vec per reader, and it is paid only by a node
        // that is routing. It is opened first, because a failure there must leave nothing behind: the
        // O_DIRECT channel would otherwise have to be closed by hand on a path that has no input to close
        // it with.
        final IndexInput fallback;
        try {
            fallback = in.openInput(name, context);
        } catch (IOException | RuntimeException e) {
            // Fail closed: without the delegate's input there is nothing to compare the file's bytes with,
            // and condition 8 is the only thing standing between a transforming layer and ciphertext served
            // as vectors. An unverifiable open declines.
            decline(declinedOpenFailed, DirectIOVectorIndexInput.DeclineReason.OPEN_FAILED);
            log.warn(
                "k-NN Direct I/O [{}]: could not open a delegate input for [{}], so its bytes cannot be "
                    + "compared with the file's; the delegate serves it",
                segmentName,
                name,
                e
            );
            return null;
        }
        // (7b) the O_DIRECT open, the only condition that touches the device.
        final DirectIOVectorIndexInput direct;
        try {
            direct = DirectIOVectorIndexInput.open(path, fallback, directIORescoreEnabled);
        } catch (RuntimeException e) {
            decline(declinedOpenFailed, DirectIOVectorIndexInput.DeclineReason.OPEN_FAILED);
            log.warn("k-NN Direct I/O [{}]: could not open [{}] with Direct I/O", segmentName, path, e);
            closeQuietly(fallback);
            return null;
        }
        if (direct == null) {
            decline(declinedOpenFailed, DirectIOVectorIndexInput.DeclineReason.OPEN_FAILED);
            log.warn("k-NN Direct I/O [{}]: Direct I/O is not available for [{}]", segmentName, path);
            closeQuietly(fallback);
            return null;
        }
        // (8) the bytes. Both routes are open, so the claim that every layer above the FSDirectory passes
        // bytes through can be checked rather than assumed from a class name.
        try {
            if (direct.contentMatches(fallback)) {
                return direct;
            }
            decline(declinedContent, DirectIOVectorIndexInput.DeclineReason.CONTENT);
            // At warn, and with the whole chain: unlike a compound decline this one means a substrate the
            // operator may not know is in the chain, and the chain is the only place the answer can be.
            log.warn(
                "k-NN Direct I/O [{}]: the bytes [{}] serves through the delegate chain {} are not the bytes "
                    + "of [{}] on disk, so a layer in that chain transforms them; the delegate serves it",
                segmentName,
                name,
                describeChain(in),
                path
            );
        } catch (IOException | RuntimeException e) {
            // Fail closed for the same reason (7a) does: a comparison that could not be made is not a
            // comparison that passed.
            decline(declinedContent, DirectIOVectorIndexInput.DeclineReason.CONTENT);
            log.warn(
                "k-NN Direct I/O [{}]: could not compare the bytes of [{}] through the delegate chain with "
                    + "[{}] on disk; the delegate serves it",
                segmentName,
                name,
                path,
                e
            );
        }
        // Both inputs go, because the routed input owns the delegate's; and through closeAbandoned rather
        // than close, so that an open which read nothing does not free the cached vectors of a file another
        // reader may be serving from the store.
        abandonQuietly(direct);
        return null;
    }

    /** Closes a delegate input nothing took ownership of, because the {@code O_DIRECT} open declined. */
    private void closeQuietly(@Nullable final IndexInput input) {
        if (input == null) {
            return;
        }
        try {
            input.close();
        } catch (IOException | RuntimeException e) {
            log.debug("k-NN Direct I/O [{}]: could not close an abandoned delegate input", segmentName, e);
        }
    }

    /**
     * Releases a routed input, and the delegate input it owns, that this open is abandoning: the content
     * decline, which is the one decline taken after both opens succeeded. Not {@code close()}, because this
     * input read nothing and the cache is keyed on the file rather than on the handle — see
     * {@link DirectIOVectorIndexInput#closeAbandoned}.
     */
    private void abandonQuietly(final DirectIOVectorIndexInput input) {
        try {
            input.closeAbandoned();
        } catch (IOException | RuntimeException e) {
            log.debug("k-NN Direct I/O [{}]: could not close an abandoned routed input", segmentName, e);
        }
    }

    /**
     * The delegate chain of {@code directory} as a list of class names, outermost first, stopping at the
     * first layer that is not a {@link FilterDirectory}. For the decline log lines only.
     */
    static List<String> describeChain(final Directory directory) {
        final List<String> chain = new ArrayList<>();
        Directory current = directory;
        for (int depth = 0; current != null && depth < MAX_CHAIN_DEPTH; depth++) {
            chain.add(current.getClass().getName());
            if (current instanceof FilterDirectory filterDirectory) {
                current = filterDirectory.getDelegate();
            } else {
                break;
            }
        }
        return chain;
    }

    /**
     * The file behind {@code name} when {@code directory}'s chain bottoms out at an {@link FSDirectory},
     * and {@code null} otherwise: condition 4.
     *
     * <p><b>Why the layer's class name is not the guard.</b> Resolving a name against the bottom directory's
     * path is correct only if no layer above it changes which bytes a reader sees, and a class name is not a
     * property of the bytes: the shape this most matters for, an encryption-at-rest directory, <em>extends</em>
     * {@link FSDirectory} rather than wrapping one, so it is the terminal case of any walk and is never asked
     * whether it may be crossed. Deciding by name also declines every chain nobody has read yet, which costs
     * the feature on a substrate that was passing bytes through all along.
     *
     * <p>So the resolution is {@link FilterDirectory#unwrap}, which walks past every layer whatever it does,
     * and the claim is made where it can be checked: condition 8 compares the bytes the chain serves with
     * the bytes in the file, after both inputs are open and before either is handed out. See
     * {@link DirectIOVectorIndexInput#contentMatches}.
     *
     * <p>An {@link FSDirectory} is the terminal case and may be a subclass of one: OpenSearch's
     * {@code HybridDirectory} extends {@code NIOFSDirectory} and chooses mmap or nio per file, but both
     * reads are of {@code getDirectory().resolve(name)}, so the path is the same either way.
     */
    @Nullable
    static Path resolvePath(final Directory directory, final String name) {
        // unwrap answers null for null, so a chain with nothing in it needs no case of its own.
        if (FilterDirectory.unwrap(directory) instanceof FSDirectory fsDirectory) {
            return fsDirectory.getDirectory().resolve(name);
        }
        return null;
    }

    /**
     * The two FP16 per-field formats' names. Spelled out here rather than read from the formats because
     * both declare their {@code FORMAT_NAME} private and one of them lives outside the files this phase may
     * change; {@code KNNDirectIODirectoryTests#halfFloatFormatNamesMatchTheFormats} asks the formats
     * themselves and fails if either is ever renamed, so these strings cannot drift in silence.
     */
    static final String HALF_FLOAT_FORMAT_NAME = "NativeEngines990HalfFloatKnnVectorsFormat";
    static final String HALF_FLOAT_SQ_FORMAT_NAME = "Faiss1040HalfFloatScalarQuantizedKnnVectorsFormat";

    /**
     * Whether {@code name} is the flat vector data of a <b>faiss or MOS</b> field — the only file this
     * routes.
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
     *
     * <h4>The two FP16 names</h4>
     * A {@code half_float} faiss or MOS field writes its vectors through
     * {@code NativeEngines990HalfFloatKnnVectorsFormat} or, with {@code sq, bits:1}, through
     * {@code Faiss1040HalfFloatScalarQuantizedKnnVectorsFormat}. Both extend the fp32 formats above and both
     * nest {@code KNN1040HalfFloatFlatVectorsFormat}, whose data extension is also {@code vec}, so the open
     * always reaches this rule — but the inserted {@code HalfFloat} breaks the fp32 substrings, which is why
     * the two names are listed. Nothing else about the rule changes: a half-float {@code .vec} is a flat file of {@code 2 x dimension} bytes per vector
     * that the rescore path reads at random, which is what this feature is for. The store keys on offset and
     * byte length, so the halved vector length needs nothing of it, and the FP16 read path's
     * {@code readBytes} is served by {@code DirectIOVectorIndexInput#isVectorRead}.
     */
    static boolean isFaissVectorData(final String name) {
        if (name == null || name.endsWith(VECTOR_DATA_EXTENSION) == false) {
            return false;
        }
        return name.contains(NativeEngines990KnnVectorsFormat.FORMAT_NAME)
            || name.contains(Faiss1040ScalarQuantizedKnnVectorsFormat.FORMAT_NAME)
            || name.contains(HALF_FLOAT_FORMAT_NAME)
            || name.contains(HALF_FLOAT_SQ_FORMAT_NAME);
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
    private void decline(final AtomicLong counter, final DirectIOVectorIndexInput.DeclineReason reason) {
        counter.incrementAndGet();
        // The delegate's own class, not this one's: an operator reading the node-wide map wants the substrate
        // that declined, and every entry in it would otherwise read KNNDirectIODirectory.
        DirectIOVectorIndexInput.STATS.recordDecline(reason, in.getClass().getSimpleName());
    }

    /** Flat vector data opens this directory served with {@code O_DIRECT}, at either precision. */
    public long routedOpens() {
        return routedOpens.get();
    }

    /**
     * Opens that satisfied conditions 1 and 2 but were declined by one of 3 to 8 — the sum of the six
     * reason counters below. An operator watching only this number sees that routing is not happening; the
     * reasons say why, and they want different answers: a compound decline is every flushed segment and is
     * working as designed, a chain decline is a substrate the feature cannot serve, a size decline is
     * working as designed, a missing file is a bug, an open failure is a filesystem that does not
     * support {@code O_DIRECT}, and a content decline is a layer in the chain that transforms bytes.
     */
    public long declinedOpens() {
        return declinedCompound.get() + declinedChain.get() + declinedMissing.get() + declinedTooSmall.get() + declinedOpenFailed.get()
            + declinedContent.get();
    }

    /** Condition 3: the segment is compound, so the {@code .vec} is a {@code .cfs} entry and not a file. */
    public long declinedCompound() {
        return declinedCompound.get();
    }

    /** Condition 4: the delegate chain does not bottom out at an {@link FSDirectory}. */
    public long declinedChain() {
        return declinedChain.get();
    }

    /** Condition 5: the resolved path does not exist, or could not be sized. */
    public long declinedMissing() {
        return declinedMissing.get();
    }

    /** Condition 6: the file is below {@code knn.direct_io.min_file_size}. */
    public long declinedTooSmall() {
        return declinedTooSmall.get();
    }

    /**
     * Condition 7: one of the two opens failed — the {@code O_DIRECT} one, or the delegate's own input
     * without which condition 8 cannot be checked. Either way the delegate served the read.
     */
    public long declinedOpenFailed() {
        return declinedOpenFailed.get();
    }

    /**
     * Condition 8: the bytes the delegate chain serves are not the bytes in the file, so some layer of the
     * chain transforms them; or the comparison itself could not be made. The one decline reason that means
     * <b>Direct I/O must not serve this file</b> rather than that it cannot: see
     * {@link DirectIOVectorIndexInput#contentMatches}.
     */
    public long declinedContent() {
        return declinedContent.get();
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
     *   <li><b>Compound segment</b> — {@code state.directory} is a {@link CompoundDirectory}, so the open is
     *       <b>declined</b> and counted as {@link #declinedCompound}; see the class javadoc for why that is
     *       the measured right answer, and why it is most segments.</li>
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
        if (rescoreEnabled == false) {
            return state;
        }
        return new SegmentReadState(
            new KNNDirectIODirectory(state.directory, state.segmentInfo.name),
            state.segmentInfo,
            state.fieldInfos,
            state.context,
            state.segmentSuffix
        );
    }
}

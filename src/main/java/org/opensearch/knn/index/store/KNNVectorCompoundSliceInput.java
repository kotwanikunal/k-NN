/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.store.FilterIndexInput;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.BooleanSupplier;

/**
 * The dispatch point <em>inside</em> a compound file: an {@link IndexInput} over a {@code .cfs} whose
 * {@code slice} decides how the bytes of one entry are fetched.
 *
 * <p><b>Why this class exists.</b> A compound segment has no {@code .vec} {@code openInput} to dispatch
 * on: the {@code .vec} bytes live inside the {@code .cfs}, so a {@code Directory} below never sees the
 * name {@code .vec} and {@link org.opensearch.knn.index.codec.KNN80Codec.KNN80CompoundDirectory} is not
 * a {@code FilterDirectory} to walk. Since every freshly flushed segment is compound, that limit
 * applied to the newest data in every index.
 *
 * <p>The limit was an artefact of looking only at {@code Directory}. Lucene's compound reader opens the
 * {@code .cfs} on the <em>outer</em> directory — the plugin's — and then serves each entry by
 * <b>slicing that handle</b>:
 *
 * <pre>
 * Lucene90CompoundReader(Directory dir, SegmentInfo si)
 *     handle = dir.openInput("_0.cfs", IOContext.DEFAULT);          // :77  reaches the plugin directory
 * Lucene90CompoundReader#openInput(String name, IOContext context)
 *     return handle.slice(name, entry.offset, entry.length, context); // :171 four-argument slice
 * </pre>
 *
 * So the plugin's own {@code IndexInput} for the {@code .cfs} is handed, per entry, the <b>logical file
 * name</b> ({@code _0_NativeEngines990KnnVectorsFormat_0.vec}). Name dispatch is therefore available on
 * a compound segment after all — one level down from where it was looked for.
 *
 * <p><b>What this class must be, and must not be.</b> It is a {@link FilterIndexInput}, which is safe
 * for a container handle and is <em>not</em> safe for the {@code .vec} entry itself:
 * <ul>
 *   <li>A production {@code FilterIndexInput} subclass is <b>not</b> unwrapped by
 *       {@code FilterIndexInput.unwrapOnlyTest}, which the SIMD scorer calls first:
 *       {@code TEST_FILTER_INPUTS} is populated only through {@code TestSecrets}, whose setter
 *       Lucene restricts to its own test framework. So a {@code FilterIndexInput} around an mmap
 *       {@code .vec} would <b>displace</b> the memory-segment SIMD scorer — for traversal as well as
 *       re-score, because the check is a type test on the object and traversal and re-score share one
 *       object. That is why this class dispatches at {@code slice} granularity and returns the
 *       <em>raw</em> delegate slice for every entry it does not route: the entries it does not route
 *       keep their zero-copy, SIMD-bound, page-cached read path exactly.</li>
 *   <li>{@link FilterIndexInput} inherits {@code IndexInput}'s no-op {@link IndexInput#prefetch} and
 *       {@code DataInput}'s per-{@code float} {@link IndexInput#readFloats} loop. A naive wrapper would
 *       therefore silently disable the shipped prefetch path and make every bulk read 768 times more
 *       virtual calls. Both are overridden below; neither is optional.</li>
 * </ul>
 *
 * <p>It routes only a slice whose name is a faiss/MOS full-precision vector file — the same predicate
 * {@link KNNVectorStorageDirectory#isFaissVectorData} applies to a non-compound segment's
 * {@code openInput}. Everything else is the delegate's own slice, so installing this on a {@code .cfs}
 * costs one virtual call per {@code slice} — once per values object, not once per read — and nothing
 * else.
 */
@Log4j2
public final class KNNVectorCompoundSliceInput extends FilterIndexInput {

    /** One {@code slice} of the compound container, in the terms a dispatch rule would be written in. */
    public record SliceObservation(String name, long offset, long length, Set<IOContext.FileOpenHint> hints, boolean carriedContext,
        boolean routedToDirectIO) {
        @Override
        public String toString() {
            return "name="
                + name
                + " offset="
                + offset
                + " length="
                + length
                + " carriedContext="
                + carriedContext
                + " routedToDirectIO="
                + routedToDirectIO
                + " hints="
                + hints;
        }
    }

    /** The container this input reads, for logging and for the Direct I/O handle. */
    private final String containerName;

    /**
     * The {@code .cfs} on disk, or {@code null} when the delegate chain does not bottom out at an
     * {@code FSDirectory} — a remote-store or in-memory directory, where Direct I/O is not available at
     * all and this class can only observe.
     */
    private final Path containerPath;

    /** Shared with the directory that created this input, and with clones. */
    private final List<SliceObservation> observations;

    /**
     * Whether every slice is recorded in {@link #observations} and logged.
     *
     * <p>Off for the production {@link KNNVectorStorageDirectory}, and that is not a detail: the list is
     * unbounded and appending to a {@link CopyOnWriteArrayList} copies it, so a spike's
     * record-everything behaviour on a shard's real slice traffic would be a leak and an O(n²). The
     * counters below are kept either way, which is all a node needs to tell dispatch from silence.
     */
    private final boolean observing;

    /**
     * Whether a faiss/MOS {@code .vec} slice should be served with {@code O_DIRECT} — condition 2
     * of {@link KNNVectorStorageDirectory}'s dispatch rule. A supplier rather than a boolean because the
     * setting behind it is dynamic and a container outlives the query that opened it.
     */
    private final BooleanSupplier routeRescoreToDirectIO;

    /** Shared with clones, so a caller counts the container's traffic and not one clone's. */
    private final AtomicLong routedSlices;
    private final AtomicLong declinedSlices;

    /** Whether this instance owns {@link #directIO} and must close it. Clones do not. */
    private final boolean owner;

    /** Opened on the first routed slice, shared by every routed slice of this container. */
    private DirectIOVectorIndexInput directIO;

    /** Set once {@link DirectIOVectorIndexInput#open} has answered {@code null}, so it is asked once. */
    private boolean directIOUnavailable;

    /** The observing form, for tests: a fixed answer to condition 3, and every slice recorded. */
    KNNVectorCompoundSliceInput(
        final IndexInput delegate,
        final String containerName,
        final Path containerPath,
        final boolean routeRescoreToDirectIO
    ) {
        this(
            delegate,
            containerName,
            containerPath,
            () -> routeRescoreToDirectIO,
            new CopyOnWriteArrayList<>(),
            true,
            new AtomicLong(),
            new AtomicLong(),
            true
        );
    }

    /** The production form: condition 3 is asked live, and nothing is recorded per slice. */
    KNNVectorCompoundSliceInput(
        final IndexInput delegate,
        final String containerName,
        final Path containerPath,
        final BooleanSupplier routeRescoreToDirectIO
    ) {
        this(
            delegate,
            containerName,
            containerPath,
            routeRescoreToDirectIO,
            new CopyOnWriteArrayList<>(),
            false,
            new AtomicLong(),
            new AtomicLong(),
            true
        );
    }

    private KNNVectorCompoundSliceInput(
        final IndexInput delegate,
        final String containerName,
        final Path containerPath,
        final BooleanSupplier routeRescoreToDirectIO,
        final List<SliceObservation> observations,
        final boolean observing,
        final AtomicLong routedSlices,
        final AtomicLong declinedSlices,
        final boolean owner
    ) {
        super("KNNVectorCompoundSliceInput(" + containerName + ")", delegate);
        this.containerName = containerName;
        this.containerPath = containerPath;
        this.routeRescoreToDirectIO = routeRescoreToDirectIO;
        this.observations = observations;
        this.observing = observing;
        this.routedSlices = routedSlices;
        this.declinedSlices = declinedSlices;
        this.owner = owner;
    }

    /** Entries of this container served with {@code O_DIRECT}. Shared with clones. */
    public long routedSlices() {
        return routedSlices.get();
    }

    /** Entries that satisfied the name rule but could not be served with {@code O_DIRECT}. */
    public long declinedSlices() {
        return declinedSlices.get();
    }

    /** Every {@code slice} of this container seen so far, in call order. Shared with clones. */
    public List<SliceObservation> sliceObservations() {
        return List.copyOf(observations);
    }

    public void clearSliceObservations() {
        observations.clear();
    }

    // ---------------------------------------------------------------------------------------------------
    // The dispatch point
    // ---------------------------------------------------------------------------------------------------

    /**
     * The channel that carries the signal: {@code sliceDescription} is the logical file name. This is
     * where a compound segment gets the dispatch that {@code openInput} gives a non-compound one.
     */
    @Override
    public IndexInput slice(final String sliceDescription, final long offset, final long length, final IOContext context)
        throws IOException {
        final boolean route = KNNVectorStorageDirectory.isFaissVectorData(sliceDescription) && routeRescoreToDirectIO.getAsBoolean();
        final IndexInput routed = route ? directIOSlice(sliceDescription, offset, length) : null;
        if (route) {
            (routed != null ? routedSlices : declinedSlices).incrementAndGet();
            // The counters are reachable only from a test that holds this object. On a node the compound
            // dispatch point has to be observable too, and the non-compound one already is
            // (KNNVectorStorageDirectory.openInput), so this is the same statement for the other shape.
            log.debug(
                "k-NN vector storage: {} compound entry [{}] of [{}] with O_DIRECT, offset {} length {}",
                routed != null ? "serving" : "could not serve",
                sliceDescription,
                containerName,
                offset,
                length
            );
        }
        record(sliceDescription, offset, length, context.hints(), true, routed != null);
        return routed != null ? routed : in.slice(sliceDescription, offset, length, context);
    }

    /**
     * The overload with no {@link IOContext}. Not a dispatch point: Lucene's compound reader serves every
     * entry of a container through the four-argument overload above, so a three-argument slice is always
     * a sub-slice of an entry already dispatched, and its description is a caller's label rather than a
     * file name. Recorded so that the distinction is visible in the evidence rather than argued.
     */
    @Override
    public IndexInput slice(final String sliceDescription, final long offset, final long length) throws IOException {
        record(sliceDescription, offset, length, Set.of(), false, false);
        return in.slice(sliceDescription, offset, length);
    }

    private void record(
        final String name,
        final long offset,
        final long length,
        final Set<IOContext.FileOpenHint> hints,
        final boolean carriedContext,
        final boolean routedToDirectIO
    ) {
        if (observing == false) {
            return;
        }
        final SliceObservation observation = new SliceObservation(name, offset, length, hints, carriedContext, routedToDirectIO);
        observations.add(observation);
        if (isVectorDataEntry(name)) {
            log.info("k-NN compound slice probe [{}] saw slice {}", containerName, observation);
        } else {
            log.debug("k-NN compound slice probe [{}] saw slice {}", containerName, observation);
        }
    }

    /**
     * A view of the entry over an {@code O_DIRECT} handle on the container, or {@code null} when Direct
     * I/O cannot serve it. Returning {@code null} rather than throwing keeps
     * {@link DirectIOVectorSource#open}'s fallback contract: a file this cannot serve costs a fall back
     * to the delegate's slice, never a failed read.
     *
     * <p>The container is opened once and every routed entry slices it, because the {@code .cfs} is one
     * file and {@link DirectIOVectorIndexInput#slice} inherits the handle rather than reopening it.
     */
    private IndexInput directIOSlice(final String name, final long offset, final long length) {
        if (containerPath == null || directIOUnavailable) {
            return null;
        }
        try {
            if (directIO == null) {
                directIO = DirectIOVectorIndexInput.open(containerPath);
                if (directIO == null) {
                    directIOUnavailable = true;
                    log.warn("k-NN compound slice probe [{}]: Direct I/O is not available for [{}]", containerName, containerPath);
                    return null;
                }
            }
            return directIO.slice(name, offset, length);
        } catch (IOException | RuntimeException e) {
            log.warn("k-NN compound slice probe [{}]: could not route [{}] to Direct I/O", containerName, name, e);
            return null;
        }
    }

    /** Whether {@code name} is one of the entries a dispatch rule would ever look at. */
    static boolean isVectorDataEntry(final String name) {
        return name != null && (name.endsWith(".vec") || name.endsWith(".veq") || name.endsWith(".faiss"));
    }

    // ---------------------------------------------------------------------------------------------------
    // Everything FilterIndexInput does not delegate, and must
    // ---------------------------------------------------------------------------------------------------

    /**
     * {@link FilterIndexInput} does not override this, so it would inherit {@link IndexInput}'s no-op and
     * silently disable the shipped prefetch path
     * ({@code PrefetchableFlatVectorScorer} → {@code PrefetchHelper} → {@code IndexInput.prefetch}) for
     * every entry of every compound segment. The single most damaging thing a naive wrapper could do,
     * and invisible: a dropped prefetch changes no bytes.
     */
    @Override
    public void prefetch(final long offset, final long length) throws IOException {
        in.prefetch(offset, length);
    }

    @Override
    public void updateIOContext(final IOContext context) throws IOException {
        in.updateIOContext(context);
    }

    @Override
    public java.util.Optional<Boolean> isLoaded() {
        return in.isLoaded();
    }

    /**
     * {@link FilterIndexInput} delegates only {@code readByte} and {@code readBytes}, so every other read
     * would fall back to {@code DataInput}'s implementation in terms of those — {@code readFloats} alone
     * becomes {@code length} calls to {@code readInt}, each four calls to {@code readByte}. Delegating the
     * bulk and fixed-width reads is what makes the wrapper cost one virtual call rather than thousands.
     */
    @Override
    public void readFloats(final float[] destination, final int offset, final int length) throws IOException {
        in.readFloats(destination, offset, length);
    }

    @Override
    public void readInts(final int[] destination, final int offset, final int length) throws IOException {
        in.readInts(destination, offset, length);
    }

    @Override
    public void readLongs(final long[] destination, final int offset, final int length) throws IOException {
        in.readLongs(destination, offset, length);
    }

    @Override
    public short readShort() throws IOException {
        return in.readShort();
    }

    @Override
    public int readInt() throws IOException {
        return in.readInt();
    }

    @Override
    public long readLong() throws IOException {
        return in.readLong();
    }

    @Override
    public int readVInt() throws IOException {
        return in.readVInt();
    }

    @Override
    public long readVLong() throws IOException {
        return in.readVLong();
    }

    @Override
    public void skipBytes(final long numBytes) throws IOException {
        in.skipBytes(numBytes);
    }

    /**
     * A clone must not be a shallow copy of this object. {@link IndexInput#clone} is
     * {@code Object.clone}, which would leave the copy sharing this object's delegate — so the copy's
     * {@code seek} would move this object's file pointer. {@code CodecUtil.checksumEntireFile} clones a
     * {@code .cfs} handle and seeks it to zero, so this is reached on a real segment open, not in theory.
     */
    @Override
    public KNNVectorCompoundSliceInput clone() {
        return new KNNVectorCompoundSliceInput(
            in.clone(),
            containerName,
            containerPath,
            routeRescoreToDirectIO,
            observations,
            observing,
            routedSlices,
            declinedSlices,
            false
        );
    }

    @Override
    public void close() throws IOException {
        try {
            if (owner && directIO != null) {
                directIO.close();
                directIO = null;
            }
        } finally {
            in.close();
        }
    }
}

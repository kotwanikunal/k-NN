/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.FilterDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.opensearch.common.Nullable;
import org.opensearch.knn.index.KNNSettings;

import java.io.IOException;
import java.nio.file.Path;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.BooleanSupplier;

/**
 * The storage layer's half of the directory design: the {@link Directory} that decides <em>how</em> a
 * vector data read is served, for reads whose <em>what</em> has already been named by
 * {@link KNNVectorReadIntent}.
 *
 * <h2>The dispatch rule, in one place</h2>
 * A read is served with {@code O_DIRECT} if and only if all four of these hold:
 *
 * <ol>
 *   <li>the file is the full-precision flat vector data — {@code name.endsWith(".vec")};
 *   <li>the caller's {@link IOContext} carries {@link KNNVectorReadIntent#RESCORE};
 *   <li>{@link KNNSettings#isDirectIORescoreEnabled()} is on — the per-substrate enablement rule, which
 *       stays a setting because Direct I/O is a win only where the {@code .vec} working set cannot stay
 *       in the page cache;
 *   <li>the delegate chain bottoms out at an {@link FSDirectory}, so there is a {@link Path} to open.
 * </ol>
 *
 * Everything else — every other file, and every other read of {@code .vec} — is the delegate's own
 * {@link IndexInput}, returned <b>unwrapped</b>. Neither condition 1 nor condition 2 is sufficient
 * alone: by name alone this would route the traversal, merge, warmup and derived-source reads of the
 * same file, and by intent alone it would route the quantized codes and the native index. The
 * conjunction is the rule.
 *
 * <h2>Two dispatch points, because a segment has two shapes</h2>
 * <table>
 *   <caption>where the name and the intent arrive together</caption>
 *   <tr><th>segment</th><th>channel</th></tr>
 *   <tr>
 *     <td>non-compound (a merged segment)</td>
 *     <td>{@code openInput(name, context)} here — Lucene's flat vectors reader opens {@code .vec}
 *         directly on the directory.</td>
 *   </tr>
 *   <tr>
 *     <td>compound (<b>every freshly flushed segment</b>)</td>
 *     <td>{@code IndexInput.slice(name, offset, length, context)} — the four-argument overload — on the
 *         {@code .cfs} container, which Lucene opens <em>here</em> and then slices per entry. That is
 *         {@link KNNVectorCompoundSliceInput}, and it is why every {@code .cfs} open is wrapped.</td>
 *   </tr>
 * </table>
 *
 * Nothing else carries both halves of the signal: the three-argument {@code slice} has no
 * {@link IOContext} parameter at all, and {@code updateIOContext} can only change access advice, for the
 * whole mapping.
 *
 * <h2>Dispatch granularity is the open, never the file</h2>
 * The decision is taken once per {@code openInput}/{@code slice} and never re-taken per read, and that
 * is forced rather than chosen. Lucene's SIMD float scorer binds by a type test on the
 * {@code IndexInput} and then reads the memory segment directly, so it never calls {@code readBytes} —
 * an input transparent enough to keep SIMD for traversal is transparent enough that a re-score read
 * would bypass the Direct I/O staging entirely, and an input opaque enough to route a re-score read
 * also declines SIMD. One {@code .vec} object cannot serve both. It does not need to: the re-score path
 * gets its <em>own</em> object, opened by its own reader
 * ({@link org.opensearch.knn.index.codec.KNNRescoreVectorsReader}), and the traversal object is
 * untouched by construction.
 *
 * <h2>Failure is a fallback</h2>
 * Every way Direct I/O can be unavailable — no {@code ExtendedOpenOption.DIRECT} on this JDK, a
 * filesystem that answers {@code EINVAL}, a remote-store or in-memory directory with no path — ends in
 * the delegate's own input and a log line. There is no configuration in which a failure here changes a
 * result, because both routes read the same bytes of the same file.
 *
 * <h2>What this class deliberately does not have</h2>
 * Its predecessor {@link KNNVectorIntentProbeDirectory} is a spike: it records every {@code openInput}
 * in an unbounded list and sets a node-wide static flag so a test can tell "never installed" from
 * "installed but unreachable". Neither belongs on a production read path — the list grows for the life
 * of the shard and the static outlives every index that set it. This class keeps bounded counters and
 * no static state at all; {@link #find(Directory)} is a walk of the caller's own chain, not a registry.
 */
@Log4j2
public final class KNNVectorStorageDirectory extends FilterDirectory {

    /** Lucene's flat full-precision vector data — the only file this routes. */
    static final String VECTOR_DATA_EXTENSION = ".vec";

    /** The container a compound segment's files are served from. */
    static final String COMPOUND_CONTAINER_EXTENSION = ".cfs";

    /** The index this directory belongs to, so a log line can be attributed to one. */
    private final String indexName;

    /**
     * Condition 3 of the dispatch rule, read at every dispatch rather than captured once, because the
     * setting is dynamic and a non-compound segment's {@code .vec} is opened once per reader while a
     * compound segment's entries are sliced once per values object.
     */
    private final BooleanSupplier directIORescoreEnabled;

    private final AtomicLong routedOpens = new AtomicLong();
    private final AtomicLong declinedOpens = new AtomicLong();
    private final AtomicLong wrappedContainers = new AtomicLong();

    public KNNVectorStorageDirectory(final Directory delegate, final String indexName) {
        this(delegate, indexName, KNNSettings::isDirectIORescoreEnabled);
    }

    /** Visible for testing, so a test can drive the rule without a cluster service. */
    KNNVectorStorageDirectory(final Directory delegate, final String indexName, final BooleanSupplier directIORescoreEnabled) {
        super(delegate);
        this.indexName = indexName;
        this.directIORescoreEnabled = directIORescoreEnabled;
    }

    public String indexName() {
        return indexName;
    }

    // -------------------------------------------------------------------------------------------------
    // Dispatch point 1: the open, which is how a non-compound segment's .vec arrives
    // -------------------------------------------------------------------------------------------------

    @Override
    public IndexInput openInput(final String name, final IOContext context) throws IOException {
        if (routesToDirectIO(name, context)) {
            final IndexInput direct = directIOInput(name);
            if (direct != null) {
                routedOpens.incrementAndGet();
                log.debug("k-NN vector storage [{}]: serving [{}] with O_DIRECT", indexName, name);
                return direct;
            }
            declinedOpens.incrementAndGet();
        }
        final IndexInput input = in.openInput(name, context);
        if (isCompoundContainer(name)) {
            // A compound segment's .vec has no openInput of its own for the intent to ride on, but every
            // entry of the container is a four-argument slice of THIS input, and that slice carries both
            // the logical name and the caller's context. So the dispatch point for a compound segment is
            // one level down, inside the container.
            wrappedContainers.incrementAndGet();
            return new KNNVectorCompoundSliceInput(input, name, resolvePath(name), directIORescoreEnabled);
        }
        return input;
    }

    /**
     * Conditions 1–3 of the dispatch rule. Condition 4 is answered by {@link #directIOInput}, because
     * "there is no path" and "the path cannot be opened with {@code O_DIRECT}" have the same consequence
     * and are better answered in one place.
     *
     * <p>Condition 3 is belt-and-braces rather than the load-bearing gate: the only thing in the plugin
     * that attaches {@link KNNVectorReadIntent#RESCORE} is {@link KNNRescoreIntentDirectory}, installed
     * only by a rescore view that already checked the setting. Keeping the check here means the whole
     * rule can be read in one method, and means an intent arriving from anywhere else still respects the
     * operator's switch.
     */
    private boolean routesToDirectIO(final String name, final IOContext context) {
        return isFullPrecisionVectorData(name)
            && KNNVectorReadIntent.of(context) == KNNVectorReadIntent.RESCORE
            && directIORescoreEnabled.getAsBoolean();
    }

    /**
     * An {@code O_DIRECT} input over {@code name}, or {@code null} when Direct I/O cannot serve it. The
     * returned input owns its own file handle and is closed by whoever closes the input, which is the
     * same ownership {@link Directory#openInput} always has.
     */
    @Nullable
    private IndexInput directIOInput(final String name) {
        final Path path = resolvePath(name);
        if (path == null) {
            log.debug("k-NN vector storage [{}]: [{}] has no filesystem path, so Direct I/O cannot serve it", indexName, name);
            return null;
        }
        try {
            final DirectIOVectorIndexInput direct = DirectIOVectorIndexInput.open(path);
            if (direct == null) {
                log.warn("k-NN vector storage [{}]: Direct I/O is not available for [{}]", indexName, path);
            }
            return direct;
        } catch (RuntimeException e) {
            log.warn("k-NN vector storage [{}]: could not open [{}] with Direct I/O", indexName, path, e);
            return null;
        }
    }

    /**
     * The file behind {@code name} when the delegate chain bottoms out at an {@link FSDirectory}, and
     * {@code null} otherwise — a remote-store or in-memory directory has no path, and the answer there is
     * to decline rather than to fail.
     */
    @Nullable
    private Path resolvePath(final String name) {
        final Directory bottom = FilterDirectory.unwrap(in);
        if (bottom instanceof FSDirectory fsDirectory) {
            return fsDirectory.getDirectory().resolve(name);
        }
        return null;
    }

    static boolean isFullPrecisionVectorData(final String name) {
        return name != null && name.endsWith(VECTOR_DATA_EXTENSION);
    }

    static boolean isCompoundContainer(final String name) {
        return name != null && name.endsWith(COMPOUND_CONTAINER_EXTENSION);
    }

    // -------------------------------------------------------------------------------------------------
    // Observability: bounded counters, no node-wide state
    // -------------------------------------------------------------------------------------------------

    /** Full-precision vector data opens this directory served with {@code O_DIRECT}. */
    public long routedOpens() {
        return routedOpens.get();
    }

    /** Opens that satisfied the intent rule but could not be served with {@code O_DIRECT}. */
    public long declinedOpens() {
        return declinedOpens.get();
    }

    /** Compound containers wrapped so that their entries can be dispatched. */
    public long wrappedContainers() {
        return wrappedContainers.get();
    }

    /**
     * The {@link KNNVectorStorageDirectory} in {@code directory}'s wrapper chain, or {@code null} when
     * there is none — which is every index that has not opted in with {@code index.store.factory}.
     * A pure walk of the chain the caller already holds: this class keeps no registry.
     */
    @Nullable
    public static KNNVectorStorageDirectory find(final Directory directory) {
        Directory current = directory;
        // Bounded because a probe must never be the thing that hangs a shard.
        for (int depth = 0; current != null && depth < 16; depth++) {
            if (current instanceof KNNVectorStorageDirectory storage) {
                return storage;
            }
            if (current instanceof FilterDirectory filterDirectory) {
                current = filterDirectory.getDelegate();
            } else {
                return null;
            }
        }
        return null;
    }
}

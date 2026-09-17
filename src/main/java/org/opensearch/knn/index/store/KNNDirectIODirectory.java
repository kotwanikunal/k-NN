/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.misc.store.DirectIODirectory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;

import java.io.IOException;
import java.nio.file.OpenOption;
import java.util.Arrays;
import java.util.OptionalLong;

/**
 * A {@link DirectIODirectory} whose routing gate is narrowed to the one file the k-NN rescore path
 * reads in bulk: the flat full-precision vector file, {@code .vec}.
 * <p>
 * Lucene's default gate is merge-only — {@code context() == MERGE && estimatedMergeBytes >=
 * minBytesDirect} — so wrapping a directory in the unmodified {@link DirectIODirectory} changes
 * nothing for search, because a query's {@link IOContext} is never
 * {@link IOContext.Context#MERGE}. Overriding {@link #useDirectIO} is the extension point that
 * makes it apply to reads, and it is the only behaviour this class changes; everything else,
 * including the delegation of every non-{@code .vec} file back to the wrapped directory, is
 * inherited from {@link DirectIODirectory} and {@code FilterDirectory}.
 * <p>
 * The gate is an explicit include list rather than an exclude list, so any file type we have not
 * reasoned about keeps its current behaviour by default.
 * <p>
 * <b>Fallback.</b> Direct I/O can be refused at three different moments, and only the first two are
 * recoverable:
 * <ol>
 *   <li>The JDK may not expose {@code com.sun.nio.file.ExtendedOpenOption.DIRECT} at all. That is a
 *       property of the runtime, is knowable before any file is opened, and is what
 *       {@link #isDirectIOOpenOptionAvailable()} reports so the factory can decline to wrap.</li>
 *   <li>The filesystem may reject the open — tmpfs and several network filesystems answer
 *       {@code O_DIRECT} with {@code EINVAL}. This surfaces per file at {@link #openInput}, so that
 *       call falls back to the delegate, logs once, and latches Direct I/O off for the rest of this
 *       directory's life rather than paying a failed open per file.</li>
 *   <li>A misaligned read throws from Lucene's {@code refill} mid-stream, after the
 *       {@link IndexInput} has been handed to the query. There is no way to resume from that, so
 *       alignment is not defended at run time — it is correct by construction, because the buffer
 *       {@link DirectIOBufferSizer} computes is always a whole number of filesystem blocks and
 *       Lucene's own input does every seek and refill in block units. Do not add a per read guard
 *       here expecting it to help.</li>
 * </ol>
 * The consequence of (1) and (2) together is that a query is never failed because Direct I/O is
 * unavailable; the worst case is that it runs exactly as it does today.
 */
@Log4j2
public final class KNNDirectIODirectory extends DirectIODirectory {

    /**
     * The suffix of Lucene's flat vector data file. This is the file
     * {@code OffHeapFloatVectorValues} reads one vector at a time during full-precision rescoring,
     * and the only file this directory serves with Direct I/O.
     * <p>
     * Matched with a plain suffix test rather than through
     * {@link org.apache.lucene.store.FileSwitchDirectory#getExtension}, because that method
     * deliberately looks *through* a trailing {@code .tmp} and reports the inner extension — so
     * {@code _0_NativeEngines990.vec.tmp} comes back as {@code vec} there. A temporary file is a
     * write target, and Direct I/O must not follow it.
     */
    static final String VECTOR_DATA_SUFFIX = ".vec";

    /**
     * {@link DirectIODirectory} keeps its own copy of this privately, so we hold a second one for
     * the gate.
     */
    private final long minBytesDirect;

    /**
     * Also kept privately by {@link DirectIODirectory}, and held here only so that a test can assert
     * the size {@link DirectIOBufferSizer} chose for this index.
     */
    private final int readBufferSize;

    /**
     * Latched to true by the first {@link #openInput} that could not open the file with Direct I/O.
     * Once set, every later open goes straight to the delegate: the reason a Direct I/O open fails is
     * almost always a property of the filesystem, so retrying it per file would cost a failed
     * {@code open} syscall each time and produce the same answer.
     * <p>
     * Volatile rather than synchronized because a benign race — two concurrent opens both attempting
     * Direct I/O once — is cheaper than serialising every open of every file in the shard.
     */
    private volatile boolean directIOUnavailable = false;

    /**
     * @param delegate      the directory that serves every file this one does not route, and the
     *                      reference for the filesystem path. Must be an {@link FSDirectory}; the
     *                      superclass casts it.
     * @param readBufferSize size of the per-{@code IndexInput} read buffer, in bytes. Named
     *                      {@code mergeBufferSize} upstream because upstream only uses it for
     *                      merges; on the read path it is the granularity of every refill, so it
     *                      is the single most important tuning knob. Callers supply it; a later
     *                      change derives it per index from the mapping's vector dimensions.
     * @param minBytesDirect files shorter than this are served by the delegate. Small segments are
     *                      likely fully page-resident already, so bypassing the page cache for
     *                      them costs syscalls and buys no cache hygiene.
     * @throws IOException if the superclass cannot read the filesystem block size
     */
    public KNNDirectIODirectory(final FSDirectory delegate, final int readBufferSize, final long minBytesDirect) throws IOException {
        super(delegate, readBufferSize, minBytesDirect);
        this.minBytesDirect = minBytesDirect;
        this.readBufferSize = readBufferSize;
    }

    /** The read buffer size this directory was built with, in bytes. */
    int getReadBufferSize() {
        return readBufferSize;
    }

    /** True once a Direct I/O open has failed and this directory has latched back to the delegate. */
    boolean isDirectIOUnavailable() {
        return directIOUnavailable;
    }

    /**
     * Opens the file, falling back to the delegate if Direct I/O is refused for it.
     * <p>
     * The superclass already routes correctly; what it does not do is survive a refusal. A
     * filesystem that does not implement {@code O_DIRECT} fails the {@code FileChannel.open} inside
     * Lucene's {@code DirectIOIndexInput} constructor with {@code EINVAL}, and a JDK without
     * {@code ExtendedOpenOption.DIRECT} throws {@link UnsupportedOperationException} from the same
     * place. Either would otherwise propagate out of a query.
     * <p>
     * The gate is evaluated here rather than left to the superclass so that the {@code catch} only
     * covers an open we actually attempted with Direct I/O. Catching around an unconditional
     * {@code super.openInput} would also swallow the delegate's own failures — a missing file, say —
     * and retry them, turning one clear exception into two confusing ones.
     * <p>
     * {@code Error} is deliberately not caught. A {@code MaxDirectMemorySize} exhaustion arrives as
     * {@link OutOfMemoryError}, and continuing on the delegate while the JVM is out of direct memory
     * would hide a misconfiguration that the operator has to fix.
     *
     * @param name    the file to open
     * @param context the open context
     * @return a Direct I/O input when the gate selects this file and the open succeeds, otherwise
     *         exactly the input the delegate would have returned
     * @throws IOException if the delegate cannot open the file either
     */
    @Override
    public IndexInput openInput(final String name, final IOContext context) throws IOException {
        ensureOpen();
        if (directIOUnavailable == false && shouldUseDirectIO(name, context, OptionalLong.of(fileLength(name)), minBytesDirect)) {
            try {
                return super.openInput(name, context);
            } catch (IOException | RuntimeException e) {
                directIOUnavailable = true;
                log.warn(
                    "Direct I/O is not usable under [{}] and will not be attempted again for this shard; "
                        + "falling back to [{}] for [{}] and every later file. Read buffer was {} bytes.",
                    getDirectory(),
                    getDelegate().getClass().getSimpleName(),
                    name,
                    readBufferSize,
                    e
                );
            }
        }
        return getDelegate().openInput(name, context);
    }

    /**
     * Whether this JDK exposes {@code com.sun.nio.file.ExtendedOpenOption.DIRECT}, which is what
     * Lucene opens Direct I/O files with.
     * <p>
     * Looked up reflectively for the same two reasons Lucene's own
     * {@code DirectIODirectory.ExtendedOpenOption_DIRECT} does — it is a proprietary OpenJDK API that
     * emits an unsuppressible warning when referenced under {@code --release}, and it is absent on
     * runtimes that do not implement it, so a direct reference would not link. Lucene's copy of this
     * lookup is package private and has no accessor, hence the second one.
     * <p>
     * Called once per shard open, so the reflection cost is irrelevant.
     *
     * @return true if Direct I/O is at least theoretically available on this runtime
     */
    static boolean isDirectIOOpenOptionAvailable() {
        try {
            final Class<? extends OpenOption> clazz = Class.forName("com.sun.nio.file.ExtendedOpenOption").asSubclass(OpenOption.class);
            return Arrays.stream(clazz.getEnumConstants()).anyMatch(option -> option.toString().equalsIgnoreCase("DIRECT"));
        } catch (ClassNotFoundException | RuntimeException e) {
            return false;
        }
    }

    /**
     * Routes only {@code .vec} reads above the size floor through Direct I/O.
     * <p>
     * The conditions are ordered deliberately and each is load bearing:
     * <ol>
     *   <li>An empty {@code fileLength} means the caller wants an {@link org.apache.lucene.store.IndexOutput},
     *       not an input. Returning {@code true} there would hand back Lucene's
     *       {@code DirectIOIndexOutput}, which opens with {@code CREATE_NEW} and so fails on any
     *       rewrite, and would make us write vectors through Direct I/O — neither of which this
     *       change is about. Writes always go to the delegate.</li>
     *   <li>Merges keep the delegate too. Merge reads of {@code .vec} are sequential and benefit
     *       from the page cache and read-ahead that Direct I/O throws away, and Lucene already
     *       offers its own merge-only gate for operators who want the opposite. Note this
     *       deliberately inverts the superclass' default rather than extending it.</li>
     *   <li>The include list. Anything that is not {@code .vec} — the FAISS index file, the doc-id
     *       map, {@code segments_N}, temporary files whose final extension is {@code .tmp} — is
     *       untouched.</li>
     *   <li>The size floor.</li>
     * </ol>
     * Unlike the superclass' implementation this never calls {@code context.mergeInfo()}, so it
     * cannot NPE on a non-merge context; do not "restore" that call.
     * <p>
     * This deliberately does <em>not</em> consult {@code directIOUnavailable}. {@link #openInput}
     * checks the latch itself and bypasses the superclass entirely when it is set, and the
     * superclass calls this method again on the way to building the input — so making it return false
     * once the latch is set would mean the one open we do attempt can never succeed.
     *
     * @param name       the file name, without any directory component
     * @param context    the open context
     * @param fileLength the file length when known, empty when an output is being requested
     * @return true if this file should be served with Direct I/O
     */
    @Override
    protected boolean useDirectIO(final String name, final IOContext context, final OptionalLong fileLength) {
        return shouldUseDirectIO(name, context, fileLength, minBytesDirect);
    }

    /**
     * The gate as a pure function, so it can be tested without a filesystem. See
     * {@link #useDirectIO} for why each condition is there.
     */
    static boolean shouldUseDirectIO(final String name, final IOContext context, final OptionalLong fileLength, final long minBytesDirect) {
        if (fileLength.isEmpty()) {
            return false;
        }
        if (context.context() == IOContext.Context.MERGE) {
            return false;
        }
        if (name.endsWith(VECTOR_DATA_SUFFIX) == false) {
            return false;
        }
        return fileLength.getAsLong() >= minBytesDirect;
    }
}

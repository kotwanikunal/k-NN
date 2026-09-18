/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.index.FloatVectorValues;
import org.opensearch.knn.index.KNNSettings;

import java.io.Closeable;
import java.io.IOException;
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
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * A Direct I/O byte source for one field's full-precision vectors in a {@code .vec} file: given an
 * ordinal, it returns that vector's floats, read with {@code O_DIRECT} so the bytes never enter the
 * page cache.
 *
 * <p>This is the loader half of the rescore seam. It is deliberately <em>not</em> an
 * {@link org.apache.lucene.store.IndexInput}: Lucene's own {@code DirectIOIndexInput} is a sequential
 * stream with a single internal buffer, and this path wants independent aligned buffers per query thread
 * over a shared file handle. The handle is opened once per segment and shared; the buffers are owned by a
 * {@link Reader}, one reader per scorer.
 *
 * <p>A {@link Reader} can also be told what it is about to be asked for, via {@link Reader#stage}, and
 * will then keep a rolling window of those reads in flight on {@link DirectIOReadPool}. That read-ahead is
 * what replaces the {@code madvise} prefetch the mmap path got for free and {@code O_DIRECT} necessarily
 * removes; without it a rescore query pays one serial device round trip per candidate.
 *
 * <h2>Why the file region has to be discovered rather than asked for</h2>
 * The seam holds Lucene's mmap-backed slice of the vector region, and neither {@code IndexInput} nor
 * {@code FloatVectorValues} exposes the file that slice came from or where in it the region starts.
 * The file <em>name</em> is recoverable from the codec's {@code SegmentReadState} (see
 * {@code Faiss1040ScalarQuantizedFlatVectorsReader}), but the region's start offset is written only into
 * the {@code .vem} metadata and kept in a private field of Lucene's reader.
 *
 * <p>So the start offset is derived instead: a flat vector file written by
 * {@code Lucene99FlatVectorsFormat} is a codec header, then each field's vector region, then a
 * {@code CodecUtil} footer, and Lucene's per-field vectors formats give each format instance its own
 * file, so in practice there is exactly one region and it ends where the footer begins:
 *
 * <pre>
 *   baseOffset = fileLength - footerLength - size * vectorByteLength
 * </pre>
 *
 * That derivation is an inference about a file layout this code does not own, so it is
 * <b>verified before use</b>: {@link #open} reads the first and last ordinals through Direct I/O and
 * compares them, bit for bit, with what the mmap values return for the same ordinals. A file with more
 * than one region, a different codec layout, or a compound segment fails that comparison (or never
 * gets a path at all) and the caller falls back to the default mmap path. The consequence is that a
 * wrong derivation costs a fallback, never a wrong score.
 *
 * <h2>Fallback</h2>
 * Every failure mode returns {@code null} from {@link #open} and is logged once: no
 * {@code ExtendedOpenOption.DIRECT} on this JDK, a filesystem that answers {@code O_DIRECT} with
 * {@code EINVAL}, a path that does not exist, a derived offset that is negative or does not verify, or
 * a vector large enough that a single-read buffer would exceed
 * {@code knn.direct_io.max_buffer_size}. {@link Error} — notably direct-buffer {@link OutOfMemoryError}
 * — is not caught, for the same reason {@link KNNDirectIODirectory} does not catch it: continuing on the
 * mmap path while the JVM is out of direct memory hides a misconfiguration the operator has to fix.
 */
@Log4j2
public final class DirectIOVectorSource implements Closeable {

    /** Length of a {@code CodecUtil} footer: magic, algorithm id and a long checksum. */
    static final int FOOTER_LENGTH = 16;

    private final Path path;
    private final FileChannel channel;
    private final long baseOffset;
    private final int size;
    private final int dimension;
    private final int vectorByteLength;
    private final int blockSize;
    private final int bufferSize;

    /**
     * Whether a vector always starts on a four-byte boundary inside its read buffer, which is what lets
     * {@link Reader#decode} copy the whole vector at once instead of a float at a time.
     *
     * <p>True for every layout this seam actually meets — {@code vectorByteLength} is {@code dimension}
     * floats and a block size is a power of two, so the only way the offset within the block can be
     * unaligned is a base offset that is not a multiple of four, which no {@code CodecUtil} header
     * produces. It is still computed rather than assumed, because the base offset is derived (see the
     * class javadoc) and a misaligned one must cost a slower decode, not a wrong read.
     */
    private final boolean bulkDecodable;

    private DirectIOVectorSource(
        final Path path,
        final FileChannel channel,
        final long baseOffset,
        final int size,
        final int dimension,
        final int vectorByteLength,
        final int blockSize,
        final int bufferSize
    ) {
        this.path = path;
        this.channel = channel;
        this.baseOffset = baseOffset;
        this.size = size;
        this.dimension = dimension;
        this.vectorByteLength = vectorByteLength;
        this.blockSize = blockSize;
        this.bufferSize = bufferSize;
        this.bulkDecodable = baseOffset % Float.BYTES == 0 && vectorByteLength % Float.BYTES == 0;
    }

    /**
     * Opens {@code path} with Direct I/O and returns a source for the vector region it is expected to
     * hold, or {@code null} when Direct I/O cannot serve those vectors for any reason.
     *
     * <p>The returned source has been verified against {@code reference}: the first and last ordinals
     * read identically through both. {@code reference} is read twice and left positioned wherever those
     * reads leave it, so callers should hand over values they are not otherwise using.
     *
     * @param path      the {@code .vec} file
     * @param reference the mmap-backed values for the same field, used only to verify the derived region
     * @return a verified source, or {@code null} to mean "fall back to the default path"
     */
    public static DirectIOVectorSource open(final Path path, final FloatVectorValues reference) {
        if (path == null || reference == null) {
            return null;
        }
        final int size = reference.size();
        final int dimension = reference.dimension();
        final int vectorByteLength = reference.getVectorByteLength();
        if (size <= 0 || dimension <= 0 || vectorByteLength <= 0) {
            return null;
        }

        final OpenOption directOption = directOpenOption();
        if (directOption == null) {
            log.warn("Direct I/O rescore is enabled but this JDK does not expose ExtendedOpenOption.DIRECT; using the default path");
            return null;
        }

        FileChannel channel = null;
        try {
            if (Files.isRegularFile(path) == false) {
                log.warn("Direct I/O rescore cannot use [{}]: not a regular file. Using the default path.", path);
                return null;
            }
            final long fileLength = Files.size(path);
            final long baseOffset = fileLength - FOOTER_LENGTH - (long) size * vectorByteLength;
            if (baseOffset < 0) {
                log.warn(
                    "Direct I/O rescore cannot use [{}]: {} bytes is too short for {} vectors of {} bytes. Using the default path.",
                    path,
                    fileLength,
                    size,
                    vectorByteLength
                );
                return null;
            }

            final int blockSize = Math.toIntExact(Files.getFileStore(path).getBlockSize());
            if (blockSize <= 0) {
                log.warn("Direct I/O rescore cannot use [{}]: the filesystem reports block size {}.", path, blockSize);
                return null;
            }
            final int bufferSize = DirectIOBufferSizer.requiredBufferSize(vectorByteLength, blockSize, baseOffset);
            final long maxBufferSize = maxBufferSize();
            if (bufferSize > maxBufferSize) {
                log.warn(
                    "Direct I/O rescore declines [{}]: serving a {} byte vector in one read needs a {} byte buffer, "
                        + "above knn.direct_io.max_buffer_size ({} bytes). Using the default path.",
                    path,
                    vectorByteLength,
                    bufferSize,
                    maxBufferSize
                );
                return null;
            }

            channel = FileChannel.open(path, StandardOpenOption.READ, directOption);
            final DirectIOVectorSource source = new DirectIOVectorSource(
                path,
                channel,
                baseOffset,
                size,
                dimension,
                vectorByteLength,
                blockSize,
                bufferSize
            );
            if (source.verifyAgainst(reference) == false) {
                channel.close();
                return null;
            }
            log.info(
                "Direct I/O rescore is serving [{}]: {} vectors of {} bytes at offset {}, {} byte reads on a {} byte block",
                path,
                size,
                vectorByteLength,
                baseOffset,
                bufferSize,
                blockSize
            );
            return source;
        } catch (IOException | RuntimeException e) {
            closeQuietly(channel);
            log.warn("Direct I/O rescore could not open [{}]; using the default path", path, e);
            return null;
        }
    }

    /**
     * Reads the first and last ordinals through Direct I/O and compares them with {@code reference}.
     *
     * <p>Both ends are checked rather than only the first, because a base offset that is wrong by a whole
     * number of vectors — which is what a second region in the same file looks like — still lines up
     * somewhere. A file whose region is shifted at all fails the first ordinal; a file whose region is
     * the wrong length fails the last.
     */
    private boolean verifyAgainst(final FloatVectorValues reference) throws IOException {
        final Reader reader = newReader();
        for (final int ord : size == 1 ? new int[] { 0 } : new int[] { 0, size - 1 }) {
            final float[] expected = reference.vectorValue(ord).clone();
            final float[] actual = reader.read(ord);
            if (Arrays.equals(expected, actual) == false) {
                log.warn(
                    "Direct I/O rescore rejected [{}]: ordinal {} read at offset {} does not match the mmap values, "
                        + "so the vector region was not where it was derived to be. Using the default path.",
                    path,
                    ord,
                    baseOffset + (long) ord * vectorByteLength
                );
                return false;
            }
        }
        return true;
    }

    /**
     * A single-threaded view over the shared file handle, owning the aligned buffers its reads go through
     * and the one {@code float[]} they decode into.
     *
     * <p>One per scorer: Lucene hands every per-leaf scoring task its own copy of the vector values, and
     * those tasks run concurrently, so a shared buffer would be a data race. Buffers are allocated on
     * first use rather than in the constructor, because the seam creates a reader for the values it
     * returns and then a second one for the private copy the scorer actually reads through — allocating
     * eagerly would pay for buffers that are never used.
     *
     * <h2>The staging ring</h2>
     * Read ahead exists because the mmap path this replaces got its speed from {@code madvise}, which has
     * no meaning without a mapping: with one blocking read per candidate, a rescore query costs 200 serial
     * device round trips. So {@link #stage} takes the batch of ordinals the scorer is about to ask for and
     * puts a rolling window of them in flight on {@link DirectIOReadPool}, and {@link #read} then waits
     * only for the one it needs while the rest are already travelling.
     *
     * <p>Each ring entry is written by exactly one background read, decoded by exactly one {@link #read},
     * and then immediately reused for the next ordinal in the batch. <b>That makes this a staging area and
     * not a cache</b>: there is no eviction policy because there is nothing to evict, and no entry
     * survives its single consume. A future cache belongs at the loader seam, above this class, not here.
     *
     * <p>Staging is an optimisation and never a correctness requirement. Any reason not to stage — the
     * setting off, no pool, a rejected submission, a window of one, a batch shorter than two, or a
     * consumer that asks for an ordinal the batch did not predict — falls back to a single blocking read,
     * which is the path {@link #read} takes when no batch is active. The vector returned is the same
     * either way.
     *
     * <p>Buffers are released when the reader becomes unreachable, by the same {@code Cleaner} that
     * releases any direct buffer.
     */
    public final class Reader {

        /** Buffer for reads that are not served from the ring. Allocated on the first such read. */
        private ByteBuffer syncBuffer;
        private FloatBuffer syncFloats;
        private final float[] value = new float[dimension];

        /**
         * Ring slots, or null before the first staged batch. Slices of one aligned allocation, which they
         * keep reachable, so this is one direct buffer and one {@code Cleaner} registration however many
         * slots there are.
         */
        private ByteBuffer[] slots;
        /**
         * A {@code float} view per slot, created with the slot and never reassigned. A view addresses the
         * byte range the buffer had when the view was made, so it is unaffected by the {@code clear()} and
         * position changes each read does, and making it once keeps {@link #decode} allocation free.
         */
        private FloatBuffer[] slotFloats;
        private Future<?>[] pending;
        private StagedRead[] staged;
        private int[] slotOrd;
        private int ringSize;
        private ExecutorService pool;

        /** The ordinals of the batch being staged, and how far submission and consumption have got. */
        private int[] batch = new int[0];
        private int batchSize;
        private int nextSubmit;
        private int nextConsume;

        private Reader() {}

        /**
         * Declares that {@code ords[0..count)} are about to be read, in that order, and puts a window of
         * them in flight.
         *
         * <p>Advisory in both directions: this may stage none of them, and a caller that then reads
         * something else, or reads them out of order, gets correct vectors from blocking reads. It must be
         * called from the thread that will do the reading.
         *
         * @param ords  ordinals in the order they will be read; only the first {@code count} are looked at
         * @param count how many of {@code ords} are meaningful
         */
        public void stage(final int[] ords, final int count) {
            // Any previous batch has to be fully quiesced before its slots can be handed to new reads:
            // cancel(false) does not stop a read that has already started writing into a slot.
            abandonBatch();
            if (ords == null || count <= 1) {
                return;
            }
            final int window = prefetchWindow();
            if (window <= 1) {
                return;
            }
            for (int i = 0; i < count; i++) {
                if (ords[i] < 0 || ords[i] >= size) {
                    // Not this class' error to report: read(ord) will reject it with the ordinal in hand.
                    return;
                }
            }
            if (ensureRing(Math.min(window, count)) == false) {
                return;
            }
            if (batch.length < count) {
                batch = new int[count];
            }
            System.arraycopy(ords, 0, batch, 0, count);
            batchSize = count;
            nextSubmit = 0;
            nextConsume = 0;
            final int inFlight = Math.min(ringSize, count);
            for (int i = 0; i < inFlight; i++) {
                if (submitNext() == false) {
                    abandonBatch();
                    return;
                }
            }
        }

        /**
         * The vector at {@code ord}: taken from the staging ring when it is the next ordinal the current
         * batch predicted, and otherwise read with one {@code pread} of a block-aligned range.
         *
         * <p>The returned array is owned by this reader and is overwritten by the next call, which is the
         * same contract {@code FloatVectorValues#vectorValue(int)} has.
         *
         * @param ord the ordinal to read
         * @return the vector's floats
         * @throws IOException if the read fails or returns too few bytes
         */
        public float[] read(final int ord) throws IOException {
            if (ord < 0 || ord >= size) {
                throw new IllegalArgumentException(
                    String.format(Locale.ROOT, "Ordinal %d is out of range for %d vectors in %s", ord, size, path)
                );
            }
            if (batchSize > 0) {
                if (nextConsume < batchSize && batch[nextConsume] == ord) {
                    return consumeStaged();
                }
                // The consumer did not follow the order it declared. Correct, but every remaining staged
                // read is now speculative, so drop the batch rather than serve from it.
                abandonBatch();
            }
            return readBlocking(ord);
        }

        /** Waits for the head of the ring, decodes it, and refills the slot it frees. */
        private float[] consumeStaged() throws IOException {
            final int slot = nextConsume % ringSize;
            final Future<?> future = pending[slot];
            pending[slot] = null;
            staged[slot] = null;
            try {
                future.get();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                abandonBatch();
                throw new IOException("Interrupted waiting for a Direct I/O read of " + path, e);
            } catch (ExecutionException e) {
                abandonBatch();
                final Throwable cause = e.getCause();
                // Error propagates for the reason open() does not catch it: a direct-buffer OutOfMemoryError
                // is a misconfiguration to fix, not a condition to read around.
                if (cause instanceof Error) {
                    throw (Error) cause;
                }
                if (cause instanceof IOException) {
                    throw (IOException) cause;
                }
                throw new IOException("Direct I/O read of " + path + " failed", cause);
            }
            decode(slots[slot], slotFloats[slot], slotOrd[slot]);
            nextConsume++;
            if (submitNext() == false) {
                // value already holds this ordinal, so the batch can be dropped without losing this read.
                abandonBatch();
            }
            return value;
        }

        /**
         * Puts the next unsubmitted ordinal of the batch in flight, in the slot the ring has just freed.
         *
         * @return false if the pool refused the read, which means the batch cannot continue
         */
        private boolean submitNext() {
            if (nextSubmit >= batchSize) {
                return true;
            }
            final int ord = batch[nextSubmit];
            final int slot = nextSubmit % ringSize;
            final StagedRead read = new StagedRead(slots[slot], ord);
            try {
                pending[slot] = pool.submit(read);
            } catch (RejectedExecutionException e) {
                pending[slot] = null;
                staged[slot] = null;
                log.debug("Direct I/O rescore read pool refused a read of {}; falling back to blocking reads", path);
                return false;
            }
            staged[slot] = read;
            slotOrd[slot] = ord;
            nextSubmit++;
            return true;
        }

        /**
         * Quiesces everything the current batch still has in flight and forgets the batch.
         *
         * <p>A slot is safe to reuse only once no read can still write into it, and {@code Future#cancel} is
         * not enough for that: cancelling a task that has already started completes the future immediately
         * while the read runs on. So each read is claimed instead — a read that has not begun is skipped by
         * the claim and one that has is waited for.
         */
        private void abandonBatch() {
            if (pending != null) {
                for (int slot = 0; slot < ringSize; slot++) {
                    final Future<?> future = pending[slot];
                    if (future == null) {
                        continue;
                    }
                    final StagedRead read = staged[slot];
                    pending[slot] = null;
                    staged[slot] = null;
                    if (read.skipIfNotStarted()) {
                        continue;
                    }
                    try {
                        future.get();
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    } catch (ExecutionException e) {
                        log.debug("Abandoned a staged Direct I/O read of {}", path);
                    }
                }
            }
            batchSize = 0;
            nextSubmit = 0;
            nextConsume = 0;
        }

        /**
         * Allocates the ring, once, at {@code desired} slots, and takes a reference to the shared pool.
         *
         * @return false when read ahead is not available, so the caller should read blocking
         */
        private boolean ensureRing(final int desired) {
            if (slots != null) {
                return true;
            }
            final ExecutorService acquired = DirectIOReadPool.executor();
            if (acquired == null) {
                return false;
            }
            // One allocation for the whole ring, over-allocated by a block so a block-aligned start exists
            // inside it. Every slot boundary is then also block aligned, because bufferSize is a multiple
            // of blockSize by construction in DirectIOBufferSizer.
            final ByteBuffer arena = ByteBuffer.allocateDirect(desired * bufferSize + blockSize - 1).alignedSlice(blockSize);
            final ByteBuffer[] allocated = new ByteBuffer[desired];
            final FloatBuffer[] views = new FloatBuffer[desired];
            for (int i = 0; i < desired; i++) {
                allocated[i] = arena.slice(i * bufferSize, bufferSize).order(ByteOrder.LITTLE_ENDIAN);
                views[i] = allocated[i].asFloatBuffer();
            }
            slotFloats = views;
            slotOrd = new int[desired];
            pending = new Future<?>[desired];
            staged = new StagedRead[desired];
            ringSize = desired;
            pool = acquired;
            slots = allocated;
            return true;
        }

        /** One blocking {@code pread}, the whole of the Phase 2 path and the fallback for every other. */
        private float[] readBlocking(final int ord) throws IOException {
            if (syncBuffer == null) {
                // Mirrors Lucene's DirectIOIndexInput#allocateBuffer: over-allocate by a block so that a
                // block-aligned slice exists inside the allocation, since O_DIRECT requires the buffer
                // address, the file offset and the length to all be block aligned.
                syncBuffer = ByteBuffer.allocateDirect(bufferSize + blockSize - 1).alignedSlice(blockSize).order(ByteOrder.LITTLE_ENDIAN);
                syncFloats = syncBuffer.asFloatBuffer();
            }
            readInto(syncBuffer, ord);
            decode(syncBuffer, syncFloats, ord);
            return value;
        }

        /**
         * Decodes the vector at {@code ord} out of a buffer that already holds its block-aligned range.
         *
         * <p>The bulk path is one copy of {@code dimension} floats rather than {@code dimension} separate
         * reads, which matters because this runs once per rescore candidate: a query decodes
         * {@code firstPassK x dimension} floats, 153,600 of them at the shape this seam was measured on.
         * A {@code FloatBuffer} view of a direct buffer in the platform's byte order implements its
         * absolute bulk get as a memory copy, so there is no per-element cost and no intermediate object.
         *
         * @param target the buffer holding the block-aligned range, used by the unaligned fallback
         * @param view   {@code target}'s {@code float} view, created once with the buffer
         * @param ord    the ordinal whose bytes {@code target} holds
         */
        private void decode(final ByteBuffer target, final FloatBuffer view, final int ord) {
            final int delta = (int) ((baseOffset + (long) ord * vectorByteLength) % blockSize);
            if (bulkDecodable) {
                view.get(delta / Float.BYTES, value, 0, value.length);
                return;
            }
            for (int i = 0; i < value.length; i++) {
                value[i] = target.getFloat(delta + (i << 2));
            }
        }
    }

    /**
     * One staged read of one ordinal into one ring slot.
     *
     * <p>The claim is what makes a slot reusable: whoever wins {@code started} owns the buffer, so a read
     * abandoned before it began never touches a slot the next batch is already reading into, and one that
     * has begun can be waited out. {@code Future#cancel} cannot express this, since it completes the future
     * while a started task runs on.
     */
    private final class StagedRead implements Callable<Void> {

        private final ByteBuffer target;
        private final int ord;
        private final AtomicBoolean started = new AtomicBoolean();

        private StagedRead(final ByteBuffer target, final int ord) {
            this.target = target;
            this.ord = ord;
        }

        @Override
        public Void call() throws IOException {
            if (started.compareAndSet(false, true) == false) {
                return null;
            }
            readInto(target, ord);
            return null;
        }

        /** @return true if this read will now never run, false if it is running or has already run */
        private boolean skipIfNotStarted() {
            return started.compareAndSet(false, true);
        }
    }

    /**
     * Reads the block-aligned range holding {@code ord} into {@code target}.
     *
     * <p>Runs on the calling thread for a blocking read and on a {@link DirectIOReadPool} thread for a
     * staged one. Safe in both cases because a {@code FileChannel} is thread safe for positional reads and
     * each buffer is written by one read at a time, with the {@code Future} publishing it to the consumer.
     */
    private void readInto(final ByteBuffer target, final int ord) throws IOException {
        final long absolute = baseOffset + (long) ord * vectorByteLength;
        final int delta = (int) (absolute % blockSize);
        final long alignedStart = absolute - delta;
        target.clear();
        final int read = channel.read(target, alignedStart);
        // A short read is only possible at end of file, and the bytes wanted here always end at least
        // FOOTER_LENGTH bytes before it, so this cannot fire for a source that verified. It is checked
        // rather than asserted because the alternative is decoding whatever the buffer last held.
        if (read < delta + vectorByteLength) {
            throw new IOException(
                String.format(
                    Locale.ROOT,
                    "Direct I/O read of %s at %d returned %d bytes, need %d for ordinal %d",
                    path,
                    alignedStart,
                    read,
                    delta + vectorByteLength,
                    ord
                )
            );
        }
    }

    /**
     * The configured read-ahead window, or 1 to mean "do not read ahead". Read defensively for the reason
     * {@link #maxBufferSize()} is: an unreadable setting must leave the query working.
     */
    private static int prefetchWindow() {
        try {
            return KNNSettings.isDirectIORescorePrefetchEnabled() ? KNNSettings.getDirectIORescorePrefetchWindow() : 1;
        } catch (Exception e) {
            log.debug("Could not read the Direct I/O rescore prefetch settings; reading without prefetch", e);
            return 1;
        }
    }

    /** A new single-threaded reader over this source. Cheap: the buffer is allocated on first use. */
    public Reader newReader() {
        return new Reader();
    }

    /** Number of vectors in the region this source serves. */
    public int size() {
        return size;
    }

    /** Dimension of the vectors this source serves. */
    public int dimension() {
        return dimension;
    }

    /** On-disk size of one vector, in bytes. */
    public int vectorByteLength() {
        return vectorByteLength;
    }

    /** File offset of ordinal 0, as derived and then verified by {@link #open}. */
    public long baseOffset() {
        return baseOffset;
    }

    /** Size of one aligned read buffer, in bytes; a {@link Reader} holds one per staging ring slot. */
    public int bufferSize() {
        return bufferSize;
    }

    /** Filesystem block size the reads are aligned to, in bytes. */
    public int blockSize() {
        return blockSize;
    }

    /** The file this source reads. */
    public Path path() {
        return path;
    }

    @Override
    public void close() throws IOException {
        channel.close();
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

    /**
     * The buffer size above which this source declines the file, from
     * {@code knn.direct_io.max_buffer_size}. Read defensively: an unreadable setting must leave the query
     * on the default path rather than fail it.
     */
    private static long maxBufferSize() {
        try {
            return KNNSettings.getDirectIOMaxBufferSize().getBytes();
        } catch (Exception e) {
            log.debug("Could not read knn.direct_io.max_buffer_size; using its default", e);
            return KNNSettings.KNN_DIRECT_IO_MAX_BUFFER_SIZE_DEFAULT_VALUE.getBytes();
        }
    }

    private static void closeQuietly(final FileChannel channel) {
        if (channel == null) {
            return;
        }
        try {
            channel.close();
        } catch (IOException e) {
            log.debug("Failed to close a Direct I/O channel that was being abandoned", e);
        }
    }
}

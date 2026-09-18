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
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.OpenOption;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.Locale;

/**
 * A Direct I/O byte source for one field's full-precision vectors in a {@code .vec} file: given an
 * ordinal, it returns that vector's floats, read with {@code O_DIRECT} so the bytes never enter the
 * page cache.
 *
 * <p>This is the loader half of the rescore seam. It is deliberately <em>not</em> an
 * {@link org.apache.lucene.store.IndexInput}: Lucene's own {@code DirectIOIndexInput} is a sequential
 * stream with a single internal buffer, and this path wants one independent aligned buffer per query
 * thread over a shared file handle. The handle is opened once per segment and shared; the buffer is
 * owned by a {@link Reader}, one per scorer.
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
     * A single-threaded view over the shared file handle, owning the one aligned buffer its reads go
     * through and the one {@code float[]} they decode into.
     *
     * <p>One per scorer: Lucene hands every per-leaf scoring task its own copy of the vector values, and
     * those tasks run concurrently, so a shared buffer would be a data race. The buffer is allocated on
     * the first read rather than in the constructor, because the seam creates a reader for the values it
     * returns and then a second one for the private copy the scorer actually reads through — allocating
     * eagerly would pay for a buffer that is never used.
     *
     * <p>The buffer is released when the reader becomes unreachable, by the same {@code Cleaner} that
     * releases any direct buffer. Nothing is retained after a read: {@link #read} overwrites the buffer
     * every time, so this is a staging area and not a cache.
     */
    public final class Reader {

        private ByteBuffer buffer;
        private final float[] value = new float[dimension];

        private Reader() {}

        /**
         * The vector at {@code ord}, read with one {@code pread} of a block-aligned range.
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
            final long absolute = baseOffset + (long) ord * vectorByteLength;
            final int delta = (int) (absolute % blockSize);
            final long alignedStart = absolute - delta;

            final ByteBuffer target = buffer();
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
            for (int i = 0; i < value.length; i++) {
                value[i] = target.getFloat(delta + (i << 2));
            }
            return value;
        }

        private ByteBuffer buffer() {
            if (buffer == null) {
                // Mirrors Lucene's DirectIOIndexInput#allocateBuffer: over-allocate by a block so that a
                // block-aligned slice exists inside the allocation, since O_DIRECT requires the buffer
                // address, the file offset and the length to all be block aligned.
                buffer = ByteBuffer.allocateDirect(bufferSize + blockSize - 1).alignedSlice(blockSize).order(ByteOrder.LITTLE_ENDIAN);
            }
            return buffer;
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

    /** Size of each {@link Reader}'s aligned buffer, in bytes. */
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

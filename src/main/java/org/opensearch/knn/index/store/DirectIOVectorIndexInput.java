/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import lombok.NonNull;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.store.AlreadyClosedException;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.RandomAccessInput;

import java.io.Closeable;
import java.io.EOFException;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.util.Objects;
import java.util.Optional;

/**
 * An {@link IndexInput} over a faiss {@code .vec} that reads with blocking positional reads on the calling thread
 * from a file channel opened with {@code O_DIRECT}, so the bytes do not go through the page cache. Every read is
 * rounded out to whole blocks of the file store's block size into one reusable direct buffer per input.
 *
 * <p>An instance of this class exists only because a slice or a clone of the file was taken while the index had
 * Direct I/O on, so the route is already decided: this input never asks the setting again, and a slice or clone of it
 * is routed too. While the index is off, Lucene holds the original directory's own inputs instead, handed out by
 * {@link SwitchableIndexInput}.
 *
 * <p>Beside the channel each instance keeps the matching input of the original directory, with the same coordinates.
 * It serves two purposes: a read that fails on the {@code O_DIRECT} channel is read again from it, from the start of
 * the whole request, and {@link #randomAccessSlice(long, long)} is always its own, because Lucene reads a sparse
 * field's ord-to-doc table through a random-access slice and a blocking aligned read per lookup would be the wrong
 * shape for it.
 *
 * <p>A read failure never fails the read. Any {@link IOException} from the channel, and a short read that leaves the
 * request uncovered, falls back to the original input, marks the file's channel failed so that every later read of
 * that file goes straight to the original input, and logs once per file. An {@link java.io.EOFException} from the
 * bounds check is thrown before any device read and is not a failure.
 *
 * <p>Neither the channel nor the original input is closed by this class: both belong to the
 * {@link SwitchableIndexInput} that opened them, which closes each exactly once.
 */
@Log4j2
public final class DirectIOVectorIndexInput extends IndexInput {

    /**
     * One file's {@code O_DIRECT} channel, shared by every slice and clone of that file. It is opened by
     * {@link KNNDirectIODirectory#openDirect} and owned by the {@link SwitchableIndexInput} that asked for it, which
     * closes it exactly once. It lives here because it carries the failed mark the read fallback reads and writes.
     */
    static final class Handle implements Closeable {

        private final Path path;
        private final FileChannel channel;
        private final int blockSize;
        private final long fileLength;
        private volatile boolean failed;

        Handle(final Path path, final FileChannel channel, final int blockSize, final long fileLength) {
            this.path = path;
            this.channel = channel;
            this.blockSize = blockSize;
            this.fileLength = fileLength;
        }

        /**
         * Marks this file's channel failed, so that every later read of the file goes to the original input, and logs
         * the first failure.
         *
         * @param cause what the channel did
         */
        void markFailed(final IOException cause) {
            if (failed) {
                return;
            }
            synchronized (this) {
                if (failed) {
                    return;
                }
                failed = true;
            }
            log.warn("Direct I/O read of {} failed, this file falls back to the original input until it is opened again", path, cause);
        }

        /**
         * Whether a read has already failed on this channel.
         *
         * @return true if the channel is not to be used again
         */
        boolean hasFailed() {
            return failed;
        }

        /**
         * The file store's block size, which every read's offset and length are rounded out to.
         *
         * @return the block size in bytes
         */
        int blockSize() {
            return blockSize;
        }

        /**
         * The path of the file this channel reads.
         *
         * @return the path
         */
        Path path() {
            return path;
        }

        /**
         * The length of the file this channel reads, as the {@code O_DIRECT} open saw it.
         *
         * @return the length in bytes
         */
        long fileLength() {
            return fileLength;
        }

        /**
         * The length of the file as the {@code O_DIRECT} channel sees it right now, which the content check compares
         * against the length the directory chain reports.
         *
         * @return the length in bytes
         * @throws IOException if the channel cannot be measured
         */
        long channelSize() throws IOException {
            return channel.size();
        }

        @Override
        public void close() throws IOException {
            channel.close();
        }
    }

    private final Handle handle;
    private final IndexInput delegateInput;
    private final long offset;
    private final long length;
    private final int bufferSize;

    private long filePointer;
    private boolean closed;
    private ByteBuffer buffer;
    private long bufferStart = -1;
    private int bufferValid;

    /**
     * Creates a routed input over a settled file. {@link SwitchableIndexInput} makes one for each slice and clone it
     * routes, and this class makes one for each of its own.
     *
     * @param description   the resource description Lucene reports
     * @param handle        the file's shared channel
     * @param delegateInput the matching input of the original directory, with the same coordinates
     * @param offset        this input's offset in the file
     * @param length        this input's length
     * @param bufferSize    the buffer size, a multiple of the block size
     * @param filePointer   where this input starts reading
     */
    DirectIOVectorIndexInput(
        final String description,
        @NonNull final Handle handle,
        @NonNull final IndexInput delegateInput,
        final long offset,
        final long length,
        final int bufferSize,
        final long filePointer
    ) {
        super(description);
        this.handle = handle;
        this.delegateInput = delegateInput;
        this.offset = offset;
        this.length = length;
        this.bufferSize = bufferSize;
        this.filePointer = filePointer;
    }

    /**
     * One block-aligned positional read of a region of the file into a fresh aligned buffer. The content check in
     * {@link KNNDirectIODirectory#contentMatches} reads through this, because an aligned read is this class's job.
     *
     * @param handle the open {@code O_DIRECT} channel
     * @param start  where the region starts in the file
     * @param length how many bytes of it to return
     * @return the bytes of the region
     * @throws IOException if the read fails or returns too little to cover the region
     */
    static byte[] readAt(final Handle handle, final long start, final int length) throws IOException {
        final int blockSize = handle.blockSize();
        final long alignedStart = start - Math.floorMod(start, (long) blockSize);
        final int span = Math.toIntExact(start - alignedStart) + length;
        final ByteBuffer buffer = allocateAligned(roundUpToBlock(span, blockSize), blockSize);
        final int read = handle.channel.read(buffer, alignedStart);
        if (read < span) {
            throw new IOException("Direct I/O read of " + handle.path() + " at " + alignedStart + " returned " + read + ", need " + span);
        }
        final byte[] bytes = new byte[length];
        buffer.get(Math.toIntExact(start - alignedStart), bytes, 0, length);
        return bytes;
    }

    /**
     * A byte count rounded up to whole blocks, which every buffer and every read length is.
     *
     * @param bytes     the byte count
     * @param blockSize the file store's block size
     * @return the count rounded up
     */
    static int roundUpToBlock(final int bytes, final int blockSize) {
        return ((bytes + blockSize - 1) / blockSize) * blockSize;
    }

    private static ByteBuffer allocateAligned(final int bytes, final int blockSize) {
        return ByteBuffer.allocateDirect(bytes + blockSize - 1).alignedSlice(blockSize).order(ByteOrder.LITTLE_ENDIAN).limit(bytes);
    }

    @Override
    public byte readByte() throws IOException {
        ensureOpen();
        if (filePointer + 1 > length) {
            throw new EOFException("read past EOF: " + this);
        }
        final long start = filePointer;
        if (handle.hasFailed() == false) {
            try {
                final int index = fill(offset + start);
                final byte value = buffer.get(index);
                filePointer = start + 1;
                return value;
            } catch (IOException e) {
                fallback(e);
            }
        }
        delegateInput.seek(start);
        final byte value = delegateInput.readByte();
        filePointer = start + 1;
        return value;
    }

    /** One request's worth of bytes for the fixed-width reads, which are not shared between inputs. */
    private final byte[] scratch = new byte[Long.BYTES];

    @Override
    public void readBytes(final byte[] b, final int bufferOffset, final int len) throws IOException {
        ensureOpen();
        Objects.checkFromIndexSize(bufferOffset, len, b.length);
        if (filePointer + len > length) {
            throw new EOFException("read past EOF: " + this);
        }
        readWithFallback(filePointer, b, bufferOffset, len);
    }

    @Override
    public short readShort() throws IOException {
        ensureOpen();
        if (filePointer + Short.BYTES > length) {
            throw new EOFException("read past EOF: " + this);
        }
        readWithFallback(filePointer, scratch, 0, Short.BYTES);
        return (short) ((scratch[0] & 0xFF) | ((scratch[1] & 0xFF) << 8));
    }

    @Override
    public int readInt() throws IOException {
        ensureOpen();
        if (filePointer + Integer.BYTES > length) {
            throw new EOFException("read past EOF: " + this);
        }
        readWithFallback(filePointer, scratch, 0, Integer.BYTES);
        return intAt(scratch, 0);
    }

    @Override
    public long readLong() throws IOException {
        ensureOpen();
        if (filePointer + Long.BYTES > length) {
            throw new EOFException("read past EOF: " + this);
        }
        readWithFallback(filePointer, scratch, 0, Long.BYTES);
        return longAt(scratch, 0);
    }

    @Override
    public void readInts(final int[] dst, final int dstOffset, final int len) throws IOException {
        ensureOpen();
        Objects.checkFromIndexSize(dstOffset, len, dst.length);
        final int bytes = len * Integer.BYTES;
        if (filePointer + bytes > length) {
            throw new EOFException("read past EOF: " + this);
        }
        final byte[] raw = new byte[bytes];
        readWithFallback(filePointer, raw, 0, bytes);
        for (int i = 0; i < len; i++) {
            dst[dstOffset + i] = intAt(raw, i * Integer.BYTES);
        }
    }

    @Override
    public void readLongs(final long[] dst, final int dstOffset, final int len) throws IOException {
        ensureOpen();
        Objects.checkFromIndexSize(dstOffset, len, dst.length);
        final int bytes = len * Long.BYTES;
        if (filePointer + bytes > length) {
            throw new EOFException("read past EOF: " + this);
        }
        final byte[] raw = new byte[bytes];
        readWithFallback(filePointer, raw, 0, bytes);
        for (int i = 0; i < len; i++) {
            dst[dstOffset + i] = longAt(raw, i * Long.BYTES);
        }
    }

    /**
     * Reads one whole request into an array, from the channel if it has not failed, and from the original input
     * otherwise or when it fails here. A fallback re-reads the request from its start, so bytes a failed read already
     * copied are overwritten, and the whole request is served once rather than byte by byte.
     *
     * @param start     where the request starts, in this input's coordinates
     * @param dst       where to copy to
     * @param dstOffset the offset in dst
     * @param len       how many bytes
     * @throws IOException if the original input fails too, which is what main would do
     */
    private void readWithFallback(final long start, final byte[] dst, final int dstOffset, final int len) throws IOException {
        if (handle.hasFailed() == false) {
            try {
                copyAt(offset + start, dst, dstOffset, len);
                filePointer = start + len;
                return;
            } catch (IOException e) {
                fallback(e);
            }
        }
        delegateInput.seek(start);
        delegateInput.readBytes(dst, dstOffset, len);
        filePointer = start + len;
    }

    private static int intAt(final byte[] raw, final int index) {
        return (raw[index] & 0xFF) | ((raw[index + 1] & 0xFF) << 8) | ((raw[index + 2] & 0xFF) << 16) | ((raw[index + 3] & 0xFF) << 24);
    }

    private static long longAt(final byte[] raw, final int index) {
        return (intAt(raw, index) & 0xFFFFFFFFL) | ((long) intAt(raw, index + Integer.BYTES) << 32);
    }

    @Override
    public void readFloats(final float[] dst, final int dstOffset, final int len) throws IOException {
        ensureOpen();
        final long bytes = (long) len * Float.BYTES;
        if (filePointer + bytes > length) {
            throw new EOFException("read past EOF: " + this);
        }
        final long start = filePointer;
        if (handle.hasFailed() == false) {
            try {
                int done = 0;
                while (done < len) {
                    final long absolute = offset + start + (long) done * Float.BYTES;
                    final int index = fill(absolute);
                    final int floatsInBuffer = (bufferValid - index) / Float.BYTES;
                    if (floatsInBuffer <= 0) {
                        // One float straddles the end of the buffer window: take it through the byte path.
                        final byte[] four = new byte[Float.BYTES];
                        copyAt(absolute, four, 0, Float.BYTES);
                        dst[dstOffset + done] = Float.intBitsToFloat(
                            (four[0] & 0xFF) | ((four[1] & 0xFF) << 8) | ((four[2] & 0xFF) << 16) | ((four[3] & 0xFF) << 24)
                        );
                        done++;
                        continue;
                    }
                    final int floats = Math.min(len - done, floatsInBuffer);
                    for (int i = 0; i < floats; i++) {
                        dst[dstOffset + done + i] = buffer.getFloat(index + i * Float.BYTES);
                    }
                    done += floats;
                }
                filePointer = start + bytes;
                return;
            } catch (IOException e) {
                fallback(e);
            }
        }
        delegateInput.seek(start);
        delegateInput.readFloats(dst, dstOffset, len);
        filePointer = start + bytes;
    }

    /**
     * Marks this file's channel failed, so that this read and every later read of the file is served by the original
     * input.
     *
     * @param cause what the channel did
     */
    private void fallback(final IOException cause) {
        handle.markFailed(cause);
        bufferStart = -1;
        bufferValid = 0;
    }

    /**
     * Copies bytes out of the aligned buffer, filling it from the device as often as the span needs.
     *
     * @param absolute  the absolute offset in the file
     * @param dst       where to copy to
     * @param dstOffset the offset in dst
     * @param len       how many bytes
     * @throws IOException if a device read fails or returns short
     */
    private void copyAt(final long absolute, final byte[] dst, final int dstOffset, final int len) throws IOException {
        int done = 0;
        while (done < len) {
            final int index = fill(absolute + done);
            final int available = Math.min(len - done, bufferValid - index);
            buffer.get(index, dst, dstOffset + done, available);
            done += available;
        }
    }

    /**
     * Makes sure the buffer holds the byte at an absolute offset, with one block-aligned positional read when it does
     * not, and returns that byte's index in the buffer.
     *
     * @param absolute the absolute offset in the file
     * @return the index of that byte in the buffer
     * @throws IOException if the read fails or returns too little to cover the offset
     */
    private int fill(final long absolute) throws IOException {
        if (bufferStart >= 0 && absolute >= bufferStart && absolute < bufferStart + bufferValid) {
            return (int) (absolute - bufferStart);
        }
        final int blockSize = handle.blockSize();
        final long alignedStart = absolute - Math.floorMod(absolute, (long) blockSize);
        final ByteBuffer target = buffer();
        target.clear().limit(bufferSize);
        bufferStart = -1;
        bufferValid = 0;
        final int read = handle.channel.read(target, alignedStart);
        if (read <= 0) {
            throw new IOException("Direct I/O read of " + handle.path() + " at " + alignedStart + " returned " + read);
        }
        final int index = (int) (absolute - alignedStart);
        if (index >= read) {
            throw new IOException(
                "Direct I/O read of " + handle.path() + " at " + alignedStart + " returned " + read + ", need " + (index + 1)
            );
        }
        bufferStart = alignedStart;
        bufferValid = read;
        return index;
    }

    private ByteBuffer buffer() {
        if (buffer == null) {
            buffer = allocateAligned(bufferSize, handle.blockSize());
        }
        return buffer;
    }

    @Override
    public long getFilePointer() {
        return filePointer;
    }

    @Override
    public void seek(final long pos) throws IOException {
        ensureOpen();
        if (pos < 0 || pos > length) {
            throw new EOFException("seek past EOF: " + this);
        }
        filePointer = pos;
    }

    @Override
    public long length() {
        return length;
    }

    @Override
    public IndexInput slice(final String sliceDescription, final long sliceOffset, final long sliceLength) throws IOException {
        ensureOpen();
        if (sliceOffset < 0 || sliceLength < 0 || sliceOffset + sliceLength > length) {
            throw new IllegalArgumentException(
                "slice() " + sliceDescription + " out of bounds: offset=" + sliceOffset + ",length=" + sliceLength + ",fileLength=" + length
            );
        }
        return new DirectIOVectorIndexInput(
            getFullSliceDescription(sliceDescription),
            handle,
            delegateInput.slice(sliceDescription, sliceOffset, sliceLength),
            offset + sliceOffset,
            sliceLength,
            bufferSize,
            0L
        );
    }

    @Override
    public DirectIOVectorIndexInput clone() {
        ensureOpen();
        return new DirectIOVectorIndexInput(
            toString() + " [slice=clone]",
            handle,
            delegateInput.clone(),
            offset,
            length,
            bufferSize,
            filePointer
        );
    }

    @Override
    public RandomAccessInput randomAccessSlice(final long sliceOffset, final long sliceLength) throws IOException {
        ensureOpen();
        // The ord-to-doc table of a sparse field is read one lookup at a time through a random access slice, which on
        // an aligned blocking input would be one device read per lookup. It is small next to the vectors and it stays
        // on the original input.
        return delegateInput.randomAccessSlice(sliceOffset, sliceLength);
    }

    @Override
    public void prefetch(final long prefetchOffset, final long prefetchLength) throws IOException {
        ensureOpen();
        // Carrying a prefetch out needs the read pool a later part adds. A blocking read on the calling thread would
        // be a read, not a hint.
    }

    @Override
    public Optional<Boolean> isLoaded() {
        return Optional.empty();
    }

    @Override
    public void close() throws IOException {
        closed = true;
        buffer = null;
        bufferStart = -1;
        bufferValid = 0;
    }

    /**
     * Fails once this input has been closed, as every {@link IndexInput} does.
     */
    void ensureOpen() {
        if (closed) {
            throw new AlreadyClosedException("this input is closed: " + this);
        }
    }
}

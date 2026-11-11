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
 * An {@link IndexInput} over flat vector data that reads with blocking positional reads on the calling thread from a
 * file channel opened with {@code O_DIRECT}, so the bytes do not go through the page cache. Every read is rounded out
 * to whole blocks of the file store's block size into one reusable direct buffer per input.
 *
 * <p>Beside the channel each instance keeps the matching input of the original directory, with the same coordinates.
 * A read that fails on the channel is read again from it, from the start of the whole request, and the file's channel
 * is marked failed so that every later read of that file goes straight to it. So a read failure never fails a read.
 * An {@link EOFException} from a bounds check is thrown before any device read and is not such a failure.
 *
 * <p>Neither the channel nor the original input is closed by this class: both belong to the
 * {@link SwitchableIndexInput} that opened them, which closes each exactly once.
 */
@Log4j2
public final class DirectIOVectorIndexInput extends IndexInput {

    // One file's O_DIRECT channel, shared by every slice and clone of that file and closed by the
    // SwitchableIndexInput that asked for it. It lives here because it carries the failed mark the fallback reads.
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

        boolean hasFailed() {
            return failed;
        }

        int blockSize() {
            return blockSize;
        }

        Path path() {
            return path;
        }

        long fileLength() {
            return fileLength;
        }

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

    // One block-aligned positional read of a region into a fresh buffer, which the directory's content check uses.
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
        final long start = filePointer;
        if (handle.hasFailed() == false) {
            try {
                final short value = (short) valueAt(offset + start, Short.BYTES);
                filePointer = start + Short.BYTES;
                return value;
            } catch (IOException e) {
                fallback(e);
            }
        }
        delegateInput.seek(start);
        final short value = delegateInput.readShort();
        filePointer = start + Short.BYTES;
        return value;
    }

    @Override
    public int readInt() throws IOException {
        ensureOpen();
        if (filePointer + Integer.BYTES > length) {
            throw new EOFException("read past EOF: " + this);
        }
        final long start = filePointer;
        if (handle.hasFailed() == false) {
            try {
                final int value = (int) valueAt(offset + start, Integer.BYTES);
                filePointer = start + Integer.BYTES;
                return value;
            } catch (IOException e) {
                fallback(e);
            }
        }
        delegateInput.seek(start);
        final int value = delegateInput.readInt();
        filePointer = start + Integer.BYTES;
        return value;
    }

    @Override
    public long readLong() throws IOException {
        ensureOpen();
        if (filePointer + Long.BYTES > length) {
            throw new EOFException("read past EOF: " + this);
        }
        final long start = filePointer;
        if (handle.hasFailed() == false) {
            try {
                final long value = valueAt(offset + start, Long.BYTES);
                filePointer = start + Long.BYTES;
                return value;
            } catch (IOException e) {
                fallback(e);
            }
        }
        delegateInput.seek(start);
        final long value = delegateInput.readLong();
        filePointer = start + Long.BYTES;
        return value;
    }

    @Override
    public void readInts(final int[] dst, final int dstOffset, final int len) throws IOException {
        ensureOpen();
        Objects.checkFromIndexSize(dstOffset, len, dst.length);
        final long bytes = (long) len * Integer.BYTES;
        if (filePointer + bytes > length) {
            throw new EOFException("read past EOF: " + this);
        }
        final long start = filePointer;
        if (handle.hasFailed() == false) {
            try {
                int done = 0;
                while (done < len) {
                    final long absolute = offset + start + (long) done * Integer.BYTES;
                    final int index = fill(absolute);
                    final int available = (bufferValid - index) / Integer.BYTES;
                    if (available <= 0) {
                        dst[dstOffset + done] = (int) valueAt(absolute, Integer.BYTES);
                        done++;
                        continue;
                    }
                    final int count = Math.min(len - done, available);
                    for (int i = 0; i < count; i++) {
                        dst[dstOffset + done + i] = buffer.getInt(index + i * Integer.BYTES);
                    }
                    done += count;
                }
                filePointer = start + bytes;
                return;
            } catch (IOException e) {
                fallback(e);
            }
        }
        delegateInput.seek(start);
        delegateInput.readInts(dst, dstOffset, len);
        filePointer = start + bytes;
    }

    @Override
    public void readLongs(final long[] dst, final int dstOffset, final int len) throws IOException {
        ensureOpen();
        Objects.checkFromIndexSize(dstOffset, len, dst.length);
        final long bytes = (long) len * Long.BYTES;
        if (filePointer + bytes > length) {
            throw new EOFException("read past EOF: " + this);
        }
        final long start = filePointer;
        if (handle.hasFailed() == false) {
            try {
                int done = 0;
                while (done < len) {
                    final long absolute = offset + start + (long) done * Long.BYTES;
                    final int index = fill(absolute);
                    final int available = (bufferValid - index) / Long.BYTES;
                    if (available <= 0) {
                        dst[dstOffset + done] = valueAt(absolute, Long.BYTES);
                        done++;
                        continue;
                    }
                    final int count = Math.min(len - done, available);
                    for (int i = 0; i < count; i++) {
                        dst[dstOffset + done + i] = buffer.getLong(index + i * Long.BYTES);
                    }
                    done += count;
                }
                filePointer = start + bytes;
                return;
            } catch (IOException e) {
                fallback(e);
            }
        }
        delegateInput.seek(start);
        delegateInput.readLongs(dst, dstOffset, len);
        filePointer = start + bytes;
    }

    // A fallback re-reads the request from its start, so bytes a failed read already copied are overwritten and the
    // file pointer ends where a read that did not fail would leave it. A failure of the original input propagates.
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

    @Override
    public void readFloats(final float[] dst, final int dstOffset, final int len) throws IOException {
        ensureOpen();
        Objects.checkFromIndexSize(dstOffset, len, dst.length);
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
                    final int available = (bufferValid - index) / Float.BYTES;
                    if (available <= 0) {
                        dst[dstOffset + done] = Float.intBitsToFloat((int) valueAt(absolute, Float.BYTES));
                        done++;
                        continue;
                    }
                    final int count = Math.min(len - done, available);
                    for (int i = 0; i < count; i++) {
                        dst[dstOffset + done + i] = buffer.getFloat(index + i * Float.BYTES);
                    }
                    done += count;
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

    // One value of two, four or eight bytes, read from the buffer. A value that straddles the end of the buffer window
    // is assembled from its bytes, which is the only case the buffer's own little-endian reads cannot serve.
    private long valueAt(final long absolute, final int width) throws IOException {
        final int index = fill(absolute);
        if (bufferValid - index >= width) {
            return switch (width) {
                case Short.BYTES -> buffer.getShort(index);
                case Integer.BYTES -> buffer.getInt(index);
                default -> buffer.getLong(index);
            };
        }
        final byte[] value = new byte[width];
        copyAt(absolute, value, 0, width);
        long assembled = 0L;
        for (int i = width - 1; i >= 0; i--) {
            assembled = (assembled << 8) | (value[i] & 0xFFL);
        }
        return assembled;
    }

    private void fallback(final IOException cause) {
        handle.markFailed(cause);
        bufferStart = -1;
        bufferValid = 0;
    }

    private void copyAt(final long absolute, final byte[] dst, final int dstOffset, final int len) throws IOException {
        int done = 0;
        while (done < len) {
            final int index = fill(absolute + done);
            final int available = Math.min(len - done, bufferValid - index);
            buffer.get(index, dst, dstOffset + done, available);
            done += available;
        }
    }

    // Makes sure the buffer holds the byte at an absolute offset, with one aligned positional read when it does not,
    // and returns that byte's index in the buffer.
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
        // A sparse field's ord-to-doc table is read through this, one lookup at a time, which here would be a device
        // read per lookup. It is small next to the vectors, so it stays on the original input.
        return delegateInput.randomAccessSlice(sliceOffset, sliceLength);
    }

    @Override
    public void prefetch(final long prefetchOffset, final long prefetchLength) throws IOException {
        ensureOpen();
        // A blocking read on the calling thread would be a read, not a hint.
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

    void ensureOpen() {
        if (closed) {
            throw new AlreadyClosedException("this input is closed: " + this);
        }
    }
}

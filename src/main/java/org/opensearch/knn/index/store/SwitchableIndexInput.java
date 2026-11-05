/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import lombok.NonNull;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.store.AlreadyClosedException;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.RandomAccessInput;
import org.apache.lucene.util.IOUtils;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Optional;

/**
 * The input {@link KNNDirectIODirectory} hands out for a faiss {@code .vec} that could be read with {@code O_DIRECT}.
 * It forwards every method to the input the original directory opened, and at each slice and each clone it hands out
 * either that input's own slice or clone or a Direct I/O reader, by the index setting, in both directions.
 *
 * <p>While the index is off every slice and clone is the original input's own object, so an index that is off reads
 * exactly as it does without this plugin change: the same bytes through the same code, the same scoring kernels and
 * the same page-cache hints. Nothing is opened.
 *
 * <p>The first slice or clone taken while the index is on settles the file, and so does the first
 * {@link #isEfficientFor(IOContext)} call, because Lucene asks that question to decide whether to reopen the file and
 * the answer depends on the route. Settling asks the directory once for the Direct I/O side, which is the
 * {@code O_DIRECT} open and the content check, keeps that answer for the rest of this input's life, and closes what
 * it opened. So the file either becomes routed or stays on the original input, and a file the
 * platform or a transforming layer rules out pays for that verdict once rather than on every slice.
 *
 * <p>Its own reads always go to the original input. What reads a file input directly rather than through a slice or a
 * clone is the codec header and the footer checksum at reader open, two small reads at the two ends of the file.
 * Every read of a vector is made through a slice and {@code checkIntegrity} through a clone, so the Direct I/O path
 * covers every read that matters.
 */
@Log4j2
public final class SwitchableIndexInput extends IndexInput implements RandomAccessInput {

    private final KNNDirectIODirectory directory;
    private final IndexInput delegateInput;
    private final Path path;
    private final long fileLength;
    private final boolean sequentialOpen;

    private boolean settled;
    private RandomAccessInput wholeFileTable;
    private DirectIOVectorIndexInput.Handle handle;
    private int bufferSize;
    private volatile boolean closed;

    /**
     * Creates the switch over a file that passed every check {@link KNNDirectIODirectory#openInput} can make without
     * a device read.
     *
     * @param directory      the delegate that opened the file
     * @param name           the file name
     * @param path           the real path of the file, which the four checks at open resolved
     * @param delegateInput  the input the original directory opened, which this object takes ownership of
     * @param fileLength     the length that input reports
     * @param sequentialOpen whether the file was opened for a merge or a sequential read
     */
    SwitchableIndexInput(
        @NonNull final KNNDirectIODirectory directory,
        @NonNull final String name,
        @NonNull final Path path,
        @NonNull final IndexInput delegateInput,
        final long fileLength,
        final boolean sequentialOpen
    ) {
        super("SwitchableIndexInput(" + name + ")");
        this.directory = directory;
        this.path = path;
        this.delegateInput = delegateInput;
        this.fileLength = fileLength;
        this.sequentialOpen = sequentialOpen;
    }

    /**
     * Asks the directory for this file's Direct I/O side, once, and keeps the answer. Called from the first slice or
     * clone taken while the index is on.
     *
     * @return true if this file reads with {@code O_DIRECT}
     */
    private synchronized boolean settle() {
        if (closed) {
            // An input that is closing or closed opens nothing: the channel it opened would have no owner left to
            // close it.
            return false;
        }
        if (settled) {
            return handle != null;
        }
        settled = true;
        final DirectIOVectorIndexInput.Handle opened = directory.openDirect(path, fileLength);
        if (opened == null) {
            return false;
        }
        if (directory.contentMatches(opened, delegateInput) == false) {
            IOUtils.closeWhileHandlingException(opened);
            // With no counters this line is the only signal an operator gets that a layer in the directory chain
            // transforms the bytes a reader sees, such as encryption at rest, so it is a warning and not a debug line.
            log.warn("Direct I/O declined for {}: the bytes read through the directory chain and through O_DIRECT differ", path);
            return false;
        }
        handle = opened;
        bufferSize = KNNDirectIODirectory.bufferSizeFor(sequentialOpen, opened.blockSize());
        return true;
    }

    /**
     * The file's channel if a Direct I/O slice or clone may be made from it right now, or null. Settling and closing
     * hold the same monitor, so a channel is opened at most once and is never left without an owner. A caller that
     * slices or clones concurrently with a close can still be handed a channel this input is closing, which Lucene's
     * own contract rules out: an {@link IndexInput} may not be used while it is being closed.
     *
     * @return the handle, or null if this file declined, is closed, or the index is off
     */
    private DirectIOVectorIndexInput.Handle routedHandle() {
        // The switch is read outside the monitor: it is a cluster state lookup and no caller should wait on another
        // caller's settle to learn that the index is off.
        if (directory.isRouteEnabled() == false) {
            return null;
        }
        synchronized (this) {
            return settle() ? handle : null;
        }
    }

    /**
     * The buffer size a Direct I/O slice or clone of this file reads through, or zero before the file settled.
     *
     * @return the buffer size in bytes
     */
    synchronized int bufferSize() {
        return bufferSize;
    }

    @Override
    public IndexInput slice(final String sliceDescription, final long offset, final long length) throws IOException {
        ensureOpen();
        final DirectIOVectorIndexInput.Handle routed = routedHandle();
        final IndexInput delegateSlice = delegateInput.slice(sliceDescription, offset, length);
        if (routed == null) {
            return delegateSlice;
        }
        return new DirectIOVectorIndexInput(
            getFullSliceDescription(sliceDescription),
            routed,
            delegateSlice,
            offset,
            length,
            bufferSize(),
            0L
        );
    }

    @Override
    public IndexInput slice(final String sliceDescription, final long offset, final long length, final IOContext context)
        throws IOException {
        ensureOpen();
        final DirectIOVectorIndexInput.Handle routed = routedHandle();
        if (routed == null) {
            // The original input's own overload, which on an mmap input applies the context's read advice to the
            // slice. Composing the three-argument slice with updateIOContext would not, because mmap does not
            // implement updateIOContext.
            return delegateInput.slice(sliceDescription, offset, length, context);
        }
        return new DirectIOVectorIndexInput(
            getFullSliceDescription(sliceDescription),
            routed,
            delegateInput.slice(sliceDescription, offset, length, context),
            offset,
            length,
            KNNDirectIODirectory.bufferSizeFor(sequentialOpen || KNNDirectIODirectory.isSequential(context), routed.blockSize()),
            0L
        );
    }

    @Override
    public IndexInput clone() {
        ensureOpen();
        final DirectIOVectorIndexInput.Handle routed = routedHandle();
        final IndexInput delegateClone = delegateInput.clone();
        if (routed == null) {
            return delegateClone;
        }
        return new DirectIOVectorIndexInput(
            toString() + " [slice=clone]",
            routed,
            delegateClone,
            0L,
            fileLength,
            bufferSize(),
            delegateClone.getFilePointer()
        );
    }

    /**
     * The random-access view of the whole file, made once from the original input. Lucene 10.6's mmap input is a
     * {@link RandomAccessInput}, so this class is one too: an index that is off must answer every question main's
     * object answers, {@code instanceof} included. Every one of these reads is the original input's in both states,
     * for the reason {@link #randomAccessSlice(long, long)} gives.
     *
     * @return the view
     * @throws IOException if the original input cannot make one
     */
    private synchronized RandomAccessInput table() throws IOException {
        if (wholeFileTable == null) {
            wholeFileTable = delegateInput.randomAccessSlice(0L, fileLength);
        }
        return wholeFileTable;
    }

    @Override
    public byte readByte(final long pos) throws IOException {
        ensureOpen();
        return table().readByte(pos);
    }

    @Override
    public short readShort(final long pos) throws IOException {
        ensureOpen();
        return table().readShort(pos);
    }

    @Override
    public int readInt(final long pos) throws IOException {
        ensureOpen();
        return table().readInt(pos);
    }

    @Override
    public long readLong(final long pos) throws IOException {
        ensureOpen();
        return table().readLong(pos);
    }

    @Override
    public void readBytes(final long pos, final byte[] bytes, final int offset, final int length) throws IOException {
        ensureOpen();
        table().readBytes(pos, bytes, offset, length);
    }

    @Override
    public RandomAccessInput randomAccessSlice(final long offset, final long length) throws IOException {
        ensureOpen();
        // The ord-to-doc table of a sparse field is read through a random access slice, one lookup at a time. It is
        // small next to the vectors and it stays on the original input in both states.
        return delegateInput.randomAccessSlice(offset, length);
    }

    /**
     * Whether Lucene should keep this input for the given context or open the file again. A file that reads with
     * {@code O_DIRECT} answers for the buffer it reads through. A file that does not, because the index is off or
     * because it declined, answers exactly what the original input answers: anything else would make Lucene open the
     * file a second and a third time for a merge and a sequential search.
     *
     * @param context the context Lucene is asking about
     * @return true if this input is efficient for it
     */
    @Override
    public boolean isEfficientFor(final IOContext context) {
        ensureOpen();
        if (routedHandle() == null) {
            return delegateInput.isEfficientFor(context);
        }
        return sequentialOpen || KNNDirectIODirectory.isSequential(context) == false;
    }

    @Override
    public byte readByte() throws IOException {
        ensureOpen();
        return delegateInput.readByte();
    }

    @Override
    public void readBytes(final byte[] b, final int offset, final int len) throws IOException {
        ensureOpen();
        delegateInput.readBytes(b, offset, len);
    }

    @Override
    public void readBytes(final byte[] b, final int offset, final int len, final boolean useBuffer) throws IOException {
        ensureOpen();
        delegateInput.readBytes(b, offset, len, useBuffer);
    }

    @Override
    public void readInts(final int[] dst, final int offset, final int len) throws IOException {
        ensureOpen();
        delegateInput.readInts(dst, offset, len);
    }

    @Override
    public void readLongs(final long[] dst, final int offset, final int len) throws IOException {
        ensureOpen();
        delegateInput.readLongs(dst, offset, len);
    }

    @Override
    public void readFloats(final float[] dst, final int offset, final int len) throws IOException {
        ensureOpen();
        delegateInput.readFloats(dst, offset, len);
    }

    @Override
    public short readShort() throws IOException {
        ensureOpen();
        return delegateInput.readShort();
    }

    @Override
    public int readInt() throws IOException {
        ensureOpen();
        return delegateInput.readInt();
    }

    @Override
    public int readVInt() throws IOException {
        ensureOpen();
        return delegateInput.readVInt();
    }

    @Override
    public long readLong() throws IOException {
        ensureOpen();
        return delegateInput.readLong();
    }

    @Override
    public long readVLong() throws IOException {
        ensureOpen();
        return delegateInput.readVLong();
    }

    @Override
    public long getFilePointer() {
        ensureOpen();
        return delegateInput.getFilePointer();
    }

    @Override
    public void seek(final long pos) throws IOException {
        ensureOpen();
        delegateInput.seek(pos);
    }

    @Override
    public void skipBytes(final long numBytes) throws IOException {
        ensureOpen();
        delegateInput.skipBytes(numBytes);
    }

    @Override
    public long length() {
        ensureOpen();
        return fileLength;
    }

    @Override
    public void prefetch(final long offset, final long length) throws IOException {
        ensureOpen();
        // This input's own reads are the original input's in both states, so its hints are too, and an index that is
        // off keeps main's madvise behaviour byte for byte.
        delegateInput.prefetch(offset, length);
    }

    @Override
    public void updateIOContext(final IOContext context) throws IOException {
        ensureOpen();
        delegateInput.updateIOContext(context);
    }

    @Override
    public Optional<Boolean> isLoaded() {
        ensureOpen();
        return delegateInput.isLoaded();
    }

    /**
     * Closes the original input and, if one was ever opened, this file's {@code O_DIRECT} channel, each exactly once.
     * A slice or a clone this input handed out closes nothing, which is Lucene's own rule for both.
     *
     * @throws IOException if either close fails
     */
    @Override
    public void close() throws IOException {
        final DirectIOVectorIndexInput.Handle toClose;
        synchronized (this) {
            if (closed) {
                return;
            }
            closed = true;
            toClose = handle;
            handle = null;
        }
        IOUtils.close(delegateInput, toClose);
    }

    private void ensureOpen() {
        if (closed) {
            throw new AlreadyClosedException("this input is closed: " + this);
        }
    }
}

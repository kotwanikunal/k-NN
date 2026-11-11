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
 * The input {@link KNNDirectIODirectory} hands out for a flat vector data file that could be read with
 * {@code O_DIRECT}. It forwards every method to the input the original directory opened, and at each slice and each
 * clone it hands out either that input's own slice or clone or a {@link DirectIOVectorIndexInput}, by the index
 * setting, in both directions.
 *
 * <p>While the index is off every slice and clone is the original input's own object, so the file reads exactly as it
 * does without this delegate and nothing is opened. The first slice, clone or efficiency question taken while the
 * index is on opens the {@code O_DIRECT} channel, runs the content check and keeps the answer for the rest of this
 * input's life.
 *
 * <p>Its own reads always go to the original input: what reads a file input directly rather than through a slice or a
 * clone is the codec header and the footer checksum at reader open.
 */
@Log4j2
public final class SwitchableIndexInput extends IndexInput implements RandomAccessInput {

    private final KNNDirectIODirectory directory;
    private final IndexInput delegateInput;
    private final Path path;
    private final long fileLength;
    private final boolean sequentialOpen;

    private boolean initialized;
    private RandomAccessInput delegateRandomAccess;
    private DirectIOVectorIndexInput.Handle handle;
    private int bufferSize;
    private volatile boolean closed;

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

    // Opens this file's O_DIRECT channel and runs the content check, once. Called from the first slice or clone taken
    // while the index is on.
    private synchronized boolean initializeDirectIO() {
        if (closed) {
            // An input that is closing or closed opens nothing: the channel would have no owner left to close it.
            return false;
        }
        if (initialized) {
            return handle != null;
        }
        initialized = true;
        final DirectIOVectorIndexInput.Handle opened = directory.openDirect(path, fileLength);
        if (opened == null) {
            return false;
        }
        if (directory.contentMatches(opened, delegateInput) == false) {
            IOUtils.closeWhileHandlingException(opened);
            // The only signal an operator gets that a layer in the chain transforms the bytes a reader sees, so it is
            // a warning rather than a debug line.
            log.warn("Direct I/O declined for {}: the bytes read through the directory chain and through O_DIRECT differ", path);
            return false;
        }
        handle = opened;
        bufferSize = KNNDirectIODirectory.bufferSizeFor(sequentialOpen, opened.blockSize());
        return true;
    }

    // Null if this file declined, is closed, or the index is off. Initializing and closing hold the same monitor, so a
    // channel is opened at most once and is never left without an owner. A caller that slices or clones concurrently
    // with a close can be handed a channel this input is closing, which Lucene's contract rules out anyway.
    private DirectIOVectorIndexInput.Handle routedHandle() {
        // The switch is read outside the monitor: it is a cluster state lookup and no caller should wait on another
        // caller's initialization to learn that the index is off.
        if (directory.isRouteEnabled() == false) {
            return null;
        }
        synchronized (this) {
            return initializeDirectIO() ? handle : null;
        }
    }

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
            // slice. Composing the three-argument slice with updateIOContext would not: mmap does not implement it.
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

    // Lucene's mmap input is a RandomAccessInput, so this class is one too, or a caller that tests for it would take
    // another path than it does without this delegate. IndexInput offers no absolute reads of its own, so the five
    // below go through one random-access view of the original input, in both states.
    private synchronized RandomAccessInput delegateRandomAccess() throws IOException {
        if (delegateRandomAccess == null) {
            delegateRandomAccess = delegateInput.randomAccessSlice(0L, fileLength);
        }
        return delegateRandomAccess;
    }

    @Override
    public byte readByte(final long pos) throws IOException {
        ensureOpen();
        return delegateRandomAccess().readByte(pos);
    }

    @Override
    public short readShort(final long pos) throws IOException {
        ensureOpen();
        return delegateRandomAccess().readShort(pos);
    }

    @Override
    public int readInt(final long pos) throws IOException {
        ensureOpen();
        return delegateRandomAccess().readInt(pos);
    }

    @Override
    public long readLong(final long pos) throws IOException {
        ensureOpen();
        return delegateRandomAccess().readLong(pos);
    }

    @Override
    public void readBytes(final long pos, final byte[] bytes, final int offset, final int length) throws IOException {
        ensureOpen();
        delegateRandomAccess().readBytes(pos, bytes, offset, length);
    }

    @Override
    public RandomAccessInput randomAccessSlice(final long offset, final long length) throws IOException {
        ensureOpen();
        // A sparse field's ord-to-doc table is read through this, one lookup at a time, which on an aligned blocking
        // input would be a device read per lookup. It is small next to the vectors, so it stays on the original input.
        return delegateInput.randomAccessSlice(offset, length);
    }

    // A file that is not routed answers exactly what the original input answers, or Lucene would open the file a
    // second and a third time for a merge and a sequential search.
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
        // This input's own reads are the original input's in both states, so its hints are too.
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

    // Closes the original input and, if one was ever opened, this file's O_DIRECT channel, each exactly once. A slice
    // or a clone this input handed out closes nothing, which is Lucene's own rule for both.
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

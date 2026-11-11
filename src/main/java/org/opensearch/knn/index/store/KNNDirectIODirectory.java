/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import lombok.NonNull;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.CompoundDirectory;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.store.DataAccessHint;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.FilterDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.IOUtils;
import org.opensearch.knn.common.ShardPathUtil;
import org.opensearch.knn.index.KNNSettings;

import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.OpenOption;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.function.BooleanSupplier;

import static org.opensearch.knn.index.codec.KNN1040Codec.KNN1040HalfFloatFlatVectorsFormat.VECTOR_DATA_EXTENSION;

/**
 * A {@link FilterDirectory} that the two faiss vector formats put in front of a segment's directory, through
 * {@link #routeVectorData}, so that the flat vector data of a faiss field can be read with {@code O_DIRECT} instead of
 * through the page cache. It decides and configures, as Lucene's {@code DirectIODirectory} does, while
 * {@link SwitchableIndexInput} picks a route per slice and clone and {@link DirectIOVectorIndexInput} reads.
 *
 * <p>The content check compares the length, the head and the codec footer, so a directory layer that transforms the
 * bytes a reader sees, such as encryption at rest, is declined. <b>A transform of interior bytes only is not
 * detected</b>, and such a layer would serve transformed vectors through this delegate.
 *
 * <p>One of these is made per segment per reader and is never closed: {@link FilterDirectory#close()} would close the
 * shard's directory, which belongs to the reader that built the {@link SegmentReadState}.
 */
@Log4j2
public final class KNNDirectIODirectory extends FilterDirectory {

    static final long MIN_FILE_SIZE_BYTES = 10L * 1024 * 1024; // Lucene's DirectIODirectory.DEFAULT_MIN_BYTES_DIRECT
    static final int SEQUENTIAL_BUFFER_BYTES = 256 * 1024; // Lucene's DirectIODirectory.DEFAULT_MERGE_BUFFER_SIZE
    static final int CONTENT_HEADER_BYTES = 4096;

    // Looked up reflectively, as Lucene's DirectIODirectory.ExtendedOpenOption_DIRECT is, so that the plugin links on
    // a JDK without the option. Null then, and no file is ever routed.
    static final OpenOption DIRECT_OPEN_OPTION;

    static {
        OpenOption option = null;
        try {
            final Class<? extends OpenOption> extended = Class.forName("com.sun.nio.file.ExtendedOpenOption").asSubclass(OpenOption.class);
            final OpenOption[] options = extended.getEnumConstants();
            if (options != null) {
                option = Arrays.stream(options)
                    .filter(candidate -> "DIRECT".equalsIgnoreCase(candidate.toString()))
                    .findFirst()
                    .orElse(null);
            }
        } catch (ClassNotFoundException | RuntimeException e) {
            option = null;
        }
        DIRECT_OPEN_OPTION = option;
        if (option == null) {
            log.warn(
                "Direct I/O is not available on this runtime: com.sun.nio.file.ExtendedOpenOption.DIRECT is absent, so "
                    + "index.knn.direct_io.rescore.enabled has no effect and vector data is read through the page cache"
            );
        }
    }

    @FunctionalInterface
    interface ChannelFactory {
        FileChannel open(Path path) throws IOException;
    }

    static final ChannelFactory DEFAULT_CHANNEL_FACTORY = KNNDirectIODirectory::openDirectChannel;

    private final long minFileSize;
    private final BooleanSupplier routeEnabled;
    private final ChannelFactory channelFactory;
    private final OpenOption directOption;

    /**
     * Puts this delegate in front of the directory of the segment a flat vector reader is about to open.
     *
     * @param state The state the faiss format was handed by Lucene.
     * @return A state whose directory is this delegate, or the given state when the shard path carries no index UUID,
     *         because the setting of such a segment can never be read.
     */
    public static SegmentReadState routeVectorData(@NonNull final SegmentReadState state) {
        final String indexUuid = ShardPathUtil.tryGetIndexUuid(state.segmentInfo.dir);
        if (indexUuid == null) {
            return state;
        }
        return new SegmentReadState(
            new KNNDirectIODirectory(state.directory, () -> KNNSettings.isDirectIORescoreEnabled(indexUuid)),
            state.segmentInfo,
            state.fieldInfos,
            state.context,
            state.segmentSuffix
        );
    }

    KNNDirectIODirectory(@NonNull final Directory in, @NonNull final BooleanSupplier routeEnabled) {
        this(in, MIN_FILE_SIZE_BYTES, routeEnabled, DEFAULT_CHANNEL_FACTORY, DIRECT_OPEN_OPTION);
    }

    KNNDirectIODirectory(
        @NonNull final Directory in,
        final long minFileSize,
        @NonNull final BooleanSupplier routeEnabled,
        @NonNull final ChannelFactory channelFactory,
        final OpenOption directOption
    ) {
        super(in);
        this.minFileSize = minFileSize;
        this.routeEnabled = routeEnabled;
        this.channelFactory = channelFactory;
        this.directOption = directOption;
    }

    @Override
    public IndexInput openInput(final String name, final IOContext context) throws IOException {
        if (IndexFileNames.matchesExtension(name, VECTOR_DATA_EXTENSION) == false || directOption == null) {
            return in.openInput(name, context);
        }
        final IndexInput delegateInput = in.openInput(name, context);
        boolean handedOver = false;
        try {
            final IndexInput result = wrapOrDecline(name, context, delegateInput);
            handedOver = true;
            return result;
        } finally {
            if (handedOver == false) {
                IOUtils.closeWhileHandlingException(delegateInput);
            }
        }
    }

    // The checks that need no device read. A decline here is normal, so it logs at debug.
    private IndexInput wrapOrDecline(final String name, final IOContext context, final IndexInput delegateInput) {
        if (in instanceof CompoundDirectory) {
            log.debug("Direct I/O declined for {}: the segment is compound", name);
            return delegateInput;
        }
        final Path path = resolvePath(in, name);
        if (path == null) {
            log.debug("Direct I/O declined for {}: the directory chain does not end in a file-system directory: {}", name, in);
            return delegateInput;
        }
        if (Files.isRegularFile(path) == false) {
            log.debug("Direct I/O declined for {}: {} is not a regular file", name, path);
            return delegateInput;
        }
        final long length = delegateInput.length();
        if (length < minFileSize) {
            log.debug("Direct I/O declined for {}: {} bytes is under the floor of {}", name, length, minFileSize);
            return delegateInput;
        }
        return new SwitchableIndexInput(this, name, path, delegateInput, length, isSequential(context));
    }

    // Returns null rather than throwing when O_DIRECT is not available for this file, so that the caller serves the
    // original input. A file that passed the checks at open was expected to route, so the decline is a warning.
    DirectIOVectorIndexInput.Handle openDirect(@NonNull final Path path, final long length) {
        FileChannel channel = null;
        try {
            final int blockSize = Math.toIntExact(Files.getFileStore(path).getBlockSize());
            if (blockSize <= 0 || Integer.bitCount(blockSize) != 1) {
                // A buffer can only be aligned to a power of two, so anything else declines here rather than
                // surfacing later as a content mismatch.
                log.warn("Direct I/O declined for {}: the file store reports a block size of {}, not a power of two", path, blockSize);
                return null;
            }
            channel = channelFactory.open(path);
            return new DirectIOVectorIndexInput.Handle(path, channel, blockSize, length);
        } catch (IOException | RuntimeException e) {
            IOUtils.closeWhileHandlingException(channel);
            log.warn("Direct I/O declined for {}: the file cannot be opened with O_DIRECT: {}", path, e.toString());
            return null;
        }
    }

    // The last check: the same length, head and codec footer through the O_DIRECT channel and through the directory
    // chain. The delegate input is left where it was.
    boolean contentMatches(@NonNull final DirectIOVectorIndexInput.Handle handle, @NonNull final IndexInput delegate) {
        try {
            if (delegate.length() != handle.channelSize()) {
                // The file on disk, measured through the O_DIRECT channel, against the length the directory chain
                // reports. Comparing against Handle#fileLength would compare the delegate's length with a copy of
                // itself.
                return false;
            }
            final int headerBytes = (int) Math.min((long) CONTENT_HEADER_BYTES, handle.fileLength());
            if (headerBytes <= 0 || regionMatches(handle, delegate, 0L, headerBytes) == false) {
                return false;
            }
            final long footerStart = handle.fileLength() - CodecUtil.footerLength();
            if (footerStart < headerBytes) {
                // The header comparison already covered the footer.
                return true;
            }
            return regionMatches(handle, delegate, footerStart, CodecUtil.footerLength());
        } catch (IOException | RuntimeException e) {
            log.warn("Direct I/O declined for {}: the content check failed: {}", handle.path(), e.toString());
            return false;
        }
    }

    private static boolean regionMatches(
        final DirectIOVectorIndexInput.Handle handle,
        final IndexInput delegate,
        final long start,
        final int length
    ) throws IOException {
        final byte[] fromChannel = DirectIOVectorIndexInput.readAt(handle, start, length);
        final byte[] fromDelegate = new byte[length];
        final IndexInput clone = delegate.clone();
        clone.seek(start);
        clone.readBytes(fromDelegate, 0, length);
        return Arrays.equals(fromChannel, fromDelegate);
    }

    // Two blocks for a random reader, which hold any vector up to one block even when it straddles a boundary, and
    // Lucene's merge buffer for a sequential one.
    static int bufferSizeFor(final boolean sequential, final int blockSize) {
        final int target = sequential ? SEQUENTIAL_BUFFER_BYTES : 2 * blockSize;
        return DirectIOVectorIndexInput.roundUpToBlock(Math.max(target, blockSize), blockSize);
    }

    static boolean isSequential(final IOContext context) {
        if (context == null) {
            return false;
        }
        return context.context() == IOContext.Context.MERGE || context.hints().contains(DataAccessHint.SEQUENTIAL);
    }

    private static FileChannel openDirectChannel(final Path path) throws IOException {
        return FileChannel.open(path, StandardOpenOption.READ, DIRECT_OPEN_OPTION);
    }

    // Read once per slice and per clone of a wrapped file, never per read.
    boolean isRouteEnabled() {
        return routeEnabled.getAsBoolean();
    }

    // The real path behind a directory chain, or null when the chain does not end in an FSDirectory. The chain is not
    // judged by the class names in it: a layer that transforms bytes is caught by comparing the bytes.
    static Path resolvePath(final Directory dir, final String name) {
        if (dir == null) {
            return null;
        }
        final Directory bottom = FilterDirectory.unwrap(dir);
        return bottom instanceof FSDirectory ? ((FSDirectory) bottom).getDirectory().resolve(name) : null;
    }
}

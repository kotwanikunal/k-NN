/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import lombok.NonNull;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.CompoundDirectory;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.store.DataAccessHint;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.FilterDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.util.IOUtils;
import org.opensearch.core.index.Index;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.codec.KNN1040Codec.Faiss1040ScalarQuantizedKnnVectorsFormat;
import org.opensearch.knn.index.codec.KNN990Codec.NativeEngines990KnnVectorsFormat;

import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.OpenOption;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.function.BooleanSupplier;
import java.util.function.LongSupplier;

/**
 * A {@link FilterDirectory} that the two faiss vector formats put in front of a segment's directory so that the
 * full-precision vector data file of a faiss field, {@code .vec}, can be read with {@code O_DIRECT} instead of
 * through the page cache. Every other file of the segment is opened from the original directory, and the shard's own
 * directory stack is untouched.
 *
 * <p>This class decides and configures, in the division Lucene's own {@code DirectIODirectory} uses: it holds the
 * {@code DIRECT} open option, the file size floor, the block size, the buffer sizes and the channel factory, and it
 * runs the eligibility checks. The only bytes it reads are the two small regions the content check compares, through
 * {@link DirectIOVectorIndexInput#readAt} and through a clone of the original input. {@link SwitchableIndexInput} chooses between the two routes for
 * each slice and clone, and {@link DirectIOVectorIndexInput} does the reading.
 *
 * <p>A {@code .vec} is read with {@code O_DIRECT} only if all six checks hold: it is the vector data file of a faiss
 * or memory-optimized-search field, the segment is not compound, the directory chain ends in an
 * {@link org.apache.lucene.store.FSDirectory} so the file has a real path, the file exists and is at least
 * {@link #MIN_FILE_SIZE_BYTES}, both the {@code O_DIRECT} open and the original directory's open succeed, and the
 * bytes read through the chain and through {@code O_DIRECT} match. The four checks that need no device read run in
 * {@link #openInput}, and a file that fails one of them is the original directory's own input, exactly as it is
 * without this plugin change. The {@code O_DIRECT} open and the content check run once per file, at the first slice or
 * clone taken while the index is on, because the file system a file lives on is the only thing left to learn about it.
 *
 * <p>The content check compares the length, the first {@link #CONTENT_HEADER_BYTES} and the last 16 bytes, so a
 * directory layer that transforms the bytes a reader sees, such as encryption at rest, changes the codec header or the
 * footer checksum and is declined. <b>A transform that leaves the header and the footer alone and changes only
 * interior bytes is not detected</b>, and such a layer would serve transformed vectors through this delegate.
 *
 * <p>This object is created per segment per reader and is never closed: {@link FilterDirectory#close()} would close
 * the shard's directory, which belongs to the reader that built the {@link SegmentReadState}.
 */
@Log4j2
public final class KNNDirectIODirectory extends FilterDirectory {

    /**
     * The smallest {@code .vec} that is worth an {@code O_DIRECT} descriptor, 10,485,760 bytes. This is Lucene's own
     * {@code DirectIODirectory.DEFAULT_MIN_BYTES_DIRECT}.
     */
    static final long MIN_FILE_SIZE_BYTES = 10L * 1024 * 1024;

    /** The name of the directory a shard's files live in, {@code <uuid>/<shard>/index}. */
    static final String INDEX_FOLDER_NAME = "index";

    /** The extension of a flat vector data file. */
    static final String VECTOR_DATA_EXTENSION = ".vec";

    /** The name of the FP16 flat vector format, which is private in its own class. */
    static final String HALF_FLOAT_FORMAT_NAME = "NativeEngines990HalfFloatKnnVectorsFormat";

    /** The name of the FP16 scalar quantized vector format, which is private in its own class. */
    static final String HALF_FLOAT_SQ_FORMAT_NAME = "Faiss1040HalfFloatScalarQuantizedKnnVectorsFormat";

    /** The buffer a merge or a sequential search reads through, Lucene's {@code DEFAULT_MERGE_BUFFER_SIZE}. */
    static final int SEQUENTIAL_BUFFER_BYTES = 256 * 1024;

    /** How many bytes of the head of the file the content check compares. */
    static final int CONTENT_HEADER_BYTES = 4096;

    /**
     * {@code com.sun.nio.file.ExtendedOpenOption.DIRECT}, looked up once per JVM by reflective class and enum lookup
     * the way Lucene's {@code DirectIODirectory.ExtendedOpenOption_DIRECT} is, so that this plugin links and runs on a
     * JDK that does not have it. Null when the option is absent, and then no {@code .vec} is ever wrapped and every
     * index is served from the page cache whatever its setting says.
     */
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

    /** How the {@code O_DIRECT} channel is opened, which the unit tests replace. */
    @FunctionalInterface
    interface ChannelFactory {
        /**
         * Opens a channel for reading with {@code O_DIRECT}.
         *
         * @param path the file to open
         * @return the channel
         * @throws IOException if the open fails, which is a decline and not a failed shard
         */
        FileChannel open(Path path) throws IOException;
    }

    /** The production channel factory: {@code FileChannel.open(path, READ, ExtendedOpenOption.DIRECT)}. */
    static final ChannelFactory DEFAULT_CHANNEL_FACTORY = KNNDirectIODirectory::openDirectChannel;

    /** Why a {@code .vec} that this delegate was asked for is read from the original directory instead. */
    enum DeclineReason {
        /** The segment is compound, so its files have no path of their own. */
        COMPOUND,
        /** The directory chain does not end in a file-system directory. */
        CHAIN,
        /** The resolved path holds no regular file. */
        MISSING,
        /** The file is smaller than {@link KNNDirectIODirectory#MIN_FILE_SIZE_BYTES}. */
        TOO_SMALL,
        /** The {@code O_DIRECT} open failed on this file system or this file. */
        OPEN_FAILED,
        /** The bytes read through the directory chain and through {@code O_DIRECT} differ. */
        CONTENT
    }

    private final String indexUuid;
    private final LongSupplier minFileSize;
    private final BooleanSupplier routeEnabled;
    private final ChannelFactory channelFactory;
    private final OpenOption directOption;

    private volatile Index index;

    /**
     * Puts this delegate in front of the directory of the segment the flat vector reader is about to open, so that
     * the reader's {@code .vec} can be routed. Returns the caller's own state when the segment's shard path yields no
     * index UUID, because no later change of the setting could be read for such a segment.
     *
     * @param state the state the faiss format was handed by Lucene
     * @return a state whose directory is this delegate, or the caller's own state
     */
    public static SegmentReadState routeVectorData(@NonNull final SegmentReadState state) {
        final String indexUuid = indexUuidOf(state.segmentInfo.dir);
        if (indexUuid == null) {
            return state;
        }
        return new SegmentReadState(
            new KNNDirectIODirectory(state.directory, indexUuid),
            state.segmentInfo,
            state.fieldInfos,
            state.context,
            state.segmentSuffix
        );
    }

    /**
     * Creates a delegate over a segment's directory.
     *
     * @param in        the directory the segment reader was built with
     * @param indexUuid the UUID of the index the segment belongs to
     */
    public KNNDirectIODirectory(@NonNull final Directory in, @NonNull final String indexUuid) {
        this(in, indexUuid, () -> MIN_FILE_SIZE_BYTES, null, DEFAULT_CHANNEL_FACTORY, DIRECT_OPEN_OPTION);
    }

    /**
     * The constructor the unit tests use, which replaces the four values a test cannot otherwise reach: the file size
     * floor, the switch, the way the {@code O_DIRECT} channel is opened, and whether this runtime has the
     * {@code DIRECT} open option at all.
     *
     * @param in             the directory the segment reader was built with
     * @param indexUuid      the UUID of the index the segment belongs to
     * @param minFileSize    the file size floor to apply
     * @param routeEnabled   the switch, or null to read {@code index.knn.direct_io.rescore.enabled} from the cluster
     *                       state as production does
     * @param channelFactory how to open the {@code O_DIRECT} channel
     * @param directOption   the {@code DIRECT} open option, or null to behave as a JDK that does not have it
     */
    KNNDirectIODirectory(
        @NonNull final Directory in,
        @NonNull final String indexUuid,
        @NonNull final LongSupplier minFileSize,
        final BooleanSupplier routeEnabled,
        @NonNull final ChannelFactory channelFactory,
        final OpenOption directOption
    ) {
        super(in);
        this.indexUuid = indexUuid;
        this.minFileSize = minFileSize;
        this.routeEnabled = routeEnabled;
        this.channelFactory = channelFactory;
        this.directOption = directOption;
    }

    /**
     * Opens a file of the segment. A {@code .vec} of a faiss or memory-optimized-search field that passes the four
     * checks needing no device read is wrapped in a {@link SwitchableIndexInput}, which decides per slice and per
     * clone whether to read it with {@code O_DIRECT}. Every other file, every {@code .vec} of another engine, and
     * every {@code .vec} that fails one of those four checks, is the original directory's own input.
     *
     * @param name    the file name
     * @param context the context Lucene is opening the file for
     * @return the input the reader will use
     * @throws IOException if the original directory's open fails
     */
    @Override
    public IndexInput openInput(final String name, final IOContext context) throws IOException {
        if (isFaissVectorData(name) == false || directOption == null) {
            // A JDK without the DIRECT option said so once, in the static initializer: nothing is logged per file.
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

    /**
     * Runs the four checks that need no device read and returns either a switch over the file or the input the
     * original directory just handed out. A decline at this point is normal and logs at debug.
     *
     * @param name          the file name
     * @param context       the context Lucene is opening the file for, which fixes the buffer size
     * @param delegateInput the input the original directory just opened
     * @return the input the reader will use
     */
    private IndexInput wrapOrDecline(final String name, final IOContext context, final IndexInput delegateInput) {
        if (in instanceof CompoundDirectory) {
            log.debug("Direct I/O declined for {} ({}): the segment is compound", name, DeclineReason.COMPOUND);
            return delegateInput;
        }
        final Path path = resolvePath(in, name);
        if (path == null) {
            log.debug(
                "Direct I/O declined for {} ({}): this directory chain has no path: {}",
                name,
                DeclineReason.CHAIN,
                describeChain(in)
            );
            return delegateInput;
        }
        if (Files.isRegularFile(path) == false) {
            log.debug("Direct I/O declined for {} ({}): {} holds no regular file", name, DeclineReason.MISSING, path);
            return delegateInput;
        }
        final long length = delegateInput.length();
        if (length < minFileSizeBytes()) {
            log.debug(
                "Direct I/O declined for {} ({}): {} bytes is under the floor of {}",
                name,
                DeclineReason.TOO_SMALL,
                length,
                minFileSizeBytes()
            );
            return delegateInput;
        }
        return new SwitchableIndexInput(this, name, path, delegateInput, length, isSequential(context));
    }

    /**
     * Opens an {@code O_DIRECT} channel on a file and reads the block size of the file store it lives on. Returns
     * null rather than throwing when Direct I/O is not available on this file system or this file, so that the caller
     * serves the original input, and logs one warn line for the file because a file that passed the four checks at
     * open was expected to route.
     *
     * @param path   the file to open
     * @param length the length the original directory's input reports
     * @return the handle, or null if the file cannot be read with {@code O_DIRECT}
     */
    DirectIOVectorIndexInput.Handle openDirect(@NonNull final Path path, final long length) {
        FileChannel channel = null;
        try {
            final int blockSize = Math.toIntExact(Files.getFileStore(path).getBlockSize());
            if (blockSize <= 0 || Integer.bitCount(blockSize) != 1) {
                // A buffer can only be aligned to a power of two, so anything else declines here rather than
                // surfacing later as a content mismatch.
                log.warn(
                    "Direct I/O declined for {} ({}): the file store reports a block size of {}",
                    path,
                    DeclineReason.OPEN_FAILED,
                    blockSize
                );
                return null;
            }
            channel = channelFactory.open(path);
            return new DirectIOVectorIndexInput.Handle(path, channel, blockSize, length);
        } catch (IOException | RuntimeException e) {
            IOUtils.closeWhileHandlingException(channel);
            log.warn("Direct I/O declined for {} ({}): {}", path, DeclineReason.OPEN_FAILED, e.toString());
            return null;
        }
    }

    /**
     * Whether the file reads the same through the {@code O_DIRECT} channel and through the directory chain: the same
     * length, the same first {@link #CONTENT_HEADER_BYTES} and the same last 16 bytes, which is Lucene's codec footer
     * and carries the checksum. This is the sixth check, and it is the reason a byte-transforming directory layer is
     * declined rather than trusted. It cannot catch a transform that changes only interior bytes.
     *
     * @param handle   the open {@code O_DIRECT} channel
     * @param delegate the input the directory chain handed out, which is left where it was
     * @return true if the two agree
     */
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
            log.warn("Direct I/O declined for {} ({}): the content check failed: {}", handle.path(), DeclineReason.CONTENT, e.toString());
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

    /**
     * The buffer size a random reader of one vector at a time uses: two blocks, which hold any vector up to one
     * block even when it straddles a block boundary.
     *
     * @param blockSize the file store's block size
     * @return the buffer size in bytes
     */
    static int randomBufferBytes(final int blockSize) {
        return 2 * blockSize;
    }

    /**
     * The buffer size for an input opened sequentially or at random: Lucene's merge buffer for a merge or a
     * sequential search, two blocks otherwise, always rounded up to whole blocks.
     *
     * @param sequential whether the file was opened for a merge or a sequential read
     * @param blockSize  the file store's block size
     * @return the buffer size in bytes
     */
    static int bufferSizeFor(final boolean sequential, final int blockSize) {
        final int target = sequential ? SEQUENTIAL_BUFFER_BYTES : randomBufferBytes(blockSize);
        return DirectIOVectorIndexInput.roundUpToBlock(Math.max(target, blockSize), blockSize);
    }

    /**
     * Whether a context asks for a sequential read of the whole file.
     *
     * @param context the context
     * @return true for a merge or a sequential hint
     */
    static boolean isSequential(final IOContext context) {
        if (context == null) {
            return false;
        }
        return context.context() == IOContext.Context.MERGE || context.hints().contains(DataAccessHint.SEQUENTIAL);
    }

    private static FileChannel openDirectChannel(final Path path) throws IOException {
        if (DIRECT_OPEN_OPTION == null) {
            throw new IOException("com.sun.nio.file.ExtendedOpenOption.DIRECT is not available on this runtime");
        }
        return FileChannel.open(path, StandardOpenOption.READ, DIRECT_OPEN_OPTION);
    }

    /**
     * Whether the index this segment belongs to has Direct I/O on right now. Read once per slice and per clone of a
     * wrapped {@code .vec}, never per read.
     *
     * @return true if the index is on
     */
    boolean isRouteEnabled() {
        if (routeEnabled != null) {
            return routeEnabled.getAsBoolean();
        }
        return KNNSettings.isDirectIORescoreEnabled(index());
    }

    /**
     * The index this segment's files belong to, found once per segment by matching the shard path's UUID against the
     * cluster state, or null while this node's cluster state has no such index.
     *
     * @return the index, or null
     */
    Index index() {
        Index resolved = index;
        if (resolved != null) {
            return resolved;
        }
        try {
            resolved = KNNSettings.findIndexByUuid(indexUuid);
        } catch (RuntimeException e) {
            // A segment reader must open whatever the cluster state can answer. An index this node cannot name reads
            // as off, which is main's read path.
            log.debug("Could not resolve the index of {} from the cluster state", indexUuid, e);
            return null;
        }
        // A null is not memoed as an answer: a reader built before the index metadata arrived resolves on a later
        // route decision rather than staying off for the life of the segment.
        index = resolved;
        return resolved;
    }

    /**
     * The file size floor this delegate applies.
     *
     * @return the floor in bytes
     */
    long minFileSizeBytes() {
        return minFileSize.getAsLong();
    }

    /**
     * The index UUID this delegate took from the segment's shard path.
     *
     * @return the UUID
     */
    String indexUuid() {
        return indexUuid;
    }

    /**
     * The index UUID carried by a directory's shard path, {@code .../<uuid>/<shard>/index}, which is the layout of
     * both the default data path and a custom {@code index.data_path}. Null for any other shape, including a
     * directory with no path at all, because an index that cannot be named is never routed.
     *
     * @param directory the directory to read, which must be the segment's own
     *                  ({@code SegmentInfo#dir}) and not the possibly compound one the codec was handed
     * @return the UUID, or null
     */
    static String indexUuidOf(final Directory directory) {
        if (directory == null) {
            return null;
        }
        final Directory bottom = FilterDirectory.unwrap(directory);
        if (bottom instanceof FSDirectory == false) {
            return null;
        }
        final Path indexFolder = ((FSDirectory) bottom).getDirectory();
        if (indexFolder == null || INDEX_FOLDER_NAME.equals(fileName(indexFolder)) == false) {
            return null;
        }
        final Path shardFolder = indexFolder.getParent();
        if (shardFolder == null || isShardNumber(fileName(shardFolder)) == false) {
            return null;
        }
        final Path indexUuidFolder = shardFolder.getParent();
        if (indexUuidFolder == null) {
            return null;
        }
        final String uuid = fileName(indexUuidFolder);
        return uuid == null || uuid.isEmpty() ? null : uuid;
    }

    private static String fileName(final Path path) {
        final Path name = path.getFileName();
        return name == null ? null : name.toString();
    }

    private static boolean isShardNumber(final String name) {
        if (name == null || name.isEmpty()) {
            return false;
        }
        for (int i = 0; i < name.length(); i++) {
            if (Character.isDigit(name.charAt(i)) == false) {
                return false;
            }
        }
        return true;
    }

    /**
     * Whether a file name is the vector data file of one of the four faiss formats. Lucene's per-field vector format
     * puts the format's name in the segment suffix of every file it writes, so the name is enough to tell a faiss
     * {@code .vec} from a Lucene-engine one.
     *
     * @param name the file name
     * @return true if the file is faiss vector data
     */
    static boolean isFaissVectorData(final String name) {
        if (name == null || name.endsWith(VECTOR_DATA_EXTENSION) == false) {
            return false;
        }
        return name.contains(NativeEngines990KnnVectorsFormat.FORMAT_NAME)
            || name.contains(Faiss1040ScalarQuantizedKnnVectorsFormat.FORMAT_NAME)
            || name.contains(HALF_FLOAT_FORMAT_NAME)
            || name.contains(HALF_FLOAT_SQ_FORMAT_NAME);
    }

    /**
     * The real path of a file behind a directory chain, found by unwrapping {@link FilterDirectory} layers until the
     * first layer that is not one and taking its directory if it is an
     * {@link org.apache.lucene.store.FSDirectory}. A chain that does not end in one has no path, and the file is not
     * routed. The chain is not judged by the class names in it: a layer that transforms bytes is caught by comparing
     * the bytes, which is the check that can actually be made.
     *
     * @param dir  the directory the file was asked of
     * @param name the file name
     * @return the path, or null if the chain has none
     */
    static Path resolvePath(final Directory dir, final String name) {
        if (dir == null) {
            return null;
        }
        final Directory bottom = FilterDirectory.unwrap(dir);
        return bottom instanceof FSDirectory ? ((FSDirectory) bottom).getDirectory().resolve(name) : null;
    }

    /**
     * The chain of directory classes behind a directory, for a log line.
     *
     * @param dir the directory
     * @return the class names, outermost first
     */
    static String describeChain(final Directory dir) {
        final StringBuilder chain = new StringBuilder();
        Directory current = dir;
        while (current != null) {
            if (chain.length() > 0) {
                chain.append(" -> ");
            }
            chain.append(current.getClass().getSimpleName());
            current = current instanceof FilterDirectory ? ((FilterDirectory) current).getDelegate() : null;
        }
        return chain.toString();
    }
}

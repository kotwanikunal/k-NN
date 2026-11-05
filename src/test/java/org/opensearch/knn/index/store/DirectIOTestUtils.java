/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.FilterDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.store.NIOFSDirectory;
import org.apache.lucene.store.RandomAccessInput;
import org.apache.lucene.tests.mockfile.FilterFileChannel;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.mockito.Mockito;
import org.opensearch.Version;
import org.opensearch.cluster.ClusterName;
import org.opensearch.cluster.ClusterState;
import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.cluster.metadata.Metadata;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.settings.Settings;
import org.opensearch.knn.index.KNNSettings;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.OpenOption;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BooleanSupplier;

import static org.junit.Assume.assumeTrue;

/**
 * The one shared helper of the Direct I/O unit tests, named the way k-NN names
 * {@code PrefetchableScorerTestUtils} and {@code FaissIndexTestUtils}. It holds the file names the four faiss formats
 * produce, a shard-shaped directory in both data path layouts, a codec file with a real header and footer, the
 * directory and input stand-ins the six checks need, a channel factory that counts and can be told to misbehave, and
 * the cached check of whether this JDK and file system can do {@code O_DIRECT} at all.
 */
final class DirectIOTestUtils {

    static final String VEC_NAME = "_0_NativeEngines990KnnVectorsFormat_0.vec";
    static final String HALF_FLOAT_VEC_NAME = "_0_NativeEngines990HalfFloatKnnVectorsFormat_0.vec";
    static final String SQ_VEC_NAME = "_0_Faiss1040ScalarQuantizedKnnVectorsFormat_0.vec";
    static final String HALF_FLOAT_SQ_VEC_NAME = "_0_Faiss1040HalfFloatScalarQuantizedKnnVectorsFormat_0.vec";
    static final String LUCENE_VEC_NAME = "_0_Lucene99HnswVectorsFormat_0.vec";
    static final String META_NAME = "_0_NativeEngines990KnnVectorsFormat_0.vemf";
    static final String QUANTIZED_NAME = "_0_Faiss1040ScalarQuantizedKnnVectorsFormat_0.veq";
    static final String FAISS_NAME = "_0_NativeEngines990KnnVectorsFormat_0.faiss";

    static final String INDEX_UUID = "V4qM8m9ZTAyRm0Nn0lOAbw";
    static final String CODEC_NAME = "DirectIOTestFile";
    static final int CODEC_VERSION = 1;

    /** A floor low enough that a test file of a few KiB is routed. */
    static final long SMALL_FLOOR = 1024L;

    /** The answer of {@link #isDirectIOSupported()}, decided once per test JVM. */
    private static volatile Boolean directIOSupported;

    private DirectIOTestUtils() {}

    /**
     * The default data path layout, {@code <root>/indices/<uuid>/0/index}.
     *
     * @param root the test's temporary directory
     * @param uuid the index UUID
     * @return the shard's index directory, created
     * @throws IOException if the directories cannot be created
     */
    static Path defaultLayoutShardPath(final Path root, final String uuid) throws IOException {
        return Files.createDirectories(root.resolve("nodes").resolve("0").resolve("indices").resolve(uuid).resolve("0").resolve("index"));
    }

    /**
     * The custom {@code index.data_path} layout, which has no {@code indices} component:
     * {@code <shared_data>/<data_path>/<lockId>/<uuid>/0/index}.
     *
     * @param root the test's temporary directory
     * @param uuid the index UUID
     * @return the shard's index directory, created
     * @throws IOException if the directories cannot be created
     */
    static Path customLayoutShardPath(final Path root, final String uuid) throws IOException {
        return Files.createDirectories(
            root.resolve("shared").resolve("my_data_path").resolve("0").resolve(uuid).resolve("0").resolve("index")
        );
    }

    /**
     * Writes a file with a Lucene codec header, a deterministic payload and a codec footer, which is the shape every
     * check and every integrity read in these tests needs.
     *
     * @param dir          the directory to write into
     * @param name         the file name
     * @param payloadBytes how many payload bytes to write between the header and the footer
     * @return the bytes of the whole file
     * @throws IOException if the write fails
     */
    static byte[] writeCodecFile(final Directory dir, final String name, final int payloadBytes) throws IOException {
        final byte[] id = new byte[16];
        new Random(payloadBytes).nextBytes(id);
        try (IndexOutput out = dir.createOutput(name, IOContext.DEFAULT)) {
            CodecUtil.writeIndexHeader(out, CODEC_NAME, CODEC_VERSION, id, "0");
            final byte[] payload = payload(payloadBytes);
            out.writeBytes(payload, payload.length);
            CodecUtil.writeFooter(out);
        }
        return readAll(dir, name);
    }

    /**
     * A deterministic payload, so that a Direct I/O read and a delegate read can be compared byte for byte.
     *
     * @param bytes how many bytes
     * @return the payload
     */
    static byte[] payload(final int bytes) {
        final byte[] payload = new byte[bytes];
        new Random(42).nextBytes(payload);
        return payload;
    }

    /**
     * Reads a whole file through the directory.
     *
     * @param dir  the directory
     * @param name the file name
     * @return the bytes
     * @throws IOException if the read fails
     */
    static byte[] readAll(final Directory dir, final String name) throws IOException {
        try (IndexInput in = dir.openInput(name, IOContext.DEFAULT)) {
            final byte[] bytes = new byte[(int) in.length()];
            in.readBytes(bytes, 0, bytes.length);
            return bytes;
        }
    }

    /**
     * Opens an {@link FSDirectory} on a path, which is what every routing test needs under the delegate.
     *
     * @param path the directory
     * @return the directory
     * @throws IOException if the open fails
     */
    static Directory fsDirectory(final Path path) throws IOException {
        return FSDirectory.open(path);
    }

    /**
     * A delegate over a real directory with the test seams set: a low file size floor, a switch the test drives, and
     * the production channel factory and open option.
     *
     * @param in    the directory to wrap
     * @param route the switch
     * @return the delegate
     */
    static KNNDirectIODirectory directory(final Directory in, final BooleanSupplier route) {
        return directory(in, route, KNNDirectIODirectory.DEFAULT_CHANNEL_FACTORY);
    }

    /**
     * A delegate whose {@code O_DIRECT} channels the test opens itself.
     *
     * @param in      the directory to wrap
     * @param route   the switch
     * @param factory how the {@code O_DIRECT} channel is opened
     * @return the delegate
     */
    static KNNDirectIODirectory directory(
        final Directory in,
        final BooleanSupplier route,
        final KNNDirectIODirectory.ChannelFactory factory
    ) {
        return directory(in, route, factory, KNNDirectIODirectory.DIRECT_OPEN_OPTION);
    }

    /**
     * A delegate with all four test seams set, including the {@code DIRECT} open option, which a test passes as null
     * to behave as a JDK that does not have it.
     *
     * @param in           the directory to wrap
     * @param route        the switch
     * @param factory      how the {@code O_DIRECT} channel is opened
     * @param directOption the open option, or null
     * @return the delegate
     */
    static KNNDirectIODirectory directory(
        final Directory in,
        final BooleanSupplier route,
        final KNNDirectIODirectory.ChannelFactory factory,
        final OpenOption directOption
    ) {
        return new KNNDirectIODirectory(in, INDEX_UUID, () -> SMALL_FLOOR, route, factory, directOption);
    }

    /**
     * Publishes one index into a mocked cluster service's state with {@code index.knn.direct_io.rescore.enabled}
     * set, which is what the {@link org.opensearch.knn.index.KNNSettings} accessor reads. The counter the caller
     * passes is incremented on every read of the state, so a test can assert how often the setting was looked up.
     *
     * @param clusterService the mock from {@code KNNTestCase}
     * @param indexName      the index name
     * @param uuid           the index UUID
     * @param enabled        the value of the setting
     * @param stateReads     incremented on each read of the cluster state, or null
     */
    static void publishIndex(
        final ClusterService clusterService,
        final String indexName,
        final String uuid,
        final boolean enabled,
        final AtomicInteger stateReads
    ) {
        final Settings settings = Settings.builder()
            .put(IndexMetadata.SETTING_VERSION_CREATED, Version.CURRENT.id)
            .put(IndexMetadata.SETTING_INDEX_UUID, uuid)
            .put(KNNSettings.INDEX_KNN_DIRECT_IO_RESCORE_ENABLED, enabled)
            .build();
        final IndexMetadata metadata = IndexMetadata.builder(indexName).settings(settings).numberOfShards(1).numberOfReplicas(0).build();
        final ClusterState clusterState = ClusterState.builder(new ClusterName("test"))
            .metadata(Metadata.builder().put(metadata, false).build())
            .build();
        KNNSettings.state().setClusterService(clusterService);
        Mockito.doAnswer(invocation -> {
            if (stateReads != null) {
                stateReads.incrementAndGet();
            }
            return clusterState;
        }).when(clusterService).state();
    }

    /**
     * Whether this JDK and this file system can open a file with {@code O_DIRECT}, decided once per test JVM in the
     * two steps of Lucene's {@code TestDirectIODirectory.checkSupported}: the production open option is not null, then
     * one probe file opened through the production channel factory. Every test directory comes from
     * {@code createTempDir} under one temp root, so one probe answers for all of them. On the development desktop this
     * must answer true.
     *
     * @return true if {@code O_DIRECT} is available
     */
    static boolean isDirectIOSupported() {
        Boolean answer = directIOSupported;
        if (answer != null) {
            return answer;
        }
        synchronized (DirectIOTestUtils.class) {
            if (directIOSupported != null) {
                return directIOSupported;
            }
            directIOSupported = probeDirectIO();
            return directIOSupported;
        }
    }

    /**
     * Skips the calling test where {@code O_DIRECT} cannot be opened, which is the construct k-NN uses for a platform
     * capability. On the development desktop no Direct I/O test may skip.
     */
    static void assumeDirectIOSupported() {
        assumeTrue("Requires O_DIRECT on this JDK and file system", isDirectIOSupported());
    }

    private static boolean probeDirectIO() {
        if (KNNDirectIODirectory.DIRECT_OPEN_OPTION == null) {
            return false;
        }
        try {
            final Path probe = Files.createTempFile(LuceneTestCase.createTempDir("direct-io-probe"), "dio-probe", ".bin");
            try {
                Files.write(probe, payload(8192));
                KNNDirectIODirectory.DEFAULT_CHANNEL_FACTORY.open(probe).close();
                return true;
            } finally {
                Files.deleteIfExists(probe);
            }
        } catch (IOException | RuntimeException e) {
            return false;
        }
    }

    /**
     * A channel factory that hands out {@link TestChannel}s over real channels, keeps every one it opened, and can be
     * told to make the reads that follow misbehave. It is how the unit tests see the device reads this plugin makes
     * now that there are no counters, and how they reach the two failure cases the read fallback is about.
     */
    static final class TestChannels implements KNNDirectIODirectory.ChannelFactory {

        private final List<TestChannel> channels = new ArrayList<>();
        private final boolean withPlainChannel;
        private volatile Behaviour behaviour = Behaviour.NORMAL;

        TestChannels() {
            this(false);
        }

        /**
         * @param withPlainChannel whether each channel also holds a channel on the same file opened without
         *                         {@code O_DIRECT}, which is the only way a {@link Behaviour#SHORT} read can return
         *                         real bytes: a 100 byte read on an {@code O_DIRECT} channel is refused with
         *                         {@code EINVAL} because the length is not a multiple of the block size
         */
        TestChannels(final boolean withPlainChannel) {
            this.withPlainChannel = withPlainChannel;
        }

        @Override
        public FileChannel open(final Path path) throws IOException {
            final OpenOption direct = KNNDirectIODirectory.DIRECT_OPEN_OPTION;
            final FileChannel real = direct == null
                ? FileChannel.open(path, StandardOpenOption.READ)
                : FileChannel.open(path, StandardOpenOption.READ, direct);
            final FileChannel plain = withPlainChannel ? FileChannel.open(path, StandardOpenOption.READ) : null;
            final TestChannel channel = new TestChannel(real, plain, this);
            synchronized (channels) {
                channels.add(channel);
            }
            return channel;
        }

        /**
         * Sets what every channel of this factory does on the reads that follow.
         *
         * @param behaviour the behaviour
         */
        void behave(final Behaviour behaviour) {
            this.behaviour = behaviour;
        }

        /**
         * Every channel this factory opened, in order.
         *
         * @return the channels
         */
        List<TestChannel> channels() {
            synchronized (channels) {
                return List.copyOf(channels);
            }
        }

        /**
         * The channel this factory opened last.
         *
         * @return the channel
         */
        TestChannel last() {
            synchronized (channels) {
                return channels.get(channels.size() - 1);
            }
        }

        /**
         * How many positional reads every channel of this factory has been asked for, which is the device reads this
         * plugin made.
         *
         * @return the count
         */
        int reads() {
            int total = 0;
            for (final TestChannel channel : channels()) {
                total += channel.reads().size();
            }
            return total;
        }

        /**
         * How many bytes those positional reads asked for.
         *
         * @return the byte count
         */
        long readBytes() {
            long total = 0;
            for (final TestChannel channel : channels()) {
                for (final Read read : channel.reads()) {
                    total += read.length();
                }
            }
            return total;
        }
    }

    /** What a {@link TestChannel} does on the next positional read. */
    enum Behaviour {
        /** Read normally. */
        NORMAL,
        /**
         * Return fewer bytes than one block, which leaves the requested offset outside what the read covered. A short
         * read that still covers the request is not a failure and the input reads on, which is why this has to be
         * smaller than a block.
         */
        SHORT,
        /** Throw an {@link IOException}. */
        THROW,
        /** Write one block of garbage into the buffer, then throw, which is a read that half happened. */
        GARBAGE_THEN_THROW
    }

    /** One positional read a {@link TestChannel} was asked to make. */
    record Read(long position, int length) {
    }

    /**
     * A channel that records every positional read, counts its closes, and misbehaves on demand. Lucene's
     * {@link FilterFileChannel} forwards every other {@link FileChannel} method, so only the positional read and the
     * close are written here.
     */
    static final class TestChannel extends FilterFileChannel {

        /** How many bytes a {@link Behaviour#SHORT} read returns, which is under any file store's block size. */
        static final int SHORT_READ_BYTES = 100;

        private final FileChannel plain;
        private final TestChannels owner;
        private final List<Read> reads = new ArrayList<>();
        private final AtomicInteger closes = new AtomicInteger();

        TestChannel(final FileChannel delegate, final FileChannel plain, final TestChannels owner) {
            super(delegate);
            this.plain = plain;
            this.owner = owner;
        }

        /**
         * Every positional read this channel was asked to make, in order.
         *
         * @return the reads
         */
        List<Read> reads() {
            synchronized (reads) {
                return List.copyOf(reads);
            }
        }

        /**
         * How many times this channel was closed.
         *
         * @return the count
         */
        int closes() {
            return closes.get();
        }

        @Override
        public int read(final ByteBuffer dst, final long position) throws IOException {
            synchronized (reads) {
                reads.add(new Read(position, dst.remaining()));
            }
            switch (owner.behaviour) {
                case THROW:
                    throw new IOException("injected read failure at " + position);
                case GARBAGE_THEN_THROW: {
                    final int garbage = Math.min(dst.remaining(), 4096);
                    for (int i = 0; i < garbage; i++) {
                        dst.put((byte) 0xFF);
                    }
                    throw new IOException("injected half-done read at " + position);
                }
                case SHORT: {
                    final int limit = Math.min(dst.remaining(), SHORT_READ_BYTES);
                    final ByteBuffer slice = dst.slice();
                    slice.limit(limit);
                    final int read = (plain == null ? delegate : plain).read(slice, position);
                    if (read > 0) {
                        dst.position(dst.position() + read);
                    }
                    return read;
                }
                default:
                    return delegate.read(dst, position);
            }
        }

        @Override
        protected void implCloseChannel() throws IOException {
            closes.incrementAndGet();
            try {
                if (plain != null) {
                    plain.close();
                }
            } finally {
                super.implCloseChannel();
            }
        }
    }

    /** The byte a transforming layer flips, chosen so that no region of a codec file is left unchanged. */
    static final byte XOR_MASK = (byte) 0x5A;

    /**
     * A directory shaped like encryption at rest: it extends {@link NIOFSDirectory}, so the directory chain still
     * ends in a file-system directory and the file still has a real path, but every byte it serves is transformed.
     * The routing rule must decline it on the content check, because the bytes an {@code O_DIRECT} read of that path
     * returns are not the bytes a reader sees.
     */
    static final class CryptoShapedFsDirectory extends NIOFSDirectory {

        CryptoShapedFsDirectory(final Path path) throws IOException {
            super(path);
        }

        @Override
        public IndexInput openInput(final String name, final IOContext context) throws IOException {
            return new XoringIndexInput(super.openInput(name, context));
        }
    }

    /**
     * A directory that extends {@link NIOFSDirectory} and serves one name out of another file, so that the name the
     * delegate opens successfully resolves to a path with no file on it. That is the shape of the {@code MISSING}
     * check: a non-default data format can place a file outside the shard's index directory.
     */
    static final class GhostNameFsDirectory extends NIOFSDirectory {

        private final String ghostName;
        private final String backingName;

        GhostNameFsDirectory(final Path path, final String ghostName, final String backingName) throws IOException {
            super(path);
            this.ghostName = ghostName;
            this.backingName = backingName;
        }

        @Override
        public IndexInput openInput(final String name, final IOContext context) throws IOException {
            return super.openInput(ghostName.equals(name) ? backingName : name, context);
        }
    }

    /**
     * A {@link org.apache.lucene.codecs.CompoundDirectory} over a real directory, which is the shape Lucene hands a
     * codec for a compound segment. It is not a {@link FilterDirectory} and not an
     * {@link org.apache.lucene.store.FSDirectory}, so the files behind it have no path of their own.
     */
    static final class TestCompoundDirectory extends org.apache.lucene.codecs.CompoundDirectory {

        private final Directory in;

        TestCompoundDirectory(final Directory in) {
            this.in = in;
        }

        @Override
        public void checkIntegrity() throws IOException {
            // nothing to check in a test stand-in
        }

        @Override
        public String[] listAll() throws IOException {
            return in.listAll();
        }

        @Override
        public long fileLength(final String name) throws IOException {
            return in.fileLength(name);
        }

        @Override
        public IndexInput openInput(final String name, final IOContext context) throws IOException {
            return in.openInput(name, context);
        }

        @Override
        public Set<String> getPendingDeletions() throws IOException {
            return Set.of();
        }

        @Override
        public void close() throws IOException {
            // the delegate belongs to the test
        }
    }

    /**
     * A plain pass-through layer, which must be crossed and routed: it changes no byte, and the content check is what
     * proves that.
     */
    static final class PassThroughDirectory extends FilterDirectory {

        PassThroughDirectory(final Directory in) {
            super(in);
        }
    }

    /**
     * A layer that counts how often the inputs it handed out were closed, which is how the close rules are asserted.
     */
    static final class CloseCountingDirectory extends FilterDirectory {

        private final AtomicInteger closes = new AtomicInteger();
        private final AtomicInteger opens = new AtomicInteger();

        CloseCountingDirectory(final Directory in) {
            super(in);
        }

        @Override
        public IndexInput openInput(final String name, final IOContext context) throws IOException {
            opens.incrementAndGet();
            return new CloseCountingIndexInput(super.openInput(name, context), closes);
        }

        /**
         * How many of the inputs this directory handed out were closed.
         *
         * @return the count
         */
        int closes() {
            return closes.get();
        }

        /**
         * How many inputs this directory handed out.
         *
         * @return the count
         */
        int opens() {
            return opens.get();
        }
    }

    /**
     * An input that transforms every byte it serves. Its clones and slices transform too, which is what makes it a
     * faithful stand-in for a layer that encrypts at rest.
     */
    static final class XoringIndexInput extends IndexInput {

        private final IndexInput in;

        XoringIndexInput(final IndexInput in) {
            super("xoring(" + in + ")");
            this.in = in;
        }

        @Override
        public byte readByte() throws IOException {
            return (byte) (in.readByte() ^ XOR_MASK);
        }

        @Override
        public void readBytes(final byte[] b, final int offset, final int len) throws IOException {
            in.readBytes(b, offset, len);
            for (int i = offset; i < offset + len; i++) {
                b[i] = (byte) (b[i] ^ XOR_MASK);
            }
        }

        @Override
        public void close() throws IOException {
            in.close();
        }

        @Override
        public long getFilePointer() {
            return in.getFilePointer();
        }

        @Override
        public void seek(final long pos) throws IOException {
            in.seek(pos);
        }

        @Override
        public long length() {
            return in.length();
        }

        @Override
        public IndexInput slice(final String sliceDescription, final long offset, final long length) throws IOException {
            return new XoringIndexInput(in.slice(sliceDescription, offset, length));
        }

        @Override
        public IndexInput clone() {
            return new XoringIndexInput(in.clone());
        }
    }

    /**
     * An input that counts its closes on a shared counter, and whose slices and clones count on the same one so that
     * a test can prove a clone closes nothing.
     */
    static final class CloseCountingIndexInput extends IndexInput {

        private final IndexInput in;
        private final AtomicInteger closes;

        CloseCountingIndexInput(final IndexInput in, final AtomicInteger closes) {
            super("closeCounting(" + in + ")");
            this.in = in;
            this.closes = closes;
        }

        @Override
        public byte readByte() throws IOException {
            return in.readByte();
        }

        @Override
        public void readBytes(final byte[] b, final int offset, final int len) throws IOException {
            in.readBytes(b, offset, len);
        }

        @Override
        public void readFloats(final float[] dst, final int offset, final int len) throws IOException {
            in.readFloats(dst, offset, len);
        }

        @Override
        public void readInts(final int[] dst, final int offset, final int len) throws IOException {
            in.readInts(dst, offset, len);
        }

        @Override
        public void readLongs(final long[] dst, final int offset, final int len) throws IOException {
            in.readLongs(dst, offset, len);
        }

        @Override
        public void close() throws IOException {
            closes.incrementAndGet();
            in.close();
        }

        @Override
        public long getFilePointer() {
            return in.getFilePointer();
        }

        @Override
        public void seek(final long pos) throws IOException {
            in.seek(pos);
        }

        @Override
        public long length() {
            return in.length();
        }

        @Override
        public IndexInput slice(final String sliceDescription, final long offset, final long length) throws IOException {
            return new CloseCountingIndexInput(in.slice(sliceDescription, offset, length), closes);
        }

        @Override
        public IndexInput clone() {
            return new CloseCountingIndexInput(in.clone(), closes);
        }

        @Override
        public RandomAccessInput randomAccessSlice(final long offset, final long length) throws IOException {
            return in.randomAccessSlice(offset, length);
        }

        @Override
        public void prefetch(final long offset, final long length) throws IOException {
            in.prefetch(offset, length);
        }

        @Override
        public Optional<Boolean> isLoaded() {
            return in.isLoaded();
        }

        @Override
        public boolean isEfficientFor(final IOContext context) {
            return in.isEfficientFor(context);
        }
    }

    /**
     * An input that records the prefetch calls it was given and whether its cursor moved, for the hint rules.
     */
    static final class PrefetchRecorder {

        private final AtomicInteger prefetches = new AtomicInteger();
        private volatile long lastOffset = -1;
        private volatile long lastLength = -1;

        /**
         * How many prefetch hints the inputs of this recorder were given.
         *
         * @return the count
         */
        int prefetches() {
            return prefetches.get();
        }

        /**
         * The offset of the last prefetch hint, or -1.
         *
         * @return the offset
         */
        long lastOffset() {
            return lastOffset;
        }

        /**
         * The length of the last prefetch hint, or -1.
         *
         * @return the length
         */
        long lastLength() {
            return lastLength;
        }
    }

    /** An input that reports every prefetch hint it is given to a shared recorder rather than carrying it out. */
    static final class PrefetchRecordingIndexInput extends IndexInput {

        private final IndexInput in;
        private final PrefetchRecorder recorder;

        PrefetchRecordingIndexInput(final IndexInput in, final PrefetchRecorder recorder) {
            super("prefetchRecording(" + in + ")");
            this.in = in;
            this.recorder = recorder;
        }

        @Override
        public void prefetch(final long offset, final long length) throws IOException {
            recorder.prefetches.incrementAndGet();
            recorder.lastOffset = offset;
            recorder.lastLength = length;
        }

        @Override
        public byte readByte() throws IOException {
            return in.readByte();
        }

        @Override
        public void readBytes(final byte[] b, final int offset, final int len) throws IOException {
            in.readBytes(b, offset, len);
        }

        @Override
        public void close() throws IOException {
            in.close();
        }

        @Override
        public long getFilePointer() {
            return in.getFilePointer();
        }

        @Override
        public void seek(final long pos) throws IOException {
            in.seek(pos);
        }

        @Override
        public long length() {
            return in.length();
        }

        @Override
        public IndexInput slice(final String sliceDescription, final long offset, final long length) throws IOException {
            return new PrefetchRecordingIndexInput(in.slice(sliceDescription, offset, length), recorder);
        }

        @Override
        public IndexInput clone() {
            return new PrefetchRecordingIndexInput(in.clone(), recorder);
        }
    }

    /**
     * A directory that hands out {@link PrefetchRecordingIndexInput}s over one shared recorder, so that a test sees
     * the hints an input, a slice and a clone made without having to hold each object.
     */
    static final class PrefetchRecordingDirectory extends FilterDirectory {

        private final PrefetchRecorder recorder = new PrefetchRecorder();

        PrefetchRecordingDirectory(final Directory in) {
            super(in);
        }

        @Override
        public IndexInput openInput(final String name, final IOContext context) throws IOException {
            return new PrefetchRecordingIndexInput(super.openInput(name, context), recorder);
        }

        /**
         * The recorder every input of this directory writes to.
         *
         * @return the recorder
         */
        PrefetchRecorder recorder() {
            return recorder;
        }
    }
}

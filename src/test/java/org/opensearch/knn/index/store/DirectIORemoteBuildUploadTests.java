/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import com.carrotsearch.randomizedtesting.annotations.ThreadLeakFilters;
import lombok.SneakyThrows;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.FieldType;
import org.apache.lucene.document.KnnByteVectorField;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FieldInfo;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.MMapDirectory;
import org.opensearch.common.StreamContext;
import org.opensearch.common.blobstore.AsyncMultiStreamBlobContainer;
import org.opensearch.common.blobstore.BlobPath;
import org.opensearch.common.blobstore.DeleteResult;
import org.opensearch.common.blobstore.fs.FsBlobContainer;
import org.opensearch.common.blobstore.fs.FsBlobStore;
import org.opensearch.common.blobstore.stream.read.ReadContext;
import org.opensearch.common.blobstore.stream.write.WriteContext;
import org.opensearch.common.io.InputStreamContainer;
import org.opensearch.core.action.ActionListener;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.KNN990Codec.NativeEngines990HalfFloatKnnVectorsFormat;
import org.opensearch.knn.index.codec.KNN990Codec.NativeEngines990KnnVectorsFormat;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexBuildStrategyFactory;
import org.opensearch.knn.index.codec.nativeindex.remote.DefaultVectorRepositoryAccessor;
import org.opensearch.knn.index.codec.util.UnitTestCodec;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;
import org.opensearch.knn.index.vectorvalues.KNNVectorValues;
import org.opensearch.knn.plugin.stats.KNNRemoteIndexBuildValue;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentWriteState;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.util.InfoStream;
import org.apache.lucene.util.Version;
import org.mockito.MockedStatic;
import org.opensearch.cluster.ClusterName;
import org.opensearch.cluster.metadata.RepositoryMetadata;
import org.opensearch.common.blobstore.BlobStore;
import org.opensearch.common.settings.ClusterSettings;
import org.opensearch.common.settings.Setting;
import org.opensearch.common.settings.Settings;
import org.opensearch.index.IndexSettings;
import org.opensearch.knn.index.codec.nativeindex.NativeIndexBuildStrategy;
import org.opensearch.knn.index.codec.nativeindex.model.BuildIndexParams;
import org.opensearch.knn.index.codec.nativeindex.remote.RemoteIndexBuildStrategy;
import org.opensearch.knn.index.engine.KNNLibraryIndexingContext;
import org.opensearch.knn.index.engine.ResolvedIndexSpec;
import org.opensearch.remoteindexbuild.client.RemoteIndexClient;
import org.opensearch.remoteindexbuild.client.RemoteIndexClientFactory;
import org.opensearch.repositories.RepositoriesService;
import org.opensearch.repositories.blobstore.BlobStoreRepository;
import org.opensearch.knn.index.vectorvalues.KNNVectorValuesFactory;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Base64;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Locale;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Supplier;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.mockStatic;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
import static org.opensearch.knn.index.codec.util.KNNCodecUtil.initializeVectorValues;

/**
 * W7's codec-level gates: what a GPU remote index build uploads when the source {@code .vec} is served with
 * {@code O_DIRECT}. The build itself is off this desktop (there is no remote build endpoint), so what is
 * measured here is the half of the path that runs on the node: the vector and doc-id blobs the uploader
 * produces, and the reads it takes to produce them.
 *
 * <p><b>Why the upload is not the merge read again.</b> A merge reads each source once, forward, through one
 * input. The upload reads them differently: with a multi-stream repository the blob is cut into
 * parts, every part takes its <em>own</em> {@code KNNVectorValues} from the supplier and so its own clone of the
 * segment's input, and the parts run concurrently on the repository's threads. So one segment is read by several
 * routed inputs at once, each positioning itself with {@code nextDoc} and then scanning forward from its own
 * start. Nothing else in the plugin reads a {@code .vec} that way, which is why the read-ahead ring's behaviour
 * under it is gate 3 rather than an assumption.
 *
 * <p>The blobs are compared as real files, by content and not only between the two routes: each is checked
 * against the segment's own vectors and document ids, read back through a plain memory-mapped reader and
 * encoded the way the uploader encodes them. The repository writes to disk, one part at a time at the offset
 * it was given, and counts the bytes each part produced, so a part that read nothing fails the upload instead
 * of leaving a region of zeros two arms would agree on.
 */
@ThreadLeakFilters(defaultFilters = true, filters = { DirectIOReadPoolTests.ReadPoolThreadFilter.class })
public class DirectIORemoteBuildUploadTests extends KNNTestCase {

    private static final String FIELD = "v";
    private static final int DIMENSION = 128;
    /** One fp32 vector is 512 B, so the identity fixture is a 256 KiB {@code .vec}: four 64 KiB parts. */
    private static final int UPLOAD_DOCS = 512;
    /** Per source segment for the ring gate: 1 MiB of fp32 vectors each, 16 ring windows. */
    private static final int RING_DOCS_PER_SOURCE = 2048;
    /** The ring's shipped window, which the arming bound is stated in multiples of. */
    private static final int WINDOW_BYTES = DirectIOVectorIndexInput.DEFAULT_READAHEAD_WINDOW_BYTES;
    /** Threads the fake repository uploads parts on, standing in for its transfer pool. */
    private static final int PART_CONCURRENCY = 4;

    private static final String VECTOR_BLOB = "test_blob" + KNNConstants.VECTOR_BLOB_FILE_EXTENSION;
    private static final String DOC_ID_BLOB = "test_blob" + KNNConstants.DOC_ID_FILE_EXTENSION;
    private static final String BLOB_NAME = "test_blob";

    /**
     * The plugin's fp32 native-engine format with the graph build skipped, as {@link DirectIOMergeIntegrityTests}
     * uses it: a negative approximate threshold makes {@code shouldSkipBuildingVectorDataStructure} answer true
     * for any doc count, so the segment holds the {@code .vec} and its {@code .vemf} sidecar and no
     * {@code .faiss}. The upload reads the {@code .vec}, which is what this measures, and the fixture stays off
     * the JNI layer.
     */
    private static final Codec FLOAT_CODEC = new UnitTestCodec(() -> new NativeEngines990KnnVectorsFormat(-1));

    /** The same, for the FP16 per-field format, whose name the dispatch rule recognises separately. */
    private static final Codec HALF_FLOAT_CODEC = new UnitTestCodec(
        () -> new NativeEngines990HalfFloatKnnVectorsFormat(-1, new NativeIndexBuildStrategyFactory())
    );

    @Override
    public void setUp() throws Exception {
        super.setUp();
        // Everything this class reads is a static that some other class in this package may have left
        // elsewhere: the ring's window count at 0, the store configured, the budget moved. Published here as
        // well as restored in tearDown, so the suite's order cannot decide this class's numbers, and the same
        // list in both places so neither side can drift.
        resetDirectIOStatics();
    }

    @Override
    public void tearDown() throws Exception {
        resetDirectIOStatics();
        super.tearDown();
    }

    /**
     * The process-wide state this class reads or moves, back to what a node starts with. The remote build
     * values are in the list because the fallback gate drives the real build strategy, whose metrics are
     * JVM-wide counters another suite asserts read 0 before anything has built.
     */
    private static void resetDirectIOStatics() {
        for (final KNNRemoteIndexBuildValue value : KNNRemoteIndexBuildValue.values()) {
            value.reset();
        }
        DirectIOVectorCache.resetForTesting();
        DirectIOReadPool.resetForTesting();
        DirectIOVectorIndexInput.resetBudgetsForTesting();
        DirectIOVectorIndexInput.publishReadaheadWindow(KNNSettings.KNN_DIRECT_IO_READAHEAD_WINDOW_DEFAULT_VALUE);
        DirectIOVectorIndexInput.STATS.reset();
    }

    @SneakyThrows
    private void assumeDirectIOWorksHere() {
        final Path probe = createTempDir().resolve("probe");
        Files.write(probe, new byte[8192]);
        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(probe)) {
            assumeTrue("O_DIRECT is not available here", input != null);
        }
    }

    // -----------------------------------------------------------------------------------------------
    // The fixture: one segment per data type, written non-compound with the graph build skipped
    // -----------------------------------------------------------------------------------------------

    /** A native-engine vector field of {@code dataType}, so the per-field format the rule looks for is faiss's. */
    private static FieldType vectorField(final VectorDataType dataType) {
        final FieldType fieldType = new FieldType();
        fieldType.setTokenized(false);
        fieldType.setIndexOptions(IndexOptions.NONE);
        fieldType.putAttribute(KNNVectorFieldMapper.KNN_FIELD, "true");
        fieldType.putAttribute(KNNConstants.KNN_METHOD, KNNConstants.METHOD_HNSW);
        fieldType.putAttribute(KNNConstants.KNN_ENGINE, KNNEngine.FAISS.getName());
        fieldType.putAttribute(
            KNNConstants.SPACE_TYPE,
            dataType == VectorDataType.BINARY ? SpaceType.HAMMING.getValue() : SpaceType.L2.getValue()
        );
        fieldType.putAttribute(KNNConstants.HNSW_ALGO_M, "32");
        fieldType.putAttribute(KNNConstants.HNSW_ALGO_EF_CONSTRUCTION, "512");
        fieldType.putAttribute(KNNConstants.VECTOR_DATA_TYPE_FIELD, dataType.getValue());
        // BYTE and BINARY are Lucene byte-encoded, and for BINARY the field's dimension is the bit count
        // while the stored vector is one byte per eight bits, which is why the encoding and the dimension
        // are taken from the data type rather than fixed.
        if (dataType == VectorDataType.BYTE || dataType == VectorDataType.BINARY) {
            fieldType.setVectorAttributes(
                dataType == VectorDataType.BINARY ? DIMENSION / 8 : DIMENSION,
                VectorEncoding.BYTE,
                VectorSimilarityFunction.EUCLIDEAN
            );
        } else {
            fieldType.setVectorAttributes(DIMENSION, VectorEncoding.FLOAT32, VectorSimilarityFunction.EUCLIDEAN);
        }
        fieldType.freeze();
        return fieldType;
    }

    private static Codec codecFor(final VectorDataType dataType) {
        return dataType == VectorDataType.HALF_FLOAT ? HALF_FLOAT_CODEC : FLOAT_CODEC;
    }

    private static IndexWriterConfig config(final VectorDataType dataType) {
        final IndexWriterConfig config = new IndexWriterConfig();
        config.setCodec(codecFor(dataType));
        // Non-compound, or the .vec is inside a .cfs and nothing routes (the dispatch rule's condition 3).
        config.setUseCompoundFile(false);
        config.setMergePolicy(NoMergePolicy.INSTANCE);
        return config;
    }

    /** One commit of {@code docs} documents of {@code dataType}, with merges off so each call is its own segment. */
    private static void writeSegment(final MMapDirectory directory, final VectorDataType dataType, final int docs, final long seed)
        throws IOException {
        final FieldType fieldType = vectorField(dataType);
        final Random random = new Random(seed);
        try (IndexWriter writer = new IndexWriter(directory, config(dataType))) {
            for (int i = 0; i < docs; i++) {
                final Document document = new Document();
                if (dataType == VectorDataType.BYTE || dataType == VectorDataType.BINARY) {
                    final byte[] bytes = new byte[dataType == VectorDataType.BINARY ? DIMENSION / 8 : DIMENSION];
                    random.nextBytes(bytes);
                    document.add(new KnnByteVectorField(FIELD, bytes, fieldType));
                } else {
                    final float[] values = new float[DIMENSION];
                    for (int d = 0; d < DIMENSION; d++) {
                        values[d] = random.nextFloat() * 20 - 10;
                    }
                    document.add(new KnnFloatVectorField(FIELD, values, fieldType));
                }
                writer.addDocument(document);
            }
            writer.commit();
        }
    }

    private static FieldInfo onlyFieldInfo(final LeafReader leaf) {
        final FieldInfo fieldInfo = leaf.getFieldInfos().fieldInfo(FIELD);
        assertNotNull("the fixture must hold the vector field", fieldInfo);
        return fieldInfo;
    }

    @SneakyThrows
    private static String sha256(final Path file) {
        return sha256(Files.readAllBytes(file));
    }

    @SneakyThrows
    private static String sha256(final byte[] bytes) {
        final MessageDigest digest = MessageDigest.getInstance("SHA-256");
        final byte[] sum = digest.digest(bytes);
        final StringBuilder hex = new StringBuilder(sum.length * 2);
        for (final byte b : sum) {
            hex.append(String.format(Locale.ROOT, "%02x", b));
        }
        return hex.toString();
    }

    // -----------------------------------------------------------------------------------------------
    // The repository: a filesystem container that really writes, and really uploads in parts
    // -----------------------------------------------------------------------------------------------

    /**
     * A filesystem blob container that takes the multi-stream path and writes the bytes. Every part is asked
     * for on its own thread, as a repository's transfer pool does, so each part's
     * {@code KNNVectorValues} is created and read from a thread of its own; a loop on the caller thread would
     * not exercise the one thing about the upload that is unlike every other reader of a {@code .vec}.
     *
     * <p>Three properties of this fake are load-bearing and each is checked rather than assumed.
     * {@code asyncBlobUpload} returns before the parts finish, as its contract says it does, so the caller's
     * doc-id write overlaps them and its own wait is a real one. The parts rendezvous before their first read,
     * so the concurrency the ring is measured under is known to have happened rather than hoped for; the peak
     * number in flight is recorded for the caller to assert. And the bytes every part produced are counted and
     * compared against the blob's declared length, because the file is pre-sized: without that count a part
     * that read nothing would leave zeros that no length check could see.
     *
     * <p>{@code partSizeBytes} is the test's, not a repository's: the real part size comes from the vendor
     * implementation, and fixing it here is what lets a test state the part layout it is asserting about.
     */
    private static final class RecordingMultiStreamBlobContainer extends FsBlobContainer implements AsyncMultiStreamBlobContainer {

        private final Path blobDirectory;
        private final long partSizeBytes;
        private final AtomicInteger parts = new AtomicInteger();
        private final AtomicInteger vectorBlobWriteBlobCalls = new AtomicInteger();
        private final AtomicLong partBytesWritten = new AtomicLong();
        private final AtomicInteger inFlight = new AtomicInteger();
        private final AtomicInteger inFlightPeak = new AtomicInteger();
        /** Daemon threads with a name of their own, so a part that outlives a test is attributable and not a leak. */
        private final ExecutorService transfer = Executors.newFixedThreadPool(PART_CONCURRENCY, runnable -> {
            final Thread thread = new Thread(runnable, "w7-part-upload");
            thread.setDaemon(true);
            return thread;
        });
        private final ExecutorService finisher = Executors.newSingleThreadExecutor(runnable -> {
            final Thread thread = new Thread(runnable, "w7-upload-finisher");
            thread.setDaemon(true);
            return thread;
        });

        RecordingMultiStreamBlobContainer(final Path blobDirectory, final long partSizeBytes) {
            super(mock(FsBlobStore.class), new BlobPath().add("w7"), blobDirectory);
            this.blobDirectory = blobDirectory;
            this.partSizeBytes = partSizeBytes;
        }

        @Override
        public void asyncBlobUpload(final WriteContext writeContext, final ActionListener<Void> listener) throws IOException {
            final StreamContext streamContext = writeContext.getStreamProvider(partSizeBytes);
            final int numberOfParts = streamContext.getNumberOfParts();
            parts.set(numberOfParts);
            final long fileSize = writeContext.getFileSize();
            final Path blob = blobDirectory.resolve(writeContext.getFileName());
            // Pre-sized, because the parts are written at their own offsets and in no fixed order. The bytes
            // they actually produce are counted in partBytesWritten, which is what makes the pre-sizing safe.
            try (RandomAccessFile file = new RandomAccessFile(blob.toFile(), "rw")) {
                file.setLength(fileSize);
            }
            // As many as can run at once, so a part never waits for one that will not start.
            final CyclicBarrier rendezvous = new CyclicBarrier(Math.min(numberOfParts, PART_CONCURRENCY));
            final AtomicReference<Throwable> failure = new AtomicReference<>();
            final CountDownLatch done = new CountDownLatch(numberOfParts);
            for (int part = 0; part < numberOfParts; part++) {
                final int partNo = part;
                transfer.execute(() -> {
                    // provideStream is called here and not on the caller thread on purpose: it is what
                    // creates the part's KNNVectorValues, and with it the clone of the routed input the
                    // part's reads are taken through.
                    try {
                        final InputStreamContainer container = streamContext.provideStream(partNo);
                        try (InputStream stream = container.getInputStream()) {
                            awaitRendezvous(rendezvous);
                            final int live = inFlight.incrementAndGet();
                            inFlightPeak.accumulateAndGet(live, Math::max);
                            try {
                                final byte[] bytes = stream.readAllBytes();
                                if (bytes.length != container.getContentLength()) {
                                    throw new IOException(
                                        "part " + partNo + " produced " + bytes.length + " B of " + container.getContentLength()
                                    );
                                }
                                synchronized (this) {
                                    try (RandomAccessFile file = new RandomAccessFile(blob.toFile(), "rw")) {
                                        file.seek(container.getOffset());
                                        file.write(bytes);
                                    }
                                }
                                partBytesWritten.addAndGet(bytes.length);
                            } finally {
                                inFlight.decrementAndGet();
                            }
                        }
                        // Throwable and not Exception: the suppliers a test hands the accessor can raise an
                        // Error, and a part that died without recording why would leave the upload reporting
                        // success over a region of zeros.
                    } catch (Throwable t) {
                        failure.compareAndSet(null, t);
                    } finally {
                        done.countDown();
                    }
                });
            }
            // Returns now, as the real contract does, and the listener is called from a thread of its own once
            // the parts are in. The caller writes its doc-id blob in the meantime and then waits on its latch.
            finisher.execute(() -> {
                try {
                    if (done.await(60, TimeUnit.SECONDS) == false) {
                        failure.compareAndSet(null, new IOException("the fake repository's parts did not finish in 60 s"));
                    } else if (failure.get() == null && partBytesWritten.get() != fileSize) {
                        failure.compareAndSet(
                            null,
                            new IOException("the parts produced " + partBytesWritten.get() + " B of a " + fileSize + " B blob")
                        );
                    }
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    failure.compareAndSet(null, e);
                }
                final Throwable thrown = failure.get();
                if (thrown == null) {
                    listener.onResponse(null);
                } else {
                    listener.onFailure(thrown instanceof Exception e ? e : new RuntimeException(thrown));
                }
            });
        }

        /** Bounded, so a part whose siblings died fails rather than holding the upload for the test's timeout. */
        private static void awaitRendezvous(final CyclicBarrier rendezvous) throws IOException {
            try {
                rendezvous.await(30, TimeUnit.SECONDS);
            } catch (Exception e) {
                throw new IOException("the parts did not meet at the rendezvous", e);
            }
        }

        @Override
        public void writeBlob(final String blobName, final InputStream inputStream, final long blobSize, final boolean failIfAlreadyExists)
            throws IOException {
            if (blobName.endsWith(KNNConstants.VECTOR_BLOB_FILE_EXTENSION)) {
                vectorBlobWriteBlobCalls.incrementAndGet();
            }
            Files.copy(inputStream, blobDirectory.resolve(blobName));
        }

        // The three the accessor never calls. They throw rather than return quietly, so a path that starts
        // calling one of them fails here instead of passing with the call silently skipped.

        @Override
        public void readBlobAsync(final String name, final ActionListener<ReadContext> listener) {
            throw new UnsupportedOperationException("not exercised by this fixture");
        }

        @Override
        public boolean remoteIntegrityCheckSupported() {
            return false;
        }

        @Override
        public void deleteAsync(final ActionListener<DeleteResult> listener) {
            throw new UnsupportedOperationException("not exercised by this fixture");
        }

        @Override
        public void deleteBlobsAsyncIgnoringIfNotExists(final List<String> names, final ActionListener<Void> listener) {
            throw new UnsupportedOperationException("not exercised by this fixture");
        }

        @SneakyThrows
        void shutdown() {
            transfer.shutdown();
            finisher.shutdown();
            assertTrue("the part threads must have finished", transfer.awaitTermination(10, TimeUnit.SECONDS));
            assertTrue("and so must the finisher", finisher.awaitTermination(10, TimeUnit.SECONDS));
        }
    }

    /** A container that cannot take the multi-stream path, for the data types the uploader sends sequentially. */
    private static final class SingleStreamBlobContainer extends FsBlobContainer {

        private final Path directory;
        private final AtomicInteger vectorBlobWriteBlobCalls = new AtomicInteger();

        SingleStreamBlobContainer(final Path directory) {
            super(mock(FsBlobStore.class), new BlobPath().add("w7"), directory);
            this.directory = directory;
        }

        @Override
        public void writeBlob(final String blobName, final InputStream inputStream, final long blobSize, final boolean failIfAlreadyExists)
            throws IOException {
            if (blobName.endsWith(KNNConstants.VECTOR_BLOB_FILE_EXTENSION)) {
                vectorBlobWriteBlobCalls.incrementAndGet();
            }
            Files.copy(inputStream, directory.resolve(blobName));
        }
    }

    // -----------------------------------------------------------------------------------------------
    // Gate 1: the uploaded blobs are byte-identical with Direct I/O on and off
    // -----------------------------------------------------------------------------------------------

    /** What one arm of the comparison produced: the two blobs' digests, and how the vectors got there. */
    private record UploadResult(String vectorSha, String docIdSha, long vectorBytes, int parts, long routedOpens, long deviceBytes,
        long ringBytes, long blockingBytes, long armed, int inFlightPeak) {
    }

    /**
     * Uploads the fixture at {@code source} once, through the delegate when {@code routed} is false and through
     * {@code KNNDirectIODirectory} when it is true, into a repository of its own.
     */
    @SneakyThrows
    private UploadResult upload(final Path source, final VectorDataType dataType, final int docs, final boolean routed) {
        final Path repository = createTempDir();
        final boolean multiStream = dataType == VectorDataType.FLOAT || dataType == VectorDataType.HALF_FLOAT;
        // A quarter of the blob, so the two multi-stream types upload in four parts of 64 KiB: several ring
        // windows each, not a part that is one read long. The part count is asserted below rather than left to
        // this arithmetic, so a change to the fixture's size cannot quietly turn four parts into two.
        final long partSize = multiStream ? (long) docs * DIMENSION * Float.BYTES / 4 : 0;
        final RecordingMultiStreamBlobContainer parallel = multiStream ? new RecordingMultiStreamBlobContainer(repository, partSize) : null;
        final SingleStreamBlobContainer sequential = multiStream ? null : new SingleStreamBlobContainer(repository);
        try {
            DirectIOVectorIndexInput.STATS.reset();
            long routedOpens = 0;
            // The wrapper owns the delegate: FilterDirectory#close closes what it wraps, so only one of the
            // two is closed here and the delegate is not in the try-with-resources on the routed path.
            final MMapDirectory delegate = new MMapDirectory(source);
            final KNNDirectIODirectory storage = routed ? new KNNDirectIODirectory(delegate, "_0", () -> true, () -> 0L) : null;
            try (Directory directory = storage != null ? storage : delegate) {
                try (DirectoryReader reader = DirectoryReader.open(directory)) {
                    assertEquals("the fixture must be one segment", 1, reader.leaves().size());
                    final LeafReader leaf = reader.leaves().get(0).reader();
                    final FieldInfo fieldInfo = onlyFieldInfo(leaf);
                    if (routed) {
                        assertEquals(
                            "the .vec must have been served with O_DIRECT, declined=" + storage.declinedOpens(),
                            1,
                            storage.routedOpens()
                        );
                        assertEquals(0, storage.declinedOpens());
                        routedOpens = storage.routedOpens();
                    }

                    // UncheckedIOException and not an assertion: this runs on a part's upload thread, where an
                    // assertion would not be the test framework's to catch.
                    final Supplier<KNNVectorValues<?>> supplier = () -> {
                        try {
                            return KNNVectorValuesFactory.getVectorValues(fieldInfo, leaf);
                        } catch (IOException e) {
                            throw new UncheckedIOException(e);
                        }
                    };
                    // The concrete class and not the interface: VectorRepositoryAccessor is package-private
                    // in the remote package, so a test outside it can only hold the implementation.
                    final DefaultVectorRepositoryAccessor accessor = new DefaultVectorRepositoryAccessor(
                        multiStream ? parallel : sequential
                    );
                    accessor.writeToRepository(BLOB_NAME, docs, dataType, supplier);
                }
            }

            // The branch the uploader took, asserted rather than assumed: DefaultVectorRepositoryAccessor
            // only takes the parallel branch for FLOAT and HALF_FLOAT, so BYTE and BINARY must arrive as one
            // sequential writeBlob of the vector blob.
            if (multiStream) {
                assertEquals("the multi-stream path must have cut the blob into four parts", 4, parallel.parts.get());
                assertEquals("the vector blob must not also have been written sequentially", 0, parallel.vectorBlobWriteBlobCalls.get());
            } else {
                assertEquals("the sequential path must write the vector blob once", 1, sequential.vectorBlobWriteBlobCalls.get());
            }

            final Path vectorBlob = repository.resolve(VECTOR_BLOB);
            final Path docIdBlob = repository.resolve(DOC_ID_BLOB);
            assertTrue("the vector blob must exist", Files.exists(vectorBlob));
            assertTrue("the doc id blob must exist", Files.exists(docIdBlob));
            assertEquals("the doc id blob is four bytes a doc", (long) docs * Integer.BYTES, Files.size(docIdBlob));
            // What the blob holds, not only how long it is: the vectors the segment holds, encoded the way the
            // uploader encodes them, read back through a plain memory-mapped reader. Without this the pair of
            // digests could only catch a difference between the two routes, and two arms that both uploaded
            // nothing would agree.
            assertEquals(
                "the vector blob must hold the segment's vectors as the uploader encodes them",
                sha256(expectedVectorBlob(source, dataType, docs)),
                sha256(vectorBlob)
            );
            assertArrayEquals(
                "the doc id blob must hold the segment's document ids, little endian",
                expectedDocIdBlob(docs),
                Files.readAllBytes(docIdBlob)
            );
            return new UploadResult(
                sha256(vectorBlob),
                sha256(docIdBlob),
                Files.size(vectorBlob),
                multiStream ? parallel.parts.get() : 0,
                routedOpens,
                DirectIOVectorIndexInput.STATS.blockingBytes() + DirectIOVectorIndexInput.STATS.readaheadBytes()
                    + DirectIOVectorIndexInput.STATS.stagedBytes(),
                DirectIOVectorIndexInput.STATS.readaheadBytes(),
                DirectIOVectorIndexInput.STATS.blockingBytes(),
                DirectIOVectorIndexInput.STATS.readaheadArmed(),
                multiStream ? parallel.inFlightPeak.get() : 0
            );
        } finally {
            if (parallel != null) {
                parallel.shutdown();
            }
        }
    }

    /**
     * The vector blob the uploader should have produced, built independently of it: every vector of the
     * segment at {@code source} read through a plain memory-mapped reader and encoded the way
     * {@code VectorValuesInputStream} encodes it, which is little-endian fp32 for both float types (a
     * {@code half_float} field uploads as fp32) and the raw bytes for the two byte types.
     */
    @SneakyThrows
    private byte[] expectedVectorBlob(final Path source, final VectorDataType dataType, final int docs) {
        final boolean floatEncoded = dataType == VectorDataType.FLOAT || dataType == VectorDataType.HALF_FLOAT;
        final int bytesPerVector = switch (dataType) {
            case FLOAT, HALF_FLOAT -> DIMENSION * Float.BYTES;
            case BYTE -> DIMENSION;
            case BINARY -> DIMENSION / 8;
        };
        final ByteBuffer blob = ByteBuffer.allocate(docs * bytesPerVector).order(ByteOrder.LITTLE_ENDIAN);
        try (MMapDirectory delegate = new MMapDirectory(source); DirectoryReader reader = DirectoryReader.open(delegate)) {
            final LeafReader leaf = reader.leaves().get(0).reader();
            if (floatEncoded) {
                final FloatVectorValues values = leaf.getFloatVectorValues(FIELD);
                final KnnVectorValues.DocIndexIterator iterator = values.iterator();
                for (int doc = iterator.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = iterator.nextDoc()) {
                    for (final float value : values.vectorValue(iterator.index())) {
                        blob.putFloat(value);
                    }
                }
            } else {
                final ByteVectorValues values = leaf.getByteVectorValues(FIELD);
                final KnnVectorValues.DocIndexIterator iterator = values.iterator();
                for (int doc = iterator.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = iterator.nextDoc()) {
                    blob.put(values.vectorValue(iterator.index()));
                }
            }
        }
        assertEquals("the reference blob must be full", 0, blob.remaining());
        return blob.array();
    }

    /** The doc-id blob of a dense segment: ids 0 to {@code docs - 1}, as {@code DocIdInputStream} writes them. */
    private static byte[] expectedDocIdBlob(final int docs) {
        final ByteBuffer blob = ByteBuffer.allocate(docs * Integer.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (int doc = 0; doc < docs; doc++) {
            blob.putInt(doc);
        }
        return blob.array();
    }

    /**
     * The gate: the same segment uploaded through the delegate and through {@code O_DIRECT} produces the same
     * two blobs, and both hold the segment's own vectors and document ids.
     */
    @SneakyThrows
    private void assertUploadedBlobsAreIdentical(final VectorDataType dataType) {
        assumeDirectIOWorksHere();
        final Path source = createTempDir();
        try (MMapDirectory directory = new MMapDirectory(source)) {
            writeSegment(directory, dataType, UPLOAD_DOCS, 20261101L);
        }

        final UploadResult throughMmap = upload(source, dataType, UPLOAD_DOCS, false);
        final UploadResult throughDirectIO = upload(source, dataType, UPLOAD_DOCS, true);

        final long vectorDataBytes = Files.size(source.resolve(onlyVectorDataFileName(source)));
        logger.info(
            "W7 gate 1 {}: vec={} B blob={} B parts={} mmap_vec_sha={} dio_vec_sha={} mmap_doc_sha={} dio_doc_sha={} "
                + "dio_device={} B ({}x the file) ring={} B blocking={} B armed={}",
            dataType.getValue(),
            vectorDataBytes,
            throughDirectIO.vectorBytes(),
            throughDirectIO.parts(),
            throughMmap.vectorSha(),
            throughDirectIO.vectorSha(),
            throughMmap.docIdSha(),
            throughDirectIO.docIdSha(),
            throughDirectIO.deviceBytes(),
            String.format(Locale.ROOT, "%.2f", (double) throughDirectIO.deviceBytes() / vectorDataBytes),
            throughDirectIO.ringBytes(),
            throughDirectIO.blockingBytes(),
            throughDirectIO.armed()
        );

        // A half_float field uploads as raw fp32, which is why the expected
        // length is four bytes a dimension for it as well, and one byte a dimension for BYTE and an eighth
        // for BINARY.
        final long expectedVectorBytes = switch (dataType) {
            case FLOAT, HALF_FLOAT -> (long) UPLOAD_DOCS * DIMENSION * Float.BYTES;
            case BYTE -> (long) UPLOAD_DOCS * DIMENSION;
            case BINARY -> (long) UPLOAD_DOCS * DIMENSION / 8;
        };
        assertEquals("the vector blob must be the length the uploader computed", expectedVectorBytes, throughDirectIO.vectorBytes());
        assertEquals("both arms must have uploaded the same number of bytes", throughMmap.vectorBytes(), throughDirectIO.vectorBytes());
        assertEquals("the vector blob must be byte-identical on both routes", throughMmap.vectorSha(), throughDirectIO.vectorSha());
        assertEquals("the doc id blob must be byte-identical on both routes", throughMmap.docIdSha(), throughDirectIO.docIdSha());
        assertEquals("the routed arm must have routed", 1L, throughDirectIO.routedOpens());
        // At least the blob's worth, not merely non-zero: the accessor reads one vector on the caller thread
        // before it picks a branch, so a "> 0" here would hold on an upload that then read nothing.
        assertTrue(
            "the routed arm must have read at least the blob off the device, it read " + throughDirectIO.deviceBytes() + " B",
            throughDirectIO.deviceBytes() >= expectedVectorBytes
        );
        assertEquals("the mmap arm must not have touched the Direct I/O path at all", 0L, throughMmap.deviceBytes());
        if (dataType == VectorDataType.FLOAT || dataType == VectorDataType.HALF_FLOAT) {
            // The parts really were in flight together, so the reads this cell reports were taken under the
            // concurrency the multi-stream path is about rather than one part at a time.
            assertTrue(
                "the parts must have overlapped, peak in flight " + throughDirectIO.inFlightPeak(),
                throughDirectIO.inFlightPeak() > 1
            );
        }
    }

    public void testUploadedBlobsAreByteIdenticalForFloat() {
        assertUploadedBlobsAreIdentical(VectorDataType.FLOAT);
    }

    public void testUploadedBlobsAreByteIdenticalForHalfFloat() {
        assertUploadedBlobsAreIdentical(VectorDataType.HALF_FLOAT);
    }

    public void testUploadedBlobsAreByteIdenticalForByte() {
        assertUploadedBlobsAreIdentical(VectorDataType.BYTE);
    }

    public void testUploadedBlobsAreByteIdenticalForBinary() {
        assertUploadedBlobsAreIdentical(VectorDataType.BINARY);
    }

    // -----------------------------------------------------------------------------------------------
    // Gate 3: the read-ahead ring under the upload's part concurrency
    // -----------------------------------------------------------------------------------------------

    /**
     * One values object over two dense source segments' leaves, with ordinals and document ids running end to
     * end. This models only the property gate 3 measures: a part's stream can read from more than one source,
     * and each source's reads go through an input of its own. Deleted documents, and the ordinal-to-document
     * remapping a real {@code MergedVectorValues} does for a sub-reader's base, are out of scope here and are
     * the remote package's own suites to cover; a {@code MergeState} cannot be built outside a merge.
     */
    private static final class ConcatenatedFloatVectorValues extends FloatVectorValues {

        private final FloatVectorValues first;
        private final FloatVectorValues second;
        private final int firstSize;

        ConcatenatedFloatVectorValues(final FloatVectorValues first, final FloatVectorValues second) {
            this.first = first;
            this.second = second;
            this.firstSize = first.size();
        }

        @Override
        public float[] vectorValue(final int ordinal) throws IOException {
            return ordinal < firstSize ? first.vectorValue(ordinal) : second.vectorValue(ordinal - firstSize);
        }

        /** Required by the base class. No path in these tests clones a concatenation. */
        @Override
        public FloatVectorValues copy() throws IOException {
            return new ConcatenatedFloatVectorValues(first.copy(), second.copy());
        }

        @Override
        public int dimension() {
            return first.dimension();
        }

        @Override
        public int size() {
            return firstSize + second.size();
        }

        @Override
        public KnnVectorValues.DocIndexIterator iterator() {
            // Dense: every document in the fixture has a vector, so ordinal and document id coincide, which
            // is also what a merge of two dense segments produces.
            return createDenseIterator();
        }
    }

    /**
     * Gate 3: during the upload the ring arms once per input that scans, and it is never refused a window at
     * the budget a node runs with. The part size is chosen so that one part straddles the boundary between the
     * two sources, so the count is not trivially "one per part": the ring lives on the input, and a part that
     * reads from both sources scans two of them.
     */
    @SneakyThrows
    public void testReadaheadIsArmedPerPartPerSourceAndIsNeverDeclined() {
        assumeDirectIOWorksHere();
        final Path source = createTempDir();
        try (MMapDirectory directory = new MMapDirectory(source)) {
            writeSegment(directory, VectorDataType.FLOAT, RING_DOCS_PER_SOURCE, 20261102L);
            writeSegment(directory, VectorDataType.FLOAT, RING_DOCS_PER_SOURCE, 20261103L);
        }
        final int docs = 2 * RING_DOCS_PER_SOURCE;
        final long vectorBytes = (long) docs * DIMENSION * Float.BYTES;
        // Three parts over two equal sources, so the middle part spans the boundary: parts at
        // [0, 7/16), [7/16, 14/16) and [14/16, 1) of a blob whose sources split it at 8/16.
        final long partSize = vectorBytes * 7 / 16;

        // The budget is the node's, not a test's: gate 3's claim is about the shipped default.
        DirectIOVectorIndexInput.resetBudgetsForTesting();
        assertEquals(
            "the budget must be the one a node runs with",
            DirectIOVectorIndexInput.DIRECT_MEMORY_BUDGET_BYTES,
            DirectIOVectorIndexInput.directMemoryBudgetBytes()
        );
        DirectIOVectorIndexInput.STATS.reset();

        final Path repository = createTempDir();
        final RecordingMultiStreamBlobContainer parallel = new RecordingMultiStreamBlobContainer(repository, partSize);
        try (MMapDirectory delegate = new MMapDirectory(source)) {
            try (
                KNNDirectIODirectory storage = new KNNDirectIODirectory(delegate, "_merge", () -> true, () -> 0L);
                DirectoryReader reader = DirectoryReader.open(storage)
            ) {
                assertEquals("two segments, two readers", 2, reader.leaves().size());
                assertEquals(
                    "both sources must have been served with O_DIRECT, declined=" + storage.declinedOpens(),
                    2,
                    storage.routedOpens()
                );
                final LeafReader firstLeaf = reader.leaves().get(0).reader();
                final LeafReader secondLeaf = reader.leaves().get(1).reader();
                final FieldInfo fieldInfo = onlyFieldInfo(firstLeaf);

                final Supplier<KNNVectorValues<?>> supplier = () -> {
                    try {
                        return KNNVectorValuesFactory.getVectorValues(
                            VectorDataType.FLOAT,
                            new ConcatenatedFloatVectorValues(firstLeaf.getFloatVectorValues(FIELD), secondLeaf.getFloatVectorValues(FIELD))
                        );
                    } catch (IOException e) {
                        throw new UncheckedIOException(e);
                    }
                };
                new DefaultVectorRepositoryAccessor(parallel).writeToRepository(BLOB_NAME, docs, VectorDataType.FLOAT, supplier);
            }

            final int parts = parallel.parts.get();
            final long armed = DirectIOVectorIndexInput.STATS.readaheadArmed();
            final long declined = DirectIOVectorIndexInput.STATS.readaheadDeclined();
            logger.info(
                "W7 gate 3: blob={} B sources=2 parts={} part_size={} B armed={} declined={} ring_reads={} ring_bytes={} "
                    + "ring_hits={} blocking_reads={} blocking_bytes={} declined_budget={}",
                vectorBytes,
                parts,
                partSize,
                armed,
                declined,
                DirectIOVectorIndexInput.STATS.readaheadReads(),
                DirectIOVectorIndexInput.STATS.readaheadBytes(),
                DirectIOVectorIndexInput.STATS.readaheadHits(),
                DirectIOVectorIndexInput.STATS.blockingReads(),
                DirectIOVectorIndexInput.STATS.blockingBytes(),
                DirectIOVectorIndexInput.STATS.declinedBudget()
            );

            assertEquals("the part layout this gate is stated against", 3, parts);
            assertTrue("the parts must have overlapped, peak in flight " + parallel.inFlightPeak.get(), parallel.inFlightPeak.get() > 1);
            assertEquals("the ring must never be refused a window at the node's budget", 0L, declined);
            assertEquals("and no ring window may be refused for direct memory", 0L, DirectIOVectorIndexInput.STATS.declinedBudget());
            // One ring per (part, source) pair a part actually reads from, which for this layout is four:
            // part 0 lies inside source 1, part 1 straddles the boundary and scans both, part 2 lies inside
            // source 2. A count of 3 would mean a straddling part stopped arming on its second source, and 6
            // would mean a part armed a ring on a source it never read.
            //
            // The equality rests on two constants besides the layout, so both are pinned here: a ring arms
            // after DEFAULT_READAHEAD_TRIGGER consecutive gapless reads, and every (part, source) span in
            // this layout has to be longer than that many reads or the span would not arm at all. The
            // smallest span is the straddling part's share of the first source, 131,072 B, which is 2 reads
            // at any block size up to 32 KiB.
            assertEquals("the trigger this count is a function of", 2, DirectIOVectorIndexInput.DEFAULT_READAHEAD_TRIGGER);
            assertEquals("the ring arms once per part per source the part reads: armed=" + armed, 4L, armed);
            // And the upload produced the blob it should have: a read-ahead number taken from a scan that
            // served nothing would measure nothing. Compared by content, because the fake pre-sizes the file.
            assertEquals(
                "the blob must hold both sources' vectors end to end",
                sha256(expectedConcatenatedBlob(source)),
                sha256(repository.resolve(VECTOR_BLOB))
            );
            assertArrayEquals(
                "and the doc id blob must hold the merged document ids",
                expectedDocIdBlob(docs),
                Files.readAllBytes(repository.resolve(DOC_ID_BLOB))
            );
        } finally {
            parallel.shutdown();
        }
    }

    /** Both sources' vectors in order as little-endian fp32, read through a plain memory-mapped reader. */
    @SneakyThrows
    private byte[] expectedConcatenatedBlob(final Path source) {
        final ByteBuffer blob = ByteBuffer.allocate(2 * RING_DOCS_PER_SOURCE * DIMENSION * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        try (MMapDirectory delegate = new MMapDirectory(source); DirectoryReader reader = DirectoryReader.open(delegate)) {
            assertEquals("two segments, two readers", 2, reader.leaves().size());
            for (final var leaf : reader.leaves()) {
                final FloatVectorValues values = leaf.reader().getFloatVectorValues(FIELD);
                final KnnVectorValues.DocIndexIterator iterator = values.iterator();
                for (int doc = iterator.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = iterator.nextDoc()) {
                    for (final float value : values.vectorValue(iterator.index())) {
                        blob.putFloat(value);
                    }
                }
            }
        }
        assertEquals("the reference blob must be full", 0, blob.remaining());
        return blob.array();
    }

    // -----------------------------------------------------------------------------------------------
    // Gate 2: the CPU fallback after a failed remote build
    // -----------------------------------------------------------------------------------------------

    /**
     * A fallback build strategy that records what it was handed instead of building a graph: every document
     * id and the bytes of its vector, in the order the values object yields them. The graph is a deterministic
     * function of this sequence and the index parameters, so comparing the sequence between the two routes is
     * what says the fallback saw the same index; the graph itself is JNI's and is built the same way on both
     * routes from identical input.
     */
    private static final class RecordingFallbackStrategy implements NativeIndexBuildStrategy {

        private final List<String> rows = new ArrayList<>();
        private final AtomicInteger calls = new AtomicInteger();

        @Override
        public void buildAndWriteIndex(final BuildIndexParams indexInfo) throws IOException {
            calls.incrementAndGet();
            final KNNVectorValues<?> values = indexInfo.getKnnVectorValuesSupplier().get();
            initializeVectorValues(values);
            for (int doc = values.docId(); doc != DocIdSetIterator.NO_MORE_DOCS && doc != -1; doc = values.nextDoc()) {
                final Object vector = values.getVector();
                final byte[] bytes;
                if (vector instanceof float[] floats) {
                    final ByteBuffer buffer = ByteBuffer.allocate(floats.length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
                    buffer.asFloatBuffer().put(floats);
                    bytes = buffer.array();
                } else {
                    bytes = ((byte[]) vector).clone();
                }
                rows.add(doc + ":" + Base64.getEncoder().encodeToString(bytes));
            }
        }
    }

    /** What one arm of gate 2 saw: the fallback's input, and the device bytes the whole attempt cost. */
    private record FallbackResult(List<String> rows, int fallbackCalls, long uploadedBytes, long deviceBytes, long ringBytes,
        long blockingBytes, long armed) {
    }

    /**
     * Drives the real {@link RemoteIndexBuildStrategy} once,
     * with the upload succeeding and the build submission refused, and returns what the fallback was handed.
     *
     * <p>The failure is injected where this desktop actually has none: there is no remote build endpoint, so
     * the client's {@code submitVectorBuild} is stubbed to throw. The strategy wraps that in a
     * {@code RuntimeException}, its catch-all swallows it, and it calls the fallback strategy. So the upload
     * has already read every source once by the time the fallback reads them again, and the device bytes this
     * returns are both passes together.
     */
    @SneakyThrows
    private FallbackResult runFallback(final Path source, final int docs, final boolean routed) {
        final Path repositoryPath = createTempDir();
        final SingleStreamBlobContainer container = new SingleStreamBlobContainer(repositoryPath);

        final BlobStoreRepository repository = mock(BlobStoreRepository.class);
        final BlobStore blobStore = mock(BlobStore.class);
        when(repository.basePath()).thenReturn(new BlobPath().add("w7"));
        when(repository.blobStore()).thenReturn(blobStore);
        when(blobStore.blobContainer(any())).thenReturn(container);
        final RepositoryMetadata repositoryMetadata = s3RepositoryMetadata();
        when(repository.getMetadata()).thenReturn(repositoryMetadata);

        final RepositoriesService repositoriesService = mock(RepositoriesService.class);
        when(repositoriesService.repository(any())).thenReturn(repository);

        // getRepository asks KNNSettings for the repository's name and throws RepositoryMissingException when
        // it is unset, which would abort before the upload and so before the read the gate is about. Published
        // through a real ClusterSettings rather than a stub, so every other setting the path reads keeps its
        // own default.
        withRemoteRepositorySetting();

        final RecordingFallbackStrategy fallback = new RecordingFallbackStrategy();
        DirectIOVectorIndexInput.STATS.reset();

        try (MMapDirectory delegate = new MMapDirectory(source)) {
            final KNNDirectIODirectory storage = routed ? new KNNDirectIODirectory(delegate, "_0", () -> true, () -> 0L) : null;
            try (DirectoryReader reader = DirectoryReader.open(routed ? storage : delegate)) {
                assertEquals("the fixture must be one segment", 1, reader.leaves().size());
                final LeafReader leaf = reader.leaves().get(0).reader();
                final FieldInfo fieldInfo = onlyFieldInfo(leaf);
                if (routed) {
                    assertEquals(
                        "the .vec must have been served with O_DIRECT, declined=" + storage.declinedOpens(),
                        1,
                        storage.routedOpens()
                    );
                }

                final Supplier<KNNVectorValues<?>> supplier = () -> {
                    try {
                        return KNNVectorValuesFactory.getVectorValues(fieldInfo, leaf);
                    } catch (IOException e) {
                        throw new UncheckedIOException(e);
                    }
                };

                try (
                    Directory outputDirectory = new MMapDirectory(createTempDir());
                    IndexOutput output = outputDirectory.createOutput("fallback.faiss", IOContext.DEFAULT)
                ) {
                    final BuildIndexParams params = BuildIndexParams.builder()
                        .indexOutputWithBuffer(new IndexOutputWithBuffer(output))
                        .knnEngine(KNNEngine.FAISS)
                        .field(FIELD)
                        .vectorDataType(VectorDataType.FLOAT)
                        .indexParameters(
                            Map.of(
                                KNNConstants.SPACE_TYPE,
                                SpaceType.L2.getValue(),
                                KNNConstants.PARAMETERS,
                                Map.of(KNNConstants.METHOD_PARAMETER_M, 16, KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION, 100)
                            )
                        )
                        .knnVectorValuesSupplier(supplier)
                        .totalLiveDocs(docs)
                        .segmentWriteState(segmentWriteState(outputDirectory))
                        .isFlush(false)
                        .build();

                    final KNNLibraryIndexingContext libraryContext = mock(KNNLibraryIndexingContext.class);
                    // What Faiss#createRemoteIndexingParameters needs to assemble a build request. Without
                    // it the request assembly throws and the strategy falls back for that reason instead, so
                    // the injected failure below would never fire and the gate would measure the wrong
                    // mechanism.
                    when(libraryContext.getLibraryParameters()).thenReturn(
                        Map.of(
                            KNNConstants.NAME,
                            KNNConstants.METHOD_HNSW,
                            KNNConstants.SPACE_TYPE,
                            SpaceType.L2.getValue(),
                            KNNConstants.PARAMETERS,
                            Map.of(
                                KNNConstants.METHOD_PARAMETER_EF_CONSTRUCTION,
                                100,
                                KNNConstants.METHOD_PARAMETER_EF_SEARCH,
                                100,
                                KNNConstants.METHOD_PARAMETER_M,
                                16
                            )
                        )
                    );
                    when(libraryContext.getResolvedSpec()).thenReturn(ResolvedIndexSpec.builder().build());

                    final RemoteIndexBuildStrategy strategy = new RemoteIndexBuildStrategy(
                        () -> repositoriesService,
                        fallback,
                        testIndexSettings(),
                        libraryContext
                    );

                    try (MockedStatic<RemoteIndexClientFactory> factory = mockStatic(RemoteIndexClientFactory.class)) {
                        final RemoteIndexClient client = mock(RemoteIndexClient.class);
                        when(client.submitVectorBuild(any())).thenThrow(new IOException("no remote build endpoint on this desktop"));
                        factory.when(() -> RemoteIndexClientFactory.getRemoteIndexClient(any())).thenReturn(client);

                        strategy.buildAndWriteIndex(params);

                        // The failure the fixture injected is the one that happened: the strategy's catch-all
                        // swallows any post-upload failure identically, so without this the two-pass number
                        // below could have been produced by a repository resolution error instead.
                        verify(client).submitVectorBuild(any());
                    }
                }
            } finally {
                if (storage != null) {
                    storage.close();
                }
            }
        }

        // The real strategy names the blob itself, with a fresh UUID in it
        // (RemoteIndexBuildStrategy#getRepositoryContext), so the uploaded file is found by extension.
        return new FallbackResult(
            fallback.rows,
            fallback.calls.get(),
            uploadedVectorBlobBytes(repositoryPath),
            DirectIOVectorIndexInput.STATS.blockingBytes() + DirectIOVectorIndexInput.STATS.readaheadBytes()
                + DirectIOVectorIndexInput.STATS.stagedBytes(),
            DirectIOVectorIndexInput.STATS.readaheadBytes(),
            DirectIOVectorIndexInput.STATS.blockingBytes(),
            DirectIOVectorIndexInput.STATS.readaheadArmed()
        );
    }

    /** Publishes a repository name, so {@code getRepository} resolves and the upload runs. */
    private void withRemoteRepositorySetting() {
        final Set<Setting<?>> registered = new HashSet<>(ClusterSettings.BUILT_IN_CLUSTER_SETTINGS);
        registered.addAll(
            KNNSettings.state()
                .getSettings()
                .stream()
                .filter(setting -> setting.getProperties().contains(Setting.Property.NodeScope))
                .toList()
        );
        final Settings settings = Settings.builder().put(KNNSettings.KNN_REMOTE_REPOSITORY, "w7-repo").build();
        when(clusterService.getClusterSettings()).thenReturn(new ClusterSettings(settings, registered));
        KNNSettings.state().setClusterService(clusterService);
    }

    /** The length of the one vector blob in {@code repositoryPath}, or -1 if the upload never wrote one. */
    @SneakyThrows
    private static long uploadedVectorBlobBytes(final Path repositoryPath) {
        final List<Path> blobs = new ArrayList<>();
        try (var files = Files.list(repositoryPath)) {
            files.filter(file -> file.getFileName().toString().endsWith(KNNConstants.VECTOR_BLOB_FILE_EXTENSION)).forEach(blobs::add);
        }
        if (blobs.isEmpty()) {
            return -1;
        }
        assertEquals("the upload must have written one vector blob, saw " + blobs, 1, blobs.size());
        return Files.size(blobs.get(0));
    }

    /** The minimum an S3-typed repository's metadata has to answer for the build request to be built. */
    private static RepositoryMetadata s3RepositoryMetadata() {
        final RepositoryMetadata metadata = mock(RepositoryMetadata.class);
        when(metadata.type()).thenReturn(KNNConstants.S3);
        when(metadata.settings()).thenReturn(Settings.builder().put(KNNConstants.BUCKET, "w7-bucket").build());
        return metadata;
    }

    private static IndexSettings testIndexSettings() {
        final IndexSettings indexSettings = mock(IndexSettings.class);
        when(indexSettings.getSettings()).thenReturn(
            Settings.builder().put(ClusterName.CLUSTER_NAME_SETTING.getKey(), "w7-cluster").build()
        );
        when(indexSettings.getUUID()).thenReturn("w7-index-uuid");
        return indexSettings;
    }

    private static SegmentWriteState segmentWriteState(final Directory directory) {
        final SegmentInfo segmentInfo = new SegmentInfo(
            directory,
            Version.LATEST,
            Version.LATEST,
            "_0",
            1,
            false,
            false,
            mock(Codec.class),
            Map.of(),
            new byte[16],
            Map.of(),
            null
        );
        return new SegmentWriteState(mock(InfoStream.class), directory, segmentInfo, mock(FieldInfos.class), null, IOContext.DEFAULT);
    }

    /**
     * Gate 2: a remote build whose submission fails falls back to the CPU build, and the fallback through
     * {@code O_DIRECT} is handed exactly the index the fallback with Direct I/O off is handed.
     *
     * <p>The device bytes cover both passes, the upload's and the fallback's, and are asserted only against a
     * lower bound of twice the file: the exact ratio depends on where the read-ahead ring stops at end of file
     * and is logged rather than pinned.
     */
    @SneakyThrows
    public void testTheCpuFallbackThroughDirectIOSeesTheSameIndex() {
        assumeDirectIOWorksHere();
        final Path source = createTempDir();
        try (MMapDirectory directory = new MMapDirectory(source)) {
            writeSegment(directory, VectorDataType.FLOAT, UPLOAD_DOCS, 20261104L);
        }
        final long vectorDataBytes = Files.size(source.resolve(onlyVectorDataFileName(source)));

        final FallbackResult throughMmap = runFallback(source, UPLOAD_DOCS, false);
        final FallbackResult throughDirectIO = runFallback(source, UPLOAD_DOCS, true);

        logger.info(
            "W7 gate 2: vec={} B uploaded={} B fallback_vectors={} mmap_calls={} dio_calls={} dio_device={} B ({}x the file) "
                + "ring={} B blocking={} B armed={}",
            vectorDataBytes,
            throughDirectIO.uploadedBytes(),
            throughDirectIO.rows().size(),
            throughMmap.fallbackCalls(),
            throughDirectIO.fallbackCalls(),
            throughDirectIO.deviceBytes(),
            String.format(Locale.ROOT, "%.2f", (double) throughDirectIO.deviceBytes() / vectorDataBytes),
            throughDirectIO.ringBytes(),
            throughDirectIO.blockingBytes(),
            throughDirectIO.armed()
        );

        assertEquals("the fallback must run exactly once with the route off", 1, throughMmap.fallbackCalls());
        assertEquals("and exactly once with the route on", 1, throughDirectIO.fallbackCalls());
        assertEquals(
            "the upload must have completed before the build failed",
            (long) UPLOAD_DOCS * DIMENSION * Float.BYTES,
            throughDirectIO.uploadedBytes()
        );
        assertEquals(
            "the upload must have completed on both routes, not only the routed one",
            throughMmap.uploadedBytes(),
            throughDirectIO.uploadedBytes()
        );
        assertEquals("the fallback must have been handed every document", UPLOAD_DOCS, throughDirectIO.rows().size());
        assertEquals(
            "the fallback must see the same documents and the same vector bytes on both routes",
            throughMmap.rows(),
            throughDirectIO.rows()
        );
        // Twice the file at least: the upload scans it once and the fallback scans it again, so a bound of
        // "> 0" would hold on a run where only one of the two passes went through the route.
        assertTrue(
            "the routed arm must have read both passes off the device, it read " + throughDirectIO.deviceBytes() + " B",
            throughDirectIO.deviceBytes() >= 2 * vectorDataBytes
        );
        assertEquals("the mmap arm must not have touched the Direct I/O path", 0L, throughMmap.deviceBytes());
    }

    /** The one {@code .vec} of a one-segment fixture, by name. */
    @SneakyThrows
    private static String onlyVectorDataFileName(final Path path) {
        final List<String> names = new ArrayList<>();
        try (var files = Files.list(path)) {
            files.map(file -> file.getFileName().toString()).filter(name -> name.endsWith(".vec")).sorted().forEach(names::add);
        }
        assertEquals("the fixture must hold exactly one .vec, saw " + names, 1, names.size());
        return names.get(0);
    }
}

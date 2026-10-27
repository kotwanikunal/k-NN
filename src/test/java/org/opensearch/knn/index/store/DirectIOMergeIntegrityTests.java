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
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.CodecReader;
import org.apache.lucene.index.CorruptIndexException;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexOptions;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.KnnVectorValues;
import org.apache.lucene.index.NoMergePolicy;
import org.apache.lucene.index.SerialMergeScheduler;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.store.MMapDirectory;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.common.KNNConstants;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.codec.KNN990Codec.NativeEngines990KnnVectorsFormat;
import org.opensearch.knn.index.codec.util.UnitTestCodec;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * What a merge and Lucene's integrity check see through {@code O_DIRECT}: a corrupt {@code .vec} is detected
 * exactly as it is through the delegate, a merge of a corrupt segment fails rather than writing a new file out
 * of bad bytes, a clean merge writes the same vectors on both routes, and one integrity check reads its file
 * once through the sequential ring without admitting any of it to the store.
 *
 * <p><b>Why these are the assertions that matter.</b> {@code CodecUtil#checksumEntireFile} reads the file
 * forward from end to end, which is the one access pattern that arms the sequential read-ahead ring, so the
 * integrity check is served almost entirely from ring buffers rather than from the reads a query takes. A ring
 * that served a stale, short or misaligned buffer would compute a checksum over bytes that are not the file's
 * and could agree with the stored one; a merge would then copy whatever it liked into the new segment. So
 * these tests are the proof that the ring is transparent, and not just that it is fast.
 *
 * <p>Both corruption positions are covered because they fail by different mechanisms: a byte in the vector
 * region changes the computed checksum, and a byte in the footer changes the stored one. The footer case also
 * pins where the failure lands: the codec footer's magic and algorithm are validated when the file is opened,
 * so only a corrupt stored checksum reaches {@code checkIntegrity}.
 *
 * <p>The clean-merge case cannot assert that the two merged files are equal byte for byte, and the assertion it
 * makes instead is explained where it is made: Lucene stamps a fresh random id into every new segment's index
 * header and the footer checksums the header with it, so two merges of identical sources always differ in
 * those two places and must differ nowhere else.
 */
@ThreadLeakFilters(defaultFilters = true, filters = { DirectIOReadPoolTests.ReadPoolThreadFilter.class })
public class DirectIOMergeIntegrityTests extends KNNTestCase {

    private static final int DIMENSION = 128;
    /** One fp32 vector is 512 B, so this is a 2 MiB {@code .vec}: about 32 ring windows. */
    private static final int DOCS = 4096;
    private static final String FIELD = "v";
    private static final int FOOTER_LENGTH = 16;
    /** Lucene's per-segment random id, as {@code StringHelper#ID_LENGTH} defines it. */
    private static final int SEGMENT_ID_LENGTH = 16;
    /** How far into a file the index header can reach, which bounds where the segment id may be. */
    private static final int HEADER_LIMIT = 128;
    /** The ring's shipped window, which the device-bytes bound is stated in multiples of. */
    private static final long WINDOW_BYTES = DirectIOVectorIndexInput.DEFAULT_READAHEAD_WINDOW_BYTES;
    /** Slabs this size so that a store target of a few megabytes is whole slabs, as the shipped 32 MiB is not. */
    private static final int TEST_SLAB_BYTES = 1024 * 1024;
    /** A store big enough to hold the whole fixture, so that admission is never refused for want of room. */
    private static final long STORE_BYTES = 8L * TEST_SLAB_BYTES;

    /**
     * The plugin's native-engine format with the graph build skipped, as
     * {@link FaissVectorDataIdentityTests} uses it: a negative approximate threshold makes
     * {@code shouldSkipBuildingVectorDataStructure} answer true for any doc count, so the segment holds the
     * fp32 {@code .vec} and its {@code .vemf} sidecar and no {@code .faiss}. The integrity check and the
     * merge read the {@code .vec}, which is what this measures, and the fixture stays off the JNI layer.
     */
    private static final Codec FAISS_CODEC = new UnitTestCodec(() -> new NativeEngines990KnnVectorsFormat(-1));

    @Override
    public void setUp() throws Exception {
        super.setUp();
        // The ring's window count is a static another class in this package may have left at 0, and a ring
        // that is off would make the read-ahead assertions below fail for a reason that is not this test's.
        // Published explicitly here and restored in tearDown, so the suite's order cannot decide the result.
        DirectIOVectorIndexInput.publishReadaheadWindow(KNNSettings.KNN_DIRECT_IO_READAHEAD_WINDOW_DEFAULT_VALUE);
    }

    @Override
    public void tearDown() throws Exception {
        DirectIOVectorCache.resetForTesting();
        DirectIOReadPool.resetForTesting();
        DirectIOVectorIndexInput.publishReadaheadWindow(KNNSettings.KNN_DIRECT_IO_READAHEAD_WINDOW_DEFAULT_VALUE);
        DirectIOVectorIndexInput.STATS.reset();
        super.tearDown();
    }

    @SneakyThrows
    private void assumeDirectIOWorksHere() {
        final Path probe = createTempDir().resolve("probe");
        Files.write(probe, new byte[8192]);
        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(probe)) {
            assumeTrue("O_DIRECT is not available here", input != null);
        }
    }

    private static float[] vector(final Random random) {
        final float[] values = new float[DIMENSION];
        for (int d = 0; d < DIMENSION; d++) {
            values[d] = random.nextFloat() * 20 - 10;
        }
        return values;
    }

    /** A native-engine vector field, so the per-field format name in the file name is the faiss one. */
    private static FieldType nativeEngineVectorField() {
        final FieldType fieldType = new FieldType();
        fieldType.setTokenized(false);
        fieldType.setIndexOptions(IndexOptions.NONE);
        fieldType.putAttribute(KNNVectorFieldMapper.KNN_FIELD, "true");
        fieldType.putAttribute(KNNConstants.KNN_METHOD, KNNConstants.METHOD_HNSW);
        fieldType.putAttribute(KNNConstants.KNN_ENGINE, KNNEngine.FAISS.getName());
        fieldType.putAttribute(KNNConstants.SPACE_TYPE, SpaceType.L2.getValue());
        fieldType.putAttribute(KNNConstants.HNSW_ALGO_M, "32");
        fieldType.putAttribute(KNNConstants.HNSW_ALGO_EF_CONSTRUCTION, "512");
        fieldType.putAttribute(KNNConstants.VECTOR_DATA_TYPE_FIELD, VectorDataType.FLOAT.getValue());
        fieldType.setVectorAttributes(DIMENSION, VectorEncoding.FLOAT32, VectorSimilarityFunction.EUCLIDEAN);
        fieldType.freeze();
        return fieldType;
    }

    private static IndexWriterConfig config() {
        final IndexWriterConfig config = new IndexWriterConfig();
        config.setCodec(FAISS_CODEC);
        // Non-compound, or the .vec is inside a .cfs and nothing routes (the dispatch rule's condition 3).
        config.setUseCompoundFile(false);
        // The merge must run on the calling thread, or a merge that fails fails on a Lucene merge thread and
        // forceMerge returns normally. The corruption gate is an assertion about what forceMerge throws.
        config.setMergeScheduler(new SerialMergeScheduler());
        return config;
    }

    /** One commit of {@code docs} documents, with merges off so each call is its own segment. */
    private static void writeSegment(final MMapDirectory directory, final int docs, final long seed) throws IOException {
        final IndexWriterConfig config = config();
        config.setMergePolicy(NoMergePolicy.INSTANCE);
        final FieldType fieldType = nativeEngineVectorField();
        final Random random = new Random(seed);
        try (IndexWriter writer = new IndexWriter(directory, config)) {
            for (int i = 0; i < docs; i++) {
                final Document document = new Document();
                document.add(new KnnFloatVectorField(FIELD, vector(random), fieldType));
                writer.addDocument(document);
            }
            writer.commit();
        }
    }

    /** Every {@code .vec} in {@code path}, by name, sorted. */
    @SneakyThrows
    private static List<String> vectorDataFileNames(final Path path) {
        final List<String> names = new ArrayList<>();
        try (var files = Files.list(path)) {
            files.map(file -> file.getFileName().toString()).filter(name -> name.endsWith(".vec")).sorted().forEach(names::add);
        }
        return names;
    }

    private static String onlyVectorDataFileName(final Path path) {
        final List<String> names = vectorDataFileNames(path);
        assertEquals("the fixture must hold exactly one .vec, saw " + names, 1, names.size());
        return names.get(0);
    }

    /** Flips the low bit of one byte of {@code file} at {@code offset}, in place. */
    private static void corruptByte(final Path file, final long offset) throws IOException {
        final byte[] bytes = Files.readAllBytes(file);
        final int at = Math.toIntExact(offset < 0 ? bytes.length + offset : offset);
        bytes[at] = (byte) (bytes[at] ^ 0x01);
        Files.write(file, bytes);
    }

    @SneakyThrows
    private static String sha256(final Path file) {
        final MessageDigest digest = MessageDigest.getInstance("SHA-256");
        final byte[] sum = digest.digest(Files.readAllBytes(file));
        final StringBuilder hex = new StringBuilder(sum.length * 2);
        for (final byte b : sum) {
            hex.append(String.format(java.util.Locale.ROOT, "%02x", b));
        }
        return hex.toString();
    }

    /** The one leaf of a one-segment fixture, as the reader the integrity check is asked of. */
    private static CodecReader onlyCodecLeaf(final DirectoryReader reader) {
        assertEquals("the fixture must be one segment", 1, reader.leaves().size());
        return (CodecReader) reader.leaves().get(0).reader();
    }

    /**
     * Asserts that {@code checkIntegrity} on a one-segment fixture at {@code path} throws
     * {@link CorruptIndexException} through the delegate and through the {@code O_DIRECT} route alike, and
     * that the routed run really did route.
     */
    private void assertCorruptionIsDetectedOnBothRoutes(final Path path, final String position) throws IOException {
        try (MMapDirectory delegate = new MMapDirectory(path)) {
            final CorruptIndexException throughDelegate = expectThrows(CorruptIndexException.class, () -> {
                try (DirectoryReader reader = DirectoryReader.open(delegate)) {
                    onlyCodecLeaf(reader).checkIntegrity();
                }
            });

            DirectIOVectorIndexInput.STATS.reset();
            try (KNNDirectIODirectory storage = new KNNDirectIODirectory(delegate, "_0", () -> true, () -> 0L)) {
                final CorruptIndexException throughDirectIO = expectThrows(CorruptIndexException.class, () -> {
                    try (DirectoryReader reader = DirectoryReader.open(storage)) {
                        assertEquals(
                            "the .vec must have been served with O_DIRECT, declined=" + storage.declinedOpens(),
                            1,
                            storage.routedOpens()
                        );
                        onlyCodecLeaf(reader).checkIntegrity();
                    }
                });
                logger.info(
                    "W5 gate 1 {}: delegate [{}] | O_DIRECT [{}] | {}",
                    position,
                    throughDelegate.getMessage(),
                    throughDirectIO.getMessage(),
                    DirectIOVectorIndexInput.STATS
                );
                // The same failure, not merely some failure: a route that threw for its own reason (a short
                // read, an alignment slip) would pass a bare expectThrows and prove nothing about the bytes.
                assertEquals(
                    "the same corruption must be reported the same way on both routes",
                    messageShape(throughDelegate),
                    messageShape(throughDirectIO)
                );
            }
        }
    }

    /**
     * A corruption message without the parts that name the input: the delegate's and the routed input's
     * {@code toString} differ by construction, and the file's offsets and checksums are what has to match.
     */
    private static String messageShape(final CorruptIndexException thrown) {
        final String message = thrown.getMessage();
        final int resource = message.indexOf(" (resource=");
        return resource < 0 ? message : message.substring(0, resource);
    }

    // -----------------------------------------------------------------------------------------------
    // Gate 1: corruption is still detected, in the vector region and in the footer
    // -----------------------------------------------------------------------------------------------

    /** One byte of the vector region, which changes the checksum the check computes. */
    @SneakyThrows
    public void testVectorRegionCorruptionIsDetectedThroughDirectIO() {
        assumeDirectIOWorksHere();
        final Path path = createTempDir();
        try (MMapDirectory directory = new MMapDirectory(path)) {
            writeSegment(directory, DOCS, 20261027L);
        }
        final Path vectorData = path.resolve(onlyVectorDataFileName(path));
        // Deep inside the region, so the check has armed the ring long before it reaches the bad byte.
        corruptByte(vectorData, Files.size(vectorData) / 2);
        assertCorruptionIsDetectedOnBothRoutes(path, "vector region");
    }

    /**
     * One byte of the stored checksum, which changes the value the check compares against. The other eight
     * footer bytes are the codec footer's magic and algorithm id, and corrupting those is caught when the
     * file is opened rather than by {@code checkIntegrity}; that half is asserted here too, because it is
     * the reason this test corrupts the last byte and not an arbitrary one of the last sixteen.
     */
    @SneakyThrows
    public void testFooterCorruptionIsDetectedThroughDirectIO() {
        assumeDirectIOWorksHere();
        final Path path = createTempDir();
        try (MMapDirectory directory = new MMapDirectory(path)) {
            writeSegment(directory, DOCS, 20261028L);
        }
        final Path vectorData = path.resolve(onlyVectorDataFileName(path));
        corruptByte(vectorData, -1);
        assertCorruptionIsDetectedOnBothRoutes(path, "stored checksum");

        // And the footer's magic, which both routes reject while the file is being opened: the reader itself
        // fails, so there is no reader to ask for an integrity check. Asserted as an open that throws, not as
        // an open-then-check that throws somewhere, or the claim above would not be pinned to a position.
        corruptByte(vectorData, -1);
        corruptByte(vectorData, -FOOTER_LENGTH);
        try (MMapDirectory delegate = new MMapDirectory(path)) {
            final CorruptIndexException delegateOpen = expectThrows(
                CorruptIndexException.class,
                () -> { DirectoryReader.open(delegate).close(); }
            );
            DirectIOVectorIndexInput.STATS.reset();
            try (KNNDirectIODirectory storage = new KNNDirectIODirectory(delegate, "_0", () -> true, () -> 0L)) {
                final CorruptIndexException routedOpen = expectThrows(
                    CorruptIndexException.class,
                    () -> { DirectoryReader.open(storage).close(); }
                );
                assertEquals("the .vec must have been served with O_DIRECT, declined=" + storage.declinedOpens(), 1, storage.routedOpens());
                assertEquals("a corrupt footer magic must read the same both ways", messageShape(delegateOpen), messageShape(routedOpen));
            }
        }
    }

    // -----------------------------------------------------------------------------------------------
    // Gate 1, third position: a merge of a corrupt segment fails rather than copying the bad bytes
    // -----------------------------------------------------------------------------------------------

    /**
     * A merge whose source {@code .vec} is corrupt must fail through the {@code O_DIRECT} route, and must
     * not leave a merged {@code .vec} behind: the failure has to arrive before the new file is committed,
     * or a corrupt vector would have been copied into a segment whose own checksum then agrees with it.
     */
    @SneakyThrows
    public void testAMergeOfACorruptSegmentFails() {
        assumeDirectIOWorksHere();
        final Path path = createTempDir();
        try (MMapDirectory directory = new MMapDirectory(path)) {
            writeSegment(directory, DOCS / 2, 20261029L);
            writeSegment(directory, DOCS / 2, 20261030L);
        }
        final List<String> sources = vectorDataFileNames(path);
        assertEquals("two segments, two .vec files, saw " + sources, 2, sources.size());
        final Path corrupt = path.resolve(sources.get(0));
        corruptByte(corrupt, Files.size(corrupt) / 2);

        DirectIOVectorIndexInput.STATS.reset();
        try (
            MMapDirectory delegate = new MMapDirectory(path);
            KNNDirectIODirectory storage = new KNNDirectIODirectory(delegate, "_merge", () -> true, () -> 0L)
        ) {
            // The exact type, not any IOException: a short read, an alignment slip or a lock failure would
            // also abort the merge, and this test would then pass while saying nothing about the checksum.
            final CorruptIndexException thrown = expectThrows(CorruptIndexException.class, () -> {
                try (IndexWriter writer = new IndexWriter(storage, config())) {
                    writer.forceMerge(1);
                    writer.commit();
                }
            });
            logger.info(
                "W5 gate 1 merge: the merge failed with [{}], routed={} | {}",
                thrown.getMessage(),
                storage.routedOpens(),
                DirectIOVectorIndexInput.STATS
            );
            // One open per source, and the reads were taken through those inputs rather than handed back to
            // the delegate: an input that fell back on every read would leave routedOpens standing alone.
            assertEquals("both sources must have been read with O_DIRECT, declined=" + storage.declinedOpens(), 2, storage.routedOpens());
            assertEquals("a routed merge read must not fall back to the delegate", 0L, DirectIOVectorIndexInput.STATS.fallbackReads());
            assertTrue(
                "the merge must have read bytes through the route, not just opened it",
                DirectIOVectorIndexInput.STATS.blockingBytes() + DirectIOVectorIndexInput.STATS.readaheadBytes() > 0
            );
        }
        // Lucene rolls the failed merge back, so the only .vec files left are the two sources. A third one
        // would be a merged segment written out of corrupt bytes.
        assertEquals("the failed merge must leave no merged .vec", sources, vectorDataFileNames(path));
    }

    // -----------------------------------------------------------------------------------------------
    // Gate 2: a clean merge writes the same vectors on both routes
    // -----------------------------------------------------------------------------------------------

    /**
     * The clean case: the same two segments merged with the route on and with it off produce the same merged
     * vectors, and files that differ only where Lucene's own per-segment id forces them to. The two runs are
     * two copies of one directory, so the only difference between them is the system call the merge's reads
     * are taken with.
     */
    @SneakyThrows
    public void testACleanMergeIsByteIdenticalWithAndWithoutDirectIO() {
        assumeDirectIOWorksHere();
        final Path routedPath = createTempDir();
        try (MMapDirectory directory = new MMapDirectory(routedPath)) {
            writeSegment(directory, DOCS / 2, 20261031L);
            writeSegment(directory, DOCS / 2, 20261032L);
        }
        // The control is the same two segments written again from the same two seeds, rather than a file
        // copy of the first: the vectors are identical, which is what the comparison needs, and nothing
        // depends on what else the test framework may have left in a temporary directory.
        final Path mmapPath = createTempDir();
        try (MMapDirectory directory = new MMapDirectory(mmapPath)) {
            writeSegment(directory, DOCS / 2, 20261031L);
            writeSegment(directory, DOCS / 2, 20261032L);
        }
        final List<String> sources = vectorDataFileNames(routedPath);
        assertEquals("two segments, two .vec files, saw " + sources, 2, sources.size());
        assertEquals("the control must hold the same two segments", sources, vectorDataFileNames(mmapPath));

        DirectIOVectorIndexInput.STATS.reset();
        final long routedOpens;
        try (
            MMapDirectory delegate = new MMapDirectory(routedPath);
            KNNDirectIODirectory storage = new KNNDirectIODirectory(delegate, "_merge", () -> true, () -> 0L)
        ) {
            try (IndexWriter writer = new IndexWriter(storage, config())) {
                writer.forceMerge(1);
                writer.commit();
            }
            routedOpens = storage.routedOpens();
            assertEquals("both sources must have been read with O_DIRECT, declined=" + storage.declinedOpens(), 2, routedOpens);
            assertEquals("a routed merge read must not fall back to the delegate", 0L, DirectIOVectorIndexInput.STATS.fallbackReads());
        }
        try (MMapDirectory control = new MMapDirectory(mmapPath); IndexWriter writer = new IndexWriter(control, config())) {
            writer.forceMerge(1);
            writer.commit();
        }

        final String merged = mergedVectorDataFileName(routedPath, sources);
        assertEquals("the two merges must have produced the same segment", merged, mergedVectorDataFileName(mmapPath, sources));
        // The two digests are recorded rather than compared, because they cannot be equal: see
        // assertSameExceptTheSegmentIdAndItsChecksum.
        final String routedSha = sha256(routedPath.resolve(merged));
        final String mmapSha = sha256(mmapPath.resolve(merged));
        logger.info(
            "W5 gate 2: merged {} routed_opens={} sha256 O_DIRECT {} | mmap {} | {}",
            merged,
            routedOpens,
            routedSha,
            mmapSha,
            DirectIOVectorIndexInput.STATS
        );
        assertSameExceptTheSegmentIdAndItsChecksum(mmapPath.resolve(merged), routedPath.resolve(merged));

        // And the content gate, which is what byte identity was standing in for: every merged vector, in
        // ordinal order, read back from both segments.
        final List<float[]> throughMmap = readAllVectors(mmapPath);
        final List<float[]> throughDirectIO = readAllVectors(routedPath);
        assertEquals("the merge must have written every document", DOCS, throughMmap.size());
        assertEquals(throughMmap.size(), throughDirectIO.size());
        for (int ordinal = 0; ordinal < throughMmap.size(); ordinal++) {
            assertArrayEquals("merged ordinal " + ordinal, throughMmap.get(ordinal), throughDirectIO.get(ordinal), 0.0f);
        }
    }

    /**
     * The byte-level half of the clean-merge gate.
     *
     * <p>Two merges of identical sources cannot write identical files, for a reason that has nothing to do
     * with Direct I/O: Lucene stamps a fresh random segment id into every new segment's index header
     * ({@code StringHelper#randomId}), and the codec footer stores a checksum of the whole file, header
     * included. So they differ in exactly two places, the 16-byte id and the 8-byte stored checksum, and what
     * is asserted is that they differ in nothing else: same length, every differing byte either inside the
     * index header's reach or in the stored checksum, and the differing header bytes no wider than one id. A
     * merge that had copied a single vector differently would fail it.
     */
    @SneakyThrows
    private void assertSameExceptTheSegmentIdAndItsChecksum(final Path expected, final Path actual) {
        final byte[] a = Files.readAllBytes(expected);
        final byte[] b = Files.readAllBytes(actual);
        assertEquals("the two merges must have written the same number of bytes", a.length, b.length);
        final List<Integer> differing = new ArrayList<>();
        for (int at = 0; at < a.length; at++) {
            if (a[at] != b[at]) {
                differing.add(at);
            }
        }
        if (differing.isEmpty()) {
            // Only reachable if two merges ever drew the same id, which would be the stronger outcome and
            // not a failure.
            logger.info("W5 gate 2: the two merged files are identical byte for byte, {} B", a.length);
            return;
        }
        final int first = differing.get(0);
        final int last = differing.get(differing.size() - 1);
        logger.info(
            "W5 gate 2: {} bytes of {} differ, first at {}, last at {} (length - {})",
            differing.size(),
            a.length,
            first,
            last,
            a.length - last
        );
        // Two windows may differ and nothing between them: the id, which is inside the index header and so
        // within its reach, and the stored checksum, which is the last eight bytes. Everything else is the
        // vector region. The head window is bounded by the header's reach rather than by where the first
        // difference landed, because the two ids agree in their first byte about one time in 256 and a
        // window anchored on the first difference would then slide past the id and leave bytes unchecked.
        final List<Integer> head = new ArrayList<>();
        for (final int at : differing) {
            if (at >= a.length - Long.BYTES) {
                continue;
            }
            assertTrue("byte " + at + " of the merged .vec differs, and it is past the index header: it is vector data", at < HEADER_LIMIT);
            head.add(at);
        }
        if (head.isEmpty() == false) {
            assertTrue(
                "the differing header bytes must fit one segment id, they span " + (head.get(head.size() - 1) - head.get(0) + 1),
                head.get(head.size() - 1) - head.get(0) < SEGMENT_ID_LENGTH
            );
        }
    }

    /** Every vector of the one segment at {@code path}, in ordinal order, read through the delegate. */
    @SneakyThrows
    private static List<float[]> readAllVectors(final Path path) {
        final List<float[]> vectors = new ArrayList<>();
        try (MMapDirectory directory = new MMapDirectory(path); DirectoryReader reader = DirectoryReader.open(directory)) {
            assertEquals("the merged fixture must be one segment", 1, reader.leaves().size());
            final FloatVectorValues values = reader.leaves().get(0).reader().getFloatVectorValues(FIELD);
            assertNotNull(values);
            final KnnVectorValues.DocIndexIterator iterator = values.iterator();
            for (int doc = iterator.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = iterator.nextDoc()) {
                vectors.add(values.vectorValue(iterator.index()).clone());
            }
        }
        return vectors;
    }

    /** The one {@code .vec} that is not one of {@code sources}: the file the merge wrote. */
    private static String mergedVectorDataFileName(final Path path, final List<String> sources) {
        final List<String> after = new ArrayList<>(vectorDataFileNames(path));
        after.removeAll(sources);
        assertEquals("the merge must have left exactly one new .vec, saw " + after, 1, after.size());
        return after.get(0);
    }

    // -----------------------------------------------------------------------------------------------
    // Gate 3: one integrity check reads the file once, through the ring, and admits nothing
    // -----------------------------------------------------------------------------------------------

    /**
     * {@code checkIntegrity} is a forward scan, so the ring serves it: the bytes it reads from the device
     * are about the file's length and not a multiple of it, the blocking reads are the handful the arm and
     * the tail take, and the store's entry count does not move, because a ring read is not admitted.
     *
     * <p>The bound on the device bytes is two windows either side of the file length. One window is the
     * ring's own grain; the slack above it is there because the last window is clipped to the file's end
     * and the first reads happen before the ring arms, and the point of the assertion is that the file is
     * read <em>once</em> rather than that a particular number of requests was issued.
     *
     * <p>The store is turned on and shown to be live before the scan runs, by a single random read that is
     * admitted. Without that the entry count would be zero for the whole test whatever the ring did, and a
     * build that admitted every window would pass.
     */
    @SneakyThrows
    public void testCheckIntegrityReadsTheFileOnceThroughTheRing() {
        assumeDirectIOWorksHere();
        final Path path = createTempDir();
        try (MMapDirectory directory = new MMapDirectory(path)) {
            writeSegment(directory, DOCS, 20261033L);
        }
        final Path vectorData = path.resolve(onlyVectorDataFileName(path));
        final long length = Files.size(vectorData);
        assertTrue("the fixture must be several ring windows long, it is " + length + " B", length > 8 * WINDOW_BYTES);

        DirectIOVectorCache.resetForTesting();
        DirectIOVectorCache.setSlabBytesForTesting(TEST_SLAB_BYTES);
        DirectIOVectorCache.publishMaxBytes(STORE_BYTES);
        DirectIOVectorIndexInput.STATS.reset();
        assertTrue("the store must be on, or the assertion that nothing was admitted is vacuous", DirectIOVectorCache.maxBytes() > 0);
        assertEquals("the store starts empty", 0L, DirectIOVectorCache.entries());

        try (
            MMapDirectory delegate = new MMapDirectory(path);
            KNNDirectIODirectory storage = new KNNDirectIODirectory(delegate, "_0", () -> true, () -> 0L);
            DirectoryReader reader = DirectoryReader.open(storage)
        ) {
            assertEquals("the .vec must have been served with O_DIRECT, declined=" + storage.declinedOpens(), 1, storage.routedOpens());
            onlyCodecLeaf(reader).checkIntegrity();

            final long ringBytes = DirectIOVectorIndexInput.STATS.readaheadBytes();
            final long blockingBytes = DirectIOVectorIndexInput.STATS.blockingBytes();
            final long blockingReads = DirectIOVectorIndexInput.STATS.blockingReads();
            final long device = ringBytes + blockingBytes + DirectIOVectorIndexInput.STATS.stagedBytes();
            final long admitted = DirectIOVectorCache.entries();
            logger.info(
                "W5 gate 3: file={} B device={} B ring={} B blocking={} B in {} reads, armed={} ring_reads={} "
                    + "ring_hits={} declined={} entries={}",
                length,
                device,
                ringBytes,
                blockingBytes,
                blockingReads,
                DirectIOVectorIndexInput.STATS.readaheadArmed(),
                DirectIOVectorIndexInput.STATS.readaheadReads(),
                DirectIOVectorIndexInput.STATS.readaheadHits(),
                DirectIOVectorIndexInput.STATS.readaheadDeclined(),
                admitted
            );

            assertTrue("the forward scan must arm the ring", DirectIOVectorIndexInput.STATS.readaheadArmed() > 0);
            assertTrue("and serve the scan from it: ring bytes were " + ringBytes + " of " + length, ringBytes > length - 4 * WINDOW_BYTES);
            assertTrue(
                "the file must be read once, not twice: " + device + " B of device reads for a " + length + " B file",
                device <= length + 2 * WINDOW_BYTES
            );
            assertEquals("the ring must not have been refused a window", 0L, DirectIOVectorIndexInput.STATS.readaheadDeclined());
            assertTrue(
                "and the scan must not degenerate into per-read blocking reads: " + blockingReads + " blocking reads",
                blockingReads <= 8
            );
            assertEquals("a ring read must not be admitted to the store", 0L, admitted);

            // And the store was able to admit: the same file, the same store, read the way the rescore path
            // reads it, which is a declared read followed by its consumption. Without this the assertion
            // above would hold on a store that could not have admitted anything anyway. Several reads,
            // because the store allocates its slabs on demand and a fill requested while its segment is
            // still allocating is dropped, so the first miss admits nothing; scattered, so they cannot arm
            // the ring themselves.
            try (DirectIOVectorIndexInput declared = DirectIOVectorIndexInput.open(vectorData)) {
                assertNotNull(declared);
                final float[] scratch = new float[DIMENSION];
                final int vectorBytes = DIMENSION * Float.BYTES;
                for (final int part : new int[] { 8, 4, 12, 2, 10, 6, 14, 1 }) {
                    final long at = (length / 16 * part) / vectorBytes * vectorBytes;
                    declared.prefetch(at, vectorBytes);
                    declared.seek(at);
                    declared.readFloats(scratch, 0, DIMENSION);
                }
                assertTrue(
                    "a declared read must be admitted, or the store was not able to admit anything: entries="
                        + DirectIOVectorCache.entries()
                        + " slabs="
                        + DirectIOVectorCache.slabs()
                        + " misses="
                        + DirectIOVectorCache.misses(),
                    DirectIOVectorCache.entries() > 0
                );
            }
        }
    }
}

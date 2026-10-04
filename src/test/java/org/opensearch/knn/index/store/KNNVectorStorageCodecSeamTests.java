/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import com.carrotsearch.randomizedtesting.annotations.ThreadLeakFilters;
import lombok.SneakyThrows;
import org.apache.lucene.codecs.CodecUtil;
import org.apache.lucene.codecs.CompoundDirectory;
import org.apache.lucene.codecs.lucene90.Lucene90CompoundFormat;
import org.apache.lucene.index.FieldInfos;
import org.apache.lucene.index.SegmentInfo;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.store.DataAccessHint;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FileDataHint;
import org.apache.lucene.store.FileTypeHint;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.util.StringHelper;
import org.apache.lucene.util.Version;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.settings.ClusterSettings;
import org.opensearch.common.settings.Setting;
import org.opensearch.common.settings.Settings;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.codec.KNN990Codec.NativeEngines990KnnVectorsFormat;

import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.OpenOption;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * {@link KNNVectorStorageDirectory#routeVectorData}, the codec-level seam — since phase 14 the plugin's
 * <b>only</b> Direct I/O engagement point, reaching the {@code .vec} open with no index setting at all.
 *
 * <p>Three properties, because three things can go wrong with a seam placed here rather than at the store:
 * it can fire when the feature is off, it can fail to fire when it is on, and it can mis-handle a compound
 * segment — where the directory a codec is handed is not the shard's directory at all but the compound
 * reader, which has no filesystem path behind it. Each is asserted here; the third is asserted against a
 * <b>real</b> {@link Lucene90CompoundFormat} container rather than a stand-in, because "the decline is
 * structural" is exactly the claim being made.
 */
@ThreadLeakFilters(defaultFilters = true, filters = { DirectIOReadPoolTests.ReadPoolThreadFilter.class })
public class KNNVectorStorageCodecSeamTests extends KNNTestCase {

    /** A native-engine (faiss / MOS) fp32 flat file: the shape the rule routes. */
    private static final String VECTOR_DATA = "_0_" + NativeEngines990KnnVectorsFormat.FORMAT_NAME + "_0.vec";

    private static final int FILE_BYTES = 16 * 1024;

    @SneakyThrows
    private void assumeDirectIOWorksHere() {
        final OpenOption direct = DirectIOVectorIndexInput.directOpenOption();
        assumeTrue("this JDK has no ExtendedOpenOption.DIRECT", direct != null);
        final Path probe = createTempDir().resolve("probe");
        Files.write(probe, new byte[8192]);
        try (FileChannel channel = FileChannel.open(probe, StandardOpenOption.READ, direct)) {
            assertNotNull(channel);
        } catch (IOException | UnsupportedOperationException e) {
            assumeNoException("this filesystem refuses O_DIRECT", e);
        }
    }

    /**
     * The node settings this seam is gated on. The size floor is set to {@code 0b} because the fixture here
     * is a 16 KiB file and the property under test is the <em>engagement</em> — that the codec seam reaches
     * k-NN's dispatch with no index setting — not the floor, which is pinned where it belongs in
     * {@link KNNVectorStorageDirectoryTests#testAFileBelowTheSizeFloorIsNotRouted}. Reset by
     * {@code tearDown}.
     */
    private void enableDirectIORouting() {
        final Set<Setting<?>> settings = new HashSet<>(ClusterSettings.BUILT_IN_CLUSTER_SETTINGS);
        settings.addAll(
            KNNSettings.state().getSettings().stream().filter(s -> s.getProperties().contains(Setting.Property.NodeScope)).toList()
        );
        final Settings enabled = Settings.builder()
            .put(KNNSettings.KNN_DIRECT_IO_RESCORE_ENABLED, true)
            .put(KNNSettings.KNN_DIRECT_IO_MIN_FILE_SIZE, "0b")
            .build();
        final ClusterService clusterService = mock(ClusterService.class);
        when(clusterService.getClusterSettings()).thenReturn(new ClusterSettings(enabled, settings));
        KNNSettings.state().setClusterService(clusterService);
        assertTrue(KNNSettings.isDirectIORescoreEnabled());
    }

    private static byte[] contentOf(final int length) {
        final byte[] bytes = new byte[length];
        for (int i = 0; i < length; i++) {
            bytes[i] = (byte) (i * 31 + 7);
        }
        return bytes;
    }

    @SneakyThrows
    private static byte[] writeFile(final Directory directory, final String name, final int length) {
        final byte[] content = contentOf(length);
        try (IndexOutput output = directory.createOutput(name, IOContext.DEFAULT)) {
            output.writeBytes(content, content.length);
        }
        return content;
    }

    /**
     * As {@link #writeFile}, but a well-formed Lucene file: codec index header, payload, codec footer.
     * {@link Lucene90CompoundFormat#write} refuses a sub-file without them
     * ({@code CodecUtil.verifyAndCopyIndexHeader}), so the compound test cannot use raw bytes. Returns the
     * file's whole content as written, which is what a reader of the compound entry must see.
     */
    @SneakyThrows
    private static byte[] writeLuceneFile(final Directory directory, final String name, final SegmentInfo info, final int payload) {
        try (IndexOutput output = directory.createOutput(name, IOContext.DEFAULT)) {
            CodecUtil.writeIndexHeader(output, "KNNCodecSeamProbe", 0, info.getId(), "");
            output.writeBytes(contentOf(payload), payload);
            CodecUtil.writeFooter(output);
        }
        try (IndexInput input = directory.openInput(name, IOContext.DEFAULT)) {
            return readAll(input);
        }
    }

    private static byte[] readAll(final IndexInput input) throws IOException {
        final byte[] bytes = new byte[Math.toIntExact(input.length())];
        input.seek(0);
        input.readBytes(bytes, 0, bytes.length);
        return bytes;
    }

    /** The context Lucene's flat vectors reader builds for its own {@code .vec} open. */
    private static IOContext vectorDataContext() {
        return IOContext.DEFAULT.withHints(FileTypeHint.DATA, FileDataHint.KNN_VECTORS, DataAccessHint.RANDOM);
    }

    private static SegmentInfo segmentInfoOver(final Directory directory, final boolean compound, final Set<String> files) {
        final SegmentInfo info = new SegmentInfo(
            directory,
            Version.LATEST,
            Version.LATEST,
            "_0",
            1,
            compound,
            false,
            null,
            Collections.emptyMap(),
            StringHelper.randomId(),
            Collections.emptyMap(),
            null
        );
        info.setFiles(files);
        return info;
    }

    private static SegmentReadState readStateOver(final Directory directory, final SegmentInfo info) {
        return new SegmentReadState(directory, info, FieldInfos.EMPTY, vectorDataContext(), "_0");
    }

    // -------------------------------------------------------------------------------------------------

    /** The feature off: the seam must hand back the caller's own state, not a wrapper of it. */
    @SneakyThrows
    public void testTheSeamIsInertWhenTheSettingIsOff() {
        assertFalse("this test's premise is the shipped default", KNNSettings.isDirectIORescoreEnabled());
        try (MMapDirectory directory = new MMapDirectory(createTempDir())) {
            writeFile(directory, VECTOR_DATA, FILE_BYTES);
            final SegmentReadState state = readStateOver(directory, segmentInfoOver(directory, false, Set.of(VECTOR_DATA)));
            assertSame(
                "the seam must be inert, not merely harmless, when the feature is off",
                state,
                KNNVectorStorageDirectory.routeVectorData(state)
            );
        }
    }

    /**
     * The whole point of the seam: with the setting on and <b>no index setting anywhere</b>, a faiss
     * {@code .vec} opened through the derived state is served with {@code O_DIRECT} and reads the same
     * bytes. Everything else on the state is carried through unchanged, because a codec that found a
     * different {@code segmentSuffix} would read a different field's file.
     */
    @SneakyThrows
    public void testTheDerivedStateServesAFaissVecWithDirectIOAndReadsTheSameBytes() {
        assumeDirectIOWorksHere();
        enableDirectIORouting();
        try (MMapDirectory directory = new MMapDirectory(createTempDir())) {
            final byte[] expected = writeFile(directory, VECTOR_DATA, FILE_BYTES);
            final SegmentInfo info = segmentInfoOver(directory, false, Set.of(VECTOR_DATA));
            final SegmentReadState state = readStateOver(directory, info);

            final SegmentReadState derived = KNNVectorStorageDirectory.routeVectorData(state);
            assertNotSame(state, derived);
            assertSame("the segment info must be carried through", state.segmentInfo, derived.segmentInfo);
            assertSame("the field infos must be carried through", state.fieldInfos, derived.fieldInfos);
            assertSame("the context must be carried through", state.context, derived.context);
            assertEquals("the segment suffix must be carried through", state.segmentSuffix, derived.segmentSuffix);

            assertTrue(
                "the derived state's directory must be k-NN's, got " + derived.directory.getClass().getName(),
                derived.directory instanceof KNNVectorStorageDirectory
            );
            final KNNVectorStorageDirectory storage = (KNNVectorStorageDirectory) derived.directory;
            assertSame("it must wrap the caller's directory, not replace it", directory, storage.getDelegate());
            assertEquals("the directory names the segment it serves, not an index", info.name, storage.segmentName());

            try (IndexInput routed = derived.directory.openInput(VECTOR_DATA, vectorDataContext())) {
                assertTrue(
                    "the codec seam must reach the same O_DIRECT input the store factory reaches, got " + routed.getClass().getName(),
                    routed instanceof DirectIOVectorIndexInput
                );
                assertArrayEquals("the codec seam must read the delegate's bytes exactly", expected, readAll(routed));
            }
            assertEquals("the routed-open counter must fire", 1, storage.routedOpens());
            assertEquals(0, storage.declinedOpens());
        }
    }

    /**
     * A compound segment, against a real {@link Lucene90CompoundFormat} container. The directory a codec
     * is handed for a compound segment is the {@link CompoundDirectory}, which is <b>not</b> a
     * {@code FilterDirectory}, so k-NN's path resolution stops there and the open is <b>declined</b> — the
     * entry is served by the compound reader's own mmap slice, and the bytes are still right.
     *
     * <p>This is the measured right answer rather than a gap: a compound segment is a freshly flushed one,
     * small, and its working set belongs in the page cache — phase 13's E4 measured a flush-sized
     * {@code .vec} at 64.7 ops/s on {@code O_DIRECT} against 518,681 ops/s from the page cache. Resolving
     * {@code .cfe} offsets inside the codec would mean re-implementing the compound reader's entry table
     * one layer above it, in order to route reads that must not be routed.
     */
    @SneakyThrows
    public void testACompoundSegmentDeclinesAndStillReadsTheRightBytes() {
        assumeDirectIOWorksHere();
        enableDirectIORouting();
        try (MMapDirectory directory = new MMapDirectory(createTempDir())) {
            final SegmentInfo info = segmentInfoOver(directory, true, Set.of(VECTOR_DATA));
            final byte[] expected = writeLuceneFile(directory, VECTOR_DATA, info, FILE_BYTES);
            new Lucene90CompoundFormat().write(directory, info, IOContext.DEFAULT);

            try (CompoundDirectory compound = new Lucene90CompoundFormat().getCompoundReader(directory, info)) {
                final SegmentReadState state = readStateOver(compound, info);
                final SegmentReadState derived = KNNVectorStorageDirectory.routeVectorData(state);
                final KNNVectorStorageDirectory storage = (KNNVectorStorageDirectory) derived.directory;

                try (IndexInput entry = derived.directory.openInput(VECTOR_DATA, vectorDataContext())) {
                    assertFalse(
                        "a compound entry has no path of its own, so the seam must decline rather than fail",
                        entry instanceof DirectIOVectorIndexInput
                    );
                    assertArrayEquals("declining must not change the bytes", expected, readAll(entry));
                }
                assertEquals("the decline must be counted, not silent", 1, storage.declinedOpens());
                assertEquals("and its reason is the chain: a CompoundDirectory is not a FilterDirectory", 1, storage.declinedChain());
                assertEquals(0, storage.routedOpens());
            }
        }
    }
}

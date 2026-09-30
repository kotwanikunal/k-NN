/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import org.apache.lucene.codecs.CompoundDirectory;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.DataAccessHint;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FileDataHint;
import org.apache.lucene.store.FileTypeHint;
import org.apache.lucene.store.FilterDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.store.MMapDirectory;
import org.apache.lucene.store.MergeInfo;
import org.apache.lucene.store.ReadAdvice;
import org.opensearch.Version;
import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.common.settings.Settings;
import org.opensearch.core.index.shard.ShardId;
import org.opensearch.env.ShardLock;
import org.opensearch.index.IndexSettings;
import org.opensearch.index.store.Store;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.codec.KNN80Codec.KNN80CompoundDirectory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.CopyOnWriteArrayList;

/**
 * The first half of the directory design: an intent named at the codec layer has to arrive, intact, at
 * the deepest {@link Directory} a plugin can supply.
 *
 * <p>On a non-compound segment {@code Directory.openInput} is the only channel that can still change how
 * bytes are fetched — a three-argument {@code slice} takes no {@link IOContext}, and
 * {@code IndexInput.updateIOContext} can only re-advise a mapping it already made. So the design needs a
 * plugin-authored {@code openInput}, and needs the hint on it to survive {@code Store$StoreDirectory} and
 * {@code ByteSizeCachingDirectory}, which a real {@link Store} puts above whatever directory it is handed.
 *
 * <p>Inside a compound file there is a second channel, and it is the one the compound route uses: the
 * context-carrying four-argument {@code slice} does have a call site —
 * {@code Lucene90CompoundReader#openInput} — so the container's {@code IndexInput} is handed each entry
 * by name with the caller's context. See {@link KNNVectorCompoundSliceInputTests}.
 *
 * <p>These tests use a real {@link Store} rather than hand-stacked wrappers precisely so that the
 * classes under test are the server's own, at the version on the classpath. The directory below the
 * store is a local recorder rather than {@link KNNVectorStorageDirectory}: what is under test here is
 * the <em>signal</em>, so the assertion has to be over the {@link IOContext} that arrived, which the
 * production directory deliberately does not retain. Its dispatch rule is pinned separately in
 * {@link KNNVectorStorageDirectoryTests}.
 */
public class KNNVectorReadIntentTests extends KNNTestCase {

    private static final String VEC_FILE = "_0_Lucene99FlatVectorsFormat_0.vec";

    /** What reached the plugin's directory, in the terms a dispatch rule would be written in. */
    private record Seen(String name, Set<IOContext.FileOpenHint> hints, IOContext.Context context, KNNVectorReadIntent intent) {
    }

    /** Records the {@code (name, context)} of every {@code openInput} that reaches it, then delegates. */
    private static final class RecordingDirectory extends FilterDirectory {

        private final List<Seen> seen = new CopyOnWriteArrayList<>();

        private RecordingDirectory(final Directory delegate) {
            super(delegate);
        }

        @Override
        public IndexInput openInput(final String name, final IOContext context) throws IOException {
            seen.add(new Seen(name, context.hints(), context.context(), KNNVectorReadIntent.of(context)));
            return in.openInput(name, context);
        }

        private List<Seen> seen() {
            return List.copyOf(seen);
        }

        private List<Seen> vectorData() {
            return seen.stream().filter(s -> s.name().endsWith(".vec")).toList();
        }

        private void clear() {
            seen.clear();
        }
    }

    /**
     * The chain from {@code directory} down, as simple class names. {@link FilterDirectory#unwrap} answers
     * only what is at the bottom; the design needs to know what is in between, because any wrapper that
     * rebuilt the {@link IOContext} rather than passing the caller's through would break the intent
     * signal. Walking {@link FilterDirectory#getDelegate()} is the only way to see them from outside the
     * server.
     */
    private static List<String> chainOf(final Directory directory) {
        final List<String> chain = new ArrayList<>();
        Directory current = directory;
        for (int depth = 0; current != null && depth < 16; depth++) {
            chain.add(current.getClass().getSimpleName());
            if (current instanceof FilterDirectory filterDirectory) {
                current = filterDirectory.getDelegate();
            } else {
                break;
            }
        }
        return List.copyOf(chain);
    }

    /** A {@link ShardLock} that owns nothing, for a {@link Store} over a directory with no shard behind it. */
    private static ShardLock noopShardLock(final ShardId shardId) {
        return new ShardLock(shardId) {
            @Override
            protected void closeInternal() {}
        };
    }

    private static IndexSettings indexSettings() {
        final Settings settings = Settings.builder()
            .put(IndexMetadata.SETTING_VERSION_CREATED, Version.CURRENT)
            .put(IndexMetadata.SETTING_NUMBER_OF_SHARDS, 1)
            .put(IndexMetadata.SETTING_NUMBER_OF_REPLICAS, 0)
            .build();
        return new IndexSettings(IndexMetadata.builder("test-index").settings(settings).build(), Settings.EMPTY);
    }

    private static void writeEmptyFile(final Directory directory, final String name) throws IOException {
        try (IndexOutput output = directory.createOutput(name, IOContext.DEFAULT)) {
            output.writeInt(0);
        }
    }

    /**
     * The gate itself. A {@code .vec} open issued with {@link KNNVectorReadIntent#RESCORE} through the
     * directory a real {@link Store} exposes must reach the plugin's directory with the same four
     * hints it was given.
     */
    public void testRescoreIntentSurvivesTheStoreWrapperChain() throws IOException {
        final ShardId shardId = new ShardId("test-index", "_na_", 0);
        final RecordingDirectory plugin = new RecordingDirectory(new ByteBuffersDirectory());
        writeEmptyFile(plugin, VEC_FILE);
        plugin.clear();

        try (Store store = new Store(shardId, indexSettings(), plugin, noopShardLock(shardId))) {
            // Exactly what the codec would do, through exactly the directory the codec is handed.
            try (IndexInput input = store.directory().openInput(VEC_FILE, KNNVectorReadIntent.RESCORE.vectorDataContext())) {
                assertNotNull(input);
            }

            final List<Seen> seen = plugin.vectorData();
            assertEquals("expected exactly one .vec open to reach the plugin directory", 1, seen.size());
            final Seen observation = seen.get(0);

            assertEquals(VEC_FILE, observation.name());
            assertEquals(KNNVectorReadIntent.RESCORE, observation.intent());
            assertEquals(IOContext.Context.DEFAULT, observation.context());
            assertEquals(
                "the wrapper chain must not add, drop or replace a hint",
                Set.of(FileTypeHint.DATA, FileDataHint.KNN_VECTORS, DataAccessHint.RANDOM, KNNVectorReadIntent.RESCORE),
                observation.hints()
            );
        }
    }

    /**
     * The chain the gate is about, named. If a future OpenSearch inserts a wrapper that rebuilds the
     * context, this is the assertion that will start failing alongside the one above, which is what
     * makes the failure diagnosable.
     */
    public void testStoreExposesTheExpectedWrapperChainAboveThePluginDirectory() throws IOException {
        final ShardId shardId = new ShardId("test-index", "_na_", 0);
        final RecordingDirectory plugin = new RecordingDirectory(new ByteBuffersDirectory());

        try (Store store = new Store(shardId, indexSettings(), plugin, noopShardLock(shardId))) {
            final List<String> chain = chainOf(store.directory());
            assertEquals(
                "chain was " + chain,
                List.of("StoreDirectory", "ByteSizeCachingDirectory", "RecordingDirectory", "ByteBuffersDirectory"),
                chain
            );
        }
    }

    /**
     * A context Lucene built for its own {@code .vec} open carries no intent, so the directory can tell
     * a plugin-authored read apart from every other read of the same file. This is the whole reason the
     * hint is a plugin type rather than a {@link FileDataHint}, whose two constants Lucene already
     * attaches to both {@code .vec} and the native index file.
     */
    public void testLuceneAuthoredContextsCarryNoIntent() {
        final IOContext luceneVectorDataContext = IOContext.DEFAULT.withHints(
            FileTypeHint.DATA,
            FileDataHint.KNN_VECTORS,
            DataAccessHint.RANDOM
        );
        assertNull(KNNVectorReadIntent.of(luceneVectorDataContext));
        assertNull(KNNVectorReadIntent.of(IOContext.DEFAULT));
        assertNull(KNNVectorReadIntent.of(IOContext.READONCE));
    }

    /**
     * A merge context cannot be made to carry the intent at all: {@code IOContext.merge(..)} ignores
     * {@code withHints}. So "a merge read must never be routed to Direct I/O" is a property of Lucene's
     * API rather than a check the dispatch rule has to get right — and, symmetrically, no amount of
     * hinting could route a merge if that were ever wanted.
     */
    public void testMergeContextCannotCarryTheIntent() {
        final IOContext merge = IOContext.merge(new MergeInfo(1, 1L, false, 1));
        final IOContext hinted = merge.withHints(FileTypeHint.DATA, FileDataHint.KNN_VECTORS, KNNVectorReadIntent.RESCORE);

        assertSame("withHints on a merge context must be inert", merge, hinted);
        assertTrue(hinted.hints().isEmpty());
        assertNull(KNNVectorReadIntent.of(hinted));
        assertEquals(IOContext.Context.MERGE, hinted.context());
    }

    /**
     * Attaching the intent must not change how a directory that does not know about it behaves. The
     * one thing on this branch that reads hints is {@code MMapDirectory.ADVISE_BY_CONTEXT}, which
     * OpenSearch installs on the default store type, so the read advice it derives has to be identical
     * with and without the intent.
     */
    public void testIntentDoesNotChangeTheReadAdviceMMapDirectoryDerives() {
        final IOContext withoutIntent = IOContext.DEFAULT.withHints(FileTypeHint.DATA, FileDataHint.KNN_VECTORS, DataAccessHint.RANDOM);
        final IOContext withIntent = KNNVectorReadIntent.RESCORE.vectorDataContext();

        final Optional<ReadAdvice> before = MMapDirectory.ADVISE_BY_CONTEXT.apply(VEC_FILE, withoutIntent);
        final Optional<ReadAdvice> after = MMapDirectory.ADVISE_BY_CONTEXT.apply(VEC_FILE, withIntent);

        assertEquals(Optional.of(ReadAdvice.RANDOM), before);
        assertEquals("an unrecognised hint must be inert to advice selection", before, after);
    }

    /**
     * At most one intent can ride on a context. {@code DefaultIOContext} rejects two hints of the same
     * class, and an enum constant with no class body reports the enum itself, so a caller cannot
     * accidentally ship an ambiguous pair — the dispatch rule never has to break a tie.
     */
    public void testAtMostOneIntentPerContext() {
        assertSame(KNNVectorReadIntent.class, KNNVectorReadIntent.RESCORE.getClass());
        final IOContext context = KNNVectorReadIntent.RESCORE.vectorDataContext();
        assertEquals(1, context.hints(KNNVectorReadIntent.class).count());
    }

    /**
     * The file suffix and the read intent are two independent facts, and the directory needs both. Suffix
     * alone cannot tell a re-score read of {@code .vec} from a traversal or warmup read of the same file,
     * which is the obstacle the intent is there to remove; so an unhinted {@code .vec} open and a hinted
     * one must arrive distinguishable, and a caller must not collapse them.
     */
    public void testFileSuffixAloneCannotSeparateAReadIntent() throws IOException {
        try (RecordingDirectory plugin = new RecordingDirectory(new ByteBuffersDirectory())) {
            writeEmptyFile(plugin, VEC_FILE);
            writeEmptyFile(plugin, "segments_1");
            plugin.clear();

            // An unhinted .vec open: what Lucene's own reader issues, and what traversal, warmup and
            // derived-source reconstruction all arrive as.
            plugin.openInput(VEC_FILE, IOContext.DEFAULT.withHints(FileTypeHint.DATA, FileDataHint.KNN_VECTORS, DataAccessHint.RANDOM))
                .close();
            // A hinted one: the re-score read.
            plugin.openInput(VEC_FILE, KNNVectorReadIntent.RESCORE.vectorDataContext()).close();
            // A file the design would never route.
            plugin.openInput("segments_1", IOContext.DEFAULT).close();

            final List<Seen> vectorData = plugin.vectorData();
            assertEquals(2, vectorData.size());
            assertNull("an unhinted .vec open is indistinguishable by suffix alone", vectorData.get(0).intent());
            assertEquals(KNNVectorReadIntent.RESCORE, vectorData.get(1).intent());
            assertEquals("segments_1 must not be counted as vector data", 3, plugin.seen().size());
        }
    }

    /**
     * The limit of <em>this</em> channel on a compound segment, which is why the design has a second
     * dispatch point at all. {@code KNN1040Codec.compoundFormat()} wraps Lucene's compound reader in a
     * {@link KNN80CompoundDirectory}, which extends {@link CompoundDirectory} — a bare {@link Directory},
     * not a {@link FilterDirectory} — so there is no {@code getDelegate()} to walk from a compound
     * segment's read state, and no {@code openInput} named {@code .vec} arrives at the plugin's directory.
     *
     * <p><b>That is not the compound-segment answer, only this channel's.</b> The compound reader below is
     * a stand-in that serves its entries from a nested {@link Directory}; Lucene's real one slices the
     * {@code .cfs} handle it opened <em>on the outer directory</em>, using the context-carrying
     * four-argument {@code slice}, so both the name and the intent do reach the plugin — at its
     * {@code IndexInput} rather than at its {@code Directory}. That is asserted against the real compound
     * reader in {@link KNNVectorCompoundSliceInputTests}.
     *
     * <p>The other thing that survives is recorded too: {@link KNN80CompoundDirectory} keeps a reference
     * to the outer directory, so the store chain is reachable from a compound segment even though the
     * walk is not.
     */
    public void testCompoundSegmentBreaksTheWalkToThePluginDirectory() throws IOException {
        final RecordingDirectory plugin = new RecordingDirectory(new ByteBuffersDirectory());
        writeEmptyFile(plugin, VEC_FILE);
        plugin.clear();

        // Stands in for Lucene's compound reader: it serves the segment's files out of a container it
        // opened itself, which is exactly why the name ".vec" never reaches the directory below.
        final Directory insideTheCompoundFile = new ByteBuffersDirectory();
        writeEmptyFile(insideTheCompoundFile, VEC_FILE);
        final CompoundDirectory luceneCompoundReader = new CompoundDirectory() {
            @Override
            public void checkIntegrity() {}

            @Override
            public String[] listAll() throws IOException {
                return insideTheCompoundFile.listAll();
            }

            @Override
            public long fileLength(final String name) throws IOException {
                return insideTheCompoundFile.fileLength(name);
            }

            @Override
            public IndexInput openInput(final String name, final IOContext context) throws IOException {
                return insideTheCompoundFile.openInput(name, context);
            }

            @Override
            public Set<String> getPendingDeletions() throws IOException {
                return insideTheCompoundFile.getPendingDeletions();
            }

            @Override
            public void close() throws IOException {
                insideTheCompoundFile.close();
            }
        };

        // What SegmentReadState.directory is for a compound segment on this branch.
        final Directory compoundSegmentDirectory = new KNN80CompoundDirectory(luceneCompoundReader, plugin);

        assertFalse("a compound segment's directory is not a FilterDirectory", compoundSegmentDirectory instanceof FilterDirectory);
        assertNull(
            "the getDelegate() walk cannot reach the plugin directory from a compound segment",
            KNNVectorStorageDirectory.find(compoundSegmentDirectory)
        );
        assertEquals(List.of("KNN80CompoundDirectory"), chainOf(compoundSegmentDirectory));

        compoundSegmentDirectory.openInput(VEC_FILE, KNNVectorReadIntent.RESCORE.vectorDataContext()).close();
        assertEquals(
            "a .vec read of a compound segment does not arrive at the plugin directory as a .vec openInput; "
                + "it arrives at the plugin's IndexInput for the .cfs as a slice -- see KNNVectorCompoundSliceInputTests",
            List.of(),
            plugin.vectorData()
        );

        assertSame(
            "the outer directory is still reachable, which is the only route a compound design has",
            plugin,
            ((KNN80CompoundDirectory) compoundSegmentDirectory).getDir()
        );

        compoundSegmentDirectory.close();
        plugin.close();
    }
}

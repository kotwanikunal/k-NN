/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import lombok.SneakyThrows;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.DataAccessHint;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FileDataHint;
import org.apache.lucene.store.FileTypeHint;
import org.apache.lucene.store.FilterDirectory;
import org.apache.lucene.store.FlushInfo;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.store.MergeInfo;
import org.opensearch.knn.KNNTestCase;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Pins what {@link KNNRescoreIntentDirectory} adds and, more importantly, what it must not change.
 *
 * <p>The class is two lines of logic, and both of them are about an API detail that fails silently if
 * it is got wrong: the intent must reach the {@code .vec} open, and the hints Lucene put on that
 * context must still be there when it does, because {@code withHints} replaces the set rather than
 * merging into it.
 */
public class KNNRescoreIntentDirectoryTests extends KNNTestCase {

    /** A directory that records the context of every {@code openInput} and then delegates unchanged. */
    private static final class RecordingDirectory extends FilterDirectory {
        private final List<IOContext> contexts = new ArrayList<>();
        private final List<String> names = new ArrayList<>();

        RecordingDirectory(final Directory delegate) {
            super(delegate);
        }

        @Override
        public IndexInput openInput(final String name, final IOContext context) throws IOException {
            names.add(name);
            contexts.add(context);
            return in.openInput(name, context);
        }

        IOContext contextFor(final String name) {
            final int index = names.indexOf(name);
            assertTrue("no openInput recorded for " + name + ", saw " + names, index >= 0);
            return contexts.get(index);
        }
    }

    private static void writeFile(final Directory directory, final String name) throws IOException {
        try (IndexOutput output = directory.createOutput(name, IOContext.DEFAULT)) {
            output.writeInt(42);
        }
    }

    /**
     * Lucene's own {@code .vec} context, reproduced: this is the hint set
     * {@code Lucene99FlatVectorsReader} builds before opening the data file, and the one the view has to
     * carry through intact.
     */
    private static IOContext luceneVectorDataContext() {
        return IOContext.DEFAULT.withHints(FileTypeHint.DATA, FileDataHint.KNN_VECTORS, DataAccessHint.RANDOM);
    }

    @SneakyThrows
    public void testVectorDataOpen_carriesTheRescoreIntent() {
        try (Directory base = new ByteBuffersDirectory()) {
            writeFile(base, "_0_KNN_0.vec");
            final RecordingDirectory recording = new RecordingDirectory(base);
            try (Directory view = new KNNRescoreIntentDirectory(recording)) {
                view.openInput("_0_KNN_0.vec", luceneVectorDataContext()).close();
            }
            assertEquals(KNNVectorReadIntent.RESCORE, KNNVectorReadIntent.of(recording.contextFor("_0_KNN_0.vec")));
        }
    }

    /**
     * The whole reason this class exists rather than a tagged {@code SegmentReadState}: hints do not
     * merge. If the view forgot to restate Lucene's three, the mapping below would be advised
     * differently and the default read path would change behaviour — with nothing throwing.
     */
    @SneakyThrows
    public void testVectorDataOpen_keepsEveryHintLuceneSet() {
        try (Directory base = new ByteBuffersDirectory()) {
            writeFile(base, "_0_KNN_0.vec");
            final RecordingDirectory recording = new RecordingDirectory(base);
            try (Directory view = new KNNRescoreIntentDirectory(recording)) {
                view.openInput("_0_KNN_0.vec", luceneVectorDataContext()).close();
            }
            final IOContext seen = recording.contextFor("_0_KNN_0.vec");
            assertTrue("FileTypeHint.DATA was dropped, hints=" + seen.hints(), seen.hints().contains(FileTypeHint.DATA));
            assertTrue("FileDataHint.KNN_VECTORS was dropped, hints=" + seen.hints(), seen.hints().contains(FileDataHint.KNN_VECTORS));
            assertTrue("DataAccessHint.RANDOM was dropped, hints=" + seen.hints(), seen.hints().contains(DataAccessHint.RANDOM));
            assertEquals("exactly one hint should have been added", 4, seen.hints().size());
        }
    }

    /**
     * Every other file the same reader opens has to arrive exactly as it would without the view: the
     * quantized codes are the traversal path's hot file and must never be routed anywhere.
     */
    @SneakyThrows
    public void testEveryOtherFileIsOpenedWithTheCallersOwnContext() {
        try (Directory base = new ByteBuffersDirectory()) {
            for (final String name : List.of("_0_KNN_0.veq", "_0_KNN_0.vemf", "_0.cfs", "_0.si")) {
                writeFile(base, name);
            }
            final RecordingDirectory recording = new RecordingDirectory(base);
            final IOContext caller = luceneVectorDataContext();
            try (Directory view = new KNNRescoreIntentDirectory(recording)) {
                for (final String name : List.of("_0_KNN_0.veq", "_0_KNN_0.vemf", "_0.cfs", "_0.si")) {
                    view.openInput(name, caller).close();
                    assertSame("the caller's own context should have been passed through for " + name, caller, recording.contextFor(name));
                    assertNull(KNNVectorReadIntent.of(recording.contextFor(name)));
                }
            }
        }
    }

    /**
     * A merge read must never carry the intent. It cannot, and the reason is Lucene's: a merge context's
     * {@code withHints} returns itself with an empty hint set. Asserted here because the design leans on
     * it in place of a check.
     */
    public void testMergeAndFlushContextsCannotCarryTheIntent() {
        final IOContext merge = IOContext.merge(new MergeInfo(10, 1024L, true, 1));
        final IOContext flush = IOContext.flush(new FlushInfo(10, 1024L));
        for (final IOContext context : List.of(merge, flush)) {
            final IOContext tagged = KNNRescoreIntentDirectory.withRescoreIntent("_0_KNN_0.vec", context);
            assertNull("a " + context.context() + " read must not be taggable", KNNVectorReadIntent.of(tagged));
            assertTrue(tagged.hints().isEmpty());
        }
    }

    /**
     * {@code DefaultIOContext} rejects two hints of the same class, so a context that already carries an
     * intent has to be passed through rather than added to. Reachable in practice: the plugin's own
     * probe issues a {@code .vec} open that is already tagged.
     */
    public void testAnAlreadyTaggedContextIsPassedThroughUnchanged() {
        final IOContext tagged = KNNVectorReadIntent.RESCORE.vectorDataContext();
        assertSame(tagged, KNNRescoreIntentDirectory.withRescoreIntent("_0_KNN_0.vec", tagged));
    }

    /** A view over a directory is still that directory for everything but the one context it edits. */
    @SneakyThrows
    public void testTheViewIsOtherwiseTransparent() {
        try (Directory base = new ByteBuffersDirectory()) {
            writeFile(base, "_0_KNN_0.vec");
            try (Directory view = new KNNRescoreIntentDirectory(base)) {
                assertEquals(base.listAll().length, view.listAll().length);
                assertEquals(base.fileLength("_0_KNN_0.vec"), view.fileLength("_0_KNN_0.vec"));
                try (IndexInput input = view.openInput("_0_KNN_0.vec", luceneVectorDataContext())) {
                    assertEquals(42, input.readInt());
                }
                assertSame(base, FilterDirectory.unwrap(view));
            }
        }
    }
}

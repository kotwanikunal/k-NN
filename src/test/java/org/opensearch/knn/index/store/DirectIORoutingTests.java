/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import org.apache.lucene.codecs.lucene95.HasIndexSlice;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.MemorySegmentAccessInput;
import org.opensearch.action.admin.indices.forcemerge.ForceMergeRequest;
import org.opensearch.action.bulk.BulkRequest;
import org.opensearch.action.bulk.BulkResponse;
import org.opensearch.action.index.IndexRequest;
import org.opensearch.action.support.WriteRequest;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.index.IndexService;
import org.opensearch.index.engine.Engine;
import org.opensearch.index.shard.IndexShard;
import org.opensearch.knn.KNNSingleNodeTestCase;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.engine.KNNEngine;

import java.io.IOException;
import java.util.Random;

/**
 * Routing in a real node. The node runs in the test JVM, so the shard's directory chain, the shard path the index
 * UUID is read from and the segment reader are the production ones, which no hand-built unit test reaches. Each case
 * reads which input a field's vectors come from the way Lucene's memory-segment scorer and k-NN's {@code WarmupUtil}
 * do: {@code ((HasIndexSlice) leafReader.getFloatVectorValues(field)).getSlice()} is a
 * {@link DirectIOVectorIndexInput} while the index is on, and the original input's own slice, a
 * {@link MemorySegmentAccessInput}, while it is off.
 *
 * <p>The base class starts a node for each test, so each case checks several fields or indices where that reads
 * naturally.
 */
public class DirectIORoutingTests extends KNNSingleNodeTestCase {

    /** 8,192 floats is 32,768 bytes, so 321 documents is 10,518,528 bytes, just over the 10 MiB floor. */
    private static final int DIMENSION = 8192;
    private static final int DOC_COUNT = 321;

    private static final String FIELD = "target_field";
    private static final String SECOND_FIELD = "second_field";
    private static final String LUCENE_FIELD = "lucene_field";

    @Override
    public void setUp() throws Exception {
        super.setUp();
        DirectIOTestUtils.assumeDirectIOSupported();
    }

    public void testAnIndexThatIsOnGivesADirectIOSlice() throws Exception {
        final IndexService indexService = createRoutableIndex("direct-io-routing-on", true, KNNEngine.FAISS);
        assertTrue(
            "a faiss .vec over the floor in a non-compound segment of an index that is on must be read with O_DIRECT",
            readsWithDirectIO(indexService, FIELD)
        );
        assertEquals("the fixture must be one merged segment, or the check above is about the wrong file", 1, leafCount(indexService));
    }

    public void testAnIndexThatIsOffGivesTheOriginalInputsSlice() throws Exception {
        final IndexService indexService = createRoutableIndex("direct-io-routing-off", false, KNNEngine.FAISS);
        // readsWithDirectIO asserts that a slice which is not a Direct I/O reader is the original input's own, a
        // MemorySegmentAccessInput, so this is both halves of the statement.
        assertFalse("an index that is off must read as main does", readsWithDirectIO(indexService, FIELD));
    }

    public void testTheFlipReachesTheOpenSegmentBothWays() throws Exception {
        final IndexService indexService = createRoutableIndex("direct-io-routing-flip", false, KNNEngine.FAISS);
        // The same open segment throughout: no refresh, no close and open, only the settings update.
        assertFalse(readsWithDirectIO(indexService, FIELD));
        setDirectIO("direct-io-routing-flip", true);
        assertTrue("the flip to on reaches a segment that is already open", readsWithDirectIO(indexService, FIELD));
        setDirectIO("direct-io-routing-flip", false);
        assertFalse("and the flip back to off stops it", readsWithDirectIO(indexService, FIELD));
    }

    public void testTwoIndicesOneOnOneOffAndTheWildcardPut() throws Exception {
        final IndexService on = createRoutableIndex("direct-io-routing-pair-on", true, KNNEngine.FAISS);
        final IndexService off = createRoutableIndex("direct-io-routing-pair-off", false, KNNEngine.FAISS);
        assertTrue("the index that is on reads with O_DIRECT", readsWithDirectIO(on, FIELD));
        assertFalse("the index that is off does not", readsWithDirectIO(off, FIELD));

        // The rollback the design asks for: one wildcard settings update turns every index off.
        setDirectIO("*", false);
        assertFalse("the wildcard put turns the on index off", readsWithDirectIO(on, FIELD));
        assertFalse(readsWithDirectIO(off, FIELD));
    }

    public void testASegmentThatFailsACheckAtOpenGivesNoDirectIOSlice() throws Exception {
        // Not force merged, so the index is several flushed segments, each a small fraction of it and so compound at
        // the shipped defaults, and each well under the floor on its own. Which of the two checks a given segment
        // fails is not fixed, so this case asserts the combined statement, and the two are isolated one at a time by
        // KNNDirectIODirectoryTests.
        final IndexService flushed = createIndexWithVectors(
            "direct-io-routing-flushed",
            true,
            KNNEngine.FAISS,
            DIMENSION,
            DOC_COUNT,
            false
        );
        assertTrue("the fixture must be more than one flushed segment", leafCount(flushed) > 1);
        assertFalse("a flushed segment must not be read with O_DIRECT", readsWithDirectIO(flushed, FIELD));

        // Force merged to one segment, so not compound, but 16 by 100 floats is 6,400 bytes, far under the
        // 10,485,760 byte floor, which is the only check left that can fail.
        final IndexService small = createIndexWithVectors("direct-io-routing-small", true, KNNEngine.FAISS, 16, 100, true);
        assertEquals("the small fixture must be one merged segment", 1, leafCount(small));
        assertFalse("a file under the floor must not be read with O_DIRECT", readsWithDirectIO(small, FIELD));
    }

    public void testOnlyFaissFieldsGetADirectIOSlice() throws Exception {
        // One index, one segment, three vector fields: a Lucene engine field, whose format never hands its files to
        // the delegate, and two faiss fields, which both route. That the choice is per file inside one segment is
        // what this case exists to show.
        final String index = "direct-io-routing-mixed";
        final IndexService mixed = createKnnIndexService(index, true);
        createKnnMapping(index, LUCENE_FIELD, DIMENSION, KNNEngine.LUCENE);
        createKnnMapping(index, FIELD, DIMENSION, KNNEngine.FAISS);
        createKnnMapping(index, SECOND_FIELD, DIMENSION, KNNEngine.FAISS);
        indexVectors(index, DIMENSION, DOC_COUNT, LUCENE_FIELD, FIELD, SECOND_FIELD);
        forceMergeToOneSegment(index);

        assertEquals("one segment, so one file per field", 1, leafCount(mixed));
        assertFalse("a Lucene engine .vec is never offered to the delegate", readsWithDirectIO(mixed, LUCENE_FIELD));
        assertTrue("the first faiss field is read with O_DIRECT", readsWithDirectIO(mixed, FIELD));
        assertTrue("and so is the second", readsWithDirectIO(mixed, SECOND_FIELD));
    }

    public void testStillRoutedAfterCloseAndOpen() throws Exception {
        final String index = "direct-io-routing-reopen";
        final IndexService indexService = createRoutableIndex(index, true, KNNEngine.FAISS);
        assertTrue(readsWithDirectIO(indexService, FIELD));

        client().admin().indices().prepareClose(index).get();
        client().admin().indices().prepareOpen(index).get();
        ensureGreen(index);

        final IndexService reopened = getInstanceFromNode(org.opensearch.indices.IndicesService.class).indexService(resolveIndex(index));
        assertNotNull("the index must be open again", reopened);
        assertTrue("the reopened segment is read with O_DIRECT again", readsWithDirectIO(reopened, FIELD));
    }

    /**
     * The input a field's vectors are read through, taken from the first leaf that has the field, which is the one
     * merged segment every fixture here produces.
     *
     * @param indexService the index
     * @param field        the vector field
     * @return the slice the flat vector values read through
     * @throws IOException if the reader cannot be read
     */
    /**
     * Whether a field's vectors are read with {@code O_DIRECT} in this node right now, read the way Lucene's
     * memory-segment scorer and k-NN's {@code WarmupUtil} read it. The slice never leaves the searcher that owns it:
     * the check happens inside the try block and only the verdict is returned.
     *
     * @param indexService the index
     * @param field        the vector field
     * @return true if the slice is a {@link DirectIOVectorIndexInput}
     * @throws IOException if the reader cannot be read
     */
    private boolean readsWithDirectIO(final IndexService indexService, final String field) throws IOException {
        final IndexShard shard = indexService.iterator().next();
        try (Engine.Searcher searcher = shard.acquireSearcher("direct-io-routing-test")) {
            for (final LeafReaderContext leaf : searcher.getIndexReader().leaves()) {
                final FloatVectorValues values = leaf.reader().getFloatVectorValues(field);
                if (values == null) {
                    continue;
                }
                assertTrue(
                    "the flat vector values of " + field + " must expose their slice, as Lucene's scorer needs",
                    values instanceof HasIndexSlice
                );
                final IndexInput slice = ((HasIndexSlice) values).getSlice();
                assertNotNull("the slice of " + field + " must not be null", slice);
                if (slice instanceof DirectIOVectorIndexInput) {
                    return true;
                }
                assertTrue(
                    "a slice that is not a Direct I/O reader must be the original input's own, not " + slice.getClass(),
                    slice instanceof MemorySegmentAccessInput
                );
                return false;
            }
        }
        throw new AssertionError("no leaf of this index has the field " + field);
    }

    /**
     * How many leaves this index has, which is how a case says which segment shape it built.
     *
     * @param indexService the index
     * @return the leaf count
     */
    private int leafCount(final IndexService indexService) {
        final IndexShard shard = indexService.iterator().next();
        try (Engine.Searcher searcher = shard.acquireSearcher("direct-io-routing-leaves")) {
            return searcher.getIndexReader().leaves().size();
        }
    }

    private IndexService createRoutableIndex(final String index, final boolean directIO, final KNNEngine engine) throws Exception {
        return createIndexWithVectors(index, directIO, engine, DIMENSION, DOC_COUNT, true);
    }

    private IndexService createIndexWithVectors(
        final String index,
        final boolean directIO,
        final KNNEngine engine,
        final int dimension,
        final int docCount,
        final boolean forceMerge
    ) throws Exception {
        final IndexService indexService = createKnnIndexService(index, directIO);
        createKnnMapping(index, FIELD, dimension, engine);
        indexVectors(index, dimension, docCount, FIELD);
        if (forceMerge) {
            forceMergeToOneSegment(index);
        }
        return indexService;
    }

    private IndexService createKnnIndexService(final String index, final boolean directIO) {
        final Settings settings = Settings.builder()
            .put(getKNNDefaultIndexSettingsBuildsGraphAlways())
            .put(KNNSettings.INDEX_KNN_DIRECT_IO_RESCORE_ENABLED, directIO)
            .build();
        return createIndex(index, settings);
    }

    private void createKnnMapping(final String index, final String field, final int dimension, final KNNEngine engine) throws IOException {
        // The base class's own overload, which puts the engine in the method and leaves the space type at its
        // default of l2.
        createKnnIndexMapping(index, field, dimension, engine);
    }

    /**
     * Indexes deterministic vectors in one bulk request per 64 documents, with one refresh at the end. Indexing one
     * document at a time with an immediate refresh would make one segment and one graph per document.
     *
     * @param index     the index
     * @param dimension the vector dimension
     * @param docCount  how many documents
     * @param fields    the vector fields, each carrying the same vector
     * @throws Exception if a bulk request fails
     */
    private void indexVectors(final String index, final int dimension, final int docCount, final String... fields) throws Exception {
        final Random random = new Random(1_234L);
        final int chunk = 64;
        for (int from = 0; from < docCount; from += chunk) {
            final int to = Math.min(from + chunk, docCount);
            final BulkRequest bulk = new BulkRequest();
            for (int doc = from; doc < to; doc++) {
                final Float[] vector = new Float[dimension];
                for (int i = 0; i < dimension; i++) {
                    vector[i] = random.nextFloat();
                }
                final XContentBuilder source = XContentFactory.jsonBuilder().startObject();
                for (final String field : fields) {
                    source.field(field, vector);
                }
                source.endObject();
                bulk.add(new IndexRequest(index).id(Integer.toString(doc)).source(source));
            }
            bulk.setRefreshPolicy(WriteRequest.RefreshPolicy.NONE);
            final BulkResponse response = client().bulk(bulk).get();
            assertFalse(response.buildFailureMessage(), response.hasFailures());
        }
        client().admin().indices().prepareRefresh(index).get();
    }

    /**
     * Force merges to one segment, which is what makes the segment non-compound: a single merged segment is the whole
     * index against a no-CFS ratio of 0.1.
     *
     * @param index the index
     */
    private void forceMergeToOneSegment(final String index) {
        client().admin().indices().forceMerge(new ForceMergeRequest(index).maxNumSegments(1)).actionGet();
        client().admin().indices().prepareRefresh(index).get();
    }

    private void setDirectIO(final String index, final boolean enabled) {
        updateIndexSetting(index, Settings.builder().put(KNNSettings.INDEX_KNN_DIRECT_IO_RESCORE_ENABLED, enabled).build());
    }
}

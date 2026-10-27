/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn;

import lombok.SneakyThrows;
import org.apache.hc.core5.http.io.entity.EntityUtils;
import org.opensearch.client.Request;
import org.opensearch.client.Response;
import org.opensearch.common.settings.Settings;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.core.xcontent.MediaTypeRegistry;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.index.VectorDataType;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;
import java.util.function.Supplier;

/**
 * What the two derived-source measurements share: the index pair, the access paths a client reconstructs a
 * vector through, the three routes they are each run under, and the {@code direct_io_stats} snapshot taken
 * around a fetch.
 *
 * <p>Derived source rebuilds {@code _source} by reading the vector back out of the {@code .vec} at fetch time,
 * so with Direct I/O engaged every vector a fetch returns is a device read. Two comparisons are made of that,
 * and they answer different questions:
 *
 * <ol>
 *   <li>the derived index against a non-derived index holding the same documents, which is ground truth: a
 *       non-derived {@code _source} is the json the client sent and never touches a {@code .vec}. This is the
 *       comparison {@link DerivedSourceTestCase#assertDocMatches} already makes, made here with the route
 *       on;</li>
 *   <li>the derived index under {@code dio-0b} and {@code dio-cache} against the same index under {@code mmap},
 *       which is whether the two routes return the same bytes.</li>
 * </ol>
 *
 * <p>Comparison 1 is made on parsed content, because a derived {@code _source} is re-serialized after the
 * vectors are injected and nothing promises it preserves the client's key order. Comparison 2 is made on the
 * bytes the {@code _source} endpoint returns, because both sides run the same serializer and a difference there
 * would be the route's doing.
 *
 * <p>The index is built under {@link DirectIOArm#DIO_CACHE} so its reader opens on the route, and the arms then
 * flip the cluster setting under the open reader, which an open input honours on its next read.
 * {@code routed_opens} is therefore asserted as a difference across the build and never as a node-lifetime
 * total.
 */
public abstract class DirectIODerivedSourceTestCase extends DerivedSourceTestCase {

    /** Every data type a {@code knn_vector} field can hold, which is what the identity matrix sweeps. */
    protected static final List<VectorDataType> DATA_TYPES = List.of(
        VectorDataType.FLOAT,
        VectorDataType.BYTE,
        VectorDataType.BINARY,
        VectorDataType.HALF_FLOAT
    );

    protected static final int DIMENSION = 64;

    /**
     * Documents per index. Large enough that every field's {@code .vec} is worth routing and that a
     * {@code size: 100} fetch is a real scan, small enough that six indices and four reindex targets stay
     * inside an integration test's budget.
     */
    protected static final int DOC_COUNT = 240;

    /** One seed for the whole fixture, so that two runs build the same documents. */
    protected static final long SEED = 20261027L;

    /** How many documents the single-document path is sampled over. */
    protected static final int GET_SAMPLE = 10;

    private final List<String> created = new ArrayList<>();

    public DirectIODerivedSourceTestCase(final CompressionTestConfig compressionConfig) {
        super(compressionConfig);
    }

    @Override
    public void tearDown() throws Exception {
        // Explicit values in both directions and never null: removing a persisted key wedges
        // _cluster/settings. Each key is restored on its own, because a failure restoring the first must not
        // leave the next at this class's value for whatever runs next on the same cluster.
        try {
            restore(KNNSettings.KNN_DIRECT_IO_RESCORE_ENABLED, false);
            restore(KNNSettings.KNN_DIRECT_IO_RESCORE_CACHE_MAX_BYTES, "0b");
            restore(KNNSettings.KNN_DIRECT_IO_MIN_FILE_SIZE, "10mb");
            for (final String index : created) {
                try {
                    deleteKNNIndex(index);
                } catch (Exception e) {
                    // The base class wipes the cluster's indices too, so a 404 here means it got there first.
                    logger.info("could not delete {}; the cluster wipe had already removed it", index, e);
                }
            }
        } finally {
            created.clear();
            super.tearDown();
        }
    }

    /** One setting back to its shipped value, loudly but without stopping the rest of the restore. */
    private void restore(final String key, final Object value) {
        try {
            updateClusterSettings(key, value);
        } catch (Exception e) {
            logger.error("could not restore {} to {}; the next test inherits this class's value", key, value, e);
        }
    }

    /**
     * Puts the cluster on {@code arm}. {@code min_file_size} is 0 B throughout, because this fixture's largest
     * {@code .vec} is a few hundred kilobytes and the shipped 10 MB default would decline it.
     */
    protected void applyArm(final DirectIOArm arm) throws Exception {
        updateClusterSettings(KNNSettings.KNN_DIRECT_IO_MIN_FILE_SIZE, "0b");
        updateClusterSettings(KNNSettings.KNN_DIRECT_IO_RESCORE_CACHE_MAX_BYTES, arm.cacheMaxBytes());
        updateClusterSettings(KNNSettings.KNN_DIRECT_IO_RESCORE_ENABLED, arm.routes());
    }

    /**
     * The {@code direct_io_stats} an operator reads, summed over the cluster's nodes, as
     * {@link DirectIOCounters}, whose javadoc states what a snapshot costs.
     */
    @SneakyThrows
    protected DirectIOCounters counters() {
        final Response response = getKnnStats(Collections.emptyList(), Collections.emptyList());
        return DirectIOCounters.of(parseNodeStatsResponse(EntityUtils.toString(response.getEntity())));
    }

    /**
     * The six index configurations the inherited helpers expect, in their order: derived on, derived off, and
     * the four reindex targets. Every vector field appears once per data type, so one index covers the whole
     * matrix; {@code nested} adds the same four fields again inside a nested object.
     *
     * <p>Each context's settings are its own plus {@code index.use_compound_file: false}, without which the
     * {@code .vec} lives inside the {@code .cfs} and the dispatch declines it {@code COMPOUND}.
     */
    protected List<DerivedSourceUtils.IndexConfigContext> contexts(final String prefix, final boolean nested) {
        final List<DerivedSourceUtils.IndexConfigContext> contexts = new ArrayList<>();
        for (final Pair<String, Boolean> index : derivedSourceVariants()) {
            final List<DerivedSourceUtils.FieldContext> fields = new ArrayList<>();
            for (final VectorDataType dataType : DATA_TYPES) {
                fields.add(vectorField(fieldName(dataType), dataType, false));
                fields.add(vectorField("update_" + fieldName(dataType), dataType, true));
            }
            if (nested) {
                final List<DerivedSourceUtils.FieldContext> children = new ArrayList<>();
                for (final VectorDataType dataType : DATA_TYPES) {
                    children.add(vectorField("nested_1." + fieldName(dataType), dataType, false));
                }
                children.add(DerivedSourceUtils.IntFieldType.builder().fieldPath("nested_1.test-int").nullProb(0).build());
                fields.add(DerivedSourceUtils.NestedFieldContext.builder().fieldPath("nested_1").children(children).nullProb(0).build());
            }
            fields.add(DerivedSourceUtils.TextFieldType.builder().fieldPath("test-text").nullProb(0).build());
            fields.add(DerivedSourceUtils.IntFieldType.builder().fieldPath("test-int").nullProb(0).build());

            final DerivedSourceUtils.IndexConfigContext context = DerivedSourceUtils.IndexConfigContext.builder()
                .derivedEnabled(index.getSecond())
                .indexName(getIndexName(prefix, index.getFirst(), false))
                .docCount(DOC_COUNT)
                .random(new Random(SEED))
                .fields(fields)
                .build();
            context.init();
            context.setSettings(Settings.builder().put(context.getSettings()).put("index.use_compound_file", false).build());
            contexts.add(context);
            created.add(context.indexName);
        }
        return contexts;
    }

    /**
     * The same six configurations carrying one {@code float} vector field instead of eight, so that a counter
     * divided by the number of documents a fetch returned is the cost of one reconstructed vector.
     *
     * @param prefix   the index name prefix, which keys the indices of one test
     * @param docCount how many documents each index holds
     */
    protected List<DerivedSourceUtils.IndexConfigContext> singleFieldContexts(final String prefix, final int docCount) {
        final List<DerivedSourceUtils.IndexConfigContext> contexts = new ArrayList<>();
        for (final Pair<String, Boolean> index : derivedSourceVariants()) {
            final DerivedSourceUtils.IndexConfigContext context = DerivedSourceUtils.IndexConfigContext.builder()
                .derivedEnabled(index.getSecond())
                .indexName(getIndexName(prefix, index.getFirst(), false))
                .docCount(docCount)
                .random(new Random(SEED))
                .fields(
                    List.of(
                        vectorField(fieldName(VectorDataType.FLOAT), VectorDataType.FLOAT, false),
                        DerivedSourceUtils.TextFieldType.builder().fieldPath("test-text").nullProb(0).build(),
                        DerivedSourceUtils.IntFieldType.builder().fieldPath("test-int").nullProb(0).build()
                    )
                )
                .build();
            context.init();
            context.setSettings(Settings.builder().put(context.getSettings()).put("index.use_compound_file", false).build());
            contexts.add(context);
            created.add(context.indexName);
        }
        return contexts;
    }

    /**
     * The six index prefixes and whether each has derived source on, in the order the inherited helpers index
     * them by: original enabled, original disabled, then the four reindex targets.
     */
    private static List<Pair<String, Boolean>> derivedSourceVariants() {
        return List.of(
            new Pair<>("original-enable-", true),
            new Pair<>("original-disable-", false),
            new Pair<>("e2e-", true),
            new Pair<>("e2d-", false),
            new Pair<>("d2e-", true),
            new Pair<>("d2d-", false)
        );
    }

    /** The field name a data type's vectors live under, which is also the name in every log line. */
    protected static String fieldName(final VectorDataType dataType) {
        return dataType.getValue() + "_vector";
    }

    /**
     * One {@code knn_vector} field of {@code dataType}, with a value supplier whose values survive the round
     * trip into the index and back.
     *
     * <p>A derived {@code _source} is the vector as it was stored and read back; a non-derived one is the json
     * the client sent. The two agree only where the value is exactly representable in the field's storage, so
     * {@code half_float} draws from a grid of sixteenths, which fp16 holds exactly;
     * {@link DerivedSourceUtils#randomVectorSupplier} covers the other three, whose values are already exact
     * (a float prints and re-parses to itself, and {@code byte} and {@code binary} are integers). That is a
     * property of derived source and not of Direct I/O, and it is why the identity matrix can be a strict
     * equality at all.
     */
    private static DerivedSourceUtils.FieldContext vectorField(final String path, final VectorDataType dataType, final boolean isUpdate) {
        final DerivedSourceUtils.KNNVectorFieldTypeContext.KNNVectorFieldTypeContextBuilder<?, ?> builder =
            DerivedSourceUtils.KNNVectorFieldTypeContext.builder()
                .fieldPath(path)
                .dimension(DIMENSION)
                .vectorDataType(dataType)
                .nullProb(0)
                .skipProb(0)
                .isUpdate(isUpdate);
        if (dataType == VectorDataType.HALF_FLOAT) {
            builder.valueSupplier(halfFloatSupplier(new Random(SEED + path.hashCode()), DIMENSION));
        }
        return builder.build();
    }

    /** Vectors on a grid of sixteenths inside +-64, every value of which fp16 holds exactly. */
    private static Supplier<Object> halfFloatSupplier(final Random random, final int dimension) {
        return () -> {
            final float[] vector = new float[dimension];
            for (int i = 0; i < dimension; i++) {
                vector[i] = (random.nextInt(2049) - 1024) / 16.0f;
            }
            return vector;
        };
    }

    /**
     * Creates and fills the first two contexts, force merges both to one segment and refreshes, which is the
     * state every arm reads: one segment per index, no compound file, nothing reopening underneath.
     *
     * <p>{@link DerivedSourceTestCase#prepareOriginalIndices} is what fills them, so the documents, the
     * per-document equality check and the ids are the ones every other derived-source test uses.
     */
    protected void prepareIndexPair(final List<DerivedSourceUtils.IndexConfigContext> contexts) throws Exception {
        prepareOriginalIndices(contexts);
        forceMergeKnnIndex(contexts.get(0).indexName, 1);
        forceMergeKnnIndex(contexts.get(1).indexName, 1);
        refreshAllIndices();
    }

    /**
     * An empty index with {@code template}'s settings and mapping, registered for deletion, which is what a
     * reindex measurement needs: a fresh target per run, because reindexing into a used one is not the same
     * work.
     */
    protected void createTargetIndex(final String name, final DerivedSourceUtils.IndexConfigContext template) throws Exception {
        createKnnIndex(name, template.getSettings(), template.getMapping());
        created.add(name);
    }

    /**
     * Every document id of an index of {@code docCount} documents, as the helpers write them: 1 to n. The
     * count is a parameter and not {@link #DOC_COUNT} because the indices a measurement builds are not all
     * the same size, and an id list short of the index would silently measure a prefix of it.
     */
    protected static List<String> allIds(final int docCount) {
        final List<String> ids = new ArrayList<>(docCount);
        for (int i = 1; i <= docCount; i++) {
            ids.add(String.valueOf(i));
        }
        return ids;
    }

    /** {@link #GET_SAMPLE} ids spread evenly over the index, for the paths that fetch one document at a time. */
    protected static List<String> sampledIds() {
        final List<String> ids = new ArrayList<>(GET_SAMPLE);
        for (int i = 0; i < GET_SAMPLE; i++) {
            ids.add(String.valueOf(1 + i * (DOC_COUNT / GET_SAMPLE)));
        }
        return ids;
    }

    /**
     * Every document's {@code _source} exactly as the {@code _source} endpoint returns it, keyed by id. This is
     * the capture comparison 2 is made on: the endpoint returns the source document alone, so the strings are
     * the bytes a client receives and nothing else.
     */
    protected Map<String, String> sourceBytes(final String index, final List<String> ids) throws Exception {
        final Map<String, String> sources = new TreeMap<>();
        for (final String id : ids) {
            final Request request = new Request("GET", String.format(Locale.ROOT, "/%s/_source/%s", index, id));
            final Response response = client().performRequest(request);
            assertEquals(RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));
            sources.put(id, EntityUtils.toString(response.getEntity()));
        }
        return sources;
    }

    /** Every document's {@code _source} as one search fetch returns it, keyed by id and canonicalised. */
    protected Map<String, String> searchFetch(final String index, final int size) throws Exception {
        final String body = String.format(Locale.ROOT, "{\"size\":%d,\"query\":{\"match_all\":{}}}", size);
        return hitSources(post(String.format(Locale.ROOT, "/%s/_search", index), body));
    }

    /** Every document's {@code _source} as one multi-get returns it, keyed by id and canonicalised. */
    protected Map<String, String> multiGet(final String index, final List<String> ids) throws Exception {
        final StringBuilder docs = new StringBuilder("{\"docs\":[");
        for (int i = 0; i < ids.size(); i++) {
            docs.append(i == 0 ? "" : ",").append(String.format(Locale.ROOT, "{\"_id\":\"%s\"}", ids.get(i)));
        }
        docs.append("]}");
        final Map<String, Object> response = parse(post(String.format(Locale.ROOT, "/%s/_mget", index), docs.toString()));
        final Map<String, String> sources = new TreeMap<>();
        for (final Object doc : asList(response.get("docs"))) {
            final Map<String, Object> entry = asMap(doc);
            assertEquals("a multi-get document was not found: " + entry, Boolean.TRUE, entry.get("found"));
            sources.put(String.valueOf(entry.get("_id")), canonical(entry.get("_source")));
        }
        return sources;
    }

    /**
     * Every document's {@code _source} as a scroll returns it, keyed by id and canonicalised. A fetch per
     * slice, which is the doc-order scan a derived-source fetch is cheapest on and dearest without.
     */
    protected Map<String, String> scroll(final String index, final int size) throws Exception {
        final Map<String, String> sources = new TreeMap<>();
        String body = String.format(Locale.ROOT, "{\"size\":%d,\"query\":{\"match_all\":{}}}", size);
        String response = post(String.format(Locale.ROOT, "/%s/_search?scroll=2m", index), body);
        String scrollId = String.valueOf(parse(response).get("_scroll_id"));
        try {
            while (true) {
                final Map<String, String> slice = hitSources(response);
                if (slice.isEmpty()) {
                    break;
                }
                sources.putAll(slice);
                body = String.format(Locale.ROOT, "{\"scroll\":\"2m\",\"scroll_id\":\"%s\"}", scrollId);
                response = post("/_search/scroll", body);
                scrollId = String.valueOf(parse(response).get("_scroll_id"));
            }
        } finally {
            final Request delete = new Request("DELETE", "/_search/scroll");
            delete.setJsonEntity(String.format(Locale.ROOT, "{\"scroll_id\":[\"%s\"]}", scrollId));
            client().performRequest(delete);
        }
        return sources;
    }

    /** The {@code _source} of every hit of a search response, keyed by id and canonicalised. */
    private Map<String, String> hitSources(final String response) throws Exception {
        final Map<String, String> sources = new TreeMap<>();
        for (final Object hit : asList(asMap(parse(response).get("hits")).get("hits"))) {
            final Map<String, Object> entry = asMap(hit);
            sources.put(String.valueOf(entry.get("_id")), canonical(entry.get("_source")));
        }
        return sources;
    }

    /** A request body straight through, as the response body. */
    protected String post(final String path, final String body) throws Exception {
        final Request request = new Request("POST", path);
        request.setJsonEntity(body);
        final Response response = client().performRequest(request);
        assertEquals(RestStatus.OK, RestStatus.fromCode(response.getStatusLine().getStatusCode()));
        return EntityUtils.toString(response.getEntity());
    }

    /** A response body as a map, the way every other helper in {@link KNNRestTestCase} parses one. */
    @SneakyThrows
    private Map<String, Object> parse(final String json) {
        return createParser(MediaTypeRegistry.getDefaultMediaType().xContent(), json).map();
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> asMap(final Object value) {
        return (Map<String, Object>) value;
    }

    @SuppressWarnings("unchecked")
    private static List<Object> asList(final Object value) {
        return (List<Object>) value;
    }

    /**
     * A parsed {@code _source} written back out with its object keys in order, which is what comparison 1 is
     * made on: a derived source is re-serialized after its vectors are injected, and nothing promises that
     * preserves the key order the client sent, so an order difference is not a content difference.
     */
    protected static String canonical(final Object value) {
        final StringBuilder out = new StringBuilder();
        append(out, value);
        return out.toString();
    }

    private static void append(final StringBuilder out, final Object value) {
        if (value instanceof Map<?, ?> map) {
            final Map<String, Object> sorted = new TreeMap<>();
            for (final Map.Entry<?, ?> entry : map.entrySet()) {
                sorted.put(String.valueOf(entry.getKey()), entry.getValue());
            }
            out.append('{');
            boolean first = true;
            for (final Map.Entry<String, Object> entry : sorted.entrySet()) {
                out.append(first ? "" : ",").append(entry.getKey()).append('=');
                append(out, entry.getValue());
                first = false;
            }
            out.append('}');
        } else if (value instanceof List<?> list) {
            out.append('[');
            for (int i = 0; i < list.size(); i++) {
                out.append(i == 0 ? "" : ",");
                append(out, list.get(i));
            }
            out.append(']');
        } else {
            out.append(value);
        }
    }

    /** Every document's {@code _source} as canonical content, keyed by id, through the single-document path. */
    protected Map<String, String> sourceContent(final String index, final List<String> ids) throws Exception {
        final Map<String, String> sources = new TreeMap<>();
        for (final Map.Entry<String, String> entry : sourceBytes(index, ids).entrySet()) {
            sources.put(entry.getKey(), canonical(parse(entry.getValue())));
        }
        return sources;
    }

    /**
     * Every access path of one arm, as canonical content keyed by document id. Keyed by path name, so a
     * mismatch names the path that produced it.
     */
    protected Map<String, Map<String, String>> readPaths(final String index, final int docCount) throws Exception {
        final Map<String, Map<String, String>> capture = new LinkedHashMap<>();
        capture.put("source-endpoint", sourceContent(index, allIds(docCount)));
        capture.put("get-sample", sourceContent(index, sampledIds()));
        capture.put("search-fetch", searchFetch(index, docCount));
        capture.put("mget", multiGet(index, allIds(docCount)));
        capture.put("scroll", scroll(index, 50));
        return capture;
    }

    /**
     * Asserts that two captures agree document by document on every path, and says which path, which document
     * and which side differ when they do.
     */
    protected void assertCapturesMatch(
        final String what,
        final Map<String, Map<String, String>> expected,
        final Map<String, Map<String, String>> actual
    ) {
        assertEquals(what + ": the two captures covered different paths", expected.keySet(), actual.keySet());
        for (final String path : expected.keySet()) {
            final Map<String, String> left = expected.get(path);
            final Map<String, String> right = actual.get(path);
            assertEquals(what + " on " + path + ": different documents came back", left.keySet(), right.keySet());
            assertFalse(what + " on " + path + " returned nothing, so it proves nothing", left.isEmpty());
            for (final String id : left.keySet()) {
                assertEquals(what + " on " + path + ", document " + id, left.get(id), right.get(id));
            }
        }
    }
}

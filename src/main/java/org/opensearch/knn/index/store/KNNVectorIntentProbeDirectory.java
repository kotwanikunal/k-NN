/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FilterDirectory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;

import java.io.IOException;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.stream.Collectors;

/**
 * A {@link Directory} that records the {@code (name, hints)} pair of every {@code openInput} that
 * reaches it and then delegates unchanged.
 *
 * <p>It exists to answer one question the directory design turns on: an {@link IOContext} is built
 * at the codec layer, several OpenSearch wrappers sit between there and the directory a plugin can
 * supply, and the design is only possible if a plugin-defined {@link KNNVectorReadIntent} survives
 * that descent. This class makes the answer observable from both a unit test and a running node,
 * without changing a single byte of I/O — every method not overridden here is
 * {@link FilterDirectory}'s pass-through, and the one method that is overridden returns the
 * delegate's {@link IndexInput} as-is.
 *
 * <p>It is deliberately not the dispatching directory. Dispatch would have to answer a second
 * question — whether an {@code IndexInput} over Direct I/O can honour {@link IndexInput#prefetch}
 * well enough to keep the queue depth mmap gets from {@code madvise(MADV_WILLNEED)} — and mixing the
 * two would make a failure of either look like a failure of the design.
 */
@Log4j2
public class KNNVectorIntentProbeDirectory extends FilterDirectory {

    /** An {@code openInput} this directory saw, in the terms the dispatch rule would be written in. */
    public record Observation(String name, Set<IOContext.FileOpenHint> hints, IOContext.Context context, KNNVectorReadIntent intent) {
        @Override
        public String toString() {
            return "name=" + name + " context=" + context + " intent=" + intent + " hints=" + hints;
        }
    }

    /**
     * Whether any index on this node has the probe installed.
     * <p>
     * It lets a caller that failed to find a probe tell "this index never opted in" from "this index
     * opted in but this segment cannot reach it", which is the difference between no evidence and the
     * compound-segment result. Without it an absent log line would be ambiguous, and the alternative —
     * logging every failed lookup — would fire on every segment open of every index on the node.
     */
    private static final java.util.concurrent.atomic.AtomicBoolean INSTALLED_ON_NODE = new java.util.concurrent.atomic.AtomicBoolean();

    private final List<Observation> observations = new CopyOnWriteArrayList<>();

    /** The index this directory belongs to, so that a log line can be attributed to one. */
    private final String indexName;

    public KNNVectorIntentProbeDirectory(final Directory delegate) {
        this(delegate, "unknown");
    }

    public KNNVectorIntentProbeDirectory(final Directory delegate, final String indexName) {
        super(delegate);
        this.indexName = indexName;
        INSTALLED_ON_NODE.set(true);
    }

    public String indexName() {
        return indexName;
    }

    /** Whether any index on this node installed a probe. See {@link #INSTALLED_ON_NODE}. */
    public static boolean isInstalledOnNode() {
        return INSTALLED_ON_NODE.get();
    }

    @Override
    public IndexInput openInput(final String name, final IOContext context) throws IOException {
        final Observation observation = new Observation(name, context.hints(), context.context(), KNNVectorReadIntent.of(context));
        observations.add(observation);
        // INFO rather than DEBUG so that a gradlew-run ingest yields the evidence without a logger
        // override, and only for the files the design would ever route, so that the segments/metadata
        // opens of a real ingest do not bury them.
        if (isVectorDataFile(name)) {
            log.info("k-NN intent probe [{}] saw openInput {} delegate={}", indexName, observation, in.getClass().getSimpleName());
        } else {
            log.debug("k-NN intent probe [{}] saw openInput {}", indexName, observation);
        }
        return in.openInput(name, context);
    }

    /**
     * Whether {@code name} is one of the files the design would dispatch on: the full-precision flat
     * vectors, the quantized codes beside them, and the native index. Named by suffix because that is
     * all a directory ever gets — and the fact that suffix alone cannot separate a re-score read of
     * {@code .vec} from a traversal, merge, warmup or derived-source read of the same file is exactly
     * why {@link KNNVectorReadIntent} exists.
     */
    static boolean isVectorDataFile(final String name) {
        return name.endsWith(".vec") || name.endsWith(".veq") || name.endsWith(".faiss");
    }

    /** Every {@code openInput} seen so far, in call order. */
    public List<Observation> observations() {
        return List.copyOf(observations);
    }

    /** Every {@code openInput} seen so far for a vector data file, in call order. */
    public List<Observation> vectorDataObservations() {
        return observations.stream().filter(o -> isVectorDataFile(o.name())).collect(Collectors.toUnmodifiableList());
    }

    public void clearObservations() {
        observations.clear();
    }

    /**
     * The chain of {@link Directory} wrappers from {@code directory} down, as simple class names.
     * <p>
     * {@link FilterDirectory#unwrap} answers only what is at the bottom; the design needs to know
     * what is in between, because any wrapper that rebuilt the {@link IOContext} rather than passing
     * the caller's through would break the intent signal. Walking
     * {@link FilterDirectory#getDelegate()} is the only way to see them from outside the server.
     */
    public static List<String> wrapperChain(final Directory directory) {
        final List<String> chain = new java.util.ArrayList<>();
        Directory current = directory;
        // Bounded: a cycle in a directory chain would already have hung the shard, but a probe must
        // not be the thing that hangs.
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

    /**
     * The {@link KNNVectorIntentProbeDirectory} in {@code directory}'s wrapper chain, or {@code null}
     * when there is none — which is every index that has not opted in with
     * {@code index.store.factory}.
     */
    public static KNNVectorIntentProbeDirectory find(final Directory directory) {
        Directory current = directory;
        for (int depth = 0; current != null && depth < 16; depth++) {
            if (current instanceof KNNVectorIntentProbeDirectory probe) {
                return probe;
            }
            if (current instanceof FilterDirectory filterDirectory) {
                current = filterDirectory.getDelegate();
            } else {
                return null;
            }
        }
        return null;
    }
}

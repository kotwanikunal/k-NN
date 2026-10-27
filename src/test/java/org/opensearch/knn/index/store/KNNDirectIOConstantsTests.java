/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.KNNSettings;

/**
 * The nine values that used to be {@code knn.direct_io.*} settings are constants now, and each one is still
 * the value the shipped default had. F-03.
 *
 * <p>The reference is a node's own answer to
 * {@code _cluster/settings?include_defaults=true&flat_settings=true} taken before the keys were removed, so
 * these are the numbers an operator would have read back, not the numbers the declarations happened to say.
 * A change here is a changed default, which is the one thing this phase may not do.
 */
public class KNNDirectIOConstantsTests extends KNNTestCase {

    /**
     * The seven constants whose value is absolute, against what the baseline node reported.
     *
     * <p>{@code queue_capacity} is not here because its default was {@code 0}, meaning "derive", and the
     * derivation is now the only behaviour: {@code DirectIOReadPoolTests} pins it. The direct-memory budget
     * is in {@link #testTheDirectMemoryBudgetIsStillHalfTheWall()} because it is derived per JVM.
     */
    public void testEachRemovedKeyIsAConstantAtItsBaselineDefault() {
        assertEquals("knn.direct_io.rescore.prefetch.staged_ranges was 64", 64, DirectIOVectorIndexInput.DEFAULT_MAX_STAGED_RANGES);
        assertEquals(
            "knn.direct_io.rescore.prefetch.max_span_bytes was 135168b",
            135168,
            DirectIOVectorIndexInput.DEFAULT_MAX_STAGED_RANGE_BYTES
        );
        assertEquals("knn.direct_io.readahead.window_bytes was 65536b", 65536, DirectIOVectorIndexInput.DEFAULT_READAHEAD_WINDOW_BYTES);
        assertEquals("knn.direct_io.readahead.trigger was 2", 2, DirectIOVectorIndexInput.DEFAULT_READAHEAD_TRIGGER);
        assertTrue("knn.direct_io.rescore.cache.span_fill was true", DirectIOVectorCache.SPAN_FILL);
        assertEquals("knn.direct_io.rescore.cache.read_around_bytes was 65536b", 65536L, DirectIOVectorCache.READ_AROUND_BYTES);
        assertEquals(
            "knn.direct_io.rescore.prefetch.wide_buffer_budget was 0b, which meant a quarter of the budget",
            DirectIOVectorIndexInput.DIRECT_MEMORY_BUDGET_BYTES / 4,
            DirectIOVectorIndexInput.wideBufferBudgetBytes()
        );
    }

    /**
     * The direct-memory budget was {@code 2147483648b} on the baseline node, which is half that node's
     * 4 GiB wall; the value is per JVM, so what carries over is the derivation and not the number.
     */
    public void testTheDirectMemoryBudgetIsStillHalfTheWall() {
        assertEquals(
            "knn.direct_io.direct_memory_budget defaulted to half the wall",
            KNNSettings.KNN_DIRECT_IO_DIRECT_MEMORY_WALL_BYTES / 2,
            DirectIOVectorIndexInput.DIRECT_MEMORY_BUDGET_BYTES
        );
        assertTrue("and the wall resolves on this JVM", KNNSettings.KNN_DIRECT_IO_DIRECT_MEMORY_WALL_BYTES > 0);
        assertEquals(
            "a node bounds itself by the constant",
            DirectIOVectorIndexInput.DIRECT_MEMORY_BUDGET_BYTES,
            DirectIOVectorIndexInput.directMemoryBudgetBytes()
        );
    }

    /**
     * The constants are not reachable through the settings surface any more, which is the point of removing
     * the keys: a key nothing reads is a key an operator can set and be misled by.
     */
    public void testTheRemovedKeysAreGoneFromTheSurface() {
        assertEquals(
            6L,
            KNNSettings.state().getSettings().stream().filter(setting -> setting.getKey().startsWith("knn.direct_io.")).count()
        );
    }
}

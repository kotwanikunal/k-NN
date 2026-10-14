/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.scorer;

import lombok.AccessLevel;
import lombok.NoArgsConstructor;
import lombok.extern.log4j.Log4j2;
import org.apache.lucene.store.IndexInput;
import org.opensearch.knn.common.featureflags.KNNFeatureFlags;
import org.opensearch.knn.index.store.DirectIOVectorIndexInput;

import java.io.IOException;
import java.util.Arrays;

/**
 * Helper class to prefetch vector data from disk to improve query performance.
 * <p>
 * Prefetches vector data using exact byte range strategy. Vectors are grouped together when
 * the end of a vector (start offset + vector size) falls within 128KB from the group's start offset.
 * This minimizes I/O operations while fetching only the exact bytes needed.
 * <p>
 * Prefetching is controlled by {@link KNNFeatureFlags#isPrefetchEnabled()}. When disabled,
 * no prefetch operations are performed.
 * <p>
 * <b>Example:</b><br>
 * Given vectors at offsets [100KB, 120KB, 300KB] with 10KB vector size:
 * <ul>
 *   <li>Group 1: Prefetch 40KB (100KB to 130KB, covers vectors at 100KB and 120KB)</li>
 *   <li>Group 2: Prefetch 10KB (300KB to 310KB, covers vector at 300KB)</li>
 * </ul>
 * Result: 2 prefetch operations, 50KB total prefetched (vs 6 random seeks without prefetch)
 * <p>
 * For a Direct I/O input two further rules apply, and neither changes what any other input sees: the
 * candidates the vector store already holds whole are dropped before the grouping (N1), and a group is closed
 * before the block-aligned span the input would have to stage for it would pass that input's span cap (P1),
 * because a group above the cap is declined and its vectors then go to the device one blocking read at a time
 * and are never admitted to the store.
 */
@Log4j2
@NoArgsConstructor(access = AccessLevel.PRIVATE)
public class PrefetchHelper {

    // TODO: If needed we can get this value via Cluster Settings
    private static final long BYTES_128 = 128 * 1024;

    /**
     * The miss list, one array per thread because the batch is per thread and the alternative is an
     * allocation per {@code bulkScore}. Grown to the largest batch a thread has seen and never shrunk; a
     * search thread's batch is Lucene's bulk size, so this settles at a few hundred ints.
     */
    private static final ThreadLocal<int[]> MISS_SCRATCH = new ThreadLocal<>();

    /**
     * Prefetches vector data from disk using exact byte range strategy.
     * <p>
     * Vectors are sorted by offset and grouped when the end of a vector falls within 128KB
     * from the group start. Each group is prefetched with exact byte range needed.
     * Returns early if prefetch is disabled, ordinals are null, or fewer than 2 vectors.
     * <p>
     * When the input is a Direct I/O input with the vector store on, the candidates the store already holds
     * whole are dropped before the grouping and only the misses are prefetched (N1). Without that, a group
     * with one missing vector in it is staged in full, because the input can only accept or decline a whole
     * group: MEASURED on the node, 73.4 staged reads per query for 77.1 missing vectors against 62.7 for
     * 76.9 for a build that grouped only its misses. The hits are still served by the input's own lookup when
     * the values are read, so nothing here serves bytes and no score can change; a vector evicted between the
     * filter and the read is a blocking read of the same bytes.
     *
     * @param indexInput the index input to prefetch from
     * @param baseOffset the base offset in the file where vectors start
     * @param oneVectorByteSize the size of one vector in bytes
     * @param ordsToPrefetch array of vector ordinals to prefetch
     * @param numOrds number of valid ordinals in the array
     * @throws IOException if an I/O error occurs during prefetch
     */
    public static void prefetch(
        final IndexInput indexInput,
        final long baseOffset,
        final long oneVectorByteSize,
        final int[] ordsToPrefetch,
        final int numOrds
    ) throws IOException {
        // The bound is on the batch as the caller gives it, and deliberately not on the miss list: a batch of
        // one is the caller's own single-vector read, while a batch of many with one miss left in it must be
        // prefetched, because the store admits only vectors a staged read covered (M3). Turning that into a
        // blocking read would leave those vectors uncached forever.
        if (ordsToPrefetch == null || numOrds <= 1) {
            return;
        }
        if (KNNFeatureFlags.isPrefetchEnabled() == false) {
            log.debug("KNNVectors Prefetch is disabled");
            return;
        }
        // The caller's array is sorted in place, exactly as it has always been and before anything else, so
        // that what the caller sees after this call does not depend on what the store happens to hold.
        Arrays.sort(ordsToPrefetch, 0, numOrds);
        int[] ords = ordsToPrefetch;
        int count = numOrds;
        if (indexInput instanceof DirectIOVectorIndexInput directIOInput) {
            final int[] misses = missScratch(numOrds);
            final int remaining = directIOInput.selectUncachedOrdinals(baseOffset, oneVectorByteSize, ords, numOrds, misses);
            if (remaining >= 0) {
                ords = misses;
                count = remaining;
            }
        }
        if (count == 0) {
            // Every candidate is cached, so there is nothing for the device to do: the reads will all hit.
            log.trace("Prefetching nothing: the vector store holds all [{}] candidates", numOrds);
            return;
        }
        prefetchExactVectorSize(indexInput, baseOffset, oneVectorByteSize, ords, count);
    }

    /** This thread's miss list, at least {@code needed} ints long. */
    private static int[] missScratch(final int needed) {
        int[] scratch = MISS_SCRATCH.get();
        if (scratch == null || scratch.length < needed) {
            scratch = new int[needed];
            MISS_SCRATCH.set(scratch);
        }
        return scratch;
    }

    /**
     * Prefetches vectors using exact byte ranges.
     * <p>
     * Groups vectors within 128KB ranges and prefetches only the exact bytes needed for each group.
     * The ordinals arrive sorted ascending, either as the caller's array or as the miss list taken from it in
     * order, so the grouping of a given set of ordinals is the same whichever way it got here.
     * <p>
     * A Direct I/O input declines a range whose block-aligned length is above its span cap, and a declined
     * group's vectors become blocking reads that the store never admits, so the decline costs the group's
     * hits and not just its read-ahead. For such an input a group is therefore <b>closed</b> before it would
     * pass the cap rather than handed over to be declined (P1): one more staged read of the same bytes,
     * where before there was one blocking read per vector and no admission. Every other input keeps today's
     * grouping exactly, because {@link #spanCap} answers {@link Long#MAX_VALUE} for it.
     * <p>
     * Under the {@code bytes} read profile (P2) a Direct I/O input's group is also closed whenever the next
     * vector's first block does not overlap or touch the group's last, so that no read carries a block
     * holding no byte of a missing vector. {@code ops}, the default, is the grouping above unchanged.
     *
     * @param indexInput the index input to prefetch from
     * @param baseOffset the base offset in the file where vectors start
     * @param oneVectorByteSize the size of one vector in bytes
     * @param ordsToPrefetch ascending array of vector ordinals to prefetch
     * @param numOrds number of valid ordinals in the array
     * @throws IOException if an I/O error occurs during prefetch
     */
    private static void prefetchExactVectorSize(
        final IndexInput indexInput,
        final long baseOffset,
        final long oneVectorByteSize,
        final int[] ordsToPrefetch,
        final int numOrds
    ) throws IOException {
        final DirectIOVectorIndexInput directIOInput = indexInput instanceof DirectIOVectorIndexInput in ? in : null;
        final long spanCap = spanCap(directIOInput);
        // Both read once per batch rather than per candidate: the span cap comes from a setting the input
        // read at open and the profile is static, so neither can change under one grouping pass.
        final boolean byBytes = directIOInput != null && directIOInput.groupsByBytes();
        int groupCount = 1;
        long groupStartOffset = baseOffset + (long) ordsToPrefetch[0] * oneVectorByteSize;
        long groupEndOffset = groupStartOffset + oneVectorByteSize;

        for (int i = 1; i < numOrds; i++) {
            long currentOffset = baseOffset + (long) ordsToPrefetch[i] * oneVectorByteSize;
            long currentEndOffset = currentOffset + oneVectorByteSize;
            // Three reasons to close the group, and the last two never fire for an input with no span cap and
            // no profile: the 128 KB group budget; the span the input would actually have to stage for it
            // (P1); and, under the bytes profile, a gap of a whole block or more between this vector and the
            // group (P2). A group of one is always handed over whatever its aligned length, because splitting
            // cannot make a single vector smaller.
            if ((currentEndOffset - groupStartOffset) > BYTES_128
                || exceedsSpanCap(directIOInput, spanCap, groupStartOffset, currentEndOffset - groupStartOffset)
                || (byBytes && leavesAGapBlock(directIOInput, groupEndOffset, currentOffset))) {
                indexInput.prefetch(groupStartOffset, groupEndOffset - groupStartOffset);
                groupCount++;
                groupStartOffset = currentOffset;
            }
            groupEndOffset = currentEndOffset;
        }
        // Prefetch final group
        indexInput.prefetch(groupStartOffset, groupEndOffset - groupStartOffset);

        log.trace("Prefetching grouped [{}] vectors where num of ords was [{}] using exact prefetch size", groupCount, numOrds);
    }

    /**
     * The largest span {@code input} will stage, or {@link Long#MAX_VALUE} when there is no such bound to
     * respect. Read once per batch: it comes from a setting the input read at open, so it cannot change
     * under one grouping pass, and reading it per candidate would be a field read per vector for no effect.
     */
    private static long spanCap(final DirectIOVectorIndexInput input) {
        return input == null ? Long.MAX_VALUE : input.maxStagedRangeBytes();
    }

    /** Whether staging {@code [offset, offset + length)} on {@code input} would be declined by its cap. */
    private static boolean exceedsSpanCap(final DirectIOVectorIndexInput input, final long spanCap, final long offset, final long length) {
        return input != null && input.alignedSpanBytes(offset, length) > spanCap;
    }

    /**
     * Whether adding the vector at {@code nextOffset} to a group ending at {@code groupEndOffset} would put a
     * block carrying none of the group's bytes and none of that vector's into the read. P2's {@code bytes}
     * rule.
     *
     * <p>The comparison is on block <em>indexes</em> and not on byte distance, because what the device reads
     * is whole blocks: two misses 10 bytes apart across a boundary are in adjacent blocks and share a read,
     * while two misses 4,000 bytes apart can be too. The group's last block is the one holding
     * {@code groupEndOffset - 1}, its last byte; the vector's first block is the one holding its first byte.
     * They may be the same block (two misses in one block, one read) or adjacent (one read); a difference of
     * two or more means at least one block in between holds nothing either of them needs, and the group
     * closes.
     */
    private static boolean leavesAGapBlock(final DirectIOVectorIndexInput input, final long groupEndOffset, final long nextOffset) {
        return input.blockIndexOf(nextOffset) > input.blockIndexOf(groupEndOffset - 1) + 1;
    }
}

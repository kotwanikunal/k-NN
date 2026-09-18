/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import lombok.extern.log4j.Log4j2;
import org.opensearch.core.common.unit.ByteSizeValue;
import org.opensearch.knn.index.KNNSettings;

import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;

/**
 * <b>The retention behind the loader seam.</b> A bounded, byte-accounted LRU of decoded full-precision
 * vectors for one {@link VectorLoaderSource} - one field in one segment - keyed by ordinal.
 *
 * <h2>Why a cache on a path built to bypass the cache</h2>
 * Reading rescore candidates with {@code O_DIRECT} removes them from the page cache, which is the point:
 * their eviction pressure is what pushes the graph and the quantized vectors out. But the page cache was
 * also serving <em>reuse</em>, and that was thrown out with the pressure. Measured on this shape: the
 * Direct I/O arm issues 200 device reads per query where the mmap arm issues 103.4, so mmap is serving
 * about 48% of candidate reads from bytes a previous query brought in. An exact LRU stack-distance
 * analysis of a 2,000-query full-candidate trace says how much of that a bounded cache recovers - 41.4%
 * of reads at a 4 MB budget, 44.2% at 8 MB, 45.3% at 16 MB, 50.3% at 128 MB - and cross-validates
 * against the mmap figure, since the 56.9% ceiling restricted to a 200-query horizon is 47.9% against
 * mmap's measured 48.3%.
 *
 * <p>So this cache is not an attempt to re-create the page cache. It is sized to hold the head of the
 * reuse distribution and nothing else: 41% of the reuse falls within 1 MB of stack depth, and about 80%
 * of re-references happen within 50 queries, which is why the 8 MB default is within 1.1 points of a
 * cache sixteen times its size.
 *
 * <h2>Granularity: vectors, not blocks</h2>
 * Entries are decoded vectors, not the 8 KiB aligned blocks the device reads. At the budgets that matter
 * this is the whole difference between a useful cache and a useless one: at a 1 MB budget vector
 * granularity measured 23.2% hit against roughly 5% for block granularity, because a block holds two
 * 3072-byte vectors of which the second is usually cold, so block granularity spends half the budget
 * retaining bytes nothing asks for.
 *
 * <h2>The bound is bytes, and it is not approximate</h2>
 * Capacity is a byte budget, never an entry count, and the charge per entry is the retained
 * {@code float[]} plus {@link #ENTRY_OVERHEAD_BYTES} for the map node that holds it, so
 * {@link Stats#bytes()} is an estimate of heap actually held rather than of payload alone. Eviction
 * happens <em>before</em> insertion, so the budget is never exceeded, not even for the window of one
 * entry: {@link #put} makes room first and then declines to insert at all if room cannot be made.
 *
 * <h2>Copy in, copy out</h2>
 * {@link #put} copies the caller's vector and {@link #load} copies back into the caller's array, rather
 * than either side handing over a reference. That costs one 3072-byte {@code memcpy} per hit - against a
 * device read, nothing - and buys the guarantee that a retained array is never aliased by a loader that
 * is about to overwrite its scratch buffer. Handing out the retained array directly would be faster and
 * is permitted by {@link VectorLoaderSource.Loader#read}'s contract, but it makes every future caller of
 * that method a potential corruptor of every other query's cached data, and the cost of getting that
 * wrong is a silently wrong score.
 *
 * <h2>Lifetime and invalidation</h2>
 * There is no invalidation, because a segment is immutable: the vector at an ordinal never changes, so a
 * hit is always correct. The cache is created with its source and dies with it, which is also what bounds
 * the node-wide footprint to the budget times the number of open rescore-served segments.
 *
 * <h2>Concurrency</h2>
 * Lucene runs per-leaf scoring tasks concurrently and every one of them gets its own loader over the same
 * shared source, so lookup and populate race by construction. Both are {@code synchronized} on this
 * object. A single lock is the right first design here rather than a striped one: the traffic is about 200
 * lookups per query per segment and each critical section is a hash lookup and an array copy, so the lock
 * is held for nanoseconds and contention is measured rather than assumed. Access-order
 * {@link LinkedHashMap} is not thread safe even for reads, since a read reorders the list, so the lock is
 * required and not merely prudent.
 */
@Log4j2
public final class LruVectorCache {

    /**
     * Bytes charged per entry on top of the vector itself: the {@link LinkedHashMap} node, its boxed
     * {@link Integer} key, and the {@code float[]} object header. An estimate, and deliberately a
     * generous one - the point of charging it at all is that the budget bounds retained heap rather than
     * payload, so an operator who sets 8 MB is not surprised by 8.3.
     */
    static final int ENTRY_OVERHEAD_BYTES = 80;

    private final long budgetBytes;
    private final int dimension;
    private final long entryBytes;

    /** Access ordered, so iteration order is least-recently-used first. Guarded by {@code this}. */
    private final LinkedHashMap<Integer, float[]> entries;

    /** Guarded by {@code this}; the running sum of {@link #entryBytes} over {@link #entries}. */
    private long heldBytes;

    // Counters. Written under the lock, read without it, so they are atomics rather than longs: an
    // operator reading the stats surface must not have to take a lock a query path is holding.
    private final AtomicLong hits = new AtomicLong();
    private final AtomicLong misses = new AtomicLong();
    private final AtomicLong evictions = new AtomicLong();

    private LruVectorCache(final int dimension, final long budgetBytes) {
        this.dimension = dimension;
        this.budgetBytes = budgetBytes;
        this.entryBytes = ENTRY_OVERHEAD_BYTES + (long) dimension * Float.BYTES;
        this.entries = new LinkedHashMap<>(16, 0.75f, true);
    }

    /**
     * A cache for a source of {@code dimension}-wide vectors at the node's configured budget, or
     * {@code null} to mean "do not cache".
     *
     * <p>Returning {@code null} rather than a zero-capacity cache is the contract that makes a budget of
     * zero a real control arm: the caller holds no cache object, so there is no lookup, no accounting and
     * no counter on the read path, and the loader is the Phase 2-5 loader exactly.
     *
     * <p>The setting is read defensively, for the reason every setting on this path is: a setting that
     * cannot be read must leave the query working, and the safe direction here is no cache.
     */
    public static LruVectorCache forSource(final int dimension) {
        final long budgetBytes;
        try {
            final ByteSizeValue configured = KNNSettings.getDirectIORescoreCacheBytesPerSource();
            budgetBytes = configured == null ? 0L : configured.getBytes();
        } catch (Exception e) {
            log.debug(
                "Could not read {}; serving the rescore seam without a cache",
                KNNSettings.KNN_DIRECT_IO_RESCORE_CACHE_BYTES_PER_SOURCE,
                e
            );
            return null;
        }
        return forSource(dimension, budgetBytes);
    }

    /**
     * A cache for a source of {@code dimension}-wide vectors at an explicit budget, or {@code null} for a
     * budget of zero or less. Package private twin of {@link #forSource(int)} for callers that have the
     * budget in hand, which in practice means tests.
     */
    static LruVectorCache forSource(final int dimension, final long budgetBytes) {
        if (dimension <= 0 || budgetBytes <= 0) {
            return null;
        }
        final LruVectorCache cache = new LruVectorCache(dimension, budgetBytes);
        if (cache.capacity() == 0) {
            log.warn(
                "The rescore vector cache budget of {} bytes cannot hold even one {}-dimension vector ({} bytes with overhead); "
                    + "serving the rescore seam without a cache",
                budgetBytes,
                dimension,
                cache.entryBytes
            );
            return null;
        }
        return cache;
    }

    /**
     * Copies the cached vector at {@code ord} into {@code dst} and marks it most recently used.
     *
     * @param dst where to copy the vector on a hit; untouched on a miss
     * @return true if {@code ord} was cached, false if it was not
     */
    public boolean load(final int ord, final float[] dst) {
        synchronized (this) {
            final float[] cached = entries.get(ord);
            if (cached == null) {
                misses.incrementAndGet();
                return false;
            }
            System.arraycopy(cached, 0, dst, 0, dimension);
        }
        hits.incrementAndGet();
        return true;
    }

    /**
     * Retains a copy of {@code vector} under {@code ord}, evicting least-recently-used entries first so
     * that the budget is not exceeded at any point.
     *
     * <p>Idempotent for an ordinal already held: the entry is refreshed to most-recently-used and nothing
     * is copied, since a segment's vector at an ordinal cannot have changed. A {@code vector} of the wrong
     * length is rejected rather than retained, because a short entry would be a wrong score later, far
     * from here.
     *
     * @param ord    the ordinal this vector belongs to
     * @param vector the decoded vector, copied rather than retained
     */
    public void put(final int ord, final float[] vector) {
        if (ord < 0 || vector == null || vector.length != dimension) {
            return;
        }
        synchronized (this) {
            // get() rather than containsKey(): on an access-ordered map only get() touches the entry, and
            // touching it is the point - an ordinal being put again is an ordinal being used again.
            if (entries.get(ord) != null) {
                return;
            }
            // Evict before inserting, so the budget is never momentarily over. An LRU that inserts and
            // then trims is over budget for as long as the insert takes, which for a cache sized against
            // an operator's memory is a promise worth not breaking.
            final Iterator<Map.Entry<Integer, float[]>> eldestFirst = entries.entrySet().iterator();
            while (heldBytes + entryBytes > budgetBytes && eldestFirst.hasNext()) {
                eldestFirst.next();
                eldestFirst.remove();
                heldBytes -= entryBytes;
                evictions.incrementAndGet();
            }
            if (heldBytes + entryBytes > budgetBytes) {
                // Nothing left to evict and it still does not fit, i.e. the budget is smaller than one
                // entry. forSource declines to build such a cache, so this is unreachable defence.
                return;
            }
            entries.put(ord, vector.clone());
            heldBytes += entryBytes;
        }
    }

    /**
     * How many entries fit in the budget. Derived from the byte budget rather than being the bound itself
     * - the bound is bytes - and exposed because "2,730 vectors" is the number an operator can reason
     * about against a candidate set of 200.
     */
    public int capacity() {
        return (int) Math.min(Integer.MAX_VALUE, budgetBytes / entryBytes);
    }

    /** The budget this cache was built with, in bytes. */
    public long budgetBytes() {
        return budgetBytes;
    }

    /**
     * A point-in-time read of the counters and the occupancy.
     *
     * <p>Not atomic across fields, and does not need to be: it exists so that an operator, an integration
     * test or a benchmark can see whether the cache is doing anything, and a hit rate skewed by the one
     * lookup that happened between two field reads is not a conclusion anyone draws.
     */
    public Stats stats() {
        final long held;
        final int count;
        synchronized (this) {
            held = heldBytes;
            count = entries.size();
        }
        return new Stats(hits.get(), misses.get(), evictions.get(), held, count, budgetBytes);
    }

    /**
     * What the cache has done and what it is holding.
     *
     * @param hits        lookups served from the cache
     * @param misses      lookups that had to read the device
     * @param evictions   entries dropped to stay within budget
     * @param bytes       bytes currently held, payload plus per-entry overhead
     * @param entries     vectors currently held
     * @param budgetBytes the budget, for reading {@code bytes} against
     */
    public record Stats(long hits, long misses, long evictions, long bytes, int entries, long budgetBytes) {

        /** Hits as a fraction of lookups, or zero before the first lookup. */
        public double hitRate() {
            final long lookups = hits + misses;
            return lookups == 0 ? 0.0d : (double) hits / lookups;
        }

        /** One line, for a log or a probe dump. */
        @Override
        public String toString() {
            return String.format(
                Locale.ROOT,
                "hits=%d misses=%d hitRate=%.4f evictions=%d entries=%d bytes=%d budget=%d",
                hits,
                misses,
                hitRate(),
                evictions,
                entries,
                bytes,
                budgetBytes
            );
        }
    }

    @Override
    public String toString() {
        return "LruVectorCache[dimension=" + dimension + ", budget=" + budgetBytes + " bytes, capacity=" + capacity() + " vectors]";
    }
}

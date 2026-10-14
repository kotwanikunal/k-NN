/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import com.google.common.cache.Cache;
import com.google.common.cache.CacheBuilder;
import com.google.common.cache.RemovalCause;
import lombok.extern.log4j.Log4j2;
import org.opensearch.core.common.unit.ByteSizeValue;
import org.opensearch.knn.index.KNNSettings;

import java.nio.ByteBuffer;
import java.util.concurrent.atomic.AtomicLong;

/**
 * The one cache of individual vectors on the Direct I/O random rescore path, for the whole node.
 *
 * <h2>Why it exists, in one measurement</h2>
 * Every phase 12-14 arm, and the store-factory arm it was compared against, ran with <b>no vector cache at
 * all</b>. An LRU one existed in phase 7 and was deleted in phase 12 as unreachable code after phase 10
 * removed the channel that reached it — on reachability, never on performance evidence. So the question
 * "would a cache help" had never been asked of the design that ships.
 *
 * <p>It was asked, at scale, before a line of this was written. MEASURED 2026-10-05 on {@code dio-10m}:
 * an exact LRU stack-distance replay at vector granularity of 5,000 real OSB query vectors against the
 * exact rescore candidate set (k=200 {@code rescore:false}, which is what a k=100 rescore asks for at 32x
 * compression and 768 dims), 1,000,000 references over 544,060 unique vectors — a 1.70 GB working set:
 *
 * <table>
 *   <caption>steady-state hit rate by budget; ceiling 45.6%</caption>
 *   <tr><th>budget</th><th>entries</th><th>hit rate</th></tr>
 *   <tr><td><b>16 MB</b></td><td>5,127</td><td><b>40.3%</b></td></tr>
 *   <tr><td>64 MB</td><td>20,510</td><td>40.9%</td></tr>
 *   <tr><td>128 MB</td><td>41,020</td><td>41.4%</td></tr>
 *   <tr><td>256 MB</td><td>82,040</td><td>42.3%</td></tr>
 *   <tr><td>1024 MB</td><td>328,160</td><td>48.1%</td></tr>
 * </table>
 *
 * 16 MB captures 84% of the achievable reuse, and replaying the same 5,000 queries three times does not
 * inflate the figure (128 MB: 39.8% on every pass), so it is inner-product HNSW hub reuse rather than an
 * artefact of the query set. The page cache under mmap did not deliver this at 10M because it also had to
 * hold the graph. INFERRED from that curve: a 64-128 MB cache should take {@code .vec} device reads per
 * 5,000 queries from 707k to ~420k, with QPS up proportionally on an IOPS-capped volume.
 *
 * <h2>Vector granularity comes for free</h2>
 * {@code Lucene99FlatVectorsReader}'s values object asks this input for exactly one vector's range — a
 * {@code prefetch(offset, vectorBytes)} and then a {@code readFloats} of the same run — so a key of
 * {@code (file, offset)} <em>is</em> one vector. Nothing has to be grouped, aligned or split, and the value
 * is the vector's bytes rather than the 8 KiB block they sit in: caching the block would cache 2.7 vectors'
 * worth of neighbours that the measured access pattern does not ask for, and would charge the budget 2.7x
 * for the 40% it buys.
 *
 * <h2>File identity: path and length, not the handle</h2>
 * The key's file part is {@code <absolute path>#<file length>} ({@link #fileId}), built once per
 * {@code .vec} open, and <b>not</b> the {@code Handle}'s object identity. Identity would be wrong in the
 * direction that matters: every {@code DirectIOVectorIndexInput#open} creates its own {@code Handle}, and
 * the codec seam opens a {@code .vec} once per reader, so two readers of the same file would never share an
 * entry — which is exactly the reuse the gate above measured and this cache exists to collect.
 *
 * <p>Path and length cannot collide across different content either, and the reason is structural rather
 * than hopeful. A shard's files live under {@code .../indices/<index-uuid>/<shard>/index/} (MEASURED: the
 * bench fixture's path is {@code .../indices/u9rQ-8RMRqKoIcFHnsJexg/0/index/}), and Lucene's segment names
 * come from a counter that only increases within an index, so a name is never reused for different bytes
 * within one index and a different index has a different UUID in its path. The file length is a second
 * discriminator on top of that. A Lucene segment file is also immutable once written, so the bytes at a
 * given offset of a given file never change.
 *
 * <h2>Entries of a file that has been closed: LRU evicts them, and that is the choice</h2>
 * Nothing invalidates a merged-away segment's entries. They are <b>never wrong</b> — the key says which
 * file's bytes they are, and that file's bytes are immutable — they are only <em>cold</em>, and they occupy
 * budget until LRU pushes them out, which costs hit rate for a bounded period and never correctness.
 *
 * <p>The alternative, track-and-invalidate on {@code Handle} close, needs a per-file index of every key
 * written, maintained on every {@code put} and walked on every close. That index is as large as the cache's
 * own key set, puts are the hot path, and closes are frequent on the merge path. For a mechanism whose
 * whole measured benefit is 40% at 16 MB, doubling the per-entry overhead and adding work to the put path
 * to recover eviction pressure that LRU recovers anyway is the wrong trade. Recorded as a decision, not an
 * omission.
 *
 * <h2>Guava divides the budget across segments, and a small budget therefore holds less than it looks</h2>
 * MEASURED while building this: a budget of three entries' weight holds <b>zero</b> entries. Guava's
 * {@code LocalCache} splits {@code maximumWeight} across {@code concurrencyLevel} segments and evicts
 * against the <em>per-segment</em> share, so an entry heavier than {@code max_bytes / concurrencyLevel} is
 * inserted and immediately evicted. With the default concurrency level of 4 and a 3,272 B entry (a 768-dim
 * fp32 vector plus {@link #ENTRY_OVERHEAD_BYTES}) the smallest budget that can hold anything at all is
 * about <b>13 KB</b>, and the smallest that behaves like an LRU of its nominal size is a good deal larger.
 *
 * <p>Two consequences worth stating rather than discovering:
 * <ul>
 *   <li>Do not set {@code max_bytes} to a few KB expecting a few entries. At the sizes the gate measured
 *       (16 MB and up) the division is immaterial — 16 MB over 4 segments is 4 MB and ~1,281 entries each,
 *       5,124 in total against the replay's 5,127.</li>
 *   <li>The gate's figure is from a <b>global</b> LRU and this is four LRUs keyed by hash, so the real hit
 *       rate will sit slightly <em>below</em> the modelled one through segment skew. The 40.3%-at-16-MB
 *       number is therefore an upper bound on what this implementation can deliver, not a prediction of it.
 *       The node arm measures the real figure; that is why the shipped default is off.</li>
 * </ul>
 *
 * <h2>Off by default, and off means off</h2>
 * {@code knn.direct_io.rescore.cache.max_bytes} defaults to {@code 0b}. When it is zero {@link #cache} is
 * {@code null} and every lookup is one volatile read and a null check — no key allocated, no setting read,
 * no counter touched. The shipped default is the node arm's to decide; this phase ships the mechanism and
 * the correctness proof.
 */
@Log4j2
public final class DirectIOVectorCache {

    /**
     * Per-entry overhead charged on top of the vector's bytes, so that
     * {@code knn.direct_io.rescore.cache.max_bytes} bounds the cache's real heap cost rather than only its
     * payload. {@code 200 B} is the figure the P15-8-A reuse replay used for its entry model — the key
     * record, the {@code byte[]} header, and Guava's own segment entry — so the hit-rate curve in the class
     * javadoc is a curve for <em>this</em> weigher and the budgets in it mean the same thing here.
     */
    static final int ENTRY_OVERHEAD_BYTES = 200;

    /** One vector's bytes in one file: the whole of the key, because the reader asks per vector. */
    private record Key(String fileId, long offset) {
    }

    /**
     * The live cache, or {@code null} when {@code max_bytes} is {@code 0}. Volatile so that the hot path is
     * a single read, and so that a rebuild is published atomically.
     */
    private static volatile Cache<Key, byte[]> cache;

    /** The byte budget {@link #cache} was built at, so a rebuild is only done when it actually changed. */
    private static volatile long configuredBytes;

    private static final AtomicLong hits = new AtomicLong();
    private static final AtomicLong misses = new AtomicLong();
    private static final AtomicLong evictions = new AtomicLong();
    /** Charged weight currently held: the sum of the weigher over live entries. */
    private static final AtomicLong bytes = new AtomicLong();

    private DirectIOVectorCache() {}

    /** The identity {@link Key} uses for a file: two opens of one {@code .vec} must produce one string. */
    static String fileId(final java.nio.file.Path path, final long fileLength) {
        return path.toString() + "#" + fileLength;
    }

    /**
     * Reads {@code knn.direct_io.rescore.cache.max_bytes} and rebuilds the cache if it has changed.
     *
     * <p>Called once per prefetch burst, from {@code DirectIOVectorIndexInput#prefetch} at the burst
     * boundary — the same cadence and the same contract as {@code DirectIOReadPool#executor()}, so an
     * operator's change takes effect on the next burst with no restart and there is no startup gap for a
     * value that was already persisted. A setting that cannot be read leaves the cache exactly as it is.
     *
     * <p><b>A change rebuilds; it does not resize.</b> Guava fixes {@code maximumWeight} at
     * {@code CacheBuilder#build()} and offers no setter, which is why
     * {@code NativeMemoryCacheManager#rebuildCache} also rebuilds. So the entries are dropped and the cache
     * warms again — on the measured curve that costs the first few hundred queries after the change and
     * nothing after. Retuning this knob is therefore not free, unlike retuning the queue capacity.
     */
    static void applyConfiguredSize() {
        final long wanted;
        try {
            final ByteSizeValue configured = KNNSettings.getDirectIORescoreCacheMaxBytes();
            wanted = configured == null ? 0L : configured.getBytes();
        } catch (RuntimeException e) {
            log.debug("Could not read knn.direct_io.rescore.cache.max_bytes; leaving the vector cache as it is", e);
            return;
        }
        if (wanted == configuredBytes) {
            return;
        }
        configure(wanted);
    }

    /**
     * Rebuilds the cache at {@code maxBytes}, or drops it when {@code maxBytes <= 0}. Synchronized so two
     * concurrent bursts seeing the same change build one cache between them.
     */
    static synchronized void configure(final long maxBytes) {
        if (maxBytes == configuredBytes) {
            return;
        }
        final Cache<Key, byte[]> previous = cache;
        if (maxBytes <= 0) {
            cache = null;
            configuredBytes = 0;
            log.info("The k-NN Direct I/O vector cache is off (knn.direct_io.rescore.cache.max_bytes=0)");
        } else {
            cache = CacheBuilder.newBuilder()
                .maximumWeight(maxBytes)
                // Weight is the payload plus the entry's own cost, so the setting bounds heap and not just
                // bytes of vector. See ENTRY_OVERHEAD_BYTES.
                .weigher((Key key, byte[] value) -> value.length + ENTRY_OVERHEAD_BYTES)
                // Deliberately NOT concurrencyLevel(1), which NativeMemoryCacheManager sets: this cache is
                // read by every search thread inside every rescore, where that one lock would serialise the
                // whole node's vector lookups behind each other.
                .removalListener(notification -> onRemoval(notification.getValue(), notification.getCause()))
                .build();
            configuredBytes = maxBytes;
            log.info("The k-NN Direct I/O vector cache holds up to {} bytes of rescore vectors", maxBytes);
        }
        if (previous != null) {
            // invalidateAll fires the removal listener for every entry, which is what returns their weight.
            previous.invalidateAll();
        }
        bytes.set(0);
    }

    /**
     * Returns a removed entry's weight to {@link #bytes}, and counts it as an eviction when the cache chose
     * to drop it rather than being told to.
     *
     * <p>{@code RemovalCause#wasEvicted()} is package private in Guava, so the cause is tested directly:
     * {@code SIZE} is the only eviction this cache can see, because it sets no expiry and holds strong
     * references. {@code EXPLICIT} and {@code REPLACED} are not evictions — they still return the weight,
     * which is what keeps {@code bytes} correct across a {@code put} that overwrites and across
     * {@code invalidateAll} on a rebuild.
     */
    private static void onRemoval(final byte[] value, final RemovalCause cause) {
        if (value != null) {
            bytes.addAndGet(-(long) (value.length + ENTRY_OVERHEAD_BYTES));
        }
        if (cause == RemovalCause.SIZE) {
            evictions.incrementAndGet();
        }
    }

    /**
     * The {@code length} bytes at {@code offset} of {@code fileId}, or {@code null} on a miss or with the
     * cache off.
     *
     * <p>{@code length} is checked against the entry rather than being part of the key, because the reader
     * always asks for one whole vector and so always asks for the same length at a given offset; a
     * mismatch therefore means a caller this cache was not designed for, and the honest answer for one is a
     * miss rather than a partial.
     */
    static byte[] get(final String fileId, final long offset, final int length) {
        final Cache<Key, byte[]> current = cache;
        if (current == null) {
            return null;
        }
        final byte[] value = current.getIfPresent(new Key(fileId, offset));
        if (value == null || value.length != length) {
            misses.incrementAndGet();
            return null;
        }
        hits.incrementAndGet();
        return value;
    }

    /** Whether a lookup can succeed at all, so a caller can skip building a key it will not use. */
    static boolean isOn() {
        return cache != null;
    }

    /**
     * Copies {@code length} bytes from {@code source} at {@code position} into the cache under
     * {@code (fileId, offset)}. A copy and not a view, because {@code source} is a staging buffer that the
     * next burst reuses.
     */
    static void put(final String fileId, final long offset, final ByteBuffer source, final int position, final int length) {
        final Cache<Key, byte[]> current = cache;
        if (current == null) {
            return;
        }
        final byte[] value = new byte[length];
        source.get(position, value, 0, length);
        bytes.addAndGet(value.length + ENTRY_OVERHEAD_BYTES);
        current.put(new Key(fileId, offset), value);
    }

    /** Lookups that returned bytes. See {@link #lookupsPerVector()} for how to read this against misses. */
    public static long hits() {
        return hits.get();
    }

    /** Lookups that did not return bytes. */
    public static long misses() {
        return misses.get();
    }

    /**
     * <b>Two.</b> One vector is looked up twice on the way to being served: once in {@code prefetch}, where
     * a hit removes it from the burst and so saves the device read, and once in {@code readFloats}, where a
     * hit serves the bytes. Both sites count, so {@code hits / (hits + misses)} is the hit <em>rate</em> and
     * is directly comparable with the 40.3%-at-16-MB figure in the class javadoc, while the absolute counts
     * are twice the number of vectors. Stated here rather than left for an operator to discover from a
     * ratio that looks right and totals that look doubled.
     */
    public static int lookupsPerVector() {
        return 2;
    }

    /** Entries LRU pushed out under {@code max_bytes}. Zero with a budget larger than the working set. */
    public static long evictions() {
        return evictions.get();
    }

    /** Charged weight held right now: vector bytes plus {@link #ENTRY_OVERHEAD_BYTES} per entry. */
    public static long bytes() {
        return bytes.get();
    }

    /** Entries held right now. */
    public static long entries() {
        final Cache<Key, byte[]> current = cache;
        return current == null ? 0L : current.size();
    }

    /** The budget the live cache was built at; {@code 0} when the cache is off. */
    public static long maxBytes() {
        return configuredBytes;
    }

    /** Test seam: drops the cache and every counter, so one test cannot read another's entries. */
    static synchronized void resetForTesting() {
        final Cache<Key, byte[]> current = cache;
        cache = null;
        configuredBytes = 0;
        if (current != null) {
            current.invalidateAll();
        }
        hits.set(0);
        misses.set(0);
        evictions.set(0);
        bytes.set(0);
    }
}

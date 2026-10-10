/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.memory;

import lombok.extern.log4j.Log4j2;
import org.opensearch.knn.jni.JNICommons;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * One fixed-size off-heap slab of full-precision vector bytes, freed when the cache says so rather than
 * when a collector gets round to it.
 *
 * <h2>Why native memory and not a direct ByteBuffer (MEASURED)</h2>
 * {@code ByteBuffer.allocateDirect} counts against {@code -XX:MaxDirectMemorySize}, which this plugin's
 * Direct I/O staging and netty already share, and frees only after a garbage collection. Phase 15 D3c is
 * the record of what that lifetime costs: a 1 GB ceiling on a gauge that was measuring arena churn awaiting
 * collection wedged the feature within two minutes on the bench node. A slab instead comes from
 * {@link JNICommons#allocateSlab(long)} and is accounted by k-NN under the native memory budget, the same
 * way a faiss index's {@code malloc} memory already is.
 *
 * <p>The other deterministic option, an FFM {@code Arena.ofShared()} in a reflectively loaded JDK22+ class
 * behind a Java 21 fallback, was evaluated and rejected on two measurements (output/bench/i1):
 *
 * <table>
 *   <caption>32 MiB slab, dev desktop, median of 100</caption>
 *   <tr><th></th><th>FFM {@code Arena.ofShared}, JDK 25</th><th>this, JDK 21</th></tr>
 *   <tr><td>allocate</td><td>6,834 us</td><td><b>4.8 us</b></td></tr>
 *   <tr><td>free</td><td>2,060 us</td><td><b>3.3 us</b></td></tr>
 *   <tr><td>VmRSS after touching 16 slabs</td><td>+525,152 KB</td><td>+525,396 KB</td></tr>
 *   <tr><td>VmRSS after freeing them</td><td>+972 KB</td><td>+1,252 KB</td></tr>
 *   <tr><td>JVM direct pool</td><td>unchanged at 0</td><td>unchanged at 8,192 B</td></tr>
 * </table>
 *
 * The three-order-of-magnitude gap is not FFM being slow: {@code Arena.allocate} <em>zero-fills</em>, which
 * the cache then immediately overwrites with vector bytes, and its {@code close} then pays to reclaim the
 * pages that zero-fill touched. Forcing the same semantics here (a {@code memset} after the allocation)
 * reproduces FFM's numbers exactly: 6,730 us and 2,066 us. The decisive reason is the other one, and it is
 * structural rather than a benchmark: this project sets {@code sourceCompatibility = targetCompatibility =
 * JavaVersion.VERSION_21} as literals in {@code build.gradle}, and {@code compileJava} excludes every
 * JDK22+ source file whenever either is at or below 21, so the exclusion is true for <em>every</em> build
 * whatever JDK runs gradle. MEASURED: the phase 15 jar contains
 * {@code MemorySegmentAddressExtractorJDK21.class} and no {@code ...JDK22.class}. An FFM slab class would
 * inherit that, i.e. it would never load, and its Java 21 fallback would have to be this class anyway.
 *
 * <h2>The slab is not zeroed, and that is safe because of how it is read</h2>
 * {@link JNICommons#allocateSlab(long)} does not clear the memory. Freshly allocated slabs therefore hold
 * whatever this process last had at those addresses. Nothing can observe it: a slot is handed to a reader
 * only after the cache's index records that a vector was written there, so every byte a reader sees is a
 * byte the cache put there. Zeroing would cost 6.7 ms per slab (above) to overwrite bytes that are about to
 * be overwritten.
 *
 * <h2>Lifetime: the read-write lock, not a reference count</h2>
 * A direct {@link ByteBuffer} over freed memory is still a perfectly valid Java object pointing at memory
 * the allocator may have handed to someone else, so a read after the free is undefined rather than an
 * exception. Two rules make that unreachable, and they are the ones
 * {@link NativeMemoryAllocation.IndexAllocation} already uses to keep a graph from being freed under a
 * search:
 * <ol>
 *   <li>A reader takes {@link #readLock()} for as long as it holds a buffer, and {@link #close()} takes the
 *       write lock, so a free cannot begin while a read is in flight.</li>
 *   <li>{@link #bytes()} and {@link #floats()} throw once the slab is closed, so a reader cannot newly
 *       obtain a view of freed memory. A reader that stashes the buffer and reads it after dropping the read
 *       lock has broken the contract; that is why those two methods are the only way to reach it and why
 *       both are documented as valid only under the read lock.</li>
 * </ol>
 * {@link #close()} is idempotent and synchronized, so a double free is a no-op rather than a crash.
 *
 * <h2>Absolute access only</h2>
 * The {@link ByteBuffer} and {@link FloatBuffer} returned here are shared by every reader of this slab, so
 * their mutable position and limit must never be touched. Callers use only the absolute forms
 * ({@code get(int)}, {@code get(int, float[], int, int)}, {@code put(int, ...)}), which is the same rule and
 * the same reason as {@code DirectIOVectorIndexInput.Source}, which carries the same pair for the same
 * staging buffer.
 */
@Log4j2
public final class VectorCacheSlab {

    /**
     * The slab size, 32 MiB. Fixed rather than a setting, because nothing an operator can measure would
     * tell them what to set it to: the figure that matters is the cache's total, which
     * {@code knn.direct_io.rescore.cache.max_bytes} and the native budget already bound, and the slab is
     * only the granularity at which that total is reclaimed. 32 MiB is the midpoint of the 16-64 MB range
     * the design note names, it is two orders of magnitude above the largest vector this holds (a
     * 4,096-dim fp32 vector is 16 KiB), and at a 4 GB cache it is 128 slabs, which is a small enough
     * population for slab-granular FIFO eviction to be a fine-grained policy rather than a blunt one.
     * A setting can be added later if a measurement ever asks for one.
     */
    public static final int SLAB_BYTES = 32 * 1024 * 1024;

    /** Slabs allocated and freed since node start, for {@code direct_io_stats} and for leak tests. */
    private static final AtomicLong allocated = new AtomicLong();
    private static final AtomicLong freed = new AtomicLong();

    private final long address;
    private final int capacity;
    private final ByteBuffer bytes;
    private final FloatBuffer floats;
    private final ReentrantReadWriteLock lock = new ReentrantReadWriteLock();
    private volatile boolean closed;

    private VectorCacheSlab(final long address, final int capacity, final ByteBuffer bytes) {
        this.address = address;
        this.capacity = capacity;
        this.bytes = bytes;
        this.floats = bytes.asFloatBuffer();
    }

    /**
     * Allocates a slab of {@link #SLAB_BYTES}.
     *
     * @return the slab
     * @throws OutOfMemoryError if the native allocation failed
     */
    public static VectorCacheSlab allocate() {
        return allocate(SLAB_BYTES);
    }

    /**
     * Allocates a slab of {@code capacity} bytes. Only tests pass anything other than
     * {@link #SLAB_BYTES}.
     *
     * @param capacity slab size in bytes, positive
     * @return the slab
     * @throws OutOfMemoryError if the native allocation failed
     */
    public static VectorCacheSlab allocate(final int capacity) {
        if (capacity <= 0) {
            throw new IllegalArgumentException("A slab's capacity must be positive, got " + capacity);
        }
        final long address = JNICommons.allocateSlab(capacity);
        if (address == 0) {
            throw new OutOfMemoryError("Could not allocate a " + capacity + " byte vector cache slab");
        }
        final ByteBuffer buffer = JNICommons.wrapSlab(address, capacity);
        if (buffer == null) {
            // NewDirectByteBuffer can fail where the JVM cannot create the object, and then the memory
            // would be unreachable and unfreeable. Give it back before reporting the failure.
            JNICommons.freeSlab(address);
            throw new OutOfMemoryError("Could not wrap a " + capacity + " byte vector cache slab");
        }
        allocated.incrementAndGet();
        // Little endian because that is the order a Lucene .vec file stores floats in, so a hit and a
        // staged miss decode the same bytes the same way. See I5.
        return new VectorCacheSlab(address, capacity, buffer.order(ByteOrder.LITTLE_ENDIAN));
    }

    /** The slab's size in bytes. */
    public int capacity() {
        return capacity;
    }

    /**
     * The slab's native address, for a caller that scores from the address directly rather than through
     * the buffer. Valid only under {@link #readLock()}.
     */
    public long address() {
        return address;
    }

    /** Whether {@link #close()} has run. */
    public boolean isClosed() {
        return closed;
    }

    /**
     * The slab's bytes. <b>Absolute access only</b>, and valid only while the caller holds
     * {@link #readLock()}.
     *
     * @throws IllegalStateException if the slab is closed
     */
    public ByteBuffer bytes() {
        ensureOpen();
        return bytes;
    }

    /**
     * The slab's bytes as floats, which is how a hit is scored. <b>Absolute access only</b>, and valid only
     * while the caller holds {@link #readLock()}.
     *
     * @throws IllegalStateException if the slab is closed
     */
    public FloatBuffer floats() {
        ensureOpen();
        return floats;
    }

    /** Taken for as long as a caller holds a buffer or the address. See the class javadoc. */
    public void readLock() {
        lock.readLock().lock();
    }

    public void readUnlock() {
        lock.readLock().unlock();
    }

    /**
     * Frees the slab. Idempotent, so a double free is a no-op; and it takes the write lock, so it cannot
     * begin while a read is in flight.
     */
    public synchronized void close() {
        if (closed) {
            return;
        }
        lock.writeLock().lock();
        try {
            // Set inside the lock so that a reader which has already passed ensureOpen() and is holding the
            // read lock finishes before the free, while a reader arriving after it sees the slab closed.
            closed = true;
            JNICommons.freeSlab(address);
            freed.incrementAndGet();
        } finally {
            lock.writeLock().unlock();
        }
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("This vector cache slab has been freed");
        }
    }

    /** Slabs allocated since node start. */
    public static long allocatedCount() {
        return allocated.get();
    }

    /** Slabs freed since node start. {@code allocatedCount() - freedCount()} is the live population. */
    public static long freedCount() {
        return freed.get();
    }

    /** Test seam: so one test's counters cannot be read by another. */
    static void resetCountersForTesting() {
        allocated.set(0);
        freed.set(0);
    }

    @Override
    public String toString() {
        return "VectorCacheSlab(capacity=" + capacity + ", closed=" + closed + ")";
    }
}

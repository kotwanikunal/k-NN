/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import lombok.extern.log4j.Log4j2;

import java.lang.reflect.Method;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.concurrent.atomic.AtomicLong;

/**
 * One fixed-size off-heap slab of vector bytes for {@link DirectIOVectorCache}, allocated without JNI and
 * freed when the cache says so rather than when a collector gets round to it.
 *
 * <h2>Two backends, chosen once, and the choice is measured</h2>
 * <ol>
 *   <li><b>{@code ffm}</b> — {@code java.lang.foreign.Arena.ofShared()}, {@code allocate(bytes, 64)} and
 *       {@code MemorySegment#asByteBuffer()}, reached <b>only by reflection</b>, with no compile-time
 *       reference to {@code java.lang.foreign} anywhere in this file. Release is {@code Arena#close()}.
 *       That restriction is not stylistic: this project pins {@code sourceCompatibility} and
 *       {@code targetCompatibility} to 21 ({@code build.gradle:234-235}) and {@code compileJava} excludes
 *       every JDK 22+ source file while either is at or below 21 ({@code build.gradle:297-299}), so a class
 *       that <em>names</em> an FFM type never reaches the jar. The same reflection pattern already ships in
 *       {@code MemorySegmentAddressExtractorJDK21}.</li>
 *   <li><b>{@code direct_buffer}</b> — {@link ByteBuffer#allocateDirect(int)}, for a runtime where
 *       {@code java.lang.foreign.Arena} is absent or an arena cannot be allocated. Release drops the
 *       reference and the memory comes back at a later collection.</li>
 * </ol>
 *
 * <h2>Why FFM is first (MEASURED 2026-10-12, dev desktop, output/bench/m1)</h2>
 * {@code java.lang.foreign} is a preview API in JDK 21 (JEP 442), but the preview gate is enforced by
 * {@code javac} and by a class file's preview minor version, and a reflective call passes through neither.
 * So the four methods resolve and work on Corretto 21.0.11 and on Temurin 25.0.4 alike, with no warning,
 * no exception and no {@code --enable-preview}.
 *
 * <table>
 *   <caption>32 MiB slab, median of 100 after 10 warm-ups; VmRSS for 16 slabs allocated and touched</caption>
 *   <tr><th></th><th>ffm, JDK 21</th><th>ffm, JDK 25</th><th>direct_buffer, JDK 21</th>
 *       <th>direct_buffer, JDK 25</th></tr>
 *   <tr><td>allocate p50</td><td>7,111 us</td><td>6,730 us</td><td>7,366 us</td><td>7,557 us</td></tr>
 *   <tr><td>release p50</td><td>2,230 us</td><td>2,060 us</td><td>0.1 us</td><td>0.1 us</td></tr>
 *   <tr><td>VmRSS, 16 touched</td><td>+529,152 kB</td><td>+536,948 kB</td><td>+529,440 kB</td>
 *       <td>+535,708 kB</td></tr>
 *   <tr><td>VmRSS after release</td><td><b>+4,964 kB</b></td><td><b>+13,024 kB</b></td>
 *       <td>+529,440 kB, unchanged</td><td>+535,708 kB, unchanged</td></tr>
 * </table>
 *
 * The last row is the decision. Neither backend is fast to allocate, because both zero-fill, and neither has
 * to be: a 4 GB cache is 128 slabs, so filling it from empty costs about 0.9 s spread over the first 128
 * misses that need a slab, against a device read of 0.3-3 ms per miss. What differs is what happens when the
 * cache shrinks or an operator sets {@code knn.direct_io.rescore.cache.max_bytes} back to {@code 0b}: FFM
 * gives the pages back inside {@link #close()}, and {@code allocateDirect} holds them until a collection
 * finds the buffer, which is exactly the GC-timed release D3c removed from the staging buffers.
 *
 * <h2>{@code MaxDirectMemorySize}: outside it on the node's JDK, inside it on 21 (MEASURED)</h2>
 * With {@code -XX:MaxDirectMemorySize=64m}, 16 FFM slabs (512 MiB) allocate fine on JDK 25, which does not
 * count them in {@code BufferPoolMXBean("direct")} either. On JDK 21 the <em>view</em> charges the limit,
 * not the arena: the third {@code asByteBuffer()} throws
 * {@code OutOfMemoryError: Cannot reserve 33554495 bytes of direct buffer memory} from
 * {@code java.nio.Bits.reserveMemory}, and {@code Arena#close} gives that reservation straight back with no
 * GC. {@code allocateDirect} is inside the limit on both. So on a JDK 21 runtime this cache shares
 * {@code MaxDirectMemorySize} with Direct I/O staging and netty, and on a JDK 22+ runtime it does not; its
 * own ceiling, {@code knn.direct_io.rescore.cache.max_bytes}, is the bound either way.
 *
 * <h2>An allocation that fails is not an error</h2>
 * {@link #allocate(int)} returns {@code null} rather than throwing when the backend will not give up the
 * memory. The caller is the cache, and the honest response there is to stay the size it is: a cache that
 * cannot grow costs hit rate, and nothing else. Propagating an {@link OutOfMemoryError} would cost a query
 * its answer for the sake of a cache entry.
 *
 * <h2>Absolute access only, and no view outlives the store's read lock</h2>
 * {@link #bytes()} and {@link #floats()} are shared by every reader of the slab, so their position and limit
 * must never be touched; callers use only the absolute {@code get}/{@code put} forms. A slab is closed only
 * under {@link DirectIOVectorCache}'s write lock, which every reader excludes by holding its read lock, so
 * no reader can be inside the slab when it goes away. That is why there is no per-slab lock here.
 *
 * <p>Reading through a buffer a caller kept from <em>before</em> the close is still a broken contract, and
 * what it does differs by backend: FFM throws {@code IllegalStateException: Already closed} (MEASURED),
 * {@code allocateDirect} returns the old bytes until the collector unmaps them. Neither crashes the JVM,
 * which is the one thing a JNI slab could not promise.
 */
@Log4j2
final class DirectIOSlab {

    /**
     * The slab size, 32 MiB, fixed rather than a setting for the reason 16-I's {@code VectorCacheSlab}
     * recorded: nothing an operator can measure would tell them what to set it to. The figure that matters
     * is the cache's total, which {@code knn.direct_io.rescore.cache.max_bytes} bounds, and the slab is only
     * the granularity at which that total is reclaimed. At a 4 GB cache it is 128 slabs, which is a fine
     * enough population for slab-granular FIFO eviction, and it is three orders of magnitude above the
     * largest vector it holds (a 4,096-dim fp32 vector is 16 KiB).
     */
    static final int SLAB_BYTES = 32 * 1024 * 1024;

    /** Alignment asked of an FFM arena. One cache line, so a slot never straddles one needlessly. */
    private static final long ALIGNMENT = 64L;

    static final String BACKEND_FFM = "ffm";
    static final String BACKEND_DIRECT_BUFFER = "direct_buffer";

    /** Slabs allocated and freed since node start, for the stats and for leak tests. */
    private static final AtomicLong allocated = new AtomicLong();
    private static final AtomicLong freed = new AtomicLong();

    /**
     * The reflective handles on {@code java.lang.foreign}, resolved once and proven once.
     *
     * <p>Resolution alone is not proof: a runtime can carry the classes and still refuse an arena, and on
     * JDK 21 the failure arrives at {@code asByteBuffer} rather than at {@code allocate}. So the holder
     * allocates one small arena, views it, writes a float, reads it back and closes it. A backend that
     * cannot do that for 1 KiB will not do it for 32 MiB.
     */
    private static final class Ffm {
        private static final Method OF_SHARED;
        private static final Method ALLOCATE;
        private static final Method AS_BYTE_BUFFER;
        private static final Method CLOSE;
        private static final boolean USABLE;

        static {
            Method ofShared = null;
            Method allocate = null;
            Method asByteBuffer = null;
            Method close = null;
            boolean usable = false;
            try {
                final Class<?> arena = Class.forName("java.lang.foreign.Arena");
                final Class<?> segment = Class.forName("java.lang.foreign.MemorySegment");
                ofShared = arena.getMethod("ofShared");
                allocate = arena.getMethod("allocate", long.class, long.class);
                asByteBuffer = segment.getMethod("asByteBuffer");
                close = arena.getMethod("close");
                final Object probe = ofShared.invoke(null);
                try {
                    final ByteBuffer view = (ByteBuffer) asByteBuffer.invoke(allocate.invoke(probe, 1024L, ALIGNMENT));
                    view.order(ByteOrder.LITTLE_ENDIAN).putFloat(0, 1.0f);
                    usable = view.getFloat(0) == 1.0f && view.isDirect();
                } finally {
                    close.invoke(probe);
                }
            } catch (ClassNotFoundException | NoSuchMethodException e) {
                log.info("This runtime has no java.lang.foreign.Arena, so the k-NN vector cache uses direct buffers");
            } catch (Throwable t) {
                log.info("A java.lang.foreign.Arena could not be used, so the k-NN vector cache uses direct buffers", t);
            }
            OF_SHARED = ofShared;
            ALLOCATE = allocate;
            AS_BYTE_BUFFER = asByteBuffer;
            CLOSE = close;
            USABLE = usable;
        }

        private Ffm() {}
    }

    /** The arena backing {@link #bytes}, or null for the {@code direct_buffer} backend. */
    private final Object arena;
    private final int capacity;
    private ByteBuffer bytes;
    private FloatBuffer floats;
    private boolean closed;

    private DirectIOSlab(final Object arena, final int capacity, final ByteBuffer bytes) {
        this.arena = arena;
        this.capacity = capacity;
        this.bytes = bytes;
        this.floats = bytes.asFloatBuffer();
    }

    /**
     * Which backend this node allocates slabs through, for
     * {@code direct_io_stats.cache.slab_memory}: {@value #BACKEND_FFM} or {@value #BACKEND_DIRECT_BUFFER}.
     *
     * @return the backend's name, never null
     */
    static String backend() {
        return Ffm.USABLE ? BACKEND_FFM : BACKEND_DIRECT_BUFFER;
    }

    /**
     * A slab of {@code capacity} bytes, little endian because that is the order a Lucene {@code .vec} stores
     * floats in, so a hit decodes the bytes exactly as a staged miss does.
     *
     * @param capacity the slab size in bytes, positive; only tests pass anything but {@link #SLAB_BYTES}
     * @return the slab, or {@code null} when the backend would not give up the memory
     */
    static DirectIOSlab allocate(final int capacity) {
        return allocate(capacity, backend());
    }

    /**
     * A slab from a named backend, so that a test can exercise the fallback on a runtime where FFM is
     * available. Production always calls {@link #allocate(int)}, which asks {@link #backend()}.
     *
     * @param capacity the slab size in bytes, positive
     * @param backend {@value #BACKEND_FFM} or {@value #BACKEND_DIRECT_BUFFER}
     * @return the slab, or {@code null} when that backend would not give up the memory
     */
    static DirectIOSlab allocate(final int capacity, final String backend) {
        if (capacity <= 0) {
            throw new IllegalArgumentException("A slab's capacity must be positive, got " + capacity);
        }
        final DirectIOSlab slab = BACKEND_FFM.equals(backend) ? allocateFfm(capacity) : allocateDirectBuffer(capacity);
        if (slab != null) {
            allocated.incrementAndGet();
        }
        return slab;
    }

    /** Whether the {@code ffm} backend resolved and proved itself on this runtime. */
    static boolean ffmUsable() {
        return Ffm.USABLE;
    }

    private static DirectIOSlab allocateFfm(final int capacity) {
        Object arena = null;
        try {
            arena = Ffm.OF_SHARED.invoke(null);
            final ByteBuffer view = (ByteBuffer) Ffm.AS_BYTE_BUFFER.invoke(Ffm.ALLOCATE.invoke(arena, (long) capacity, ALIGNMENT));
            return new DirectIOSlab(arena, capacity, view.order(ByteOrder.LITTLE_ENDIAN));
        } catch (Throwable t) {
            // On JDK 21 this is where MaxDirectMemorySize is enforced, inside asByteBuffer, so the arena may
            // be open with memory already behind it. Give it back before reporting no slab.
            if (arena != null) {
                try {
                    Ffm.CLOSE.invoke(arena);
                } catch (Throwable ignored) {
                    log.debug("A vector cache arena could not be closed after a failed allocation", ignored);
                }
            }
            log.debug("Could not allocate a {} byte vector cache slab through java.lang.foreign", capacity, t);
            return null;
        }
    }

    private static DirectIOSlab allocateDirectBuffer(final int capacity) {
        try {
            return new DirectIOSlab(null, capacity, ByteBuffer.allocateDirect(capacity).order(ByteOrder.LITTLE_ENDIAN));
        } catch (OutOfMemoryError e) {
            // MaxDirectMemorySize, which a direct buffer is always inside. The JVM has already forced a
            // collection trying to make room, so there is nothing further to try here.
            log.debug("Could not allocate a {} byte direct buffer for a vector cache slab", capacity, e);
            return null;
        }
    }

    /** The slab's size in bytes. */
    int capacity() {
        return capacity;
    }

    /** Whether {@link #close()} has run. */
    boolean isClosed() {
        return closed;
    }

    /**
     * The slab's bytes. <b>Absolute access only.</b>
     *
     * @return the buffer, little endian
     * @throws IllegalStateException if the slab has been released
     */
    ByteBuffer bytes() {
        ensureOpen();
        return bytes;
    }

    /**
     * The slab's bytes as floats, which is how a hit is copied out. <b>Absolute access only.</b>
     *
     * @return the float view, little endian
     * @throws IllegalStateException if the slab has been released
     */
    FloatBuffer floats() {
        ensureOpen();
        return floats;
    }

    /**
     * Releases the slab: {@code Arena#close()} on the {@code ffm} backend, which gives the pages back here
     * and now, or dropping this object's reference to the buffer on {@code direct_buffer}, which leaves the
     * memory to the next collection that finds it.
     *
     * <p>Idempotent, because a shared arena's second {@code close()} throws
     * {@code IllegalStateException: Already closed} (MEASURED) and a double release has to be a no-op: it is
     * reachable from a resize that races a shutdown.
     */
    void close() {
        if (closed) {
            return;
        }
        closed = true;
        bytes = null;
        floats = null;
        if (arena == null) {
            freed.incrementAndGet();
            return;
        }
        try {
            Ffm.CLOSE.invoke(arena);
            freed.incrementAndGet();
        } catch (Throwable t) {
            // Nothing a node can do about a failed close, and the slab is out of the cache's hands either
            // way. Counting it as freed would make the leak test pass on a leak, so it is not counted.
            log.warn("Could not release a {} byte vector cache slab", capacity, t);
        }
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException("This vector cache slab has been released");
        }
    }

    /** Slabs allocated since node start. */
    static long allocatedCount() {
        return allocated.get();
    }

    /** Slabs released since node start. {@code allocatedCount() - freedCount()} is the live population. */
    static long freedCount() {
        return freed.get();
    }

    /** Test seam: so one test's counters cannot be read by another. */
    static void resetCountersForTesting() {
        allocated.set(0);
        freed.set(0);
    }

    @Override
    public String toString() {
        return "DirectIOSlab(" + backend() + ", capacity=" + capacity + ", closed=" + closed + ")";
    }
}

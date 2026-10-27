/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import lombok.extern.log4j.Log4j2;
import org.opensearch.knn.index.KNNSettings;

import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * The one pool of threads that performs the rescore seam's Direct I/O reads, for the whole node.
 *
 * <h2>Why node wide and not per segment</h2>
 * A rescore query dispatches one scoring task per leaf and runs them concurrently
 * ({@code NativeEngineKnnVectorQuery} uses {@code getTaskExecutor().invokeAll}), so total in-flight
 * reads are {@code segments x window}, not {@code window}. On the host this was tuned for the device
 * queue holds 63 requests ({@code nr_requests}), and eight segments at a window of 16 would offer 128 —
 * past the point where more in-flight I/O helps, and into the region where it measurably hurts: a
 * 200-wide fan-out measured p90 2.2x and p99 2.0x worse than a 64-wide one. A per-segment pool cannot
 * express that bound, because no segment knows how many others are being scored beside it. This one can,
 * and the bound is the published {@code knn.direct_io.rescore.prefetch_threads}.
 *
 * <h2>Lifecycle</h2>
 * Created on the first read that needs it and never shut down, because there is no node-lifecycle hook
 * at this layer and nothing to shut down when it is empty: threads are created on demand and, with
 * {@code allowCoreThreadTimeOut}, reaped after {@link #KEEP_ALIVE_SECONDS} idle. A node whose queries
 * never engage the rescore seam therefore holds one executor object and zero threads.
 *
 * <p>The idle timeout is safe here for a reason worth stating rather than assuming: reaping a thread can
 * only lose work if the queue can be non-empty with no worker to drain it, and
 * {@link ThreadPoolExecutor} replaces a worker that exits while work is queued.
 *
 * <h2>The queue is bounded, and why the old argument for leaving it unbounded no longer holds</h2>
 * This pool used to be an {@code Executors.newFixedThreadPool}, i.e. an unbounded
 * {@code LinkedBlockingQueue}, on the argument that every task is awaited by the thread that submitted it
 * before that thread can finish its batch, so the queue could not grow past
 * {@code concurrent scorers x window} entries. Two things falsify that:
 *
 * <ul>
 *   <li>{@code DirectIOVectorIndexInput}'s sequential read-ahead ring also submits, and a ring window can
 *       be <em>retired without ever being awaited</em>, so not every task has a waiting submitter.</li>
 *   <li>{@code concurrent scorers x window} is not small. Measured: a 16-thread probe driving the real
 *       stack on 4 pinned cores with the shipped {@code staged_ranges=64} held <b>654-857 tasks
 *       queued</b> with all 64 threads active, sustained. Every staged read in that state waits behind
 *       about thirteen others, on a search thread, with no bound.</li>
 * </ul>
 *
 * So the queue is an {@link ArrayBlockingQueue} of {@link #queueCapacity()}, and a submit past it is
 * refused with the default abort policy. That is not a new failure mode: both submit sites already catch
 * {@link java.util.concurrent.RejectedExecutionException} and fall back — {@code prefetch} counts
 * {@code stageDeclined} and lets the bytes be served by a blocking read, {@code refillReadahead} counts
 * {@code readaheadDeclined} and leaves the scan at queue depth one. Bounding the queue makes those
 * already-written paths reachable rather than adding anything.
 *
 * <p><b>Declining is not the cheaper fallback, and the node measured it.</b> Phase 14 recorded the opposite
 * here as a design claim. On the Oct 4-5 sweep (r7g.xlarge, 4 vCPU, gp3 3000 IOPS, 16 clients, k=100, 5,000
 * queries) the same jar at an effective capacity of 64 declined 46-52% of staged reads and ran
 * 769-777 / 1060-1069 / 1322-1428 ms at 16.7-17.4 qps against <b>783k</b> device reads; at a capacity of
 * 1,024 it declined <b>0%</b> and ran 634-650 / 877-896 / 954-970 at 17.4-19.5 qps against <b>707k</b>. The
 * 76k extra device reads are the declined bursts falling back to individual blocking reads. The claim is
 * withdrawn and the default is derived from burst demand — see {@link #queueCapacity()}.
 *
 * <h2>The thread count is live, and the queue is swapped rather than resized</h2>
 * A changed {@code knn.direct_io.rescore.prefetch_threads} resizes the pool in place
 * ({@link #publishThreads}, {@link #resize}) the moment the cluster applier accepts it. A changed capacity
 * cannot be applied that way — {@link ArrayBlockingQueue} has no resize — so {@link #rebuild} builds a
 * <em>new</em> executor at the new capacity, publishes it so every later submit goes there, and
 * {@code shutdown()}s the old one so reads already queued on it run to completion on its own threads.
 * Nothing waits on the old pool, and no read is lost or run twice. The capacity is derived rather than set
 * (see {@link #queueCapacity()}), so it moves only when the node's own search thread pools are rebuilt,
 * and the swap then happens on the next {@link #executor()} call, which
 * {@code DirectIOVectorIndexInput#ensureStagingTable} makes once per prefetch burst.
 */
@Log4j2
public final class DirectIOReadPool {

    /** How long an idle reader thread is kept before it is reaped. */
    static final long KEEP_ALIVE_SECONDS = 60L;

    /**
     * Floor on the queue. Four whole prefetch bursts, where a burst is {@code PrefetchHelper}'s 64-ordinal
     * Lucene bulk batch: task-15 measured that a table which declines the tail of every burst makes 5.4
     * blocking reads per query and a paced p99 of 30 ms where one that holds the burst makes 0.0 and 11 ms,
     * so one burst is the hard minimum and the floor is above it because a single-processor host still
     * answers several leaves at once. Raised from 64 in phase 15: at 64 the node declined 46-52% of staged
     * reads (see the class javadoc).
     */
    static final int MIN_QUEUE_CAPACITY = 256;

    /** Reader threads are named from this so that a thread dump attributes the read to this seam. */
    static final String THREAD_NAME_PREFIX = "knn-direct-io-rescore-";

    private static final AtomicInteger THREAD_NUMBER = new AtomicInteger();

    private static final ThreadFactory THREAD_FACTORY = runnable -> {
        final Thread thread = new Thread(runnable, THREAD_NAME_PREFIX + THREAD_NUMBER.incrementAndGet());
        // Daemon because this pool is never shut down: a non-daemon reader thread idling out its keep
        // alive would hold up JVM exit for no reason.
        thread.setDaemon(true);
        return thread;
    };

    /**
     * The live pool together with the capacity its queue was built at, because that capacity cannot be read
     * back from an {@link ArrayBlockingQueue} once work is in it ({@code remainingCapacity() + size()} races
     * with the pool's own threads) and it is what {@link #executor()} compares against to decide a rebuild.
     */
    private record Pool(ThreadPoolExecutor executor, int capacity) {
    }

    private static volatile Pool pool;

    /**
     * {@code knn.direct_io.rescore.prefetch_threads}, as the node last published it. Published by
     * {@link KNNSettings} on a change and seeded there at startup; until then it is the setting's own
     * default, which is what a JVM with no cluster service runs on.
     */
    private static volatile int configuredThreads = KNNSettings.KNN_DIRECT_IO_RESCORE_PREFETCH_THREADS_DEFAULT_VALUE;

    private DirectIOReadPool() {}

    /**
     * The shared pool, resized to the published thread count and rebuilt if the derived capacity has moved,
     * or {@code null} when it cannot be provided — in which case the caller must read synchronously rather
     * than fail the query.
     */
    static ExecutorService executor() {
        final int capacity = queueCapacity();
        try {
            final Pool current = pool;
            if (current != null && current.capacity == capacity) {
                resize(current.executor);
                return current.executor;
            }
            return rebuild(configuredThreads, capacity).executor;
        } catch (Exception e) {
            log.warn("Could not provide the Direct I/O rescore read pool; reading without prefetch", e);
            return null;
        }
    }

    /** The thread count the node last published. */
    public static int configuredThreads() {
        return configuredThreads;
    }

    /**
     * The live pool's maximum, or 0 when no burst has built one yet. What {@link #configuredThreads()} has
     * actually been applied to, which is the pair an operator reads to see a change land.
     */
    public static int activeThreads() {
        final Pool current = pool;
        return current == null ? 0 : current.executor.getMaximumPoolSize();
    }

    /**
     * Publishes {@code knn.direct_io.rescore.prefetch_threads} and applies it to the live pool. Called by
     * {@link KNNSettings} from the cluster applier thread: {@link #resize} is a core-and-maximum change on
     * a {@code ThreadPoolExecutor}, which does not wait for a running task, so there is nothing here to
     * hand to another thread. With no pool yet, the next burst builds one at the new count.
     *
     * <p>Not clamped: the setting's own bounds are 2 to 1,024.
     */
    public static void publishThreads(final int threads) {
        configuredThreads = threads;
        final Pool current = pool;
        if (current != null) {
            resize(current.executor);
        }
    }

    /**
     * Builds a pool at {@code capacity} and retires the one it replaces, under this class's lock so that two
     * concurrent {@link #executor()} callers during a capacity change create exactly one pool between them.
     *
     * <p><b>{@code shutdown()}, never {@code shutdownNow()}.</b> The old pool's queue holds reads that a
     * search thread is already waiting on in {@code DirectIOVectorIndexInput#await}; interrupting them would
     * turn a settings change into a burst of bounded-wait timeouts. {@code shutdown()} refuses new submits
     * and runs the queue out on the old pool's own threads, which then exit. Nothing joins it, and nothing
     * needs to: the futures the waiting threads hold are the old pool's and complete normally.
     */
    private static synchronized Pool rebuild(final int threads, final int capacity) {
        final Pool current = pool;
        if (current != null && current.capacity == capacity) {
            // The caller's count, not the published one: the test seam builds at an explicit size, and this
            // method already holds the monitor the published count is read under.
            applyThreads(current.executor, threads);
            return current;
        }
        final ThreadPoolExecutor created = new ThreadPoolExecutor(
            threads,
            threads,
            KEEP_ALIVE_SECONDS,
            TimeUnit.SECONDS,
            new ArrayBlockingQueue<>(capacity),
            THREAD_FACTORY
        );
        created.allowCoreThreadTimeOut(true);
        final Pool replacement = new Pool(created, capacity);
        // Publish before retiring, so there is no window in which a submit can reach a shut-down pool.
        pool = replacement;
        if (current != null) {
            current.executor.shutdown();
            DirectIOVectorIndexInput.STATS.recordPoolRebuild();
        }
        // The operator's only verification that a capacity change took effect, which is why it is emitted on
        // every rebuild and not only at startup: the sweep cron greps for exactly this line.
        log.info("Direct I/O rescore reads will run on a shared pool of up to {} threads with at most {} reads queued", threads, capacity);
        return replacement;
    }

    /**
     * How many reads may wait behind the pool, derived from burst demand. A thread count that cannot be read
     * falls back to the floor, the same contract every other Direct I/O value has: this must never be the
     * reason a read fails.
     *
     * <h4>Derived from burst demand, not from the pool size</h4>
     * It used to be {@code max(64, 8 x pool_threads)}, which was <b>the wrong variable</b>. Pool threads set
     * the drain <em>rate</em>; the queue has to absorb the peak <em>burst</em>, and a burst is
     * {@code staged_ranges} ranges submitted by one scorer before it reads any of them. So the demand is
     * {@code (threads that can be inside a rescore at once) x staged_ranges}, and those threads are the
     * {@code search} pool plus — under concurrent segment search, where
     * {@code NativeEngineKnnVectorQuery} dispatches one task per leaf through
     * {@code getTaskExecutor().invokeAll} — the {@code index_searcher} pool. This method's own earlier
     * Javadoc already stated the demand correctly ("16 scorers x 64 staged ranges want 1,024 slots") while
     * the code computed {@code 8 x 16 = 128}.
     *
     * <p>Rounded up to a power of two and floored at {@link #MIN_QUEUE_CAPACITY}, because excess capacity is
     * free: a submit past capacity is refused immediately (default abort policy) rather than waited on, so a
     * queue that never fills costs only the task references it does not hold. The node measured 0 declines at
     * both 1,024 and 4,096, i.e. the queue filled at neither, and the p99 difference between those two arms
     * is therefore not a queue effect. Buffer bytes are bounded separately by
     * {@link DirectIOVectorIndexInput#DIRECT_MEMORY_BUDGET_BYTES}.
     *
     * <p>On the 4-vCPU node this is {@code (7 + 8) x 64 = 960 -> 1024}, which is the value the sweep measured
     * as best. On a 16-vCPU host it is {@code (25 + 32) x 64 = 3648 -> 4096}.
     */
    static int queueCapacity() {
        try {
            return derivedQueueCapacity(KNNSettings.getDirectIORescoreBurstThreads(), DirectIOVectorIndexInput.DEFAULT_MAX_STAGED_RANGES);
        } catch (Exception e) {
            log.debug("Could not derive the Direct I/O queue capacity from the thread pools; using the floor", e);
            return MIN_QUEUE_CAPACITY;
        }
    }

    /**
     * {@code max(MIN_QUEUE_CAPACITY, nextPowerOfTwo(burstThreads x stagedRanges))}. Pure, so that the
     * formula can be pinned at a processor count this host does not have.
     */
    static int derivedQueueCapacity(final int burstThreads, final int stagedRanges) {
        final long want = (long) Math.max(1, burstThreads) * Math.max(1, stagedRanges);
        if (want >= Integer.MAX_VALUE / 2) {
            return Integer.MAX_VALUE / 2;
        }
        return Math.max(MIN_QUEUE_CAPACITY, Integer.highestOneBit(Math.toIntExact(want) - 1) << 1);
    }

    /** Pool rebuilds since node start — the counter that makes a live capacity change visible. */
    static long poolRebuilds() {
        return DirectIOVectorIndexInput.STATS.poolRebuilds();
    }

    /**
     * Samples {@code executor}'s queue occupancy into the node-wide high-water mark, so that the derived
     * capacity above can be validated against what a real burst actually offers rather than against a
     * formula. Called by both submit sites immediately after a successful submit, which is the only moment
     * occupancy is known to be at least one. Costs one {@link ArrayBlockingQueue} lock acquisition — the same
     * lock the submit just released — against a device read of ~100 us.
     */
    static void sampleQueueDepth(final ExecutorService executor) {
        if (executor instanceof ThreadPoolExecutor threadPool) {
            DirectIOVectorIndexInput.STATS.recordQueuePeak(threadPool.getQueue().size());
        }
    }

    /**
     * Applies the published thread count. Order matters when growing: the maximum has to rise before the
     * core size, because {@code ThreadPoolExecutor} rejects a core size above the maximum.
     *
     * <p><b>The count is read inside the monitor, not passed in.</b> Two threads reach here: the cluster
     * applier, which has just published a new count, and a reading thread at a burst boundary, which may
     * have read the field before that. A count passed in from outside could be the stale one and would put
     * the pool back to the size the operator has just changed — and would log that size as the one that took
     * effect, which is the line the operations harness greps for. Reading the field here makes the last
     * publish win whichever order the two threads arrive in.
     */
    private static void resize(final ThreadPoolExecutor pool) {
        if (pool.getCorePoolSize() == configuredThreads) {
            return;
        }
        synchronized (DirectIOReadPool.class) {
            applyThreads(pool, configuredThreads);
        }
    }

    /**
     * Sets {@code pool}'s core and maximum to {@code threads}, with this class's monitor held. Order matters
     * when growing: the maximum has to rise before the core size, because {@code ThreadPoolExecutor} rejects
     * a core size above the maximum.
     */
    private static void applyThreads(final ThreadPoolExecutor pool, final int threads) {
        if (pool.getCorePoolSize() == threads) {
            return;
        }
        if (threads > pool.getMaximumPoolSize()) {
            pool.setMaximumPoolSize(threads);
            pool.setCorePoolSize(threads);
        } else {
            pool.setCorePoolSize(threads);
            pool.setMaximumPoolSize(threads);
        }
        log.info("Direct I/O rescore read pool resized to up to {} threads", threads);
    }

    /** Test seam: drops the pool so that a test can observe it being created with a different size. */
    static synchronized void resetForTesting() {
        final Pool current = pool;
        pool = null;
        if (current != null) {
            current.executor.shutdown();
        }
    }

    /**
     * Test seam: drives {@link #rebuild} at an explicit capacity, because a unit test has no cluster service
     * to change the setting through and the swap is the behaviour under test rather than the setting read.
     */
    static ThreadPoolExecutor rebuildForTesting(final int threads, final int capacity) {
        return rebuild(threads, capacity).executor;
    }

    /** Test seam: the capacity the live pool's queue was built at, or 0 when there is no pool. */
    static int liveCapacityForTesting() {
        final Pool current = pool;
        return current == null ? 0 : current.capacity;
    }
}

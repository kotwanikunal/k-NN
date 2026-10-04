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
 * and the bound is {@link KNNSettings#getDirectIORescorePrefetchThreads()}.
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
 * So the queue is an {@link ArrayBlockingQueue} of {@link #queueCapacity}, and a submit past it is
 * refused with the default abort policy. That is not a new failure mode: both submit sites already catch
 * {@link java.util.concurrent.RejectedExecutionException} and fall back — {@code prefetch} counts
 * {@code stageDeclined} and lets the bytes be served by a blocking read, {@code refillReadahead} counts
 * {@code readaheadDeclined} and leaves the scan at queue depth one. Bounding the queue makes those
 * already-written paths reachable rather than adding anything. Declining is also the <em>cheaper</em>
 * fallback: a declined prefetch span is replaced by blocking reads of only the blocks a vector actually
 * occupies, where caller-running the submit would read the whole span inline.
 *
 * <p>Resizing is live: a change to the setting is applied on the next {@link #executor()} call, which
 * happens once per scorer, so the pool follows the setting without a restart. Shrinking is honoured the
 * way {@code ThreadPoolExecutor} honours it — threads above the new bound exit as they go idle.
 */
@Log4j2
final class DirectIOReadPool {

    /** How long an idle reader thread is kept before it is reaped. */
    static final long KEEP_ALIVE_SECONDS = 60L;

    /**
     * Reads one reader thread may have waiting behind it. Past about this depth a queued read costs more to
     * wait for than to issue on the calling thread, and more than that to wait for than to decline.
     */
    static final int QUEUED_READS_PER_THREAD = 8;

    /**
     * Floor on the queue, so that one whole prefetch burst from one scorer always fits. This is a
     * compatibility bound rather than a tuning one: {@code PrefetchHelper} is driven by Lucene's
     * 64-ordinal bulk batch, and task-15 measured that a table which declines the tail of every burst
     * makes 5.4 blocking reads per query and a paced p99 of 30 ms where one that holds the burst makes 0.0
     * and 11 ms. A queue that could not hold a burst would reintroduce exactly that.
     */
    static final int MIN_QUEUE_CAPACITY = 64;

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

    private static volatile ThreadPoolExecutor executor;

    private DirectIOReadPool() {}

    /**
     * The shared pool, resized to the current setting, or {@code null} when it cannot be provided — in
     * which case the caller must read synchronously rather than fail the query.
     */
    static ExecutorService executor() {
        final int threads;
        try {
            threads = KNNSettings.getDirectIORescorePrefetchThreads();
        } catch (Exception e) {
            log.debug("Could not read knn.direct_io.rescore.prefetch_threads; reading without prefetch", e);
            return null;
        }
        try {
            final ThreadPoolExecutor current = executor;
            if (current != null) {
                resize(current, threads);
                return current;
            }
            return create(threads);
        } catch (Exception e) {
            log.warn("Could not provide the Direct I/O rescore read pool; reading without prefetch", e);
            return null;
        }
    }

    private static synchronized ThreadPoolExecutor create(final int threads) {
        if (executor != null) {
            resize(executor, threads);
            return executor;
        }
        final int capacity = queueCapacity(threads);
        final ThreadPoolExecutor created = new ThreadPoolExecutor(
            threads,
            threads,
            KEEP_ALIVE_SECONDS,
            TimeUnit.SECONDS,
            new ArrayBlockingQueue<>(capacity),
            THREAD_FACTORY
        );
        created.allowCoreThreadTimeOut(true);
        log.info("Direct I/O rescore reads will run on a shared pool of up to {} threads with at most {} reads queued", threads, capacity);
        executor = created;
        return created;
    }

    /**
     * How many reads may wait behind the pool: {@code knn.direct_io.rescore.prefetch.queue_capacity} when
     * an operator has set it, and otherwise {@code max(64, 8 x threads)} derived from the size the pool is
     * created at. Not resized when either setting changes, because {@link ArrayBlockingQueue} has no resize
     * and because the safe direction is the one that happens anyway: growing the pool leaves the queue
     * relatively shallower.
     *
     * <p><b>The default is unchanged by design.</b> Phase 13's audit row 7 and phase 14's S3 both measured
     * this as the binding constraint at concurrent-scorer load — 44%, 43.8% and 43.9% of declared reads
     * refused at 16 scorers, because 16 scorers x 64 staged ranges want 1,024 slots and the derived
     * capacity is 128 — but the right value is a trade of queue depth against wait on a specific device and
     * core count, which this desktop cannot settle for the node. So the knob exists and the shipped
     * behaviour does not move. A setting that cannot be read falls back to the derived value, the same
     * contract every other Direct I/O setting has.
     */
    static int queueCapacity(final int threads) {
        final int configured;
        try {
            configured = KNNSettings.getDirectIORescorePrefetchQueueCapacity();
        } catch (Exception e) {
            log.debug("Could not read knn.direct_io.rescore.prefetch.queue_capacity; deriving the capacity", e);
            return Math.max(MIN_QUEUE_CAPACITY, QUEUED_READS_PER_THREAD * threads);
        }
        if (configured > 0) {
            return configured;
        }
        return Math.max(MIN_QUEUE_CAPACITY, QUEUED_READS_PER_THREAD * threads);
    }

    /**
     * Applies a changed thread count. Order matters when growing: the maximum has to rise before the core
     * size, because {@code ThreadPoolExecutor} rejects a core size above the maximum.
     */
    private static void resize(final ThreadPoolExecutor pool, final int threads) {
        if (pool.getCorePoolSize() == threads) {
            return;
        }
        synchronized (DirectIOReadPool.class) {
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
    }

    /** Test seam: drops the pool so that a test can observe it being created with a different size. */
    static synchronized void resetForTesting() {
        final ThreadPoolExecutor current = executor;
        executor = null;
        if (current != null) {
            current.shutdown();
        }
    }
}

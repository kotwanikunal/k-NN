/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import lombok.extern.log4j.Log4j2;
import org.opensearch.knn.index.KNNSettings;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
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
 * {@link ThreadPoolExecutor} replaces a worker that exits while work is queued. Every task is also
 * awaited by the thread that submitted it before that thread can finish its batch, so an unbounded queue
 * cannot grow past {@code concurrent scorers x window} entries.
 *
 * <p>Resizing is live: a change to the setting is applied on the next {@link #executor()} call, which
 * happens once per scorer, so the pool follows the setting without a restart. Shrinking is honoured the
 * way {@code ThreadPoolExecutor} honours it — threads above the new bound exit as they go idle.
 */
@Log4j2
final class DirectIOReadPool {

    /** How long an idle reader thread is kept before it is reaped. */
    static final long KEEP_ALIVE_SECONDS = 60L;

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
        final ThreadPoolExecutor created = (ThreadPoolExecutor) Executors.newFixedThreadPool(threads, THREAD_FACTORY);
        created.setKeepAliveTime(KEEP_ALIVE_SECONDS, TimeUnit.SECONDS);
        created.allowCoreThreadTimeOut(true);
        log.info("Direct I/O rescore reads will run on a shared pool of up to {} threads", threads);
        executor = created;
        return created;
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

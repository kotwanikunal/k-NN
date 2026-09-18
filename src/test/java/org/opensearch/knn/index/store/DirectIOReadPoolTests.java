/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import com.carrotsearch.randomizedtesting.ThreadFilter;
import com.carrotsearch.randomizedtesting.annotations.ThreadLeakFilters;
import lombok.SneakyThrows;
import org.opensearch.knn.KNNTestCase;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

/**
 * The node-wide read pool. What matters about it is the properties a wrong pool would break silently: that
 * it is shared rather than per caller (a per-segment pool cannot bound in-flight I/O across segments), that
 * its threads are daemons (it is never shut down), and that submitted reads actually run.
 */
@ThreadLeakFilters(defaultFilters = true, filters = { DirectIOReadPoolTests.ReadPoolThreadFilter.class })
public class DirectIOReadPoolTests extends KNNTestCase {

    /**
     * The pool outlives any one query, so on a node it is created once and never shut down and its threads
     * are daemons that idle out. A test suite that used it therefore ends with those threads still parked,
     * which is the intended lifecycle rather than a leak.
     */
    public static final class ReadPoolThreadFilter implements ThreadFilter {
        @Override
        public boolean reject(final Thread thread) {
            return thread.getName().startsWith(DirectIOReadPool.THREAD_NAME_PREFIX);
        }
    }

    @Override
    public void tearDown() throws Exception {
        DirectIOReadPool.resetForTesting();
        super.tearDown();
    }

    public void testTheSamePoolIsHandedToEveryCaller() {
        final ExecutorService first = DirectIOReadPool.executor();
        assertNotNull(first);
        assertSame("a per-caller pool cannot bound in-flight reads across segments", first, DirectIOReadPool.executor());
    }

    @SneakyThrows
    public void testSubmittedWorkRuns() {
        final ExecutorService pool = DirectIOReadPool.executor();
        final List<Future<Integer>> results = new ArrayList<>();
        for (int i = 0; i < 32; i++) {
            final int value = i;
            results.add(pool.submit(() -> value * 2));
        }
        for (int i = 0; i < results.size(); i++) {
            assertEquals(Integer.valueOf(i * 2), results.get(i).get(10, TimeUnit.SECONDS));
        }
    }

    /**
     * The pool is created lazily and never shut down, so a non-daemon thread in it would hold up JVM exit
     * for its whole keep-alive after the last read.
     */
    @SneakyThrows
    public void testReaderThreadsAreDaemonsAndNamed() {
        final ExecutorService pool = DirectIOReadPool.executor();
        final CountDownLatch seen = new CountDownLatch(1);
        final Thread[] worker = new Thread[1];
        pool.submit(() -> {
            worker[0] = Thread.currentThread();
            seen.countDown();
        });
        assertTrue(seen.await(10, TimeUnit.SECONDS));
        assertTrue("reader threads must be daemons", worker[0].isDaemon());
        assertTrue("got thread name " + worker[0].getName(), worker[0].getName().startsWith(DirectIOReadPool.THREAD_NAME_PREFIX));
    }

    /**
     * The bound is a dynamic setting, so it has to be applied to the live pool rather than only at creation.
     * With no cluster service in a unit test the setting reads its default, so what is pinned here is that
     * the pool is sized from the setting and that re-reading it does not resize or replace the pool.
     */
    public void testThePoolIsSizedFromTheSetting() {
        final ThreadPoolExecutor pool = (ThreadPoolExecutor) DirectIOReadPool.executor();
        assertNotNull(pool);
        final int expected = org.opensearch.knn.index.KNNSettings.getDirectIORescorePrefetchThreads();
        assertEquals(expected, pool.getCorePoolSize());
        assertEquals(expected, pool.getMaximumPoolSize());
        assertTrue("idle reader threads must be reclaimable", pool.allowsCoreThreadTimeOut());
        assertEquals(DirectIOReadPool.KEEP_ALIVE_SECONDS, pool.getKeepAliveTime(TimeUnit.SECONDS));

        assertSame(pool, DirectIOReadPool.executor());
        assertEquals(expected, pool.getCorePoolSize());
    }
}

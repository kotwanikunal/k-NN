/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.memory;

import org.junit.After;
import org.junit.Before;
import org.opensearch.action.admin.cluster.settings.ClusterUpdateSettingsRequest;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.settings.ClusterSettings;
import org.opensearch.common.settings.Setting;
import org.opensearch.common.settings.Settings;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.knn.plugin.KNNPlugin;
import org.opensearch.plugins.Plugin;
import org.opensearch.test.OpenSearchSingleNodeTestCase;
import org.opensearch.threadpool.ThreadPool;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.ExecutionException;
import java.util.stream.Collectors;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * I3's tests: the vector cache's slabs are granted and accounted by {@link NativeMemoryCacheManager} under
 * the k-NN native budget, and the four rules the design note asks of the manager hold.
 *
 * <p>The setup mirrors {@link NativeMemoryCacheManagerTests}, which is the suite whose behaviour must not
 * change, so that these tests exercise the same manager under the same settings.
 */
public class VectorCacheSlabAccountingTests extends OpenSearchSingleNodeTestCase {

    private static final long SLAB_KB = VectorCacheSlab.SLAB_BYTES / 1024L;

    private ThreadPool threadPool;
    private ClusterService clusterService;
    private Set<Setting<?>> registeredSettings;
    private NativeMemoryCacheManager manager;

    @Before
    public void setUp() throws Exception {
        super.setUp();
        clusterService = mock(ClusterService.class);
        registeredSettings = new HashSet<>(ClusterSettings.BUILT_IN_CLUSTER_SETTINGS);
        registeredSettings.addAll(
            KNNSettings.state()
                .getSettings()
                .stream()
                .filter(s -> s.getProperties().contains(Setting.Property.NodeScope))
                .collect(Collectors.toList())
        );
        KNNSettings.state().setClusterService(clusterService);
        // A pinned limit rather than the default percentage of the host's memory, because the default on
        // this desktop is tens of gigabytes and these tests fill the budget with real slabs. Four slabs is
        // the smallest limit the scenarios below need.
        setConfiguredCacheSize("0b");
        threadPool = new ThreadPool(Settings.builder().put("node.name", "VectorCacheSlabAccountingTests").build());
        NativeMemoryCacheManager.setThreadPool(threadPool);
        manager = new NativeMemoryCacheManager();
        assertEquals("the pinned limit did not reach the manager", 4 * SLAB_KB, manager.getMaxCacheSizeInKilobytes());
    }

    /**
     * Sets {@code knn.direct_io.rescore.cache.max_bytes}, and pins the circuit breaker limit at four slabs,
     * as the manager reads them: through {@code ClusterService#getClusterSettings}. The cache setting
     * defaults to {@code 0b}, so a test that wants the manager to give the cache a target has to say so.
     */
    private void setConfiguredCacheSize(final String size) {
        final Settings settings = Settings.builder()
            .put(KNNSettings.KNN_DIRECT_IO_RESCORE_CACHE_MAX_BYTES, size)
            .put(KNNSettings.KNN_MEMORY_CIRCUIT_BREAKER_CLUSTER_LIMIT, (4 * SLAB_KB) + "kb")
            .build();
        when(clusterService.getClusterSettings()).thenReturn(new ClusterSettings(settings, registeredSettings));
        assertEquals(
            "the test's own setting plumbing is broken",
            settings.getAsBytesSize(KNNSettings.KNN_DIRECT_IO_RESCORE_CACHE_MAX_BYTES, null).getBytes(),
            KNNSettings.getDirectIORescoreCacheMaxBytes().getBytes()
        );
    }

    @After
    public void shutdown() throws Exception {
        if (manager != null) {
            manager.close();
            manager = null;
        }
        terminate(threadPool);
        tearDown();
    }

    @Override
    public void tearDown() throws Exception {
        final ClusterUpdateSettingsRequest request = new ClusterUpdateSettingsRequest();
        request.persistentSettings(Settings.builder().putNull(KNNSettings.KNN_CIRCUIT_BREAKER_TRIGGERED).build());
        client().admin().cluster().updateSettings(request).get();
        NativeMemoryCacheManager.getInstance().close();
        super.tearDown();
    }

    @Override
    protected Collection<Class<? extends Plugin>> getPlugins() {
        return Collections.singletonList(KNNPlugin.class);
    }

    // ---------------------------------------------------------------------------------------------------
    // Accounting after grant, release and reclaim
    // ---------------------------------------------------------------------------------------------------

    public void testSlabsAreAccountedOnGrantAndGivenBackOnRelease() {
        final SlabVectorCache cache = manager.getVectorCache();
        assertEquals(0, manager.getVectorCacheSlabCount());
        assertEquals(0L, manager.getVectorCacheSlabSizeInKilobytes());

        cache.configure(3 * VectorCacheSlab.SLAB_BYTES);
        fillSlabs(cache, 3);

        assertEquals(3, manager.getVectorCacheSlabCount());
        assertEquals(3 * SLAB_KB, manager.getVectorCacheSlabSizeInKilobytes());
        assertEquals(3 * SLAB_KB, manager.getTotalNativeMemoryInKilobytes() - manager.getCacheSizeInKilobytes());
        assertEquals(3L * VectorCacheSlab.SLAB_BYTES, cache.slabBytes());

        cache.configure(VectorCacheSlab.SLAB_BYTES);
        assertEquals(1, manager.getVectorCacheSlabCount());
        assertEquals(SLAB_KB, manager.getVectorCacheSlabSizeInKilobytes());

        cache.configure(0);
        assertEquals(0, manager.getVectorCacheSlabCount());
        assertEquals(0L, manager.getVectorCacheSlabSizeInKilobytes());
        assertEquals(manager.getCacheSizeInKilobytes(), manager.getTotalNativeMemoryInKilobytes());
    }

    /** Rule three: the running counter has to equal the sum it replaced, at every step. */
    public void testTheRunningCounterEqualsTheOldSum() throws ExecutionException {
        assertEquals(manager.getCacheSizeInKilobytesByScan(), manager.getCacheSizeInKilobytes());

        manager.get(new SizedEntryContext("graph-1", 10), true);
        assertEquals(10, manager.getCacheSizeInKilobytes());
        assertEquals(manager.getCacheSizeInKilobytesByScan(), manager.getCacheSizeInKilobytes());

        manager.get(new SizedEntryContext("graph-2", 25), true);
        assertEquals(35, manager.getCacheSizeInKilobytes());
        assertEquals(manager.getCacheSizeInKilobytesByScan(), manager.getCacheSizeInKilobytes());

        // A second get of a key already loaded must not count twice.
        manager.get(new SizedEntryContext("graph-2", 25), true);
        assertEquals(35, manager.getCacheSizeInKilobytes());
        assertEquals(manager.getCacheSizeInKilobytesByScan(), manager.getCacheSizeInKilobytes());

        manager.invalidate("graph-1");
        assertEquals(25, manager.getCacheSizeInKilobytes());
        assertEquals(manager.getCacheSizeInKilobytesByScan(), manager.getCacheSizeInKilobytes());

        // Slabs are not part of this figure, which is what keeps the circuit breaker's unset test honest.
        manager.getVectorCache().configure(VectorCacheSlab.SLAB_BYTES);
        fillSlabs(manager.getVectorCache(), 1);
        assertEquals(25, manager.getCacheSizeInKilobytes());
        assertEquals(manager.getCacheSizeInKilobytesByScan(), manager.getCacheSizeInKilobytes());
        assertEquals(25 + SLAB_KB, manager.getTotalNativeMemoryInKilobytes());

        manager.invalidateAll();
        assertEquals(0, manager.getCacheSizeInKilobytes());
        assertEquals(manager.getCacheSizeInKilobytesByScan(), manager.getCacheSizeInKilobytes());
    }

    // ---------------------------------------------------------------------------------------------------
    // Rule one: slab churn never trips the breaker
    // ---------------------------------------------------------------------------------------------------

    public void testSlabChurnAtTheLimitDoesNotTripTheBreaker() {
        final SlabVectorCache cache = manager.getVectorCache();
        // One slab and two slabs' worth of vectors per round, so that every round both reclaims the slab
        // under eviction pressure and releases it on the resize to zero.
        cache.configure(VectorCacheSlab.SLAB_BYTES);
        for (int round = 0; round < 10; round++) {
            fillSlabs(cache, 2);
            cache.configure(0);
            cache.configure(VectorCacheSlab.SLAB_BYTES);
        }
        assertTrue("the churn should have reclaimed", cache.reclaimedSlabs() > 0);
        assertFalse("a slab release must not set capacity reached", manager.isCacheCapacityReached());
        assertFalse(
            "a slab release must not trip the breaker",
            Boolean.TRUE.equals(KNNSettings.state().getSettingValue(KNNSettings.KNN_CIRCUIT_BREAKER_TRIGGERED))
        );
    }

    // ---------------------------------------------------------------------------------------------------
    // Rule two: a graph takes its room from the slabs, not from another graph
    // ---------------------------------------------------------------------------------------------------

    public void testAGraphLoadReleasesSlabsAndEvictsNoGraph() throws ExecutionException {
        final SlabVectorCache cache = manager.getVectorCache();
        final long limit = manager.getMaxCacheSizeInKilobytes();
        final int affordable = (int) (limit / SLAB_KB);
        assertTrue("this test needs a limit of at least four slabs, got " + limit + " KB", affordable >= 4);
        setConfiguredCacheSize(limit + "kb");

        // One graph of a slab's worth that must survive, and then slabs until the manager refuses.
        manager.get(new SizedEntryContext("graph-keep", (int) SLAB_KB), true);
        cache.configure(limit * 1024);
        fillSlabs(cache, affordable);

        final int slabsBefore = manager.getVectorCacheSlabCount();
        final long totalBefore = manager.getTotalNativeMemoryInKilobytes();
        assertTrue("the cache should have taken slabs, it took " + slabsBefore, slabsBefore > 0);
        assertTrue("the grant must not overshoot the limit, " + totalBefore + " of " + limit, totalBefore <= limit);
        assertTrue("and must have filled it, " + totalBefore + " of " + limit, totalBefore > limit - 2 * SLAB_KB);

        // A second graph of two slabs' worth, against under two slabs of headroom: it only fits if slabs go
        // first.
        manager.get(new SizedEntryContext("graph-new", (int) (2 * SLAB_KB)), true);

        assertEquals("both graphs must still be loaded", 3 * SLAB_KB, manager.getCacheSizeInKilobytes());
        assertTrue("slabs should have been released", manager.getVectorCacheSlabCount() < slabsBefore);
        assertTrue(
            "the total must be back inside the limit, got " + manager.getTotalNativeMemoryInKilobytes(),
            manager.getTotalNativeMemoryInKilobytes() <= limit
        );
        assertFalse("no graph was evicted, so nothing may be flagged", manager.isCacheCapacityReached());
    }

    /** Rule four: the target is the limit minus the non-slab allocations, and it is applied by resizing. */
    public void testTheTargetIsTheBudgetLeftOverAndResizingKeepsWhatFits() throws ExecutionException {
        final SlabVectorCache cache = manager.getVectorCache();
        final long limit = manager.getMaxCacheSizeInKilobytes();
        final int affordable = (int) (limit / SLAB_KB);
        assertTrue("this test needs a limit of at least four slabs, got " + limit + " KB", affordable >= 4);

        // Nothing else in the cache, so the configured size is the one that wins.
        setConfiguredCacheSize((2 * SLAB_KB) + "kb");
        manager.updateVectorCacheTarget();
        assertEquals(2 * SLAB_KB * 1024, cache.maxBytes());
        fillSlabs(cache, 2);
        assertEquals(2, manager.getVectorCacheSlabCount());
        final long entriesBefore = cache.entries();
        assertTrue(entriesBefore > 0);

        // A graph that leaves less room than the configured size forces the other term to win.
        manager.get(new SizedEntryContext("graph-big", (int) ((affordable - 1) * SLAB_KB)), true);
        final long leftoverKb = limit - (affordable - 1) * SLAB_KB;
        assertEquals("the target is the leftover budget", leftoverKb / SLAB_KB * SLAB_KB * 1024, cache.maxBytes());
        assertEquals((int) (leftoverKb / SLAB_KB), manager.getVectorCacheSlabCount());
        assertTrue("resizing down keeps the newest slab's vectors", cache.entries() > 0);
        assertTrue("and drops the older slab's", cache.entries() < entriesBefore);
        cache.checkInvariantsForTesting();

        // Freeing the graph raises the leftover budget again, and the target follows it back up.
        manager.invalidate("graph-big");
        manager.updateVectorCacheTarget();
        assertEquals(2 * SLAB_KB * 1024, cache.maxBytes());
        fillSlabs(cache, 2);
        assertEquals(2, manager.getVectorCacheSlabCount());
    }

    /** The grant has to refuse rather than overshoot when the budget moved under the cache's target. */
    public void testAGrantIsRefusedWhenTheBudgetIsGone() throws ExecutionException {
        final SlabVectorCache cache = manager.getVectorCache();
        final long limit = manager.getMaxCacheSizeInKilobytes();
        // A graph that leaves under one slab of room, with the cache's target left high on purpose.
        manager.get(new SizedEntryContext("graph-all", (int) (limit - SLAB_KB / 2)), true);
        cache.configure(8L * VectorCacheSlab.SLAB_BYTES);
        fillSlabs(cache, 8);
        assertEquals("no slab should have been granted", 0, manager.getVectorCacheSlabCount());
        assertEquals(0L, cache.entries());
        assertEquals(0L, manager.getVectorCacheSlabSizeInKilobytes());
    }

    // ---------------------------------------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------------------------------------

    /** Fills enough 768-dim vectors to take {@code slabs} slabs, if the manager will grant them. */
    private static void fillSlabs(final SlabVectorCache cache, final int slabs) {
        final int dimension = 768;
        final int bytes = dimension * Float.BYTES;
        final int perSlab = VectorCacheSlab.SLAB_BYTES / bytes;
        final Object core = new Object();
        final ByteBuffer staging = ByteBuffer.allocate(bytes).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < dimension; i++) {
            staging.putFloat(i * Float.BYTES, i);
        }
        for (int ordinal = 0; ordinal < slabs * perSlab; ordinal++) {
            cache.put(core, 0, ordinal, staging, 0, bytes);
        }
    }

    /** The smallest allocation that can stand in for a graph: a size and nothing else. */
    private static final class SizedAllocation implements NativeMemoryAllocation {
        private final int sizeKb;

        private SizedAllocation(final int sizeKb) {
            this.sizeKb = sizeKb;
        }

        @Override
        public void close() {}

        @Override
        public boolean isClosed() {
            return false;
        }

        @Override
        public long getMemoryAddress() {
            return 0;
        }

        @Override
        public void readLock() {}

        @Override
        public void writeLock() {}

        @Override
        public void readUnlock() {}

        @Override
        public void writeUnlock() {}

        @Override
        public int getSizeInKB() {
            return sizeKb;
        }
    }

    private static final class SizedEntryContext extends NativeMemoryEntryContext<SizedAllocation> {
        private final int sizeKb;

        private SizedEntryContext(final String key, final int sizeKb) {
            super(key);
            this.sizeKb = sizeKb;
        }

        @Override
        public Integer calculateSizeInKB() {
            return sizeKb;
        }

        @Override
        public void open() {}

        @Override
        public SizedAllocation load() throws IOException {
            return new SizedAllocation(sizeKb);
        }
    }
}

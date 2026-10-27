/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import com.carrotsearch.randomizedtesting.ThreadFilter;
import com.carrotsearch.randomizedtesting.annotations.ThreadLeakFilters;
import lombok.SneakyThrows;
import org.apache.logging.log4j.Level;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.core.Appender;
import org.apache.logging.log4j.core.LogEvent;
import org.apache.logging.log4j.core.Logger;
import org.apache.logging.log4j.core.appender.AbstractAppender;
import org.apache.logging.log4j.core.config.Configurator;
import org.opensearch.common.settings.ClusterSettings;
import org.opensearch.common.settings.Setting;
import org.opensearch.common.settings.Settings;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.KNNSettings;
import org.opensearch.transport.client.Client;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.OpenOption;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * W3: every dynamic Direct I/O setting reaches its component through a cluster settings update consumer,
 * and the read path reads a field rather than a settings map.
 *
 * <p><b>Why the real applier and not the publish methods directly.</b> These cases drive
 * {@link ClusterSettings#applySettings}, which is what an operator's {@code _cluster/settings} PUT ends in,
 * so they prove the registration as well as the effect. A test that called {@code publishX} itself would
 * pass with no consumer registered at all, which is the bug this phase is for: before W3 nothing was
 * registered and a {@code knn.direct_io.rescore.enabled: false} left 4.5 GiB of slabs held on the node
 * (ref/kb/CHALLENGES.md 8 and 19).
 *
 * <p>The bound each case asserts is stated in it. All five consumers publish and signal only, so for four
 * of them the bound is "on the consumer's return"; the cache's shrink is the one that hands work to another
 * thread, and {@code consumerReturnsBeforeALargeReleaseFinishes} is the case that holds it to that.
 */
@ThreadLeakFilters(defaultFilters = true, filters = { DirectIOSettingsConsumerTests.ReadPoolThreadFilter.class })
public class DirectIOSettingsConsumerTests extends KNNTestCase {

    /** The read pool is never shut down by design; its daemon threads outlive any one suite. */
    public static final class ReadPoolThreadFilter implements ThreadFilter {
        @Override
        public boolean reject(final Thread thread) {
            return thread.getName().startsWith(DirectIOReadPool.THREAD_NAME_PREFIX);
        }
    }

    /** 64 KiB: small enough that a few hundred slabs is a cheap release to watch. */
    private static final int TEST_SLAB_BYTES = 64 * 1024;
    private static final long HEADER_LENGTH = 44L;
    private static final int DIMENSION = 32;
    private static final int VECTOR_BYTES = DIMENSION * Float.BYTES;
    private static final int VECTOR_COUNT = 64;

    /** The live applier these cases change settings through, rebuilt per case by {@link #applierOver}. */
    private ClusterSettings clusterSettings;

    @Override
    public void tearDown() throws Exception {
        DirectIOVectorCache.resetForTesting();
        DirectIOReadPool.resetForTesting();
        // Every published value goes back to its default in KNNTestCase#initKNNSettings, which re-seeds from
        // Settings.EMPTY, so there is nothing more to undo here.
        super.tearDown();
    }

    /**
     * A real {@link ClusterSettings} over {@code persisted}, installed on the plugin and initialized, which
     * is what registers the consumers and seeds the published values once.
     */
    private void applierOver(final Settings persisted) {
        final Set<Setting<?>> registered = new HashSet<>(ClusterSettings.BUILT_IN_CLUSTER_SETTINGS);
        registered.addAll(
            KNNSettings.state().getSettings().stream().filter(s -> s.getProperties().contains(Setting.Property.NodeScope)).toList()
        );
        clusterSettings = new ClusterSettings(persisted, registered);
        when(clusterService.getClusterSettings()).thenReturn(clusterSettings);
        KNNSettings.state().initialize(mock(Client.class), clusterService);
    }

    /** An operator's PUT: the applier accepts it and runs the consumers on the calling thread. */
    private void apply(final Settings settings) {
        clusterSettings.applySettings(settings);
    }

    // ---------------------------------------------------------------------------------------------------
    // The five consumers
    // ---------------------------------------------------------------------------------------------------

    /**
     * {@code rescore.enabled} reaches the directory's dispatch rule. The bound is the consumer's return:
     * {@link ClusterSettings#applySettings} runs it on the calling thread, so the new answer is in place
     * when {@code apply} returns and no reopen and no query is needed to collect it.
     */
    public void testRescoreEnabledReachesTheDirectoryWithinOneApply() {
        applierOver(Settings.EMPTY);
        assertFalse("the shipped default is off", KNNDirectIODirectory.isRescoreEnabled());

        apply(Settings.builder().put(KNNSettings.KNN_DIRECT_IO_RESCORE_ENABLED, true).build());
        assertTrue("on, as of the consumer's return", KNNDirectIODirectory.isRescoreEnabled());

        apply(Settings.builder().put(KNNSettings.KNN_DIRECT_IO_RESCORE_ENABLED, false).build());
        assertFalse("and off again", KNNDirectIODirectory.isRescoreEnabled());
    }

    /** {@code min_file_size} reaches the directory's size floor, same shape and same bound. */
    public void testMinFileSizeReachesTheDirectory() {
        applierOver(Settings.EMPTY);
        assertEquals(KNNSettings.KNN_DIRECT_IO_MIN_FILE_SIZE_DEFAULT_VALUE.getBytes(), KNNDirectIODirectory.minFileSizeBytes());

        apply(Settings.builder().put(KNNSettings.KNN_DIRECT_IO_MIN_FILE_SIZE, "0b").build());
        assertEquals(0L, KNNDirectIODirectory.minFileSizeBytes());

        apply(Settings.builder().put(KNNSettings.KNN_DIRECT_IO_MIN_FILE_SIZE, "4mb").build());
        assertEquals(4L * 1024 * 1024, KNNDirectIODirectory.minFileSizeBytes());
    }

    /**
     * {@code prefetch_threads} resizes the live pool in place: the maximum is the new value on the
     * consumer's return, and {@code pool_rebuilds} has not moved, because the queue capacity is derived
     * from the node's own search pools and a thread-count change does not touch it. In place matters
     * because a rebuild costs the reads already queued on the old executor their place in line.
     */
    public void testPrefetchThreadsResizesThePoolInPlace() {
        applierOver(Settings.EMPTY);
        DirectIOReadPool.resetForTesting();
        final ThreadPoolExecutor pool = DirectIOReadPool.rebuildForTesting(4, DirectIOReadPool.MIN_QUEUE_CAPACITY);
        final long rebuildsBefore = DirectIOReadPool.poolRebuilds();
        final int capacityBefore = DirectIOReadPool.liveCapacityForTesting();

        apply(Settings.builder().put(KNNSettings.KNN_DIRECT_IO_RESCORE_PREFETCH_THREADS, 9).build());

        assertEquals("the published count", 9, DirectIOReadPool.configuredThreads());
        assertEquals("the live pool's maximum, on the consumer's return", 9, pool.getMaximumPoolSize());
        assertEquals("and its core size with it", 9, pool.getCorePoolSize());
        assertEquals("the same executor, not a replacement", 9, DirectIOReadPool.activeThreads());
        assertEquals("so no rebuild was needed", rebuildsBefore, DirectIOReadPool.poolRebuilds());
        assertEquals("and the queue was not replaced", capacityBefore, DirectIOReadPool.liveCapacityForTesting());

        apply(Settings.builder().put(KNNSettings.KNN_DIRECT_IO_RESCORE_PREFETCH_THREADS, 2).build());
        assertEquals("shrinking is in place too", 2, pool.getMaximumPoolSize());
        assertEquals(rebuildsBefore, DirectIOReadPool.poolRebuilds());
    }

    /**
     * The {@code "resized to up to N threads"} line is the campaign harness's only verification that a live
     * thread change took effect on a node ({@code ref/campaign/node/node.py} greps the node log for it), so
     * its text is pinned here rather than left to drift with an edit to the message.
     */
    public void testTheResizeLogLineKeepsTheTextTheHarnessGrepsFor() {
        applierOver(Settings.EMPTY);
        DirectIOReadPool.resetForTesting();
        DirectIOReadPool.rebuildForTesting(4, DirectIOReadPool.MIN_QUEUE_CAPACITY);

        final List<String> lines = Collections.synchronizedList(new ArrayList<>());
        final Logger logger = (Logger) LogManager.getLogger(DirectIOReadPool.class);
        final Level restore = logger.getLevel();
        final Appender capture = new AbstractAppender("w3-resize-capture", null, null, true, null) {
            @Override
            public void append(final LogEvent event) {
                lines.add(event.getMessage().getFormattedMessage());
            }
        };
        capture.start();
        logger.addAppender(capture);
        Configurator.setLevel(DirectIOReadPool.class.getName(), Level.INFO);
        try {
            apply(Settings.builder().put(KNNSettings.KNN_DIRECT_IO_RESCORE_PREFETCH_THREADS, 11).build());
        } finally {
            Configurator.setLevel(DirectIOReadPool.class.getName(), restore);
            logger.removeAppender(capture);
            capture.stop();
        }
        assertTrue(
            "the harness greps for 'read pool resized to up to', saw " + lines,
            lines.stream().anyMatch(line -> line.contains("Direct I/O rescore read pool resized to up to 11 threads"))
        );
    }

    /**
     * {@code cache.max_bytes} grows and shrinks the store. Growing keeps what is held; shrinking to zero
     * empties it. Both are observable on the consumer's return: {@code configure} detaches the slabs under
     * the broker's lock before it hands them to the release thread, so the index is already empty even
     * though the memory has not gone back yet.
     */
    public void testCacheMaxBytesGrowsAndShrinks() {
        applierOver(Settings.EMPTY);
        DirectIOVectorCache.resetForTesting();
        DirectIOVectorCache.setSlabBytesForTesting(TEST_SLAB_BYTES);
        assertEquals("the shipped default is off", 0L, DirectIOVectorCache.configuredMaxBytes());
        assertEquals(0L, DirectIOVectorCache.maxBytes());

        apply(Settings.builder().put(KNNSettings.KNN_DIRECT_IO_RESCORE_CACHE_MAX_BYTES, (4 * TEST_SLAB_BYTES) + "b").build());
        assertEquals(4L * TEST_SLAB_BYTES, DirectIOVectorCache.configuredMaxBytes());
        assertEquals("active equals configured after a grow", 4L * TEST_SLAB_BYTES, DirectIOVectorCache.maxBytes());

        apply(Settings.builder().put(KNNSettings.KNN_DIRECT_IO_RESCORE_CACHE_MAX_BYTES, (2 * TEST_SLAB_BYTES) + "b").build());
        assertEquals(2L * TEST_SLAB_BYTES, DirectIOVectorCache.configuredMaxBytes());
        assertEquals("and after a shrink", 2L * TEST_SLAB_BYTES, DirectIOVectorCache.maxBytes());

        apply(Settings.builder().put(KNNSettings.KNN_DIRECT_IO_RESCORE_CACHE_MAX_BYTES, "0b").build());
        assertEquals(0L, DirectIOVectorCache.maxBytes());
        assertEquals("and the index is empty on the consumer's return", 0L, DirectIOVectorCache.entries());
        assertEquals(0L, DirectIOVectorCache.slabs());
    }

    /**
     * The directive's W3 change 2, as a bound: a shrink from a large cache is hundreds of 32 MiB frees at
     * about 2 ms each, and the cluster applier thread must not wait for them. The release backend here
     * blocks until this test lets it go, so the consumer can only return if the frees are somebody else's
     * work. 100 ms is the stated bound; the frees then finish afterwards.
     */
    public void testConsumerReturnsBeforeALargeReleaseFinishes() {
        applierOver(Settings.EMPTY);
        DirectIOVectorCache.resetForTesting();
        DirectIOVectorCache.setSlabBytesForTesting(TEST_SLAB_BYTES);

        final CountDownLatch held = new CountDownLatch(1);
        final AtomicInteger released = new AtomicInteger();
        DirectIOVectorCache.setSlabMemoryForTesting(new DirectIOVectorCache.SlabMemory() {
            @Override
            public DirectIOSlab allocate(final int bytes) {
                return DirectIOSlab.allocate(bytes);
            }

            @Override
            public void release(final DirectIOSlab slab) {
                try {
                    assertTrue("the release must not be the applier's work", held.await(30, TimeUnit.SECONDS));
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                }
                released.incrementAndGet();
                slab.close();
            }
        });

        final int slabs = 64;
        apply(Settings.builder().put(KNNSettings.KNN_DIRECT_IO_RESCORE_CACHE_MAX_BYTES, ((long) slabs * TEST_SLAB_BYTES) + "b").build());
        fillEverySlab(slabs);
        assertEquals("the store must be full, so the shrink has something to free", slabs, DirectIOVectorCache.slabs());

        final long started = System.nanoTime();
        apply(Settings.builder().put(KNNSettings.KNN_DIRECT_IO_RESCORE_CACHE_MAX_BYTES, "0b").build());
        final long elapsedMillis = (System.nanoTime() - started) / 1_000_000;

        assertTrue("the consumer returned in " + elapsedMillis + " ms, bound 100 ms", elapsedMillis < 100);
        assertEquals("with the index already empty", 0L, DirectIOVectorCache.entries());
        assertEquals("and no slab freed yet, because the backend is still held", 0, released.get());
        held.countDown();
        DirectIOVectorCache.awaitReleasesForTesting();
        assertEquals("the release thread then frees every one of them", slabs, released.get());
    }

    /**
     * {@code readahead.window} reaches the next open and not the inputs already open, which is exactly the
     * reach the W8 limitation line claims: the value sizes a ring's buffers, so a ring built for one window
     * count is never re-sized to another.
     */
    @SneakyThrows
    public void testReadaheadWindowReachesTheNextOpen() {
        assumeDirectIOWorksHere();
        applierOver(Settings.EMPTY);
        final Path path = writeFixture();

        try (DirectIOVectorIndexInput before = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(before);
            assertEquals(KNNSettings.KNN_DIRECT_IO_READAHEAD_WINDOW_DEFAULT_VALUE, before.readaheadWindows());

            apply(Settings.builder().put(KNNSettings.KNN_DIRECT_IO_READAHEAD_WINDOW, 3).build());
            assertEquals("the input already open keeps the ring it was built with", 8, before.readaheadWindows());

            try (DirectIOVectorIndexInput after = DirectIOVectorIndexInput.open(path)) {
                assertNotNull(after);
                assertEquals("and the next open takes the new one", 3, after.readaheadWindows());
            }

            apply(Settings.builder().put(KNNSettings.KNN_DIRECT_IO_READAHEAD_WINDOW, 0).build());
            try (DirectIOVectorIndexInput off = DirectIOVectorIndexInput.open(path)) {
                assertNotNull(off);
                assertEquals("zero turns the ring off", 0, off.readaheadWindows());
            }
        }
    }

    // ---------------------------------------------------------------------------------------------------
    // The property the whole phase is for
    // ---------------------------------------------------------------------------------------------------

    /**
     * No Direct I/O setting is read on the read path -- per query, per burst or per open -- asserted from
     * the compiled form rather than from a grep: the node's settings are installed with every Direct I/O
     * key at a non-default value and <b>nothing is published</b>, and every read site still answers the
     * default. A read path that consulted {@link KNNSettings} could not pass this, and the last two
     * assertions confirm the settings really do say otherwise, so the case cannot pass by the settings
     * having been ignored.
     *
     * <p>The two {@code KNNSettings} references {@code index/store/} keeps are outside this property and
     * named here so the next reader does not go looking: {@code KNN_DIRECT_IO_DIRECT_MEMORY_WALL_BYTES},
     * a fact about this JVM read once at class load, and the {@code read_profile} resolve, which is static
     * rather than dynamic and is cached after the first answer.
     */
    @SneakyThrows
    public void testNoSettingIsReadOnTheReadPath() {
        assumeDirectIOWorksHere();
        final Settings persisted = Settings.builder()
            .put(KNNSettings.KNN_DIRECT_IO_RESCORE_ENABLED, true)
            .put(KNNSettings.KNN_DIRECT_IO_MIN_FILE_SIZE, "0b")
            .put(KNNSettings.KNN_DIRECT_IO_RESCORE_PREFETCH_THREADS, 5)
            .put(KNNSettings.KNN_DIRECT_IO_RESCORE_CACHE_MAX_BYTES, (4 * TEST_SLAB_BYTES) + "b")
            .put(KNNSettings.KNN_DIRECT_IO_READAHEAD_WINDOW, 1)
            .build();
        // The settings are installed, but initialize() -- which is what seeds the published values -- is
        // not called, so every owner is still at its field initializer.
        final Set<Setting<?>> registered = new HashSet<>(ClusterSettings.BUILT_IN_CLUSTER_SETTINGS);
        registered.addAll(
            KNNSettings.state().getSettings().stream().filter(s -> s.getProperties().contains(Setting.Property.NodeScope)).toList()
        );
        when(clusterService.getClusterSettings()).thenReturn(new ClusterSettings(persisted, registered));
        KNNSettings.state().setClusterService(clusterService);

        assertFalse("the dispatch rule did not read the setting", KNNDirectIODirectory.isRescoreEnabled());
        assertEquals(
            "nor the size floor",
            KNNSettings.KNN_DIRECT_IO_MIN_FILE_SIZE_DEFAULT_VALUE.getBytes(),
            KNNDirectIODirectory.minFileSizeBytes()
        );
        assertEquals(
            "nor the pool",
            KNNSettings.KNN_DIRECT_IO_RESCORE_PREFETCH_THREADS_DEFAULT_VALUE,
            DirectIOReadPool.configuredThreads()
        );
        assertEquals("nor the store, at the burst boundary", 0L, DirectIOVectorCache.configuredMaxBytes());
        DirectIOVectorCache.applyConfiguredSize(true);
        assertEquals("so a burst leaves the store off", 0L, DirectIOVectorCache.maxBytes());

        final Path path = writeFixture();
        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(input);
            assertEquals("nor the ring, at the open", KNNSettings.KNN_DIRECT_IO_READAHEAD_WINDOW_DEFAULT_VALUE, input.readaheadWindows());
        }

        assertTrue("and the settings really do say otherwise", KNNSettings.isDirectIORescoreEnabled());
        assertEquals(1, KNNSettings.getDirectIOReadaheadWindow());
    }

    /**
     * The startup seed, which is the other half of the same property: {@code addSettingsUpdateConsumer}
     * does not fire at registration, so without the seeding call a value an operator had already persisted
     * would be ignored until the next time they changed it. All five are asserted, each at a value that is
     * not its default, so dropping any one of the five publishes from the seeding call fails this case.
     */
    @SneakyThrows
    public void testStartupSeedingPublishesEveryPersistedValue() {
        assumeDirectIOWorksHere();
        final Path path = writeFixture();
        applierOver(
            Settings.builder()
                .put(KNNSettings.KNN_DIRECT_IO_RESCORE_ENABLED, true)
                .put(KNNSettings.KNN_DIRECT_IO_MIN_FILE_SIZE, "0b")
                .put(KNNSettings.KNN_DIRECT_IO_RESCORE_PREFETCH_THREADS, 6)
                .put(KNNSettings.KNN_DIRECT_IO_RESCORE_CACHE_MAX_BYTES, "64mb")
                .put(KNNSettings.KNN_DIRECT_IO_READAHEAD_WINDOW, 2)
                .build()
        );
        assertTrue(KNNDirectIODirectory.isRescoreEnabled());
        assertEquals(0L, KNNDirectIODirectory.minFileSizeBytes());
        assertEquals(6, DirectIOReadPool.configuredThreads());
        assertEquals(64L * 1024 * 1024, DirectIOVectorCache.configuredMaxBytes());
        // The window has no getter, because nothing on the read path asks for it twice; the open is where it
        // is read, so the open is where a test can see it.
        try (DirectIOVectorIndexInput input = DirectIOVectorIndexInput.open(path)) {
            assertNotNull(input);
            assertEquals(2, input.readaheadWindows());
        }
    }

    /**
     * The route goes last in the seeding call, so that it sees the byte target the operator persisted: a
     * node with the route off ends startup with the store at zero whatever {@code cache.max_bytes} says,
     * which is also what it holds, because nothing routes and so nothing fills.
     */
    public void testSeedingLeavesTheStoreOffWhenTheRouteIsOff() {
        DirectIOVectorCache.resetForTesting();
        applierOver(Settings.builder().put(KNNSettings.KNN_DIRECT_IO_RESCORE_CACHE_MAX_BYTES, "64mb").build());
        assertFalse("the route is at its default, which is off", KNNDirectIODirectory.isRescoreEnabled());
        assertEquals("the operator's target is still reported", 64L * 1024 * 1024, DirectIOVectorCache.configuredMaxBytes());
        assertEquals("but the store holds to nothing", 0L, DirectIOVectorCache.maxBytes());

        // And the route coming on restores it, with no query and no reopen.
        apply(
            Settings.builder()
                .put(KNNSettings.KNN_DIRECT_IO_RESCORE_CACHE_MAX_BYTES, "64mb")
                .put(KNNSettings.KNN_DIRECT_IO_RESCORE_ENABLED, true)
                .build()
        );
        assertEquals(64L * 1024 * 1024, DirectIOVectorCache.maxBytes());

        // Off again frees it, which is the finding W3 exists for: a node that stops routing stops holding.
        apply(Settings.builder().put(KNNSettings.KNN_DIRECT_IO_RESCORE_CACHE_MAX_BYTES, "64mb").build());
        assertFalse(KNNDirectIODirectory.isRescoreEnabled());
        assertEquals("the slabs went back without a burst", 0L, DirectIOVectorCache.maxBytes());
        assertEquals("while the target the operator set is unchanged", 64L * 1024 * 1024, DirectIOVectorCache.configuredMaxBytes());
    }

    /**
     * A thread change published before any burst has built a pool. {@code active_prefetch_threads} is 0
     * until then, which is what an operator reads on a node that has not yet re-scored, and the first pool
     * must come up at the published count rather than at the one that was current when the node started.
     */
    public void testAThreadChangeBeforeTheFirstPoolIsTakenByTheFirstPool() {
        applierOver(Settings.EMPTY);
        DirectIOReadPool.resetForTesting();
        assertEquals("no pool yet", 0, DirectIOReadPool.activeThreads());

        apply(Settings.builder().put(KNNSettings.KNN_DIRECT_IO_RESCORE_PREFETCH_THREADS, 7).build());
        assertEquals(7, DirectIOReadPool.configuredThreads());
        assertEquals("still no pool, so nothing to be active", 0, DirectIOReadPool.activeThreads());

        final ThreadPoolExecutor built = DirectIOReadPool.rebuildForTesting(
            DirectIOReadPool.configuredThreads(),
            DirectIOReadPool.MIN_QUEUE_CAPACITY
        );
        assertEquals("the first pool comes up at the published count", 7, built.getMaximumPoolSize());
        assertEquals(7, DirectIOReadPool.activeThreads());
    }

    // ---------------------------------------------------------------------------------------------------
    // Fixture
    // ---------------------------------------------------------------------------------------------------

    @SneakyThrows
    private void assumeDirectIOWorksHere() {
        final OpenOption direct = DirectIOVectorIndexInput.directOpenOption();
        assumeTrue("this JDK has no ExtendedOpenOption.DIRECT", direct != null);
        final Path probe = createTempDir().resolve("direct-io-probe");
        Files.write(probe, new byte[8192]);
        try (FileChannel channel = FileChannel.open(probe, StandardOpenOption.READ, direct)) {
            channel.read(ByteBuffer.allocateDirect(8192).alignedSlice(4096));
        } catch (IOException | UnsupportedOperationException e) {
            assumeNoException("this filesystem refuses O_DIRECT", e);
        }
    }

    /** A {@code .vec}-shaped file, large enough that a ring of any window count has somewhere to read. */
    @SneakyThrows
    private Path writeFixture() {
        final Path path = createTempDir().resolve("consumer.vec");
        final ByteBuffer buffer = ByteBuffer.allocate((int) HEADER_LENGTH + VECTOR_COUNT * VECTOR_BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < HEADER_LENGTH; i++) {
            buffer.put((byte) (0xA0 + (i & 0x0F)));
        }
        for (int v = 0; v < VECTOR_COUNT; v++) {
            for (int d = 0; d < DIMENSION; d++) {
                buffer.putFloat(v + d / 100.0f);
            }
        }
        Files.write(path, buffer.array());
        return path;
    }

    /** Admits enough distinct vectors to take every slab the target allows. */
    private void fillEverySlab(final int slabs) {
        final String fileId = "consumer-fill.vec#1";
        final int perSlab = TEST_SLAB_BYTES / VECTOR_BYTES;
        final ByteBuffer staging = ByteBuffer.allocate(VECTOR_BYTES).order(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i <= slabs * perSlab; i++) {
            for (int d = 0; d < DIMENSION; d++) {
                staging.putFloat(d * Float.BYTES, i + d);
            }
            DirectIOVectorCache.put(fileId, HEADER_LENGTH + (long) i * VECTOR_BYTES, staging, 0, VECTOR_BYTES);
            if (DirectIOVectorCache.slabs() == slabs) {
                return;
            }
        }
    }
}

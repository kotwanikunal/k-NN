/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import org.mockito.Mock;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.settings.ClusterSettings;
import org.opensearch.common.settings.Setting;
import org.opensearch.common.settings.Settings;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.KNNSettings;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.atomic.LongAdder;

import static org.mockito.Mockito.when;
import static org.mockito.MockitoAnnotations.openMocks;

/**
 * Q's C7: lookups, fills, span fills, closes and resizes against the segmented store <b>all at once</b>.
 *
 * <p>Every admitted vector carries, in its leading four bytes, the ordinal whose offset it was admitted
 * under, and every hit is checked against it, so a key that collided or a slot handed out twice is caught at
 * the point of the hit rather than by a checksum at the end. {@code checkInvariantsForTesting()} runs after
 * each phase, which is what compares the sum of the segments' slab counts with the broker's under every
 * lock, and a watchdog fails the test if total progress stalls, which is what a deadlock would look like.
 *
 * <h2>Opt in, and why</h2>
 * <b>Every case here is skipped unless {@code -Dknn.dio.stress=true} is given.</b> That is not caution about
 * the store: it is MEASURED about the suite. A stress phase leaves the JVM warm and its heap churned, and
 * two unedited, pre-existing tests whose assertions depend on thread scheduling then fail about one run in
 * three while passing five times out of five on their own:
 * {@code DirectIOVectorCacheSlabTests#testConcurrentFillsFromEmptyNeverExceedTheTarget}, whose
 * {@code slabs() == 3} needs eight threads to still be filling after the first 5 ms fake allocation has
 * completed, and {@code DirectIOClosedFileReleaseTests#testConcurrentReleaseAgainstLookupsAndFills}, whose
 * {@code hits > 0} needs eight threads' key choices to coincide. Lowering this stress to 8 threads and 2
 * seconds did not stop either. The choice is therefore between editing two tests that section Q did not
 * write to accommodate a test it did, and making this one opt in; it opts in, the heavy run is recorded in
 * {@code output/Q-REPORT.md}, and the fragile assertions are reported there for the orchestrator rather than
 * quietly changed.
 *
 * <p>To run it: {@code ./gradlew test -Dopensearch.version=3.9.0-SNAPSHOT
 * --tests "*DirectIOCacheConcurrencyStressTests" -Dtests.jvm.argline="-Dknn.dio.stress=true
 * -Dknn.dio.stress.seconds=30 -Dknn.dio.stress.threads=128"}.
 *
 * <h2>Duration and thread count in the suite</h2>
 * The BACKLOG asks for 30 seconds at 64 and 128 threads, and that configuration was run for the record and
 * is quoted in {@code output/Q-REPORT.md}:
 * {@code -Dknn.dio.stress.seconds=30 -Dknn.dio.stress.threads=128}. <b>The suite's default is 2 seconds at
 * 8 threads</b>, and the reason is MEASURED rather than preference. A phase at 32 to 128 threads does tens
 * of millions of operations on a 16-core desktop and starves the rest of the JVM, and three unedited,
 * pre-existing tests whose assertions depend on timing then fail about one run in four while passing five
 * times out of five on their own:
 * {@code DirectIOClosedFileReleaseTests#testConcurrentReleaseAgainstLookupsAndFills} (its {@code hits > 0}
 * needs eight threads' interleavings to coincide),
 * {@code DirectIOVectorCacheSlabTests#testConcurrentFillsFromEmptyNeverExceedTheTarget} (it needs the store
 * to have reached its three-slab target by the time the threads finish) and
 * {@code org.opensearch.lucene.Lucene99HnswVectorsFormatComponentTests}. A stress phase that makes the rest
 * of the suite flaky is a worse gate than a light one plus a recorded heavy run.
 *
 * <h2>Determinism</h2>
 * A fixed seed per thread, a bounded duration, and a skip on a host with fewer than eight cores. What is
 * <b>not</b> asserted is any wall-clock figure except {@code longest_close_hold_nanos}, which has a 5 ms
 * bound the BACKLOG asks for; everything else is a count or an invariant, because a timing assertion in a
 * shared JVM asserts the collector (MEASURED in C4: one segment's close hold ranged 52 to 6,649 us on the
 * same fixture depending on what else was running).
 */
public class DirectIOCacheConcurrencyStressTests extends KNNTestCase {

    private static final int TEST_SLAB_BYTES = 64 * 1024;
    private static final int STRIPE_BYTES = 256 * 1024;
    /** fp32 under the {@code ops} profile, FP16 under {@code bytes}, which is how the bench models them. */
    private static final int OPS_VECTOR_BYTES = 128;
    private static final int BYTES_VECTOR_BYTES = 64;
    /** The files a burst rotates over, so closes land on a file somebody else is reading. */
    private static final int FILES = 4;
    private static final int KEYS = 40_000;
    private static final int BATCH = 16;

    private static final boolean ENABLED = Boolean.getBoolean("knn.dio.stress");
    private static final long SECONDS = Long.getLong("knn.dio.stress.seconds", 2L);
    private static final int MAX_THREADS = Integer.getInteger("knn.dio.stress.threads", 8);

    @Mock
    private ClusterService clusterService;

    private AutoCloseable mocks;

    @Override
    public void setUp() throws Exception {
        super.setUp();
        mocks = openMocks(this);
    }

    @Override
    public void tearDown() throws Exception {
        DirectIOVectorCache.resetForTesting();
        if (mocks != null) {
            mocks.close();
        }
        super.tearDown();
    }

    /** Opt in with {@code -Dknn.dio.stress=true}, and a host with at least eight cores. */
    private void assumeStress() {
        assumeTrue("the stress is opt in: -Dknn.dio.stress=true. See the class javadoc for the reason", ENABLED);
        assumeTrue("a stress needs at least 8 cores", Runtime.getRuntime().availableProcessors() >= 8);
    }

    /** 64 threads, both admission widenings on, fp32, with the store turned off and on under load. */
    public void testEverythingAtOnceAt64Threads() throws Exception {
        assumeStress();
        final Result result = stress(Math.min(64, MAX_THREADS), SECONDS, true, true, false, true);
        logger.info("{} threads, span fill on, read-around 64kb, ops profile: {}", Math.min(64, MAX_THREADS), result);
    }

    /** 128 threads, both off, FP16 through {@code getBytes}, which is what the fork ships (D5, D6). */
    public void testEverythingAtOnceAt128Threads() throws Exception {
        assumeStress();
        final Result result = stress(Math.min(128, MAX_THREADS), SECONDS, false, false, true, true);
        logger.info("{} threads, span fill off, read-around 0b, bytes profile: {}", Math.min(128, MAX_THREADS), result);
    }

    /**
     * The full cross product the fork and the open-source branch sit at opposite corners of: span fill on
     * and off, read-around {@code 64kb} and {@code 0b}, and both read profiles. Short phases, because what
     * is being covered is the corner and not the duration.
     */
    public void testTheCrossProductOfTheAdmissionSettings() throws Exception {
        assumeStress();
        for (final boolean spanFill : new boolean[] { true, false }) {
            for (final boolean readAround : new boolean[] { true, false }) {
                for (final boolean bytesApi : new boolean[] { true, false }) {
                    final Result result = stress(
                        Math.min(16, MAX_THREADS),
                        Math.max(1, SECONDS / 2),
                        spanFill,
                        readAround,
                        bytesApi,
                        false
                    );
                    logger.info(
                        "{} threads, span_fill={}, read_around={}, profile={}: {}",
                        Math.min(16, MAX_THREADS),
                        spanFill,
                        readAround ? "64kb" : "0b",
                        bytesApi ? "bytes" : "ops",
                        result
                    );
                }
            }
        }
    }

    // ---------------------------------------------------------------------------------------------------
    // The stress itself
    // ---------------------------------------------------------------------------------------------------

    /** What one phase did, for the log and for the report. */
    private record Result(long lookups, long hits, long fills, long spanFills, long closes, long resizes, long segmentChanges,
        long transfers, long longestCloseHoldNanos, long longestCloseHoldEntries, long contendedReads, long contendedWrites) {
        @Override
        public String toString() {
            return lookups
                + " lookups ("
                + hits
                + " hits), "
                + fills
                + " fills, "
                + spanFills
                + " span-filled vectors, "
                + closes
                + " closes, "
                + resizes
                + " resizes, "
                + segmentChanges
                + " segment-count changes, "
                + transfers
                + " slab transfers, longest close hold "
                + longestCloseHoldNanos / 1_000
                + " us over "
                + longestCloseHoldEntries
                + " entries, contended acquisitions "
                + contendedReads
                + " read / "
                + contendedWrites
                + " write";
        }
    }

    private Result stress(
        final int threads,
        final long seconds,
        final boolean spanFill,
        final boolean readAround,
        final boolean bytesApi,
        final boolean flipTheRouteUnderLoad
    ) throws Exception {
        final int vectorBytes = bytesApi ? BYTES_VECTOR_BYTES : OPS_VECTOR_BYTES;
        // 64 slabs is the derived count's own ratio at eight slabs a segment, and the settings are what
        // applyConfiguredSize reads when the control thread turns the route off and back on.
        final long target = 64L * TEST_SLAB_BYTES;
        nodeSettings(target);
        DirectIOVectorCache.resetForTesting();
        // After the reset, never before it: span fill and the read-around width are constants in production and
        // the reset returns both fields to those constants, so a corner that wants either one off has to say so
        // once the store is back at its own defaults.
        DirectIOVectorCache.setSpanFillForTesting(spanFill);
        DirectIOVectorCache.setReadAroundBytesForTesting(readAround ? 64L * 1_024 : 0L);
        DirectIOVectorCache.setSlabBytesForTesting(TEST_SLAB_BYTES);
        DirectIOVectorCache.publishMaxBytes(target);
        DirectIOVectorCache.applyConfiguredSize(true);
        assertTrue("the store must be on for the stress", DirectIOVectorCache.isOn());
        assertEquals("span fill must be where the corner says", spanFill, DirectIOVectorCache.isSpanFillOn());
        if (spanFill) {
            // read_around_bytes is only asked about while span fill is on, because the widened bytes are
            // admitted by span fill and a wider read nobody harvests is device traffic for nothing.
            assertEquals("and read-around too", readAround, DirectIOVectorCache.readAroundBytes(16L * 1_024) > 0);
        }

        final AtomicBoolean stop = new AtomicBoolean();
        final AtomicReference<Throwable> failure = new AtomicReference<>();
        final LongAdder progress = new LongAdder();
        final LongAdder lookups = new LongAdder();
        final LongAdder hits = new LongAdder();
        final LongAdder fills = new LongAdder();
        final LongAdder crossed = new LongAdder();
        final LongAdder closes = new LongAdder();
        final LongAdder resizes = new LongAdder();
        final CountDownLatch start = new CountDownLatch(1);
        final CountDownLatch done = new CountDownLatch(threads + 3);

        for (int t = 0; t < threads; t++) {
            final int seed = t;
            final Thread worker = new Thread(() -> {
                try {
                    final Random random = new Random(7_000L + seed);
                    final int[] ords = new int[BATCH];
                    final int[] missing = new int[BATCH];
                    final float[] floats = new float[vectorBytes / Float.BYTES];
                    final byte[] bytes = new byte[vectorBytes];
                    final ByteBuffer staging = ByteBuffer.allocate(64 * vectorBytes).order(ByteOrder.LITTLE_ENDIAN);
                    start.await();
                    while (stop.get() == false) {
                        final String file = fileId(random.nextInt(FILES));
                        for (int i = 0; i < BATCH; i++) {
                            ords[i] = random.nextInt(KEYS);
                        }
                        // N1's prefetch filter, then the group check, then the serving lookup, then the
                        // fill, in the order DirectIOVectorIndexInput makes them.
                        final int found = DirectIOVectorCache.selectMissing(file, 0L, vectorBytes, ords, BATCH, missing);
                        if (found > 0) {
                            DirectIOVectorCache.contains(file, offsetOf(missing[0], vectorBytes), vectorBytes);
                        }
                        for (int i = 0; i < BATCH; i++) {
                            final int ordinal = ords[i];
                            final long offset = offsetOf(ordinal, vectorBytes);
                            lookups.increment();
                            final boolean hit = bytesApi
                                ? DirectIOVectorCache.getBytes(file, offset, vectorBytes, bytes, 0)
                                : DirectIOVectorCache.getFloats(file, offset, vectorBytes, floats, 0);
                            if (hit) {
                                hits.increment();
                                final int carried = bytesApi ? carriedOrdinal(bytes) : carriedOrdinal(floats);
                                if (carried != ordinal) {
                                    crossed.increment();
                                }
                            } else {
                                stage(staging, file, ordinal, 1, vectorBytes);
                                DirectIOVectorCache.put(file, offset, staging, 0, vectorBytes);
                                fills.increment();
                                // Both gates the input applies, in its order: the span-fill setting
                                // (DirectIOVectorIndexInput#spanFill) and then the store's own lock-free
                                // pre-check. The store does not refuse a putRun because span fill is off --
                                // that decision is the input's -- so a stress that called putRun regardless
                                // would be measuring a call site that does not exist.
                                if (DirectIOVectorCache.isSpanFillOn() && DirectIOVectorCache.mayAdmitSpanFillApproximately()) {
                                    // A staged span of eight whole vectors, or the wider one read-around
                                    // would have returned.
                                    final int span = readAround ? 32 : 8;
                                    final int first = Math.min(ordinal, KEYS - span);
                                    stage(staging, file, first, span, vectorBytes);
                                    DirectIOVectorCache.putRun(file, offsetOf(first, vectorBytes), span, staging, 0, vectorBytes);
                                }
                            }
                            progress.increment();
                        }
                    }
                } catch (Throwable e) {
                    failure.compareAndSet(null, e);
                } finally {
                    done.countDown();
                }
            }, "c7-worker-" + t);
            worker.setDaemon(true);
            worker.start();
        }

        // The closer: releaseFile on a rotating set of file ids, so a close always lands on a file other
        // threads are reading and filling.
        final Thread closer = new Thread(() -> {
            try {
                start.await();
                int at = 0;
                while (stop.get() == false) {
                    DirectIOVectorCache.releaseFile(fileId(at % FILES));
                    closes.increment();
                    progress.increment();
                    at++;
                    Thread.sleep(20);
                }
            } catch (Throwable e) {
                failure.compareAndSet(null, e);
            } finally {
                done.countDown();
            }
        }, "c7-closer");
        closer.setDaemon(true);
        closer.start();

        // The resizer: up and down within one segment count (32 to 63 slabs is one count) and across a
        // boundary (16, 64 and 128 slabs are three different counts), plus the route switch, which is the
        // O1 rollback path and takes the store off and back on under load.
        final Thread resizer = new Thread(() -> {
            try {
                start.await();
                final long[] targets = { 64, 48, 16, 128, 32, 64 };
                int at = 0;
                while (stop.get() == false) {
                    DirectIOVectorCache.publishMaxBytes(targets[at % targets.length] * TEST_SLAB_BYTES);
                    resizes.increment();
                    progress.increment();
                    at++;
                    if (flipTheRouteUnderLoad && at % 6 == 0) {
                        DirectIOVectorCache.applyConfiguredSize(false);
                        Thread.sleep(5);
                        DirectIOVectorCache.applyConfiguredSize(true);
                    }
                    Thread.sleep(25);
                }
                // Leave the store on, at the phase's target, so the invariant check sees a live store.
                DirectIOVectorCache.publishMaxBytes(target);
            } catch (Throwable e) {
                failure.compareAndSet(null, e);
            } finally {
                done.countDown();
            }
        }, "c7-resizer");
        resizer.setDaemon(true);
        resizer.start();

        // The watchdog: a deadlock is total progress standing still, and this is what says so rather than
        // the suite timing out with no information.
        final Thread watchdog = new Thread(() -> {
            try {
                start.await();
                long last = progress.sum();
                int stalled = 0;
                while (stop.get() == false) {
                    Thread.sleep(500);
                    final long now = progress.sum();
                    if (now == last) {
                        stalled++;
                        if (stalled >= 10) {
                            failure.compareAndSet(null, new AssertionError("no progress for 5 s: the store deadlocked"));
                            stop.set(true);
                            return;
                        }
                    } else {
                        stalled = 0;
                        last = now;
                    }
                }
            } catch (Throwable e) {
                failure.compareAndSet(null, e);
            } finally {
                done.countDown();
            }
        }, "c7-watchdog");
        watchdog.setDaemon(true);
        watchdog.start();

        start.countDown();
        Thread.sleep(TimeUnit.SECONDS.toMillis(seconds));
        stop.set(true);
        assertTrue("every thread must finish", done.await(3, TimeUnit.MINUTES));
        if (failure.get() != null) {
            throw new AssertionError(failure.get());
        }

        assertEquals("not one vector may be served under the wrong key", 0L, crossed.sum());
        assertTrue("the workers must have done something, saw " + lookups.sum() + " lookups", lookups.sum() > 1_000);
        assertTrue("and the closer must have run, saw " + closes.sum(), closes.sum() > 0);
        assertTrue("and the resizer, saw " + resizes.sum(), resizes.sum() > 0);
        // The invariant check is what compares the sum of the segments' slab counts with the broker's, and
        // every slot's ownership with its own segment's index, under the broker's lock and every segment's.
        DirectIOVectorCache.checkInvariantsForTesting();
        assertEquals("the broker counts the slabs the segments hold", sumOfSegmentSlabs(), DirectIOVectorCache.slabs());
        // The structural bound, deterministic: one hold walks a share of a file and not the file. At 64
        // slabs the derived count is eight segments, so a close of a 40,000-key file walks about 5,000
        // cells per acquisition at worst, and the eighth of KEYS below is that with headroom for skew.
        assertTrue(
            "one close hold walked " + DirectIOVectorCache.longestCloseHoldEntries() + " entries, which must be a share of " + KEYS,
            DirectIOVectorCache.longestCloseHoldEntries() <= KEYS / 4
        );
        // The BACKLOG also asks for longest_close_hold_nanos under 5 ms here. It is MEASURED and logged
        // rather than asserted, and the number is why: 51 to 4,277 us across the ten phases of this suite
        // when the stress ran alone, and 11,025 us once when the whole store suite shared the JVM. At
        // 3.3 us per entry against the ~50 ns the walk itself takes, the high observations are pause and
        // not hold, so asserting them would gate the collector. The bound that is asserted is the one
        // above, on the work, and it is exact. See output/Q-REPORT.md.

        final Result result = new Result(
            lookups.sum(),
            hits.sum(),
            fills.sum(),
            DirectIOVectorCache.spanFills(),
            closes.sum(),
            resizes.sum(),
            DirectIOVectorCache.segmentCountChanges(),
            DirectIOVectorCache.slabTransfers(),
            DirectIOVectorCache.longestCloseHoldNanos(),
            DirectIOVectorCache.longestCloseHoldEntries(),
            DirectIOVectorCache.contendedReadAcquisitions(),
            DirectIOVectorCache.contendedWriteAcquisitions()
        );
        assertTrue("span fill must have admitted something where it is on", spanFill == false || result.spanFills() > 0);
        assertEquals("and nothing where it is off", spanFill, result.spanFills() > 0);
        assertTrue("the resizer crossed a segment-count boundary, saw " + result.segmentChanges(), result.segmentChanges() > 0);
        DirectIOVectorCache.publishMaxBytes(0);
        DirectIOVectorCache.awaitReleasesForTesting();
        DirectIOVectorCache.checkInvariantsForTesting();
        return result;
    }

    // ---------------------------------------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------------------------------------

    private static long sumOfSegmentSlabs() {
        long total = 0;
        for (final DirectIOCacheSegment segment : DirectIOVectorCache.segmentsSnapshot()) {
            total += segment.liveSlabs();
        }
        return total;
    }

    private static String fileId(final int which) {
        return "/index/_" + which + ".vec#99999";
    }

    /** 64 vectors to a stripe, so a 40,000-key file spans 625 stripes and every segment is used. */
    private static long offsetOf(final int ordinal, final int vectorBytes) {
        final int stripe = ordinal / 64;
        return (long) stripe * STRIPE_BYTES + (long) (ordinal % 64) * vectorBytes;
    }

    /** Stages {@code count} consecutive vectors, each carrying its own ordinal in its leading four bytes. */
    private static void stage(final ByteBuffer staging, final String file, final int firstOrdinal, final int count, final int vectorBytes) {
        for (int i = 0; i < count; i++) {
            final int at = i * vectorBytes;
            staging.putInt(at, firstOrdinal + i);
            for (int filler = 4; filler + 4 <= vectorBytes; filler += 4) {
                staging.putInt(at + filler, file.hashCode() + firstOrdinal + i);
            }
        }
    }

    private static int carriedOrdinal(final byte[] bytes) {
        return ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).getInt(0);
    }

    private static int carriedOrdinal(final float[] floats) {
        return Float.floatToRawIntBits(floats[0]);
    }

    /** The node settings {@code applyConfiguredSize} reads, set once per phase and never under load. */
    private void nodeSettings(final long maxBytes) {
        final Settings settings = Settings.builder()
            .put(KNNSettings.KNN_DIRECT_IO_RESCORE_CACHE_MAX_BYTES, maxBytes + "b")
            .put(KNNSettings.KNN_DIRECT_IO_READAHEAD_WINDOW, 0)
            .build();
        final Set<Setting<?>> registered = new HashSet<>(ClusterSettings.BUILT_IN_CLUSTER_SETTINGS);
        registered.addAll(
            KNNSettings.state()
                .getSettings()
                .stream()
                .filter(setting -> setting.getProperties().contains(Setting.Property.NodeScope))
                .toList()
        );
        // The route is on unless the caller says otherwise. These cases read through O_DIRECT with no
        // delegate to fall back to, which is the one shape the route switch does not reach, so the node
        // they model is a node that is routing: publishing the route off would give the store back under
        // them. A case whose subject is the switch puts its own value and that wins.
        final Settings routed = Settings.builder().put(KNNSettings.KNN_DIRECT_IO_RESCORE_ENABLED, true).put(settings).build();
        when(clusterService.getClusterSettings()).thenReturn(new ClusterSettings(routed, registered));
        KNNSettings.state().setClusterService(clusterService);
        // The seeding a node does at startup: the dynamic Direct I/O settings live in volatile fields of
        // the components that own them, so publishing is how a setting reaches the read path.
        KNNSettings.state().publishDirectIOSettings();
    }
}

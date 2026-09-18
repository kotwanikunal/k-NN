/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

/**
 * <b>The staging seam.</b> "I am about to want these ordinals": where read ahead sits.
 *
 * <p>A {@link VectorLoaderSource.Loader} may also implement this. When it does, the query path tells it
 * which ordinals the next batch of scores will read, in the order they will be read, so those reads can be
 * in flight before they are needed. This exists because the mmap path this seam replaces got its speed from
 * a {@code madvise} prefetch, which has no meaning without a mapping: with one blocking read per candidate
 * a rescore query pays 200 serial device round trips, measured at 123 ms against 33 ms staged on the same
 * index.
 *
 * <h2>Advisory, in both directions</h2>
 * This is the whole safety argument for read ahead, and it must not erode:
 * <ul>
 *   <li>A loader may stage <b>none</b> of what it is offered, for any reason it likes.</li>
 *   <li>A caller may then read something else, or read the same ordinals out of order, and must still get
 *       correct vectors.</li>
 * </ul>
 * So {@link #stage} returns nothing and promises nothing, and no score may depend on staging having
 * happened. A loader that does not implement this interface at all is fully functional, just slower — the
 * query path checks for it and carries on without it.
 *
 * <h2>Why this is not part of the loader seam</h2>
 * {@link VectorLoaderSource} and this interface have opposite lifecycles, and a cache at the loader seam is
 * the reason to care:
 * <table border="1">
 *   <caption>opposite lifecycles</caption>
 *   <tr><th></th><th>staging</th><th>a cache at the loader seam</th></tr>
 *   <tr><td>capacity</td><td>fixed ring, sized by the read-ahead window</td><td>sized by memory budget and
 *       hit rate</td></tr>
 *   <tr><td>retention</td><td>each entry consumed exactly once, then reused immediately</td><td>entries
 *       retained across queries, in the hope of reuse</td></tr>
 *   <tr><td>eviction</td><td>none — there is nothing to evict</td><td>a policy, the main thing a cache
 *       is</td></tr>
 *   <tr><td>success measure</td><td>queue depth</td><td>hit rate</td></tr>
 * </table>
 *
 * <p>Fused into one interface, those two sets of concerns would share an implementation, and adding a cache
 * would mean rewriting read ahead rather than implementing an interface next to it. Kept separate, a cache
 * implements {@link VectorLoaderSource} and either offers staging or does not.
 */
public interface VectorStagingArea {

    /**
     * Declares that {@code ords[0..count)} are about to be read, in that order.
     *
     * <p>Must be called from the thread that will do the reading, and is free to do nothing. It returns no
     * indication of what was staged, on purpose: a caller that could tell would be tempted to depend on it.
     *
     * @param ords  ordinals in the order they will be read; only the first {@code count} are meaningful
     * @param count how many entries of {@code ords} to look at
     */
    void stage(int[] ords, int count);
}

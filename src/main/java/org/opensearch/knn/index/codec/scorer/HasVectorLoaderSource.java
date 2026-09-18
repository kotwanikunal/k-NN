/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.scorer;

import org.apache.lucene.index.KnnVectorValues;
import org.opensearch.knn.index.store.VectorLoaderSource;

/**
 * Implemented by {@link KnnVectorValues} that can name a {@link VectorLoaderSource} for the same
 * full-precision vectors they serve through mmap, so a query that knows those reads have no reuse can read
 * them without filling the page cache.
 *
 * <p>This is the codec side of the loader seam, and the point where the loader implementation is chosen: a
 * future cache would be introduced by returning one here instead of the Direct I/O source, with nothing on
 * the query path changing. It exists as a capability interface because the query seam that decides whether a
 * read is a rescore read holds only a {@link KnnVectorValues}, while the file those vectors live in is known
 * only to the codec reader that opened it — {@code IndexInput} does not expose its file, and the vector
 * region's offset within that file is written into the {@code .vem} metadata and then kept private. So the
 * codec reader answers the question instead of the query layer guessing.
 *
 * <p>Two properties of the contract are load bearing:
 * <ul>
 *   <li><b>Lazy.</b> The source must not be established until this method is first called. The Direct I/O
 *       rescore flag is a dynamic node setting, and with it off nothing may call this, so nothing may
 *       open a second file handle. That is what makes "flag off is bit-identical to today" a property of
 *       the code rather than a claim about it.</li>
 *   <li><b>Shared and segment-scoped.</b> One source per field per segment, reused by every query, and
 *       closed when the reader closes. Vector values are created per query, so a source owned by them
 *       would open and leak a file descriptor per query. The per-read state lives in a
 *       {@link VectorLoaderSource.Loader} instead.</li>
 * </ul>
 *
 * <p>{@code null} is an ordinary answer, not an error: it means this segment's vectors cannot be served
 * through the loader seam — a compound file, an unexpected file layout, a filesystem that refuses
 * {@code O_DIRECT} — and the caller reads them the way it does today.
 *
 * @see HasFullPrecisionVectorValues
 */
public interface HasVectorLoaderSource {

    /**
     * The loader seam for these values' full-precision vectors, or {@code null} when there is none.
     * Establishes it on the first call and returns the same instance afterwards.
     *
     * @return a source sharing this values object's ordinal space, or {@code null} to mean "use the
     *         default path"
     */
    VectorLoaderSource vectorLoaderSource();
}

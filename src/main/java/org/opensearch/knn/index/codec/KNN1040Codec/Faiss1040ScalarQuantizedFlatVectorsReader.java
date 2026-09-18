/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.codec.KNN1040Codec;

import lombok.extern.log4j.Log4j2;
import org.apache.lucene.codecs.hnsw.FlatVectorsReader;
import org.apache.lucene.codecs.hnsw.FlatVectorsScorer;
import org.apache.lucene.index.ByteVectorValues;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexFileNames;
import org.apache.lucene.index.SegmentReadState;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.store.FilterDirectory;
import org.apache.lucene.util.hnsw.RandomVectorScorer;
import org.opensearch.common.Nullable;
import org.opensearch.knn.index.store.DirectIOVectorSource;
import org.opensearch.knn.index.store.VectorLoaderSource;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

/**
 * A {@link FlatVectorsReader} wrapper for Faiss SQ vector fields that exposes both the
 * full-precision floats and the quantized codes on the {@link FloatVectorValues} returned by
 * {@link #getFloatVectorValues(String)}.
 *
 * <p>Lucene's {@code Lucene104ScalarQuantizedVectorsReader} returns a {@code ScalarQuantizedVectorValues}
 * that hides its quantized byte delegate behind a private field. Callers on the plugin side
 * (warmup, native scoring) need both the full-precision {@code .vec} floats and the quantized
 * {@code .veq} codes, so this reader wraps the delegate's values with
 * {@link ScalarQuantizedFloatVectorValues}, which carries both delegates explicitly.
 *
 * <p>The resulting reader hierarchy is:
 * <pre>
 *   Faiss1040ScalarQuantizedKnnVectorsReader
 *     └─ Faiss1040ScalarQuantizedFlatVectorsReader  (this class)
 *          └─ Lucene104ScalarQuantizedVectorsReader  (delegate)
 * </pre>
 *
 * <p>All other operations are delegated directly to the underlying reader.
 */
@Log4j2
public class Faiss1040ScalarQuantizedFlatVectorsReader extends FlatVectorsReader {

    /** Extension of Lucene's flat full-precision vector file, the one the rescore path reads. */
    private static final String VECTOR_DATA_EXTENSION = "vec";

    private final FlatVectorsReader delegateFlatVectorsReader;

    /**
     * Filesystem path of this segment's {@code .vec} file, or {@code null} when it cannot be named — a
     * compound segment, a directory that is not filesystem backed, or a reader built without a
     * {@link SegmentReadState}. Computed once, without touching the filesystem.
     */
    @Nullable
    private final Path vectorDataPath;

    /**
     * Loader seams by field, established on first use and shared for this reader's life. Empty until a
     * Direct I/O rescore query asks for one, so a node with the flag off never opens a second handle.
     * {@link Optional#empty()} caches "there is none", so a field that cannot be served is not retried
     * once per query.
     *
     * <p>Typed as {@link VectorLoaderSource} rather than the Direct I/O implementation because this is the
     * one place the implementation behind the seam is chosen: a future full-precision vector cache would be
     * introduced by constructing a different one in {@link #vectorLoaderSource(String)}, with nothing on the
     * query path changing.
     */
    private final Map<String, Optional<VectorLoaderSource>> vectorLoaderSources = new ConcurrentHashMap<>();

    /**
     * @param lucene104ScalarQuantizedVectorsReader the delegate reader whose {@link FloatVectorValues}
     *                                              will be wrapped to implement {@code HasIndexSlice}
     */
    protected Faiss1040ScalarQuantizedFlatVectorsReader(final FlatVectorsReader lucene104ScalarQuantizedVectorsReader) {
        this(lucene104ScalarQuantizedVectorsReader, null);
    }

    /**
     * @param lucene104ScalarQuantizedVectorsReader the delegate reader whose {@link FloatVectorValues}
     *                                              will be wrapped to implement {@code HasIndexSlice}
     * @param state the read state the segment is being opened with, used only to name the {@code .vec}
     *              file for Direct I/O rescoring. Passing {@code null} builds a reader that offers no
     *              Direct I/O source, which is exactly the behaviour before that path existed.
     */
    protected Faiss1040ScalarQuantizedFlatVectorsReader(
        final FlatVectorsReader lucene104ScalarQuantizedVectorsReader,
        @Nullable final SegmentReadState state
    ) {
        super();
        this.delegateFlatVectorsReader = lucene104ScalarQuantizedVectorsReader;
        this.vectorDataPath = resolveVectorDataPath(state);
    }

    /**
     * The {@code .vec} file this segment's full-precision vectors live in, or {@code null}.
     * <p>
     * Lucene's per-field vectors format gives each format instance its own {@code segmentSuffix}, and the
     * flat vectors reader nested inside the scalar-quantized one is handed the same
     * {@link SegmentReadState}, so the file it opened is the one this name resolves to.
     * <p>
     * A compound segment resolves to {@code null} by construction: its files are regions of a
     * {@code .cfs}, its directory is Lucene's compound reader rather than an {@link FSDirectory}, and
     * {@link FilterDirectory#unwrap} does not see through it. That is the graceful compound-file fallback
     * the design calls for, and it costs nothing to get.
     */
    @Nullable
    static Path resolveVectorDataPath(@Nullable final SegmentReadState state) {
        if (state == null) {
            return null;
        }
        try {
            final Directory unwrapped = FilterDirectory.unwrap(state.directory);
            if (unwrapped instanceof FSDirectory fsDirectory) {
                final String name = IndexFileNames.segmentFileName(state.segmentInfo.name, state.segmentSuffix, VECTOR_DATA_EXTENSION);
                return fsDirectory.getDirectory().resolve(name);
            }
            log.debug(
                "Direct I/O rescore has no path for [{}]: directory is [{}]",
                state.segmentInfo.name,
                unwrapped.getClass().getSimpleName()
            );
            return null;
        } catch (RuntimeException e) {
            // Naming a file must never fail opening a segment.
            log.debug("Could not name the .vec file for Direct I/O rescoring", e);
            return null;
        }
    }

    /**
     * The loader seam for {@code field}, establishing and verifying it on the first call and returning the
     * same instance afterwards. {@code null} means this field's vectors cannot be served that way and the
     * caller should read them through mmap as usual.
     * <p>
     * Verification needs full-precision values to compare against, and this method makes its own rather
     * than borrowing the caller's: the caller's are about to be read by a scorer, and
     * {@code vectorValue(int)} mutates the values it is called on.
     */
    @Nullable
    VectorLoaderSource vectorLoaderSource(final String field) {
        if (vectorDataPath == null) {
            return null;
        }
        return vectorLoaderSources.computeIfAbsent(field, name -> {
            try {
                final FloatVectorValues reference = delegateFlatVectorsReader.getFloatVectorValues(name);
                return Optional.ofNullable(DirectIOVectorSource.open(vectorDataPath, reference));
            } catch (IOException | RuntimeException e) {
                log.warn("Direct I/O rescore could not verify [{}] for field [{}]; using the default path", vectorDataPath, name, e);
                return Optional.empty();
            }
        }).orElse(null);
    }

    @Override
    public RandomVectorScorer getRandomVectorScorer(String field, float[] target) throws IOException {
        return delegateFlatVectorsReader.getRandomVectorScorer(field, target);
    }

    @Override
    public RandomVectorScorer getRandomVectorScorer(String field, byte[] target) throws IOException {
        return delegateFlatVectorsReader.getRandomVectorScorer(field, target);
    }

    @Override
    public void checkIntegrity() throws IOException {
        delegateFlatVectorsReader.checkIntegrity();
    }

    /**
     * Returns {@link FloatVectorValues} wrapped with {@link ScalarQuantizedFloatVectorValues},
     * which exposes both the full-precision float delegate and the quantized byte delegate via
     * dedicated getters. Empty values are wrapped with no quantized backing because Lucene does
     * not expose one.
     */
    @Override
    public FloatVectorValues getFloatVectorValues(String field) throws IOException {
        final FloatVectorValues floatVectorValues = delegateFlatVectorsReader.getFloatVectorValues(field);
        if (floatVectorValues == null) {
            return null;
        }

        if (floatVectorValues.size() == 0) {
            return new ScalarQuantizedFloatVectorValues(floatVectorValues, null);
        }

        return new ScalarQuantizedFloatVectorValues(
            floatVectorValues,
            KNN1040ScalarQuantizedUtils.extractQuantizedByteVectorValues(floatVectorValues),
            () -> vectorLoaderSource(field)
        );
    }

    @Override
    public ByteVectorValues getByteVectorValues(String field) throws IOException {
        return delegateFlatVectorsReader.getByteVectorValues(field);
    }

    /**
     * Closes the delegate and any loader seams this reader established. The delegate is closed even if a
     * seam fails to close, since it owns every file the default path reads.
     */
    @Override
    public void close() throws IOException {
        try {
            for (final Map.Entry<String, Optional<VectorLoaderSource>> entry : vectorLoaderSources.entrySet()) {
                final Optional<VectorLoaderSource> source = entry.getValue();
                if (source.isPresent()) {
                    try {
                        source.get().close();
                    } catch (IOException e) {
                        log.warn("Failed to close the vector loader source for field [{}]: {}", entry.getKey(), source.get(), e);
                    }
                }
            }
            vectorLoaderSources.clear();
        } finally {
            delegateFlatVectorsReader.close();
        }
    }

    @Override
    public long ramBytesUsed() {
        return delegateFlatVectorsReader.ramBytesUsed();
    }

    @Override
    public FlatVectorsScorer getFlatVectorScorer(String field) throws IOException {
        return delegateFlatVectorsReader.getFlatVectorScorer(field);
    }

}

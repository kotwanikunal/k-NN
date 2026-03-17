/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine.lucene;

import org.opensearch.Version;
import org.opensearch.common.ValidationException;
import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.SpaceType;
import org.opensearch.knn.index.VectorDataType;
import org.opensearch.knn.index.engine.KNNEngine;
import org.opensearch.knn.index.engine.KNNMethodConfigContext;
import org.opensearch.knn.index.engine.KNNMethodContext;
import org.opensearch.knn.index.engine.MethodComponentContext;
import org.opensearch.knn.index.engine.MethodResolver;
import org.opensearch.knn.index.engine.ResolvedMethodContext;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.Mode;

import java.util.Map;

import static org.opensearch.knn.common.KNNConstants.ENCODER_SQ;
import static org.opensearch.knn.common.KNNConstants.METHOD_ENCODER_PARAMETER;
import static org.opensearch.knn.common.KNNConstants.METHOD_HNSW;

public class LuceneHNSWMethodResolverTests extends KNNTestCase {
    MethodResolver TEST_RESOLVER = new LuceneHNSWMethodResolver();

    public void testResolveMethod_whenValid_thenResolve() {
        ResolvedMethodContext resolvedMethodContext = TEST_RESOLVER.resolveMethod(
            null,
            KNNMethodConfigContext.builder().vectorDataType(VectorDataType.FLOAT).versionCreated(Version.CURRENT).build(),
            false,
            SpaceType.INNER_PRODUCT
        );
        assertEquals(METHOD_HNSW, resolvedMethodContext.getKnnMethodContext().getMethodComponentContext().getName());
        assertFalse(resolvedMethodContext.getKnnMethodContext().getMethodComponentContext().getParameters().isEmpty());
        assertEquals(KNNEngine.LUCENE, resolvedMethodContext.getKnnMethodContext().getKnnEngine());
        assertEquals(SpaceType.INNER_PRODUCT, resolvedMethodContext.getKnnMethodContext().getSpaceType());
        assertEquals(CompressionLevel.x32, resolvedMethodContext.getCompressionLevel());

        resolvedMethodContext = TEST_RESOLVER.resolveMethod(
            null,
            KNNMethodConfigContext.builder()
                .vectorDataType(VectorDataType.FLOAT)
                .versionCreated(Version.CURRENT)
                .mode(Mode.ON_DISK)
                .build(),
            false,
            SpaceType.INNER_PRODUCT
        );
        assertEquals(METHOD_HNSW, resolvedMethodContext.getKnnMethodContext().getMethodComponentContext().getName());
        assertFalse(resolvedMethodContext.getKnnMethodContext().getMethodComponentContext().getParameters().isEmpty());
        assertTrue(
            resolvedMethodContext.getKnnMethodContext().getMethodComponentContext().getParameters().containsKey(METHOD_ENCODER_PARAMETER)
        );
        assertEquals(KNNEngine.LUCENE, resolvedMethodContext.getKnnMethodContext().getKnnEngine());
        assertEquals(SpaceType.INNER_PRODUCT, resolvedMethodContext.getKnnMethodContext().getSpaceType());
        assertEquals(CompressionLevel.x32, resolvedMethodContext.getCompressionLevel());

        resolvedMethodContext = TEST_RESOLVER.resolveMethod(
            null,
            KNNMethodConfigContext.builder()
                .vectorDataType(VectorDataType.FLOAT)
                .versionCreated(Version.CURRENT)
                .compressionLevel(CompressionLevel.x4)
                .build(),
            false,
            SpaceType.INNER_PRODUCT
        );
        assertEquals(METHOD_HNSW, resolvedMethodContext.getKnnMethodContext().getMethodComponentContext().getName());
        assertFalse(resolvedMethodContext.getKnnMethodContext().getMethodComponentContext().getParameters().isEmpty());
        assertTrue(
            resolvedMethodContext.getKnnMethodContext().getMethodComponentContext().getParameters().containsKey(METHOD_ENCODER_PARAMETER)
        );
        assertEquals(KNNEngine.LUCENE, resolvedMethodContext.getKnnMethodContext().getKnnEngine());
        assertEquals(SpaceType.INNER_PRODUCT, resolvedMethodContext.getKnnMethodContext().getSpaceType());
        assertEquals(CompressionLevel.x4, resolvedMethodContext.getCompressionLevel());

        KNNMethodContext knnMethodContext = new KNNMethodContext(
            KNNEngine.LUCENE,
            SpaceType.INNER_PRODUCT,
            new MethodComponentContext(METHOD_HNSW, Map.of())
        );
        resolvedMethodContext = TEST_RESOLVER.resolveMethod(
            knnMethodContext,
            KNNMethodConfigContext.builder()
                .vectorDataType(VectorDataType.FLOAT)
                .versionCreated(Version.CURRENT)
                .mode(Mode.ON_DISK)
                .build(),
            false,
            SpaceType.INNER_PRODUCT
        );
        assertEquals(METHOD_HNSW, resolvedMethodContext.getKnnMethodContext().getMethodComponentContext().getName());
        assertFalse(resolvedMethodContext.getKnnMethodContext().getMethodComponentContext().getParameters().isEmpty());
        assertTrue(
            resolvedMethodContext.getKnnMethodContext().getMethodComponentContext().getParameters().containsKey(METHOD_ENCODER_PARAMETER)
        );
        assertEquals(KNNEngine.LUCENE, resolvedMethodContext.getKnnMethodContext().getKnnEngine());
        assertEquals(SpaceType.INNER_PRODUCT, resolvedMethodContext.getKnnMethodContext().getSpaceType());
        assertEquals(CompressionLevel.x32, resolvedMethodContext.getCompressionLevel());
        assertNotEquals(knnMethodContext, resolvedMethodContext.getKnnMethodContext());

        knnMethodContext = new KNNMethodContext(
            KNNEngine.LUCENE,
            SpaceType.INNER_PRODUCT,
            new MethodComponentContext(METHOD_HNSW, Map.of())
        );
        resolvedMethodContext = TEST_RESOLVER.resolveMethod(
            knnMethodContext,
            KNNMethodConfigContext.builder()
                .vectorDataType(VectorDataType.FLOAT)
                .versionCreated(Version.CURRENT)
                .compressionLevel(CompressionLevel.x4)
                .build(),
            false,
            SpaceType.INNER_PRODUCT
        );
        assertEquals(METHOD_HNSW, resolvedMethodContext.getKnnMethodContext().getMethodComponentContext().getName());
        assertFalse(resolvedMethodContext.getKnnMethodContext().getMethodComponentContext().getParameters().isEmpty());
        assertTrue(
            resolvedMethodContext.getKnnMethodContext().getMethodComponentContext().getParameters().containsKey(METHOD_ENCODER_PARAMETER)
        );
        assertEquals(KNNEngine.LUCENE, resolvedMethodContext.getKnnMethodContext().getKnnEngine());
        assertEquals(SpaceType.INNER_PRODUCT, resolvedMethodContext.getKnnMethodContext().getSpaceType());
        assertEquals(CompressionLevel.x4, resolvedMethodContext.getCompressionLevel());
        assertNotEquals(knnMethodContext, resolvedMethodContext.getKnnMethodContext());

        knnMethodContext = new KNNMethodContext(
            KNNEngine.LUCENE,
            SpaceType.INNER_PRODUCT,
            new MethodComponentContext(METHOD_HNSW, Map.of(METHOD_ENCODER_PARAMETER, new MethodComponentContext(ENCODER_SQ, Map.of())))
        );
        resolvedMethodContext = TEST_RESOLVER.resolveMethod(
            knnMethodContext,
            KNNMethodConfigContext.builder().vectorDataType(VectorDataType.FLOAT).versionCreated(Version.CURRENT).build(),
            false,
            SpaceType.INNER_PRODUCT
        );
        assertEquals(METHOD_HNSW, resolvedMethodContext.getKnnMethodContext().getMethodComponentContext().getName());
        assertFalse(resolvedMethodContext.getKnnMethodContext().getMethodComponentContext().getParameters().isEmpty());
        assertTrue(
            resolvedMethodContext.getKnnMethodContext().getMethodComponentContext().getParameters().containsKey(METHOD_ENCODER_PARAMETER)
        );
        assertEquals(KNNEngine.LUCENE, resolvedMethodContext.getKnnMethodContext().getKnnEngine());
        assertEquals(SpaceType.INNER_PRODUCT, resolvedMethodContext.getKnnMethodContext().getSpaceType());
        assertEquals(CompressionLevel.x4, resolvedMethodContext.getCompressionLevel());
        assertNotEquals(knnMethodContext, resolvedMethodContext.getKnnMethodContext());

        resolvedMethodContext = TEST_RESOLVER.resolveMethod(
            null,
            KNNMethodConfigContext.builder().vectorDataType(VectorDataType.BYTE).versionCreated(Version.CURRENT).build(),
            false,
            SpaceType.INNER_PRODUCT
        );
        assertEquals(METHOD_HNSW, resolvedMethodContext.getKnnMethodContext().getMethodComponentContext().getName());
        assertFalse(resolvedMethodContext.getKnnMethodContext().getMethodComponentContext().getParameters().isEmpty());
        assertFalse(
            resolvedMethodContext.getKnnMethodContext().getMethodComponentContext().getParameters().containsKey(METHOD_ENCODER_PARAMETER)
        );
        assertEquals(KNNEngine.LUCENE, resolvedMethodContext.getKnnMethodContext().getKnnEngine());
        assertEquals(SpaceType.INNER_PRODUCT, resolvedMethodContext.getKnnMethodContext().getSpaceType());
        assertEquals(CompressionLevel.x1, resolvedMethodContext.getCompressionLevel());
    }

    public void testResolveMethod_whenInvalid_thenThrow() {
        // Invalid training context
        expectThrows(
            ValidationException.class,
            () -> TEST_RESOLVER.resolveMethod(
                null,
                KNNMethodConfigContext.builder().vectorDataType(VectorDataType.FLOAT).versionCreated(Version.CURRENT).build(),
                true,
                SpaceType.L2
            )
        );

        // Invalid spec ondisk and compression is 1
        expectThrows(
            ValidationException.class,
            () -> TEST_RESOLVER.resolveMethod(
                null,
                KNNMethodConfigContext.builder()
                    .vectorDataType(VectorDataType.FLOAT)
                    .mode(Mode.ON_DISK)
                    .compressionLevel(CompressionLevel.x1)
                    .versionCreated(Version.CURRENT)
                    .build(),
                false,
                SpaceType.L2
            )
        );
    }

    public void testResolveMethod_whenPreV360_thenUseLegacyDefaults() {
        // Pre-3.6.0: minimal config should resolve to x1 with no encoder
        ResolvedMethodContext resolvedMethodContext = TEST_RESOLVER.resolveMethod(
            null,
            KNNMethodConfigContext.builder().vectorDataType(VectorDataType.FLOAT).versionCreated(Version.V_3_5_0).build(),
            false,
            SpaceType.L2
        );
        assertEquals(CompressionLevel.x1, resolvedMethodContext.getCompressionLevel());
        assertFalse(
            resolvedMethodContext.getKnnMethodContext().getMethodComponentContext().getParameters().containsKey(METHOD_ENCODER_PARAMETER)
        );

        // Pre-3.6.0: ON_DISK should resolve to x4 with SQ encoder (not BBQ)
        resolvedMethodContext = TEST_RESOLVER.resolveMethod(
            null,
            KNNMethodConfigContext.builder()
                .vectorDataType(VectorDataType.FLOAT)
                .mode(Mode.ON_DISK)
                .versionCreated(Version.V_3_5_0)
                .build(),
            false,
            SpaceType.L2
        );
        assertEquals(CompressionLevel.x4, resolvedMethodContext.getCompressionLevel());
        assertEquals(
            ENCODER_SQ,
            ((MethodComponentContext) resolvedMethodContext.getKnnMethodContext()
                .getMethodComponentContext()
                .getParameters()
                .get(METHOD_ENCODER_PARAMETER)).getName()
        );
    }

    public void testResolveMethod_whenV360DefaultConfig_thenModeIsOnDisk() {
        KNNMethodConfigContext configContext = KNNMethodConfigContext.builder()
            .vectorDataType(VectorDataType.FLOAT)
            .versionCreated(Version.CURRENT)
            .build();
        TEST_RESOLVER.resolveMethod(null, configContext, false, SpaceType.L2);
        assertEquals(Mode.ON_DISK, configContext.getMode());
    }

    public void testResolveMethod_whenPreV360DefaultConfig_thenModeNotChanged() {
        KNNMethodConfigContext configContext = KNNMethodConfigContext.builder()
            .vectorDataType(VectorDataType.FLOAT)
            .versionCreated(Version.V_3_5_0)
            .build();
        TEST_RESOLVER.resolveMethod(null, configContext, false, SpaceType.L2);
        assertEquals(Mode.NOT_CONFIGURED, configContext.getMode());
    }

    public void testResolveMethod_whenV360WithExplicitX4_thenUseSQ() {
        // Explicit x4 compression on 3.6.0+ should still use SQ encoder
        ResolvedMethodContext resolvedMethodContext = TEST_RESOLVER.resolveMethod(
            null,
            KNNMethodConfigContext.builder()
                .vectorDataType(VectorDataType.FLOAT)
                .compressionLevel(CompressionLevel.x4)
                .versionCreated(Version.CURRENT)
                .build(),
            false,
            SpaceType.L2
        );
        assertEquals(CompressionLevel.x4, resolvedMethodContext.getCompressionLevel());
        assertEquals(
            ENCODER_SQ,
            ((MethodComponentContext) resolvedMethodContext.getKnnMethodContext()
                .getMethodComponentContext()
                .getParameters()
                .get(METHOD_ENCODER_PARAMETER)).getName()
        );
    }
}

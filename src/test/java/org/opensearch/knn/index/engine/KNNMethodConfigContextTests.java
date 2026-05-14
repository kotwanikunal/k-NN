/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.engine;

import org.opensearch.knn.KNNTestCase;
import org.opensearch.knn.index.mapper.CompressionLevel;
import org.opensearch.knn.index.mapper.Mode;

public class KNNMethodConfigContextTests extends KNNTestCase {

    public void testDeriveMode_whenNotConfigured_thenNotConfigured() {
        assertEquals(Mode.NOT_CONFIGURED, KNNMethodConfigContext.deriveMode(CompressionLevel.NOT_CONFIGURED));
    }

    public void testDeriveMode_whenX1_thenInMemory() {
        assertEquals(Mode.IN_MEMORY, KNNMethodConfigContext.deriveMode(CompressionLevel.x1));
    }

    public void testDeriveMode_whenX2_thenInMemory() {
        assertEquals(Mode.IN_MEMORY, KNNMethodConfigContext.deriveMode(CompressionLevel.x2));
    }

    public void testDeriveMode_whenX4_thenOnDisk() {
        assertEquals(Mode.ON_DISK, KNNMethodConfigContext.deriveMode(CompressionLevel.x4));
    }

    public void testDeriveMode_whenX16_thenOnDisk() {
        assertEquals(Mode.ON_DISK, KNNMethodConfigContext.deriveMode(CompressionLevel.x16));
    }

    public void testDeriveMode_whenX32_thenOnDisk() {
        assertEquals(Mode.ON_DISK, KNNMethodConfigContext.deriveMode(CompressionLevel.x32));
    }
}

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn;

import java.util.List;

/**
 * Direct I/O on or off, and how big its store may be, as an operator would set them. One arm of a desktop
 * measurement is one of these applied to a running cluster.
 *
 * <p>The three are declared once, here, because more than one measurement compares the same three routes and
 * they must not drift apart: {@code DirectIONestedTestCase} and {@code DirectIODerivedSourceTestCase} both
 * read them.
 */
public enum DirectIOArm {
    MMAP("mmap", false, "0b"),
    DIO_0B("dio-0b", true, "0b"),
    DIO_CACHE("dio-cache", true, "64mb");

    /**
     * The arms in the order they must run: the delegate first, so it is the reference the two Direct I/O arms
     * are compared against, and the sized store last, so it starts from a store the 0 B arm has released.
     */
    public static final List<DirectIOArm> ORDER = List.of(MMAP, DIO_0B, DIO_CACHE);

    private final String label;
    private final boolean enabled;
    private final String cacheMaxBytes;

    DirectIOArm(final String label, final boolean enabled, final String cacheMaxBytes) {
        this.label = label;
        this.enabled = enabled;
        this.cacheMaxBytes = cacheMaxBytes;
    }

    /** The arm's name in a log line and a table row. */
    public String label() {
        return label;
    }

    /** Whether this arm puts the {@code .vec} on the {@code O_DIRECT} route. */
    public boolean routes() {
        return enabled;
    }

    /** What {@code knn.direct_io.rescore.cache.max_bytes} is set to for this arm. */
    public String cacheMaxBytes() {
        return cacheMaxBytes;
    }
}

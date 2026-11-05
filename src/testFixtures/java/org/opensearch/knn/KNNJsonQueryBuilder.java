/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn;

import lombok.Builder;
import lombok.NonNull;
import org.opensearch.common.xcontent.XContentFactory;
import org.opensearch.core.xcontent.XContentBuilder;

import java.io.IOException;

/**
 * Helper method to create knn query in json string
 * Here, we don't use any of the internal class so that it can mimic user request more closely.
 */
@Builder
public class KNNJsonQueryBuilder {
    @NonNull
    private String fieldName;
    @NonNull
    private Object[] vector;
    private Integer k;
    private Float minScore;
    private Float maxDistance;
    private String nestedFieldName;
    private String filterFieldName;
    private String filterValue;
    /** A filter on a child field, which goes into the k-NN filter as a nested query over {@code nestedFieldName}. */
    private String childFilterFieldName;
    private Boolean rescoreEnabled;
    private Float oversampleFactor;
    private Boolean expandNested;
    private Integer innerHitsSize;
    private String scoreMode;

    public String getQueryString() throws IOException {
        XContentBuilder builder = XContentFactory.jsonBuilder().startObject().startObject("query");
        if (nestedFieldName != null) {
            builder.startObject("nested");
            builder.field("path", nestedFieldName);
            if (scoreMode != null) {
                builder.field("score_mode", scoreMode);
            }
            builder.startObject("query");
            builder.startObject("knn");
            builder.startObject(nestedFieldName + "." + fieldName);
        } else {
            builder.startObject("knn");
            builder.startObject(fieldName);
        }

        builder.field("vector", vector);
        if (k != null) {
            builder.field("k", k);
        }
        if (minScore != null) {
            builder.field("min_score", minScore);
        }
        if (maxDistance != null) {
            builder.field("max_distance", maxDistance);
        }
        if (expandNested != null) {
            builder.field("expand_nested_docs", expandNested);
        }

        if (filterValue != null && (filterFieldName != null || childFilterFieldName != null)) {
            builder.startObject("filter");
            if (childFilterFieldName != null) {
                builder.startObject("nested").field("path", nestedFieldName).startObject("query");
                builder.startObject("term").field(childFilterFieldName, filterValue).endObject();
                builder.endObject().endObject();
            } else {
                builder.startObject("term").field(filterFieldName, filterValue).endObject();
            }
            builder.endObject();
        }

        // One rescore key either way: a boolean turns the whole stage on or off, an oversample factor configures it.
        if (oversampleFactor != null) {
            builder.startObject("rescore").field("oversample_factor", oversampleFactor).endObject();
        } else if (rescoreEnabled != null) {
            builder.field("rescore", rescoreEnabled);
        }

        // the vector field object, then the knn object
        builder.endObject().endObject();
        if (nestedFieldName != null) {
            // the query inside the nested object, then the inner hits beside it, then the nested object
            builder.endObject();
            if (innerHitsSize != null) {
                builder.field("inner_hits").startObject().field("size", innerHitsSize).endObject();
            }
            builder.endObject();
        }
        // the outer query, then the root
        builder.endObject().endObject();

        return builder.toString();
    }
}

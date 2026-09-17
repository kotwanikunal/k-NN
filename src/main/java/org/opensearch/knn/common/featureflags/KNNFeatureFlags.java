/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 *
 */

package org.opensearch.knn.common.featureflags;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import lombok.experimental.UtilityClass;
import org.opensearch.common.Booleans;
import org.opensearch.common.settings.Setting;
import org.opensearch.knn.index.KNNSettings;

import java.util.List;

import static org.opensearch.common.settings.Setting.Property.Dynamic;
import static org.opensearch.common.settings.Setting.Property.NodeScope;

/**
 * Class to manage KNN feature flags
 */
@UtilityClass
public class KNNFeatureFlags {

    // Feature flags
    private static final String KNN_FORCE_EVICT_CACHE_ENABLED = "knn.feature.cache.force_evict.enabled";
    private static final String KNN_PREFETCH_ENABLED = "knn.feature.prefetch.enabled";
    private static final boolean KNN_PREFETCH_ENABLED_DEFAULT_VALUE = true;
    private static final String KNN_DIRECT_IO_ENABLED = "knn.feature.direct_io.enabled";
    private static final boolean KNN_DIRECT_IO_ENABLED_DEFAULT_VALUE = true;

    @VisibleForTesting
    public static final Setting<Boolean> KNN_FORCE_EVICT_CACHE_ENABLED_SETTING = Setting.boolSetting(
        KNN_FORCE_EVICT_CACHE_ENABLED,
        false,
        NodeScope,
        Dynamic
    );

    public static final Setting<Boolean> KNN_PREFETCH_ENABLED_SETTING = Setting.boolSetting(
        KNN_PREFETCH_ENABLED,
        KNN_PREFETCH_ENABLED_DEFAULT_VALUE,
        NodeScope,
        Dynamic
    );

    /**
     * Node level kill switch for the "knn_direct_io" store type. Because the Directory is built when a
     * shard opens, flipping this flag takes effect on the next close and reopen of an index, with no
     * node restart.
     */
    public static final Setting<Boolean> KNN_DIRECT_IO_ENABLED_SETTING = Setting.boolSetting(
        KNN_DIRECT_IO_ENABLED,
        KNN_DIRECT_IO_ENABLED_DEFAULT_VALUE,
        NodeScope,
        Dynamic
    );

    /**
     * All feature flags which needs to be provided as setting should be added here.
     * @return List of Feature flag settings
     */
    public static List<Setting<?>> getFeatureFlags() {
        return ImmutableList.of(KNN_FORCE_EVICT_CACHE_ENABLED_SETTING, KNN_PREFETCH_ENABLED_SETTING, KNN_DIRECT_IO_ENABLED_SETTING);
    }

    /**
     * All feature flags which when changed should trigger a cache rebuild
     * @return List of Feature flag settings
     */
    public static List<Setting<?>> getFeatureFlagsWhichRebuildsCache() {
        return ImmutableList.of(KNN_FORCE_EVICT_CACHE_ENABLED_SETTING);
    }

    /**
     * Checks if force evict for cache is enabled by executing a check against cluster settings
     * @return true if force evict setting is set to true
     */
    public static boolean isForceEvictCacheEnabled() {
        return Booleans.parseBoolean(KNNSettings.state().getSettingValue(KNN_FORCE_EVICT_CACHE_ENABLED).toString(), false);
    }

    public static boolean isPrefetchEnabled() {
        return Booleans.parseBoolean(
            KNNSettings.state().getSettingValue(KNN_PREFETCH_ENABLED).toString(),
            KNN_PREFETCH_ENABLED_DEFAULT_VALUE
        );
    }

    /**
     * Checks the node level Direct I/O kill switch. Unlike the other flags this one is read while a
     * shard's Directory is being built, which can happen before {@link KNNSettings} has a
     * {@code ClusterService} - so an unreadable setting falls back to the default rather than
     * propagating and failing shard open.
     *
     * @return true if Direct I/O may be used on this node
     */
    public static boolean isDirectIOEnabled() {
        try {
            return Booleans.parseBoolean(
                KNNSettings.state().getSettingValue(KNN_DIRECT_IO_ENABLED).toString(),
                KNN_DIRECT_IO_ENABLED_DEFAULT_VALUE
            );
        } catch (Exception e) {
            return KNN_DIRECT_IO_ENABLED_DEFAULT_VALUE;
        }
    }
}

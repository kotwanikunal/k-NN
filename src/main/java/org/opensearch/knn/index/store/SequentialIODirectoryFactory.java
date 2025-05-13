/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import org.apache.lucene.store.Directory;
import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.common.settings.Settings;
import org.opensearch.index.IndexModule;
import org.opensearch.index.IndexSettings;
import org.opensearch.index.shard.ShardPath;
import org.opensearch.index.store.FsDirectoryFactory;

import java.io.IOException;

public class SequentialIODirectoryFactory extends FsDirectoryFactory {

    public SequentialIODirectoryFactory() {
        super();
    }

    public Directory newDirectory(IndexSettings indexSettings, ShardPath shardPath) throws IOException {
        // Override the settings here for the FSDirectory calls to work correctly
        // It rechecks the STORE TYPE setting

        Settings settings = Settings.builder()
            .put(indexSettings.getSettings())
            .put(IndexModule.INDEX_STORE_TYPE_SETTING.getKey(), IndexModule.Type.FS.getSettingsKey())
            .build();
        Settings metadataSettings = Settings.builder()
            .put(indexSettings.getIndexMetadata().getSettings())
            .put(IndexModule.INDEX_STORE_TYPE_SETTING.getKey(), IndexModule.Type.FS.getSettingsKey())
            .build();
        IndexMetadata indexMetadata = IndexMetadata.builder(indexSettings.getIndexMetadata()).settings(metadataSettings).build();
        IndexSettings indexSettings1 = new IndexSettings(indexMetadata, settings);

        Directory directory = super.newDirectory(indexSettings1, shardPath);
        return new SequentialIODirectory(directory);
    }
}

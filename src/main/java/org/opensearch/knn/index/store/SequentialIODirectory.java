/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.index.store;

import org.apache.lucene.store.Directory;
import org.apache.lucene.store.IOContext;
import org.apache.lucene.store.IndexInput;
import org.apache.lucene.store.IndexOutput;
import org.apache.lucene.store.Lock;

import java.io.IOException;
import java.util.Collection;
import java.util.Set;

public class SequentialIODirectory extends Directory {

    private final Directory delegateDirectory;

    public SequentialIODirectory(Directory directory) {
        this.delegateDirectory = directory;
    }

    @Override
    public String[] listAll() throws IOException {
        return delegateDirectory.listAll();
    }

    @Override
    public void deleteFile(String s) throws IOException {
        delegateDirectory.deleteFile(s);
    }

    @Override
    public long fileLength(String s) throws IOException {
        return delegateDirectory.fileLength(s);
    }

    @Override
    public IndexOutput createOutput(String s, IOContext ioContext) throws IOException {
        return delegateDirectory.createOutput(s, ioContext);
    }

    @Override
    public IndexOutput createTempOutput(String s, String s1, IOContext ioContext) throws IOException {
        return delegateDirectory.createTempOutput(s, s1, ioContext);
    }

    @Override
    public void sync(Collection<String> collection) throws IOException {
        delegateDirectory.sync(collection);
    }

    @Override
    public void syncMetaData() throws IOException {
        delegateDirectory.syncMetaData();
    }

    @Override
    public void rename(String s, String s1) throws IOException {
        delegateDirectory.rename(s, s1);
    }

    @Override
    public IndexInput openInput(String s, IOContext ioContext) throws IOException {
        // OVERRIDE HERE FOR SEQUENTIAL READS
        return delegateDirectory.openInput(s, IOContext.READ);
    }

    @Override
    public Lock obtainLock(String s) throws IOException {
        return delegateDirectory.obtainLock(s);
    }

    @Override
    public void close() throws IOException {
        delegateDirectory.close();
    }

    @Override
    public Set<String> getPendingDeletions() throws IOException {
        return delegateDirectory.getPendingDeletions();
    }
}

/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.jni;

import org.opensearch.knn.common.KNNConstants;

import java.security.AccessController;
import java.security.PrivilegedAction;

import static org.opensearch.knn.index.KNNSettings.*;
import static org.opensearch.knn.jni.PlatformUtils.*;

public class NativeLuceneService {
    static {
        AccessController.doPrivileged((PrivilegedAction<Void>) () -> {

            // Even if the underlying system supports AVX512 and AVX2, users can override and disable it by setting
            // 'knn.faiss.avx2.disabled', 'knn.faiss.avx512.disabled', or 'knn.faiss.avx512_spr.disabled' to true in the opensearch.yml
            // configuration
            System.loadLibrary(KNNConstants.NATIVE_LUCENE_JNI_LIBRARY_NAME);

            return null;
        });
    }

    public static native float innerProductScaledNativeOffHeapPinnedQuery(long queryVectorAddress, long inputVectorAddress, long dimension);

    public static native long allocatePinnedQueryVector(float[] queryVector, long dimension);

    public static native void deallocatePinnedQueryVector(long queryVectorAddress);
}

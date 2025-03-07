#include <jni.h>
#include "org_opensearch_knn_jni_NativeLuceneService.h"

JNIEXPORT jlong JNICALL Java_org_opensearch_knn_jni_NativeLuceneService_allocatePinnedQueryVector(JNIEnv * env, jclass cls,
jfloatArray queryVector, jlong dimension) {
    auto * addr = new float[dimension];
    env->GetFloatArrayRegion(queryVector, 0, dimension, addr);
    return reinterpret_cast<jlong>(addr);
}

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_NativeLuceneService_deallocatePinnedQueryVector(JNIEnv * env, jclass cls,
jlong address) {
    auto * addr = reinterpret_cast<float*>(address);
    delete[] addr;
}

JNIEXPORT jfloat JNICALL Java_org_opensearch_knn_jni_NativeLuceneService_innerProductScaledNativeOffHeapPinnedQuery
  (JNIEnv *env, jclass cls, jlong queryAddr, jlong address, jlong dimension) {
    jfloat *queryArr = reinterpret_cast<jfloat*>(queryAddr);
    jfloat *inputArr = reinterpret_cast<jfloat*>(address);

    jsize length = (jsize) dimension;

    float sum = 0.0f;

    for (jsize i = 0; i < length; i++) {
        // FMA pattern: sum = sum + (queryArr[i] * inputArr[i])
        sum += queryArr[i] * inputArr[i];
//        sum = std::fma(queryArr[i], inputArr[i], sum);
    }

    // scale due to lucene restrictions
    sum = (sum < 0.0f) ? (1.0f / (1.0f - sum)) : (sum + 1.0f);

    return sum;
  }
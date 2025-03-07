#include <jni.h>

#ifndef _Included_org_opensearch_knn_jni_NativeLuceneService
#define _Included_org_opensearch_knn_jni_NativeLuceneService
#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT jlong JNICALL Java_org_opensearch_knn_jni_NativeLuceneService_allocatePinnedQueryVector(JNIEnv * env, jclass cls,
jfloatArray queryVector, jlong dimension);

JNIEXPORT void JNICALL Java_org_opensearch_knn_jni_NativeLuceneService_deallocatePinnedQueryVector(JNIEnv * env, jclass cls,
jlong address);

JNIEXPORT jfloat JNICALL Java_org_opensearch_knn_jni_NativeLuceneService_innerProductScaledNativeOffHeapPinnedQuery
  (JNIEnv *env, jclass cls, jlong queryAddr, jlong address, jlong dimension);

 #ifdef __cplusplus
}
#endif
#endif
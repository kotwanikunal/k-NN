#include <jni.h>
#include <cmath>
#include <stdexcept>
#include <arm_neon.h>
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

    // Calculate dot product
    /* for (jsize i = 0; i < length; i++) {
        float diff = queryArr[i] - inputArr[i];
        sum = std::fma(diff, diff, sum);
    }

    // Scale using Lucene's exact formula
    float result = 1 / (1 + sum);

    return result;*/

    #if defined(__AVX__) || defined(__AVX2__)  // AVX2 on x86_64 (Intel, AMD)
        __m256 sumVec = _mm256_setzero_ps();
        jsize i = 0;
        for (; i <= length - 8; i += 8) { // Process 8 floats per iteration
            __m256 queryVec = _mm256_loadu_ps(&queryArr[i]);
            __m256 inputVec = _mm256_loadu_ps(&inputArr[i]);
            __m256 diffVec = _mm256_sub_ps(queryVec, inputVec);
            sumVec = _mm256_fmadd_ps(diffVec, diffVec, sumVec);
        }
        // Horizontal sum of sumVec
        float temp[8];
        _mm256_storeu_ps(temp, sumVec);
        for (int j = 0; j < 8; j++) sum += temp[j];

    #elif defined(__ARM_NEON)  // NEON on ARM (Graviton, Apple M-series)
        float32x4_t sumVec = vdupq_n_f32(0.0f);
        jsize i = 0;
        for (; i <= length - 4; i += 4) { // Process 4 floats per iteration
            float32x4_t queryVec = vld1q_f32(&queryArr[i]);
            float32x4_t inputVec = vld1q_f32(&inputArr[i]);
            float32x4_t diffVec = vsubq_f32(queryVec, inputVec);
            sumVec = vfmaq_f32(sumVec, diffVec, diffVec);
        }
        // Reduce sumVec to scalar
        sum += vaddvq_f32(sumVec);

    #else  // Scalar fallback (if no SIMD support)
        jsize i = 0;
    #endif

        // Handle remaining elements (if any)
        for (; i < length; i++) {
            float diff = queryArr[i] - inputArr[i];
            sum = std::fma(diff, diff, sum);
        }

        // Scale using Lucene's formula
        float result = 1 / (1 + sum);
        return result;
  }
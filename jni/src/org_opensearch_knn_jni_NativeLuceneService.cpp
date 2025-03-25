#include <jni.h>
#include <cmath>
#include <stdexcept>
#if defined(__AVX__) || defined(__AVX2__)  // AVX for Intel/AMD (x86_64)
#include <immintrin.h>
#elif defined(__ARM_NEON)  // NEON for ARM (Graviton, Apple M-series)
#include <arm_neon.h>
#endif
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
    jsize length = static_cast<jsize>(dimension);

    float sum = 0.0f;

    #if defined(__AVX__) || defined(__AVX2__)  // AVX2 on Intel/AMD
        __m256 sumVec = _mm256_setzero_ps();
        jsize i = 0;
        for (; i <= length - 8; i += 8) {  // Process 8 floats per iteration
            __m256 queryVec = _mm256_loadu_ps(&queryArr[i]);
            __m256 inputVec = _mm256_loadu_ps(&inputArr[i]);
            sumVec = _mm256_fmadd_ps(queryVec, inputVec, sumVec);
        }
        // Reduce the vector sum to a scalar
        float temp[8];
        _mm256_storeu_ps(temp, sumVec);
        for (int j = 0; j < 8; j++) sum += temp[j];

    #elif defined(__ARM_NEON)  // NEON on ARM (Graviton, Apple M-series)
        float32x4_t sumVec = vdupq_n_f32(0.0f);
        jsize i = 0;
        for (; i <= length - 4; i += 4) {  // Process 4 floats per iteration
            float32x4_t queryVec = vld1q_f32(&queryArr[i]);
            float32x4_t inputVec = vld1q_f32(&inputArr[i]);
            sumVec = vfmaq_f32(sumVec, queryVec, inputVec);
        }
        // Reduce sumVec to scalar
        sum += vaddvq_f32(sumVec);

    #else  // Scalar fallback (if no SIMD support)
        jsize i = 0;
    #endif

        // Handle remaining elements
        for (; i < length; i++) {
            sum = std::fma(queryArr[i], inputArr[i], sum);
        }

        // Scale using Lucene's exact formula
        float result = (sum < 0) ? (1 / (1 + (-1 * sum))) : (sum + 1);
        return result;
}
/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 *
 * Modifications Copyright OpenSearch Contributors. See
 * GitHub history for details.
 */

#ifndef OPENSEARCH_KNN_COMMONS_H
#define OPENSEARCH_KNN_COMMONS_H

#include "jni_util.h"
#include <jni.h>
namespace knn_jni {
    namespace commons {
        /**
         * This is utility function that can be used to store data in native memory. This function will allocate memory for
         * the data(rows*columns) with initialCapacity and return the memory address where the data is stored.
         * If you are using this function for first time use memoryAddress = 0 to ensure that a new memory location is created.
         * For subsequent calls you can pass the same memoryAddress. If the data cannot be stored in the memory location
         * will throw Exception.
         *
         * append tells the method to keep appending to the existing vector. Passing the value as false will clear the vector
         * without reallocating new memory. This helps with reducing memory frangmentation and overhead of allocating
         * and deallocating when the memory address needs to be reused.
         *
         * CAUTION: The behavior is undefined if the memory address is deallocated and the method is called
         *
         * @param memoryAddress The address of the memory location where data will be stored.
         * @param data 2D float array containing data to be stored in native memory.
         * @param initialCapacity The initial capacity of the memory location.
         * @param append whether to append or start from index 0 when called subsequently with the same address
         * @return memory address of std::vector<float> where the data is stored.
         */
        jlong storeVectorData(knn_jni::JNIUtilInterface *, JNIEnv *, jlong , jobjectArray, jlong, jboolean);

        /**
         * This is utility function that can be used to store data in native memory. This function will allocate memory for
         * the data(rows*columns) with initialCapacity and return the memory address where the data is stored.
         * If you are using this function for first time use memoryAddress = 0 to ensure that a new memory location is created.
         * For subsequent calls you can pass the same memoryAddress. If the data cannot be stored in the memory location
         * will throw Exception.
         *
         *  append tells the method to keep appending to the existing vector. Passing the value as false will clear the vector
         * without reallocating new memory. This helps with reducing memory frangmentation and overhead of allocating
         * and deallocating when the memory address needs to be reused.
         *
         * CAUTION: The behavior is undefined if the memory address is deallocated and the method is called
         *
         * @param memoryAddress The address of the memory location where data will be stored.
         * @param data 2D byte array containing binary data to be stored in native memory.
         * @param initialCapacity The initial capacity of the memory location.
         * @param append whether to append or start from index 0 when called subsequently with the same address
         * @return memory address of std::vector<uint8_t> where the data is stored.
         */
        jlong storeBinaryVectorData(knn_jni::JNIUtilInterface *, JNIEnv *, jlong , jobjectArray, jlong, jboolean);

        /**
        * This is utility function that can be used to store signed int8 data in native memory. This function will allocate memory for
        * the data(rows*columns) with initialCapacity and return the memory address where the data is stored.
        * If you are using this function for first time use memoryAddress = 0 to ensure that a new memory location is created.
        * For subsequent calls you can pass the same memoryAddress. If the data cannot be stored in the memory location
        * will throw Exception.
        *
        * @param memoryAddress The address of the memory location where data will be stored.
        * @param data 2D byte array containing int8 data to be stored in native memory.
        * @param initialCapacity The initial capacity of the memory location.
        * @param append whether to append or start from index 0 when called subsequently with the same address
        * @return memory address of std::vector<int8_t> where the data is stored.
        */
        jlong storeByteVectorData(knn_jni::JNIUtilInterface *, JNIEnv *, jlong , jobjectArray, jlong, jboolean);

        /**
         * Free up the memory allocated for the data stored in memory address. This function should be used with the memory
         * address returned by {@link JNICommons#storeVectorData(long, float[][], long, long)}
         *
         * @param memoryAddress address to be freed.
         */
        void freeVectorData(jlong);

        /**
         * Free up the memory allocated for the data stored in memory address. This function should be used with the memory
         * address returned by {@link JNICommons#storeByteVectorData(long, byte[][], long, long)}
         *
         * @param memoryAddress address to be freed.
         */
        void freeByteVectorData(jlong);

        /**
         * Free up the memory allocated for the data stored in memory address. This function should be used with the memory
         * address returned by {@link JNICommons#storeBinaryVectorData(long, byte[][], long, long)}
         *
         * @param memoryAddress address to be freed.
         */
        void freeBinaryVectorData(jlong);

        /**
         * Allocates one fixed-size off-heap slab for the full-precision vector cache and returns its
         * address, or 0 when the allocation failed.
         *
         * Deliberately a plain malloc rather than a Java direct ByteBuffer: a direct ByteBuffer counts
         * against -XX:MaxDirectMemorySize, which this plugin's Direct I/O staging and netty already share,
         * and it is reclaimed only after a garbage collection. A slab is accounted by k-NN under the
         * native memory budget, the same way a faiss index's malloc memory is, and freeSlab returns it at
         * a point the cache chooses.
         *
         * The memory is NOT zeroed. The cache reads only the slots it has written, and zeroing a 32 MiB
         * slab costs about 6.7 ms on the dev desktop (MEASURED, output/bench/i1) which the cache would
         * then immediately overwrite with vector bytes.
         *
         * @param bytes size of the slab. Must be positive.
         * @return address of the slab, or 0 on failure.
         */
        jlong allocateSlab(jlong bytes);

        /**
         * Frees a slab returned by allocateSlab. A 0 address is ignored, so a double free that has already
         * cleared its address is safe.
         *
         * @param address address returned by allocateSlab.
         */
        void freeSlab(jlong address);

        /**
         * Wraps a slab in a direct java.nio.ByteBuffer over the slab's own memory, so the scorer reads the
         * vector bytes in place rather than through a copy. The buffer carries no cleaner: the slab's
         * address is the only thing that frees the memory, through freeSlab.
         *
         * @param address address returned by allocateSlab.
         * @param bytes the slab's size.
         * @return the direct ByteBuffer, or nullptr when the address is 0.
         */
        jobject wrapSlab(JNIEnv *, jlong address, jlong bytes);

        /**
         * Extracts query time efSearch from method parameters
         **/
        int getIntegerMethodParameter(JNIEnv *, knn_jni::JNIUtilInterface *, std::unordered_map<std::string, jobject>, std::string, int);
    }
}

#endif
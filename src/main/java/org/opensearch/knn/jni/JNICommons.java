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

package org.opensearch.knn.jni;

/**
 * Common class for providing the JNI related functionality to various JNIServices.
 */
public class JNICommons {

    static {
        KNNLibraryLoader.loadCommonLibrary();
    }

    /**
     * This is utility function that can be used to store data in native memory. This function will allocate memory for
     * the data(rows*columns) with initialCapacity and return the memory address where the data is stored.
     * If you are using this function for first time use memoryAddress = 0 to ensure that a new memory location is created.
     * For subsequent calls you can pass the same memoryAddress. If the data cannot be stored in the memory location
     * will throw Exception.
     *
     * <p>
     * The function is not threadsafe. If multiple threads are trying to insert on same memory location, then it can
     * lead to data corruption.
     * </p>
     *
     * @param memoryAddress   The address of the memory location where data will be stored.
     * @param data            2D float array containing data to be stored in native memory.
     * @param initialCapacity The initial capacity of the memory location.
     * @return memory address where the data is stored.
     */
    public static long storeVectorData(long memoryAddress, float[][] data, long initialCapacity) {
        return storeVectorData(memoryAddress, data, initialCapacity, true);
    }

    /**
     * This is utility function that can be used to store data in native memory. This function will allocate memory for
     * the data(rows*columns) with initialCapacity and return the memory address where the data is stored.
     * If you are using this function for first time use memoryAddress = 0 to ensure that a new memory location is created.
     * For subsequent calls you can pass the same memoryAddress. If the data cannot be stored in the memory location
     * will throw Exception.
     *
     * <p>
     * The function is not threadsafe. If multiple threads are trying to insert on same memory location, then it can
     * lead to data corruption.
     * </p>
     *
     * @param memoryAddress   The address of the memory location where data will be stored.
     * @param data            2D float array containing data to be stored in native memory.
     * @param initialCapacity The initial capacity of the memory location.
     * @param append          append the data or rewrite the memory location
     * @return memory address where the data is stored.
     */
    public static native long storeVectorData(long memoryAddress, float[][] data, long initialCapacity, boolean append);

    /**
     * This is utility function that can be used to store binary data in native memory. This function will allocate memory for
     * the data(rows*columns) with initialCapacity and return the memory address where the data is stored.
     * If you are using this function for first time use memoryAddress = 0 to ensure that a new memory location is created.
     * For subsequent calls you can pass the same memoryAddress. If the data cannot be stored in the memory location
     * will throw Exception.
     *
     * <p>
     * The function is not threadsafe. If multiple threads are trying to insert on same memory location, then it can
     * lead to data corruption.
     * </p>
     *
     * @param memoryAddress   The address of the memory location where data will be stored.
     * @param data            2D byte array containing binary data to be stored in native memory.
     * @param initialCapacity The initial capacity of the memory location.
     * @return memory address where the data is stored.
     */
    public static long storeBinaryVectorData(long memoryAddress, byte[][] data, long initialCapacity) {
        return storeBinaryVectorData(memoryAddress, data, initialCapacity, true);
    }

    /**
     * This is utility function that can be used to store binary data in native memory. This function will allocate memory for
     * the data(rows*columns) with initialCapacity and return the memory address where the data is stored.
     * If you are using this function for first time use memoryAddress = 0 to ensure that a new memory location is created.
     * For subsequent calls you can pass the same memoryAddress. If the data cannot be stored in the memory location
     * will throw Exception.
     *
     * <p>
     * The function is not threadsafe. If multiple threads are trying to insert on same memory location, then it can
     * lead to data corruption.
     * </p>
     *
     * @param memoryAddress   The address of the memory location where data will be stored.
     * @param data            2D byte array containing binary data to be stored in native memory.
     * @param initialCapacity The initial capacity of the memory location.
     * @param append          append the data or rewrite the memory location
     * @return memory address where the data is stored.
     */
    public static native long storeBinaryVectorData(long memoryAddress, byte[][] data, long initialCapacity, boolean append);

    /**
     * This is utility function that can be used to store byte data in native memory. This function will allocate memory for
     * the data(rows*columns) with initialCapacity and return the memory address where the data is stored.
     * If you are using this function for first time use memoryAddress = 0 to ensure that a new memory location is created.
     * For subsequent calls you can pass the same memoryAddress. If the data cannot be stored in the memory location
     * will throw Exception.
     *
     * <p>
     * The function is not threadsafe. If multiple threads are trying to insert on same memory location, then it can
     * lead to data corruption.
     * </p>
     *
     * @param memoryAddress   The address of the memory location where data will be stored.
     * @param data            2D byte array containing byte data to be stored in native memory.
     * @param initialCapacity The initial capacity of the memory location.
     * @return memory address where the data is stored.
     */
    public static long storeByteVectorData(long memoryAddress, byte[][] data, long initialCapacity) {
        return storeByteVectorData(memoryAddress, data, initialCapacity, true);
    }

    /**
     * This is utility function that can be used to store byte data in native memory. This function will allocate memory for
     * the data(rows*columns) with initialCapacity and return the memory address where the data is stored.
     * If you are using this function for first time use memoryAddress = 0 to ensure that a new memory location is created.
     * For subsequent calls you can pass the same memoryAddress. If the data cannot be stored in the memory location
     * will throw Exception.
     *
     * <p>
     * The function is not threadsafe. If multiple threads are trying to insert on same memory location, then it can
     * lead to data corruption.
     * </p>
     *
     * @param memoryAddress   The address of the memory location where data will be stored.
     * @param data            2D byte array containing byte data to be stored in native memory.
     * @param initialCapacity The initial capacity of the memory location.
     * @param append          append the data or rewrite the memory location
     * @return memory address where the data is stored.
     */
    public static native long storeByteVectorData(long memoryAddress, byte[][] data, long initialCapacity, boolean append);

    /**
     * Free up the memory allocated for the data stored in memory address. This function should be used with the memory
     * address returned by {@link JNICommons#storeVectorData(long, float[][], long, boolean)}
     *
     * <p>
     * The function is not threadsafe. If multiple threads are trying to free up same memory location, then it can
     * lead to errors.
     * </p>
     *
     * @param memoryAddress address to be freed.
     */
    public static native void freeVectorData(long memoryAddress);

    /**
     * Free up the memory allocated for the binary data stored in memory address. This function should be used with the memory
     * address returned by {@link JNICommons#storeBinaryVectorData(long, byte[][], long)}
     *
     * <p>
     * The function is not threadsafe. If multiple threads are trying to free up same memory location, then it can
     * lead to errors.
     * </p>
     *
     * @param memoryAddress address to be freed.
     */
    public static native void freeBinaryVectorData(long memoryAddress);

    /**
     * Free up the memory allocated for the byte data stored in memory address. This function should be used with the memory
     * address returned by {@link JNICommons#storeByteVectorData(long, byte[][], long)}
     *
     * <p>
     *  The function is not threadsafe. If multiple threads are trying to free up same memory location, then it can
     *  lead to errors.
     * </p>
     *
     * @param memoryAddress address to be freed.
     */
    public static native void freeByteVectorData(long memoryAddress);

    /**
     * Allocates one fixed-size off-heap slab for the full-precision vector cache and returns its address,
     * or {@code 0} when the allocation failed.
     *
     * <p>This is the allocation primitive the vector cache's slabs are built on. It is native rather than
     * a Java direct {@link java.nio.ByteBuffer} for two reasons, both measured:
     * <ul>
     *   <li>A direct ByteBuffer counts against {@code -XX:MaxDirectMemorySize}, which this plugin's
     *       Direct I/O staging and netty already share, and is reclaimed only after a garbage collection.
     *       Phase 15 D3c is the record of what that costs: a gauge measuring churn awaiting collection
     *       rather than memory in use. A slab is accounted by k-NN under the native memory budget, the
     *       same way a faiss index's {@code malloc} memory is, and {@link #freeSlab} returns it at a point
     *       the cache chooses.</li>
     *   <li>The alternative that is also deterministic, an FFM {@code Arena.ofShared()}, cannot reach the
     *       production jar: this project compiles at {@code sourceCompatibility = 21} unconditionally, so
     *       the build excludes every JDK22+ source file. MEASURED in output/bench/i1: FFM also zero-fills,
     *       which costs 6.8 ms per 32 MiB slab that the cache overwrites immediately, against 4.8 us here.
     *       </li>
     * </ul>
     *
     * <p><b>The memory is not zeroed.</b> The cache reads only slots it has written. See
     * {@code VectorCacheSlab} for the accessor that enforces that.
     *
     * @param bytes size of the slab in bytes. Must be positive.
     * @return address of the slab, or {@code 0} on failure.
     */
    public static native long allocateSlab(long bytes);

    /**
     * Frees a slab returned by {@link #allocateSlab}. A {@code 0} address is ignored.
     *
     * <p>Not threadsafe for one address: the caller must serialise the free with every read of the slab.
     * {@code VectorCacheSlab} does that with a read-write lock, the same way
     * {@code NativeMemoryAllocation.IndexAllocation} serialises a graph's free against its searches.
     *
     * @param address address returned by {@link #allocateSlab}.
     */
    public static native void freeSlab(long address);

    /**
     * Wraps a slab in a direct {@link java.nio.ByteBuffer} over the slab's own memory, so a cache hit is
     * read in place rather than through a copy.
     *
     * <p>The buffer carries no cleaner, which is the point: nothing frees this memory except
     * {@link #freeSlab}, so the slab's lifetime is the cache's to decide rather than the collector's. It
     * is also why reading the buffer after the slab is freed is undefined, and why {@code VectorCacheSlab}
     * never hands the buffer out once it is closed.
     *
     * @param address address returned by {@link #allocateSlab}.
     * @param bytes the slab's size in bytes.
     * @return the direct buffer, or {@code null} when {@code address} is {@code 0}.
     */
    public static native java.nio.ByteBuffer wrapSlab(long address, long bytes);
}

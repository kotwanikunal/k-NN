/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.ffm;

/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
//package org.ap/ache.lucene.internal.vectorization;

import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_INT;

import java.lang.foreign.AddressLayout;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.security.AccessController;
import java.security.PrivilegedAction;
//public final class FFMInterface {
//    private FFMInterface() {}
//
//    public static final AddressLayout POINTER =
//            ValueLayout.ADDRESS.withTargetLayout(MemoryLayout.sequenceLayout(JAVA_BYTE));
//
//    private static final Linker LINKER = Linker.nativeLinker();
//    private static final SymbolLookup SYMBOL_LOOKUP;
//
//    static {
//        System.loadLibrary("ffm_native");
//        SymbolLookup loaderLookup = SymbolLookup.loaderLookup();
//        SYMBOL_LOOKUP = name -> loaderLookup.find(name).or(() -> LINKER.defaultLookup().find(name));
//    }
//
//    private static final FunctionDescriptor innerProdDesc =
//            FunctionDescriptor.of(ValueLayout.JAVA_FLOAT, POINTER, POINTER, JAVA_INT);
//
//    private static final MethodHandle innerProdMH =
//            SYMBOL_LOOKUP
//                    .find("innerprod_native")
//                    .map(addr -> LINKER.downcallHandle(addr, innerProdDesc))
//                    .orElse(null);
//
//    static final MethodHandle INNER_PRODUCT_IMPL = innerProdMH;
//
//    public static float computeInnerProduct(long queryAddress, long inputAddress, int length) {
//        try {
//            return (float) INNER_PRODUCT_IMPL.invokeExact(queryAddress, inputAddress, length);
//        } catch (Throwable e) {
//            throw new RuntimeException("Failed to compute inner product", e);
//        }
//    }
//}

public final class FFMInterface {
    private FFMInterface() {}

    // Update this line to use correct sequenceLayout syntax
    public static final AddressLayout POINTER =
            ValueLayout.ADDRESS.withTargetLayout(MemoryLayout.sequenceLayout(Long.MAX_VALUE, ValueLayout.JAVA_BYTE));

    private static final Linker LINKER = Linker.nativeLinker();
    private static final SymbolLookup SYMBOL_LOOKUP;

    static {
//        System.loadLibrary("ffm_native");
        try {
            AccessController.doPrivileged((PrivilegedAction<Void>) () -> {
                System.loadLibrary("ffm_native");
                return null;
            });
        } catch (Exception e) {
            throw new RuntimeException("Failed to load ffm_native library", e);
        }
        SymbolLookup loaderLookup = SymbolLookup.loaderLookup();
        SYMBOL_LOOKUP = name -> loaderLookup.find(name).or(() -> LINKER.defaultLookup().find(name));
    }

    private static final FunctionDescriptor innerProdDesc =
            FunctionDescriptor.of(ValueLayout.JAVA_FLOAT, POINTER, POINTER, JAVA_INT);

    private static final MethodHandle innerProdMH =
            SYMBOL_LOOKUP
                    .find("innerprod_native")
                    .map(addr -> LINKER.downcallHandle(addr, innerProdDesc))
                    .orElse(null);

    static final MethodHandle INNER_PRODUCT_IMPL = innerProdMH;

    public static float computeInnerProduct(long queryAddress, long inputAddress, int length) {
        try {
            return (float) INNER_PRODUCT_IMPL.invokeExact(queryAddress, inputAddress, length);
        } catch (Throwable e) {
            throw new RuntimeException("Failed to compute inner product", e);
        }
    }
}




///// *
//// * Copyright OpenSearch Contributors
//// * SPDX-License-Identifier: Apache-2.0
//// */
////
// package org.opensearch.knn.index.codec.KNN9120Codec;
////
//
// import java.lang.foreign.MemorySegment;
// import org.apache.lucene.codecs.lucene95.OffHeapFloatVectorValues;
// import org.apache.lucene.index.FloatVectorValues;
// import org.apache.lucene.index.VectorSimilarityFunction;
// import org.apache.lucene.store.IndexInput;
// import org.apache.lucene.store.MemorySegmentAccessInput;
// import org.apache.lucene.util.hnsw.RandomVectorScorer;
// import org.apache.lucene.util.hnsw.RandomVectorScorerSupplier;
// import org.junit.Test;
// import org.opensearch.knn.KNNTestCase;
//
// import java.io.IOException;
//
// import static org.mockito.ArgumentMatchers.anyInt;
// import static org.mockito.ArgumentMatchers.anyLong;
// import static org.mockito.Mockito.mock;
// import static org.mockito.Mockito.when;
//
///// *
//// * Copyright OpenSearch Contributors
//// * SPDX-License-Identifier: Apache-2.0
//// */
////
//// import org.apache.lucene.document.Field;
//// import org.apache.lucene.document.KnnVectorField;
//// import org.apache.lucene.index.DirectoryReader;
//// import org.apache.lucene.index.IndexWriter;
//// import org.apache.lucene.index.IndexWriterConfig;
//// import org.apache.lucene.store.Directory;
//// import org.apache.lucene.store.FSDirectory;
//// import org.junit.Before;
//// import org.junit.Test;
//// import org.opensearch.knn.KNNTestCase;
//// import org.opensearch.knn.index.VectorField;
//// import org.opensearch.test.OpenSearchTestCase;
//// import org.opensearch.knn.index.VectorDataType;
////
//// import java.io.IOException;
//// import java.nio.file.Path;
//// import java.util.Arrays;
////
//// import static org.hamcrest.Matchers.equalTo;
////
// public class NativeLuceneScorerTests extends KNNTestCase {
//
// @Test
// public void testKNN9120NativeLuceneScorer() throws IOException {
// KNN9120NativeLuceneScorer scorer = new KNN9120NativeLuceneScorer();
//
// try {
// // Create a simple target vector for testing
// float[] target = new float[] { 1.0f, 1.0f, 1.0f };
//
// // Test getting a vector scorer supplier - we need to use Mockito for the FloatVectorValues
// // KNNFloatVectorValues vectorValues = createFloatVectorValues(3,5,9);
// FloatVectorValues vectorValues = mock(FloatVectorValues.class);
// when(vectorValues.size()).thenReturn(3);
// when(vectorValues.vectorValue(anyInt())).thenReturn(new float[] { 1.0f, 2.0f, 3.0f });
// // Test RandomVectorScorerSupplier creation
// RandomVectorScorerSupplier scorerSupplier = scorer.getRandomVectorScorerSupplier(
// VectorSimilarityFunction.EUCLIDEAN,
// vectorValues
// );
// assertNotNull(scorerSupplier);
// // assertTrue(scorerSupplier instanceof KNN9120NativeLuceneScorer.NativeLuceneRandomVectorScorerSupplier);
//
// // Test RandomVectorScorer creation
// RandomVectorScorer vectorScorer = scorer.getRandomVectorScorer(VectorSimilarityFunction.EUCLIDEAN, vectorValues, target);
// assertNotNull(vectorScorer);
// // assertTrue(vectorScorer instanceof KNN9120NativeLuceneScorer.NativeLuceneVectorScorer);
//
// float out = vectorScorer.score(0);
//
// // vectorScorer = null;
// System.gc();
// System.out.println("we're here");
// // Verify proper cleanup is handled automatically through Java's finalize mechanism
// // The test passing without errors indicates proper resource management
// } catch (Exception e) {
// throw new RuntimeException(e);
// }
// }
//
// @Test
// public void testKNN9120NativeLuceneScorerOffHeap() throws IOException {
// KNN9120NativeLuceneScorer scorer = new KNN9120NativeLuceneScorer();
//
// try {
// // Create a simple target vector for testing
// float[] target = new float[] { 1.0f, 1.0f, 1.0f };
//
// // Test getting a vector scorer supplier - we need to use Mockito for the FloatVectorValues
// // KNNFloatVectorValues vectorValues = createFloatVectorValues(3,5,9);
// OffHeapFloatVectorValues vectorValues = mock(OffHeapFloatVectorValues.class);
// when(vectorValues.size()).thenReturn(3);
// when(vectorValues.vectorValue(anyInt())).thenReturn(new float[] { 1.0f, 2.0f, 3.0f });
// MemorySegmentAccessInput input = mock(MemorySegmentAccessInput.class);
// when(vectorValues.getSlice()).thenReturn((IndexInput) input);
// when(input.length()).thenReturn(3L);
// MemorySegment seg = mock(MemorySegment.class);
// when(input.segmentSliceOrNull(anyLong(), anyLong())).thenReturn(seg);
// when(input.address()).thenReturn(0L);
// // Test RandomVectorScorerSupplier creation
// RandomVectorScorerSupplier scorerSupplier = scorer.getRandomVectorScorerSupplier(
// VectorSimilarityFunction.EUCLIDEAN,
// vectorValues
// );
// assertNotNull(scorerSupplier);
// // assertTrue(scorerSupplier instanceof KNN9120NativeLuceneScorer.NativeLuceneRandomVectorScorerSupplier);
//
// // Test RandomVectorScorer creation
// RandomVectorScorer vectorScorer = scorer.getRandomVectorScorer(VectorSimilarityFunction.EUCLIDEAN, vectorValues, target);
// assertNotNull(vectorScorer);
// // assertTrue(vectorScorer instanceof KNN9120NativeLuceneScorer.NativeLuceneVectorScorer);
//
// float out = vectorScorer.score(0);
//
// vectorScorer = null;
// System.gc();
// System.out.println("we're here");
// // Verify proper cleanup is handled automatically through Java's finalize mechanism
// // The test passing without errors indicates proper resource management
// } catch (Exception e) {
// throw new RuntimeException(e);
// }
// }
// }
////
//// private Directory directory;
//// private IndexWriter writer;
//// private KNN9120HnswNativeLuceneVectorsFormat vectorsFormat;
//// private static final int DIMENSIONS = 128;
////
//// @Before
//// public void setUp() throws Exception {
//// super.setUp();
//// directory = FSDirectory.open(createTempDir());
//// vectorsFormat = new KNN9120HnswNativeLuceneVectorsFormat();
//// IndexWriterConfig config = new IndexWriterConfig();
//// config.setCodec(new KNN9120Codec());
//// writer = new IndexWriter(directory, config);
//// }
////
//// @Test
//// public void testVectorFormatBasicOperations() throws IOException {
//// // Test vector writing and reading
//// float[] vector = new float[DIMENSIONS];
//// Arrays.fill(vector, 0.1f);
////
//// // Create and add document with vector
//// KnnVectorField vectorField = new KnnVectorField("vector_field", vector);
//// writer.addDocument(Arrays.asList(vectorField));
//// writer.commit();
////
//// // Read and verify
//// try (DirectoryReader reader = DirectoryReader.open(writer)) {
//// assertEquals(1, reader.numDocs());
//// // Add verification of vector content
//// }
//// }
////
//// @Test
//// public void testNativeLuceneScorerSimilarityCalculation() throws IOException {
//// // Prepare test vectors
//// float[] queryVector = new float[DIMENSIONS];
//// Arrays.fill(queryVector, 0.5f);
////
//// float[] docVector = new float[DIMENSIONS];
//// Arrays.fill(docVector, 0.7f);
////
//// // Add document with known vector
//// KnnVectorField vectorField = new KnnVectorField("vector_field", docVector);
//// writer.addDocument(Arrays.asList(vectorField));
//// writer.commit();
////
//// // Test similarity calculation
//// try (DirectoryReader reader = DirectoryReader.open(writer)) {
//// KNN9120NativeLuceneScorer scorer = new KNN9120NativeLuceneScorer(
//// reader.leaves().get(0),
//// queryVector,
//// VectorDataType.FLOAT
//// );
////
//// // Verify similarity score
//// float score = scorer.score();
//// assertTrue("Score should be between 0 and 1", score >= 0 && score <= 1);
//// }
//// }
////
//// @Test
//// public void testMultipleVectorSearching() throws IOException {
//// // Add multiple vectors
//// int numDocs = 10;
//// for (int i = 0; i < numDocs; i++) {
//// float[] vector = new float[DIMENSIONS];
//// Arrays.fill(vector, (float) i / numDocs);
//// KnnVectorField vectorField = new KnnVectorField("vector_field", vector);
//// writer.addDocument(Arrays.asList(vectorField));
//// }
//// writer.commit();
////
//// // Search with query vector
//// float[] queryVector = new float[DIMENSIONS];
//// Arrays.fill(queryVector, 0.5f);
////
//// try (DirectoryReader reader = DirectoryReader.open(writer)) {
//// KNN9120NativeLuceneScorer scorer = new KNN9120NativeLuceneScorer(
//// reader.leaves().get(0),
//// queryVector,
//// VectorDataType.FLOAT
//// );
////
//// // Test scoring multiple documents
//// for (int i = 0; i < numDocs; i++) {
//// float score = scorer.score();
//// assertTrue("Score should be between 0 and 1", score >= 0 && score <= 1);
//// scorer.nextDoc();
//// }
//// }
//// }
////
//// @Test
//// public void testVectorFormatWithDifferentDimensions() throws IOException {
//// // Test with different dimension sizes
//// int[] dimensionSizes = {32, 64, 128, 256};
////
//// for (int dim : dimensionSizes) {
//// float[] vector = new float[dim];
//// Arrays.fill(vector, 0.1f);
////
//// KnnVectorField vectorField = new KnnVectorField("vector_field_" + dim, vector);
//// writer.addDocument(Arrays.asList(vectorField));
//// }
//// writer.commit();
////
//// try (DirectoryReader reader = DirectoryReader.open(writer)) {
//// assertEquals(dimensionSizes.length, reader.numDocs());
//// // Verify each vector dimension
//// }
//// }
////
//// @Test
//// public void testErrorHandling() {
//// // Test invalid vector dimensions
//// assertThrows(IllegalArgumentException.class, () -> {
//// float[] invalidVector = new float[0];
//// new KnnVectorField("invalid_field", invalidVector);
//// });
////
//// // Test null vector
//// assertThrows(NullPointerException.class, () -> {
//// new KnnVectorField("null_field", null);
//// });
//// }
////
//// @Override
//// public void tearDown() throws Exception {
//// writer.close();
//// directory.close();
//// super.tearDown();
//// }
//// }

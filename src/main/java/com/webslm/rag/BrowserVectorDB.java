package com.webslm.rag;

import java.util.ArrayList;
import java.util.List;

/**
 * Lightweight, pure-Java in-browser vector store.
 *
 * <p>Owns the document chunking, the in-memory primitive vector index, and the
 * direct cosine-similarity ranking + context assembly &mdash; all using raw
 * {@code float} primitives. No reflection, no dynamic classloading, so TeaVM
 * compiles this statically.</p>
 *
 * <p>Embeddings themselves are produced asynchronously by Transformers.js on
 * the JavaScript side; this class is handed the already-computed
 * {@code float[]} vectors via {@link #indexChunk}.</p>
 */
public class BrowserVectorDB {

    private final List<DocumentChunk> index = new ArrayList<>();

    public static class DocumentChunk {
        public final String text;
        public final float[] vector;

        public DocumentChunk(String text, float[] vector) {
            this.text = text;
            this.vector = vector;
        }
    }

    /**
     * Sliding-window character segmenter. Emits each chunk to the JavaScript
     * layer (which embeds it asynchronously and calls back into
     * {@link #indexChunk}).
     */
    public void chunkAndEmit(String rawContent, int chunkSize, int overlap) {
        if (rawContent == null || rawContent.trim().isEmpty()) return;

        int step = Math.max(1, chunkSize - overlap);
        int start = 0;
        int produced = 0;
        while (start < rawContent.length()) {
            int end = Math.min(start + chunkSize, rawContent.length());
            String textBlock = rawContent.substring(start, end).trim();
            if (!textBlock.isEmpty()) {
                NativeAIBridge.emitChunk(textBlock);
                produced++;
            }
            start += step;
        }
        NativeAIBridge.logFromWasm("chunkAndEmit(): split " + rawContent.length()
            + " chars into " + produced + " chunk(s) [window=" + chunkSize + ", overlap=" + overlap + "]");
    }

    /** Stores a chunk together with its precomputed embedding vector. */
    public void indexChunk(String text, float[] vector) {
        if (text == null || vector == null || vector.length == 0) return;
        index.add(new DocumentChunk(text, vector));
        NativeAIBridge.logFromWasm("indexChunk(): vector #" + index.size()
            + " stored (dim=" + vector.length + ", |v|=" + round4(magnitude(vector)) + ")");
    }

    public int size() {
        return index.size();
    }

    /**
     * Ranks every indexed chunk by descending cosine similarity to the query
     * vector and assembles the top-K chunks into a structured context string.
     */
    public String searchTopContext(float[] queryVector, int topK) {
        if (index.isEmpty() || queryVector == null) return "";

        List<DocumentChunk> matches = new ArrayList<>(index);
        // Sort explicitly by descending similarity values
        matches.sort((a, b) -> Double.compare(
            cosineSimilarity(b.vector, queryVector),
            cosineSimilarity(a.vector, queryVector)
        ));

        StringBuilder contextBuilder = new StringBuilder();
        StringBuilder scoreLog = new StringBuilder();
        int limit = Math.min(topK, matches.size());
        for (int i = 0; i < limit; i++) {
            contextBuilder.append(matches.get(i).text).append("\n---\n");
            if (i > 0) scoreLog.append(", ");
            scoreLog.append(round4(cosineSimilarity(matches.get(i).vector, queryVector)));
        }
        NativeAIBridge.logFromWasm("searchTopContext(): cosine-ranked " + matches.size()
            + " vectors; top-" + limit + " similarity = [" + scoreLog + "]");
        return contextBuilder.toString();
    }

    /** Cosine similarity of two equal-length vectors (pure Java, no libraries). */
    public double cosineSimilarity(float[] vecA, float[] vecB) {
        if (vecA.length != vecB.length) return 0.0;
        double dotProduct = 0.0, normA = 0.0, normB = 0.0;
        for (int i = 0; i < vecA.length; i++) {
            dotProduct += vecA[i] * vecB[i];
            normA += vecA[i] * vecA[i];
            normB += vecB[i] * vecB[i];
        }
        if (normA == 0.0 || normB == 0.0) return 0.0;
        return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    }

    private static double magnitude(float[] v) {
        double sum = 0.0;
        for (float x : v) sum += x * x;
        return Math.sqrt(sum);
    }

    /** Rounds to 4 decimal places without relying on String.format (TeaVM-friendly). */
    static double round4(double value) {
        return Math.round(value * 10000.0) / 10000.0;
    }

    public void clearIndex() {
        this.index.clear();
    }
}

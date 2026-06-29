package com.webslm.rag;

import java.util.ArrayList;
import java.util.List;

/**
 * Lightweight, pure-Java in-browser vector store.
 *
 * <p>Handles text chunking, an in-memory primitive vector index, and direct
 * cosine-similarity distance computations using raw {@code float} primitives.
 * No reflection, no dynamic classloading &mdash; TeaVM compiles this statically.</p>
 */
public class BrowserVectorDB {

    private final List<DocumentChunk> index = new ArrayList<>();

    public static class DocumentChunk {
        public String text;
        public float[] vector;

        public DocumentChunk(String text, float[] vector) {
            this.text = text;
            this.vector = vector;
        }
    }

    /**
     * Segments raw content with a sliding character window and pushes each
     * chunk out to the JavaScript embedding engine for vectorization.
     */
    public void chunkAndIndexDocument(String rawContent, int chunkSize, int overlap) {
        if (rawContent == null || rawContent.trim().isEmpty()) return;

        // Simple sliding-window character index segmenter
        int start = 0;
        while (start < rawContent.length()) {
            int end = Math.min(start + chunkSize, rawContent.length());
            String textBlock = rawContent.substring(start, end).trim();

            if (!textBlock.isEmpty()) {
                NativeAIBridge.updateUIStatus("Vectorizing chunk at index: " + start);
                float[] embedding = NativeAIBridge.fetchEmbeddingFromBrowser(textBlock);
                if (embedding != null) {
                    index.add(new DocumentChunk(textBlock, embedding));
                }
            }
            start += (chunkSize - overlap);
        }
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
            calculateCosineSimilarity(b.vector, queryVector),
            calculateCosineSimilarity(a.vector, queryVector)
        ));

        StringBuilder contextBuilder = new StringBuilder();
        int limit = Math.min(topK, matches.size());
        for (int i = 0; i < limit; i++) {
            contextBuilder.append(matches.get(i).text).append("\n---\n");
        }
        return contextBuilder.toString();
    }

    private double calculateCosineSimilarity(float[] vecA, float[] vecB) {
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

    public void clearIndex() {
        this.index.clear();
    }
}

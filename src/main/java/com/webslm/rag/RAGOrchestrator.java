package com.webslm.rag;

import org.teavm.jso.JSExport;

/**
 * Main application entry point for the Java Wasm RAG core.
 *
 * <p>Exposes the synchronous, pure-compute surface (chunking, indexing,
 * similarity-based context assembly + prompt routing) to the JavaScript
 * hardware engine layer via {@code @JSExport}. JavaScript owns the
 * asynchronous model work (embeddings + WebGPU inference) and drives these
 * exports.</p>
 *
 * <p>Embedding vectors cross the Wasm boundary as comma-separated
 * {@code String}s &mdash; only {@code String}/{@code int} are exchanged, which
 * marshals reliably on the WasmGC backend.</p>
 */
public class RAGOrchestrator {

    private static final BrowserVectorDB db = new BrowserVectorDB();

    public static void main(String[] args) {
        NativeAIBridge.updateUIStatus("Java WasmGC Engine Mounted Successfully.");
        NativeAIBridge.logFromWasm("main(): RAGOrchestrator booted inside WebAssembly GC.");
    }

    /**
     * Deterministic self-check executed entirely in Java/Wasm. Returns a string
     * built by the Wasm core and computes a known cosine value (~0.7071), so a
     * correct result is proof the compiled Java is running.
     */
    @JSExport
    public static String selfTest() {
        float[] a = { 1f, 0f, 0f };
        float[] b = { 1f, 1f, 0f };
        double cos = BrowserVectorDB.round4(db.cosineSimilarity(a, b));
        NativeAIBridge.logFromWasm("selfTest(): cosine([1,0,0],[1,1,0]) = " + cos + " (expected 0.7071)");
        return "Java WasmGC core ALIVE — computed cosine([1,0,0],[1,1,0]) = " + cos
            + " (expected 0.7071); chunks currently indexed = " + db.size();
    }

    /** Drops every indexed chunk. */
    @JSExport
    public static void clearIndex() {
        db.clearIndex();
    }

    /**
     * Chunks the raw document. Each chunk is emitted back to JavaScript (via
     * {@code window.__onChunk}) so it can be embedded asynchronously.
     */
    @JSExport
    public static void emitChunks(String content) {
        db.chunkAndEmit(content, 500, 100);
    }

    /** Stores one chunk with its precomputed embedding (CSV of floats). */
    @JSExport
    public static void indexChunk(String text, String vectorCsv) {
        db.indexChunk(text, parseVector(vectorCsv));
    }

    /** Number of chunks currently held in the in-memory vector store. */
    @JSExport
    public static int indexSize() {
        return db.size();
    }

    /**
     * Runs cosine-similarity retrieval against the precomputed query embedding,
     * assembles the structured prompt, and routes it to the WebGPU SLM.
     */
    @JSExport
    public static void executeRAGQuery(String queryText, String queryVectorCsv) {
        NativeAIBridge.updateUIStatus("Scanning local vector space...");
        float[] queryEmbedding = parseVector(queryVectorCsv);
        NativeAIBridge.logFromWasm("executeRAGQuery(): parsed query embedding (dim=" + queryEmbedding.length + ")");
        String retrievedContext = db.searchTopContext(queryEmbedding, 3);

        String systemPersona = "You are an advanced domain-specific assistant. "
            + "Use only the provided context to answer the user request.";
        NativeAIBridge.updateUIStatus("Routing parameters to WebGPU SLM context engine...");
        NativeAIBridge.executeSLM(systemPersona, queryText, retrievedContext);
    }

    private static float[] parseVector(String vectorCsv) {
        if (vectorCsv == null || vectorCsv.isEmpty()) return new float[0];
        String[] parts = vectorCsv.split(",");
        float[] vector = new float[parts.length];
        for (int i = 0; i < parts.length; i++) {
            vector[i] = Float.parseFloat(parts[i]);
        }
        return vector;
    }
}

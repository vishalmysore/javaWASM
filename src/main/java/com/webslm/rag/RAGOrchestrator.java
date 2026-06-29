package com.webslm.rag;

import org.teavm.interop.Export;

/**
 * Main application entry point for the Java Wasm RAG core.
 *
 * <p>Maps state initialization to the runtime loop and exposes the two
 * document/query entry points to the JavaScript hardware engine layer via
 * {@code @Export}.</p>
 */
public class RAGOrchestrator {

    private static final BrowserVectorDB db = new BrowserVectorDB();

    public static void main(String[] args) {
        // Run internal setup checks during initialization
        NativeAIBridge.updateUIStatus("Java WasmGC Engine Mounted Successfully.");
    }

    @Export(name = "processIncomingDocument")
    public static void processIncomingDocument(String content) {
        db.clearIndex();
        db.chunkAndIndexDocument(content, 500, 100);
        NativeAIBridge.updateUIStatus("Indexing Complete. Elements stored securely in Wasm Memory.");
    }

    @Export(name = "executeRAGQuery")
    public static void executeRAGQuery(String query) {
        NativeAIBridge.updateUIStatus("Computing embedding for user query...");
        float[] queryEmbedding = NativeAIBridge.fetchEmbeddingFromBrowser(query);

        NativeAIBridge.updateUIStatus("Scanning local vector space...");
        String retrievedContext = db.searchTopContext(queryEmbedding, 3);

        String systemPersona = "You are an advanced domain-specific assistant. "
            + "Use only the provided context to answer the user request.";
        NativeAIBridge.updateUIStatus("Routing parameters to WebGPU SLM context engine...");
        NativeAIBridge.executeSLM(systemPersona, query, retrievedContext);
    }
}

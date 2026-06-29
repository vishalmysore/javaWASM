package com.webslm.rag;

import org.teavm.jso.JSBody;

/**
 * TeaVM JSInterop binding definitions (Java &rarr; JavaScript).
 *
 * <p>Uses {@code @JSBody} so the Java Wasm core can drive the JavaScript
 * hardware engine layer. Only {@code String} values cross the boundary here,
 * which marshals cleanly on the WasmGC backend.</p>
 */
public class NativeAIBridge {

    /** Hands a freshly produced text chunk to the JS side for embedding. */
    @JSBody(params = { "text" }, script = "window.__onChunk(text);")
    public static native void emitChunk(String text);

    /** Routes the assembled RAG prompt to the WebGPU SLM (WebLLM). */
    @JSBody(params = { "systemPrompt", "userQuery", "contextBlocks" },
            script = "window.streamSLMInference(systemPrompt, userQuery, contextBlocks);")
    public static native void executeSLM(String systemPrompt, String userQuery, String contextBlocks);

    /** Pushes a human-readable status line up to the UI. */
    @JSBody(params = { "status" }, script = "window.updateJavaStatusIndicator(status);")
    public static native void updateUIStatus(String status);

    /**
     * Emits a line into the "Java Wasm Core Activity" panel. Every call
     * originates inside the compiled {@code .wasm}, so anything shown there is
     * proof the WasmGC core executed.
     */
    @JSBody(params = { "line" }, script = "window.__wasmLog(line);")
    public static native void logFromWasm(String line);

    // ---- sqlite-vec memory engine bridges (synchronous; in-memory DB) ----

    /** Inserts a remembered fact + embedding (CSV) into the sqlite-vec store. */
    @JSBody(params = { "text", "csv" }, script = "window.__vecInsert(text, csv);")
    public static native void vecInsert(String text, String csv);

    /** Runs a sqlite-vec KNN over the query embedding (CSV); returns top-K facts. */
    @JSBody(params = { "csv", "k" }, script = "return window.__vecSearch(csv, k);")
    public static native String vecSearch(String csv, int k);

    /** Number of vectors currently in the sqlite-vec store. */
    @JSBody(params = {}, script = "return window.__vecCount();")
    public static native int vecCount();

    /** Empties the sqlite-vec store. */
    @JSBody(params = {}, script = "window.__vecClear();")
    public static native void vecClear();
}

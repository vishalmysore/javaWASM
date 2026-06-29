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
}

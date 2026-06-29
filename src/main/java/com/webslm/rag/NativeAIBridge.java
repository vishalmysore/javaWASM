package com.webslm.rag;

import org.teavm.jso.JSBody;

/**
 * TeaVM JSInterop binding definitions.
 *
 * <p>Uses {@code @JSBody} annotations to create high-speed links down into
 * browser memory. Primitive arrays are passed directly across the Wasm
 * boundary, avoiding manual serialization penalties.</p>
 */
public class NativeAIBridge {

    @JSBody(params = { "text" }, script = "return window.computeLocalEmbedding(text);")
    public static native float[] fetchEmbeddingFromBrowser(String text);

    @JSBody(params = { "systemPrompt", "userQuery", "contextBlocks" },
            script = "window.streamSLMInference(systemPrompt, userQuery, contextBlocks);")
    public static native void executeSLM(String systemPrompt, String userQuery, String contextBlocks);

    @JSBody(params = { "status" }, script = "window.updateJavaStatusIndicator(status);")
    public static native void updateUIStatus(String status);
}

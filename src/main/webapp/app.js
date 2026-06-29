// JavaScript Hardware Engine Layer
// ---------------------------------
// Hosts Transformers.js (local text embeddings) and WebLLM (Qwen2.5-0.5B-Instruct
// on WebGPU), and exposes clean window-scope bindings for the Java Wasm core.
//
// Versions are pinned exactly to protect the compilation ABI surface.
import * as webllm from "https://esm.run/@mlc-ai/web-llm@0.2.79";
import { pipeline } from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2";

let embeddingPipeline = null;
let llmEngine = null;
let javaAppInstance = null;

// Initialize Hardware Enclaves
async function initializeHardwareRuntimes() {
    window.updateJavaStatusIndicator("Spawning Feature Extractor (Xenova/all-MiniLM-L6-v2)...");
    embeddingPipeline = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');

    window.updateJavaStatusIndicator("Spawning WebGPU SLM Layer (Qwen2.5-0.5B-Instruct)...");
    const appConfig = {
        model_list: [{
            model: "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-MLC",
            model_id: "Qwen2.5-0.5B-Instruct-q4f16_1-webgpu",
        }]
    };

    llmEngine = new webllm.MLCEngine({
        appConfig,
        initProgressCallback: (report) => {
            window.updateJavaStatusIndicator(`Loading LLM: ${Math.round(report.progress * 100)}% - ${report.text}`);
        }
    });
    await llmEngine.reload("Qwen2.5-0.5B-Instruct-q4f16_1-webgpu");

    window.updateJavaStatusIndicator("All engines warm. Loading Java Wasm Application Core...");
    bootstrapJavaWasmApplication();
}

// Instantiate Wasm Module via generated TeaVM Runtime wrapper
async function bootstrapJavaWasmApplication() {
    javaAppInstance = await TeaVM.wasmGC.load("wasm-gc/classes.wasm");
    // Start main execution loop
    javaAppInstance.exports.main([]);
}

// Window Bridge Hook Implementations
window.computeLocalEmbedding = async function(text) {
    if (!embeddingPipeline) return null;
    const output = await embeddingPipeline(text, { pooling: 'mean', normalize: true });
    // Strip raw JavaScript TypedArray floats and pass straight to Wasm memory
    return Array.from(output.data);
};

window.streamSLMInference = async function(systemPrompt, userQuery, contextBlocks) {
    const outputBox = document.getElementById("chat-output");
    outputBox.innerHTML = "<b>Assistant:</b> ";

    const stream = await llmEngine.chat.completions.create({
        messages: [
            { role: "system", content: systemPrompt },
            { role: "user", content: `Context:\n${contextBlocks}\n\nQuestion: ${userQuery}` }
        ],
        stream: true,
        temperature: 0.7,
        top_p: 0.8
    });

    for await (const chunk of stream) {
        const word = chunk.choices[0]?.delta?.content || "";
        outputBox.innerHTML += word;
    }
    window.updateJavaStatusIndicator("Inference finished cleanly.");
};

window.updateJavaStatusIndicator = function(status) {
    document.getElementById("status-log").innerText = `[System Status]: ${status}`;
};

// UI Triggers mapping direct to exported Java context methods
window.triggerDocumentProcess = function() {
    const content = document.getElementById("doc-input").value;
    javaAppInstance.exports.processIncomingDocument(content);
};

window.triggerRAGSearch = function() {
    const query = document.getElementById("query-input").value;
    javaAppInstance.exports.executeRAGQuery(query);
};

// Auto-boot on load
window.addEventListener("DOMContentLoaded", initializeHardwareRuntimes);

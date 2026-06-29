// JavaScript Hardware Engine Layer
// ---------------------------------
// Hosts Transformers.js (local text embeddings) and WebLLM (Qwen2.5-0.5B-Instruct
// on WebGPU). The Java Wasm core owns chunking + vector math; this layer owns the
// asynchronous model work and orchestrates the exported Java methods.
//
// Versions are pinned exactly to protect the ABI surface.
import * as webllm from "https://esm.run/@mlc-ai/web-llm@0.2.79";
import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2";

// Fetch models from the Hugging Face CDN, NOT from this static Pages site
// (otherwise Transformers.js 404s on /models/Xenova/...).
env.allowLocalModels = false;

const EMBED_MODEL_ID = "Xenova/all-MiniLM-L6-v2";
const LLM_MODEL_ID = "Qwen2.5-0.5B-Instruct-q4f16_1-MLC"; // prebuilt in web-llm (incl. model_lib)

let embeddingPipeline = null;
let llmEngine = null;
let javaAppInstance = null;
let pendingChunks = [];

async function initializeHardwareRuntimes() {
    try {
        // 1. Boot the Java Wasm core FIRST so chunking/index math is ready immediately.
        window.updateJavaStatusIndicator("Loading Java Wasm Application Core...");
        javaAppInstance = await TeaVM.wasmGC.load("wasm-gc/classes.wasm");
        javaAppInstance.exports.main([]);

        // 2. Embedding model (Transformers.js, streamed from the HF CDN).
        window.updateJavaStatusIndicator(`Spawning Feature Extractor (${EMBED_MODEL_ID})...`);
        embeddingPipeline = await pipeline("feature-extraction", EMBED_MODEL_ID);

        // 3. WebGPU SLM (WebLLM prebuilt Qwen2.5-0.5B).
        if (!navigator.gpu) {
            window.updateJavaStatusIndicator(
                "WebGPU unavailable here — embedding + indexing work, but answer generation is disabled. Try Chrome/Edge 113+.");
        } else {
            window.updateJavaStatusIndicator(`Spawning WebGPU SLM Layer (${LLM_MODEL_ID})...`);
            llmEngine = await webllm.CreateMLCEngine(LLM_MODEL_ID, {
                initProgressCallback: (report) => {
                    window.updateJavaStatusIndicator(
                        `Loading LLM: ${Math.round(report.progress * 100)}% - ${report.text}`);
                }
            });
        }

        window.updateJavaStatusIndicator("✅ All engines warm. Ingest a document to begin.");
    } catch (err) {
        console.error(err);
        window.updateJavaStatusIndicator("Initialization error: " + (err && err.message ? err.message : err));
    }
}

async function computeEmbedding(text) {
    const output = await embeddingPipeline(text, { pooling: "mean", normalize: true });
    // Plain JS number array -> handed to the Wasm core as a CSV string.
    return Array.from(output.data);
}

function enginesReady() {
    if (!javaAppInstance || !embeddingPipeline) {
        window.updateJavaStatusIndicator("Engines are still warming up — please wait a moment.");
        return false;
    }
    return true;
}

// ---- Window bridges invoked FROM Java (@JSBody) ----
window.__onChunk = function(text) {
    pendingChunks.push(text);
};

window.updateJavaStatusIndicator = function(status) {
    document.getElementById("status-log").innerText = `[System Status]: ${status}`;
};

window.streamSLMInference = async function(systemPrompt, userQuery, contextBlocks) {
    const outputBox = document.getElementById("chat-output");
    if (!llmEngine) {
        outputBox.innerText = "LLM engine unavailable (WebGPU required for answer generation).";
        return;
    }
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
        outputBox.innerHTML += chunk.choices[0]?.delta?.content || "";
    }
    window.updateJavaStatusIndicator("Inference finished cleanly.");
};

// ---- UI triggers: async orchestration of the exported Java core ----
window.triggerDocumentProcess = async function() {
    if (!enginesReady()) return;
    const content = document.getElementById("doc-input").value;
    if (!content.trim()) {
        window.updateJavaStatusIndicator("Paste a knowledge document first.");
        return;
    }

    // Java chunks the document; each chunk arrives via window.__onChunk.
    javaAppInstance.exports.clearIndex();
    pendingChunks = [];
    javaAppInstance.exports.emitChunks(content);

    // Embed every chunk (async) and push the vector back into the Wasm core.
    for (let i = 0; i < pendingChunks.length; i++) {
        window.updateJavaStatusIndicator(`Vectorizing chunk ${i + 1}/${pendingChunks.length}...`);
        const vector = await computeEmbedding(pendingChunks[i]);
        javaAppInstance.exports.indexChunk(pendingChunks[i], vector.join(","));
    }

    const stored = javaAppInstance.exports.indexSize();
    window.updateJavaStatusIndicator(`Indexing complete. ${stored} chunks stored securely in Wasm memory.`);
};

window.triggerRAGSearch = async function() {
    if (!enginesReady()) return;
    const query = document.getElementById("query-input").value;
    if (!query.trim()) {
        window.updateJavaStatusIndicator("Type a question first.");
        return;
    }
    if (javaAppInstance.exports.indexSize() === 0) {
        window.updateJavaStatusIndicator("Index is empty — ingest a document before querying.");
        return;
    }

    window.updateJavaStatusIndicator("Computing embedding for user query...");
    const vector = await computeEmbedding(query);
    // Java retrieves context (cosine similarity) and routes the prompt to WebLLM.
    javaAppInstance.exports.executeRAGQuery(query, vector.join(","));
};

// Auto-boot on load
window.addEventListener("DOMContentLoaded", initializeHardwareRuntimes);

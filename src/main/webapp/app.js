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
const PDF_VER = "4.10.38"; // pinned pdf.js (ESM build) for PDF text extraction
const EMBED_DIM = 384;     // Xenova/all-MiniLM-L6-v2 output dimension
const SQLITE_VEC_URL = "https://cdn.jsdelivr.net/npm/sqlite-vec-wasm-demo@0.1.9/sqlite3.mjs";

let embeddingPipeline = null;
let llmEngine = null;
let javaAppInstance = null;
let pendingChunks = [];
let pdfjsLib = null;
let sqlite3Vec = null;
let vecDB = null;

async function initializeHardwareRuntimes() {
    // Wire the file picker + drag-and-drop immediately (independent of model loads).
    document.getElementById("file-input").addEventListener("change", (e) => loadFile(e.target.files[0]));
    setupDragAndDrop();
    try {
        // 1. Boot the Java Wasm core FIRST so chunking/index math is ready immediately.
        window.updateJavaStatusIndicator("Loading Java Wasm Application Core...");
        javaAppInstance = await TeaVM.wasmGC.load("wasm-gc/classes.wasm");
        javaAppInstance.exports.main([]);

        // Bring up the sqlite-vec engine, then restore memory from IndexedDB into it.
        await initSqliteVec();
        await rehydrateMemory();

        // Let the Java/Wasm core build its own UI panel (DOM + events from Java).
        javaAppInstance.exports.mountWasmUI();

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

// Drop a .pdf/.txt anywhere on the textarea to load + auto-index it.
function setupDragAndDrop() {
    const box = document.getElementById("doc-input");
    const stop = (e) => { e.preventDefault(); e.stopPropagation(); };

    box.addEventListener("dragover", (e) => { stop(e); box.classList.add("dragover"); });
    box.addEventListener("dragleave", (e) => { stop(e); box.classList.remove("dragover"); });
    box.addEventListener("drop", (e) => {
        stop(e);
        box.classList.remove("dragover");
        const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
        if (file) loadFile(file);
    });

    // Stop a near-miss drop elsewhere from navigating the browser away from the app.
    window.addEventListener("dragover", (e) => e.preventDefault());
    window.addEventListener("drop", (e) => e.preventDefault());
}

// Extract text from a dropped/selected file (.pdf via pdf.js, otherwise plain text),
// load it into the textarea, then auto-index through the existing Java pipeline.
async function loadFile(file) {
    if (!file) return;
    const name = (file.name || "").toLowerCase();
    try {
        let text;
        if (name.endsWith(".pdf") || file.type === "application/pdf") {
            text = await extractPdfText(file);
        } else {
            text = await file.text();
        }
        document.getElementById("doc-input").value = text;

        if (javaAppInstance && embeddingPipeline) {
            window.updateJavaStatusIndicator(`Loaded ${text.length} chars from "${file.name}". Auto-indexing...`);
            await window.triggerDocumentProcess();
        } else {
            window.updateJavaStatusIndicator(
                `Loaded ${text.length} chars from "${file.name}". Engines still warming — click "Chunk & Index" when ready.`);
        }
    } catch (err) {
        console.error(err);
        window.updateJavaStatusIndicator("File load error: " + (err && err.message ? err.message : err));
    }
}

async function extractPdfText(file) {
    if (!pdfjsLib) {
        pdfjsLib = await import(`https://cdn.jsdelivr.net/npm/pdfjs-dist@${PDF_VER}/build/pdf.min.mjs`);
        // Load the worker via a same-origin blob that imports the pinned CDN worker
        // (avoids cross-origin Worker restrictions on static hosts).
        const workerUrl = `https://cdn.jsdelivr.net/npm/pdfjs-dist@${PDF_VER}/build/pdf.worker.min.mjs`;
        pdfjsLib.GlobalWorkerOptions.workerSrc =
            URL.createObjectURL(new Blob([`import "${workerUrl}";`], { type: "text/javascript" }));
    }
    const data = new Uint8Array(await file.arrayBuffer());
    const pdf = await pdfjsLib.getDocument({ data }).promise;
    let out = "";
    for (let p = 1; p <= pdf.numPages; p++) {
        window.updateJavaStatusIndicator(`Extracting text: page ${p}/${pdf.numPages}...`);
        const page = await pdf.getPage(p);
        const content = await page.getTextContent();
        out += content.items.map((it) => it.str).join(" ") + "\n\n";
    }
    return out.trim();
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

// Sink for log lines emitted from INSIDE the .wasm (proof the Java core ran).
window.__wasmLog = function(line) {
    const box = document.getElementById("wasm-log");
    const stamp = new Date().toLocaleTimeString();
    if (box.textContent === "awaiting core boot...") box.textContent = "";
    box.textContent += `[${stamp}] ${line}\n`;
    box.scrollTop = box.scrollHeight;
    // Also surface in the console; the call stack originates in classes.wasm-runtime.js.
    console.log("%c[JAVA→WASM]", "color:#f59e0b;font-weight:bold", line);
};

// Calls a deterministic Java/Wasm computation and shows the value it returns.
window.verifyWasmCore = function() {
    if (!javaAppInstance) {
        window.updateJavaStatusIndicator("Java Wasm core not loaded yet.");
        return;
    }
    const result = javaAppInstance.exports.selfTest(); // String built inside the .wasm
    window.updateJavaStatusIndicator(result);
};

// ============================================================
//  Persistent long-term memory
//  Engine: sqlite-vec (real KNN, in a WASM SQLite, driven by the Java core).
//  Durability: IndexedDB stores {text, vector}; rehydrated into sqlite-vec on boot.
// ============================================================
const MEM_DB_NAME = "javawasm-memory";
const MEM_STORE = "facts";

// ---- sqlite-vec engine (called from the Java core via window.__vec* bridges) ----
async function initSqliteVec() {
    try {
        const mod = await import(SQLITE_VEC_URL);
        sqlite3Vec = await mod.default();
        vecDB = new sqlite3Vec.oo1.DB(":memory:");
        vecDB.exec("CREATE TABLE IF NOT EXISTS memories(id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT, ts INTEGER)");
        vecDB.exec("CREATE VIRTUAL TABLE IF NOT EXISTS vec_memories USING vec0(embedding float[" + EMBED_DIM + "])");
        console.log("sqlite-vec ready:", vecDB.selectValue("select vec_version()"));
    } catch (err) {
        console.error("sqlite-vec init failed:", err);
        window.updateJavaStatusIndicator("sqlite-vec init failed: " + (err && err.message ? err.message : err));
    }
}

function csvToFloat32(csv) {
    const parts = csv.split(",");
    const f = new Float32Array(parts.length);
    for (let i = 0; i < parts.length; i++) f[i] = parseFloat(parts[i]);
    return f;
}

// Insert one fact + embedding. Embedding -> separate vec0 row (auto rowid),
// text -> memories row keyed by that rowid.
window.__vecInsert = function(text, csv) {
    if (!vecDB) return;
    const f = csvToFloat32(csv);
    const ins = vecDB.prepare("INSERT INTO vec_memories(embedding) VALUES (?)");
    try { ins.bind(1, f.buffer).step(); } finally { ins.finalize(); }
    const rowid = vecDB.selectValue("select last_insert_rowid()");
    vecDB.exec({ sql: "INSERT INTO memories(id, text, ts) VALUES (?, ?, ?)", bind: [rowid, text, Date.now()] });
};

// KNN over the query embedding; returns the top-K fact texts joined for the LLM.
window.__vecSearch = function(csv, k) {
    if (!vecDB) return "";
    const f = csvToFloat32(csv);
    const sql = "SELECT m.text AS text FROM vec_memories " +
                "JOIN memories m ON m.id = vec_memories.rowid " +
                "WHERE vec_memories.embedding MATCH ? AND k = " + (k | 0) + " ORDER BY distance";
    const stmt = vecDB.prepare(sql);
    const lines = [];
    try { stmt.bind(1, f.buffer); while (stmt.step()) lines.push(stmt.get(0)); } finally { stmt.finalize(); }
    return lines.join("\n---\n");
};

window.__vecCount = function() {
    if (!vecDB) return 0;
    return vecDB.selectValue("select count(*) from vec_memories") | 0;
};

window.__vecClear = function() {
    if (!vecDB) return;
    vecDB.exec("DELETE FROM vec_memories");
    vecDB.exec("DELETE FROM memories");
};

function openMemoryDB() {
    return new Promise((resolve, reject) => {
        const req = indexedDB.open(MEM_DB_NAME, 1);
        req.onupgradeneeded = () => req.result.createObjectStore(MEM_STORE, { keyPath: "id", autoIncrement: true });
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error);
    });
}

async function memAdd(text, vector) {
    const db = await openMemoryDB();
    return new Promise((resolve, reject) => {
        const tx = db.transaction(MEM_STORE, "readwrite");
        tx.objectStore(MEM_STORE).add({ text, vector, ts: Date.now() });
        tx.oncomplete = () => resolve();
        tx.onerror = () => reject(tx.error);
    });
}

async function memGetAll() {
    const db = await openMemoryDB();
    return new Promise((resolve, reject) => {
        const req = db.transaction(MEM_STORE, "readonly").objectStore(MEM_STORE).getAll();
        req.onsuccess = () => resolve(req.result || []);
        req.onerror = () => reject(req.error);
    });
}

async function memClearAll() {
    const db = await openMemoryDB();
    return new Promise((resolve, reject) => {
        const tx = db.transaction(MEM_STORE, "readwrite");
        tx.objectStore(MEM_STORE).clear();
        tx.oncomplete = () => resolve();
        tx.onerror = () => reject(tx.error);
    });
}

// On every boot: pull persisted facts from IndexedDB and replay them into the
// Wasm vector store using the stored vectors (no re-embedding needed).
async function rehydrateMemory() {
    try {
        const facts = await memGetAll();
        for (const f of facts) {
            javaAppInstance.exports.rememberFact(f.text, f.vector.join(","));
        }
        updateMemoryCount();
        if (facts.length) {
            window.updateJavaStatusIndicator(`Restored ${facts.length} memory item(s) from IndexedDB into the sqlite-vec engine.`);
        }
    } catch (err) {
        console.warn("Memory rehydrate skipped:", err);
    }
}

function updateMemoryCount() {
    const el = document.getElementById("memory-count");
    if (el && javaAppInstance) el.textContent = `Memories stored: ${javaAppInstance.exports.memoryCount()}`;
    if (javaAppInstance) javaAppInstance.exports.refreshWasmUI(); // repaint the Java-built panel
}

function escapeHtml(s) {
    return (s || "").replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
}

window.triggerRemember = async function() {
    if (!enginesReady()) return;
    const input = document.getElementById("memory-input");
    const text = input.value.trim();
    if (!text) { window.updateJavaStatusIndicator("Type something for me to remember."); return; }

    window.updateJavaStatusIndicator("Embedding & persisting memory...");
    const vector = await computeEmbedding(text);
    await memAdd(text, vector);                                   // persisted in IndexedDB
    javaAppInstance.exports.rememberFact(text, vector.join(",")); // live in the Wasm store
    input.value = "";
    updateMemoryCount();
    window.updateJavaStatusIndicator(
        `Remembered. ${javaAppInstance.exports.memoryCount()} item(s) saved — they survive a browser restart.`);
};

window.triggerRecall = async function() {
    if (!enginesReady()) return;
    const query = document.getElementById("memory-query").value.trim();
    if (!query) { window.updateJavaStatusIndicator("Ask a question to recall."); return; }
    if (javaAppInstance.exports.memoryCount() === 0) {
        window.updateJavaStatusIndicator("No memories stored yet — tell me something to remember first.");
        return;
    }

    window.updateJavaStatusIndicator("Computing query embedding...");
    const vector = await computeEmbedding(query);
    const recalled = javaAppInstance.exports.recallMemory(vector.join(","), 3); // Java cosine recall

    const box = document.getElementById("memory-output");
    box.innerHTML = `<b>Recalled via sqlite-vec KNN (driven by the Java core):</b><pre>${escapeHtml(recalled) || "(nothing relevant)"}</pre>`;

    if (llmEngine && recalled) {
        box.innerHTML += "<b>Assistant:</b> ";
        const stream = await llmEngine.chat.completions.create({
            messages: [
                { role: "system", content: "You are the user's personal memory assistant. Answer using ONLY the remembered facts provided. If the answer isn't in them, say you don't have it in memory." },
                { role: "user", content: `Remembered facts:\n${recalled}\n\nQuestion: ${query}` }
            ],
            stream: true, temperature: 0.3, top_p: 0.8
        });
        for await (const chunk of stream) {
            box.innerHTML += chunk.choices[0]?.delta?.content || "";
        }
    }
    window.updateJavaStatusIndicator("Recall complete.");
};

window.clearMemory = async function() {
    await memClearAll();
    if (javaAppInstance) javaAppInstance.exports.clearMemory();
    updateMemoryCount();
    document.getElementById("memory-output").innerText = "Memory cleared.";
    window.updateJavaStatusIndicator("Long-term memory cleared (IndexedDB + Wasm store).");
};

// ============================================================
//  Multi-agent society  (Supervisor in Wasm drives JS LLM turns)
// ============================================================
async function runAgent(role, task, priorContext, paneId) {
    const system = javaAppInstance.exports.agentSystemPrompt(role); // role prompt authored in Wasm
    const user = priorContext ? `Task: ${task}\n\n${priorContext}` : `Task: ${task}`;
    const pane = document.getElementById(paneId);
    pane.textContent = "";
    let text = "";
    const stream = await llmEngine.chat.completions.create({
        messages: [{ role: "system", content: system }, { role: "user", content: user }],
        stream: true, temperature: 0.5, top_p: 0.9
    });
    for await (const chunk of stream) {
        const word = chunk.choices[0]?.delta?.content || "";
        text += word;
        pane.textContent += word;
        pane.scrollTop = pane.scrollHeight;
    }
    return text;
}

window.triggerAgentSociety = async function() {
    if (!enginesReady()) return;
    if (!llmEngine) { window.updateJavaStatusIndicator("WebGPU LLM required to run the agent society."); return; }
    const task = document.getElementById("agent-task").value.trim();
    if (!task) { window.updateJavaStatusIndicator("Describe a task for the agents."); return; }

    ["agent-researcher", "agent-coder", "agent-critic", "agent-final"].forEach((id) => {
        document.getElementById(id).textContent = "…";
    });

    const plan = javaAppInstance.exports.supervisorPlan(task);   // Supervisor (Wasm) plans the pipeline
    window.updateJavaStatusIndicator(`Supervisor plan: [${plan}]. Researcher working...`);
    const research = await runAgent("researcher", task, "", "agent-researcher");

    window.updateJavaStatusIndicator("Coder agent working...");
    const code = await runAgent("coder", task, "Requirements from the Researcher:\n" + research, "agent-coder");

    window.updateJavaStatusIndicator("Critic agent reviewing...");
    const critique = await runAgent("critic", task,
        "Requirements:\n" + research + "\n\nProposed code:\n" + code, "agent-critic");

    window.updateJavaStatusIndicator("Supervisor merging agent outputs...");
    const report = javaAppInstance.exports.assembleFinal(task, research, code, critique); // Supervisor (Wasm) merges
    document.getElementById("agent-final").textContent = report;
    window.updateJavaStatusIndicator("Agent society finished.");
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
    javaAppInstance.exports.refreshWasmUI(); // repaint the Java-built dashboard
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
    const compress = document.getElementById("compress-toggle").checked ? 1 : 0;
    // Java retrieves context, optionally compresses it (Headroom-style), and routes to WebLLM.
    javaAppInstance.exports.executeRAGQuery(query, vector.join(","), compress);
};

// Auto-boot on load
window.addEventListener("DOMContentLoaded", initializeHardwareRuntimes);

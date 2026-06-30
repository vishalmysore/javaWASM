# Serverless AI in a Browser Tab: Java → WebAssembly + Local WebGPU LLMs

### A deep technical whitepaper on building a zero-infrastructure RAG architecture where the business logic is Java compiled to WebAssembly and the intelligence is a quantized LLM running on your own GPU

**Reference implementation:** [github.com/vishalmysore/javaWASM](https://github.com/vishalmysore/javaWASM) · **Live demo:** [vishalmysore.github.io/javaWASM](https://vishalmysore.github.io/javaWASM/)

---

## Abstract

For a decade the default shape of an "AI application" has been fixed: a thin client in the browser, a fat backend on someone else's servers, and a metered API call to a model hosted in a data center you will never see. This paper describes an architecture that inverts that shape completely. The entire system — document parsing, text chunking, vector storage, similarity search, context compression, multi-agent orchestration, *and* large-language-model inference — runs inside a single browser tab, on the user's own hardware, with **no backend, no database service, and no API key**.

Two technologies make this possible. **WebAssembly (Wasm)** lets us compile a real, statically-typed business-logic core written in **Java** down to a compact, near-native binary that runs in the browser sandbox. **WebGPU** gives that same tab direct access to the machine's GPU, so a quantized small language model can generate tokens locally. We show how to weld these together with on-device embeddings and an in-browser vector database to produce a Retrieval-Augmented Generation (RAG) pipeline that is private by construction and costs nothing to operate.

This is not a thought experiment. Every claim here is backed by a deployed, open-source reference implementation, and we are deliberately honest about the sharp edges we hit along the way.

---

## 1. The problem with the current architecture

The conventional AI stack has four structural taxes:

1. **Cost.** Every inference is a billable event. A feature that "summarizes the user's notes" has a unit economics problem the moment it has users.
2. **Privacy.** To get an answer, the user's data must leave their device and transit a third party. For legal documents, medical notes, source code, or personal journals, that is often a non-starter.
3. **Latency & availability.** A network round-trip sits on the critical path of every interaction, and your app is only as available as the upstream API.
4. **Operational drag.** Servers, autoscaling, key rotation, rate-limit handling, vector-DB clusters — infrastructure that must be built, secured, paid for, and kept alive.

The interesting observation in 2026 is that **none of these taxes are fundamental** any more. Browsers have quietly become capable of running compute-heavy code at near-native speed (Wasm) and of driving the GPU directly (WebGPU). Embedding models have shrunk to tens of megabytes; instruction-tuned LLMs now ship in sub-gigabyte quantized form. The pieces to move the *entire* stack into the client exist — they just have not been assembled into a coherent architecture. That assembly is the subject of this paper.

---

## 2. What is WebAssembly?

WebAssembly is a **portable binary instruction format for a stack-based virtual machine**. It is not a language you write; it is a compilation *target*. C, C++, Rust, Go, Kotlin, and — crucially for us — Java can all be compiled into a `.wasm` module that any modern browser (≈96% of global sessions) can load and execute.

Three properties matter:

- **Near-native performance.** Wasm is designed to be decoded and JIT/AOT-compiled extremely fast, with a predictable instruction set close to real CPU semantics. For compute-bound work it is typically **1.5×–20× faster than equivalent JavaScript**, with far less variance from garbage-collection pauses and de-optimization.
- **A capability-secure sandbox.** A Wasm module has *no* ambient access to the DOM, the network, the filesystem, or memory outside its own linear heap. Everything it can touch must be explicitly imported from the host. This is a security model, not an afterthought — it is why running untrusted compiled code in a browser is safe.
- **Language independence.** Wasm breaks the JavaScript monoculture of the web. You can bring a mature, statically-typed, heavily-tested codebase in another language to the front end without a rewrite.

### 2.1 WasmGC: the part that makes Java practical

Early Wasm had a flat linear memory and no notion of managed objects. A language like Java — built around a garbage-collected object heap — had to *ship its own GC and memory manager* compiled into the module. That worked, but it bloated binaries and fought the host.

**WasmGC** (the WebAssembly Garbage Collection proposal, now broadly shipped) adds first-class managed heap types to the VM itself. Managed languages can emit `struct` and `array` types that the **browser's own garbage collector** manages. The payoff is dramatic: a Java program compiles to a *lean* module (our entire business core is a few hundred kilobytes) that shares the host GC, interoperates cleanly with JavaScript objects, and starts fast. WasmGC is the enabling technology that turns "Java in the browser" from a curiosity into an engineering choice.

---

## 3. Why Wasm is a game changer

It is tempting to frame Wasm as merely "faster JavaScript." That undersells it. The shift is architectural:

| Dimension | Before (JS-only) | With Wasm |
|---|---|---|
| **Languages** | JavaScript / TypeScript | Any language that compiles to Wasm |
| **Performance** | JIT, GC-pause-prone | Near-native, predictable |
| **Code reuse** | Rewrite backend logic in JS | Compile existing Java/Rust/C++ as-is |
| **Trust boundary** | Same-origin scripts | Capability-secure sandbox by default |
| **Where it runs** | Browser | Browser, edge, serverless, embedded, plugins |

The deepest consequence is that **Wasm moves the unit of deployment from "a service" to "a portable binary."** The same compiled core can run in a browser tab, on an edge node, inside a serverless function, or embedded in another application — unchanged. For our purposes, the relevant instance of that generality is simple and radical: *the business logic that used to require a server can now ship to, and run on, the client.*

---

## 4. Writing the business core in Java

We chose Java for the core because it is exactly the kind of language Wasm was supposed to liberate: statically typed, with a vast standard library, decades of tooling, and an enormous corpus of existing business logic. The compiler is **[TeaVM](https://teavm.org)** (v0.15), which takes Java bytecode and emits a WasmGC module.

### 4.1 Separation of concerns

The architecture draws one hard line:

```
┌──────────────────────────────────────────────┐
│  JAVA WASM CORE  (the "brain" — TeaVM/WasmGC)  │
│  • document chunking                           │
│  • vector storage + cosine similarity          │
│  • top-K retrieval + context assembly          │
│  • context compression (PCA, k-means, etc.)    │
│  • multi-agent supervisor logic                │
│  • UI construction (DOM + canvas)              │
└──────────────────────────────────────────────┘
                 ▲   │   @JSExport / @JSBody
                 │   ▼
┌──────────────────────────────────────────────┐
│  JS HARDWARE LAYER  (the "muscles")            │
│  • Transformers.js — text embeddings           │
│  • WebLLM — LLM inference on WebGPU             │
│  • sqlite-vec (WASM) — vector KNN engine        │
│  • IndexedDB — durable storage                 │
└──────────────────────────────────────────────┘
```

The Java core owns the **deterministic, algorithmic, business logic**. The JavaScript layer owns the **asynchronous, hardware-bound, I/O work** — the things browsers are natively good at. Neither layer reaches into the other's domain; they communicate only across a narrow, explicit boundary.

### 4.2 The interop surface

TeaVM's JavaScript Object (JSO) layer provides two annotations that constitute the entire bridge:

**`@JSExport`** — expose a Java method so JavaScript can call it. After the module loads, exported methods appear on `instance.exports`:

```java
public class RAGOrchestrator {
    @JSExport
    public static String buildContext(String queryText, String queryCsv, int compress) {
        float[] qv = parseVector(queryCsv);
        String retrieved = db.searchTopContext(qv, 3);          // Java cosine retrieval
        if (compress != 0) {
            return ContextCompressor.compress(queryText, retrieved, 600).text;
        }
        return retrieved;
    }
}
```

**`@JSBody`** — call a JavaScript function from Java by inlining a snippet:

```java
public class NativeAIBridge {
    @JSBody(params = {"systemPrompt", "userQuery", "context"},
            script = "window.streamSLMInference(systemPrompt, userQuery, context);")
    public static native void executeSLM(String systemPrompt, String userQuery, String context);
}
```

On the JavaScript side, loading the module is three lines:

```javascript
const teavm = await TeaVM.wasmGC.load("wasm-gc/classes.wasm");
teavm.exports.main([]);
const context = teavm.exports.buildContext(query, csv, 1); // calls into Java/Wasm
```

### 4.3 A hard-won lesson: the synchronous boundary

The single most important design constraint is this: **you cannot block on a JavaScript Promise from inside synchronous Wasm code.** Embedding a sentence (Transformers.js) and generating a token (WebLLM) are inherently asynchronous. If Java calls an `async` JS function expecting a return value, it gets a `Promise`, not the data.

The resolution is a clean rule that shaped the whole system: **JavaScript owns the async orchestration; Java owns the synchronous compute.** When Java needs many embeddings, it does *not* pull them. Instead it pushes work outward — it splits text and emits each fragment to JS via `@JSBody`; JS embeds asynchronously and calls *back* into a Java `@JSExport` with the finished vector. Across the boundary we pass only `String` and `int`/`double` (vectors travel as comma-separated strings), because those primitive types marshal reliably on WasmGC. This pattern — *Java plans, JavaScript awaits, Java computes* — recurs in indexing, semantic compression, and agent orchestration alike.

---

## 5. The serverless architecture

"Serverless" here is meant literally — not "someone else's servers (FaaS)," but **no servers at all**.

The build pipeline compiles `RAGOrchestrator.java` and friends to `classes.wasm` + a runtime loader, and a GitHub Actions workflow publishes the static bundle (`index.html`, `app.js`, the Wasm artifacts) straight to a CDN (GitHub Pages) with no branch tracking. There is no origin server in the request path. The "deployment" is a handful of static files behind a CDN edge.

Everything that a traditional stack would put on the server now lives in the tab:

| Traditional tier | In this architecture |
|---|---|
| API gateway / app server | Java core compiled to Wasm |
| Vector database (Pinecone, etc.) | sqlite-vec (WASM) + a Java cosine store |
| Embedding service | Transformers.js on the local CPU |
| LLM API (OpenAI, etc.) | WebLLM on the local GPU |
| Durable storage (Postgres) | IndexedDB / OPFS |
| CDN | CDN (the only thing left) |

The economic and privacy consequences fall straight out of the diagram: **marginal cost per inference is zero**, and **no user data ever leaves the device**.

---

## 6. Local intelligence: WebGPU, on-device embeddings, and a browser LLM

The "intelligence" tier is where 2026's browser capabilities earn their keep.

### 6.1 WebGPU

WebGPU is the modern successor to WebGL — a low-level API that exposes the GPU for both rendering *and* general-purpose **compute shaders**. It is what makes practical LLM inference in a tab possible: the matrix multiplications at the heart of a transformer map onto GPU compute kernels.

### 6.2 WebLLM + a quantized SLM

We run **`Qwen2.5-0.5B-Instruct`** quantized to `q4f16_1` (4-bit weights, ~945 MB VRAM) via **[WebLLM](https://github.com/mlc-ai/web-llm)** (MLC). The model weights download once into the browser cache; thereafter inference is fully local and offline-capable. Tokens stream from the GPU directly into the DOM:

```javascript
const engine = await webllm.CreateMLCEngine("Qwen2.5-0.5B-Instruct-q4f16_1-MLC", { initProgressCallback });
const stream = await engine.chat.completions.create({ messages, stream: true });
for await (const chunk of stream) outputBox.innerHTML += chunk.choices[0]?.delta?.content || "";
```

### 6.3 On-device embeddings

Retrieval needs vectors. **Transformers.js** runs `Xenova/all-MiniLM-L6-v2` (384-dimensional, mean-pooled, L2-normalized) on the CPU via ONNX-runtime-web — a ~25 MB model that produces sentence embeddings in milliseconds, entirely client-side.

The division of labor is deliberate: **embeddings on the CPU** (small, frequent, latency-sensitive), **generation on the GPU** (large, occasional, throughput-bound), **everything else in Wasm**.

---

## 7. The combined architecture: serverless RAG

Putting the tiers together yields a full Retrieval-Augmented Generation loop with no server in sight.

```
[User uploads text / PDF]
        │
        ▼
┌───────────────────────────────┐
│  JAVA WASM CORE                │  1. chunk the document (sliding window)
└───────┬───────────────────────┘
        │  emit each chunk ──► JS
        ▼
┌───────────────────────────────┐
│  JS: Transformers.js           │  2. embed each chunk → float[384]
└───────┬───────────────────────┘
        │  vector (CSV) ──► back into Wasm
        ▼
┌───────────────────────────────┐
│  JAVA WASM CORE                │  3. index the vector
│                                │  4. on query: cosine rank + top-K
│                                │  5. compress context (Headroom-style)
└───────┬───────────────────────┘
        │  assembled prompt ──► JS
        ▼
┌───────────────────────────────┐
│  JS: WebLLM (WebGPU)           │  6. stream the answer locally
└───────────────────────────────┘
```

On top of this spine the reference implementation layers several capabilities, each chosen to demonstrate that *real* work — not glue — runs in Java/Wasm:

- **Persistent memory.** A "remember/recall" facility backed by **sqlite-vec** (a vector-search SQLite extension compiled to WASM) for KNN, with **IndexedDB** as the durable layer; the store is rehydrated into the engine on every boot, so memory survives closing the browser. The Java core orchestrates it; a brute-force JS cosine store stands by as a fallback so the feature degrades gracefully if the WASM engine fails to load.
- **Context compression** ("Headroom-style"). Before the prompt reaches the model, Java trims retrieved context to the query-relevant sentences, drops near-duplicates (Jaccard), and enforces a token budget — in two modes: **lexical** (term-overlap scoring) and **semantic** (Java cosine-scores each sentence against the query embedding). Less noise for a small model; fewer tokens for a tight context window.
- **A multi-agent society.** A *Supervisor* implemented in Java plans a pipeline and authors the role prompts for *Researcher → Coder → Critic* agents; JavaScript runs the asynchronous LLM turns; Java merges the outputs. The orchestration logic is Wasm; the inference is WebGPU.
- **A semantic map.** Java implements **PCA** (top-2 components via power iteration, never materializing the 384×384 covariance) and **k-means**, from scratch, on the chunk embeddings, then draws an interactive, topic-clustered 2D scatter on an HTML `<canvas>` — real machine learning *and* graphics, both inside the Wasm core.
- **A UI built in WebAssembly.** A live dashboard whose DOM is created, styled, and event-wired entirely from Java via TeaVM's JSO DOM API — proof that Wasm can own presentation, not just computation.

### 7.1 Why the Java core is genuine business logic, not a wrapper

A fair skeptic asks: *is Java actually doing anything, or just relaying?* The answer is concrete. The cosine similarity, the top-K ranking, the sliding-window chunker, the sentence-level compression scoring, the PCA eigen-decomposition, the k-means clustering, and the agent merge are all **pure Java numerical/algorithmic code executing in the `.wasm`**. The system even surfaces this: a "Wasm Core Activity" panel streams lines emitted from *inside* the module (e.g. cosine scores that exist nowhere else), and a `selfTest()` export computes a known value to prove the compiled code is live.

---

## 8. Analysis

### 8.1 Privacy

The strongest property is structural, not promised: **data cannot leak because it never has anywhere to go.** Documents, embeddings, memories, and prompts live in the tab's heap, IndexedDB, and GPU memory. The only network traffic is the one-time, cacheable download of static assets and model weights from a CDN. This is a qualitatively different privacy posture than "we don't train on your data."

### 8.2 Cost

Capital and marginal cost of inference is **zero** to the operator; compute is donated by the client's own hardware. Hosting is a static CDN bucket. This changes which products are viable — features whose per-call economics would sink a server-backed app are free here.

### 8.3 Performance & honest trade-offs

We are deliberately not selling a miracle:

- **Model capability.** A 0.5B-parameter model is not GPT-class. It is excellent for grounded, context-bounded RAG answers and demonstrations; it is not a general reasoner. The architecture is model-agnostic — swap in a larger quantized model as client GPUs allow.
- **Cold start.** The first visit downloads model weights (hundreds of MB). Subsequent loads are cache-fast and offline-capable, but the first run is heavy.
- **Hardware floor.** WebGPU requires a reasonably modern browser/GPU (Chrome/Edge 113+). Where WebGPU is absent, retrieval and embeddings still work; generation is disabled.
- **Lexical vs semantic compression.** Term-overlap compression is fast but can drop a relevant sentence on vocabulary mismatch; the semantic mode fixes that at the cost of N extra embedding calls.
- **Ecosystem maturity.** The bleeding edge bites. We hit a real Emscripten 4.0.0 `postRun` regression in one sqlite-vec WASM build and had to move to a build carrying the upstream fix — and we keep a JS fallback precisely because browser-WASM library stability is still uneven.

These are engineering constraints to plan around, not refutations of the thesis.

---

## 9. When to reach for this architecture

This pattern is a strong fit when:

- **privacy is paramount** (legal, medical, financial, personal, on-device enterprise data);
- **per-inference cost must be zero or near-zero** at scale;
- **offline or air-gapped operation** is valuable;
- you want to **reuse a mature, typed business-logic codebase** (Java/Kotlin/Rust) rather than rewrite it in JS;
- the task is **bounded and retrieval-grounded** rather than open-ended frontier reasoning.

It is a poor fit when you genuinely need frontier-model capability, must support thin/old clients with no WebGPU, or require centralized data aggregation.

---

## 10. Future directions

- **Larger local models** as quantization and client GPUs improve (1.5B–3B class in the same architecture).
- **OPFS-backed persistence** to drop the IndexedDB rehydration step and keep a true SQLite file on disk.
- **WebGPU compute from Wasm directly**, letting the Java core dispatch its own kernels for the heavy linear algebra.
- **Embedding caches** so semantic compression and the semantic map avoid recomputation.
- **A portable core** — the same Wasm business module redeployed unchanged to edge functions and native hosts, fulfilling Wasm's "write once, run at any tier" promise.

---

## 11. Conclusion

The center of gravity of software is shifting back toward the client — not because the cloud failed, but because the browser quietly became a capable, GPU-accelerated, multi-language runtime. **WebAssembly** lets us put a real, statically-typed business core — written in **Java** — on that runtime as a lean, sandboxed, near-native binary. **WebGPU** puts a language model on the same tab's GPU. Stitched together with on-device embeddings and an in-browser vector database, the result is an AI application that is **private by construction, free to operate, and dependent on no server at all.**

The reference implementation proves it is buildable today, sharp edges and all. The interesting question is no longer *whether* the full AI stack can live in a browser tab — it can — but *which* applications should. For anything where privacy, cost, or offline operation matters, the answer is increasingly: this one.

---

## Appendix: reference stack

| Layer | Technology | Role |
|---|---|---|
| Business core | Java 17 → **TeaVM 0.15** (WasmGC) | Chunking, vectors, retrieval, compression, agents, UI |
| LLM inference | **WebLLM** (MLC) + `Qwen2.5-0.5B-Instruct-q4f16_1` | Local generation on WebGPU |
| Embeddings | **Transformers.js** + `Xenova/all-MiniLM-L6-v2` (384-d) | On-device CPU vectors |
| Vector engine | **sqlite-vec** (WASM) `vec0` KNN + Java cosine fallback | Similarity search |
| Durable storage | **IndexedDB** | Persistent memory |
| UI / graphics | **TeaVM JSO DOM + Canvas** | DOM and the semantic-map canvas, built in Java |
| Delivery | **GitHub Actions → GitHub Pages (CDN)** | Static, serverless deployment |

**Repository:** [github.com/vishalmysore/javaWASM](https://github.com/vishalmysore/javaWASM) — Apache-2.0.

*This whitepaper documents a working system; its architecture, code snippets, and trade-offs are taken directly from the reference implementation.*

# Serverless Java-Wasm RAG Pipeline (WebGPU)

A 100% local, serverless, zero-infrastructure Retrieval-Augmented Generation (RAG)
system that runs entirely inside a single browser tab. There is **no backend** —
the entire pipeline executes on the client using WebAssembly and WebGPU.

The architecture is split into two cleanly separated layers:

| Layer | Tech | Responsibility |
| ----- | ---- | -------------- |
| **Java Wasm Core** (orchestrator) | Java → WasmGC via **TeaVM 0.15.0** | Document model, sliding-window chunking, in-memory primitive vector store, cosine-similarity search, context assembly |
| **JavaScript Engine Layer** (hardware) | **Transformers.js** + **WebLLM** | Local text embeddings (`Xenova/all-MiniLM-L6-v2`) and SLM inference (`Qwen2.5-0.5B-Instruct` on WebGPU) |

## Data Flow

```
[User uploads text]
        │
        ▼
┌─────────────────────────────────────────────┐
│ JAVA WASM CORE (TeaVM)                       │
│ 1. Content splitter → chunks                 │
└───────┬─────────────────────────────────────┘
        │  chunk string ──► JSInterop ──► JS
        ▼
┌─────────────────────────────────────────────┐
│ JS HARDWARE ENGINE                           │
│ 2. Transformers.js → float[] embedding       │
└───────┬─────────────────────────────────────┘
        │  float[] ──► back into Wasm memory
        ▼
┌─────────────────────────────────────────────┐
│ JAVA WASM CORE (TeaVM)                        │
│ 3. Vector indexer                            │
│ 4. Query vectorizer → cosine similarity      │
│ 5. Context assembler → structured prompt     │
└───────┬─────────────────────────────────────┘
        │  context + prompt ──► WebLLM
        ▼
┌─────────────────────────────────────────────┐
│ JS HARDWARE ENGINE                           │
│ 6. WebLLM → Qwen2.5-0.5B via WebGPU (stream) │
└─────────────────────────────────────────────┘
```

## Project Layout

```
.
├── .github/workflows/deploy.yml      # Branchless GitHub Pages deployment
├── pom.xml                           # Maven + TeaVM WasmGC build
└── src/main/
    ├── java/com/webslm/rag/
    │   ├── BrowserVectorDB.java       # Chunking, vector math, indexer
    │   ├── NativeAIBridge.java        # TeaVM JSInterop bindings
    │   └── RAGOrchestrator.java       # Entry point + exported methods
    └── webapp/
        ├── index.html                # Static UI
        └── app.js                    # Transformers.js + WebLLM integration
```

## Build

```bash
mvn clean package
```

This compiles the Java core to `target/dist/wasm-gc/classes.wasm` and copies the
TeaVM WasmGC runtime (`classes.wasm-runtime.js`) alongside it.

## Run Locally

WebGPU and ES modules require the page to be served over HTTP (not `file://`):

```bash
mvn clean package
mkdir -p site && cp src/main/webapp/* site/ && cp -r target/dist/wasm-gc site/
cd site && python -m http.server 8000
# open http://localhost:8000 in a WebGPU-capable browser (Chrome/Edge 113+)
```

## Deploy

Pushing to `main` triggers `.github/workflows/deploy.yml`, which compiles the Wasm
on `ubuntu-latest`, bundles the static assets, and publishes them straight to the
GitHub Pages CDN via `actions/deploy-pages` (no `gh-pages` branch involved).

> Enable **Settings → Pages → Build and deployment → Source: GitHub Actions** on
> the repository once, so the workflow can publish.

## Requirements

- A browser with **WebGPU** enabled (Chrome / Edge 113+).
- The first run downloads the embedding model and the quantized Qwen SLM weights
  to the browser cache; subsequent loads are fast.

## Design Guardrails

- **Zero reflection** — no `java.lang.reflect.*` / dynamic classloading, so TeaVM
  compiles statically to a lean binary.
- **Pinned versions** — `web-llm@0.2.79` and `@xenova/transformers@2.17.2` are
  locked to protect the ABI surface.
- **Self-contained paths** — all relative paths match the build output layout for
  native GitHub Actions routing.

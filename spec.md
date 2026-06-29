\# Specification: Serverless Java-Wasm RAG Pipeline via WebGPU



\## 1. Executive Summary \& Architecture

The objective is to build a 100% local, serverless, zero-infrastructure-cost Retrieval-Augmented Generation (RAG) system running entirely inside a browser tab. 



To ensure clean design lines and extreme memory efficiency, the pipeline architecture must be strictly split into a high-performance \*\*Java processing core compiled to WebAssembly (WasmGC)\*\* and a \*\*native JavaScript browser-runtime engine layer\*\*. 



\### 1.1 Separation of Concerns

1\. \*\*Java Wasm Core (Orchestrator):\*\* Implements document data models, text chunking algorithms, string token segmentation, an in-memory primitive vector storage layout, and direct cosine similarity distance computations. It compiles down to a lean, sandboxed binary using TeaVM.

2\. \*\*JavaScript Engine Layer (Hardware Core):\*\* Handles heavy multi-threaded GPU/Wasm matrix operations by hosting \*\*Transformers.js\*\* (for running local text embeddings) and \*\*WebLLM\*\* (for running the Qwen2.5-0.5B-Instruct SLM on WebGPU). 



\### 1.2 Data Flow Blueprint

\[User Uploads Text/PDF Document]│▼┌────────────────────────────────────────────────────────┐│            JAVA WASM CORE (Compiled via TeaVM)         ││  1. Content Splitter: Breaks text into chunks          │└───────┬────────────────────────────────────────────────┘│├─► \[Looping Chunk Blocks] ──► Passes string out to JS via JSInterop│                                               ││  ┌────────────────────────────────────────────┘│  ▼┌────────────────────────────────────────────────────────┐│           JAVASCRIPT HARDWARE ENGINE CONTAINER         ││  2. Transformers.js: Computes high-dim float\[] vector │└───────┬────────────────────────────────────────────────┘│└─► Returns raw primitive float\[] array back to Java core memory│▼┌────────────────────────────────────────────────────────┐│            JAVA WASM CORE (Compiled via TeaVM)         ││  3. Vector Indexer: Inserts payload into memory array ││  4. Query Vectorizer: Computes Cosine Similarity      ││  5. Context Assembler: Builds structured prompt string│└───────┬────────────────────────────────────────────────┘│▼ Passes assembled context + prompt to WebLLM┌────────────────────────────────────────────────────────┐│           JAVASCRIPT HARDWARE ENGINE CONTAINER         ││  6. WebLLM Engine: Executes Qwen2.5-0.5B via WebGPU    │└────────────────────────────────────────────────────────┘

\---



\## 2. Directory Structure

Generate the project according to this strict layout:

java-wasm-rag/├── .github/│   └── workflows/│       └── deploy.yml              # Branchless GitHub Actions deployment├── pom.xml                         # Maven configuration with TeaVM pluginsrc/├── main/│   ├── java/│   │   └── com/│   │       └── webslm/│   │           └── rag/│   │               ├── BrowserVectorDB.java   # Chunking, Vector math, indexer│   │               ├── NativeAIBridge.java    # TeaVM JSInterop binding definitions│   │               └── RAGOrchestrator.java   # Main application entry point│   └── webapp/│       ├── index.html              # Static layout UI container│       └── app.js                  # Transformers.js + WebLLM integration hooks

\---



\## 3. Build \& Deployment Automations



\### 3.1 `pom.xml` (TeaVM WasmGC Targeting)

Configure the Maven build to output a native WasmGC binary targeting modern browser engines. Ensure it copies the accompanying JS worker runtimes into the build artifact using `copy-webassembly-gc-runtime`. Do not include reflection-heavy libraries.



```xml

<?xml version="1.0" encoding="UTF-8"?>

<project xmlns="\[http://maven.apache.org/POM/4.0.0](http://maven.apache.org/POM/4.0.0)"

&#x20;        xmlns:xsi="\[http://www.w3.org/2001/XMLSchema-instance](http://www.w3.org/2001/XMLSchema-instance)"

&#x20;        xsi:schemaLocation="\[http://maven.apache.org/POM/4.0.0](http://maven.apache.org/POM/4.0.0) \[http://maven.apache.org/xsd/maven-4.0.0.xsd](http://maven.apache.org/xsd/maven-4.0.0.xsd)">

&#x20;   <modelVersion>4.0.0</modelVersion>



&#x20;   <groupId>com.webslm</groupId>

&#x20;   <artifactId>java-wasm-rag</artifactId>

&#x20;   <version>1.0.0</version>

&#x20;   <packaging>jar</packaging>



&#x20;   <properties>

&#x20;       <maven.compiler.source>17</maven.compiler.source>

&#x20;       <maven.compiler.target>17</maven.compiler.target>

&#x20;       <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>

&#x20;       <teavm.version>0.15.0</teavm.version>

&#x20;   </properties>



&#x20;   <dependencies>

&#x20;       <!-- TeaVM JSO APIs for native JavaScript Interoperability -->

&#x20;       <dependency>

&#x20;           <groupId>org.teavm</groupId>

&#x20;           <artifactId>teavm-jso-apis</artifactId>

&#x20;           <version>${teavm.version}</version>

&#x20;           <scope>provided</scope>

&#x20;       </dependency>

&#x20;   </dependencies>



&#x20;   <build>

&#x20;       <plugins>

&#x20;           <plugin>

&#x20;               <groupId>org.teavm</groupId>

&#x20;               <artifactId>teavm-maven-plugin</artifactId>

&#x20;               <version>${teavm.version}</version>

&#x20;               <executions>

&#x20;                   <execution>

&#x20;                       <id>compile-wasm-gc</id>

&#x20;                       <goals>

&#x20;                           <goal>compile</goal>

&#x20;                       </goals>

&#x20;                       <configuration>

&#x20;                           <mainClass>com.webslm.rag.RAGOrchestrator</mainClass>

&#x20;                           <targetDirectory>${project.build.directory}/dist/wasm-gc</targetDirectory>

&#x20;                           <targetType>WEBASSEMBLY\_GC</targetType>

&#x20;                           <optimizationLevel>ADVANCED</optimizationLevel>

&#x20;                           <minifying>true</minifying>

&#x20;                       </configuration>

&#x20;                   </execution>

&#x20;                   <execution>

&#x20;                       <id>copy-runtime</id>

&#x20;                       <goals>

&#x20;                           <goal>copy-webassembly-gc-runtime</goal>

&#x20;                       </goals>

&#x20;                       <configuration>

&#x20;                           <targetDirectory>${project.build.directory}/dist/wasm-gc</targetDirectory>

&#x20;                       </configuration>

&#x20;                   </execution>

&#x20;               </plugins>

&#x20;           </plugin>

&#x20;       </plugins>

&#x20;   </build>

</project>

3.2 .github/workflows/deploy.yml (Actions-Native Pages Deployment)This continuous deployment workflow bypasses branch tracking. It compiles Java bytecode directly to WebAssembly and moves the output straight onto the production GitHub Pages CDN using temporary, branchless artifact streams.YAMLname: Compile Java RAG to Wasm \& Deploy



on:

&#x20; push:

&#x20;   branches: \[ main ]



permissions:

&#x20; contents: read

&#x20; pages: write

&#x20; id-token: write



concurrency:

&#x20; group: "pages"

&#x20; cancel-in-progress: true



jobs:

&#x20; build-and-compile:

&#x20;   runs-on: ubuntu-latest

&#x20;   steps:

&#x20;     - name: Checkout Source Code

&#x20;       uses: actions/checkout@v4



&#x20;     - name: Set up Java JDK

&#x20;       uses: actions/setup-java@v4

&#x20;       with:

&#x20;         distribution: 'temurin'

&#x20;         java-version: '17'

&#x20;         cache: 'maven'



&#x20;     - name: Compile Java to WasmGC via TeaVM

&#x20;       run: mvn clean package



&#x20;     - name: Stage Deployment Artifact Bundle

&#x20;       run: |

&#x20;         mkdir -p staging

&#x20;         cp src/main/webapp/index.html staging/

&#x20;         cp src/main/webapp/app.js staging/

&#x20;         cp -r target/dist/wasm-gc staging/

&#x20;         touch staging/.nojekyll



&#x20;     - name: Upload Static Web Bundle

&#x20;       uses: actions/upload-pages-artifact@v3

&#x20;       with:

&#x20;         path: ./staging



&#x20; deploy-to-pages-cdn:

&#x20;   environment:

&#x20;     name: github-pages

&#x20;     url: ${{ steps.deployment.outputs.page\_url }}

&#x20;   runs-on: ubuntu-latest

&#x20;   needs: build-and-compile

&#x20;   steps:

&#x20;     - name: Broadcast directly to CDN Environment

&#x20;       id: deployment

&#x20;       uses: actions/deploy-pages@v4

4\. Source Code Blueprint4.1 NativeAIBridge.javaUse JavaScript Interop (org.teavm.jso) annotations to create high-speed link vectors down into browser memory. This avoids manual serialization penalties by passing primitive arrays directly over the Wasm boundary.Javapackage com.webslm.rag;



import org.teavm.jso.JSBody;

import org.teavm.jso.JSObject;



public class NativeAIBridge {

&#x20;   

&#x20;   @JSBody(params = { "text" }, script = "return window.computeLocalEmbedding(text);")

&#x20;   public static native float\[] fetchEmbeddingFromBrowser(String text);

&#x20;   

&#x20;   @JSBody(params = { "systemPrompt", "userQuery", "contextBlocks" }, 

&#x20;           script = "window.streamSLMInference(systemPrompt, userQuery, contextBlocks);")

&#x20;   public static native void executeSLM(String systemPrompt, String userQuery, String contextBlocks);



&#x20;   @JSBody(params = { "status" }, script = "window.updateJavaStatusIndicator(status);")

&#x20;   public static native void updateUIStatus(String status);

}

4.2 BrowserVectorDB.javaImplement a lightweight, pure-Java object-oriented layer for processing text strings, tracking vector stores, and parsing matrix equations using raw float primitives.Javapackage com.webslm.rag;



import java.util.ArrayList;

import java.util.List;



public class BrowserVectorDB {

&#x20;   private final List<DocumentChunk> index = new ArrayList<>();



&#x20;   public static class DocumentChunk {

&#x20;       public String text;

&#x20;       public float\[] vector;



&#x20;       public DocumentChunk(String text, float\[] vector) {

&#x20;           this.text = text;

&#x20;           this.vector = vector;

&#x20;       }

&#x20;   }



&#x20;   public void chunkAndIndexDocument(String rawContent, int chunkSize, int overlap) {

&#x20;       if (rawContent == null || rawContent.trim().isEmpty()) return;

&#x20;       

&#x20;       // Simple sliding-window character index segmenter

&#x20;       int start = 0;

&#x20;       while (start < rawContent.length()) {

&#x20;           int end = Math.min(start + chunkSize, rawContent.length());

&#x20;           String textBlock = rawContent.substring(start, end).trim();

&#x20;           

&#x20;           if (!textBlock.isEmpty()) {

&#x20;               NativeAIBridge.updateUIStatus("Vectorizing chunk at index: " + start);

&#x20;               float\[] embedding = NativeAIBridge.fetchEmbeddingFromBrowser(textBlock);

&#x20;               if (embedding != null) {

&#x20;                   index.add(new DocumentChunk(textBlock, embedding));

&#x20;               }

&#x20;           }

&#x20;           start += (chunkSize - overlap);

&#x20;       }

&#x20;   }



&#x20;   public String searchTopContext(float\[] queryVector, int topK) {

&#x20;       if (index.isEmpty() || queryVector == null) return "";

&#x20;       

&#x20;       List<DocumentChunk> matches = new ArrayList<>(index);

&#x20;       // Sort explicitly by descending similarity values

&#x20;       matches.sort((a, b) -> Double.compare(

&#x20;           calculateCosineSimilarity(b.vector, queryVector),

&#x20;           calculateCosineSimilarity(a.vector, queryVector)

&#x20;       ));



&#x20;       StringBuilder contextBuilder = new StringBuilder();

&#x20;       int limit = Math.min(topK, matches.size());

&#x20;       for (int i = 0; i < limit; i++) {

&#x20;           contextBuilder.append(matches.get(i).text).append("\\n---\\n");

&#x20;       }

&#x20;       return contextBuilder.toString();

&#x20;   }



&#x20;   private double calculateCosineSimilarity(float\[] vecA, float\[] vecB) {

&#x20;       if (vecA.length != vecB.length) return 0.0;

&#x20;       double dotProduct = 0.0, normA = 0.0, normB = 0.0;

&#x20;       for (int i = 0; i < vecA.length; i++) {

&#x20;           dotProduct += vecA\[i] \* vecB\[i];

&#x20;           normA += vecA\[i] \* vecA\[i];

&#x20;           normB += vecB\[i] \* vecB\[i];

&#x20;       }

&#x20;       if (normA == 0.0 || normB == 0.0) return 0.0;

&#x20;       return dotProduct / (Math.sqrt(normA) \* Math.sqrt(normB));

&#x20;   }



&#x20;   public void clearIndex() {

&#x20;       this.index.clear();

&#x20;   }

}

4.3 RAGOrchestrator.javaMain entry point for Java class orchestration mapping state initializations to the runtime loop.Javapackage com.webslm.rag;



import org.teavm.interop.Export;



public class RAGOrchestrator {

&#x20;   private static final BrowserVectorDB db = new BrowserVectorDB();



&#x20;   public static void main(String\[] args) {

&#x20;       // Run internal setup checks during initialization

&#x20;       NativeAIBridge.updateUIStatus("Java WasmGC Engine Mounted Successfully.");

&#x20;   }



&#x20;   @Export(name = "processIncomingDocument")

&#x20;   public static void processIncomingDocument(String content) {

&#x20;       db.clearIndex();

&#x20;       db.chunkAndIndexDocument(content, 500, 100);

&#x20;       NativeAIBridge.updateUIStatus("Indexing Complete. Elements stored securely in Wasm Memory.");

&#x20;   }



&#x20;   @Export(name = "executeRAGQuery")

&#x20;   public static void executeRAGQuery(String query) {

&#x20;       NativeAIBridge.updateUIStatus("Computing embedding for user query...");

&#x20;       float\[] queryEmbedding = NativeAIBridge.fetchEmbeddingFromBrowser(query);

&#x20;       

&#x20;       NativeAIBridge.updateUIStatus("Scanning local vector space...");

&#x20;       String retrievedContext = db.searchTopContext(queryEmbedding, 3);

&#x20;       

&#x20;       String systemPersona = "You are an advanced domain-specific assistant. Use only the provided context to answer the user request.";

&#x20;       NativeAIBridge.updateUIStatus("Routing parameters to WebGPU SLM context engine...");

&#x20;       NativeAIBridge.executeSLM(systemPersona, query, retrievedContext);

&#x20;   }

}

4.4 app.js (WebGPU / Transformers.js Integration Layer)Provide standard JavaScript browser-engine mappings. Pin WebLLM strictly to version 0.2.79 to protect the compilation ABI surface, and expose simple, clean runtime bindings to the window scope for Java interaction.JavaScriptimport \* as webllm from "\[https://esm.run/@mlc-ai/web-llm@0.2.79](https://esm.run/@mlc-ai/web-llm@0.2.79)";

import { pipeline } from "\[https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2](https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2)";



let embeddingPipeline = null;

let llmEngine = null;

let javaAppInstance = null;



// Initialize Hardware Enclaves

async function initializeHardwareRuntimes() {

&#x20;   window.updateJavaStatusIndicator("Spawning Feature Extractor (Xenova/all-MiniLM-L6-v2)...");

&#x20;   embeddingPipeline = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');



&#x20;   window.updateJavaStatusIndicator("Spawning WebGPU SLM Layer (Qwen2.5-0.5B-Instruct)...");

&#x20;   const appConfig = {

&#x20;       model\_list: \[{

&#x20;           model: "\[https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-MLC](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-MLC)",

&#x20;           model\_id: "Qwen2.5-0.5B-Instruct-q4f16\_1-webgpu",

&#x20;       }]

&#x20;   };

&#x20;   

&#x20;   llmEngine = new webllm.MLCEngine({ 

&#x20;       appConfig, 

&#x20;       initProgressCallback: (report) => {

&#x20;           window.updateJavaStatusIndicator(`Loading LLM: ${Math.round(report.progress \* 100)}% - ${report.text}`);

&#x20;       }

&#x20;   });

&#x20;   await llmEngine.reload("Qwen2.5-0.5B-Instruct-q4f16\_1-webgpu");

&#x20;   

&#x20;   window.updateJavaStatusIndicator("All engines warm. Loading Java Wasm Application Core...");

&#x20;   bootstrapJavaWasmApplication();

}



// Instantiate Wasm Module via generated TeaVM Runtime wrapper

async function bootstrapJavaWasmApplication() {

&#x20;   javaAppInstance = await TeaVM.wasmGC.load("wasm-gc/classes.wasm");

&#x20;   // Start Main execution loop

&#x20;   javaAppInstance.exports.main(\[]);

}



// Window Bridge Hook Implementations

window.computeLocalEmbedding = async function(text) {

&#x20;   if (!embeddingPipeline) return null;

&#x20;   const output = await embeddingPipeline(text, { pooling: 'mean', normalize: true });

&#x20;   // Strip raw Javascript TypedArray floats and pass straight to Wasm memory

&#x20;   return Array.from(output.data);

};



window.streamSLMInference = async function(systemPrompt, userQuery, contextBlocks) {

&#x20;   const outputBox = document.getElementById("chat-output");

&#x20;   outputBox.innerHTML = "<b>Assistant:</b> ";

&#x20;   

&#x20;   const stream = await llmEngine.chat.completions.create({

&#x20;       messages: \[

&#x20;           { role: "system", content: systemPrompt },

&#x20;           { role: "user", content: `Context:\\n${contextBlocks}\\n\\nQuestion: ${userQuery}` }

&#x20;       ],

&#x20;       stream: true,

&#x20;       temperature: 0.7,

&#x20;       top\_p: 0.8

&#x20;   });



&#x20;   for await (const chunk of stream) {

&#x20;       const word = chunk.choices\[0]?.delta?.content || "";

&#x20;       outputBox.innerHTML += word;

&#x20;   }

&#x20;   window.updateJavaStatusIndicator("Inference finished cleanly.");

};



window.updateJavaStatusIndicator = function(status) {

&#x20;   document.getElementById("status-log").innerText = `\[System Status]: ${status}`;

};



// UI Triggers mapping direct to exported Java context methods

window.triggerDocumentProcess = function() {

&#x20;   const content = document.getElementById("doc-input").value;

&#x20;   javaAppInstance.exports.processIncomingDocument(content);

};



window.triggerRAGSearch = function() {

&#x20;   const query = document.getElementById("query-input").value;

&#x20;   javaAppInstance.exports.executeRAGQuery(query);

};



// Auto-boot on load

window.addEventListener("DOMContentLoaded", initializeHardwareRuntimes);

4.5 index.html (Minimal Static Web UI Interface)HTML<!DOCTYPE html>

<html lang="en">

<head>

&#x20;   <meta charset="UTF-8">

&#x20;   <title>Serverless Java-Wasm GPU RAG Sandbox</title>

&#x20;   <style>

&#x20;       body { font-family: system-ui, sans-serif; background: #0f172a; color: #f8fafc; max-width: 800px; margin: 2rem auto; padding: 0 1rem; }

&#x20;       textarea, input { width: 100%; background: #1e293b; border: 1px solid #334155; color: #f8fafc; padding: 0.75rem; border-radius: 6px; box-sizing: border-box; font-size: 1rem; }

&#x20;       button { background: #2563eb; color: white; border: none; padding: 0.75rem 1.5rem; font-weight: bold; border-radius: 6px; cursor: pointer; margin-top: 0.5rem; }

&#x20;       button:hover { background: #1d4ed8; }

&#x20;       #status-log { color: #38bdf8; font-family: monospace; padding: 1rem; background: #1e293b; border-radius: 6px; margin: 1rem 0; }

&#x20;       #chat-output { background: #1e293b; padding: 1rem; border-radius: 6px; min-height: 100px; border-left: 4px solid #10b981; margin-top: 1rem; line-height: 1.5; }

&#x20;   </style>

&#x20;   <!-- Include TeaVM generated Wasm-GC runtime library loader -->

&#x20;   <script type="text/javascript" src="wasm-gc/classes.wasm-runtime.js"></script>

&#x20;   <script type="module" src="app.js"></script>

</head>

<body>

&#x20;   <h2>🤖 Serverless Java-Wasm GPU RAG Sandbox</h2>

&#x20;   <div id="status-log">\[System Status]: Initializing underlying client environments...</div>



&#x20;   <h3>1. Ingest Knowledge Document</h3>

&#x20;   <textarea id="doc-input" rows="6" placeholder="Paste comprehensive knowledge text baseline here..."></textarea>

&#x20;   <button onclick="window.triggerDocumentProcess()">Chunk \& Index Document via Java Wasm</button>



&#x20;   <h3>2. Query Engine</h3>

&#x20;   <input type="text" id="query-input" placeholder="Ask a targeted contextual domain question...">

&#x20;   <button onclick="window.triggerRAGSearch()">Execute Contextual WebGPU Query</button>



&#x20;   <h3>Output Stream</h3>

&#x20;   <div id="chat-output">System idle. Ingest a document and fire a question to stream results.</div>

</body>

</html>

5\. Strict Execution GuardrailsWhen building out this codebase, ensure you adhere to the following architectural requirements:Zero-Reflection Rule: Java code must never use standard Reflection (java.lang.reflect.\*) or dynamic classloading. TeaVM compiles statically, and reflection structures will inflate the output size or break the application at compile time.  Explicit Version Locking: Ensure all CDN imports in JavaScript copy the exact pinned structures (web-llm@0.2.79). Do not use unpinned version tags.No Local Hardcode Paths: The configuration file and compilation steps must be fully self-contained. The final file system structure must ensure that relative path boundaries match the build output directory structure cleanly for native GitHub Actions routing.


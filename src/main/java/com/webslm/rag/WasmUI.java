package com.webslm.rag;

import org.teavm.jso.dom.events.EventListener;
import org.teavm.jso.dom.events.MouseEvent;
import org.teavm.jso.dom.html.HTMLDocument;
import org.teavm.jso.dom.html.HTMLElement;

/**
 * A rich UI panel built and driven entirely by Java compiled to WebAssembly,
 * using TeaVM's JSO DOM API.
 *
 * <p>Nothing in this panel is authored in HTML/CSS/JS &mdash; the Java/Wasm core
 * creates every element, wires the click handlers, and updates the live values
 * (reading the same business functions that power the rest of the app). This is
 * the "rich UI in WebAssembly + Java business logic" demonstration.</p>
 */
public class WasmUI {

    private static final HTMLDocument doc = HTMLDocument.current();
    private static HTMLElement chunksValue;
    private static HTMLElement memValue;
    private static HTMLElement selfTestValue;
    private static HTMLElement memBar;

    /** Builds the whole panel inside the empty #wasm-ui-root container. */
    static void mount() {
        HTMLElement root = doc.getElementById("wasm-ui-root");
        if (root == null) return;
        root.setInnerHTML("");

        HTMLElement note = doc.createElement("div");
        note.setAttribute("class", "wasm-ui-note");
        note.setInnerHTML("Every element below was created and is driven by Java (TeaVM) "
            + "compiled to WebAssembly &mdash; DOM construction, event handling, and live "
            + "updates, with no hand-written HTML/JS.");
        root.appendChild(note);

        HTMLElement grid = doc.createElement("div");
        grid.setAttribute("class", "wasm-ui-grid");
        root.appendChild(grid);
        chunksValue = appendCard(grid, "Doc chunks indexed");
        memValue = appendCard(grid, "Memories (sqlite-vec)");
        selfTestValue = appendCard(grid, "Self-test cosine");

        HTMLElement barWrap = doc.createElement("div");
        barWrap.setAttribute("class", "wasm-ui-barwrap");
        memBar = doc.createElement("div");
        memBar.setAttribute("class", "wasm-ui-bar");
        barWrap.appendChild(memBar);
        root.appendChild(barWrap);

        HTMLElement controls = doc.createElement("div");
        controls.setAttribute("class", "wasm-ui-controls");
        root.appendChild(controls);

        HTMLElement refreshBtn = doc.createElement("button");
        refreshBtn.setAttribute("class", "secondary");
        refreshBtn.setInnerHTML("↻ Refresh (Java reads business state)");
        EventListener<MouseEvent> onRefresh = (MouseEvent e) -> refresh();
        refreshBtn.addEventListener("click", onRefresh);
        controls.appendChild(refreshBtn);

        HTMLElement selfTestBtn = doc.createElement("button");
        selfTestBtn.setInnerHTML("▶ Run self-test (Java compute)");
        EventListener<MouseEvent> onSelfTest = (MouseEvent e) -> {
            double cos = RAGOrchestrator.selfTestCosineValue();
            if (selfTestValue != null) selfTestValue.setInnerHTML(Double.toString(cos));
            NativeAIBridge.logFromWasm("WasmUI: self-test button computed cosine=" + cos + " inside Wasm");
        };
        selfTestBtn.addEventListener("click", onSelfTest);
        controls.appendChild(selfTestBtn);

        NativeAIBridge.logFromWasm("WasmUI.mount(): built dashboard DOM from Java (TeaVM JSO).");
        refresh();
    }

    private static HTMLElement appendCard(HTMLElement grid, String label) {
        HTMLElement card = doc.createElement("div");
        card.setAttribute("class", "wasm-ui-card");
        HTMLElement value = doc.createElement("div");
        value.setAttribute("class", "wasm-ui-value");
        value.setInnerHTML("—");
        HTMLElement lbl = doc.createElement("div");
        lbl.setAttribute("class", "wasm-ui-label");
        lbl.setInnerHTML(label);
        card.appendChild(value);
        card.appendChild(lbl);
        grid.appendChild(card);
        return value;
    }

    /** Re-reads the live business state and repaints the values + bar. */
    static void refresh() {
        int chunks = RAGOrchestrator.docChunkCount();
        int mem = NativeAIBridge.vecCount();
        if (chunksValue != null) chunksValue.setInnerHTML(Integer.toString(chunks));
        if (memValue != null) memValue.setInnerHTML(Integer.toString(mem));
        if (memBar != null) {
            int pct = Math.min(100, mem * 10); // 10 memories fills the bar
            memBar.getStyle().setProperty("width", pct + "%");
        }
        NativeAIBridge.logFromWasm("WasmUI.refresh(): repainted from Java (chunks=" + chunks + ", memories=" + mem + ")");
    }
}

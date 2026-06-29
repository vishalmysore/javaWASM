package com.webslm.rag;

/**
 * The Supervisor side of an in-browser multi-agent society, running inside
 * WebAssembly GC.
 *
 * <p>Java owns the <b>orchestration logic</b>: it plans the agent pipeline,
 * defines each agent's role/system prompt, and merges the agents' outputs into
 * a final supervisor report. JavaScript drives the actual (asynchronous) LLM
 * turns on WebGPU, one agent at a time, following the Java-defined plan.</p>
 *
 * <pre>
 *                 Supervisor (this class, in Wasm)
 *                          |
 *        ---------------------------------------
 *        |                 |                   |
 *   Researcher          Coder               Critic
 * </pre>
 */
public class AgentSociety {

    /** Touched from {@code RAGOrchestrator.main} so TeaVM keeps this class reachable. */
    static void boot() {
        NativeAIBridge.logFromWasm("AgentSociety: supervisor registered roster [researcher, coder, critic].");
    }

    /**
     * The Supervisor decides the pipeline for a task. Returns an ordered,
     * comma-separated list of agent roles for JS to execute in turn.
     */
    public static String supervisorPlan(String task) {
        NativeAIBridge.logFromWasm("supervisorPlan(): task=\"" + preview(task) + "\" -> [researcher, coder, critic]");
        return "researcher,coder,critic";
    }

    /** Role-specific system prompt for each agent (authored in the Wasm core). */
    public static String agentSystemPrompt(String role) {
        switch (role) {
            case "researcher":
                return "You are the Researcher agent in a multi-agent team. "
                    + "Break the user's task into a short numbered list of concrete requirements "
                    + "and the key technical decisions. Be concise. Do NOT write code.";
            case "coder":
                return "You are the Coder agent in a multi-agent team. "
                    + "Implement the task using the Researcher's requirements. "
                    + "Return ONE self-contained solution inside a single fenced code block. "
                    + "Keep prose outside the code block to a minimum.";
            case "critic":
                return "You are the Critic agent in a multi-agent team. "
                    + "Review the Coder's solution against the requirements. "
                    + "Return a short bulleted list of concrete bugs, gaps, and improvements.";
            default:
                return "You are a helpful assistant.";
        }
    }

    /** Supervisor merge step: combines the three agents' outputs into one report. */
    public static String assembleFinal(String task, String research, String code, String critique) {
        String r = nullSafe(research);
        String c = nullSafe(code);
        String k = nullSafe(critique);

        StringBuilder sb = new StringBuilder();
        sb.append("# Supervisor Report\n\n");
        sb.append("Task: ").append(nullSafe(task)).append("\n\n");
        sb.append("## 1. Requirements (Researcher)\n").append(r).append("\n\n");
        sb.append("## 2. Implementation (Coder)\n").append(c).append("\n\n");
        sb.append("## 3. Review (Critic)\n").append(k).append("\n\n");
        sb.append("---\nMerged by the Supervisor agent inside Java WasmGC.\n");

        NativeAIBridge.logFromWasm("assembleFinal(): supervisor merged 3 agent outputs ("
            + r.length() + "+" + c.length() + "+" + k.length() + " chars).");
        return sb.toString();
    }

    private static String nullSafe(String s) {
        return s == null ? "" : s;
    }

    private static String preview(String s) {
        if (s == null) return "";
        String t = s.trim();
        return t.length() > 40 ? t.substring(0, 40) + "..." : t;
    }
}

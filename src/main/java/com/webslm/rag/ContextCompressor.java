package com.webslm.rag;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Headroom-inspired context compression layer, implemented in pure Java/Wasm.
 *
 * <p>Sits between retrieval and the LLM: it squeezes the retrieved RAG context
 * down to the sentences most relevant to the query, drops redundant/near-duplicate
 * sentences, and enforces a character budget &mdash; cutting tokens while keeping
 * the answer-bearing content. This is the statistical/extractive text compressor
 * (no embeddings, no regex), so it runs synchronously inside the WasmGC core.</p>
 */
public final class ContextCompressor {

    private static final Set<String> STOPWORDS = new HashSet<>();
    static {
        String[] sw = { "the", "a", "an", "and", "or", "but", "of", "to", "in", "on", "for", "with",
            "is", "are", "was", "were", "be", "been", "by", "at", "as", "it", "this", "that", "these",
            "those", "from", "into", "what", "which", "who", "whom", "how", "why", "when", "where",
            "do", "does", "did", "you", "your", "i", "we", "they", "he", "she", "them", "his", "her",
            "its", "their", "our", "about", "can", "could", "would", "should", "will", "may", "not" };
        for (String s : sw) STOPWORDS.add(s);
    }

    /** Outcome of a compression pass, including the before/after sizes. */
    public static final class Result {
        public final String text;
        public final int originalChars;
        public final int compressedChars;
        public final int totalSentences;
        public final int keptSentences;

        Result(String text, int originalChars, int compressedChars, int totalSentences, int keptSentences) {
            this.text = text;
            this.originalChars = originalChars;
            this.compressedChars = compressedChars;
            this.totalSentences = totalSentences;
            this.keptSentences = keptSentences;
        }

        public int reductionPct() {
            if (originalChars == 0) return 0;
            return (int) Math.round(100.0 * (originalChars - compressedChars) / originalChars);
        }
    }

    /** Compresses the retrieved context against the query, within a char budget. */
    public static Result compress(String query, String context, int charBudget) {
        if (context == null || context.isEmpty()) return new Result("", 0, 0, 0, 0);
        int originalChars = context.length();

        List<String> sentences = splitSentences(context);
        Set<String> queryTerms = terms(query);
        boolean hasQueryTerms = !queryTerms.isEmpty();

        List<Scored> scored = new ArrayList<>();
        for (int i = 0; i < sentences.size(); i++) {
            scored.add(new Scored(i, sentences.get(i), score(sentences.get(i), queryTerms)));
        }
        // Most query-relevant first (stable sort keeps original order among ties).
        scored.sort((a, b) -> Double.compare(b.score, a.score));

        List<Scored> picked = new ArrayList<>();
        List<Set<String>> pickedTokens = new ArrayList<>();
        int used = 0;
        for (Scored cand : scored) {
            if (hasQueryTerms && cand.score <= 0.0 && !picked.isEmpty()) break; // stop once irrelevant
            Set<String> toks = terms(cand.sentence);
            boolean duplicate = false;
            for (Set<String> prev : pickedTokens) {
                if (jaccard(toks, prev) > 0.8) { duplicate = true; break; } // drop near-duplicates
            }
            if (duplicate) continue;
            if (!picked.isEmpty() && used + cand.sentence.length() > charBudget) continue;
            picked.add(cand);
            pickedTokens.add(toks);
            used += cand.sentence.length();
            if (used >= charBudget) break;
        }
        if (picked.isEmpty() && !scored.isEmpty()) picked.add(scored.get(0)); // never return empty

        // Restore original reading order for coherence.
        picked.sort((a, b) -> Integer.compare(a.index, b.index));
        StringBuilder sb = new StringBuilder();
        for (Scored p : picked) sb.append(p.sentence.trim()).append(' ');
        String out = sb.toString().trim();

        return new Result(out, originalChars, out.length(), sentences.size(), picked.size());
    }

    private static final class Scored {
        final int index;
        final String sentence;
        final double score;
        Scored(int index, String sentence, double score) {
            this.index = index;
            this.sentence = sentence;
            this.score = score;
        }
    }

    /** Distinct query-term coverage, lightly normalized to prefer dense sentences. */
    private static double score(String sentence, Set<String> queryTerms) {
        if (queryTerms.isEmpty()) return 0.0;
        Set<String> st = terms(sentence);
        int hits = 0;
        for (String q : queryTerms) if (st.contains(q)) hits++;
        double lengthPenalty = 1.0 / (1.0 + Math.log(1.0 + sentence.length() / 80.0));
        return hits * (0.7 + 0.3 * lengthPenalty);
    }

    private static double jaccard(Set<String> a, Set<String> b) {
        if (a.isEmpty() || b.isEmpty()) return 0.0;
        int inter = 0;
        for (String x : a) if (b.contains(x)) inter++;
        int union = a.size() + b.size() - inter;
        return union == 0 ? 0.0 : (double) inter / union;
    }

    /** Lowercased alphanumeric tokens of length > 2, minus stopwords. No regex. */
    private static Set<String> terms(String s) {
        Set<String> out = new HashSet<>();
        if (s == null) return out;
        String lower = s.toLowerCase();
        StringBuilder cur = new StringBuilder();
        for (int i = 0; i <= lower.length(); i++) {
            char c = i < lower.length() ? lower.charAt(i) : ' ';
            if ((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')) {
                cur.append(c);
            } else {
                if (cur.length() > 2) {
                    String t = cur.toString();
                    if (!STOPWORDS.contains(t)) out.add(t);
                }
                cur.setLength(0);
            }
        }
        return out;
    }

    /** Splits into sentences after collapsing the chunk separators + whitespace. No regex. */
    private static List<String> splitSentences(String text) {
        String stripped = text.replace("\n---\n", " ");
        StringBuilder norm = new StringBuilder();
        boolean prevSpace = false;
        for (int i = 0; i < stripped.length(); i++) {
            char c = stripped.charAt(i);
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f') {
                if (!prevSpace) { norm.append(' '); prevSpace = true; }
            } else {
                norm.append(c);
                prevSpace = false;
            }
        }
        String s = norm.toString().trim();

        List<String> out = new ArrayList<>();
        int start = 0;
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            if (c == '.' || c == '!' || c == '?') {
                String sentence = s.substring(start, i + 1).trim();
                if (!sentence.isEmpty()) out.add(sentence);
                start = i + 1;
            }
        }
        if (start < s.length()) {
            String tail = s.substring(start).trim();
            if (!tail.isEmpty()) out.add(tail);
        }
        return out;
    }

    private ContextCompressor() {}
}

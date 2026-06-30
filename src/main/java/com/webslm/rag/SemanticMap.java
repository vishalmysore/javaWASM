package com.webslm.rag;

import java.util.List;

import org.teavm.jso.canvas.CanvasRenderingContext2D;
import org.teavm.jso.dom.html.HTMLCanvasElement;
import org.teavm.jso.dom.html.HTMLDocument;

/**
 * An interactive "semantic map" of the indexed document, computed and drawn
 * entirely by Java compiled to WebAssembly.
 *
 * <p>Java runs <b>PCA</b> (top-2 principal components via power iteration) to
 * project the 384-dim chunk embeddings down to 2D, and <b>k-means</b> to cluster
 * them into topics &mdash; both implemented from scratch, no libraries. It then
 * paints the scatter onto an HTML {@code <canvas>} via TeaVM's JSO Canvas API and
 * answers hover/click hit-tests. The dimensionality reduction + clustering are
 * the business logic; the rendering is UI, both in the Wasm core.</p>
 */
public final class SemanticMap {

    private static final String[] PALETTE = {
        "#38bdf8", "#f59e0b", "#10b981", "#a855f7", "#ef4444", "#eab308"
    };

    // Rendered point data, kept for hover/click hit-testing.
    private static double[] px = new double[0];
    private static double[] py = new double[0];
    private static int[] cluster = new int[0];
    private static String[] texts = new String[0];
    private static int count = 0;

    static void render(BrowserVectorDB db) {
        HTMLDocument doc = HTMLDocument.current();
        HTMLCanvasElement canvas = (HTMLCanvasElement) doc.getElementById("map-canvas");
        if (canvas == null) return;
        CanvasRenderingContext2D ctx = (CanvasRenderingContext2D) canvas.getContext("2d");
        int w = canvas.getWidth();
        int h = canvas.getHeight();

        ctx.setFillStyle("#0f172a");
        ctx.fillRect(0, 0, w, h);

        List<BrowserVectorDB.DocumentChunk> chunks = db.chunks();
        int n = chunks.size();
        if (n < 2) {
            count = 0;
            ctx.setFillStyle("#94a3b8");
            ctx.setFont("14px sans-serif");
            ctx.fillText("Index a document (2+ chunks) first, then render the map.", 20, h / 2.0);
            NativeAIBridge.logFromWasm("SemanticMap: not enough chunks to map (" + n + ").");
            return;
        }

        int dim = chunks.get(0).vector.length;
        double[][] data = new double[n][dim];
        double[] mean = new double[dim];
        for (int i = 0; i < n; i++) {
            float[] v = chunks.get(i).vector;
            for (int d = 0; d < dim; d++) {
                data[i][d] = v[d];
                mean[d] += v[d];
            }
        }
        for (int d = 0; d < dim; d++) mean[d] /= n;
        for (int i = 0; i < n; i++) {
            for (int d = 0; d < dim; d++) data[i][d] -= mean[d];
        }

        // PCA: top-2 principal components by power iteration.
        double[] pc1 = powerIteration(data, null, 32);
        double[] pc2 = powerIteration(data, pc1, 32);

        double[] c1 = new double[n];
        double[] c2 = new double[n];
        for (int i = 0; i < n; i++) {
            c1[i] = dot(data[i], pc1);
            c2[i] = dot(data[i], pc2);
        }

        // k-means clustering for the colors.
        int k = Math.min(PALETTE.length, n);
        int[] assign = kmeans(data, k, 16);

        double min1 = Double.MAX_VALUE, max1 = -Double.MAX_VALUE;
        double min2 = Double.MAX_VALUE, max2 = -Double.MAX_VALUE;
        for (int i = 0; i < n; i++) {
            min1 = Math.min(min1, c1[i]); max1 = Math.max(max1, c1[i]);
            min2 = Math.min(min2, c2[i]); max2 = Math.max(max2, c2[i]);
        }

        double pad = 34;
        px = new double[n];
        py = new double[n];
        cluster = new int[n];
        texts = new String[n];
        count = n;
        for (int i = 0; i < n; i++) {
            px[i] = mapRange(c1[i], min1, max1, pad, w - pad);
            py[i] = mapRange(c2[i], min2, max2, h - pad, pad); // invert y so "up" is positive
            cluster[i] = assign[i];
            texts[i] = chunks.get(i).text;
        }

        for (int i = 0; i < n; i++) {
            ctx.setFillStyle(PALETTE[cluster[i] % PALETTE.length]);
            ctx.beginPath();
            ctx.arc(px[i], py[i], 5, 0, Math.PI * 2);
            ctx.fill();
        }

        ctx.setFillStyle("#94a3b8");
        ctx.setFont("12px sans-serif");
        ctx.fillText("PCA + k-means computed in Java/Wasm — " + n + " chunks, " + k + " clusters. Hover/click a point.", 12, 18);

        NativeAIBridge.logFromWasm("SemanticMap: PCA(2D) + k-means in Java rendered "
            + n + " chunks into " + k + " clusters on canvas.");
    }

    /** Nearest chunk text to a canvas (x,y), or "" if no point is close enough. */
    static String nearestText(double x, double y) {
        double best = Double.MAX_VALUE;
        int bi = -1;
        for (int i = 0; i < count; i++) {
            double dx = px[i] - x, dy = py[i] - y;
            double d = dx * dx + dy * dy;
            if (d < best) { best = d; bi = i; }
        }
        return (bi >= 0 && best <= 196.0) ? texts[bi] : ""; // within 14px
    }

    // ---- linear algebra (pure Java) ----

    private static double[] powerIteration(double[][] data, double[] ortho, int iters) {
        int n = data.length, dim = data[0].length;
        double[] v = new double[dim];
        for (int d = 0; d < dim; d++) v[d] = 1.0 + 0.001 * d; // deterministic, non-degenerate
        normalize(v);
        for (int it = 0; it < iters; it++) {
            double[] cv = new double[dim];
            for (int i = 0; i < n; i++) {
                double s = dot(data[i], v);
                double[] row = data[i];
                for (int d = 0; d < dim; d++) cv[d] += s * row[d];
            }
            for (int d = 0; d < dim; d++) cv[d] /= n;
            if (ortho != null) { // deflate: keep orthogonal to the first component
                double dp = dot(cv, ortho);
                for (int d = 0; d < dim; d++) cv[d] -= dp * ortho[d];
            }
            normalize(cv);
            v = cv;
        }
        return v;
    }

    private static int[] kmeans(double[][] data, int k, int iters) {
        int n = data.length, dim = data[0].length;
        double[][] cent = new double[k][dim];
        for (int c = 0; c < k; c++) {
            int idx = (int) ((long) c * n / k);
            System.arraycopy(data[idx], 0, cent[c], 0, dim);
        }
        int[] assign = new int[n];
        for (int it = 0; it < iters; it++) {
            for (int i = 0; i < n; i++) {
                double best = Double.MAX_VALUE;
                int bc = 0;
                for (int c = 0; c < k; c++) {
                    double dd = dist2(data[i], cent[c]);
                    if (dd < best) { best = dd; bc = c; }
                }
                assign[i] = bc;
            }
            double[][] sum = new double[k][dim];
            int[] cnt = new int[k];
            for (int i = 0; i < n; i++) {
                int c = assign[i];
                cnt[c]++;
                for (int d = 0; d < dim; d++) sum[c][d] += data[i][d];
            }
            for (int c = 0; c < k; c++) {
                if (cnt[c] > 0) {
                    for (int d = 0; d < dim; d++) cent[c][d] = sum[c][d] / cnt[c];
                }
            }
        }
        return assign;
    }

    private static double dot(double[] a, double[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) s += a[i] * b[i];
        return s;
    }

    private static double dist2(double[] a, double[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            double d = a[i] - b[i];
            s += d * d;
        }
        return s;
    }

    private static void normalize(double[] v) {
        double s = 0;
        for (double x : v) s += x * x;
        double norm = Math.sqrt(s);
        if (norm > 0) {
            for (int i = 0; i < v.length; i++) v[i] /= norm;
        }
    }

    private static double mapRange(double v, double inMin, double inMax, double outMin, double outMax) {
        if (inMax - inMin < 1e-12) return (outMin + outMax) / 2.0;
        return outMin + (v - inMin) / (inMax - inMin) * (outMax - outMin);
    }

    private SemanticMap() {}
}

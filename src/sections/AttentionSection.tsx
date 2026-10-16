import { useMemo, useState } from "react";
import { dot, softmax } from "@/lib/linalg";

// deterministic pseudo-random embedding per token (d = 6)
function tokenVec(tok: string): number[] {
  let h = 2166136261;
  for (const c of tok) {
    h ^= c.charCodeAt(0);
    h = Math.imul(h, 16777619);
  }
  return Array.from({ length: 6 }, (_, i) => {
    h ^= h << 13;
    h ^= h >>> 17;
    h ^= h << 5;
    return ((h >>> 0) / 4294967295 - 0.5) * 4 + i * 0.1;
  });
}

export default function AttentionSection() {
  const [text, setText] = useState("the cat sat on the mat because it was tired");
  const tokens = useMemo(
    () =>
      text
        .toLowerCase()
        .replace(/[^a-z0-9'\s]/g, "")
        .split(/\s+/)
        .filter(Boolean)
        .slice(0, 12),
    [text]
  );
  const [focus, setFocus] = useState<number | null>(null);

  const { scores } = useMemo(() => {
    const vecs = tokens.map(tokenVec);
    const d = 6;
    const raw = vecs.map((q) => vecs.map((k) => dot(q, k) / Math.sqrt(d)));
    const scores = raw.map((row) => softmax(row));
    return { scores };
  }, [tokens]);

  const cell = 34;
  const label = 78;

  return (
    <section id="attention" className="mx-auto max-w-6xl px-6 py-20">
      <p className="font-mono text-sm text-pink-400">03 · attention</p>
      <h2 className="mt-2 text-3xl font-bold">Attention is a softmax over dot products</h2>
      <p className="mt-3 max-w-2xl text-zinc-400">
        The core of a Transformer: attention(Q,K,V) = softmax(QKᵀ/√d)·V. Each
        token computes a dot product against every other token, and a softmax
        turns those scores into weights. Edit the sentence — the heatmap is
        computed live with real matrix math.
      </p>

      <div className="mt-8">
        <input
          value={text}
          onChange={(e) => setText(e.target.value)}
          className="w-full max-w-xl rounded-lg border border-zinc-700 bg-zinc-900 px-4 py-2.5 font-mono text-sm text-zinc-200 outline-none focus:border-pink-400"
          placeholder="type a sentence (up to 12 words)…"
        />
      </div>

      <div className="mt-8 grid gap-8 lg:grid-cols-[1.4fr_1fr]">
        <div className="overflow-x-auto rounded-xl border border-zinc-800 bg-zinc-900/50 p-4">
          {tokens.length === 0 ? (
            <p className="p-8 text-center text-zinc-500">type something above ↑</p>
          ) : (
            <svg
              width={label + tokens.length * cell + 10}
              height={label + tokens.length * cell + 30}
              className="mx-auto"
            >
              {tokens.map((t, i) => (
                <text
                  key={`c${i}`}
                  x={label + i * cell + cell / 2}
                  y={label - 8}
                  textAnchor="middle"
                  fontSize={11}
                  className="fill-zinc-300 font-mono"
                >
                  {t}
                </text>
              ))}
              {tokens.map((t, i) => (
                <text
                  key={`r${i}`}
                  x={label - 8}
                  y={label + i * cell + cell / 2 + 4}
                  textAnchor="end"
                  fontSize={11}
                  className="fill-zinc-300 font-mono"
                >
                  {t}
                </text>
              ))}
              {scores.map((row, i) =>
                row.map((s, j) => (
                  <g key={`${i}-${j}`}>
                    <rect
                      x={label + j * cell + 1}
                      y={label + i * cell + 1}
                      width={cell - 2}
                      height={cell - 2}
                      rx={4}
                      fill={`rgba(244,114,182,${0.08 + 0.92 * s})`}
                      onMouseEnter={() => setFocus(i * 100 + j)}
                      onMouseLeave={() => setFocus(null)}
                    />
                    <text
                      x={label + j * cell + cell / 2}
                      y={label + i * cell + cell / 2 + 3}
                      textAnchor="middle"
                      fontSize={9}
                      className={`pointer-events-none font-mono ${s > 0.5 ? "fill-zinc-900" : "fill-zinc-400"}`}
                    >
                      {s.toFixed(2)}
                    </text>
                  </g>
                ))
              )}
            </svg>
          )}
          <p className="mt-2 text-center text-xs text-zinc-500">
            row = query token · column = key token · value = softmax weight (each row sums to 1)
          </p>
        </div>

        <div className="flex flex-col justify-center rounded-xl border border-zinc-800 bg-zinc-900/50 p-6">
          <h3 className="font-mono text-sm text-zinc-300">
            attention(Q,K,V) = softmax( QKᵀ / √d ) · V
          </h3>
          <ol className="mt-5 list-decimal space-y-3 pl-5 text-sm text-zinc-400">
            <li>
              Each token gets three vectors: a <span className="text-zinc-200">query</span>, a{" "}
              <span className="text-zinc-200">key</span> and a{" "}
              <span className="text-zinc-200">value</span> — all learned matrices times the
              token's embedding.
            </li>
            <li>
              Every query dots with every key → a score matrix (QKᵀ).
            </li>
            <li>
              Softmax normalizes each row into weights that sum to 1 — this heatmap.
            </li>
            <li>
              The output is a weighted sum of the value vectors: each token pulls
              context from the tokens it "attends" to.
            </li>
          </ol>
          <p className="mt-5 text-xs text-zinc-500">
            {focus !== null
              ? `"${tokens[Math.floor(focus / 100)]}" attends to "${tokens[focus % 100]}" with weight ${scores[Math.floor(focus / 100)][focus % 100].toFixed(3)}`
              : "hover a cell to read the attention weight"}
          </p>
        </div>
      </div>
    </section>
  );
}

import { useEffect, useMemo, useRef, useState } from "react";
import { dot, softmax } from "@/lib/linalg";

/* ------------------------------------------------------------------ */
/* Toy language model — everything below runs real math in the browser */
/* ------------------------------------------------------------------ */

const D = 8; // embedding dimension

// deterministic pseudo-embedding per token
function embed(tok: string): number[] {
  let h = 2166136261;
  for (const c of tok) {
    h ^= c.charCodeAt(0);
    h = Math.imul(h, 16777619);
  }
  return Array.from({ length: D }, (_, i) => {
    h ^= h << 13;
    h ^= h >>> 17;
    h ^= h << 5;
    return ((h >>> 0) / 4294967295 - 0.5) * 3 + Math.sin(i + tok.length) * 0.5;
  });
}

// tiny output vocabulary the model may "say"
const VOCAB = [
  "the", "a", "of", "and", "is", "to", "in", "it", "that", "with",
  "vector", "matrix", "token", "model", "data", "learning", "network",
  "space", "number", "word", "weight", "value", "layer", "signal",
  ".", ",", "…",
];

// simple tokenizer: words + punctuation
function tokenize(text: string): string[] {
  return (
    text
      .toLowerCase()
      .match(/[a-z0-9']+|[.,!?…]/g) ?? []
  ).slice(0, 16);
}

// attention context: last token as query, softmax over all keys, weighted sum of values
function attend(tokens: string[]): { ctx: number[]; weights: number[] } {
  const vecs = tokens.map(embed);
  const q = vecs[vecs.length - 1];
  const raw = vecs.map((k) => dot(q, k) / Math.sqrt(D));
  const weights = softmax(raw);
  const ctx = Array.from({ length: D }, (_, i) =>
    vecs.reduce((s, v, j) => s + weights[j] * v[i], 0)
  );
  return { ctx, weights };
}

// next-token distribution: context · vocab embeddings, plus a bigram-ish bias
function nextTokenDist(tokens: string[]): { word: string; p: number }[] {
  const { ctx } = attend(tokens);
  const last = tokens[tokens.length - 1] ?? "";
  const scores = VOCAB.map((w) => {
    let s = dot(ctx, embed(w)) / D;
    // bigram-ish priors so output reads plausibly
    if (/[.,!?…]/.test(last) && /^[a-z]/.test(w)) s += 0.8;
    if (["is", "are", "was"].includes(last) && ["a", "the"].includes(w)) s += 1.2;
    if (["a", "the"].includes(last) && !["a", "the", "is", "of"].includes(w)) s += 1.0;
    if (/[.,!?…]/.test(w) && /^[a-z]/.test(last)) s += 0.4;
    if (tokens.length > 4 && w === ".") s += 0.6;
    return s;
  });
  const probs = softmax(scores, 0.8);
  return VOCAB.map((w, i) => ({ word: w, p: probs[i] })).sort((a, b) => b.p - a.p);
}

/* ------------------------------------------------------------------ */

type Stage = "idle" | "tokenize" | "embed" | "attend" | "predict" | "generate" | "done";

const STAGES: { id: Exclude<Stage, "idle" | "done">; label: string; blurb: string }[] = [
  { id: "tokenize", label: "1 · tokenize", blurb: "Text is chopped into tokens — the atoms of the model." },
  { id: "embed", label: "2 · embed", blurb: "Each token becomes a vector of numbers." },
  { id: "attend", label: "3 · attend", blurb: "Tokens look at each other: softmax(QKᵀ/√d)·V mixes context." },
  { id: "predict", label: "4 · predict", blurb: "The context vector scores every word in the vocabulary." },
  { id: "generate", label: "5 · generate", blurb: "One token is picked, appended, and the loop repeats." },
];

export default function PipelineSection() {
  const [prompt, setPrompt] = useState("a neural network is");
  const [stage, setStage] = useState<Stage>("idle");
  const [output, setOutput] = useState<string[]>([]);
  const [step, setStep] = useState(0);
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

  const tokens = useMemo(() => tokenize(prompt), [prompt]);
  const { weights } = useMemo(() => attend(tokens.length ? tokens : ["·"]), [tokens]);
  const dist = useMemo(() => nextTokenDist([...tokens, ...output]), [tokens, output]);

  const run = () => {
    clearTimeout(timer.current);
    setOutput([]);
    setStep(0);
    if (tokens.length === 0) return;
    setStage("tokenize");
  };

  // stage sequencing
  useEffect(() => {
    if (stage === "idle" || stage === "done") return;
    const durations: Record<string, number> = { tokenize: 1400, embed: 1800, attend: 1800, predict: 1600, generate: 700 };
    timer.current = setTimeout(() => {
      if (stage === "generate") {
        if (step >= 7) {
          setStage("done");
          return;
        }
        // greedy pick from current distribution
        const pick = nextTokenDist([...tokens, ...output])[0].word;
        setOutput((o) => [...o, pick]);
        setStep((s) => s + 1);
        return;
      }
      const order: Stage[] = ["tokenize", "embed", "attend", "predict", "generate"];
      const next = order[order.indexOf(stage) + 1];
      setStage(next);
    }, durations[stage]);
    return () => clearTimeout(timer.current);
  }, [stage, step, tokens, output]);

  const stageIdx = stage === "idle" ? -1 : stage === "done" ? 5 : STAGES.findIndex((s) => s.id === stage);
  const show = (i: number) => stageIdx >= i;

  return (
    <section id="pipeline" className="mx-auto max-w-6xl px-6 py-20">
      <p className="font-mono text-sm text-violet-400">05 · the full loop</p>
      <h2 className="mt-2 text-3xl font-bold">Watch a prompt become an answer</h2>
      <p className="mt-3 max-w-2xl text-zinc-400">
        Type a prompt and run it through a toy language model. Every step below
        is real math happening live in your browser — a miniature, slowed-down
        version of what happens millions of times per second inside an LLM.
      </p>

      <div className="mt-8 flex gap-2">
        <input
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && run()}
          placeholder="type a prompt…"
          className="min-w-0 flex-1 rounded-lg border border-zinc-700 bg-zinc-900 px-4 py-2.5 font-mono text-sm text-zinc-200 outline-none focus:border-violet-400"
        />
        <button
          onClick={run}
          className="rounded-lg bg-violet-500 px-6 py-2.5 text-sm font-semibold text-zinc-950 hover:bg-violet-400"
        >
          run ▸
        </button>
      </div>

      {/* stage tracker */}
      <div className="mt-8 flex flex-wrap gap-2">
        {STAGES.map((s, i) => (
          <div
            key={s.id}
            className={`rounded-full border px-4 py-1.5 font-mono text-xs transition ${
              i === stageIdx
                ? "border-violet-400 bg-violet-400/10 text-violet-300"
                : i < stageIdx
                  ? "border-zinc-700 text-zinc-500 line-through"
                  : "border-zinc-800 text-zinc-600"
            }`}
          >
            {s.label}
          </div>
        ))}
      </div>
      {stage !== "idle" && (
        <p className="mt-3 text-sm text-zinc-400">
          {stage === "done" ? "Done — the model generated its answer one token at a time." : STAGES[stageIdx]?.blurb}
        </p>
      )}

      <div className="mt-8 grid gap-6 lg:grid-cols-2">
        {/* 1-2: tokens + embeddings */}
        <div className={`rounded-xl border border-zinc-800 bg-zinc-900/50 p-5 transition-opacity ${show(0) ? "opacity-100" : "opacity-30"}`}>
          <h3 className="font-mono text-sm text-zinc-300">tokens → vectors</h3>
          <div className="mt-4 flex flex-wrap gap-2">
            {tokens.map((t, i) => (
              <div key={i} className="rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2">
                <span className={`font-mono text-sm ${i === tokens.length - 1 ? "text-violet-300" : "text-zinc-200"}`}>
                  {t}
                </span>
                {show(1) && (
                  <div className="mt-1.5 flex gap-[3px]">
                    {embed(t).map((v, j) => (
                      <div
                        key={j}
                        className="h-4 w-[7px] rounded-sm"
                        style={{ backgroundColor: v >= 0 ? `rgba(52,211,153,${Math.min(Math.abs(v) / 1.5, 1)})` : `rgba(244,114,182,${Math.min(Math.abs(v) / 1.5, 1)})` }}
                        title={v.toFixed(2)}
                      />
                    ))}
                  </div>
                )}
              </div>
            ))}
            {tokens.length === 0 && <p className="text-sm text-zinc-600">type a prompt and press run ▸</p>}
          </div>
          {show(1) && (
            <p className="mt-3 text-xs text-zinc-500">
              Each bar strip is a token's {D}-dimensional embedding — green positive, pink negative.
            </p>
          )}
        </div>

        {/* 3: attention weights */}
        <div className={`rounded-xl border border-zinc-800 bg-zinc-900/50 p-5 transition-opacity ${show(2) ? "opacity-100" : "opacity-30"}`}>
          <h3 className="font-mono text-sm text-zinc-300">context via attention</h3>
          <div className="mt-4 space-y-1.5">
            {tokens.map((t, i) => (
              <div key={i} className="flex items-center gap-2">
                <span className="w-16 truncate text-right font-mono text-xs text-zinc-400">{t}</span>
                <div className="h-3 flex-1 overflow-hidden rounded bg-zinc-800">
                  <div
                    className="h-full rounded bg-gradient-to-r from-violet-500 to-fuchsia-400 transition-all duration-500"
                    style={{ width: show(2) ? `${weights[i] * 100}%` : "0%" }}
                  />
                </div>
                <span className="w-12 font-mono text-xs text-zinc-500">{show(2) ? weights[i].toFixed(2) : ""}</span>
              </div>
            ))}
          </div>
          {show(2) && (
            <p className="mt-3 text-xs text-zinc-500">
              How much the last token "listens" to each previous token. The
              weighted mix is the context vector used for prediction.
            </p>
          )}
        </div>

        {/* 4: prediction distribution */}
        <div className={`rounded-xl border border-zinc-800 bg-zinc-900/50 p-5 transition-opacity ${show(3) ? "opacity-100" : "opacity-30"}`}>
          <h3 className="font-mono text-sm text-zinc-300">next-token probabilities</h3>
          <div className="mt-4 space-y-1.5">
            {dist.slice(0, 8).map(({ word, p }) => (
              <div key={word} className="flex items-center gap-2">
                <span className="w-16 truncate text-right font-mono text-xs text-zinc-400">{word === " " ? "␣" : word}</span>
                <div className="h-3 flex-1 overflow-hidden rounded bg-zinc-800">
                  <div
                    className="h-full rounded bg-gradient-to-r from-sky-500 to-violet-400 transition-all duration-500"
                    style={{ width: show(3) ? `${p * 100}%` : "0%" }}
                  />
                </div>
                <span className="w-12 font-mono text-xs text-zinc-500">{show(3) ? p.toFixed(2) : ""}</span>
              </div>
            ))}
          </div>
          {show(3) && (
            <p className="mt-3 text-xs text-zinc-500">
              softmax over dot products between the context vector and every
              vocabulary embedding.
            </p>
          )}
        </div>

        {/* 5: generated output */}
        <div className={`rounded-xl border border-zinc-800 bg-zinc-900/50 p-5 transition-opacity ${show(4) ? "opacity-100" : "opacity-30"}`}>
          <h3 className="font-mono text-sm text-zinc-300">generated answer</h3>
          <div className="mt-4 min-h-16 rounded-lg border border-zinc-700 bg-zinc-950 p-4 font-mono text-sm leading-relaxed">
            <span className="text-zinc-500">{prompt} </span>
            {output.map((t, i) => (
              <span key={i} className={i === output.length - 1 ? "text-violet-300" : "text-zinc-200"}>
                {/^[.,!?…]/.test(t) ? t : " " + t}
              </span>
            ))}
            {(stage === "generate") && <span className="animate-pulse text-violet-400">▌</span>}
            {stage === "done" && <span className="text-zinc-600"> ▏end</span>}
          </div>
          <p className="mt-3 text-xs text-zinc-500">
            Each new token is appended to the input and the whole loop —
            embed → attend → predict — runs again. That loop is all an LLM does.
          </p>
        </div>
      </div>
    </section>
  );
}

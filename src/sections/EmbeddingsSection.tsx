import { useEffect, useMemo, useRef, useState } from "react";
import { cosineSim, pca2 } from "@/lib/linalg";

// Hand-crafted 8D "embeddings": each semantic axis is a dimension.
// axes: [royalty, animal, food, tech, gender-masc, size, warmth, urbanity]
const BASE: Record<string, number[]> = {
  king: [1.0, 0.1, 0.0, 0.1, 0.9, 0.7, 0.3, 0.8],
  queen: [1.0, 0.1, 0.0, 0.1, 0.1, 0.7, 0.4, 0.8],
  prince: [0.9, 0.1, 0.0, 0.1, 0.9, 0.4, 0.3, 0.8],
  princess: [0.9, 0.1, 0.0, 0.1, 0.1, 0.4, 0.4, 0.8],
  man: [0.3, 0.1, 0.0, 0.2, 0.9, 0.7, 0.4, 0.7],
  woman: [0.3, 0.1, 0.0, 0.2, 0.1, 0.7, 0.5, 0.7],
  cat: [0.0, 0.95, 0.2, 0.0, 0.5, 0.2, 0.7, 0.7],
  dog: [0.0, 0.95, 0.2, 0.0, 0.5, 0.4, 0.8, 0.6],
  lion: [0.4, 0.9, 0.1, 0.0, 0.6, 0.8, 0.2, 0.2],
  tiger: [0.2, 0.9, 0.1, 0.0, 0.5, 0.8, 0.1, 0.2],
  pizza: [0.0, 0.0, 1.0, 0.1, 0.5, 0.5, 0.8, 0.7],
  pasta: [0.0, 0.0, 0.95, 0.1, 0.5, 0.4, 0.8, 0.6],
  sushi: [0.1, 0.1, 0.9, 0.2, 0.5, 0.3, 0.5, 0.9],
  burger: [0.0, 0.0, 0.9, 0.1, 0.5, 0.5, 0.7, 0.7],
  computer: [0.0, 0.0, 0.0, 1.0, 0.5, 0.3, 0.1, 0.9],
  robot: [0.1, 0.1, 0.0, 0.95, 0.5, 0.6, 0.1, 0.8],
  software: [0.0, 0.0, 0.0, 0.9, 0.5, 0.2, 0.1, 0.9],
  algorithm: [0.1, 0.0, 0.0, 0.85, 0.5, 0.1, 0.0, 0.8],
};

// deterministic 8D vector for user-added words
function hashVec(word: string): number[] {
  let h = 2166136261;
  for (const c of word) {
    h ^= c.charCodeAt(0);
    h = Math.imul(h, 16777619);
  }
  return Array.from({ length: 8 }, () => {
    h ^= h << 13;
    h ^= h >>> 17;
    h ^= h << 5;
    return (h >>> 0) / 4294967295;
  });
}

export default function EmbeddingsSection() {
  const [selected, setSelected] = useState<string>("king");
  const [extra, setExtra] = useState<Record<string, number[]>>({});
  const [newWord, setNewWord] = useState("");
  // arithmetic: a - b + c
  const [a, setA] = useState("king");
  const [b, setB] = useState("man");
  const [c, setC] = useState("woman");
  const [anim, setAnim] = useState(0); // 0..1 progress of result marker
  const animRef = useRef<number | undefined>(undefined);

  const EMB = useMemo(() => ({ ...BASE, ...extra }), [extra]);
  const words = useMemo(() => Object.keys(EMB), [EMB]);
  const points = useMemo(() => pca2(words.map((w) => EMB[w])), [words, EMB]);

  const posOf = useMemo(() => {
    const maxAbs = Math.max(...points.flatMap((p) => [Math.abs(p.x), Math.abs(p.y)]), 0.001);
    const map: Record<string, { x: number; y: number }> = {};
    points.forEach((p, i) => {
      map[words[i]] = { x: (p.x / maxAbs) * 190, y: (-p.y / maxAbs) * 150 };
    });
    return map;
  }, [points, words]);

  // arithmetic result
  const result = useMemo(() => {
    const r = EMB[a].map((v, i) => v - EMB[b][i] + EMB[c][i]);
    let best = "";
    let bestSim = -Infinity;
    for (const w of words) {
      if (w === a || w === b || w === c) continue;
      const s = cosineSim(r, EMB[w]);
      if (s > bestSim) {
        bestSim = s;
        best = w;
      }
    }
    return { vec: r, nearest: best, sim: bestSim };
  }, [a, b, c, EMB, words]);

  // animate result marker from a to nearest
  useEffect(() => {
    cancelAnimationFrame(animRef.current!);
    const start = performance.now();
    const loop = (now: number) => {
      const t = Math.min((now - start) / 900, 1);
      setAnim(t * t * (3 - 2 * t));
      if (t < 1) animRef.current = requestAnimationFrame(loop);
    };
    animRef.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animRef.current!);
  }, [result.nearest, a, b, c]);

  const sims = useMemo(
    () =>
      words.map((w) => ({ word: w, sim: cosineSim(EMB[selected], EMB[w]) })),
    [selected, words, EMB]
  );

  const addWord = () => {
    const w = newWord.trim().toLowerCase();
    if (!w || EMB[w]) return;
    setExtra((e) => ({ ...e, [w]: hashVec(w) }));
    setNewWord("");
    setSelected(w);
  };

  const pA = posOf[a];
  const pR = posOf[result.nearest];
  const marker = pA && pR ? { x: pA.x + (pR.x - pA.x) * anim, y: pA.y + (pR.y - pA.y) * anim } : null;

  return (
    <section id="embeddings" className="mx-auto max-w-6xl px-6 py-20">
      <p className="font-mono text-sm text-emerald-400">01 · vectors</p>
      <h2 className="mt-2 text-3xl font-bold">Words become vectors</h2>
      <p className="mt-3 max-w-2xl text-zinc-400">
        AI models turn every word into a list of numbers — a point in a
        high-dimensional space. Meaning is geometry: similar words land close
        together, and you can even do math with them.
      </p>

      <div className="mt-10 grid gap-8 lg:grid-cols-[1.4fr_1fr]">
        <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-4">
          <svg viewBox="-220 -180 440 360" className="w-full">
            <line x1={-220} y1={0} x2={220} y2={0} stroke="#3f3f46" strokeWidth={0.6} />
            <line x1={0} y1={-180} x2={0} y2={180} stroke="#3f3f46" strokeWidth={0.6} />
            {/* arithmetic trail */}
            {pA && pR && (
              <line x1={pA.x} y1={pA.y} x2={pR.x} y2={pR.y} stroke="#fbbf24" strokeWidth={1} strokeDasharray="4 4" opacity={0.5} />
            )}
            {points.map((p, i) => {
              const x = (p.x / Math.max(...points.map((q) => Math.abs(q.x)), 0.001)) * 190;
              const y = (-p.y / Math.max(...points.map((q) => Math.abs(q.y)), 0.001)) * 150;
              const sim = sims[i].sim;
              const w = words[i];
              const isSel = w === selected;
              const isResult = w === result.nearest;
              return (
                <g key={w} onClick={() => setSelected(w)} className="cursor-pointer">
                  <circle
                    cx={x}
                    cy={y}
                    r={isSel ? 8 : isResult ? 7 : 5}
                    fill={isSel ? "#34d399" : isResult ? "#fbbf24" : `rgba(96,165,250,${0.25 + 0.75 * Math.max(sim, 0)})`}
                  >
                    <title>{`${w} · similarity ${sim.toFixed(3)}`}</title>
                  </circle>
                  <text x={x + 9} y={y + 4} fontSize={11} fill={isSel ? "#34d399" : isResult ? "#fbbf24" : "#d4d4d8"} className="select-none font-mono">
                    {w}
                  </text>
                </g>
              );
            })}
            {marker && <circle cx={marker.x} cy={marker.y} r={4} fill="none" stroke="#fbbf24" strokeWidth={1.5} />}
          </svg>
          <p className="mt-2 text-center text-xs text-zinc-500">
            8-dimensional embeddings projected to 2D with PCA (computed live in your browser)
          </p>
        </div>

        <div className="space-y-6">
          {/* vector arithmetic */}
          <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-5">
            <h3 className="font-mono text-sm text-zinc-300">vector arithmetic</h3>
            <div className="mt-3 flex items-center gap-2 font-mono text-sm">
              <WordSelect value={a} onChange={setA} words={words} accent="text-emerald-300" />
              <span className="text-zinc-500">−</span>
              <WordSelect value={b} onChange={setB} words={words} accent="text-sky-300" />
              <span className="text-zinc-500">+</span>
              <WordSelect value={c} onChange={setC} words={words} accent="text-pink-300" />
            </div>
            <p className="mt-4 text-center font-mono text-sm">
              ≈ <span className="rounded bg-amber-400/10 px-2 py-1 text-amber-300">{result.nearest}</span>
              <span className="ml-2 text-xs text-zinc-500">cos = {result.sim.toFixed(3)}</span>
            </p>
            <p className="mt-3 text-xs text-zinc-500">
              The classic king − man + woman ≈ queen works because relationships
              are directions in this space.
            </p>
          </div>

          {/* similarity ranking */}
          <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-5">
            <h3 className="font-mono text-sm text-zinc-300">
              cosine similarity · <span className="text-emerald-400">{selected}</span>
            </h3>
            <div className="mt-4 space-y-2">
              {[...sims].sort((x, y) => y.sim - x.sim).slice(0, 6).map(({ word, sim }) => (
                <div key={word} className="flex items-center gap-3">
                  <span className="w-20 font-mono text-xs text-zinc-300">{word}</span>
                  <div className="h-2 flex-1 overflow-hidden rounded bg-zinc-800">
                    <div className="h-full rounded bg-gradient-to-r from-sky-500 to-emerald-400 transition-all duration-300" style={{ width: `${Math.max(sim, 0) * 100}%` }} />
                  </div>
                  <span className="w-14 text-right font-mono text-xs text-zinc-400">{sim.toFixed(3)}</span>
                </div>
              ))}
            </div>
            <div className="mt-4 flex gap-2">
              <input
                value={newWord}
                onChange={(e) => setNewWord(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && addWord()}
                placeholder="add your own word…"
                className="min-w-0 flex-1 rounded-lg border border-zinc-700 bg-zinc-900 px-3 py-1.5 font-mono text-xs text-zinc-200 outline-none focus:border-emerald-400"
              />
              <button onClick={addWord} className="rounded-lg bg-emerald-500 px-3 py-1.5 text-xs font-semibold text-zinc-900 hover:bg-emerald-400">
                add
              </button>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

function WordSelect({ value, onChange, words, accent }: { value: string; onChange: (v: string) => void; words: string[]; accent: string }) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className={`min-w-0 flex-1 rounded-lg border border-zinc-700 bg-zinc-900 px-2 py-1.5 text-xs outline-none ${accent}`}
    >
      {words.map((w) => (
        <option key={w} value={w}>{w}</option>
      ))}
    </select>
  );
}

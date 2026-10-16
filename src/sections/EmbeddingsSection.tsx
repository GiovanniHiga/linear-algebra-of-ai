import { useMemo, useState } from "react";
import { cosineSim, pca2 } from "@/lib/linalg";

// Hand-crafted 8D "embeddings": each semantic axis is a dimension.
// axes: [royalty, animal, food, tech, gender-masc, size, warmth, urbanity]
const EMBEDDINGS: Record<string, number[]> = {
  king: [1.0, 0.1, 0.0, 0.1, 0.9, 0.7, 0.3, 0.8],
  queen: [1.0, 0.1, 0.0, 0.1, 0.1, 0.7, 0.4, 0.8],
  prince: [0.9, 0.1, 0.0, 0.1, 0.9, 0.4, 0.3, 0.8],
  princess: [0.9, 0.1, 0.0, 0.1, 0.1, 0.4, 0.4, 0.8],
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

const words = Object.keys(EMBEDDINGS);

export default function EmbeddingsSection() {
  const [selected, setSelected] = useState<string>("king");
  const points = useMemo(() => pca2(words.map((w) => EMBEDDINGS[w])), []);

  const sims = useMemo(
    () =>
      words.map((w) => ({
        word: w,
        sim: cosineSim(EMBEDDINGS[selected], EMBEDDINGS[w]),
      })),
    [selected]
  );
  const maxAbs = Math.max(...points.flatMap((p) => [Math.abs(p.x), Math.abs(p.y)]), 0.001);

  return (
    <section id="embeddings" className="mx-auto max-w-6xl px-6 py-20">
      <p className="font-mono text-sm text-emerald-400">01 · vectors</p>
      <h2 className="mt-2 text-3xl font-bold">Words become vectors</h2>
      <p className="mt-3 max-w-2xl text-zinc-400">
        AI models turn every word into a list of numbers — a point in a
        high-dimensional space. Meaning is geometry: similar words land close
        together. Click a word to see its cosine similarity to every other word.
      </p>

      <div className="mt-10 grid gap-8 lg:grid-cols-[1.4fr_1fr]">
        <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-4">
          <svg viewBox="-220 -180 440 360" className="w-full">
            <line x1={-220} y1={0} x2={220} y2={0} stroke="#3f3f46" strokeWidth={0.6} />
            <line x1={0} y1={-180} x2={0} y2={180} stroke="#3f3f46" strokeWidth={0.6} />
            {points.map((p, i) => {
              const x = (p.x / maxAbs) * 190;
              const y = (-p.y / maxAbs) * 150;
              const sim = sims[i].sim;
              const isSel = words[i] === selected;
              return (
                <g
                  key={words[i]}
                  onClick={() => setSelected(words[i])}
                  className="cursor-pointer"
                >
                  <circle
                    cx={x}
                    cy={y}
                    r={isSel ? 8 : 5}
                    fill={isSel ? "#34d399" : `rgba(96,165,250,${0.25 + 0.75 * sim})`}
                    stroke={isSel ? "#34d399" : "transparent"}
                  >
                    <title>{`${words[i]} · similarity ${sim.toFixed(3)}`}</title>
                  </circle>
                  <text
                    x={x + 9}
                    y={y + 4}
                    fontSize={11}
                    fill={isSel ? "#34d399" : "#d4d4d8"}
                    className="select-none font-mono"
                  >
                    {words[i]}
                  </text>
                </g>
              );
            })}
          </svg>
          <p className="mt-2 text-center text-xs text-zinc-500">
            8-dimensional embeddings projected to 2D with PCA (computed live in your browser)
          </p>
        </div>

        <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-5">
          <h3 className="font-mono text-sm text-zinc-300">
            cosine similarity · <span className="text-emerald-400">{selected}</span>
          </h3>
          <div className="mt-4 space-y-2">
            {[...sims]
              .sort((a, b) => b.sim - a.sim)
              .slice(0, 8)
              .map(({ word, sim }) => (
                <div key={word} className="flex items-center gap-3">
                  <span className="w-20 font-mono text-xs text-zinc-300">{word}</span>
                  <div className="h-2 flex-1 overflow-hidden rounded bg-zinc-800">
                    <div
                      className="h-full rounded bg-gradient-to-r from-sky-500 to-emerald-400 transition-all duration-300"
                      style={{ width: `${Math.max(sim, 0) * 100}%` }}
                    />
                  </div>
                  <span className="w-14 text-right font-mono text-xs text-zinc-400">
                    {sim.toFixed(3)}
                  </span>
                </div>
              ))}
          </div>
          <p className="mt-5 text-xs leading-relaxed text-zinc-500">
            cos(θ) = (a·b) / (‖a‖‖b‖). A dot product measures how much two
            vectors point in the same direction — the basis of semantic search
            and recommendation systems.
          </p>
        </div>
      </div>
    </section>
  );
}

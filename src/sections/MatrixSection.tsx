import { useEffect, useMemo, useState } from "react";
import { matVec } from "@/lib/linalg";

type Mat2 = [number, number, number, number];

const PRESETS: { name: string; m: Mat2 }[] = [
  { name: "rotation", m: [0, -1, 1, 0] },
  { name: "shear", m: [1, 1, 0, 1] },
  { name: "scale", m: [1.6, 0, 0, 0.6] },
  { name: "reflection", m: [1, 0, 0, -1] },
];

export default function MatrixSection() {
  const [preset, setPreset] = useState(1);
  const [t, setT] = useState(0);

  useEffect(() => {
    let raf: number;
    let start = performance.now();
    const loop = (now: number) => {
      const cycle = ((now - start) / 3200) % 1;
      // ease in-out, hold at both ends
      const u = Math.min(Math.max((cycle - 0.15) / 0.6, 0), 1);
      setT(u * u * (3 - 2 * u));
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [preset]);

  const A = useMemo(() => {
    const m = PRESETS[preset].m;
    // interpolate I -> m
    const I: Mat2 = [1, 0, 0, 1];
    return I.map((v, i) => v + (m[i] - v) * t) as Mat2;
  }, [preset, t]);

  const M = useMemo(
    () => [
      [A[0], A[1]],
      [A[2], A[3]],
    ],
    [A]
  );

  const scale = 55;
  const lines: { x1: number; y1: number; x2: number; y2: number }[] = [];
  const N = 4;
  for (let i = -N; i <= N; i++) {
    for (const j of [-N, N]) {
      const p1 = matVec(M, [i, -N]);
      const p2 = matVec(M, [i, N]);
      lines.push({ x1: p1[0], y1: p1[1], x2: p2[0], y2: p2[1] });
      const q1 = matVec(M, [-N, j * (i === -N || i === N ? 1 : 0) + i]);
      void q1;
      const r1 = matVec(M, [-N, i]);
      const r2 = matVec(M, [N, i]);
      lines.push({ x1: r1[0], y1: r1[1], x2: r2[0], y2: r2[1] });
    }
  }

  const v: [number, number] = [0.8, 0.6];
  const w = matVec(M, v);
  const basisI = matVec(M, [1, 0]);
  const basisJ = matVec(M, [0, 1]);

  return (
    <section id="matrices" className="mx-auto max-w-6xl px-6 py-20">
      <p className="font-mono text-sm text-sky-400">02 · matrices</p>
      <h2 className="mt-2 text-3xl font-bold">Matrices transform space</h2>
      <p className="mt-3 max-w-2xl text-zinc-400">
        Every layer of a neural network is a matrix multiplying a vector:
        y = Wx + b. Geometrically, that is the whole space being rotated,
        sheared and stretched. Watch the identity morph into a transformation.
      </p>

      <div className="mt-10 grid gap-8 lg:grid-cols-[1.4fr_1fr]">
        <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-4">
          <svg viewBox="-260 -260 520 520" className="w-full">
            <g>
              {lines.map((l, i) => (
                <line
                  key={i}
                  x1={l.x1 * scale}
                  y1={-l.y1 * scale}
                  x2={l.x2 * scale}
                  y2={-l.y2 * scale}
                  stroke="#27272a"
                  strokeWidth={1}
                />
              ))}
            </g>
            <line x1={-260} y1={0} x2={260} y2={0} stroke="#3f3f46" strokeWidth={1} />
            <line x1={0} y1={-260} x2={0} y2={260} stroke="#3f3f46" strokeWidth={1} />
            {/* basis vectors */}
            <line x1={0} y1={0} x2={basisI[0] * scale} y2={-basisI[1] * scale} stroke="#34d399" strokeWidth={2.5} markerEnd="url(#arrG)" />
            <line x1={0} y1={0} x2={basisJ[0] * scale} y2={-basisJ[1] * scale} stroke="#f472b6" strokeWidth={2.5} markerEnd="url(#arrP)" />
            {/* transformed vector */}
            <line x1={0} y1={0} x2={w[0] * scale} y2={-w[1] * scale} stroke="#60a5fa" strokeWidth={3} markerEnd="url(#arrB)" />
            <circle cx={w[0] * scale} cy={-w[1] * scale} r={4} fill="#60a5fa" />
            <defs>
              <marker id="arrG" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6" fill="none" stroke="#34d399" /></marker>
              <marker id="arrP" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6" fill="none" stroke="#f472b6" /></marker>
              <marker id="arrB" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6" fill="none" stroke="#60a5fa" /></marker>
            </defs>
          </svg>
          <div className="mt-3 flex flex-wrap justify-center gap-2">
            {PRESETS.map((p, i) => (
              <button
                key={p.name}
                onClick={() => setPreset(i)}
                className={`rounded-full border px-4 py-1.5 font-mono text-xs transition ${
                  i === preset
                    ? "border-sky-400 bg-sky-400/10 text-sky-300"
                    : "border-zinc-700 text-zinc-400 hover:border-zinc-500"
                }`}
              >
                {p.name}
              </button>
            ))}
          </div>
        </div>

        <div className="flex flex-col justify-center rounded-xl border border-zinc-800 bg-zinc-900/50 p-6 font-mono">
          <h3 className="text-sm text-zinc-300">y = W · x</h3>
          <div className="mt-6 flex items-center justify-center gap-4 text-lg">
            <MatrixView m={M} />
            <span className="text-zinc-500">×</span>
            <VectorView v={v as unknown as number[]} color="text-zinc-300" />
            <span className="text-zinc-500">=</span>
            <VectorView v={w} color="text-sky-300" />
          </div>
          <div className="mt-8 space-y-2 text-xs text-zinc-500">
            <p>
              <span className="text-emerald-400">— green</span> column 1: where î lands
            </p>
            <p>
              <span className="text-pink-400">— pink</span> column 2: where ĵ lands
            </p>
            <p>
              <span className="text-sky-400">— blue</span> the input vector x after W
            </p>
          </div>
          <p className="mt-6 font-sans text-xs leading-relaxed text-zinc-500">
            Stacking these transformations (with a nonlinear squish between
            them) is all a deep network does — millions of these multiplies per
            token.
          </p>
        </div>
      </div>
    </section>
  );
}

function MatrixView({ m }: { m: number[][] }) {
  return (
    <div className="grid grid-cols-2 gap-x-4 gap-y-1 rounded border border-zinc-700 p-2 text-center">
      {m.flat().map((x, i) => (
        <span key={i} className={i % 2 === 0 ? "text-emerald-300" : "text-pink-300"}>
          {x.toFixed(2)}
        </span>
      ))}
    </div>
  );
}

function VectorView({ v, color }: { v: number[]; color: string }) {
  return (
    <div className={`flex flex-col gap-1 rounded border border-zinc-700 p-2 ${color}`}>
      {v.map((x, i) => (
        <span key={i}>{x.toFixed(2)}</span>
      ))}
    </div>
  );
}

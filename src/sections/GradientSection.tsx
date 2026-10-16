import { useEffect, useRef, useState } from "react";

// A tiny "model": y = w·x + b, trained on real data with gradient descent.
const DATA: [number, number][] = [
  [0.5, 1.4], [1.0, 2.1], [1.5, 2.6], [2.0, 3.3], [2.5, 3.4],
  [3.0, 4.1], [3.5, 4.0], [4.0, 4.9], [4.5, 5.2], [5.0, 5.6],
];

function loss(w: number, b: number): number {
  return DATA.reduce((s, [x, y]) => s + (w * x + b - y) ** 2, 0) / DATA.length;
}

export default function GradientSection() {
  const [running, setRunning] = useState(false);
  const [params, setParams] = useState({ w: -0.4, b: 4.5 });
  const [lr, setLr] = useState(0.05);
  const [path, setPath] = useState<{ w: number; b: number }[]>([{ w: -0.4, b: 4.5 }]);
  const raf = useRef<number | undefined>(undefined);

  useEffect(() => {
    if (!running) return;
    let last = 0;
    const step = (now: number) => {
      if (now - last > 120) {
        last = now;
        setParams((p) => {
          // numerical gradients of MSE
          let gw = 0;
          let gb = 0;
          for (const [x, y] of DATA) {
            const err = wOf(p.w) * x + p.b - y;
            gw += (2 / DATA.length) * err * x;
            gb += (2 / DATA.length) * err;
          }
          const nw = p.w - lr * gw;
          const nb = p.b - lr * gb;
          setPath((pt) => (pt.length > 200 ? pt : [...pt, { w: nw, b: nb }]));
          return { w: nw, b: nb };
        });
      }
      raf.current = requestAnimationFrame(step);
    };
    raf.current = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf.current!);
  }, [running, lr]);

  const wOf = (w: number) => w;
  const reset = () => {
    setRunning(false);
    setParams({ w: -0.4, b: 4.5 });
    setPath([{ w: -0.4, b: 4.5 }]);
  };

  // plot scales
  const px = (x: number) => 40 + x * 62; // data x: 0..5.4
  const py = (y: number) => 300 - y * 42; // data y: 0..6.8

  // loss contour plot: w in [-1.2, 2.2], b in [0, 6]
  const cw = (w: number) => ((w + 1.2) / 3.4) * 300 + 10;
  const cb = (b: number) => 310 - (b / 6) * 300 + 10;

  return (
    <section id="training" className="mx-auto max-w-6xl px-6 py-20">
      <p className="font-mono text-sm text-amber-400">04 · learning</p>
      <h2 className="mt-2 text-3xl font-bold">Learning is sliding downhill</h2>
      <p className="mt-3 max-w-2xl text-zinc-400">
        Training means minimizing a loss function. The gradient ∇L — a vector of
        partial derivatives — points uphill, so we step the other way. Here a
        real gradient descent loop fits a line to data, live.
      </p>

      <div className="mt-10 grid gap-8 lg:grid-cols-2">
        <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-4">
          <svg viewBox="0 0 380 330" className="w-full">
            <text x={190} y={18} textAnchor="middle" fontSize={11} className="fill-zinc-500 font-mono">
              data & fitted line y = wx + b
            </text>
            <line x1={30} y1={310} x2={370} y2={310} stroke="#3f3f46" />
            <line x1={30} y1={310} x2={30} y2={20} stroke="#3f3f46" />
            <line
              x1={px(0)}
              y1={py(params.b)}
              x2={px(5.4)}
              y2={py(params.w * 5.4 + params.b)}
              stroke="#fbbf24"
              strokeWidth={2.5}
            />
            {DATA.map(([x, y], i) => (
              <circle key={i} cx={px(x)} cy={py(y)} r={4.5} fill="#60a5fa" />
            ))}
          </svg>
          <p className="text-center font-mono text-xs text-zinc-400">
            w = {params.w.toFixed(3)} · b = {params.b.toFixed(3)} · loss ={" "}
            {loss(params.w, params.b).toFixed(4)}
          </p>
        </div>

        <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-4">
          <svg viewBox="0 0 380 330" className="w-full">
            <text x={190} y={18} textAnchor="middle" fontSize={11} className="fill-zinc-500 font-mono">
              loss landscape L(w, b) — gradient descent path
            </text>
            {[0.05, 0.2, 0.5, 1.2, 2.5, 5].map((lv, i) => (
              <ellipse
                key={i}
                cx={cw(0.97)}
                cy={cb(3.1)}
                rx={30 + lv * 62}
                ry={8 + lv * 34}
                fill="none"
                stroke={`rgba(251,191,36,${0.5 - i * 0.07})`}
                strokeWidth={1}
              />
            ))}
            <polyline
              points={path.map((p) => `${cw(p.w)},${cb(p.b)}`).join(" ")}
              fill="none"
              stroke="#34d399"
              strokeWidth={2}
            />
            {path.map((p, i) => (
              <circle key={i} cx={cw(p.w)} cy={cb(p.b)} r={2.5} fill="#34d399" />
            ))}
            <text x={cw(0.97) + 8} y={cb(3.1)} fontSize={10} className="fill-amber-300 font-mono">
              minimum
            </text>
          </svg>
          <div className="mt-2 flex items-center justify-center gap-4">
            <button
              onClick={() => setRunning((r) => !r)}
              className="rounded-full bg-emerald-500 px-5 py-1.5 text-xs font-semibold text-zinc-900 hover:bg-emerald-400"
            >
              {running ? "pause" : "train"}
            </button>
            <button
              onClick={reset}
              className="rounded-full border border-zinc-600 px-5 py-1.5 text-xs text-zinc-300 hover:border-zinc-400"
            >
              reset
            </button>
            <label className="flex items-center gap-2 font-mono text-xs text-zinc-400">
              lr
              <input
                type="range"
                min={0.005}
                max={0.15}
                step={0.005}
                value={lr}
                onChange={(e) => setLr(+e.target.value)}
                className="w-24 accent-emerald-400"
              />
              {lr.toFixed(3)}
            </label>
          </div>
        </div>
      </div>
    </section>
  );
}

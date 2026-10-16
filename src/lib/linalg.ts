export type Vec = number[];
export type Mat = number[][];

export const dot = (a: Vec, b: Vec): number =>
  a.reduce((s, v, i) => s + v * b[i], 0);

export const norm = (a: Vec): number => Math.sqrt(dot(a, a));

export const cosineSim = (a: Vec, b: Vec): number =>
  dot(a, b) / (norm(a) * norm(b) || 1);

export const matVec = (m: Mat, v: Vec): Vec => m.map((row) => dot(row, v));

export const matMul = (a: Mat, b: Mat): Mat =>
  a.map((row) =>
    b[0].map((_, j) => row.reduce((s, v, k) => s + v * b[k][j], 0))
  );

export const transpose = (m: Mat): Mat => m[0].map((_, j) => m.map((r) => r[j]));

export const softmax = (v: Vec, temp = 1): Vec => {
  const scaled = v.map((x) => x / temp);
  const max = Math.max(...scaled);
  const exps = scaled.map((x) => Math.exp(x - max));
  const sum = exps.reduce((s, x) => s + x, 0);
  return exps.map((x) => x / sum);
};

/** PCA via covariance + power iteration. Returns 2D projections. */
export function pca2(data: Mat): { x: number; y: number }[] {
  const n = data.length;
  const d = data[0].length;
  const mean: Vec = Array(d).fill(0);
  data.forEach((r) => r.forEach((v, j) => (mean[j] += v / n)));
  const centered = data.map((r) => r.map((v, j) => v - mean[j]));

  // covariance matrix d x d
  const cov: Mat = Array.from({ length: d }, (_, i) =>
    Array.from({ length: d }, (_, j) =>
      centered.reduce((s, r) => s + r[i] * r[j], 0) / n
    )
  );

  const powerIter = (m: Mat, deflated?: Mat): Vec => {
    let v: Vec = Array.from({ length: d }, () => Math.random() + 0.5);
    const mm = deflated ?? m;
    for (let it = 0; it < 200; it++) {
      const w = matVec(mm, v);
      const l = norm(w) || 1;
      v = w.map((x) => x / l);
    }
    return v;
  };

  const pc1 = powerIter(cov);
  const l1 = dot(pc1, matVec(cov, pc1));
  // deflate
  const def: Mat = cov.map((row, i) => row.map((v, j) => v - l1 * pc1[i] * pc1[j]));
  const pc2 = powerIter(cov, def);

  return centered.map((r) => ({ x: dot(r, pc1), y: dot(r, pc2) }));
}

// Shared helpers for the Lecture 1 demos (no dependencies).
const NS = "http://www.w3.org/2000/svg";

function el(tag, attrs = {}, parent = null) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}

// Seeded RNG (mulberry32) so graphs are identical every lecture.
function rng(seed) {
  return function () {
    seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Cyclic Jacobi eigen-solver for small symmetric matrices.
// Returns {values, vectors} sorted ascending; vectors[k] is the k-th eigenvector.
function eigSym(Ain) {
  const n = Ain.length;
  const A = Ain.map(r => r.slice());
  const V = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
  for (let sweep = 0; sweep < 100; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += A[p][q] * A[p][q];
    if (off < 1e-22) break;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        if (Math.abs(A[p][q]) < 1e-300) continue;
        const theta = (A[q][q] - A[p][p]) / (2 * A[p][q]);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let k = 0; k < n; k++) {
          const akp = A[k][p], akq = A[k][q];
          A[k][p] = c * akp - s * akq; A[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = A[p][k], aqk = A[q][k];
          A[p][k] = c * apk - s * aqk; A[q][k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = V[k][p], vkq = V[k][q];
          V[k][p] = c * vkp - s * vkq; V[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
  const idx = [...Array(n).keys()].sort((a, b) => A[a][a] - A[b][b]);
  return {
    values: idx.map(i => A[i][i]),
    vectors: idx.map(i => V.map(row => row[i])),
  };
}

function lerp(a, b, t) { return a + (b - a) * t; }
function hex(c) { return "#" + c.map(x => Math.round(Math.max(0, Math.min(255, x))).toString(16).padStart(2, "0")).join(""); }
function mix(c1, c2, t) { return [0, 1, 2].map(i => lerp(c1[i], c2[i], t)); }

// Diverging: blue (-1) — near-white (0) — orange (+1)
const C_NEG = [37, 99, 235], C_MID = [241, 241, 238], C_POS = [234, 88, 12];
function diverging(x) { // x in [-1,1]
  x = Math.max(-1, Math.min(1, x));
  return hex(x < 0 ? mix(C_MID, C_NEG, -x) : mix(C_MID, C_POS, x));
}
// Sequential: near-white (0) → deep red (1)
const S0 = [248, 246, 240], S1 = [253, 186, 116], S2 = [185, 28, 28];
function sequential(x) {
  x = Math.max(0, Math.min(1, x));
  return hex(x < 0.5 ? mix(S0, S1, x * 2) : mix(S1, S2, (x - 0.5) * 2));
}

function fmt(x, d = 3) {
  if (x === 0) return "0";
  if (Math.abs(x) < 1e-3 || Math.abs(x) >= 1e4) return x.toExponential(1);
  return x.toFixed(d);
}

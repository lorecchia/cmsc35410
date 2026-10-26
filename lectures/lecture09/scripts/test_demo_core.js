// Tests the numerical core of demos/laplacian-solvers-demo.html (run: node test_demo_core.js)
const fs = require('fs'), path = require('path');
const html = fs.readFileSync(path.join(__dirname, '..', 'demos', 'laplacian-solvers-demo.html'), 'utf8');
const src = html.split('<script>')[1].split('</script>')[0];
const mod = { exports: {} }; new Function('module', src)(mod); const C = mod.exports;
let fails = 0; const check = (name, ok, info) => { console.log((ok ? 'ok   ' : 'FAIL ') + name + (info ? '  ' + info : '')); if (!ok) fails++; };

// 1. centroid structure vs naive sum of weighted tree distances
{ const rng = C.mulberry32(5), G = C.gridGraph(9, 1.5, rng), T = C.bfsTree(G, 3), tr = C.rootedTree(G, T, 0), ds = C.centroidDS(tr, G);
  const q = new Float64Array(G.n).map(() => rng() - 0.5); let mu = 0; q.forEach(v => mu += v); mu /= G.n; for (let i = 0; i < G.n; i++) { q[i] -= mu; ds.add(i, q[i]); }
  let err = 0, derr = 0; for (let u = 0; u < G.n; u++) { let s = 0; for (let v = 0; v < G.n; v++) s += q[v] * tr.dist(u, v); err = Math.max(err, Math.abs(s - ds.psi(u)));
    for (let v = 0; v < G.n; v += 7) derr = Math.max(derr, Math.abs(ds.dist(u, v) - tr.dist(u, v))); }
  check('Psi via centroids = naive', err < 1e-10, `max err ${err.toExponential(2)}, depth ${ds.depth} (log2 n = ${Math.log2(G.n).toFixed(2)})`);
  check('R_T via centroid LCA = via tree LCA', derr < 1e-10, derr.toExponential(2)); }

// 2. trees: sizes, stretch numbers match scripts/core.py (verify_bounds.py §7)
for (const [k, bfsAvg, recAvg] of [[16, 8.5, 3.61], [32, 16.5, 4.88], [64, 32.5, 6.34]]) {
  const G = C.gridGraph(k, 0, C.mulberry32(1));
  const res = [C.bfsTree(G, 0), C.recursiveGridTree(G)].map(T => { const tr = C.rootedTree(G, T, 0); let s = 0; for (let e = 0; e < G.m; e++) s += tr.dist(G.U[e], G.V[e]); return [T.length, s / G.m]; });
  check(`grid ${k}: trees spanning, avg stretch BFS ${res[0][1].toFixed(2)} rec ${res[1][1].toFixed(2)}`,
    res[0][0] === G.n - 1 && res[1][0] === G.n - 1 && Math.abs(res[0][1] - bfsAvg) < 0.01 && Math.abs(res[1][1] - recAvg) < 0.01); }

// 3. KOSZ: energy decreases, potentials converge, flow feasible, tau matches python (grid 64 recursive: 51014)
{ const G = C.gridGraph(64, 0, C.mulberry32(1)), chi = new Float64Array(G.n); chi[0] = 1; chi[G.n - 1] = -1;
  const K = C.kosz(G, C.recursiveGridTree(G), chi, C.mulberry32(2)); check('tau(64x64, recursive) = 51014', Math.round(K.tau) === 51014, K.tau.toFixed(1)); }
{ const G = C.gridGraph(12, 2, C.mulberry32(3)), chi = new Float64Array(G.n); chi[0] = 1; chi[G.n - 1] = -1;
  const tr = C.rootedTree(G, C.maxWeightTree(G), 0), xs = C.pcg(G, chi, null, 5000, r => tr.solve(r), 1e-14).x, tmp = new Float64Array(G.n);
  const Es = C.dot(xs, C.matvec(G, xs, tmp)), K = C.kosz(G, C.bfsTree(G, 0), chi, C.mulberry32(4));
  let prev = K.energy(), mono = true; const NS_ = Math.ceil(6 * K.tau); for (let i = 0; i < NS_; i++) { K.step(); if (i % 50 === 0) { const E = K.energy(); if (E > prev + 1e-12) mono = false; prev = E; } }
  const f = K.flow(), div = new Float64Array(G.n); for (let e = 0; e < G.m; e++) { div[G.U[e]] += f[e]; div[G.V[e]] -= f[e]; }
  let fe = 0; for (let u = 0; u < G.n; u++) fe = Math.max(fe, Math.abs(div[u] - chi[u]));
  const x = K.potentials(), err = C.Lerr(G, x, xs, tmp) / Math.sqrt(Es);
  check('KOSZ energy monotone', mono); check('KOSZ flow routes chi', fe < 1e-10, fe.toExponential(2));
  check(`KOSZ converges: after ${(NS_ / K.tau).toFixed(1)} tau, rel L-error ${err.toExponential(2)}`, err < 1e-2);
  check('E(f0) <= st_T(G) E*', C.kosz(G, C.bfsTree(G, 0), chi, C.mulberry32(4)).energy() <= K.stTotal() * Es); }

// 4. CG / PCG / Ritz values (grid 32: kappa 828.7, CG 73 its to 1e-6 for corner-to-corner)
{ const G = C.gridGraph(32, 0, C.mulberry32(1)), chi = new Float64Array(G.n); chi[0] = 1; chi[G.n - 1] = -1;
  const tr = C.rootedTree(G, C.maxWeightTree(G), 0), xs = C.pcg(G, chi, null, 5000, r => tr.solve(r), 1e-14).x;
  const cg = C.pcg(G, chi, xs, 2000, null, 1e-12), e = cg.errs, i6 = e.findIndex(v => v <= 1e-6 * e[0]);
  const [l2, ln] = C.ritzExtremes(cg.al, cg.be);
  check(`CG its to 1e-6 on 32x32 grid = ${i6} (python: 73)`, Math.abs(i6 - 73) <= 1);
  check(`Ritz kappa ${(ln / l2).toFixed(1)} (python: 828.7)`, Math.abs(ln / l2 - 828.7) / 828.7 < 0.02, `l2=${l2.toFixed(5)} ln=${ln.toFixed(3)}`); }

// 5. timing of a full demo run at the largest default size
{ const t0 = Date.now(), G = C.gridGraph(48, 4, C.mulberry32(9)), chi = new Float64Array(G.n); chi[0] = 1; chi[G.n - 1] = -1;
  const K = C.kosz(G, C.maxWeightTree(G), chi, C.mulberry32(1)); for (let i = 0; i < 200000; i++) K.step();
  check('200k toggles on 48x48 grid', true, `${Date.now() - t0} ms, tau/m = ${(K.tau / G.m).toFixed(2)}`); }
process.exit(fails ? 1 : 0);

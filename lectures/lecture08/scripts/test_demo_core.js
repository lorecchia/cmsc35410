// Extracts the numerical core of the demo and dumps instances + results for test_demo_core.py
const fs=require('fs'), path=require('path');
const html=fs.readFileSync(path.join(__dirname,'..','demos','spectral-clustering-demo.html'),'utf8');
const src=html.split('<script>')[1].split('</script>')[0];
const mod={exports:{}}; new Function('module',src)(mod); const C=mod.exports;
const out=[]; const t0=Date.now();
const dumpG=G=>{ const E=[]; for(let u=0;u<G.n;u++) G.nbr[u].forEach((v,t)=>{ if(v>u) E.push([u,v,G.wt[u][t]]); }); return E; };
for(const [n,k,p,q] of [[300,2,0.3,0.1],[300,3,0.3,0.15],[200,4,0.4,0.05],[300,2,0.3,0.0],[500,2,0.1,0.06]]){
  const G=C.sbmGraph(n,k,p,q,7); const t=Date.now(); const R=C.analyzeSBM(G,C.mulberry32(3)); const dt=Date.now()-t;
  const t2=Date.now(); const spec=C.denseSpectrum(G); const dt2=Date.now()-t2;
  out.push({type:'sbm',n,k,p,q,edges:dumpG(G),lab:Array.from(G.lab),normE:R.normE,delta0:R.delta0,sinOp:R.sinOp,sinF:R.sinF,ritz:R.ritzA,resid:R.residA,mis:R.mis,bad:R.bad,spec:spec.slice(0,4),specMin:spec[spec.length-1],ms:dt,msSpec:dt2});
}
for(const [k,m,deg,c,mode] of [[3,80,20,40,'spread'],[3,80,20,100,'hubs'],[2,60,10,0,'spread'],[4,120,30,400,'spread']]){
  const G=C.plantedGraph(k,m,deg,c,mode,5); const t=Date.now(); const R=C.analyzePlanted(G,C.mulberry32(3)); const dt=Date.now()-t;
  out.push({type:'pl',k,m,deg,c,mode,n:G.n,edges:dumpG(G),lab:Array.from(G.lab),normE:R.normE,delta0:R.delta0,mu1:R.mu1,sinOp:R.sinOp,sinF:R.sinF,energy:R.energy,epsOut:R.epsOut,lam:R.lam,resid:R.residN,mis:R.mis,ms:dt});
}
fs.writeFileSync('/tmp/demo_out.json',JSON.stringify(out)); console.log('total ms',Date.now()-t0);

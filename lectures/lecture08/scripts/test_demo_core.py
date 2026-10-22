import json, numpy as np
D=json.load(open('/tmp/demo_out.json'))
def sinth(V,Vh):
    s=np.clip(np.linalg.svd(Vh.T@V,compute_uv=False),0,1); t=np.sqrt(1-s**2); return t.max(), np.sqrt((t**2).sum())
for r in D:
    n=r['n']; k=r['k']; A=np.zeros((n,n))
    for u,v,w in r['edges']: A[u,v]=A[v,u]=w
    lab=np.array(r['lab'])
    if r['type']=='sbm':
        p,q=r['p'],r['q']; P=np.where(lab[:,None]==lab[None,:],p,q); np.fill_diagonal(P,0)
        w,U=np.linalg.eigh(A); w=w[::-1]; U=U[:,::-1]
        V=np.stack([(lab==j)/np.sqrt((lab==j).sum()) for j in range(k)],1)
        s,sF=sinth(V,U[:,:k]); nE=np.linalg.norm(A-P,2); wp=np.linalg.eigvalsh(P)[::-1]
        print(f"SBM n={n} k={k} q={q}: |E| {r['normE']:.4f} vs {nE:.4f} | d0 {r['delta0']:.3f} vs {wp[k-1]-wp[k]:.3f} | sin {r['sinOp']:.4f} vs {s:.4f} | sinF {r['sinF']:.4f} vs {sF:.4f} | top {np.round(r['ritz'],3)} vs {np.round(w[:k+1],3)} | spec {np.allclose(r['spec'],w[:4])},{abs(r['specMin']-w[-1])<1e-8} | resid {r['resid']:.1e} | mis {r['mis']} bad {r['bad']} | {r['ms']}ms +spec {r['msSpec']}ms")
    else:
        d=A.sum(1); Dm=np.diag(d**-.5); same=lab[:,None]==lab[None,:]
        Aout=A*~same; dout=Aout.sum(1); N=Dm@A@Dm; N0=Dm@(A*same+np.diag(dout))@Dm
        w,U=np.linalg.eigh(N); w=w[::-1]; U=U[:,::-1]; w0=np.linalg.eigvalsh(N0)[::-1]
        vol=np.array([d[lab==j].sum() for j in range(k)])
        V=np.stack([np.where(lab==j,np.sqrt(d/vol[j]),0) for j in range(k)],1)
        s,sF=sinth(V,U[:,:k]); nE=np.linalg.norm(N-N0,2)
        Phi=np.array([dout[lab==j].sum()/vol[j] for j in range(k)])
        print(f"PL k={k} c={r['c']} {r['mode']}: |E| {r['normE']:.4f} vs {nE:.4f} | d0 {r['delta0']:.4f} vs {w0[k-1]-w0[k]:.4f} | mu1 {r['mu1']:.4f} vs {1-w[k]:.4f} | sin {r['sinOp']:.4f} vs {s:.4f} | sinF {r['sinF']:.4f} vs {sF:.4f} | energy {r['energy']:.4f} vs {np.sqrt(Phi.sum()/(1-w[k])):.4f} | eps {r['epsOut']:.3f} vs {(dout/d).max():.3f} | resid {r['resid']:.1e} | mis {r['mis']} | {r['ms']}ms")

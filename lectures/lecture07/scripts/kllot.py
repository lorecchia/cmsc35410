# Helpers for Lecture 7 (KLLOT 2013): normalized Laplacian, Rayleigh quotient, sweep, Lemma 3.1 greedy step approximation.
import numpy as np, networkx as nx
def setup(G):
    n=G.number_of_nodes(); A=nx.to_numpy_array(G,nodelist=range(n),weight='weight')
    d=A.sum(1); Dm=np.diag(d**-0.5); L=np.eye(n)-Dm@A@Dm
    lam,U=np.linalg.eigh(L); return A,d,lam,U
def R(A,d,f):
    E=0.5*np.sum(A*(f[:,None]-f[None,:])**2); return E/np.sum(d*f*f)
def cond(A,d,S):
    m=np.zeros(len(d),bool); m[S]=True
    cut=A[m][:,~m].sum(); return cut/min(d[m].sum(),d[~m].sum())
def sweep(A,d,f):
    order=np.argsort(-f); best=np.inf; vol=d.sum()
    m=np.zeros(len(d),bool); cut=0.0; vS=0.0
    for i,v in enumerate(order[:-1]):
        # add v
        cut+= A[v][~m].sum()-A[v][m].sum() - A[v,v]
        m[v]=True; vS+=d[v]
        if i+1<len(order) and f[order[i+1]]==f[v]: continue
        if f[v]<=0 and (f>0).any() and False: pass
        best=min(best,cut/min(vS,vol-vS))
    return best
def sweep_pos(A,d,f):
    # thresholds t>0 only (threshold sets of nonnegative f)
    order=np.argsort(-f); best=np.inf; vol=d.sum(); m=np.zeros(len(d),bool); cut=0; vS=0
    for i,v in enumerate(order):
        if f[v]<=0: break
        cut+= A[v][~m].sum()-A[v][m].sum(); m[v]=True; vS+=d[v]
        if i+1<len(order) and f[order[i+1]]==f[v]: continue
        best=min(best,cut/min(vS,vol-vS))
    return best
def lemma3_vector(A,d,lam,U):
    """Lecture 6, Lemma 3: shift the Fiedler vector f by its pi-median, keep the half with
    smaller R_2, and normalize D(x)=1. Returns (x, f)."""
    f=U[:,1]/np.sqrt(d); order=np.argsort(f); cum=np.cumsum(d[order])/d.sum()
    med=f[order[np.searchsorted(cum,0.5)]]
    a=np.maximum(f-med,0); b=np.maximum(med-f,0)
    x = a if R(A,d,a)<=R(A,d,b) else b
    return x/np.sqrt(np.sum(d*x*x)), f
fplus=lemma3_vector  # backwards-compatible name
def psi(x,ts):
    ts=np.asarray(ts); return ts[np.argmin(np.abs(x[:,None]-ts[None,:]),1)]
def piece_mass(f,d,a,b):
    m=(f>=a)&(f<=b)
    if not m.any(): return 0.0
    x=f[m]; return np.sum(d[m]*np.minimum(x-a,b-x)**2)
def lemma31(A,d,f,k,lamk):
    Rf=R(A,d,f); M=f.max(); C=2*Rf/(k*lamk); ts=[0.0]
    for i in range(2*k):
        a=ts[-1]
        if piece_mass(f,d,a,M)<C: ts.append(M); 
        else:
            lo,hi=a,M
            for _ in range(100):
                mid=(lo+hi)/2
                if piece_mass(f,d,a,mid)>=C: hi=mid
                else: lo=mid
            ts.append(hi)
        if ts[-1]>=M: 
            ts += [M]*(2*k-i-1); break
    succeeded = ts[-1]>=M-1e-15
    g=psi(f,ts); err=np.sum(d*(f-g)**2)
    return ts,g,err,succeeded,C

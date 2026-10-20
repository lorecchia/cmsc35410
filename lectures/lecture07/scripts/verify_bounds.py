import sys; import os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from kllot import *
rng=np.random.default_rng(0)
def dumbbell(n,dg,bridges,seed=0):
    G1=nx.random_regular_graph(dg,n,seed=seed); G2=nx.random_regular_graph(dg,n,seed=seed+1)
    G=nx.disjoint_union(G1,G2); r=np.random.default_rng(seed)
    for _ in range(bridges): G.add_edge(int(r.integers(n)),n+int(r.integers(n)))
    return G
def sbm(n,p,q,seed=0): return nx.convert_node_labels_to_integers(nx.stochastic_block_model([n,n],[[p,q],[q,p]],seed=seed))
cases={'cycle200':nx.cycle_graph(200),'path150':nx.path_graph(150),'grid15x15':nx.convert_node_labels_to_integers(nx.grid_2d_graph(15,15)),
 'dumbbell':dumbbell(60,4,3),'sbm':sbm(80,0.2,0.01),'3clusters':None}
G3=nx.disjoint_union_all([nx.random_regular_graph(4,50,seed=s) for s in range(3)])
for a,b in [(0,50),(55,110),(3,120)]: G3.add_edge(a,b)
cases['3clusters']=G3
ok=True
for name,G in cases.items():
    if not nx.is_connected(G): print(name,'disconnected'); continue
    A,d,lam,U=setup(G); f,fall=fplus(A,d,lam,U); Rf=R(A,d,f)
    phif=sweep_pos(A,d,f); phiall=sweep(A,d,fall)
    print(f"{name:10s} n={len(d)} l2={lam[1]:.2e} l3={lam[2]:.2e} R2(x)={Rf:.2e}<=lam:{Rf<=lam[1]+1e-12} Phi_sw(x)={phif:.3e} phi(full sweep)={phiall:.3e} sqrt(2l2)={np.sqrt(2*lam[1]):.3e}")
    for k in [2,3,4,6]:
        lk=lam[k-1]; ts,g,err,succ,C=lemma31(A,d,f,k,lk)
        b31=4*Rf/lk; b32=4*k*Rf+4*np.sqrt(2)*k*np.sqrt(err*Rf); bthm=12*np.sqrt(2)*k*Rf/np.sqrt(lk)
        c=(err<=b31+1e-12) and (phif<=b32+1e-12) and (phif<=bthm)
        ok&=c and succ
        print(f"   k={k}: succ={succ} D(x-y)={err:.2e} <= 4R/lk={b31:.2e}; Phi_sw(x)={phif:.2e} <= Lem2prime {b32:.2e}; ratio phi/(k l2/sqrt lk)={phif/(k*lam[1]/np.sqrt(lk)):.3f}  phi/sqrt(l2)={phif/np.sqrt(lam[1]):.3f}")
print("ALL OK" if ok else "FAIL")
# cycle tightness
for n in [100,400,1600]:
    A,d,lam,U=setup(nx.cycle_graph(n))
    print(n,[round(k*lam[1]/np.sqrt(lam[k-1])*n,3) for k in [2,3,5,9,17]], 'phi*n=',2/n*n)

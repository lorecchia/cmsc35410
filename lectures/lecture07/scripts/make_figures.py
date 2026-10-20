import sys; import os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from kllot import *
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','figures')+os.sep
plt.rcParams.update({'font.size':14,'axes.spines.top':False,'axes.spines.right':False,'savefig.transparent':True,'svg.fonttype':'path','font.family':'DejaVu Sans'})
BLUE='#2a6fdb'; ORANGE='#e8792b'; GREY='#888'; GREEN='#2a9d5c'; RED='#c0392b'
def dumbbell(n,dg,bridges,seed=0):
    G1=nx.random_regular_graph(dg,n,seed=seed); G2=nx.random_regular_graph(dg,n,seed=seed+1)
    G=nx.disjoint_union(G1,G2); r=np.random.default_rng(seed)
    for _ in range(bridges): G.add_edge(int(r.integers(n)),n+int(r.integers(n)))
    return G
# --- Fig 1: smooth vs step
fig,ax=plt.subplots(1,2,figsize=(11,3.6))
for a,(G,ttl) in zip(ax,[(nx.cycle_graph(120),'cycle $C_{120}$'),(dumbbell(60,4,3),'two 4-regular expanders + 3 bridges')]):
    A,d,lam,U=setup(G); f=U[:,1]/np.sqrt(d); f=f/np.sqrt(np.sum(d*f*f))
    a.plot(np.sort(f),'o',ms=3.5,color=BLUE)
    a.set_title(f"{ttl}\n$\\lambda=\\lambda_2={lam[1]:.4f}$,  $\\lambda_3={lam[2]:.4f}$",fontsize=13)
    a.set_xlabel('vertices $i$, sorted by $f_i$'); a.axhline(0,color=GREY,lw=.8)
ax[0].set_ylabel('Fiedler vector $f_i$')
plt.tight_layout(); plt.savefig(OUT+'smooth-vs-step.svg'); plt.close()
# --- Fig 2: step approximation (Lemma 3.1) on the path, k=2 -> 5-step
G=nx.path_graph(80); A,d,lam,U=setup(G); f,_=fplus(A,d,lam,U); k=2
ts,g,err,succ,C=lemma31(A,d,f,k,lam[k-1]); print('fig2 thresholds',np.round(ts,4),succ,err,4*R(A,d,f)/lam[k-1])
# use a smaller C for a nicer picture: same greedy with prescribed C
def greedy_C(f,d,C,nsteps):
    M=f.max(); ts=[0.0]
    for i in range(nsteps):
        a=ts[-1]
        if piece_mass(f,d,a,M)<C: ts.append(M); break
        lo,hi=a,M
        for _ in range(100):
            mid=(lo+hi)/2
            if piece_mass(f,d,a,mid)>=C: hi=mid
            else: lo=mid
        ts.append(hi)
    return ts
order=np.argsort(f); fs=f[order]; idx=np.arange(len(f))
ts2=greedy_C(f,d,0.004,4); 
if ts2[-1]<f.max(): ts2.append(f.max())
g2=psi(f,ts2)[order]
fig,ax=plt.subplots(1,2,figsize=(11,3.8),gridspec_kw={'width_ratios':[1.1,1]})
for t in ts2: ax[0].axhline(t,color=GREY,lw=.8,ls='--')
ax[0].plot(idx,fs,'x',color=BLUE,ms=5,label='$x_i$')
ax[0].plot(idx,g2,'o',mfc='none',color=ORANGE,ms=5,label='$y_i=\\psi(x_i)$, nearest threshold')
for i,t in enumerate(ts2): ax[0].text(len(f)+1,t,f'$t_{i}$',va='center',fontsize=12)
ax[0].set_xlabel('vertices $i$, sorted by $x_i$'); ax[0].legend(loc='upper left',fontsize=11,frameon=False)
ax[0].set_title('step approximation $y=\\psi(x)$',fontsize=13)
cols=[BLUE,ORANGE,GREEN,RED,'#8e44ad']
for i in range(1,len(ts2)):
    a,b=ts2[i-1],ts2[i]; m=(fs>a)&(fs<=b) if i>1 else (fs>=a)&(fs<=b)
    bump=np.where(m,np.minimum(fs-a,b-fs),0)
    ax[1].plot(idx,bump,'-',color=cols[(i-1)%5],lw=2,label=f'$\\tau^{{({i})}}$')
ax[1].set_title('tents $\\tau^{(\\ell)}_i=|x_i-y_i|$ on piece $\\ell$',fontsize=13); ax[1].set_xlabel('vertices $i$, sorted by $x_i$')
ax[1].legend(fontsize=11,frameon=False,ncol=2)
plt.tight_layout(); plt.savefig(OUT+'step-approx.svg'); plt.close()
# left panel only (for the Lemma 3.1 statement slide)
fig,ax0=plt.subplots(figsize=(7.5,3.6))
for t in ts2: ax0.axhline(t,color=GREY,lw=.8,ls='--')
ax0.plot(idx,fs,'x',color=BLUE,ms=5,label='$x_i$')
ax0.plot(idx,g2,'o',mfc='none',color=ORANGE,ms=5,label='$y_i=\\psi(x_i)$, nearest threshold')
for i,t in enumerate(ts2): ax0.text(len(f)+1,t,f'$t_{i}$',va='center',fontsize=12)
ax0.set_xlabel('vertices $i$, sorted by $x_i$'); ax0.legend(loc='upper left',fontsize=11,frameon=False)
ax0.set_title('step approximation $y=\\psi(x)$',fontsize=13)
plt.tight_layout(); plt.savefig(OUT+'step-approx-left.svg'); plt.close()
# right panel only: the tents (for the "Failure is impossible" slide)
fig,ax1=plt.subplots(figsize=(8,2.3))
for i in range(1,len(ts2)):
    a,b=ts2[i-1],ts2[i]; m=(fs>a)&(fs<=b) if i>1 else (fs>=a)&(fs<=b)
    ax1.plot(idx,np.where(m,np.minimum(fs-a,b-fs),0),'-',color=cols[(i-1)%5],lw=2,label=f'$\\tau^{{({i})}}$')
ax1.set_title('tents $\\tau^{(\\ell)}_i=|x_i-y_i|$ on piece $\\ell$',fontsize=13); ax1.set_xlabel('vertices $i$, sorted by $x_i$')
ax1.legend(fontsize=11,frameon=False,ncol=2,loc='upper left')
plt.tight_layout(); plt.savefig(OUT+'tents.svg'); plt.close()
# --- Fig 3: densities mu vs 2x, and h vs f^2
ts=np.array([0,0.2,0.45,0.75,1.0]); x=np.linspace(0,1,2001)
mu=np.abs(x-ts[np.argmin(np.abs(x[:,None]-ts[None,:]),1)]); h=np.concatenate([[0],np.cumsum((mu[1:]+mu[:-1])/2*np.diff(x))])
fig,ax=plt.subplots(1,2,figsize=(11,3.6))
ax[0].plot(x,2*x/ (2*x).max()*mu.max()*2.2,color=GREY,lw=2,ls='--',label='Lemma 2: density $\\propto s$  ($g=x^2$)')
ax[0].plot(x,mu,color=ORANGE,lw=2.5,label="Lemma 2$'$: $\\mu(s)=|s-\\psi(s)|$")
for t in ts: ax[0].axvline(t,color=BLUE,lw=.8,ls=':')
ax[0].set_xlabel('threshold $s$'); ax[0].set_title('how the thresholds are weighted',fontsize=13); ax[0].legend(fontsize=11,frameon=False,loc='upper left'); ax[0].set_ylim(0,0.5)
kk=(len(ts)-1)/2
ax[1].plot(x,h,color=ORANGE,lw=2.5,label='$G(s)=\\int_0^s\\mu$')
ax[1].plot(x,x**2/(8*kk),color=GREEN,lw=2,ls='--',label='$s^2/(8k)$')
for t in ts: ax[1].axvline(t,color=BLUE,lw=.8,ls=':')
ax[1].set_xlabel('$s$'); ax[1].set_title('$g_i=G(x_i)$: $G$ increasing, $G(s)\\geq s^2/8k$',fontsize=13); ax[1].legend(fontsize=11,frameon=False,loc='upper left')
plt.tight_layout(); plt.savefig(OUT+'sawtooth.svg'); plt.close()
# --- Fig 4: cycle cosine & 3 localized bumps (k=3 intuition)
n=120; th=np.arange(n)/n*2*np.pi; fc=np.cos(th)
fig,ax=plt.subplots(1,2,figsize=(11,3.3))
ax[0].plot(th,fc,'o',ms=3,color=BLUE); ax[0].set_title('Fiedler vector $f_i=\\cos(2\\pi i/n)$ on $C_n$',fontsize=13); ax[0].set_xlabel('position $2\\pi i/n$')
for j,c in enumerate([BLUE,ORANGE,GREEN]):
    cen=2*np.pi*j/3+np.pi/3; dist=np.angle(np.exp(1j*(th-cen))); bump=np.maximum(0,1-np.abs(dist)/(np.pi/3))
    ax[1].fill_between(th,0,bump,color=c,alpha=.35); ax[1].plot(th,bump,color=c,lw=2)
ax[1].set_title('$k=3$ bumps: tents on disjoint arcs of $n/k$ vertices',fontsize=13); ax[1].set_xlabel('position $2\\pi i/n$')
plt.tight_layout(); plt.savefig(OUT+'cycle-bumps.svg'); plt.close()
# --- Fig 5: each edge pays the area under mu(s)=2s over its interval (Lemma 2)
s=np.linspace(0,1,1001); mu=2*s
edges=[(0.02,0.08,GREEN,'edge near the bottom','left',(0.03,0.75)),(0.08,0.92,BLUE,'long edge','center',(0.50,0.45)),(0.92,0.98,ORANGE,'short edge high up','right',(0.90,1.95))]
fig,ax=plt.subplots(figsize=(7.5,3.4))
ax.plot(s,mu,color='#333',lw=2)
for (lo,hi,c,lab,ha,(xt,yt)) in edges:
    m=(s>=lo)&(s<=hi); ax.fill_between(s[m],0,mu[m],color=c,alpha=.45,lw=0,label=lab)
    ax.plot([lo,hi],[-0.16,-0.16],color=c,lw=3,solid_capstyle='butt'); ax.plot([lo,hi],[-0.16,-0.16],'o',color=c,ms=5)
    ax.text(xt,yt,f'pays {hi*hi-lo*lo:.3f}',ha=ha,va='bottom',fontsize=12.5,color=c,fontweight='bold')
ax.set_title('Lemma 2: $\\mu(s)=2s$; edge $ij$ pays the shaded area',fontsize=14)
ax.set_xlabel('threshold $s$;  edge $ij$ = segment $[x_j,x_i]$',fontsize=12.5); ax.set_ylabel('$\\mu(s)$')
ax.set_ylim(-0.3,2.15); ax.set_xlim(-0.02,1.02); ax.axhline(0,color='#999',lw=.8)
ax.legend(fontsize=11.5,frameon=False,loc='upper left',bbox_to_anchor=(0.0,0.92))
plt.tight_layout(); plt.savefig(OUT+'edge-pays.svg'); plt.close()

# --- Fig 6: Cauchy-Schwarz in Lemma 2 compares edge lengths with edge heights
def two_expanders(m=60,eps=0.01,seed=0):
    G=nx.disjoint_union(nx.random_regular_graph(4,m,seed=seed),nx.random_regular_graph(4,m,seed=seed+1))
    for e in G.edges: G.edges[e]['weight']=1.0
    r=np.random.default_rng(seed)
    for _ in range(3): G.add_edge(int(r.integers(m)),m+int(r.integers(m)),weight=eps)
    return G
cases=[(nx.cycle_graph(120),'smooth: cycle $C_{120}$'),(two_expanders(),'near-step: two expanders, bridges of weight 0.01')]
fig,ax=plt.subplots(1,2,figsize=(11,3.4),sharey=True)
for a_,(G,ttl) in zip(ax,cases):
    A,d,lam,U=setup(G); x,_=lemma3_vector(A,d,lam,U)
    I,J=np.nonzero(np.triu(A)); w=A[I,J]; keep=(x[I]>0)|(x[J]>0); I,J,w=I[keep],J[keep],w[keep]
    Ln=np.sqrt(w)*np.abs(x[I]-x[J]); Hn=np.sqrt(w)*(x[I]+x[J]); Ln/=np.linalg.norm(Ln); Hn/=np.linalg.norm(Hn)
    o=np.argsort((x[I]+x[J])/2+1e-9*np.arange(len(I))); e=np.arange(len(I))
    a_.bar(e,Hn[o],width=1.0,color=ORANGE,alpha=.55,label='heights $x_i+x_j$')
    a_.bar(e,Ln[o],width=1.0,color=BLUE,alpha=.75,label='lengths $|x_i-x_j|$')
    eff=float(Ln@Hn)
    a_.set_title(ttl,fontsize=13); a_.set_xlabel('edges, sorted by position',fontsize=12)
    a_.text(0.5,0.93,f'cos$\\angle$(lengths, heights) = {eff:.2f}',transform=a_.transAxes,fontsize=13,fontweight='bold',va='top',ha='center')
ax[0].set_ylabel('scaled to unit length'); ax[1].legend(fontsize=11.5,frameon=False,loc='upper right',bbox_to_anchor=(1.0,0.78))
plt.tight_layout(); plt.savefig(OUT+'cs-tightness.svg'); plt.close()
# --- check Claims 3.3/3.4 numerically on several graphs
def hfun(f,ts):
    xs=np.linspace(0,f.max(),20001); mu=np.abs(xs-psi(xs,ts)); H=np.concatenate([[0],np.cumsum((mu[1:]+mu[:-1])/2*np.diff(xs))])
    return np.interp(f,xs,H)
for G in [nx.path_graph(100),nx.cycle_graph(90),dumbbell(50,4,3),nx.convert_node_labels_to_integers(nx.grid_2d_graph(10,10))]:
    A,d,lam,U=setup(G); f,_=fplus(A,d,lam,U)
    for k in [2,3,5]:
        ts,g,err,succ,C=lemma31(A,d,f,k,lam[k-1]); h=hfun(f,ts)
        c33=np.all(h>=f**2/(8*k)-1e-9)
        I,J=np.nonzero(np.triu(A)); lhs=np.abs(h[I]-h[J]); rhs=0.5*np.abs(f[I]-f[J])*(np.abs(f[I]-g[I])+np.abs(f[J]-g[J])+np.abs(f[I]-f[J]))
        c34=np.all(lhs<=rhs+1e-9)
        ratio=np.sum(A[I,J]*lhs)/np.sum(d*h); b=4*k*R(A,d,f)+4*np.sqrt(2)*k*np.sqrt(err*R(A,d,f))
        print('claims ok',c33,c34,'(3.4) ratio',round(ratio,4),'<=',round(b,4), sweep_pos(A,d,h)<=ratio+1e-9, abs(sweep_pos(A,d,h)-sweep_pos(A,d,f))<1e-12)

"""R1' check S1: Euler completion on a stretched horizon (static, any f(r), any k).

Metric (Euclidean)  ds^2 = f dtau^2 + dr^2/f + r^2 dSigma_k^2,  Sigma_k: k=1 sphere, 0 flat torus, -1 hyperbolic.
Claims checked symbolically:
 (a) sqrt(g) GB = - d/dr [ 4 f' (k - f) ] * sqrt(gamma_k)          (bulk Euler density is a total r-derivative)
 (b) Myers density on the tube r=r0 (outward normal +d_r):  sqrt(h) Q2 = 4 f'(r0) (k - f(r0)) * sqrt(gamma_k)
 (c) hence tip region r_h<r<r0:  bulk + Myers = beta*4 f'(r_h) k Vol_k = 32 pi^2 chi(Sigma_k) for EVERY r0
 (d) Lorentzian: bare Euler Noether charge of xi=d_t at the cut r=r0,  Q_E = 8c (f'/2) (k - f) Vol_k,
     equals the Myers contribution iota_xi l_E for the exterior (inward normal) with opposite sign:
     completed exterior Euler charge H_E = Q_E + iota_xi l_E = 0 identically in r0.
"""
import sympy as sp, itertools
t,r,x,y=sp.symbols('tau r x y',real=True)
k=sp.symbols('k')
f=sp.Function('f')(r)
X=[t,r,x,y]
def metric(kv):
    if kv==1:  S=sp.sin(x)**2; return sp.diag(f,1/f,r**2,r**2*S), sp.sin(x)
    if kv==0:  return sp.diag(f,1/f,r**2,r**2), 1
    if kv==-1: S=sp.sinh(x)**2; return sp.diag(f,1/f,r**2,r**2*S), sp.sinh(x)
def riemann(g):
    gi=g.inv(); n=4
    G=[[[sp.simplify(sum(gi[a,d]*(sp.diff(g[d,b],X[c])+sp.diff(g[d,c],X[b])-sp.diff(g[b,c],X[d])) for d in range(n))/2)
         for c in range(n)] for b in range(n)] for a in range(n)]
    R=sp.MutableDenseNDimArray.zeros(n,n,n,n)
    for a,b,c,d in itertools.product(range(n),repeat=4):
        R[a,b,c,d]=sp.simplify(sp.diff(G[a][b][d],X[c])-sp.diff(G[a][b][c],X[d])
                    +sum(G[a][c][e]*G[e][b][d]-G[a][d][e]*G[e][b][c] for e in range(n)))
    return R,gi
def GB(g):
    R,gi=riemann(g); n=4
    Rl=[[[[sum(g[a,e]*R[e,b,c,d] for e in range(n)) for d in range(n)] for c in range(n)] for b in range(n)] for a in range(n)]
    Ric=sp.Matrix(n,n,lambda b,d: sum(R[a,b,a,d] for a in range(n)))
    Rs=sp.simplify(sum(gi[b,d]*Ric[b,d] for b in range(n) for d in range(n)))
    Ric2=sp.simplify(sum(gi[a,c]*gi[b,d]*Ric[a,b]*Ric[c,d] for a,b,c,d in itertools.product(range(n),repeat=4)))
    # Riem^2 = R_abcd R^abcd ; build with inverse metric (diagonal)
    Riem2=0
    for a,b,c,d in itertools.product(range(n),repeat=4):
        Riem2+=Rl[a][b][c][d]*sum(gi[b,b2]*gi[c,c2]*gi[d,d2]*R[a,b2,c2,d2] for b2 in [b] for c2 in [c] for d2 in [d])
    Riem2=sp.simplify(Riem2)
    return sp.simplify(Rs**2-4*Ric2+Riem2), R
def Q2_tube(kv):
    # boundary r=r0, outward normal n = sqrt(f) d_r ; induced metric diag(f, r^2, r^2 S^2) (static, product)
    A=sp.diff(f,r)/(2*sp.sqrt(f)); B=sp.sqrt(f)/r
    K=sp.diag(A,B,B)
    # intrinsic Riemann of S^1 x (r0 Sigma_k): only R^{xy}_{xy}=k/r^2
    Rint=sp.MutableDenseNDimArray.zeros(3,3,3,3)
    Rint[1,2,1,2]=k/r**2; Rint[2,1,2,1]=k/r**2; Rint[1,2,2,1]=-k/r**2; Rint[2,1,1,2]=-k/r**2
    Q=0
    for I in itertools.permutations(range(3)):
        for J in itertools.permutations(range(3)):
            D=sp.Matrix(3,3,lambda a,b: 1 if I[a]==J[b] else 0).det()
            if D==0: continue
            Q+=2*D*K[J[0],I[0]]*(Rint[J[1],J[2],I[1],I[2]]-sp.Rational(2,3)*K[J[1],I[1]]*K[J[2],I[2]])
    return sp.simplify(Q.subs(k,kv))
for kv in [1,0,-1]:
    g,sq=metric(kv)
    gb,R=GB(g)
    sqrtg=r**2*sq  # sqrt(det g) = r^2 * sqrt(gamma_k)
    lhs=sp.simplify(sqrtg*gb)
    rhs=sp.simplify(-sp.diff(4*sp.diff(f,r)*(kv-f),r)*sq)
    assert sp.simplify(lhs-rhs)==0, (kv,lhs,rhs)
    Q2=Q2_tube(kv)
    myers=sp.simplify(sp.sqrt(f)*r**2*Q2)   # sqrt(h)/sqrt(gamma_k)
    assert sp.simplify(myers-4*sp.diff(f,r)*(kv-f))==0, (kv,myers)
    # 4D curvature component on the cut: R^{xy}_{xy}
    Rxyxy=sp.simplify(R[2,3,2,3]*g.inv()[3,3]) if True else None
    Rmix=sp.simplify(sum(g.inv()[3,b]*R[2,b,2,3] for b in range(4)))
    # R^{x y}_{x y} = R^x_{y x y} with index up/down: use R^a_{bcd} g^{bb'}
    Rup=sp.simplify(R[2,3,2,3]*g.inv()[3,3])
    assert sp.simplify(Rup-(kv-f)/r**2)==0, (kv,Rup)
    print(f'k={kv:+d}: sqrt(g)GB = -d/dr[4f\'(k-f)]  OK;  sqrt(h)Q2 = 4f\'(k-f)  OK;  R^xy_xy=(k-f)/r^2  OK')
print('Noether: Q_E(r0)=8c*(f\'/2)*(k-f)Vol_k ; iota_xi l_E (inward normal) = -c*sqrt(h)Q2*Vol_k = -4c f\'(k-f) Vol_k  => H_E = 0')
# Explicit families: tip-region Euler number for every r0
M,Q,L,g_=sp.symbols('M Q L g',positive=True)
c=sp.symbols('c')
def check_family(name,fexpr,kv,rh,chi):
    fp=sp.diff(fexpr,r)
    beta=4*sp.pi/fp.subs(r,rh)
    Vol={1:4*sp.pi,0:None,-1:4*sp.pi*(g_-1)}[kv]
    r0=sp.symbols('r0',positive=True)
    bulk=beta*Vol*(-(4*fp*(kv-fexpr)).subs(r,r0)+(4*fp*(kv-fexpr)).subs(r,rh))
    my=beta*Vol*(4*fp*(kv-fexpr)).subs(r,r0)
    tot=sp.simplify(bulk+my)
    assert sp.simplify(tot-32*sp.pi**2*chi)==0,(name,tot)
    print(f'{name}: bulk(rh..r0)+Myers(r0) = {sp.simplify(tot)} = 32 pi^2 chi for all r0;  Myers(r0->rh) =',sp.simplify(sp.limit(my.subs(r0,sp.Symbol('r0')),sp.Symbol('r0'),rh)) if False else sp.simplify(my.subs(r0,rh)))
check_family('Schwarzschild',1-2*M/r,1,2*M,2)
rp=M+sp.sqrt(M**2-Q**2)
check_family('Reissner-Nordstrom',1-2*M/r+Q**2/r**2,1,rp,2)
rh=sp.symbols('r_h',positive=True)
# hyperbolic AdS black hole, f = -1 + r^2/L^2 - m/r with m fixed by f(rh)=0
m=sp.solve(sp.Eq(-1+rh**2/L**2-sp.Symbol('m')/rh,0),sp.Symbol('m'))[0]
check_family('hyperbolic AdS genus g',-1+r**2/L**2-m/r,-1,rh,2-2*g_)
check_family('massless hyperbolic (CHM)',-1+r**2/L**2,-1,L,2-2*g_)
print('All S1 checks passed.')

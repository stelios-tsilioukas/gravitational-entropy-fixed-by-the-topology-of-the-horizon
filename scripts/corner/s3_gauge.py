"""R1' check S3: local Lorentz behaviour of the bare and completed Euler corner forms.
 (i)   bare form: Omega_E(delta, delta_lambda) = -2c oint P(lambda, delta R) on a closed corner,
       i.e. every local Lorentz transformation at the corner has an Euler charge H_lambda = -2c oint P(lambda,R)
       (boost lambda^{01}: 8c oint lambda R^{23};  rotation lambda^{23}: 8c oint lambda R^{01}).
 (ii)  their bracket closes: Omega(delta_lambda, delta_mu) = -2c oint P(lambda,[R,mu])  (= charge of the commutator)
 (iii) P vanishes identically on the stabilizer of a unit spacelike or timelike vector and of a null VECTOR (iso(2)),
       but not on the stabilizer of a null RAY (sim(2) = iso(2)+boost): the boost/rotation pairing survives.
 (iv)  adapted gauge: completed form 2c oint P(dw_bar, dw_bar) = 0 for all variations, and residual gauge
       transformations (lambda in stab(u)) have zero completed Euler charge, even when field dependent."""
import sympy as s, itertools, random
from forms import F, pairs, eta
x,y,e=s.symbols('x y epsilon'); f=F([x,y]); c=s.Symbol('c')
random.seed(7)
V=[x,y]
def rnd1():
    return {ij:{(0,):s.Integer(random.randint(-2,2))+random.randint(-1,1)*V[random.randrange(2)]**random.randint(1,2),
                (1,):s.Integer(random.randint(-2,2))+random.randint(-1,1)*V[random.randrange(2)]} for ij in pairs}
def rnd0():
    return {ij:{():s.Integer(random.randint(-2,2))+random.randint(-1,1)*x*y+random.randint(-1,1)*V[random.randrange(2)]} for ij in pairs}
w,dw,lam,mu=rnd1(),rnd1(),rnd0(),rnd0()
R=f.curv(w)
# delta R = D(delta w) at first order
Rp=f.curv(f.add_alg(w,{ij:f.scale(dw[ij],e) for ij in pairs}))
dR={ij:f.tidy({I:s.diff(v,e).subs(e,0) for I,v in Rp[ij].items()}) for ij in pairs}
Dl=f.D(lam,w); Dm=f.D(mu,w)
def P0(l,A):  # eps lam^{ab} A^{cd}, lam a 0-form
    return f.P(l,A)
# (i)
lhs=f.scale(f.P(dw,f.lin((-1,Dl))),2*c)                  # 2c P(dw ^ delta_lambda w), delta_lambda w = -D lambda
rhs=f.scale(P0(lam,dR),-2*c)
exact=f.add(lhs,f.scale(rhs,-1))
# exact must be d of the 1-form -2c P(lambda, dw)
assert not f.add(exact,f.scale(f.d(P0(lam,dw)),-2*c)), 'identity (i) failed'
print('PASS (i)  2c P(dw ^ delta_lambda w) = -2c P(lambda, delta R) + d[2c P(lambda,dw)]')
# (ii)
lhs2=f.scale(f.P(Dl,Dm),2*c)
br=f.comm(R,mu)
rhs2=f.scale(P0(lam,br),-2*c)
assert not f.add(lhs2,f.scale(rhs2,-1),f.scale(f.d(P0(lam,Dm)),-2*c)), 'identity (ii) failed'
lm=f.comm(lam,mu)
alt=f.scale(P0(lm,R),-2*c)
same=not f.add(rhs2,f.scale(alt,-1)); opp=not f.add(rhs2,alt)
print('PASS (ii) Omega(delta_lambda,delta_mu) = -2c oint P(lambda,[R,mu]) ; equals',
      'H_[lambda,mu]' if same else ('-H_[lambda,mu]' if opp else '??'))
# (iii) stabilizers
def gen(i,j):
    M=s.zeros(4); M[i,j]=1; M[j,i]=-1   # X^{ij} upper components
    return M
basis=[(i,j) for i,j in pairs]
def act(Xup,u):  # (X u)^a = X^{a b} eta_bb u^b
    return s.Matrix([sum(Xup[a,b]*eta[b]*u[b] for b in range(4)) for a in range(4)])
def Pform(X,Y):
    return sum(s.LeviCivita(a,b,cc,d)*X[a,b]*Y[cc,d] for a,b,cc,d in itertools.permutations(range(4)))
def stab(u,ray=False):
    co=s.symbols('a0:6'); X=sum((co[k]*gen(*basis[k]) for k in range(6)),s.zeros(4))
    Xu=act(X,u)
    if ray:
        lamb=s.Symbol('lamb'); eqs=list(Xu-lamb*s.Matrix(u)); sol=s.solve(eqs,list(co)+[lamb],dict=True)[0]
    else:
        sol=s.solve(list(Xu),co,dict=True)[0]
    Xs=X.subs(sol); free=[q for q in co if Xs.has(q)]
    return [Xs.subs({q:(1 if q==p else 0) for q in free}) for p in free]
for name,u,ray in [('spacelike e1',[0,1,0,0],False),('timelike e0',[1,0,0,0],False),
                   ('null vector l',[1,1,0,0],False),('null ray l (sim(2))',[1,1,0,0],True)]:
    B=stab(u,False)+([gen(0,1)] if ray else [])   # sim(2) = iso(2) + boost along l
    if ray: assert act(gen(0,1),u)==s.Matrix(u) or act(gen(0,1),u)==-s.Matrix(u)
    M=s.Matrix(len(B),len(B),lambda i,j:Pform(B[i],B[j]))
    print(f'(iii) stab({name}): dim {len(B)}, P restricted to it is', 'ZERO' if M==s.zeros(len(B)) else f'NONZERO, rank {M.rank()}')
# (iv) adapted gauge for u=e1 (timelike face): product reference = drop components with index 1
def proj(A): return {ij:(A[ij] if 1 not in ij else {}) for ij in pairs}
dw2=rnd1()
assert not f.P(proj(dw),proj(dw2)), 'P on stab(e1) should vanish'
lam_s=proj(rnd0())
Dls=f.D(lam_s,proj(w))  # D_bar lambda for lambda in stab
assert all((1 not in ij) or not Dls[ij] for ij in pairs)
assert not f.P(proj(dw),f.lin((-1,Dls)))
print('PASS (iv) completed corner form and residual (stab(u)) Lorentz charges vanish in the adapted gauge')
print('All S3 checks passed.')

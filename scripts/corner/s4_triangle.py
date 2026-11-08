"""R1' check S4: triangle identity for Euler transgressions (gluing at joints).
T(A1,A0) = 2c int_0^1 P(A1-A0, F_t) dt  satisfies L(A1)-L(A0) = dT(A1,A0).
Claim: T(A2,A0) - T(A2,A1) - T(A1,A0) = d B,  B = c P((A1-A0) ^ (A2-A1)).
So facewise completions with different references glue with joint term -B (plus terms of the joint data only)."""
import sympy as s, itertools
from forms import F, pairs
x,y,z,w,t=s.symbols('x y z w t'); f=F([x,y,z,w]); c=s.Symbol('c')
def rnd(seed):
    import random; random.seed(seed)
    v=[x,y,z,w]
    return {ij:{(random.randrange(4),):s.Integer(random.randint(-2,2))+random.randint(-1,1)*v[random.randrange(4)] for _ in range(2)} for ij in pairs}
A0,A1,A2=rnd(1),rnd(2),rnd(3)
def T(A,Ab):
    B=f.lin((1,A),(-1,Ab)); At=f.add_alg(Ab,{ij:f.scale(B[ij],t) for ij in pairs})
    integ=f.scale(f.P(B,f.curv(At)),2*c)
    return {I:s.integrate(v,(t,0,1)) for I,v in integ.items()}
# Chern-Weil check
L=lambda A: f.scale(f.P(f.curv(A),f.curv(A)),c)
assert not f.add(f.d(T(A1,A0)),f.scale(L(A1),-1),L(A0))
lhs=f.add(T(A2,A0),f.scale(T(A2,A1),-1),f.scale(T(A1,A0),-1))
for kappa in [1,-1,s.Rational(1,2),-s.Rational(1,2),2,-2]:
    B=f.scale(f.P(f.lin((1,A1),(-1,A0)),f.lin((1,A2),(-1,A1))),c*kappa)
    if not f.add(lhs,f.scale(f.d(B),-1)):
        print('PASS triangle identity with B = c*(%s)*P((A1-A0)^(A2-A1))'%kappa); break
else:
    raise SystemExit('no coefficient found')
# Same identity in the notation of the Paper A insert: references wb1, wb2, dynamical w
w_,wb1,wb2=A2,A1,A0
lhs=f.add(T(w_,wb2),f.scale(T(w_,wb1),-1),f.scale(T(wb1,wb2),-1))
rhs=f.scale(f.d(f.P(f.lin((1,wb2),(-1,wb1)),f.lin((1,w_),(-1,wb1)))),c)
assert not f.add(lhs,f.scale(rhs,-1))
print('PASS insert form: T(w,wb2)-T(w,wb1)-T(wb1,wb2) = c d eps (wb2-wb1)^(w-wb1)')

"""Table I, row 3 and Appendix A (iii): Euclidean Schwarzschild.

int_M GB sqrt(g) = 64 pi^2 (chi = 2), oint ^(2)R dA = 8 pi on the horizon, and the
Gauss-Bonnet part of the Wald entropy, 2 pi alpha chi / G = 4 pi alpha / G (Eq. (Stop)).
"""
import sympy as sp
from curvature import curvature

M,r,tau,th,ph=sp.symbols('M r tau theta phi',positive=True)
f=1-2*M/r
g=sp.diag(f, 1/f, r**2, r**2*sp.sin(th)**2)
x=[tau,r,th,ph]
Rs,Ric2,Riem2,GB=curvature(g,x)
print("Euclidean Schwarzschild: R =",Rs,"  Ric^2 =",Ric2,"  Riem^2 (Kretschmann) =",sp.simplify(Riem2),"  GB =",sp.simplify(GB))
beta=8*sp.pi*M
sqrtg=sp.sqrt(sp.simplify(g.det()))
I=sp.integrate(sp.integrate(sp.integrate(sp.integrate(GB*sqrtg,(ph,0,2*sp.pi)),(th,0,sp.pi)),(r,2*M,sp.oo)),(tau,0,beta))
print("int_M GB sqrt(g) d^4x =",sp.simplify(I),"   => chi = ",sp.simplify(I/(32*sp.pi**2)))

# horizon 2-surface: S^2 of radius r_h=2M ; intrinsic Ricci scalar and its integral
rh=2*M
R2=2/rh**2
area=4*sp.pi*rh**2
print("horizon: ^(2)R =",R2,"  Area =",area,"  oint ^(2)R dA =",sp.simplify(R2*area),"  = 4 pi chi(Sigma) with chi =",sp.simplify(R2*area/(4*sp.pi)))

# Wald entropy of EGB on this horizon
alpha,G=sp.symbols('alpha G',positive=True)
S=area/(4*G)+sp.Rational(1,4)/G*2*alpha*R2*area
print("S_Wald = A/4G +",sp.simplify(S-area/(4*G))," ; predicted 2 pi alpha chi/G =",2*sp.pi*alpha*2/G)

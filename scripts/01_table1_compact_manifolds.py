"""Table I, rows 1-2: curvature invariants and the Euler characteristic of S^4 and S^2 x S^2.

Checks int_M GB sqrt(g) d^4x = 32 pi^2 chi(M) (Chern-Gauss-Bonnet, Eq. (CGB)) and the
entries of Table I and Appendix A (i)-(ii) of the paper.
"""
import sympy as sp
from curvature import curvature

a,b=sp.symbols('a b',positive=True)
th,ph,ps,ch=sp.symbols('theta phi psi chi',positive=True)

print("=== S^4 radius a ===")
x=[ch,ps,th,ph]
g=sp.diag(a**2, a**2*sp.sin(ch)**2, a**2*sp.sin(ch)**2*sp.sin(ps)**2,
          a**2*sp.sin(ch)**2*sp.sin(ps)**2*sp.sin(th)**2)
Rs,Ric2,Riem2,GB=curvature(g,x)
vol=sp.integrate(sp.sqrt(sp.simplify(g.det())),(ph,0,2*sp.pi),(th,0,sp.pi),(ps,0,sp.pi),(ch,0,sp.pi))
print(" R =",Rs," Ric^2 =",Ric2," Riem^2 =",Riem2," GB =",GB)
print(" Vol =",sp.simplify(vol)," int GB =",sp.simplify(GB*vol)," chi =",sp.simplify(GB*vol/(32*sp.pi**2)))

print("=== S^2(a) x S^2(b) ===")
t1,p1,t2,p2=sp.symbols('theta_1 phi_1 theta_2 phi_2',positive=True)
x=[t1,p1,t2,p2]
g=sp.diag(a**2, a**2*sp.sin(t1)**2, b**2, b**2*sp.sin(t2)**2)
Rs,Ric2,Riem2,GB=curvature(g,x)
vol=(4*sp.pi*a**2)*(4*sp.pi*b**2)
print(" R =",Rs," Ric^2 =",Ric2," Riem^2 =",Riem2," GB =",GB)
print(" Vol =",vol," int GB =",sp.simplify(GB*vol)," chi =",sp.simplify(GB*vol/(32*sp.pi**2)))

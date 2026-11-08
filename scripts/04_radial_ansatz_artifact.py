"""Paper Sec. VI (Eq. (conicalresult)) and the subsection "Cuts with extrinsic curvature".

Metric:  ds^2 = drho^2 + h(rho)^2 dtheta^2 + b(rho)^2 dOmega_2^2,
with h(0)=0, h'(0)=1, h'(inf)=n (regularised cone) and b(rho) the radius of the S^2.

1. For ARBITRARY h and b the Gauss-Bonnet density is an exact total derivative,
       GB sqrt(g) / sin(theta_1) = d/drho[ -8 h' (1 - b'^2) ],
   so int GB sqrt(g) = 64 pi^2 [ (1 - b'(0)^2) - n (1 - b'(inf)^2) ]  (Eq. (conicalresult)).
2. Numerically, a profile with b'(0) != 0 gives a tip coefficient 64 pi^2 (1 - b'(0)^2).
3. This factor is NOT an extrinsic-curvature correction. With b = b0 + b1 rho + ...,
   b1 != 0, the metric is not differentiable at rho = 0 (b = b0 + b1 sqrt(x^2+y^2) in smooth
   normal coordinates), and the curvature diverges at the tip already for n = 1:
   rho * R -> -4 b'(0)/b(0) as rho -> 0.  A smooth cut with extrinsic curvature must be
   treated with smooth squashed cones, see 05_squashed_cone_fps.py.
"""
import numpy as np
import sympy as sp
from scipy.integrate import quad
from curvature import curvature

rho, th, t1, p1 = sp.symbols('rho theta theta_1 phi_1', positive=True)
h = sp.Function('h')(rho)
b = sp.Function('b')(rho)
g = sp.diag(1, h**2, b**2, b**2 * sp.sin(t1)**2)
x = [rho, th, t1, p1]
R, Ric2, Riem2, GB = curvature(g, x)
dens = sp.simplify(sp.expand(GB * b**2 * sp.sin(t1) * h))

print("1. Total-derivative form for arbitrary h(rho), b(rho):")
prim = -8 * sp.diff(h, rho) * (1 - sp.diff(b, rho)**2)
print("   d/drho[-8 h'(1-b'^2)] - GB sqrt(g)/sin(theta_1) =",
      sp.simplify(sp.expand(sp.diff(prim, rho) - dens / sp.sin(t1))))


def run(n, a, bp0, L=3.0):
    hp = lambda r: n + (1 - n) / np.cosh(r / a)**2
    hpp = lambda r: (1 - n) * (-2 / a) * np.tanh(r / a) / np.cosh(r / a)**2
    bp = lambda r: bp0 / np.cosh(r / L)**2
    bpp = lambda r: bp0 * (-2 / L) * np.tanh(r / L) / np.cosh(r / L)**2
    d = lambda r: 8 * bp(r)**2 * hpp(r) + 16 * bp(r) * bpp(r) * hp(r) - 8 * hpp(r)
    with np.errstate(over='ignore'):
        I, _ = quad(d, 0, 60, limit=2000)
    total = I * 2 * np.pi * 4 * np.pi
    pred = 64 * np.pi**2 * ((1 - bp0**2) - n)
    return total, pred


print("\n2. Numerical integral with b'(0) != 0 (radial ansatz):")
for bp0 in [0.0, 0.2, 0.5, 0.8]:
    for n_val in [0.7, 1.3, 2.0]:
        tot, pred = run(n_val, 0.5, bp0)
        print(f"   b'(0)={bp0:3.1f}  n={n_val:4.2f}   integral={tot:12.4f}   "
              f"64pi^2[(1-b'(0)^2)-n]={pred:12.4f}   diff={abs(tot - pred):.2e}")

print("\n3. The radial ansatz with b'(0) != 0 is singular at the tip (n = 1, h = rho):")
b0, b1, b2 = sp.symbols('b0 b1 b2', positive=True)
Rtip = sp.simplify(R.subs(h, rho).doit())
Rtip = sp.simplify(Rtip.subs(b, b0 + b1 * rho + b2 * rho**2).doit())
lim = sp.limit(sp.simplify(rho * Rtip), rho, 0)
print("   lim_{rho->0} rho * R =", lim, "  (nonzero unless b'(0) = b1 = 0)")
print("   => the factor (1 - b'(0)^2) records a conical singularity of the ansatz,")
print("      not an extrinsic-curvature correction to the entropy.")

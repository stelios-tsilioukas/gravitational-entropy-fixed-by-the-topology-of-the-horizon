"""Paper Sec. VII.B, Proposition 2 (rigidity): oint K dA for deformed cuts (Y20, Y31,
generic l=2,3,4 superposition, eps up to 0.4), the Kerr horizon (a/M up to 0.99) and a torus.

Test 2, step 2: rigidity of the topological Noether charge.

The GB part of the Iyer-Wald charge on a cut Sigma is Q_top = (alpha/2G) oint ^{(2)}R dA.
If it is to be central it must not respond to ANY smooth deformation of the cut or of
the fields.  Here we deform the cut arbitrarily (non-axisymmetric, large amplitude) and
check oint ^{(2)}R dA numerically; and we do a genus-1 cut as a control.
"""
import numpy as np, sympy as sp

th,ph,eps = sp.symbols('theta phi epsilon')

def euler_from_radial(rfun, name, epsval, Nth=400, Nph=400):
    """closed surface X(th,ph) = r(th,ph) * n(th,ph) embedded in R^3."""
    r = rfun
    X = sp.Matrix([r*sp.sin(th)*sp.cos(ph), r*sp.sin(th)*sp.sin(ph), r*sp.cos(th)])
    Xt, Xp = X.diff(th), X.diff(ph)
    E,F,G_ = (Xt.T*Xt)[0], (Xt.T*Xp)[0], (Xp.T*Xp)[0]
    nvec = Xt.cross(Xp); W = sp.sqrt(E*G_-F**2)
    Xtt,Xtp,Xpp = X.diff(th,2), X.diff(th,ph), X.diff(ph,2)
    L = (Xtt.T*nvec)[0]/W; Mm = (Xtp.T*nvec)[0]/W; N = (Xpp.T*nvec)[0]/W
    K = (L*N-Mm**2)/(E*G_-F**2)                      # Gaussian curvature
    integrand = sp.lambdify((th,ph),(K*W).subs(eps,epsval),'numpy')
    # Gauss-Legendre in theta (avoid poles), uniform in phi
    xg,wg = np.polynomial.legendre.leggauss(Nth)
    tg = 0.5*np.pi*(xg+1); wt = 0.5*np.pi*wg
    pg = (np.arange(Nph)+0.5)*2*np.pi/Nph; wp = 2*np.pi/Nph
    T,P = np.meshgrid(tg,pg,indexing='ij')
    val = np.sum(integrand(T,P)*wt[:,None])*wp
    print(f"  {name:38s} eps={epsval:5.2f}   oint K dA = {val: .12f}   oint R dA = {2*val: .12f}   2piChi = {4*np.pi:.6f}")
    return val

print("Deformed cuts of a horizon (chi = 2, so oint K dA should be 4pi = %.9f):"%(4*np.pi))
Y20 = (3*sp.cos(th)**2-1)
Y31 = sp.sin(th)*(5*sp.cos(th)**2-1)*sp.cos(ph)
Y43 = sp.sin(th)**3*sp.cos(th)*sp.sin(3*ph)
for e in (0.0,0.05,0.2,0.4):
    euler_from_radial(1+eps*Y20, "axisymmetric  1+e*Y20", e)
for e in (0.05,0.2,0.4):
    euler_from_radial(1+eps*Y31, "non-axisym    1+e*Y31", e)
for e in (0.1,0.3):
    euler_from_radial(1+eps*(Y20+Y31+Y43), "generic       1+e*(Y20+Y31+Y43)", e)

print("\nControl: genus-1 cut (torus of revolution, chi = 0, so oint K dA = 0):")
R0,a0 = 3.0,1.0
u,v = sp.symbols('u v')
Xt_ = sp.Matrix([(R0+a0*sp.cos(u))*sp.cos(v),(R0+a0*sp.cos(u))*sp.sin(v),a0*sp.sin(u)])
Ku = sp.cos(u)/(a0*(R0+a0*sp.cos(u)));  dA = a0*(R0+a0*sp.cos(u))
print("  torus  oint K dA =", sp.simplify(sp.integrate(sp.integrate(Ku*dA,(u,0,2*sp.pi)),(v,0,2*sp.pi))))

print("\nControl: handle attachment, chi -> chi-2 changes the charge by the quantum 2*(2pi)*... :")
print("  Delta(oint K dA) = 2pi*Delta(chi) = %.9f  for Delta(chi) = -2"%(2*np.pi*(-2)))

print("\nPhysical cut: the Kerr horizon (a genuinely non-round 2-metric):")
import sympy as sp
a_,rh_ = sp.symbols('a r_h', positive=True)
rho2 = rh_**2 + a_**2*sp.cos(th)**2
gth  = rho2
gph  = (rh_**2+a_**2)**2*sp.sin(th)**2/rho2
sq   = sp.sqrt(sp.simplify(gth*gph))
# 2d Ricci scalar of diag(gth,gph) with both depending on theta only
R2k = sp.simplify(-2/sq*sp.diff(sp.diff(sp.sqrt(gph),th)/sp.sqrt(gth),th)/sp.sqrt(gth)*sp.sqrt(gth))
R2k = sp.simplify(-(2/sp.sqrt(gth*gph))*sp.diff( sp.diff(sp.sqrt(gph),th)/sp.sqrt(gth), th))
I = sp.simplify(sp.integrate(sp.simplify(R2k*sq),(th,0,sp.pi))*2*sp.pi)
print("   oint ^{(2)}R dA over the Kerr horizon =", sp.simplify(I), "  (4 pi chi = 8 pi)")
for aval in (0.0,0.3,0.7,0.99):
    v=sp.N(I.subs({a_:aval,rh_:1+sp.sqrt(1-sp.Rational(int(aval*100),100)**2)}) if aval else I.subs({a_:0,rh_:2}))
    print(f"   a/M={aval:4.2f}:  {float(v): .12f}    8pi = {float(8*sp.pi):.12f}")

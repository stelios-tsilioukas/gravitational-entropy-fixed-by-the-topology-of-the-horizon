"""Paper Sec. VI (Test 1), Eqs. (conemetric)-(conicalresult): exact linearity of the
Gauss-Bonnet conical term in (1-n), two tip regularisations (tanh, arctan), a = 0.1, 1, 5.

Test 1: the Gauss-Bonnet conical expansion.

Strategy: build a REGULARISED cone C_n (angle 2 pi n at the tip, smooth everywhere)
times a round S^2, compute the Gauss-Bonnet density from the metric directly, and
integrate.  Compare with the Fursaev-Solodukhin singular term
      int_{M_n} GB  =  n int_{M_1} GB  +  8 pi (1-n) oint_Sigma ^{(2)}R dA .
"""
import sympy as sp

rho, th, t1, p1, b, n, a = sp.symbols('rho theta theta_1 phi_1 b n a', positive=True)
h = sp.Function('h')(rho)

g = sp.diag(1, h**2, b**2, b**2*sp.sin(t1)**2)
x = [rho, th, t1, p1]
nn = 4
ginv = g.inv()
Gam=[[[sp.simplify(sum(ginv[l,s]*(sp.diff(g[s,i],x[j])+sp.diff(g[s,j],x[i])-sp.diff(g[i,j],x[s])) for s in range(nn))/2)
       for j in range(nn)] for i in range(nn)] for l in range(nn)]
Riem=[[[[sp.simplify(sp.diff(Gam[l][j][k],x[i])-sp.diff(Gam[l][i][k],x[j])
         +sum(Gam[l][i][s]*Gam[s][j][k]-Gam[l][j][s]*Gam[s][i][k] for s in range(nn)))
         for k in range(nn)] for j in range(nn)] for i in range(nn)] for l in range(nn)]
Rd=[[[[sp.simplify(sum(g[l,m]*Riem[m][i][j][k] for m in range(nn))) for k in range(nn)] for j in range(nn)] for i in range(nn)] for l in range(nn)]
Ric=sp.Matrix(nn,nn,lambda i,j: sp.simplify(sum(Riem[l][l][i][j] for l in range(nn))))
R=sp.simplify(sum(ginv[i,j]*Ric[i,j] for i in range(nn) for j in range(nn)))
Ric2=sp.simplify(sum(ginv[i,k]*ginv[j,l]*Ric[i,j]*Ric[k,l] for i in range(nn) for j in range(nn) for k in range(nn) for l in range(nn)))
Riem2=0
for A in range(nn):
 for B in range(nn):
  for C in range(nn):
   for D in range(nn):
    if Rd[A][B][C][D]==0: continue
    for E in range(nn):
     for F in range(nn):
      for G in range(nn):
       for H in range(nn):
        if Rd[E][F][G][H]==0: continue
        Riem2+=ginv[A,E]*ginv[B,F]*ginv[C,G]*ginv[D,H]*Rd[A][B][C][D]*Rd[E][F][G][H]
Riem2=sp.simplify(Riem2)
GB=sp.simplify(R**2-4*Ric2+Riem2)
print("R      =", sp.simplify(R))
print("GB     =", sp.simplify(GB))
K1 = -sp.diff(h,rho,2)/h          # Gaussian curvature of the 2d factor
K2 = 1/b**2                        # Gaussian curvature of S^2(b)
print("GB - 8*K1*K2 =", sp.simplify(GB - 8*K1*K2), "   <-- must be 0")

sqrtg = sp.sqrt(sp.simplify(g.det()))
print("sqrt(g) =", sp.simplify(sqrtg))
integrand = sp.simplify(GB*sqrtg)
print("GB*sqrt(g) =", integrand)
# integrate over theta (2 pi), S^2 (4 pi b^2 / b^2 ... ) and rho
I_rho = sp.integrate(sp.simplify(integrand/(sp.sin(t1))), (rho, 0, sp.oo))
print("\n int dr [GB sqrt(g)/sin(theta_1)] =", sp.simplify(I_rho))


# ---------------- numerical verification -------------------------------
import numpy as np
from scipy.integrate import quad
print("\n=== numerical check:  int GB sqrt(g) d^4x  vs  64 pi^2 (1-n) ===")

def profiles(n, a):
    """smooth h(rho) with h(0)=0, h'(0)=1, h'(inf)=n  (two different regularisations)"""
    P = {}
    P['tanh']   = (lambda r: n*r + (1-n)*a*np.tanh(r/a),
                   lambda r: (1-n)*(-2/a)*np.tanh(r/a)/np.cosh(r/a)**2)
    P['arctan'] = (lambda r: n*r + (1-n)*a*np.arctan(r/a),
                   lambda r: (1-n)*(-2*r/a**2)/(1+(r/a)**2)**2)
    return P

target = lambda n: 64*np.pi**2*(1-n)
for n_val in [0.5, 0.8, 1.0, 1.3, 2.0]:
    for a_val in [0.1, 1.0, 5.0]:
        for name,(h_f,hpp) in profiles(n_val,a_val).items():
            I,err = quad(lambda r: -8*hpp(r), 0, np.inf, limit=800)
            total = I*2*np.pi*4*np.pi
            print(f"  n={n_val:4.2f} a={a_val:4.1f} [{name:6s}]  integral={total:12.4f}   "
                  f"64pi^2(1-n)={target(n_val):12.4f}   diff={abs(total-target(n_val)):.2e}")
    print()

# entropy from the replica construction
import sympy as sp
alpha,G,nn_ = sp.symbols('alpha G n', positive=True)
logZ_n = (alpha/(16*sp.pi*G))*64*sp.pi**2*(1-nn_)      # GB part of log Z_n (flat cone: no bulk piece)
S = sp.simplify(-sp.diff(logZ_n - nn_*logZ_n.subs(nn_,1), nn_))
print("replica entropy from the GB term:  S =", S, "   [Wald value 2 pi alpha chi/G with chi=2 is", 4*sp.pi*alpha/G, "]")

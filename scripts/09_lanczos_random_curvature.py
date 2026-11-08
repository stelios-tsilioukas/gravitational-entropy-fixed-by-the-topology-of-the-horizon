"""Paper Sec. VIII.B, Proposition 4(a): the Lanczos tensor vanishes identically in D=4
and not in D=5,6, tested on random algebraic curvature tensors (Kulkarni-Nomizu);
also GB = 0 on S^1 x S^3 (Sec. VIII.G).

Building the crossed product, step 1: the Gauss-Bonnet term changes no equation
of motion in D=4, and does change them in D>4.

Gauss-Bonnet (Lanczos) field-equation tensor:
  H_ab = 2[ R R_ab - 2 R_ac R^c{}_b - 2 R^{cd} R_acbd + R_a{}^{cde} R_bcde ]
         - (1/2) g_ab GB .
H_ab = 0 identically in D=4 means the solution space -- hence the algebra of
observables on a background, hence its modular flow -- is exactly that of Einstein
gravity, so the crossed product is inherited rather than rebuilt.

The identity is algebraic (pointwise in the curvature), so we test it on random
ALGEBRAIC CURVATURE TENSORS: Kulkarni-Nomizu products of random symmetric tensors,
which carry all Riemann symmetries including the first Bianchi identity and span
the space of such tensors.
"""
import numpy as np, sympy as sp
rng = np.random.default_rng(11)

def kn(h, k):
    return (np.einsum('ac,bd->abcd', h, k) + np.einsum('bd,ac->abcd', h, k)
          - np.einsum('ad,bc->abcd', h, k) - np.einsum('bc,ad->abcd', h, k))

def random_curvature(D, nterms=4):
    R = np.zeros((D,)*4)
    for _ in range(nterms):
        h = rng.normal(size=(D, D)); h = h + h.T
        k = rng.normal(size=(D, D)); k = k + k.T
        R += kn(h, k)
    return R

def lanczos(R, g):
    ginv = np.linalg.inv(g)
    Ric  = np.einsum('ac,abcd->bd', ginv, R)
    Rs   = np.einsum('bd,bd->', ginv, Ric)
    Ruu  = np.einsum('ai,bj,ck,dl,ijkl->abcd', ginv, ginv, ginv, ginv, R)
    Ricu = np.einsum('ai,bj,ij->ab', ginv, ginv, Ric)
    GB   = Rs**2 - 4*np.einsum('ab,ab->', Ricu, Ric) + np.einsum('abcd,abcd->', Ruu, R)
    core = (Rs*Ric
            - 2*np.einsum('ac,cd,db->ab', Ric, ginv, Ric)
            - 2*np.einsum('cd,acbd->ab', Ricu, R)
            + np.einsum('acde,bfgh,cf,dg,eh->ab', R, R, ginv, ginv, ginv))
    H = 2*core - 0.5*g*GB
    # Euler's theorem cross-check on dGB/dR_abcd
    P = (2*Ruu
         - 2*(np.einsum('ac,bd->abcd', ginv, Ricu) - np.einsum('ad,bc->abcd', ginv, Ricu)
            - np.einsum('bc,ad->abcd', ginv, Ricu) + np.einsum('bd,ac->abcd', ginv, Ricu))
         + Rs*(np.einsum('ac,bd->abcd', ginv, ginv) - np.einsum('ad,bc->abcd', ginv, ginv)))
    euler = np.einsum('abcd,abcd->', R, P) - 2*GB
    sym = max(np.abs(R + np.swapaxes(R, 0, 1)).max(),
              np.abs(R - np.einsum('cdab->abcd', R)).max(),
              np.abs(R + np.einsum('acdb->abcd', R) + np.einsum('adbc->abcd', R)).max())
    return sym, GB, H, np.abs(2*core).max(), euler

print("Lanczos tensor for random algebraic curvature tensors "
      "(H_ab = 0 identically <=> Gauss-Bonnet is topological)\n")
print(f"{'D':>3} {'symmetry err':>13} {'Euler err':>11} {'GB':>12} "
      f"{'max|H_ab|':>13} {'scale':>11} {'relative':>11}")
for D in (4, 5, 6):
    for _ in range(3):
        g = np.diag([-1.0] + [1.0]*(D-1)); A = rng.normal(size=(D, D))*0.2
        g = g + A + A.T
        R = random_curvature(D)
        sym, GB, H, scale, euler = lanczos(R, g)
        print(f"{D:3d} {sym:13.2e} {abs(euler)/abs(GB):11.2e} {GB:12.4g} "
              f"{np.abs(H).max():13.5e} {scale:11.4g} {np.abs(H).max()/scale:11.2e}")

print("\nConsistency: the handle geometry S^1 x S^3 has chi = 0, and the "
      "Gauss-Bonnet density vanishes there identically.")
a, chi_, th_, ph_, ps_ = sp.symbols('a chi theta phi psi', positive=True)
# S^1 x S^3(a) with the round S^3
g4 = sp.diag(1, a**2, a**2*sp.sin(chi_)**2, a**2*sp.sin(chi_)**2*sp.sin(th_)**2)
x = [ps_, chi_, th_, ph_]; n = 4; gi = g4.inv()
Gam = [[[sp.simplify(sum(gi[l, s]*(sp.diff(g4[s, i], x[j]) + sp.diff(g4[s, j], x[i])
        - sp.diff(g4[i, j], x[s])) for s in range(n))/2) for j in range(n)]
        for i in range(n)] for l in range(n)]
def Rup(r_, s_, m_, nu):
    return sp.simplify(sp.diff(Gam[r_][nu][s_], x[m_]) - sp.diff(Gam[r_][m_][s_], x[nu])
        + sum(Gam[r_][m_][l]*Gam[l][nu][s_] - Gam[r_][nu][l]*Gam[l][m_][s_] for l in range(n)))
Ru = [[[[Rup(A_, B_, C_, D_) for D_ in range(n)] for C_ in range(n)] for B_ in range(n)] for A_ in range(n)]
Rd = [[[[sp.simplify(sum(g4[A_, m]*Ru[m][B_][C_][D_] for m in range(n))) for D_ in range(n)]
        for C_ in range(n)] for B_ in range(n)] for A_ in range(n)]
Ric = sp.Matrix(n, n, lambda s_, v: sp.simplify(sum(Ru[m][s_][m][v] for m in range(n))))
Rsc = sp.simplify(sum(gi[i, j]*Ric[i, j] for i in range(n) for j in range(n)))
Ricu = sp.Matrix(n, n, lambda i, j: sp.simplify(sum(gi[i, A_]*gi[j, B_]*Ric[A_, B_]
        for A_ in range(n) for B_ in range(n))))
Ruu = [[[[sp.simplify(sum(gi[A_, i]*gi[B_, j]*gi[C_, k]*gi[D_, l]*Rd[i][j][k][l]
        for i in range(n) for j in range(n) for k in range(n) for l in range(n)))
        for D_ in range(n)] for C_ in range(n)] for B_ in range(n)] for A_ in range(n)]
GB4 = sp.simplify(Rsc**2 - 4*sum(Ricu[i, j]*Ric[i, j] for i in range(n) for j in range(n))
      + sum(Ruu[A_][B_][C_][D_]*Rd[A_][B_][C_][D_] for A_ in range(n) for B_ in range(n)
            for C_ in range(n) for D_ in range(n)))
print(f"   R = {Rsc},   Ric^2 = {sp.simplify(sum(Ricu[i,j]*Ric[i,j] for i in range(n) for j in range(n)))},"
      f"   GB = {GB4}")

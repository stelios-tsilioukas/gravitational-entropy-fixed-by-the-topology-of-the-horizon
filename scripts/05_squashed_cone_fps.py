"""Paper Sec. III.D and Sec. VI ("Cuts with extrinsic curvature"), Eq. (squashedFPS).

On smooth squashed cones the Fursaev-Patrushev-Solodukhin defect formulas read, to the
order relevant for the entropy (k_a = trace of the extrinsic curvature K^(a)_ij along the
unit normal a, R_aa and R_abab the ambient curvature in the normal directions):
    int R^2          -> n int R^2          + 8 pi (1-n) oint R
    int Ric^2        -> n int Ric^2        + 4 pi (1-n) oint ( R_aa - k_a k_a / 2 )
    int Riem^2       -> n int Riem^2       + 8 pi (1-n) oint ( R_abab - Tr K^(a) K^(a) )
Part 1 combines them with the Gauss-Bonnet coefficients (1, -4, 1).
Part 2 checks symbolically that the combination is the intrinsic curvature of the cut,
    R_Sigma = R - 2 R_aa + R_abab + k_a k_a - Tr K^(a) K^(a)      (Gauss-Codazzi),
for cuts Sigma = {x0 = const, x1 = const} of a GENERIC metric
    ds^2 = A dx0^2 + B dx1^2 + C (dtheta^2 + sin^2 theta dphi^2),
with A, B, C arbitrary functions of (x0, x1), so that both normals carry extrinsic curvature.
Hence the Gauss-Bonnet defect is 8 pi (1-n) oint R_Sigma = 32 pi^2 (1-n) chi(Sigma) on every
smooth closed cut, with no extrinsic-curvature correction.
"""
import sympy as sp
from curvature import riemann_up

# ---------------------------------------------------------------- part 1
n = sp.symbols('n')
R, Raa, Rabab, kk, trKK = sp.symbols('R R_aa R_abab k_ak_a TrKK')
defect = (8*sp.pi*(1-n)*R
          - 4*4*sp.pi*(1-n)*(Raa - kk/2)
          + 8*sp.pi*(1-n)*(Rabab - trKK))
target = 8*sp.pi*(1-n)*(R - 2*Raa + Rabab + kk - trKK)
print("1. GB defect = 8 pi (1-n) [R - 2R_aa + R_abab + k_a k_a - Tr K K] :",
      sp.simplify(defect - target) == 0)

# ---------------------------------------------------------------- part 2
x0, x1, th, ph = sp.symbols('x0 x1 theta phi', real=True)
X = [x0, x1, th, ph]
A = sp.Function('A')(x0, x1)
B = sp.Function('B')(x0, x1)
C = sp.Function('C')(x0, x1)
g = sp.diag(A, B, C, C*sp.sin(th)**2)
ginv = g.inv()
Ru = riemann_up(g, X)
# riemann_up stores Ru[l][i][j][k] = R^l_{k i j}, with R(d_i, d_j) d_k = R^l_{k i j} d_l
Rlow = lambda l, k, i, j: sp.simplify(sum(g[l, m]*Ru[m][i][j][k] for m in range(4)))   # R_{lkij}
Ric = sp.Matrix(4, 4, lambda i, j: sp.simplify(sum(Ru[l][l][i][j] for l in range(4))))
Rs = sp.simplify(sum(ginv[i, j]*Ric[i, j] for i in range(4) for j in range(4)))

# orthonormal normals e_0 = A^{-1/2} d_0, e_1 = B^{-1/2} d_1
R_aa = sp.simplify(Ric[0, 0]/A + Ric[1, 1]/B)
# R_abab = sum over ordered normal pairs = 2 K(e0, e1), with K(X,Y) = g(R(X,Y)Y, X) = R_{0101}/(A B)
R_abab = sp.simplify(2*Rlow(0, 1, 0, 1)/(A*B))
# extrinsic curvature of Sigma along e_a: K^(a)_ij = (1/2) e_a(h_ij), h = C dOmega^2
h = sp.diag(C, C*sp.sin(th)**2)
hinv = h.inv()
kk, trKK = 0, 0
for xa, Na in [(x0, A), (x1, B)]:
    K = h.diff(xa)/(2*sp.sqrt(Na))
    KK = hinv*K
    kk += KK.trace()**2
    trKK += (KK*KK).trace()
R_Sigma = 2/C                       # intrinsic scalar curvature of a round S^2 of area 4 pi C
gauss = sp.simplify(Rs - 2*R_aa + R_abab + kk - trKK - R_Sigma)
print("2. R - 2R_aa + R_abab + k_a k_a - Tr K K - R_Sigma =", gauss,
      "  (generic A, B, C)")
print("   k_a k_a - Tr K K =", sp.simplify(kk - trKK), " (extrinsic part, nonzero in general)")
print("   => GB defect = 8 pi (1-n) oint R_Sigma dA = 32 pi^2 (1-n) chi(Sigma) on every smooth cut")

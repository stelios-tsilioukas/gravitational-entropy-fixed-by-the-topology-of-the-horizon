"""Paper Sec. XII: topological black holes in AdS_4, Eq. (Sads), first law within a
sector, entropy positivity bound, Hawking-Page shift Eqs. (HPshift)-(THP).
(The numerical r_h,min values for |Gamma| = 2,3,4 use the companion-paper conjecture.)

The anti-de Sitter route.

Hawking's topology theorem forbids genus > 0 horizon cross-sections for stationary
asymptotically flat black holes obeying the dominant energy condition, so the two
configurations Eq. (Stopinv) compares do not both exist there. In AdS they do:
topological black holes

    ds^2 = -f dt^2 + dr^2/f + r^2 dSigma_{g,k}^2,   f = k - 2GM/r + r^2/L^2

with k = -1 and Sigma a compact hyperbolic surface of genus g >= 2 are exact
solutions. Since Gauss-Bonnet is topological in four dimensions the solutions are
untouched and only the entropy moves.
"""
import sympy as sp

r, rh, L, G, al, M, T, g, k = sp.symbols('r r_h L G alpha M T g k', positive=True)

# ---------------------------------------------------------------- generic k
f  = k - 2*G*M/r + r**2/L**2
Ms = sp.solve(sp.Eq(f.subs(r, rh), 0), M)[0]
Ts = sp.simplify(sp.diff(f.subs(M, Ms), r).subs(r, rh)/(4*sp.pi))
print("Topological black holes, general k:")
print("   2GM =", sp.simplify(2*G*Ms), "      T =", sp.simplify(Ts))

# ---------------------------------------------------------------- hyperbolic
print("\n" + "="*74)
print("Hyperbolic family, k = -1: Sigma_g compact, unit curvature, area 4 pi (g-1)\n")
Vg   = 4*sp.pi*(g-1)                      # area of the unit-curvature Sigma_g
A    = rh**2*Vg
chi  = 2 - 2*g
S    = sp.simplify(A/(4*G) + 2*sp.pi*al*chi/G)
print("   A            =", sp.simplify(A), "     chi =", chi)
print("   S = A/4G + 2 pi alpha chi/G =", sp.factor(sp.simplify(S)))
print("\n   The genus factors out completely and the coupling enters as a shift of")
print("   the horizon radius squared:  r_h^2 -> r_h^2 - 4 alpha.")

# first law, within a sector (g fixed, so delta chi = 0)
Mh = sp.simplify(Ms.subs(k, -1))
Th = sp.simplify(Ts.subs(k, -1))
Etot = sp.simplify(Mh*Vg/(4*sp.pi))       # mass with the transverse volume factor
dM = sp.diff(Etot, rh); dS = sp.diff(S, rh)
print("\n   First law check within a sector:")
print("      dE/dr_h / (T dS/dr_h) =", sp.simplify(dM/(Th*dS)))
print("      d(alpha-part of S)/dr_h =", sp.diff(2*sp.pi*al*chi/G, rh),
      "   -> the topological charge has no conjugate variation")

# entropy positivity
print("\n   Entropy positivity (alpha > 0) requires r_h^2 >= 4 alpha, i.e.")
print("      r_h >= 2 sqrt(alpha) = l_P sqrt(log|Gamma|/pi)   [using alpha = l_P^2 log|Gamma|/4pi]")
import math
for N in (2, 3, 4):
    print(f"      |Gamma| = {N}:  r_h,min/l_P = {math.sqrt(math.log(N)/math.pi):.4f}")

# ---------------------------------------------------------------- saddles
print("\n" + "="*74)
print("The topological weight distinguishes bulk fillings of the same boundary\n")
print("   boundary S^1 x Sigma_g;   black-hole filling D^2 x Sigma_g has chi = 2-2g,")
print("   a thermal-AdS-type filling S^1 x M_3 has chi(S^1 x M_3) = 0.")
print("   The Gauss-Bonnet term weights them by e^{2 pi alpha chi/G}: the four-")
print("   dimensional analogue of e^{S_0 chi} in JT.\n")

# spherical case: Hawking-Page with the topological term
print("   Spherical case k=+1 (chi = 2), where the transition is standard:")
Mk = sp.simplify(Ms.subs(k, 1)); Tk = sp.simplify(Ts.subs(k, 1))
Sk = sp.pi*rh**2/G                                   # A/4G with A = 4 pi r_h^2
F0 = sp.simplify(Mk - Tk*Sk)
print("      F_0(r_h) =", sp.factor(sp.simplify(F0)), "   (zero at r_h = L: standard)")
# Gauss-Bonnet free energy: I_GB = -2 pi alpha chi/G is beta-independent, F = I/beta
FGB_bh  = -2*sp.pi*al*2/G*Tk            # chi = 2 for D^2 x S^2
FGB_ads = 0                             # chi = 0 for S^1 x B^3
dF = sp.simplify(F0 + FGB_bh - FGB_ads)
x, a = sp.symbols('x a', positive=True)      # x = r_h^2/L^2 , a = alpha/L^2
cond = sp.simplify(sp.expand(dF.subs(rh, L*sp.sqrt(x)).subs(al, a*L**2)*4*G/L))
print("      transition condition (times 4G/L):", sp.factor(sp.simplify(cond)))
poly = sp.simplify(sp.numer(sp.together(cond)))
print("      as a polynomial in x:", sp.factor(sp.expand(sp.simplify(poly/sp.sqrt(x)))))
sol = sp.solve(sp.Eq(x*(1-x) - 4*a*(1+3*x), 0), x)
print("      exact roots:", [sp.simplify(s) for s in sol])
ser = [sp.series(s, a, 0, 2).removeO() for s in sol]
print("      small-alpha:", ser)
THP = sp.simplify(Tk.subs(rh, L*sp.sqrt(1-16*a)))
print("      T_HP =", sp.simplify(sp.series(THP.subs(a, al/L**2), al, 0, 2).removeO()),
      "  -> fractional shift -4 alpha/L^2")

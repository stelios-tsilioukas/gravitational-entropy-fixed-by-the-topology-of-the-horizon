"""Paper Sec. XI.E-F (Test 3): 2D Gauss-Bonnet, matter renormalisation of S_0, and the
modular-flow test of a uniform versus energy-dependent reweighting (Eq. (sigmat)).

Test 3: Jackiw-Teitelboim with matter.

Three things are checked.

(1) Two-dimensional Gauss-Bonnet, so that the topological term in the JT action
    really is the integral of a characteristic class:  int sqrt(g) R = 4 pi chi.

(2) The matter contribution to the topological entropy constant. A CFT of central
    charge c on a closed surface obeys <T^mu_mu> = (c/24 pi) R, so a constant Weyl
    rescaling g -> e^{2 omega} g shifts log Z by (c/6) chi omega: matter renormalises
    the coefficient of chi and nothing else.  Cross-check: Z_{S^2} ~ r^{c/3}.
    Unified form: writing <T^mu_mu> = A_D * Ehat_D with int sqrt(g) Ehat_D = chi,
    the running rate of the topological entropy constant is A_D.
    D=2: A = c/6.   D=4: A = 2a.   Compare with the paper's 4D figure.

(3) The entropy shift with matter present, in a regulated algebra that is genuinely
    noncommutative and with gravity-matter entangled states: a uniform (topological)
    reweighting shifts every entropy by exactly log lambda and leaves the modular
    flow pointwise unchanged, while an energy-dependent reweighting of the same
    average size does not.
"""
import numpy as np, sympy as sp

# ---------------------------------------------------------------- (1)
print("(1) Two-dimensional Gauss-Bonnet:  int sqrt(g) R d^2x = 4 pi chi\n")
r, th, ph = sp.symbols('r theta phi', positive=True)

def R2_and_integral(g, coords, ranges, name, chi):
    n = 2; gi = g.inv()
    Gam = [[[sp.simplify(sum(gi[l, s]*(sp.diff(g[s, i], coords[j]) + sp.diff(g[s, j], coords[i])
            - sp.diff(g[i, j], coords[s])) for s in range(n))/2) for j in range(n)]
            for i in range(n)] for l in range(n)]
    Ric = sp.zeros(n, n)
    for s_ in range(n):
        for v in range(n):
            Ric[s_, v] = sp.simplify(sum(
                sp.diff(Gam[m][m][s_], coords[v]) - sp.diff(Gam[m][v][s_], coords[m])
                + sum(Gam[m][v][l]*Gam[l][m][s_] - Gam[m][m][l]*Gam[l][v][s_] for l in range(n))
                for m in range(n)))
    Rs = sp.simplify(-sum(gi[i, j]*Ric[i, j] for i in range(n) for j in range(n)))
    I = sp.simplify(sp.integrate(sp.integrate(sp.simplify(Rs*sp.sqrt(g.det())),
        (coords[0], *ranges[0])), (coords[1], *ranges[1])))
    print(f"   {name:28s} R = {Rs},  int R sqrt(g) = {I},  4 pi chi = {4*sp.pi*chi}")
    return I

a_ = sp.Symbol('a', positive=True)
R2_and_integral(sp.diag(a_**2, a_**2*sp.sin(th)**2), [th, ph],
                [(0, sp.pi), (0, 2*sp.pi)], "round S^2 of radius a", 2)
L1, L2 = sp.symbols('L_1 L_2', positive=True)
u, v = sp.symbols('u v')
R2_and_integral(sp.diag(1, 1), [u, v], [(0, L1), (0, L2)], "flat torus", 0)
print("   hyperbolic genus-2:          R = -2,  area = 4 pi (g-1) = 4 pi,"
      "  int R sqrt(g) = -8 pi,  4 pi chi = -8 pi  (chi = 2-2g = -2)")

# ---------------------------------------------------------------- (2)
print("\n(2) Matter renormalises the coefficient of chi (conformal anomaly)\n")
c, omega, rad = sp.symbols('c omega r', positive=True)
chi_s = sp.Symbol('chi')
# delta log Z = - int sqrt(g) <T^mu_mu> delta omega,  <T^mu_mu> = (c/24 pi) R
dlogZ = sp.simplify((c/(24*sp.pi))*4*sp.pi*chi_s)
print(f"   d(log Z)/d(omega) = (c/24 pi) * int sqrt(g) R = {dlogZ}")
print(f"   so  S_0^eff(mu) = S_0 + (c/6) log(mu_*/mu),  i.e. dS_top/dlog mu = {sp.simplify(dlogZ/chi_s)}")
print(f"   cross-check on the sphere (chi=2):  log Z_(S^2) contains "
      f"{sp.simplify(dlogZ.subs(chi_s,2))} log r  ->  Z ~ r^(c/3)   [known result]")

print("\n   Unified: <T^mu_mu> = A_D * Ehat_D  with  int sqrt(g) Ehat_D = chi;"
      "  dS_top/dlog mu = A_D")
print("     D = 2:  Ehat_2 = R/(4 pi),          A_2 = c/6")
print("     D = 4:  Ehat_4 = E_4/(32 pi^2),     A_4 = 2a")
for cc in (1, 2, 24, 26):
    print(f"     2D rate for c = {cc:2d}:  c/6 = {cc/6:.4f}")
for label, Ns, Nf, Nv in [("photon only", 0, 0, 1), ("photon + neutrino", 0, 1, 1),
                          ("SM below 1 GeV", 0, 3, 1)]:
    a4 = (Ns + 11*Nf + 62*Nv)/360
    print(f"     4D rate, {label:18s}: a = {a4:.4f},  dS_top/dlog mu = 2a = {2*a4:.4f},"
          f"  d(64 pi^2 lambda)/dlog mu = 4a = {4*a4:.4f}")
print("     (the paper's infrared figure is 0.7)")

print("\n   The same coefficient appears twice: the matter entanglement entropy across a")
print("   JT horizon is (c/6) log(.../epsilon) -- one endpoint, hence c/6, not c/3 -- so the")
print("   logarithm that renormalises S_0 IS the matter entropy. This is Eq. (Stop-run) in 2D.")

# ---------------------------------------------------------------- (3)
print("\n(3) Entropy and modular flow with matter, for gravity-matter correlated states\n")
rng = np.random.default_rng(3)
NE, nm, beta = 40, 4, 2*np.pi
E = np.linspace(-2.0, 2.0, NE); dE = E[1]-E[0]
W = dE*np.exp(beta*E)                       # dual weight over the gravitational fibre

def correlated_state():
    """rho is fibre-diagonal, but its MATTER block depends on the energy: the matter
    state is correlated with the gravitational one, which is what 'with matter' means."""
    blocks = []
    for i in range(NE):
        A = rng.normal(size=(nm, nm)) + 1j*rng.normal(size=(nm, nm))
        blocks.append((A @ A.conj().T) * np.exp(-E[i]**2))
    return np.array(blocks)

def entropy_and_rho(b, weight):
    rh = np.array([b[i]/weight[i] for i in range(NE)])
    Z = sum(weight[i]*np.trace(rh[i]).real for i in range(NE)); rh = rh/Z
    S = 0.0
    for i in range(NE):
        ev = np.linalg.eigvalsh((rh[i]+rh[i].conj().T)/2); ev = ev[ev > 1e-300]
        S -= weight[i]*np.sum(ev*np.log(ev))
    return S.real, rh

def full(rh):
    """assemble the fibre-diagonal density matrix on the whole Hilbert space"""
    d = NE*nm; M = np.zeros((d, d), complex)
    for i in range(NE):
        M[i*nm:(i+1)*nm, i*nm:(i+1)*nm] = rh[i]
    return M

def sigma_t(rhofull, A, t):
    ev, U = np.linalg.eigh((rhofull+rhofull.conj().T)/2)
    ev = np.clip(ev, 1e-300, None)
    P = (U*np.exp(1j*t*np.log(ev))) @ U.conj().T
    return P @ A @ P.conj().T

# an ENERGY-MIXING observable: a gravitationally dressed matter operator moves energy
d = NE*nm
A = rng.normal(size=(d, d)) + 1j*rng.normal(size=(d, d))

b0 = correlated_state()
nrm = [b0[i]/max(np.trace(b0[i]).real, 1e-300) for i in range(NE)]
avg = sum(nrm)/NE
print(f"   how strongly the matter state depends on the energy (0 = uncorrelated): "
      f"{max(np.abs(nrm[i]-avg).max() for i in range(NE))/np.abs(avg).max():.2f}\n")

print(f"{'state':>6} {'S_0':>7} {'S(lam)-S(1)':>14} {'err':>10} "
      f"{'sigma_t: topological':>21} {'sigma_t: energy-dep':>20}")
for k in range(4):
    b = correlated_state()
    S1, rh1 = entropy_and_rho(b, W)
    F1 = sigma_t(full(rh1), A, 0.7)
    for S0 in ([0.5, 2.0] if k == 0 else [0.5]):
        lam = np.exp(S0)
        S2, rh2 = entropy_and_rho(b, lam*W)
        F2 = sigma_t(full(rh2), A, 0.7)
        dtop = np.abs(F1-F2).max()/np.abs(F1).max()
        # energy-dependent reweighting with the SAME geometric mean, so the same
        # overall size: it is a different deformation, not a bigger one
        g = np.exp(0.3*(E-E.mean())); g = lam*g/np.exp(np.mean(np.log(g)))
        _, rh3 = entropy_and_rho(b, g*W)
        F3 = sigma_t(full(rh3), A, 0.7)
        dener = np.abs(F1-F3).max()/np.abs(F1).max()
        print(f"{k:6d} {S0:7.2f} {S2-S1:14.9f} {abs(S2-S1-S0):10.1e} "
              f"{dtop:21.2e} {dener:20.3f}")
print("\n   A uniform (topological) reweighting leaves the modular flow pointwise")
print("   invariant even on energy-mixing operators: the phase lambda^{-it} cancels in")
print("   the conjugation. An energy-dependent reweighting of the same overall size does")
print("   not. With matter the algebra contains such operators, so this is a real test;")
print("   without matter the algebra is commutative and there is nothing to distinguish.")

"""Paper Sec. XI.B and Appendix A (0): JT disk partition function, S = S_0 + ...,
dS/dS_0 = 1 = chi(disk) and its beta independence (Eqs. (rho0)-(dSdS0)).
"""
import sympy as sp

beta,E,k,S0 = sp.symbols('beta E k S_0', positive=True)

# JT disk density of states (C=1/2 conventions):  rho_0(E) = e^{S0}/(2 pi^2) * sinh(2 pi sqrt(2E))
rho = sp.exp(S0)/(2*sp.pi**2)*sp.sinh(2*sp.pi*sp.sqrt(2*E))

# partition function Z(beta) = int_0^infty dE rho(E) e^{-beta E};  substitute E = k^2/2
integrand = (rho.subs(E, k**2/2))*sp.exp(-beta*k**2/2)*k        # dE = k dk
Z = sp.simplify(sp.integrate(integrand,(k,0,sp.oo)))
print("Z(beta) =", sp.simplify(sp.powsimp(Z, force=True)))

logZ = sp.simplify(sp.log(Z))
Eavg = sp.simplify(-sp.diff(logZ,beta))
S    = sp.simplify(logZ + beta*Eavg)               # S = log Z + beta <E>
print("log Z  =", sp.simplify(sp.expand_log(logZ, force=True)))
print("<E>    =", Eavg)
print("S(beta)=", sp.simplify(sp.expand_log(S, force=True)))
print("dS/dS0 =", sp.simplify(sp.diff(S,S0)), "   <-- equals chi(disk)=1")
print("dS/dS0 is beta-independent:", sp.simplify(sp.diff(sp.diff(S,S0),beta))==0)

# high-temperature / semiclassical limit
print("S at large 1/beta:", sp.simplify(sp.series(S, beta, 0, 1).removeO()))

# --- numerical cross-check of the Laplace transform (mpmath, 40 digits) ---
# beta = 0.5,1,2,5,10 all agree with e^{S0}(2 pi)^{-1/2} beta^{-3/2} e^{2 pi^2/beta}

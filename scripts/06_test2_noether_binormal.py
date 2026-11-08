"""Paper Sec. VII.A (Test 2), Eqs. (waldbinormal)-(GBcontraction): Gauss-Bonnet Wald
entropy from the binormal contraction on Schwarzschild, and Eq. (nofirstlaw).

Test 2, step 1: Gauss-Bonnet Wald entropy from the Noether charge,
binormal contraction done explicitly.   S = -2 pi oint P^abcd eps_ab eps_cd dA,
P^abcd = dL/dR_abcd,  eps_ab eps^ab = -2.
"""
import sympy as sp
M,r,t,th,ph,al,G = sp.symbols('M r t theta phi alpha G', positive=True)
n=4; x=[t,r,th,ph]
f = 1-2*M/r
g = sp.diag(-f, 1/f, r**2, r**2*sp.sin(th)**2); ginv=g.inv()

Gam=[[[sp.simplify(sum(ginv[l,s]*(sp.diff(g[s,i],x[j])+sp.diff(g[s,j],x[i])-sp.diff(g[i,j],x[s])) for s in range(n))/2)
       for j in range(n)] for i in range(n)] for l in range(n)]
# R^rho_{sigma mu nu} = d_mu Gam^rho_{nu sigma} - d_nu Gam^rho_{mu sigma} + Gam Gam
def Rup(rho,sig,mu,nu):
    return sp.simplify(sp.diff(Gam[rho][nu][sig],x[mu])-sp.diff(Gam[rho][mu][sig],x[nu])
        +sum(Gam[rho][mu][l]*Gam[l][nu][sig]-Gam[rho][nu][l]*Gam[l][mu][sig] for l in range(n)))
Ru=[[[[Rup(a,b,c,d) for d in range(n)] for c in range(n)] for b in range(n)] for a in range(n)]
Rdddd=[[[[sp.simplify(sum(g[a,m]*Ru[m][b][c][d] for m in range(n))) for d in range(n)] for c in range(n)] for b in range(n)] for a in range(n)]
# symmetry checks
s1=max(sp.simplify(Rdddd[a][b][c][d]+Rdddd[b][a][c][d]) for a in range(n) for b in range(n) for c in range(n) for d in range(n))
s2=max(sp.simplify(Rdddd[a][b][c][d]-Rdddd[c][d][a][b]) for a in range(n) for b in range(n) for c in range(n) for d in range(n))
print("Riemann symmetry checks (both should be 0):", s1, s2)

Ric=sp.Matrix(n,n,lambda s,v: sp.simplify(sum(Ru[m][s][m][v] for m in range(n))))
Rs=sp.simplify(sum(ginv[i,j]*Ric[i,j] for i in range(n) for j in range(n)))
print("Ricci, R for Schwarzschild:", sp.simplify(Ric), Rs)
Ricuu=sp.Matrix(n,n,lambda i,j: sp.simplify(sum(ginv[i,a]*ginv[j,b]*Ric[a,b] for a in range(n) for b in range(n))))
Ruuuu=[[[[sp.simplify(sum(ginv[a,i]*ginv[b,j]*ginv[c,k]*ginv[d,l]*Rdddd[i][j][k][l]
           for i in range(n) for j in range(n) for k in range(n) for l in range(n)))
           for d in range(n)] for c in range(n)] for b in range(n)] for a in range(n)]

# binormal of the (t,r) plane
eps=sp.zeros(n,n); eps[0,1]=1; eps[1,0]=-1
nrm=sp.simplify(sum(ginv[a,c]*ginv[b,d]*eps[a,b]*eps[c,d] for a in range(n) for b in range(n) for c in range(n) for d in range(n)))
eps=eps/sp.sqrt(sp.simplify(-nrm/2))
print("eps_ab eps^ab =", sp.simplify(sum(ginv[a,c]*ginv[b,d]*eps[a,b]*eps[c,d] for a in range(n) for b in range(n) for c in range(n) for d in range(n))))

P_EH=[[[[sp.Rational(1,2)*(ginv[a,c]*ginv[b,d]-ginv[a,d]*ginv[b,c]) for d in range(n)] for c in range(n)] for b in range(n)] for a in range(n)]
P_GB=[[[[sp.simplify(2*Ruuuu[a][b][c][d]
        -2*(ginv[a,c]*Ricuu[b,d]-ginv[a,d]*Ricuu[b,c]-ginv[b,c]*Ricuu[a,d]+ginv[b,d]*Ricuu[a,c])
        +Rs*(ginv[a,c]*ginv[b,d]-ginv[a,d]*ginv[b,c])) for d in range(n)] for c in range(n)] for b in range(n)] for a in range(n)]
# Euler check: R_abcd P^abcd = 2*GB
GB=sp.simplify(Rs**2-4*sum(Ricuu[i,j]*Ric[i,j] for i in range(n) for j in range(n))
   +sum(Ruuuu[a][b][c][d]*Rdddd[a][b][c][d] for a in range(n) for b in range(n) for c in range(n) for d in range(n)))
eul=sp.simplify(sum(Rdddd[a][b][c][d]*P_GB[a][b][c][d] for a in range(n) for b in range(n) for c in range(n) for d in range(n))-2*GB)
print("GB scalar =", GB, "   Euler-theorem residual R_abcd P^abcd - 2GB =", eul)

def contract(P): return sp.simplify(sum(P[a][b][c][d]*eps[a,b]*eps[c,d] for a in range(n) for b in range(n) for c in range(n) for d in range(n)))
cEH,cGB=contract(P_EH),contract(P_GB)
R2=2/r**2
print("\nP_EH.eps.eps =", cEH, "  -> s_EH = -2pi*(1/16piG)*that =", sp.simplify(-2*sp.pi*cEH/(16*sp.pi*G)))
print("P_GB.eps.eps =", cGB, "   -4*^{(2)}R =", sp.simplify(-4*R2), "   difference:", sp.simplify(cGB+4*R2))
print("at the horizon r=2M:", sp.simplify(cGB.subs(r,2*M)))
A=4*sp.pi*(2*M)**2
print("S_GB =", sp.simplify(-2*sp.pi*(al/(16*sp.pi*G))*cGB.subs(r,2*M)*A), "   expected 2 pi alpha chi/G =", 4*sp.pi*al/G)

print("\n--- the GB charge drops out of the first law ---")
S_W = 4*sp.pi*M**2/G + 4*sp.pi*al/G            # A/4G + 2 pi alpha chi/G, chi=2
T   = 1/(8*sp.pi*M)
print("S_W(M) =", S_W, "   T dS_W/dM =", sp.simplify(T*sp.diff(S_W,M)), "  (= dM/G, alpha-free)")
print("d(S_top)/dM =", sp.diff(4*sp.pi*al/G,M), "  -> the topological charge has no conjugate variation")

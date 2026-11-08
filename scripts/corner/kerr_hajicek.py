"""Hajicek one-form on Kerr horizon cross-sections v=const, r=r+ (ingoing Kerr coordinates),
with l the Killing generator d_v + Omega_H d_phi and n the second null normal, l.n=-1.
Findings: omega_phi != 0 for a != 0 (rotation), all components vanish for a = 0, and omega_theta
is nonzero on these cuts. omega_theta dtheta is exact on the axisymmetric cut (cut and boost-gauge
dependent); on the bifurcation surface it vanishes by the (t,phi)->(-t,-phi) isometry.
The theta component therefore must not be used to argue that the bare form vanishes cut by cut;
the vanishing on the Kerr family follows from dilatation invariance (kerr_dilatation_check.py)."""
import sympy as sp
v,r,th,ph=sp.symbols('v r theta phi',real=True)
X=[v,r,th,ph]
def check(Mv,av):
    M,a=sp.nsimplify(Mv),sp.nsimplify(av)
    rho2=r**2+a**2*sp.cos(th)**2; Delta=r**2-2*M*r+a**2
    g=sp.zeros(4)
    g[0,0]=-(1-2*M*r/rho2); g[0,1]=g[1,0]=1
    g[0,3]=g[3,0]=-2*M*a*r*sp.sin(th)**2/rho2
    g[1,3]=g[3,1]=-a*sp.sin(th)**2
    g[2,2]=rho2
    g[3,3]=((r**2+a**2)**2-Delta*a**2*sp.sin(th)**2)*sp.sin(th)**2/rho2
    rp=M+sp.sqrt(M**2-a**2); OH=a/(rp**2+a**2)
    gi=g.inv()
    Gam=[[[sum(gi[m,k]*(sp.diff(g[k,i],X[j])+sp.diff(g[k,j],X[i])-sp.diff(g[i,j],X[k])) for k in range(4))/2
           for j in range(4)] for i in range(4)] for m in range(4)]
    out=[]
    for thv in [0.3,0.9,1.4,2.2]:
        sub={r:rp,th:thv}
        G=sp.Matrix(4,4,lambda i,j:g[i,j].subs(sub)).evalf(30)
        l=sp.Matrix([1,0,0,OH]).evalf(30)
        lcov=G*l
        A,B,C=sp.symbols('A B C')
        n=sp.Matrix([A,B,0,C])
        sol=sp.nsolve([ (n.T*G*sp.Matrix([0,0,0,1]))[0], (n.T*G*n)[0], (n.T*G*l)[0]+1],[A,B,C],[0.5,-0.5,0.1],prec=30)
        nv=sp.Matrix([sol[0],sol[1],0,sol[2]]); ncov=G*nv
        def omega(t):
            return -sum(ncov[m]*sum(Gam[m][t][nu].subs(sub)*l[nu] for nu in range(4)) for m in range(4)).evalf(30)
        out.append((thv, float(lcov[2]), float(lcov[3]), float(omega(2)), float(omega(3))))
    return out
for Mv,av in [(1,0.5),(1,0.9),(1,0.0)]:
    print('M=',Mv,'a=',av)
    for row in check(Mv,av):
        print('  theta=%.2f  l_theta=%.1e l_phi=%.1e  omega_theta=%.2e  omega_phi=%.6f'%row)

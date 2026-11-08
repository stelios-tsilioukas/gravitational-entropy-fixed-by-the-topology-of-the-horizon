"""Bare Euler corner form on Kerr cross-sections v=const, r=r+ (ingoing Kerr coords).
Computes Omega_E(delta_a, delta_M)/(8c) = int [delta_a(a_th) delta_M(b_ph) - delta_M(a_th) delta_a(b_ph)] dth dph
with fixed (theta,phi) identification and adapted frame e2=rho dth, e3=sqrt(h) dph.
Result: the integrand vanishes pointwise (Jacobian zero) to ~1e-13, because a, b and k^pm are
invariant under constant rescaling of the metric, so on the Kerr family they depend on a/M only.
The global dilatation (the area mode) is annihilated by the Euler corner form.
On the bifurcation surface a_th=0 in addition, by the (t,phi)->(-t,-phi) isometry."""
import sympy as sp, mpmath as mp
mp.mp.dps=30
v,r,th,ph,M,a=sp.symbols('v r theta phi M a',real=True)
X=[v,r,th,ph]
rho2=r**2+a**2*sp.cos(th)**2; Delta=r**2-2*M*r+a**2
g=sp.zeros(4)
g[0,0]=-(1-2*M*r/rho2); g[0,1]=g[1,0]=1
g[0,3]=g[3,0]=-2*M*a*r*sp.sin(th)**2/rho2
g[1,3]=g[3,1]=-a*sp.sin(th)**2
g[2,2]=rho2
g[3,3]=((r**2+a**2)**2-Delta*a**2*sp.sin(th)**2)*sp.sin(th)**2/rho2
gi=sp.simplify(g.inv())
# Gamma^m_{th nu} only
Gth=[[sp.simplify(sum(gi[m,k]*(sp.diff(g[k,2],X[nu])+sp.diff(g[k,nu],th)-sp.diff(g[2,nu],X[k])) for k in range(4))/2) for nu in range(4)] for m in range(4)]
gf=sp.lambdify((r,th,M,a),g,'mpmath'); Gf=sp.lambdify((r,th,M,a),Gth,'mpmath')
hf=sp.lambdify((r,th,M,a),g[3,3],'mpmath')
def a_theta(thv,Mv,av):
    rp=Mv+mp.sqrt(Mv**2-av**2); OH=av/(rp**2+av**2)
    G=mp.matrix(gf(rp,thv,Mv,av)); l=mp.matrix([1,0,0,OH])
    # n = A dv + B dr + C dphi (vector components), solve linear+quadratic
    def eqs(A,B,C):
        n=mp.matrix([A,B,0,C])
        return [ (n.T*G*mp.matrix([0,0,0,1]))[0], (n.T*G*n)[0], (n.T*G*l)[0]+1 ]
    A,B,C=mp.findroot(eqs,(0.5,-0.5,0.1))
    ncov=G*mp.matrix([A,B,0,C]); Gm=Gf(rp,thv,Mv,av)
    return -sum(ncov[m]*sum(Gm[m][nu]*l[nu] for nu in range(4)) for m in range(4))
def b_phi(thv,Mv,av):
    rp=Mv+mp.sqrt(Mv**2-av**2)
    f=lambda t: mp.sqrt(hf(rp,t,Mv,av))
    return -mp.diff(f,thv)/mp.sqrt(rp**2+av**2*mp.cos(thv)**2)
def Omega(Mv,av,eps=mp.mpf('1e-6')):
    def integrand(t):
        da_a=(a_theta(t,Mv,av+eps)-a_theta(t,Mv,av-eps))/(2*eps)
        dM_a=(a_theta(t,Mv+eps,av)-a_theta(t,Mv-eps,av))/(2*eps)
        da_b=(b_phi(t,Mv,av+eps)-b_phi(t,Mv,av-eps))/(2*eps)
        dM_b=(b_phi(t,Mv+eps,av)-b_phi(t,Mv-eps,av))/(2*eps)
        return da_a*dM_b-dM_a*da_b
    return 2*mp.pi*mp.quad(integrand,[mp.mpf('1e-4'),mp.pi/2,mp.pi-mp.mpf('1e-4')])
for av in ['0.3','0.6','0.9']:
    print('a/M=',av,' Omega_E(delta_a,delta_M)/(8c) on v=const cut =',mp.nstr(Omega(mp.mpf(1),mp.mpf(av)),10))

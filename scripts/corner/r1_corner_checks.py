"""Exact R1 corner checks. Run with Python 3, sympy and mpmath.

These checks cover local identities and specified tangent vectors, not the
existence of a gravitational quantum algebra or a complete edge theory.
The antisymmetric connection uses upper internal indices and epsilon_0123=+1.
"""
import itertools
import json
import sympy as sp
import mpmath as mp

mp.mp.dps = 120
c, alpha, G = sp.symbols('c alpha G', nonzero=True, real=True)
pi = sp.pi
eta = [-1, 1, 1, 1]
pairs = list(itertools.combinations(range(4), 2))
checks = []

def passed(name, result, blind):
    checks.append(dict(check=name, result=str(result), does_not_test=blind))
    print(f'PASS {name}: {result}\n  Does not test: {blind}')

def w(x, y):
    return sp.expand(x[0]*y[1]-x[1]*y[0])

def signed(conn, a, b):
    if a == b:
        return sp.zeros(2, 1)
    return conn[a,b] if a < b else -conn[b,a]

def eps_pair(x, y):
    return sp.expand(sum(sp.LeviCivita(a,b,d,e)*w(signed(x,a,b),signed(y,d,e))
                         for a,b,d,e in itertools.permutations(range(4))))

x = {(a,b):sp.Matrix(sp.symbols(f'x{a}{b}0 x{a}{b}1')) for a,b in pairs}
y = {(a,b):sp.Matrix(sp.symbols(f'y{a}{b}0 y{a}{b}1')) for a,b in pairs}
full = 2*c*eps_pair(x,y)
split = 8*c*(w(x[0,1],y[2,3])-w(y[0,1],x[2,3])
             -w(x[0,2],y[1,3])+w(y[0,2],x[1,3])
             +w(x[0,3],y[1,2])-w(y[0,3],x[1,2]))
assert sp.expand(full-split) == 0
passed('Full Lorentz epsilon expansion', 'mixed coefficient 8c; six complementary terms',
       'boundary constraints, global frame identifications or quantization')

kp_x={i:(x[0,i]-x[1,i])/sp.sqrt(2) for i in [2,3]}
km_x={i:(x[0,i]+x[1,i])/sp.sqrt(2) for i in [2,3]}
kp_y={i:(y[0,i]-y[1,i])/sp.sqrt(2) for i in [2,3]}
km_y={i:(y[0,i]+y[1,i])/sp.sqrt(2) for i in [2,3]}
def E(kp,km):
    return w(kp[2],km[3])-w(kp[3],km[2])
null_split=8*c*(w(x[0,1],y[2,3])-w(y[0,1],x[2,3])-E(kp_x,km_y)+E(kp_y,km_x))
assert sp.expand(full-null_split) == 0
gauss_connection = sum(eta[j]*w(signed(x,2,j),signed(x,j,3)) for j in range(4))
assert sp.simplify(gauss_connection-E(kp_x,km_x)) == 0
passed('Null split and Gauss equation', 'R23 = db + epsilon_ij K+_i wedge K-_j',
       'smoothness or field equations of a chosen spacetime')

# General Gauss contraction for symmetric second fundamental forms.
a,b,d,e,f,h=sp.symbols('a b d e f h')
P=sp.Matrix([[a,b],[b,d]])
Q=sp.Matrix([[e,f],[f,h]])
eps_PQ=E({2:P.row(0),3:P.row(1)},{2:Q.row(0),3:Q.row(1)})
assert sp.expand(eps_PQ-(sp.trace(P)*sp.trace(Q)-sp.trace(P*Q))) == 0
passed('Gauss extrinsic scalar contraction', eps_PQ,
       'ambient Ricci or Weyl curvature restrictions')

# Constant sectional curvature: epsilon R R = 24 K^2 vol.
curvature=sp.symbols('K',real=True)
euler_scalar=sum(sp.LeviCivita(*perm)**2*curvature**2
                 for perm in itertools.permutations(range(4)))
radius=sp.symbols('r',positive=True)
assert euler_scalar == 24*curvature**2
euler_s4=sp.simplify(euler_scalar.subs(curvature,1/radius**2)*8*pi**2*radius**4/3)
assert euler_s4 == 64*pi**2
passed('Closed S4 Euler normalization', euler_s4,
       'a general boundary or joint completion')

# Frame Noether contractions, including Einstein-Hilbert normalization first.
kappa,area,chi=sp.symbols('kappa area chi',nonzero=True)
def boost_component(a,b):
    return kappa if (a,b)==(0,1) else (-kappa if (a,b)==(1,0) else 0)
def tangent_component(a,b,value):
    return value if (a,b)==(2,3) else (-value if (a,b)==(3,2) else 0)
def noether_epsilon_sum(value):
    return sum(sp.LeviCivita(a,b,d,e)*boost_component(a,b)*tangent_component(d,e,value)
               for a,b,d,e in itertools.permutations(range(4)))
q_EH=noether_epsilon_sum(area)/(32*pi*G)
S_EH=sp.simplify(2*pi*q_EH/kappa)
assert S_EH == area/(4*G)
Q_GB=2*c*noether_epsilon_sum(2*pi*chi)
S_GB=sp.simplify((2*pi*Q_GB/kappa).subs(c,alpha/(16*pi*G)))
assert S_GB == 2*pi*alpha*chi/G
assert sp.diff(q_EH/kappa,area)==1/(8*pi*G)
assert sp.diff(Q_GB/kappa,area)==0
passed('Wald and Jacobson Myers normalization', f'EH={S_EH}; GB={S_GB}',
       'equality of bare Wald and JM away from vanishing cross-extrinsic term')

# Spherical bifurcation family. Radius is free; intrinsic connection is not.
theta,phi=sp.symbols('theta phi',real=True)
b_sphere=sp.Matrix([0,-sp.cos(theta)])
assert sp.diff(b_sphere,radius) == sp.zeros(2,1)
area_sphere=4*pi*radius**2
assert sp.diff(area_sphere,radius) == 8*pi*radius
R_surface=2/radius**2
R_integral=sp.integrate(R_surface*radius**2*sp.sin(theta),(theta,0,pi),(phi,0,2*pi))
assert R_integral == 8*pi
passed('Static spherical horizon with changing radius',
       'Omega_E=0, dArea/dr=8*pi*r, integral R=8*pi',
       'nonspherical or nonstationary perturbations; this example alone is symmetry blind')

# Spherical symmetry by itself leaves two independent extrinsic null jets.
P_iso=sp.eye(2)
Q_iso=sp.eye(2)
omega_iso=sp.simplify(-8*c*(sp.trace(P_iso)*sp.trace(Q_iso)-sp.trace(P_iso*Q_iso))*area)
assert omega_iso == -16*c*area
passed('Spherical extrinsic negative control',omega_iso,
       'Einstein equations or persistence of a Killing bifurcation surface for this local jet family')

# Nonzero mixed form at K+=K-=0, including fixed area to first order.
lam=sp.cos(theta)
fshape=sp.cos(theta)
dlam=sp.Matrix([sp.diff(lam,theta),sp.diff(lam,phi)])
delta_b=sp.Matrix([sp.diff(fshape,phi)/sp.sin(theta),-sp.sin(theta)*sp.diff(fshape,theta)])
omega_mixed=sp.simplify(8*c*sp.integrate(w(dlam,delta_b),(theta,0,pi),(phi,0,2*pi)))
delta_area=sp.integrate(2*fshape*radius**2*sp.sin(theta),(theta,0,pi),(phi,0,2*pi))
assert omega_mixed == -64*pi*c/3 and delta_area == 0
passed('Boost versus conformal shape negative control',omega_mixed,
       'whether a chosen completed theory quotients this local boost as gauge')

# Independent TT waves, exact linearized vacuum equations on R1,1 x T2.
u,v,X,Y=sp.symbols('u v X Y',real=True)
coord=[u,v,X,Y]
flat=sp.Matrix([[0,-1,0,0],[-1,0,0,0],[0,0,1,0],[0,0,0,1]])
inv=flat.inv()
wave_checks=[]
extrinsic_checks=[]
for z in [u,v]:
    perturb=sp.diag(0,0,sp.sin(z),-sp.sin(z))
    trace=sum(inv[i,j]*perturb[i,j] for i in range(4) for j in range(4))
    div=[sum(inv[r,s]*sp.diff(perturb[r,j],coord[s]) for r in range(4) for s in range(4)) for j in range(4)]
    assert trace == 0 and all(val == 0 for val in div)
    for i in range(4):
        for j in range(4):
            box=sum(inv[r,s]*sp.diff(perturb[i,j],coord[r],coord[s]) for r in range(4) for s in range(4))
            Ricci=sp.simplify((sp.diff(div[j],coord[i])+sp.diff(div[i],coord[j])-box-sp.diff(trace,coord[i],coord[j]))/2)
            assert Ricci == 0
    assert perturb.subs({u:0,v:0}) == sp.zeros(4)
    def delta_Gamma(upper,i,j):
        return sp.simplify(sum(inv[upper,k]*(sp.diff(perturb[k,j],coord[i])
                         +sp.diff(perturb[k,i],coord[j])-sp.diff(perturb[i,j],coord[k]))
                         for k in range(4))/2)
    # a_i=-n_mu nabla_i l^mu=Gamma^u_{iu} in this fixed null gauge.
    for i in [2,3]:
        assert delta_Gamma(0,i,0)==0
    Kplus=sp.Matrix(2,2,lambda i,j:delta_Gamma(1,i+2,j+2)).subs({u:0,v:0})
    Kminus=sp.Matrix(2,2,lambda i,j:delta_Gamma(0,i+2,j+2)).subs({u:0,v:0})
    assert Kplus==(perturb[2:4,2:4].diff(u)/2).subs({u:0,v:0})
    assert Kminus==(perturb[2:4,2:4].diff(v)/2).subs({u:0,v:0})
    extrinsic_checks.append((Kplus,Kminus))
    wave_checks.append(perturb)
P=extrinsic_checks[0][0]
Q=extrinsic_checks[1][1]
assert P==sp.diag(sp.Rational(1,2),-sp.Rational(1,2)) and Q==P
assert extrinsic_checks[0][1]==sp.zeros(2) and extrinsic_checks[1][0]==sp.zeros(2)
omega_wave=sp.simplify(-8*c*(sp.trace(P)*sp.trace(Q)-sp.trace(P*Q))*area)
assert omega_wave == 4*c*area
passed('On shell linearized TT counterexample',omega_wave,
       'a global nonlinear two-parameter black-hole family; no null or completed boundary condition imposed')

# Two-form cancellation is a field-space derivative, not adding delta(boundary action) alone.
bare=2*c*eps_pair(x,y)
delta_corner=c*(eps_pair(x,y)-eps_pair(y,x))
assert sp.expand(bare-delta_corner) == 0
passed('Fixed reference transgression corner cancellation','Omega_bare - integral delta C_E = 0',
       'selection of this completion by the gravitational algebra, moving normals, or corner gluing')

# Conditional compact BF quantization. Each conjugate holonomy pair is tested.
level=sp.symbols('k',integer=True,nonzero=True)
Kmatrix=sp.Matrix([[0,level],[level,0]])
assert Kmatrix.det() == -level**2
prequantum_period=sp.simplify(8*c*(2*pi)**2/(2*pi))
assert prequantum_period == 16*pi*c
passed('Conditional compact BF level','k=16*pi*c=s/(2*pi); det K=-k^2; dim H_g=|k|^(2g)',
       'compactness, flatness or independence of the actual gravitational connections')
for kval in [1,2,3,5]:
    s_value=2*mp.pi*kval
    assert mp.log(kval) < s_value
    print('  k=',kval,'g=0..3 dimensions=',[kval**(2*g) for g in range(4)],
          'log(handle dimension)=',mp.nstr(2*mp.log(kval),22),
          '2|s|=',mp.nstr(2*s_value,22))
passed('Compact BF growth mismatch','log|k| != 2*pi*|k| for every positive integer k (analytic inequality)',
       'other interacting or nonabelian edge theories with additional input')

# Requested cross-task regression entries: arithmetic only; no R2/R3 research.
L,g=sp.symbols('L g',positive=True)
F=pi*L**2/(2*G)-2*pi*alpha/G
S_hyperbolic=pi*(g-1)*(L**2-4*alpha)/G
assert sp.simplify(S_hyperbolic+F*(2-2*g))==0
assert 4*sp.Rational(1,360)==sp.Rational(1,90)
passed('Supplied cross-task normalization arithmetic',
       'S_hyperbolic=-F*chi at rh=L; 4*(1/360)=1/90',
       'CFT quotient universality, finite-N effects, anomaly determinants or one-loop replica equality')

print('\nAll R1 symbolic and numerical checks passed.')
print(json.dumps(checks,indent=2))

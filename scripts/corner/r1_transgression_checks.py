"""Nonabelian differential-form checks of the R1 boundary completion.

Polynomial connections are deliberately generic; no torsion or field equation
restriction is needed for the Chern-Weil identities. This is a check of local
identities, not a test of a global gravitational gauge quotient or null limit.
"""
import itertools
import sympy as s

x,y,z,w,p,q,t=s.symbols('x y z w p q t')
coords=[x,y,z,w]
eta=[-1,1,1,1]
pairs=list(itertools.combinations(range(4),2))

def tidy(f):
    return {i:s.expand(v) for i,v in f.items() if s.expand(v)!=0}

def add(*fs):
    result={}
    for f in fs:
        for i,v in f.items():
            result[i]=result.get(i,0)+v
    return tidy(result)

def scale(f,c):
    return tidy({i:c*v for i,v in f.items()})

def wedge(a,b):
    result={}
    for I,v in a.items():
        for J,u in b.items():
            K=I+J
            if len(set(K))<len(K):
                continue
            sign=(-1)**sum(K[i]>K[j] for i in range(len(K)) for j in range(i+1,len(K)))
            key=tuple(sorted(K))
            result[key]=result.get(key,0)+sign*v*u
    return tidy(result)

def d(a):
    return add(*(wedge({(i,):1},{I:s.diff(v,coord)})
                 for I,v in a.items() for i,coord in enumerate(coords)))

def diff(a,param):
    return tidy({I:s.diff(v,param) for I,v in a.items()})

def conn(a,i,j):
    if i==j:
        return {}
    return a[i,j] if i<j else scale(a[j,i],-1)

def pairing(a,b):
    return add(*(scale(wedge(conn(a,i,j),conn(b,k,l)),s.LeviCivita(i,j,k,l))
                 for i,j,k,l in itertools.permutations(range(4))))

def curvature(a):
    return {(i,j):add(d(a[i,j]),*(scale(wedge(conn(a,i,k),conn(a,k,j)),eta[k]) for k in range(4)))
            for i,j in pairs}

def variation(a,param):
    return {ij:diff(a[ij],param) for ij in pairs}

def difference(a,b):
    return {ij:add(a[ij],scale(b[ij],-1)) for ij in pairs}

A={(0,1):{(1,):1+p*x,(2,):q},
   (0,2):{(2,):p,(3,):x},
   (0,3):{(1,):q,(0,):z},
   (1,2):{(2,):1+q*y},
   (1,3):{(1,):x,(3,):p},
   (2,3):{(0,):1+p*z,(3,):q*x}}
fixed={(i,j):{} for i,j in pairs}
fixed.update({(0,1):{(0,):y},(2,3):{(3,):z},(0,2):{(1,):w},(1,3):{(2,):x}})
product={ij:(A[ij] if 0 not in ij else {}) for ij in pairs}
F=curvature(A)

for label,bar in [('fixed generic reference',fixed),('varying product reference',product)]:
    B=difference(A,bar)
    At={ij:add(bar[ij],scale(B[ij],t)) for ij in pairs}
    Ft=curvature(At)
    integrand=scale(pairing(B,Ft),2)
    T={I:s.integrate(v,(t,0,1)) for I,v in integrand.items()}
    Fbar=curvature(bar)
    assert not add(d(T),scale(pairing(F,F),-1),pairing(Fbar,Fbar))
    print('PASS Chern-Weil transgression:',label)
    Cs={}
    omitted=[]
    for param in [p,q]:
        deltaA=variation(A,param)
        deltaBar=variation(bar,param)
        C=pairing(B,{ij:add(deltaA[ij],deltaBar[ij]) for ij in pairs})
        theta=scale(pairing(deltaA,F),2)
        thetaBar=scale(pairing(deltaBar,Fbar),2)
        residual=add(diff(T,param),scale(theta,-1),thetaBar,d(C))
        assert not residual
        omitted.append(add(diff(T,param),scale(theta,-1),thetaBar))
        Cs[param]=C
    assert any(omitted), 'Test would be blind to omitting the corner potential'
    print('PASS delta T = theta - theta_bar - d C for both independent variations')
    print('PASS negative control: omitting d C leaves nonzero residual')
    fieldcurvature=add(diff(Cs[q],p),scale(diff(Cs[p],q),-1))
    target=scale(add(pairing(variation(A,p),variation(A,q)),
                     scale(pairing(variation(bar,p),variation(bar,q)),-1)),2)
    assert not add(fieldcurvature,scale(target,-1))
    if label.startswith('varying'):
        assert not pairing(variation(bar,p),variation(bar,q))
        assert not pairing(variation(bar,p),Fbar)
    print('PASS completed Euler corner form cancels; product-reference Euler pairings vanish')
    print('Does not test: moving boundary frames, large gauge quotient, gluing or a null limit.')

# A boundary check independent of the connection interpolation above.
r=s.symbols('r',positive=True)
delta=s.KroneckerDelta
def determinant_delta(I,J):
    return s.det(s.Matrix([[delta(a,b) for b in J] for a in I]))

Q2=0
for I in itertools.product(range(3),repeat=3):
    for J in itertools.product(range(3),repeat=3):
        D=determinant_delta(I,J)
        if D==0:
            continue
        curvature3=(delta(J[1],I[1])*delta(J[2],I[2])-delta(J[1],I[2])*delta(J[2],I[1]))/r**2
        Q2 += 2*D*(delta(J[0],I[0])/r)*(curvature3-s.Rational(2,3)*delta(J[1],I[1])*delta(J[2],I[2])/r**2)
Q2=s.simplify(Q2)
assert Q2==16/r**3
flat_ball=s.simplify(Q2*2*s.pi**2*r**3)
assert flat_ball==32*s.pi**2 and s.diff(flat_ball,r)==0
print('PASS flat B4: Myers Q2=16/r^3, integral=32*pi^2, radius derivative=0')

# Product rectangle x S2: all Euler curvature is in four right-angle joints.
surface_R_integral=8*s.pi
joint_total=4*(4*(s.pi/2)*surface_R_integral)
assert joint_total==64*s.pi**2
assert s.diff(4*s.pi*r**2,r)!=0
print('PASS rectangle x S2: four angle joints total64*pi^2; S2 area remains variable')
print('Does not test: generic nonproduct joints; their extrinsic endpoint terms are required.')
print('All transgression and geometric completion checks passed.')

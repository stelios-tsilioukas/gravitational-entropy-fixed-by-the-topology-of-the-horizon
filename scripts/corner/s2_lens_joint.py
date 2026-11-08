"""R1' check S2: gluing of facewise Euler completions at a joint.

Joint term derived as the limit of the Myers term over a rounded edge (radius delta -> 0).
In flat ambient space Myers Q2 = 16 det K. On the rounded piece {p + delta u(theta)}, K = diag(1/delta, k(theta))
with k(theta) the second fundamental form of the joint J along u(theta), volume delta dtheta dA:
     J2 = 16 int_fan dtheta int_J det k(theta) dA      (delta-independent in the limit)
Test: 4D lens = intersection of two balls of radius R in R^4 with centres at (0,0,0,+-d), chi = 1.
Caps: S^3 caps of angular radius theta0 (cos theta0 = d/R); joint: S^2 of radius a = R sin theta0,
k(theta) = cos(theta)/a * I_2, fan theta in [-Theta/2, Theta/2], Theta = pi - 2 theta0.
Also: general flat-ambient decomposition J2 = 4 Theta R_J + endpoint (extrinsic) terms.
"""
import sympy as sp
th0,R,th=sp.symbols('theta0 R theta',positive=True)
a=R*sp.sin(th0)
cap=16/R**3*4*sp.pi*R**3*sp.integrate(sp.sin(th)**2,(th,0,th0))   # Myers on one cap
Theta=sp.pi-2*th0
joint=16*sp.integrate((sp.cos(th)/a)**2,(th,-Theta/2,Theta/2))*4*sp.pi*a**2
total=sp.simplify(2*cap+joint)
print('caps =',sp.simplify(2*cap)); print('joint =',sp.simplify(joint)); print('total =',total)
assert sp.simplify(total-32*sp.pi**2)==0
# angle part alone (4 Theta int R_J) and the endpoint remainder
angle_part=4*Theta*8*sp.pi
print('angle part 4*Theta*int R_J =',sp.simplify(angle_part),'; endpoint (extrinsic) part =',sp.simplify(joint-angle_part))
assert sp.simplify(joint-angle_part-32*sp.pi*sp.sin(2*th0))==0
# general flat ambient: 8 int E(k(theta),k(theta)) with k = cos k1 + sin k2 (2x2 symmetric)
p=sp.symbols('p1:4'); q=sp.symbols('q1:4'); t1,t2=sp.symbols('t1 t2')
k1=sp.Matrix([[p[0],p[1]],[p[1],p[2]]]); k2=sp.Matrix([[q[0],q[1]],[q[1],q[2]]])
E=lambda P,Q:(P.trace()*Q.trace()-(P*Q).trace())
kt=sp.cos(th)*k1+sp.sin(th)*k2
J2=sp.integrate(sp.expand(8*E(kt,kt)),(th,t1,t2))
RJ=E(k1,k1)+E(k2,k2)          # Gauss equation, flat ambient: R_J = sum_a [(tr k_a)^2 - tr k_a^2]
endpoint=sp.simplify(J2-4*(t2-t1)*RJ)
print('general: J2 - 4*Theta*R_J =',sp.factor(sp.simplify(endpoint.rewrite(sp.cos))))
check=sp.simplify(endpoint-( 2*(sp.sin(2*t2)-sp.sin(2*t1))*(E(k1,k1)-E(k2,k2)) -4*(sp.cos(2*t2)-sp.cos(2*t1))*E(k1,k2) ))
assert check==0
print('=> J2 = 4 Theta R_J + 2[sin 2t]_{t1}^{t2}(E11-E22) - 4[cos 2t]_{t1}^{t2} E12   (flat ambient)')
def Euv(tt):
    ku=sp.cos(tt)*k1+sp.sin(tt)*k2; kv=-sp.sin(tt)*k1+sp.cos(tt)*k2
    return E(ku,kv)
cano=4*((t2-t1)*RJ+Euv(t1)-Euv(t2))
assert sp.simplify(sp.expand(J2-cano))==0
print('=> equivalently J2 = 4[ Theta R_J + E(k_u1,k_v1) - E(k_u2,k_v2) ],  E(L,Q)=trL trQ - tr LQ = 2 L^[A_A Q^B]_B :')
print('   the structure of Cano (arXiv:1803.00172) joint term, here DERIVED as the rounded-edge limit of the Myers term.')
print('Lorentzian joints: theta -> i psi (boost angle), cos -> cosh, sin -> i sinh.')
print('All S2 checks passed.')

"""Paper Sec. VII.C, Lemma 2 and Eq. (dualweight): trace rescaling in a discretised
crossed product, entropy shift log(lambda), state independence, invariance of
expectation values and relative entropies.

Test 2, step 3: what a c-number in the boost charge does to the type II trace.

Crossed product Mhat = M x_sigma R, realised on H (x) L^2(R) with the dual-weight
(canonical) trace
        Tr(a) = int dx  e^{beta x} Tr_H( <x| a |x> ).
Shifting the adjoined generator X -> X + c (a c-number addition to the boost charge)
sends the weight e^{beta x} -> e^{beta(x+c)}, i.e. Tr -> lambda Tr with lambda = e^{beta c}.
We check, for random states, that
   (i)  expectation values are unchanged,
   (ii) rho_hat -> rho_hat / lambda  and  S -> S + log lambda,  STATE-INDEPENDENTLY,
   (iii) relative entropies are unchanged (the constant is invisible to modular dynamics).
"""
import numpy as np
rng=np.random.default_rng(7)
N,Mx,beta,L = 3,240,2*np.pi,2.0
xs=np.linspace(-L,L,Mx); dx=xs[1]-xs[0]
W=dx*np.exp(beta*xs)                       # dual-weight density on the R factor

def randstate():
    blocks=[]
    for i in range(Mx):
        A=rng.normal(size=(N,N))+1j*rng.normal(size=(N,N)); B=A@A.conj().T
        blocks.append(B*np.exp(-xs[i]**2))               # normalisable profile
    return np.array(blocks)

def entropy(blocks,lam):
    """rho_hat_x = blocks_x /(lam*W_x)  normalised by Tr_lam; returns (S, normalised rho)."""
    rh=np.array([blocks[i]/(lam*W[i]) for i in range(Mx)])
    Z=sum(lam*W[i]*np.trace(rh[i]).real for i in range(Mx))
    rh=rh/Z
    S=0.0
    for i in range(Mx):
        ev=np.linalg.eigvalsh((rh[i]+rh[i].conj().T)/2); ev=ev[ev>1e-300]
        S-= lam*W[i]*np.sum(ev*np.log(ev))
    return S.real, rh

def relent(b1,b2,lam):
    _,r1=entropy(b1,lam); _,r2=entropy(b2,lam); s=0.0
    for i in range(Mx):
        e1,U1=np.linalg.eigh((r1[i]+r1[i].conj().T)/2)
        m=e1>1e-300
        lr2=np.zeros_like(r2[i])
        e2,U2=np.linalg.eigh((r2[i]+r2[i].conj().T)/2)
        m2=e2>1e-300
        if m2.sum()==0: continue
        lr2=(U2[:,m2]*np.log(e2[m2]))@U2[:,m2].conj().T
        s+= lam*W[i]*( np.sum(e1[m]*np.log(e1[m])) - np.trace(r1[i]@lr2).real )
    return s.real

print("shift c of the boost charge  ->  lambda = e^{beta c};   check  S(lambda)-S(1) = log lambda\n")
print(f"{'state':>7} {'c':>7} {'lambda':>12} {'S(lambda)':>14} {'S(1)':>14} {'S(l)-S(1)':>13} {'log lambda':>13} {'err':>10}")
states=[randstate() for _ in range(4)]
for k,b in enumerate(states):
    S1,_=entropy(b,1.0)
    for c in (0.1,-0.25,1.0):
        lam=np.exp(beta*c); S,_=entropy(b,lam)
        print(f"{k:7d} {c:7.2f} {lam:12.5f} {S:14.8f} {S1:14.8f} {S-S1:13.8f} {np.log(lam):13.8f} {abs(S-S1-np.log(lam)):10.2e}")

print("\nstate-independence of the shift: max spread over states =",
      max(abs((entropy(b,np.exp(beta*0.37))[0]-entropy(b,1.0)[0])-beta*0.37) for b in states))

print("\nexpectation values and relative entropy are untouched:")
obs=[np.array([rng.normal(size=(N,N)) for _ in range(Mx)]) for _ in range(2)]
for lam in (1.0,np.exp(beta*0.5),np.exp(-beta*0.3)):
    _,rh=entropy(states[0],lam)
    ev=sum(lam*W[i]*np.trace(rh[i]@obs[0][i]).real for i in range(Mx))
    print(f"   lambda={lam:10.5f}   <O> = {ev: .12f}   S(rho0||rho1) = {relent(states[0],states[1],lam): .12f}")

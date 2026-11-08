"""Minimal so(3,1)-valued differential forms on R^n (dict of sorted index tuples -> sympy expr)."""
import itertools, sympy as s
eta=[-1,1,1,1]
pairs=list(itertools.combinations(range(4),2))
class F:
    def __init__(self,coords): self.X=coords
    def tidy(self,f): return {i:s.expand(v) for i,v in f.items() if s.expand(v)!=0}
    def add(self,*fs):
        r={}
        for f in fs:
            for i,v in f.items(): r[i]=r.get(i,0)+v
        return self.tidy(r)
    def scale(self,f,c): return self.tidy({i:c*v for i,v in f.items()})
    def wedge(self,a,b):
        r={}
        for I,v in a.items():
            for J,u in b.items():
                K=I+J
                if len(set(K))<len(K): continue
                sg=(-1)**sum(K[i]>K[j] for i in range(len(K)) for j in range(i+1,len(K)))
                key=tuple(sorted(K)); r[key]=r.get(key,0)+sg*v*u
        return self.tidy(r)
    def d(self,a):
        return self.add(*(self.wedge({(i,):1},{I:s.diff(v,x)}) for I,v in a.items() for i,x in enumerate(self.X)))
    def dp(self,a,p): return self.tidy({I:s.diff(v,p) for I,v in a.items()})
    # Lie-algebra valued: dict (i,j)->form for i<j (upper indices, antisymmetric)
    def comp(self,A,i,j):
        if i==j: return {}
        return A[i,j] if i<j else self.scale(A[j,i],-1)
    def P(self,A,B):   # eps_abcd A^ab ^ B^cd
        return self.add(*(self.scale(self.wedge(self.comp(A,i,j),self.comp(B,k,l)),s.LeviCivita(i,j,k,l))
                          for i,j,k,l in itertools.permutations(range(4))))
    def curv(self,A):
        return {(i,j):self.add(self.d(A[i,j]),*(self.scale(self.wedge(self.comp(A,i,k),self.comp(A,k,j)),eta[k]) for k in range(4))) for i,j in pairs}
    def lin(self,*terms):  # sum of c*A
        return {ij:self.add(*(self.scale(A[ij],c) for c,A in terms)) for ij in pairs}
    def comm(self,A,B):  # [A,B]^{ij} = A^i_k ^ B^{kj} - (-1)^{pq} B^i_k ^ A^{kj}  (A,B forms)
        out={}
        for i,j in pairs:
            t1=self.add(*(self.scale(self.wedge(self.comp(A,i,k),self.comp(B,k,j)),eta[k]) for k in range(4)))
            t2=self.add(*(self.scale(self.wedge(self.comp(B,i,k),self.comp(A,k,j)),eta[k]) for k in range(4)))
            deg=lambda f: (len(next(iter(f))) if f else 0)
            pA=max([deg(A[x]) for x in pairs if A[x]]+[0]); pB=max([deg(B[x]) for x in pairs if B[x]]+[0])
            out[i,j]=self.add(t1,self.scale(t2,-(-1)**(pA*pB)))
        return out
    def D(self,lam,A):  # covariant derivative of adjoint 0-form lam
        dl={ij:self.d(lam[ij]) for ij in pairs}
        return self.add_alg(dl,self.comm(A,lam))
    def add_alg(self,*As): return {ij:self.add(*(A[ij] for A in As)) for ij in pairs}

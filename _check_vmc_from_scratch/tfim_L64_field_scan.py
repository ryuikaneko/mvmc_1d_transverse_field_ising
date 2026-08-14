#!/usr/bin/env python3
"""tfim_L64_field_scan.py

Full-scratch fast VMC field scan for the periodic 1D transverse-field Ising model
  H = -J sum_i X_i X_{i+1} - h sum_i Z_i.

For every field this program saves and plots
  * energy per site: finite-L exact Jordan-Wigner and VMC
  * field-direction magnetization per site <Z>: finite-L exact and VMC

The supplied projected-BCS state is the exact even-parity/NS ground state, so
its local-energy variance is zero up to roundoff. Magnetization is diagonal and
has a genuine Monte Carlo error bar.

There is no finite exact saturation field in this model. By default the scan
endpoint is chosen operationally as the smallest h with exact m_z >= 0.99.
Use --h-max to set a fixed endpoint or --saturation-target to change 0.99.

Outputs (under --output-prefix PREFIX):
  PREFIX.csv
  PREFIX_energy.png, PREFIX_energy.pdf
  PREFIX_magnetization.png, PREFIX_magnetization.pdf

Requires: NumPy and Matplotlib. No VMC or Pfaffian package is used.
"""
from __future__ import print_function
import argparse, csv, math, time
import numpy as np


def popcount(x): return bin(int(x)).count("1")
def ns_momenta(L): return 2*np.pi*(np.arange(L,dtype=float)+0.5)/float(L)


def pairing_k(k,g):
    kq=np.asarray(k,dtype=np.longdouble); gq=np.longdouble(g)
    s=np.sin(kq); x=gq-np.cos(kq); e=np.sqrt(x*x+s*s)
    d=e+x; alt=e-x
    out=np.empty(kq.shape,dtype=np.clongdouble)
    use=np.abs(d)>=np.abs(s)
    out[use]=-1j*s[use]/d[use]
    out[~use]=-1j*alt[~use]/s[~use]
    return np.asarray(out,dtype=np.complex128)


def pairing_matrix(L,g):
    k=ns_momenta(L); fk=pairing_k(k,g)
    idx=np.arange(L); delta=idx[None,:]-idx[:,None]
    F=np.sum(np.exp(1j*k[:,None,None]*delta[None,:,:])*fk[:,None,None],axis=0)/float(L)
    F=0.5*(F-F.T); np.fill_diagonal(F,0.0)
    return np.asarray(F,dtype=np.complex128)


def pfaffian(A,tol=1e-13):
    """O(n^3) pivoted antisymmetric Gaussian-elimination Pfaffian."""
    A=np.array(A,dtype=np.complex128,copy=True); n=A.shape[0]
    if A.shape!=(n,n): raise ValueError("Pfaffian input must be square")
    if n%2: return 0j
    if n==0: return 1+0j
    ans=1+0j
    for k in range(0,n-1,2):
        p=k+1+int(np.argmax(np.abs(A[k,k+1:])))
        if abs(A[k,p])<tol: return 0j
        if p!=k+1:
            A[[k+1,p],:]=A[[p,k+1],:]; A[:,[k+1,p]]=A[:,[p,k+1]]; ans=-ans
        pivot=A[k,k+1]; ans*=pivot
        if k+2<n:
            u=A[k,k+2:].copy(); v=A[k+1,k+2:].copy()
            A[k+2:,k+2:] += (np.outer(v,u)-np.outer(u,v))/pivot
            B=A[k+2:,k+2:]; A[k+2:,k+2:]=0.5*(B-B.T)
    return ans


def permutation_sign_to_sorted(seq):
    inv=sum(seq[i]>seq[j] for i in range(len(seq)) for j in range(i+1,len(seq)))
    return -1.0 if inv%2 else 1.0


class FastProjectedBCS(object):
    def __init__(self,L,J,h,start_state=0):
        if L<2 or L%2: raise ValueError("L must be even and >= 2")
        if J<=0 or h<0: raise ValueError("Use J>0 and h>=0")
        self.L=int(L); self.J=float(J); self.h=float(h); self.F=pairing_matrix(L,h/J)
        self.set_state(start_state)
    def set_state(self,state):
        state=int(state)
        if state<0 or state>=(1<<self.L) or popcount(state)%2:
            raise ValueError("start state must be an even-parity L-bit integer")
        self.state=state; self.occ=[i for i in range(self.L) if (state>>i)&1]; self._rebuild()
    def _rebuild(self):
        if not self.occ:
            self.A=np.empty((0,0),complex); self.Ainv=np.empty((0,0),complex); self.amp=1+0j; return
        self.A=self.F[np.ix_(self.occ,self.occ)]; self.amp=pfaffian(self.A)
        if abs(self.amp)<1e-14: raise FloatingPointError("zero/singular accepted amplitude")
        self.Ainv=np.linalg.inv(self.A); self.Ainv=0.5*(self.Ainv-self.Ainv.T)
    def flip_state(self,i):
        j=(i+1)%self.L; return self.state^(1<<i)^(1<<j)
    def ratio_bond(self,i):
        p=i; q=(i+1)%self.L; pin=(self.state>>p)&1; qin=(self.state>>q)&1; n=len(self.occ)
        if not pin and not qin:
            if n==0: raw=self.F[p,q]
            else:
                b=self.F[np.ix_(self.occ,[p])][:,0]; c=self.F[np.ix_(self.occ,[q])][:,0]
                raw=self.F[p,q]+np.dot(b,np.dot(self.Ainv,c))
            return permutation_sign_to_sorted(self.occ+[p,q])*raw
        if pin and qin:
            a=self.occ.index(p); b=self.occ.index(q)
            if a>b: a,b=b,a
            return ((-1.0)**(a+b))*self.Ainv[a,b]
        r=p if pin else q; s=q if pin else p; a=self.occ.index(r)
        v=self.F[s,self.occ].copy(); v[a]=0.0
        raw=np.dot(v,self.Ainv[:,a]); new=list(self.occ); new[a]=s
        return permutation_sign_to_sorted(new)*raw
    def accept_bond(self,i):
        self.state=self.flip_state(i); self.occ=[x for x in range(self.L) if (self.state>>x)&1]; self._rebuild()
    def local_energy(self):
        out=-self.h*(self.L-2*len(self.occ))+0j
        for i in range(self.L): out += -self.J*self.ratio_bond(i)
        return out
    def magnetization_z(self): return float(self.L-2*len(self.occ))/self.L


def exact_observables(L,J,h):
    """Finite-L NS energy/site and field-axis magnetization/site."""
    k=ns_momenta(L); g=h/J
    d=np.sqrt(1+g*g-2*g*np.cos(k))
    energy=-J*np.sum(d)
    mz=np.sum((g-np.cos(k))/d)/float(L)
    return float(energy/L),float(mz)


def find_operational_hmax(L,J,target):
    if not (0<target<1): raise ValueError("saturation target must satisfy 0<target<1")
    lo=0.0; hi=J
    while exact_observables(L,J,hi)[1]<target: hi*=2.0
    for unused in range(80):
        mid=0.5*(lo+hi)
        if exact_observables(L,J,mid)[1]>=target: hi=mid
        else: lo=mid
    return hi


def block_mean_error(values,block_size):
    x=np.asarray(values,dtype=float); nb=len(x)//block_size
    if nb<2: raise ValueError("need at least two complete blocks")
    b=x[:nb*block_size].reshape(nb,block_size).mean(axis=1)
    return float(b.mean()),float(b.std(ddof=1)/math.sqrt(nb)),nb


def sample_one_field(L,J,h,measurements,burn_in,thin,block_size,seed):
    wf=FastProjectedBCS(L,J,h,0); rng=np.random.RandomState(seed)
    energies=[]; mags=[]; accepted=0; proposals=0
    total=burn_in+measurements*thin; t0=time.time()
    for sweep in range(total):
        for unused in range(L):
            i=int(rng.randint(0,L)); ratio=wf.ratio_bond(i); prob=abs(ratio)**2; proposals+=1
            if prob>=1.0 or rng.random_sample()<prob:
                wf.accept_bond(i); accepted+=1
        if sweep>=burn_in and (sweep-burn_in)%thin==0:
            energies.append(wf.local_energy().real/L); mags.append(wf.magnetization_z())
    ee,dee,nb=block_mean_error(energies,block_size)
    mm,dmm,nb2=block_mean_error(mags,block_size)
    ex_e,ex_m=exact_observables(L,J,h)
    return dict(h=h,exact_energy=ex_e,vmc_energy=ee,vmc_energy_error=dee,
                exact_magnetization=ex_m,vmc_magnetization=mm,vmc_magnetization_error=dmm,
                acceptance=float(accepted)/proposals,blocks=nb,
                max_abs_local_energy_error=float(np.max(np.abs(np.asarray(energies)-ex_e))),
                elapsed_seconds=time.time()-t0)


def save_csv(rows,path,L,J,measurements,burn_in,thin,block_size):
    fields=['h','h_over_J','exact_energy_per_site','vmc_energy_per_site','vmc_energy_error',
            'exact_magnetization_z','vmc_magnetization_z','vmc_magnetization_error',
            'acceptance','max_abs_local_energy_error','elapsed_seconds']
    with open(path,'w',newline='') as f:
        f.write('# periodic 1D TFIM, L=%d, J=%.17g\n'%(L,J))
        f.write('# measurements=%d, burn_in=%d, thin=%d, block_size=%d\n'%(measurements,burn_in,thin,block_size))
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
        for r in rows:
            w.writerow(dict(h='%.17g'%r['h'],h_over_J='%.17g'%(r['h']/J),
                exact_energy_per_site='%.17g'%r['exact_energy'],vmc_energy_per_site='%.17g'%r['vmc_energy'],
                vmc_energy_error='%.17g'%r['vmc_energy_error'],exact_magnetization_z='%.17g'%r['exact_magnetization'],
                vmc_magnetization_z='%.17g'%r['vmc_magnetization'],vmc_magnetization_error='%.17g'%r['vmc_magnetization_error'],
                acceptance='%.17g'%r['acceptance'],max_abs_local_energy_error='%.17g'%r['max_abs_local_energy_error'],
                elapsed_seconds='%.17g'%r['elapsed_seconds']))


def make_plots(rows,prefix,J):
    import matplotlib.pyplot as plt
    h=np.array([r['h']/J for r in rows]); ee=np.array([r['exact_energy'] for r in rows])
    ev=np.array([r['vmc_energy'] for r in rows]); de=np.array([r['vmc_energy_error'] for r in rows])
    me=np.array([r['exact_magnetization'] for r in rows]); mv=np.array([r['vmc_magnetization'] for r in rows])
    dm=np.array([r['vmc_magnetization_error'] for r in rows])
    fig,ax=plt.subplots(); ax.plot(h,ee,label='Exact (finite-L JW/NS)')
    ax.errorbar(h,ev,yerr=de,fmt='o',ms=3,capsize=2,label='VMC')
    ax.set_xlabel(r'$h/J$'); ax.set_ylabel(r'$E/L$'); ax.legend(); fig.tight_layout()
    fig.savefig(prefix+'_energy.png',dpi=200); fig.savefig(prefix+'_energy.pdf'); plt.close(fig)
    fig,ax=plt.subplots(); ax.plot(h,me,label='Exact (finite-L JW/NS)')
    ax.errorbar(h,mv,yerr=dm,fmt='o',ms=3,capsize=2,label='VMC')
    ax.set_xlabel(r'$h/J$'); ax.set_ylabel(r'$m_z=\langle\sum_i\sigma_i^z\rangle/L$')
    ax.set_ylim(-0.03,1.03); ax.legend(); fig.tight_layout()
    fig.savefig(prefix+'_magnetization.png',dpi=200); fig.savefig(prefix+'_magnetization.pdf'); plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--L',type=int,default=64); p.add_argument('--J',type=float,default=1.0)
    p.add_argument('--h-min',type=float,default=0.0); p.add_argument('--h-max',type=float,default=None)
    p.add_argument('--num-fields',type=int,default=31); p.add_argument('--saturation-target',type=float,default=0.99)
    p.add_argument('--measurements',type=int,default=1000); p.add_argument('--burn-in',type=int,default=300)
    p.add_argument('--thin',type=int,default=1); p.add_argument('--block-size',type=int,default=50)
    p.add_argument('--seed',type=int,default=12345); p.add_argument('--output-prefix',default='tfim_L64_scan')
    a=p.parse_args()
    if a.L%2 or a.L<2: p.error('L must be even and >=2')
    if a.h_min<0: p.error('h-min must be nonnegative')
    if a.measurements<2*a.block_size: p.error('measurements must contain at least two blocks')
    hmax=a.h_max if a.h_max is not None else find_operational_hmax(a.L,a.J,a.saturation_target)
    if hmax<=a.h_min: p.error('h-max must exceed h-min')
    hs=np.linspace(a.h_min,hmax,a.num_fields); rows=[]
    print('L=%d; field endpoint h/J=%.10g (operational m_z target %.6g)'%(a.L,hmax/a.J,a.saturation_target))
    for n,h in enumerate(hs):
        r=sample_one_field(a.L,a.J,float(h),a.measurements,a.burn_in,a.thin,a.block_size,a.seed+n)
        rows.append(r)
        print('%3d/%3d h/J=%7.4f E_vmc=%+.10f E_ex=%+.10f mz_vmc=%.7f mz_ex=%.7f acc=%.3f'%
              (n+1,len(hs),h/a.J,r['vmc_energy'],r['exact_energy'],r['vmc_magnetization'],r['exact_magnetization'],r['acceptance']))
    save_csv(rows,a.output_prefix+'.csv',a.L,a.J,a.measurements,a.burn_in,a.thin,a.block_size)
    make_plots(rows,a.output_prefix,a.J)
    print('saved:',a.output_prefix+'.csv',a.output_prefix+'_energy.[png,pdf]',a.output_prefix+'_magnetization.[png,pdf]')

if __name__=='__main__': main()

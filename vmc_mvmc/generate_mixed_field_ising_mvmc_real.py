#!/usr/bin/env python3
"""Generate all mVMC definition files for
H=-J sum_i X_i X_(i+1)-hx sum_i X_i-hz sum_i Z_i (periodic chain).

Real, distance-tied OrbitalGeneral convention:
  p = p(si,sj,min(|j-i|,L-|j-i|))
  F(i,si;j,sj) = sign(i,j) * f[p]
  sign(i,j) = -1 for j<i, +1 otherwise.
The OrbitalGeneral mapping rows are written as: i si j sj p sign.
NMPTrans=-1 activates the mVMC anti-periodic-condition mode.
"""
from __future__ import annotations
import argparse
import subprocess
from pathlib import Path
import numpy as np

SEP = "============================================="


def phi_real_space_hx0(L, J, hz):
    """Exact NS-sector TFIM antisymmetric pair matrix at hx=0."""
    g = hz / J
    k = 2*np.pi*(np.arange(L)+0.5)/L
    eps = np.sqrt(1+g*g-2*g*np.cos(k))
    phi_k = -1j*(g-np.cos(k)+eps)/np.sin(k)
    phi = np.empty((L,L), dtype=np.complex128)
    for i in range(L):
        for j in range(L):
            phi[i,j] = np.sum(np.exp(1j*k*(j-i))*phi_k)/L
    phi = 0.5*(phi-phi.T)
    phi[np.abs(phi)<5e-15] = 0
    if np.max(np.abs(phi+phi.T)) > 1e-12:
        raise RuntimeError("phi is not antisymmetric")
    max_imag = float(np.max(np.abs(phi.imag)))
    if max_imag > 1e-12:
        raise RuntimeError(f"phi is not real: max |Im(phi)|={max_imag:.3e}")
    return phi.real.astype(np.float64)


def s_value(i,j):
    return 1.0 if i<j else (-1.0 if i>j else 0.0)


def independent_pairs(L):
    rows=[]
    for a in range(2*L):
        i,si=a%L,a//L
        for b in range(a+1,2*L):
            j,sj=b%L,b//L
            rows.append((i,si,j,sj,a,b))
    if len(rows) != 2*L*L-L:
        raise RuntimeError("wrong independent-pair count")
    return rows


def ring_distance(i,j,L):
    d = abs(j-i)
    return min(d, L-d)

def orbital_key(i,si,j,sj,L):
    return si,sj,ring_distance(i,j,L)


def orbital_sign(i,j):
    return -1 if j<i else 1


def build_classes(L,pairs):
    keys=sorted({orbital_key(i,si,j,sj,L) for i,si,j,sj,a,b in pairs})
    key_to_p={key:p for p,key in enumerate(keys)}
    pair_to_p={}
    pair_to_sign={}
    for i,si,j,sj,a,b in pairs:
        pair_to_p[(a,b)] = key_to_p[orbital_key(i,si,j,sj,L)]
        pair_to_sign[(a,b)] = orbital_sign(i,j)
    return pair_to_p,pair_to_sign,key_to_p,keys


def tfim_target(i,si,j,sj,phi):
    if si==0 and sj==0:
        return float(phi[i,j])
    if si==1 and sj==1:
        return s_value(i,j)
    return 0j


def product_target(i,si,j,sj,u,v):
    return s_value(i,j)*(u,v)[si]*(u,v)[sj]


def signed_class_initial(L,pairs,pair_to_p,pair_to_sign,p_to_key,phi,
                         kind,lam,u,v,tol=1e-11):
    values=[None]*len(p_to_key)
    reps=[None]*len(p_to_key)
    raw=[]
    maxres=np.zeros(len(p_to_key))
    for i,si,j,sj,a,b in pairs:
        z0=tfim_target(i,si,j,sj,phi)
        zp=product_target(i,si,j,sj,u,v)
        target=z0 if kind=="tfim" else (zp if kind=="product" else (1-lam)*z0+lam*zp)
        p=pair_to_p[(a,b)]
        sign=pair_to_sign[(a,b)]
        canonical=sign*target
        if values[p] is None:
            values[p]=canonical
            reps[p]=(i,si,j,sj)
        else:
            r=abs(canonical-values[p])
            maxres[p]=max(maxres[p],r)
            if r>tol:
                raise RuntimeError(f"signed-class inconsistency p={p} key={p_to_key[p]} rep={reps[p]} row={(i,si,j,sj)} residual={r:.3e}")
        raw.append((i,si,j,sj,a,b,p,sign,target,canonical))
    if any(z is None for z in values):
        raise RuntimeError("uninitialized class")
    return np.asarray(values),maxres,raw


def validate(L,pairs,pair_to_p,pair_to_sign,p_to_key):
    seen=set(); used=set()
    for i,si,j,sj,a,b in pairs:
        if a!=i+si*L or b!=j+sj*L or not a<b:
            raise RuntimeError("bad combined-index mapping")
        if (a,b) in seen:
            raise RuntimeError("duplicate pair")
        seen.add((a,b))
        p=pair_to_p[(a,b)]
        if p_to_key[p] != orbital_key(i,si,j,sj,L):
            raise RuntimeError("bad class mapping")
        if pair_to_sign[(a,b)] != orbital_sign(i,j):
            raise RuntimeError("bad sign mapping")
        used.add(p)
    if used != set(range(len(p_to_key))):
        raise RuntimeError("unused class")


def header5(fp,key,n):
    fp.write(f"{SEP}\n{key} {n}\n{SEP}\n{SEP}\n{SEP}\n")


def exact_ns_energy(L,J,hz):
    k=2*np.pi*(np.arange(L)+0.5)/L
    eps=np.sqrt(1+(hz/J)**2-2*(hz/J)*np.cos(k))
    return float(-J*np.sum(eps))


def generate(a):
    L,J,hx,hz=a.L,a.J,a.hx,a.hz
    if L<=0 or L%2: raise ValueError("L must be a positive even integer")
    if J<=0: raise ValueError("J must be positive")
    if not 0<=a.mix<=1: raise ValueError("mix must be in [0,1]")
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    phi=phi_real_space_hx0(L,J,hz)
    pairs=independent_pairs(L)
    pair_to_p,pair_to_sign,key_to_p,p_to_key=build_classes(L,pairs)
    validate(L,pairs,pair_to_p,pair_to_sign,p_to_key)
    # Ground-state spinor of the one-site field -hx X - hz Z.
    # This gives <X>=hx/sqrt(hx^2+hz^2) and <Z>=hz/sqrt(hx^2+hz^2).
    theta=np.arctan2(hx,hz)
    u=float(np.cos(theta/2)); v=float(np.sin(theta/2))
    field_norm=float(np.hypot(hx,hz))
    x_product=2.0*u*v
    z_product=u*u-v*v
    if field_norm > 0.0:
        expected_x=hx/field_norm
        expected_z=hz/field_norm
        if abs(x_product-expected_x) > 1e-12:
            raise RuntimeError(
                f"product-state X direction mismatch: <X>={x_product}, expected={expected_x}"
            )
        if abs(z_product-expected_z) > 1e-12:
            raise RuntimeError(
                f"product-state Z direction mismatch: <Z>={z_product}, expected={expected_z}"
            )
    lam=0.0 if a.initial=="tfim" else (1.0 if a.initial=="product" else a.mix)
    initial,maxres,raw=signed_class_initial(L,pairs,pair_to_p,pair_to_sign,
        p_to_key,phi,a.initial,lam,u,v)
    npar=len(p_to_key)

    (out/"namelist.def").write_text(
      "ModPara modpara.def\nLocSpin locspn.def\nTrans trans.def\n"
      "InterAll interall.def\nOrbitalGeneral orbitalgeneralidx.def\n"
      "InOrbitalGeneral inorbitalgeneral.def\nTransSym qptransidx.def\n",encoding="ascii")
    # Post-optimization expectation-value calculation using zqp_orbital_general_opt.dat.
    (out/"namelist_aft.def").write_text(
      "ModPara modpara_aft.def\nLocSpin locspn.def\nTrans trans.def\n"
      "InterAll interall.def\nOrbitalGeneral orbitalgeneralidx.def\n"
      "InOrbitalGeneral ./output/zqp_orbital_general_opt.dat\nTransSym qptransidx.def\n",encoding="ascii")
    (out/"modpara.def").write_text(f"""--------------------
Model_Parameters 0
--------------------
VMC_Cal_Parameters
--------------------
CDataFileHead zvo
CParaFileHead zqp
--------------------
NVMCCalMode {a.mode}
NLanczosMode 0
--------------------
NDataIdxStart 1
NDataQtySmp 1
--------------------
Nsite {L}
Ncond 0
NMPTrans -1
NSROptItrStep {a.sr_steps}
NSROptItrSmp {a.sr_average}
DSROptRedCut 0.000001
DSROptStaDel 0.02
DSROptStepDt 0.02
NVMCWarmUp {a.warmup}
NVMCInterval {a.interval}
NVMCSample {a.samples}
NExUpdatePath 2
RndSeed {a.seed}
NSplitSize 1
NStore 1
NSRCG {a.srcg}
""",encoding="ascii")
    (out/"modpara_aft.def").write_text(f"""--------------------
Model_Parameters 0
--------------------
VMC_Cal_Parameters
--------------------
CDataFileHead zvo_aft
CParaFileHead zqp_aft
--------------------
NVMCCalMode 1
NLanczosMode 0
--------------------
NDataIdxStart 1
NDataQtySmp 10
--------------------
Nsite {L}
Ncond 0
NMPTrans -1
NSROptItrStep {a.sr_steps}
NSROptItrSmp {a.sr_average}
DSROptRedCut 0.000001
DSROptStaDel 0.02
DSROptStepDt 0.02
NVMCWarmUp {a.warmup}
NVMCInterval {a.interval}
NVMCSample 1000
NExUpdatePath 2
RndSeed {a.seed}
NSplitSize 1
NStore 1
NSRCG {a.srcg}
""",encoding="ascii")
    with (out/"locspn.def").open("w",encoding="ascii") as fp:
        header5(fp,"NlocalSpin",L)
        for i in range(L): fp.write(f"{i} 1\n")
    with (out/"trans.def").open("w",encoding="ascii") as fp:
        header5(fp,"NTransfer",4*L)
        for i in range(L):
            fp.write(f"{i} 0 {i} 0 {hz:.17g} 0\n")
            fp.write(f"{i} 1 {i} 1 {-hz:.17g} 0\n")
            # mVMC uses H_Trans = -sum(t * c^dagger c), so +hx here gives -hx X_i.
            fp.write(f"{i} 0 {i} 1 {hx:.17g} 0\n")
            fp.write(f"{i} 1 {i} 0 {hx:.17g} 0\n")
    terms=[]
    for i in range(L):
        j=(i+1)%L
        for si in (0,1):
            for sj in (0,1):
                terms.append((i,1-si,i,si,j,1-sj,j,sj,-J,0.0))
    with (out/"interall.def").open("w",encoding="ascii") as fp:
        header5(fp,"NInterAll",len(terms))
        for x in terms: fp.write("%d %d %d %d %d %d %d %d %.17g %.17g\n"%x)
    with (out/"orbitalgeneralidx.def").open("w",encoding="ascii") as fp:
        fp.write(f"{SEP}\nNOrbitalIdx          {npar}\nComplexType 0\n{SEP}\n{SEP}\n")
        for i,si,j,sj,aidx,bidx in pairs:
            fp.write(f"{i} {si} {j} {sj} {pair_to_p[(aidx,bidx)]} {pair_to_sign[(aidx,bidx)]}\n")
        for p in range(npar): fp.write(f"{p} {a.orbital_opt}\n")
    with (out/"inorbitalgeneral.def").open("w",encoding="ascii") as fp:
        fp.write(f"{SEP}\nNOrbitalIdx  {npar}\n{SEP}\n{SEP}\n{SEP}\n")
        for p,z in enumerate(initial): fp.write(f"{p} {z.real:.17g} {z.imag:.17g}\n")
    with (out/"qptransidx.def").open("w",encoding="ascii") as fp:
        fp.write(f"{SEP}\nNQPTrans          1\n{SEP}\n")
        fp.write("======== TrIdx_TrWeight_and_TrIdx_i_xi_Phase ======\n")
        fp.write(f"{SEP}\n     0   1.000000000000   0.000000000000\n")
        for i in range(L): fp.write(f"{0:6d} {i:7d} {i:7d} {1.0:16.12f} {0.0:16.12f}\n")

    np.save(out/"phi_matrix_hx0.npy",phi)
    np.savetxt(out/"phi_matrix_hx0.txt",phi)
    with (out/"orbitalgeneral_class_table.txt").open("w",encoding="ascii") as fp:
        fp.write("# p si sj distance Re(initial) Im(initial) max_signed_residual\n")
        for p,(si,sj,d) in enumerate(p_to_key):
            z=initial[p]; fp.write(f"{p} {si} {sj} {d} {z.real:.17g} {z.imag:.17g} {maxres[p]:.17g}\n")
    with (out/"orbitalgeneral_mapping_check.txt").open("w",encoding="ascii") as fp:
        fp.write("# i si j sj a b distance p sign Re(class) Im(class) Re(target) Im(target) Re(sign_class) Im(sign_class) residual\n")
        for i,si,j,sj,aidx,bidx,p,sign,target,canonical in raw:
            z=initial[p]; represented=sign*z
            fp.write(f"{i} {si} {j} {sj} {aidx} {bidx} {ring_distance(i,j,L)} {p} {sign} {z.real:.17g} {z.imag:.17g} {target.real:.17g} {target.imag:.17g} {represented.real:.17g} {represented.imag:.17g} {abs(target-represented):.17g}\n")
    e0=exact_ns_energy(L,J,hz)
    (out/"README.txt").write_text(f"""Periodic mixed-field Ising chain
H = -J sum_i X_i X_(i+1) - hx sum_i X_i - hz sum_i Z_i
L={L}, J={J:.17g}, hx={hx:.17g}, hz={hz:.17g}
Real distance tying: F(i,si;j,sj)=sign*f[si,sj,min(|j-i|,L-|j-i|)]
sign=-1 for j<i and +1 otherwise
ComplexType=0; all initial orbital parameters are real
OrbitalGeneral mapping row format: i si j sj p sign
NMPTrans=-1
NOrbitalIdx={npar}; mapping rows={len(pairs)}
initial={a.initial}; lambda={lam:.17g}; theta={theta:.17g}; u={u:.17g}; v={v:.17g}
product-state expectations: <X>={x_product:.17g}; <Z>={z_product:.17g}
field-direction targets: X={hx/field_norm if field_norm else 0.0:.17g}; Z={hz/field_norm if field_norm else 0.0:.17g}
maximum signed-class residual={float(np.max(maxres)):.17g}
Exact NS energy only at hx=0: E={e0:.16g}, E/L={e0/L:.16g}
""",encoding="ascii")
    return out


def main():
    p=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--L",type=int,default=24)
    p.add_argument("--J",type=float,default=1.0)
    p.add_argument("--hx",type=float,default=0.2)
    p.add_argument("--hz",type=float,default=1.0)
    p.add_argument("-o","--out",default=None,
                   help="output directory; default: dat_vmc_L[L]_J[J]_hz[hz]_hx[hx]")
    p.add_argument("--initial",choices=("tfim","interpolate","product"),default="interpolate")
    p.add_argument("--mix",type=float,default=None)
    p.add_argument("--seed",type=int,default=271828)
    p.add_argument("--orbital-opt",type=int,choices=(0,1),default=1)
    p.add_argument("--mode",type=int,choices=(0,1),default=0)
    p.add_argument("--warmup",type=int,default=1000)
    p.add_argument("--interval",type=int,default=1)
    p.add_argument("--samples",type=int,default=500)
    p.add_argument("--sr-steps",type=int,default=500)
    p.add_argument("--sr-average",type=int,default=100)
    p.add_argument("--srcg",type=int,choices=(0,1),default=1)
    p.add_argument("--run",action="store_true")
    p.add_argument("--vmc-command",default="vmc.out")
    a=p.parse_args()
    if a.out is None:
#        a.out = f"dat_vmc_L{a.L}_J{a.J:g}_hz{a.hz:g}_hx{a.hx:g}"
        a.out = f"dat_vmc_L{a.L}_J{a.J:g}_hz{a.hz:g}_hx{a.hx:.1f}"
    if a.mix is None:
        den=abs(a.hx)+a.J+abs(a.hz)
        a.mix=abs(a.hx)/den if den else 0.0
    out=generate(a)
    print(f"Wrote all files to {out}")
    if a.run:
        with (out/"vmc_console.log").open("w",encoding="utf-8") as fp:
            cp=subprocess.run([a.vmc_command,"namelist.def"],cwd=out,stdout=fp,stderr=subprocess.STDOUT,check=False)
        print(f"mVMC optimization exit code: {cp.returncode}")
        if cp.returncode:
            raise SystemExit(cp.returncode)
        with (out/"vmc_console_aft.log").open("w",encoding="utf-8") as fp:
            cp_aft=subprocess.run([a.vmc_command,"namelist_aft.def"],cwd=out,stdout=fp,stderr=subprocess.STDOUT,check=False)
        print(f"mVMC expectation-value exit code: {cp_aft.returncode}")
        if cp_aft.returncode:
            raise SystemExit(cp_aft.returncode)

if __name__=="__main__": main()

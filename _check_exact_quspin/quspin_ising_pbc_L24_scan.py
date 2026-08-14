#!/usr/bin/env python3
"""Ground-state energy of the mixed-field Ising chain with PBC using QuSpin.

H = -J sum_i X_i X_{i+1} - hx sum_i X_i - hz sum_i Z_i
Defaults: L=24, J=hz=1, hx=0,...,2 in steps of 0.02.

The calculation uses the k=0, reflection-even (p=+1) sector and ARPACK Lanczos.
Outputs:
  ising_pbc_L24_E0.csv
  ising_pbc_L24_runinfo.txt
"""

import argparse
import gc
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import scipy
from scipy.sparse.linalg import eigsh
from quspin.basis import spin_basis_1d
from quspin.operators import hamiltonian


def rss_gib():
    """Current resident-set size in GiB."""
    return psutil.Process(os.getpid()).memory_info().rss / 2**30


def build_operators(L, J, hz, dtype=np.float64):
    # For J>0, hx>=0, hz arbitrary, the finite-size GS is in k=0, p=+1.
    # pauli=1 means x,z are Pauli matrices, not spin operators S=sigma/2.
    basis = spin_basis_1d(
        L=L, S="1/2", pauli=1, a=1, kblock=0, pblock=1
    )

    xx = [[-J, i, (i + 1) % L] for i in range(L)]
    zf = [[-hz, i] for i in range(L)]
    xf = [[-1.0, i] for i in range(L)]  # multiply this operator by hx

    common = dict(
        basis=basis,
        dtype=dtype,
        check_herm=False,
        check_symm=False,
        check_pcon=False,
    )
    H0_qs = hamiltonian([["xx", xx], ["z", zf]], [], **common)
    Hx_qs = hamiltonian([["x", xf]], [], **common)

    # Work directly with SciPy CSR matrices during the scan.
    H0 = H0_qs.tocsr()
    Hx = Hx_qs.tocsr()
    del H0_qs, Hx_qs
    gc.collect()
    return basis, H0, Hx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--L", type=int, default=24)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--hz", type=float, default=1.0)
    parser.add_argument("--hx-min", type=float, default=0.0)
    parser.add_argument("--hx-max", type=float, default=2.0)
    parser.add_argument("--dhx", type=float, default=0.02)
    parser.add_argument("--tol", type=float, default=1e-12)
    parser.add_argument("--maxiter", type=int, default=200000)
    parser.add_argument("--ncv", type=int, default=20)
    parser.add_argument("--output", default="ising_pbc_L24_E0.csv")
    args = parser.parse_args()

    if args.L < 2:
        raise ValueError("L must be at least 2.")
    if args.J <= 0:
        raise ValueError("This k=0,p=+1-only script assumes ferromagnetic J>0.")
    if args.hx_min < 0:
        raise ValueError("Use hx>=0; the spectrum is even in hx.")
    if args.dhx <= 0 or args.hx_max < args.hx_min:
        raise ValueError("Require dhx>0 and hx_max>=hx_min.")

    nstep = int(round((args.hx_max - args.hx_min) / args.dhx))
    hx_values = args.hx_min + args.dhx * np.arange(nstep + 1)
    if not np.isclose(hx_values[-1], args.hx_max, atol=1e-13, rtol=0):
        raise ValueError("(hx_max-hx_min)/dhx must be an integer.")
    hx_values[-1] = args.hx_max

    rss_start = rss_gib()
    t_build = time.perf_counter()
    basis, H0, Hx = build_operators(args.L, args.J, args.hz)
    build_seconds = time.perf_counter() - t_build
    rss_after_build = rss_gib()

    print(f"L={args.L}, basis Ns={basis.Ns:,}")
    print(f"H0: shape={H0.shape}, nnz={H0.nnz:,}, storage={sum(x.nbytes for x in (H0.data,H0.indices,H0.indptr))/2**30:.3f} GiB")
    print(f"Hx: shape={Hx.shape}, nnz={Hx.nnz:,}, storage={sum(x.nbytes for x in (Hx.data,Hx.indices,Hx.indptr))/2**30:.3f} GiB")
    print(f"RSS after construction: {rss_after_build:.3f} GiB")

    rows = []
    v0 = None
    peak_rss = rss_after_build

    for ih, hx in enumerate(hx_values):
        t0 = time.perf_counter()
        H = H0 + float(hx) * Hx

        evals, evecs = eigsh(
            H,
            k=1,
            which="SA",
            v0=v0,
            tol=args.tol,
            maxiter=args.maxiter,
            ncv=args.ncv,
            return_eigenvectors=True,
        )
        E0 = float(evals[0])
        psi = evecs[:, 0]

        # A direct posterior check, more informative than tol alone.
        residual = np.linalg.norm(H @ psi - E0 * psi)
        elapsed = time.perf_counter() - t0
        current_rss = rss_gib()
        peak_rss = max(peak_rss, current_rss)

        rows.append((hx, E0, E0 / args.L, residual, elapsed, current_rss))
        print(
            f"{ih+1:3d}/{len(hx_values)}  hx={hx:7.4f}  "
            f"E0={E0: .15f}  E0/L={E0/args.L: .15f}  "
            f"res={residual:.3e}  RSS={current_rss:.3f} GiB"
        )

        # Warm start at the next field value.
        v0 = psi.copy()
        del H, evecs, psi
        gc.collect()

    out = Path(args.output)
    data = np.asarray(rows, dtype=float)
    np.savetxt(
        out,
        data,
        delimiter=",",
        header="hx,E0,E0_per_site,residual_norm,elapsed_seconds,rss_GiB",
        comments="",
        fmt=["%.15g", "%.17g", "%.17g", "%.8e", "%.6f", "%.6f"],
    )

    info = out.with_name(out.stem + "_runinfo.txt")
    info.write_text(
        "\n".join([
            f"Hamiltonian: -J sum XX - hx sum X - hz sum Z, PBC",
            f"L={args.L}",
            f"J={args.J:.17g}",
            f"hz={args.hz:.17g}",
            f"hx_min={args.hx_min:.17g}",
            f"hx_max={args.hx_max:.17g}",
            f"dhx={args.dhx:.17g}",
            f"number_of_points={len(hx_values)}",
            f"sector=k0,p+1",
            f"basis_Ns={basis.Ns}",
            f"H0_nnz={H0.nnz}",
            f"Hx_nnz={Hx.nnz}",
            f"rss_start_GiB={rss_start:.6f}",
            f"rss_after_build_GiB={rss_after_build:.6f}",
            f"observed_peak_rss_GiB={peak_rss:.6f}",
            f"build_seconds={build_seconds:.6f}",
            f"python={sys.version.split()[0]}",
            f"numpy={np.__version__}",
            f"scipy={scipy.__version__}",
            f"platform={platform.platform()}",
        ]) + "\n",
        encoding="utf-8",
    )
    print(f"\nWrote: {out.resolve()}")
    print(f"Wrote: {info.resolve()}")
    print(f"Observed peak RSS: {peak_rss:.3f} GiB")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Collect mVMC energy and energy-variance data for each hx.

Expected directory layout:
  ./dat_vmc_L64_J1_hz1_hx[hx]/output/zvo_aft_out_*.dat

The first and third numerical columns (0-based indices 0 and 2) are read as
<H> and <H^2>, respectively.  Each file is treated as one independent bin;
if a file has multiple valid numerical rows, their column-wise mean is used.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np

DIR_RE = re.compile(r"^dat_vmc_L64_J1_hz1_hx(.+)$")


def parse_float(text: str) -> float:
    """Read ordinary or Fortran-style floating-point notation."""
    return float(text.replace("D", "E").replace("d", "e"))


def read_one_file(path: Path, energy_col: int, h2_col: int) -> tuple[float, float]:
    rows: list[tuple[float, float]] = []
    needed = max(energy_col, h2_col)

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) <= needed:
                print(
                    f"warning: skip {path}:{line_no}: only {len(fields)} columns",
                    file=sys.stderr,
                )
                continue
            try:
                energy = parse_float(fields[energy_col])
                h2 = parse_float(fields[h2_col])
            except ValueError:
                print(
                    f"warning: skip {path}:{line_no}: non-numeric target column",
                    file=sys.stderr,
                )
                continue
            if math.isfinite(energy) and math.isfinite(h2):
                rows.append((energy, h2))
            else:
                print(
                    f"warning: skip {path}:{line_no}: NaN or Inf",
                    file=sys.stderr,
                )

    if not rows:
        raise ValueError(f"no usable numerical row in {path}")

    values = np.asarray(rows, dtype=float)
    # One output file is one independent bin.  Multiple rows in a file are
    # averaged first, so files with more rows do not receive extra weight.
    return float(values[:, 0].mean()), float(values[:, 1].mean())


def sem(x: np.ndarray) -> float:
    """Unbiased sample standard error of the mean."""
    n = x.size
    return float(x.std(ddof=1) / np.sqrt(n)) if n >= 2 else float("nan")


def variance_and_jackknife_error(
    energy: np.ndarray, h2: np.ndarray
) -> tuple[float, float]:
    """Return <H^2>-<H>^2 and its delete-one-bin jackknife error.

    The jackknife keeps the covariance between <H> and <H^2>, unlike an
    independent-error propagation formula.
    """
    n = energy.size
    value = float(h2.mean() - energy.mean() ** 2)
    if n < 2:
        return value, float("nan")

    sum_e = float(energy.sum())
    sum_h2 = float(h2.sum())
    e_loo = (sum_e - energy) / (n - 1)
    h2_loo = (sum_h2 - h2) / (n - 1)
    theta_loo = h2_loo - e_loo**2
    theta_bar = float(theta_loo.mean())
    error = math.sqrt((n - 1) / n * float(np.sum((theta_loo - theta_bar) ** 2)))
    return value, error


def hx_from_dir(path: Path) -> tuple[float, str]:
    match = DIR_RE.match(path.name)
    if match is None:
        raise ValueError(f"unexpected directory name: {path.name}")
    hx_text = match.group(1)
    try:
        hx_value = parse_float(hx_text)
    except ValueError as exc:
        raise ValueError(f"cannot parse hx from directory {path.name}") from exc
    return hx_value, hx_text


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize <H> and <H^2>-<H>^2 for every mVMC hx directory."
    )
    parser.add_argument(
        "--root", type=Path, default=Path("."),
        help="parent directory containing dat_vmc_L64_J1_hz1_hx* (default: .)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("dat_hx_ene_err_var_err"),
        help="output file (default: dat_hx_ene_err_var_err)",
    )
    parser.add_argument(
        "--energy-col", type=int, default=0,
        help="zero-based column index of <H> (default: 0)",
    )
    parser.add_argument(
        "--h2-col", type=int, default=2,
        help="zero-based column index of <H^2> (default: 2)",
    )
    args = parser.parse_args()

    if args.energy_col < 0 or args.h2_col < 0:
        parser.error("column indices must be non-negative")

    directories: list[tuple[float, str, Path]] = []
    for path in args.root.glob("dat_vmc_L64_J1_hz1_hx*"):
        if not path.is_dir():
            continue
        try:
            hx_value, hx_text = hx_from_dir(path)
        except ValueError as exc:
            print(f"warning: {exc}; skipped", file=sys.stderr)
            continue
        directories.append((hx_value, hx_text, path))
    directories.sort(key=lambda item: item[0])

    if not directories:
        print("error: no hx directories found", file=sys.stderr)
        return 1

    results: list[tuple[float, float, float, float, float, int]] = []
    for hx_value, hx_text, directory in directories:
        files = sorted((directory / "output").glob("zvo_aft_out_*.dat"))
        if not files:
            print(f"warning: hx={hx_text}: no zvo_aft_out_*.dat; skipped", file=sys.stderr)
            continue

        energies: list[float] = []
        h2_values: list[float] = []
        for path in files:
            try:
                e, h2 = read_one_file(path, args.energy_col, args.h2_col)
            except (OSError, ValueError) as exc:
                print(f"warning: {exc}; skipped", file=sys.stderr)
                continue
            energies.append(e)
            h2_values.append(h2)

        if not energies:
            print(f"warning: hx={hx_text}: no usable files; skipped", file=sys.stderr)
            continue

        e_arr = np.asarray(energies, dtype=float)
        h2_arr = np.asarray(h2_values, dtype=float)
        variance, variance_err = variance_and_jackknife_error(e_arr, h2_arr)
        results.append(
            (hx_value, float(e_arr.mean()), sem(e_arr), variance, variance_err, e_arr.size)
        )

    if not results:
        print("error: no usable data found", file=sys.stderr)
        return 1

    output = args.output if args.output.is_absolute() else args.root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        f.write("# hx  <H>  err_<H>  <H^2>-<H>^2  err_variance  n_files\n")
        for hx, energy, energy_err, variance, variance_err, n_files in results:
            f.write(
                f"{hx:.16g} {energy:.16e} {energy_err:.16e} "
                f"{variance:.16e} {variance_err:.16e} {n_files:d}\n"
            )

    print(f"wrote {output} ({len(results)} hx values)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

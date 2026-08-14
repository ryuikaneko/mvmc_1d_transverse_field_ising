using ITensors
using ITensorMPS
using Random
using Printf

# ============================================================
# Parameters
# ============================================================

const L  = 64
const J  = 1.0
const hz = 1.0

const outfile = "dat_hx_ene_enedens"

# DMRG settings
const maxdim_schedule = [20, 50, 100, 200, 400, 800, 1200]
const cutoff_value = 1.0e-12
const noise_schedule = [1.0e-6, 1.0e-7, 1.0e-8, 0.0, 0.0, 0.0, 0.0]


# ============================================================
# Construct Hamiltonian
#
# H = -J sum_i X_i X_{i+1}
#     -hx sum_i X_i
#     -hz sum_i Z_i
#
# Periodic boundary:
# X_L X_1 is explicitly included.
# ============================================================

function make_hamiltonian(
    sites;
    L::Int,
    J::Float64,
    hx::Float64,
    hz::Float64,
)
    os = OpSum()

    # Bonds i -- i+1
    for i in 1:(L - 1)
        os += -J, "X", i, "X", i + 1
    end

    # Periodic closing bond L -- 1
    os += -J, "X", L, "X", 1

    # Fields
    for i in 1:L
        os += -hx, "X", i
        os += -hz, "Z", i
    end

    return MPO(os, sites)
end


# ============================================================
# Main scan
# ============================================================

function run_hx_scan()
    Random.seed!(1234)

    # hx != 0 breaks the parity symmetry, so do not conserve QNs.
    sites = siteinds("Qubit", L; conserve_qns=false)

    # Initial MPS used only for hx = 0.
    # Subsequent calculations reuse the previous optimized MPS.
    psi = random_mps(sites; linkdims=10)

    # Create CSV and write header.
    open(outfile, "w") do io
        println(io, "# hx E0 E0_per_site H_expect H2_expect variance")
        flush(io)
    end

    println("============================================================")
    println("Periodic Ising model ground-state energy scan")
    println("L  = ", L)
    println("J  = ", J)
    println("hz = ", hz)
    println("Output file: ", abspath(outfile))
    println("============================================================")

    # Integer loop avoids floating-point accumulation errors.
    # k = 0,...,100 gives hx = 0.00,...,2.00.
    for k in 0:100
        hx = 0.02 * k

        println()
        @printf(
            "Point %3d / 101: hx = %.2f\n",
            k + 1,
            hx,
        )

        H = make_hamiltonian(
            sites;
            L=L,
            J=J,
            hx=hx,
            hz=hz,
        )

        energy, psi = dmrg(
            H,
            psi;
            nsweeps=length(maxdim_schedule),
            maxdim=maxdim_schedule,
            cutoff=cutoff_value,
            noise=noise_schedule,
            eigsolve_krylovdim=8,
            outputlevel=0,
        )

        # Normalize expectation values explicitly.
        psi_norm2 = real(inner(psi, psi))
        H_expect = real(inner(psi', H, psi)) / psi_norm2

        # Since H is Hermitian,
        # <H^2> = <H psi | H psi> / <psi | psi>.
        # The cutoff controls only the MPO-MPS application used here.
        Hpsi = apply(H, psi; cutoff=1.0e-14)
        H2_expect = real(inner(Hpsi, Hpsi)) / psi_norm2
        variance = H2_expect - H_expect^2

        # Remove only a tiny negative value caused by roundoff/truncation.
        variance_tol = 100.0 * eps(Float64) * max(abs(H2_expect), H_expect^2, 1.0)
        if variance < 0.0 && abs(variance) <= variance_tol
            variance = 0.0
        end

        E0_per_site = energy / L

        @printf("  E0             = %.16f\n", energy)
        @printf("  E0/L           = %.16f\n", E0_per_site)
        @printf("  <H>            = %.16f\n", H_expect)
        @printf("  <H^2>          = %.16f\n", H2_expect)
        @printf("  <H^2>-<H>^2    = %.16e\n", variance)
        @printf("  |E0 - <H>|     = %.6e\n", abs(energy - H_expect))
        @printf("  max bond       = %d\n", maxlinkdim(psi))

        # Append immediately so completed points remain saved even
        # if the calculation is interrupted later.
        open(outfile, "a") do io
            @printf(
                io,
                "%.2f %.16f %.16f %.16f %.16f %.16e\n",
                hx,
                energy,
                E0_per_site,
                H_expect,
                H2_expect,
                variance,
            )
            flush(io)
        end
    end

    println()
    println("============================================================")
    println("Scan completed.")
    println("Results saved to:")
    println(abspath(outfile))
    println("============================================================")

    return psi
end


# Run calculation
psi_final = run_hx_scan()

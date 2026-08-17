#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.container import ErrorbarContainer
from matplotlib.legend_handler import HandlerErrorbar

class HandlerErrorbarBarsAbove(HandlerErrorbar):

    def create_artists(
        self,
        legend,
        orig_handle,
        xdescent,
        ydescent,
        width,
        height,
        fontsize,
        trans,
    ):
        artists = super().create_artists(
            legend,
            orig_handle,
            xdescent,
            ydescent,
            width,
            height,
            fontsize,
            trans,
        )

        marker_artists = []
        errorbar_artists = []
        other_artists = []

        for artist in artists:
            if isinstance(artist, LineCollection):
                errorbar_artists.append(artist)

            elif isinstance(artist, Line2D):
                marker = artist.get_marker()

                if marker in ("_", "|"):
                    errorbar_artists.append(artist)

                elif marker not in (None, "None", "", " "):
                    marker_artists.append(artist)

                else:
                    other_artists.append(artist)

            else:
                other_artists.append(artist)

        ordered_artists = (
            other_artists
            + marker_artists
            + errorbar_artists
        )

        return ordered_artists



def scientific_tick_formatter(value, position):
    """Format each tick value explicitly in scientific notation."""
    if np.isclose(value, 0.0, rtol=0.0, atol=1.0e-30):
        return r"$0$"

    exponent = int(np.floor(np.log10(abs(value))))
    coefficient = value / 10.0**exponent

    if np.isclose(coefficient, round(coefficient), atol=1.0e-10):
        coefficient_text = f"{round(coefficient):d}"
    else:
        coefficient_text = f"{coefficient:.3g}"

    return rf"${coefficient_text}\times 10^{{{exponent}}}$"


def symmetric_ylim_with_margin(values, errors, scale=1.8):
    """Return symmetric y limits including error bars and extra margin."""
    lower = np.min(values - errors)
    upper = np.max(values + errors)

    maximum_absolute_value = max(abs(lower), abs(upper))

    if np.isclose(maximum_absolute_value, 0.0):
        maximum_absolute_value = 1.0

    limit = scale * maximum_absolute_value

    return -limit, limit


def asymmetric_ylim_with_margin(
    values,
    errors,
    lower_scale=0.25,
    upper_scale=1.0,
):
    """Return y limits with independently controlled margins."""
    lower = np.min(values - errors)
    upper = np.max(values + errors)

    data_range = upper - lower

    if np.isclose(data_range, 0.0):
        data_range = max(abs(lower), abs(upper), 1.0)

    y_min = lower - lower_scale * data_range
    y_max = upper + upper_scale * data_range

    return y_min, y_max


def main():
    # ========================================================
    # Publication-style plotting parameters
    # ========================================================
    plt.rcParams.update(
        {
            "font.size": 25,
            "axes.labelsize": 30,
            "xtick.labelsize": 25,
            "ytick.labelsize": 25,
            "legend.fontsize": 25,
            "axes.linewidth": 2.1,
            "lines.linewidth": 2.8,
            "xtick.major.width": 1.5,
            "ytick.major.width": 1.5,
            "xtick.minor.width": 1.2,
            "ytick.minor.width": 1.2,
            "xtick.major.size": 6.0,
            "ytick.major.size": 6.0,
            "xtick.minor.size": 3.5,
            "ytick.minor.size": 3.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    # ========================================================
    # Input and output files
    # ========================================================
    dmrg_file = Path("../dat/dat_dmrg_hx_ene_enedens")
    vmc_file = Path("../dat/dat_vmc_hx_ene_err_var_err")
    output_file = Path("fig.pdf")

    # Absolute tolerance used to match magnetic-field values
    hx_atol = 1.0e-10

    # Common color used for all VMC data
#    vmc_color = "tab:blue"
    vmc_color = "tab:red"

    # ========================================================
    # Load data
    #
    # DMRG file:
    #   Column 1: hx
    #   Column 2: energy
    #
    # VMC file:
    #   Column 1: hx
    #   Column 2: <H>
    #   Column 3: error of <H>
    #   Column 4: <H^2> - <H>^2
    #   Column 5: error of the variance
    #   Column 6: number of files
    # ========================================================
    dmrg = np.loadtxt(
        dmrg_file,
        comments="#",
        ndmin=2,
    )

    vmc = np.loadtxt(
        vmc_file,
        comments="#",
        ndmin=2,
    )

    # Validate the number of columns
    if dmrg.shape[1] < 2:
        raise ValueError(
            f"{dmrg_file} must contain at least two columns. "
            f"Actual number of columns: {dmrg.shape[1]}"
        )

    if vmc.shape[1] < 5:
        raise ValueError(
            f"{vmc_file} must contain at least five columns. "
            f"Actual number of columns: {vmc.shape[1]}"
        )

    hx_dmrg = dmrg[:, 0]
    ene_dmrg = dmrg[:, 1]

    hx_vmc = vmc[:, 0]
    ene_vmc = vmc[:, 1]
    ene_error_vmc = vmc[:, 2]
    variance_vmc = vmc[:, 3]
    variance_error_vmc = vmc[:, 4]

    # ========================================================
    # Validate numerical values
    # ========================================================
    if not np.all(np.isfinite(dmrg[:, :2])):
        raise ValueError(
            "The DMRG data contain non-finite values in hx or energy."
        )

    if not np.all(np.isfinite(vmc[:, :5])):
        raise ValueError(
            "The VMC data contain non-finite values in hx, energy, "
            "energy error, variance, or variance error."
        )

    if np.any(ene_error_vmc < 0.0):
        raise ValueError(
            "The VMC energy errors contain negative values."
        )

    if np.any(variance_error_vmc < 0.0):
        raise ValueError(
            "The VMC variance errors contain negative values."
        )

    # ========================================================
    # Sort the original data in ascending order of hx
    # ========================================================
    dmrg_order = np.argsort(hx_dmrg)

    hx_dmrg = hx_dmrg[dmrg_order]
    ene_dmrg = ene_dmrg[dmrg_order]

    vmc_order = np.argsort(hx_vmc)

    hx_vmc = hx_vmc[vmc_order]
    ene_vmc = ene_vmc[vmc_order]
    ene_error_vmc = ene_error_vmc[vmc_order]
    variance_vmc = variance_vmc[vmc_order]
    variance_error_vmc = variance_error_vmc[vmc_order]

    # ========================================================
    # Match only the hx values present in both data sets
    # ========================================================
    matched_hx = []
    matched_energy_difference = []
    matched_energy_error = []

    for i, hx_value in enumerate(hx_vmc):
        indices = np.where(
            np.isclose(
                hx_dmrg,
                hx_value,
                rtol=0.0,
                atol=hx_atol,
            )
        )[0]

        if len(indices) == 0:
            print(
                f"Warning: hx = {hx_value:.16g} is not present in "
                "the DMRG data and will be excluded from the "
                "energy-difference panel."
            )
            continue

        if len(indices) > 1:
            raise ValueError(
                f"Multiple DMRG points match hx = {hx_value:.16g}: "
                f"{len(indices)} matches were found."
            )

        dmrg_index = indices[0]

        matched_hx.append(hx_value)

        matched_energy_difference.append(
            ene_vmc[i] - ene_dmrg[dmrg_index]
        )

        matched_energy_error.append(
            ene_error_vmc[i]
        )

    if len(matched_hx) == 0:
        raise ValueError(
            "No matching hx values were found between the DMRG and "
            f"VMC data sets. The current tolerance is {hx_atol:.1e}."
        )

    hx_common = np.asarray(matched_hx)

    energy_difference = np.asarray(
        matched_energy_difference
    )

    energy_difference_error = np.asarray(
        matched_energy_error
    )

    # Sort the matched data in ascending order of hx
    common_order = np.argsort(hx_common)

    hx_common = hx_common[common_order]
    energy_difference = energy_difference[common_order]
    energy_difference_error = energy_difference_error[common_order]

    print(f"Number of DMRG points: {len(hx_dmrg)}")
    print(f"Number of VMC points: {len(hx_vmc)}")
    print(f"Number of matched hx points: {len(hx_common)}")

    # ========================================================
    # Determine y-axis limits with additional margins
    # ========================================================

    # Use symmetric limits around zero for the energy difference
    difference_ylim = symmetric_ylim_with_margin(
        energy_difference,
        energy_difference_error,
#        scale=2.5,
        scale=8.0,
    )

    # Add a moderate lower margin and a larger upper margin
    variance_ylim = asymmetric_ylim_with_margin(
        variance_vmc,
        variance_error_vmc,
        lower_scale=0.25,
#        upper_scale=1.5,
        upper_scale=1.1,
    )

    # ========================================================
    # Create compact, vertically stacked panels
    # ========================================================
    fig, (
        ax_energy,
        ax_difference,
        ax_variance,
    ) = plt.subplots(
        nrows=3,
        ncols=1,
        sharex=True,
#        figsize=(8.0, 8.0),
        figsize=(10.75, 4.0),
        gridspec_kw={
            "height_ratios": [1.0, 1.0, 1.0],
            "hspace": 0.0,
        },
    )

    # Remove all space between neighboring panels
    fig.subplots_adjust(
        left=0.24,
        right=0.97,
        bottom=0.13,
        top=0.98,
        hspace=0.0,
    )

    # ========================================================
    # Upper panel: DMRG and VMC energies
    # ========================================================
    ax_energy.plot(
        hx_dmrg,
        ene_dmrg,
        color="black",
        linewidth=2.8,
        label="DMRG",
        zorder=0,
    )

    ax_energy.errorbar(
        hx_vmc,
        ene_vmc,
        yerr=ene_error_vmc,
        fmt="o",
        markersize=14.0,
        markerfacecolor="none",
        markeredgecolor=vmc_color,
        markeredgewidth=2.8,
        color=vmc_color,
        ecolor=vmc_color,
        elinewidth=2.8,
        capsize=8.0,
        capthick=2.8,
        barsabove=True,
        label="VMC",
    )

    ax_energy.set_ylabel(r"$\langle H\rangle$",rotation=0,ha="right",va="center")
#    ax_energy.legend(frameon=False)
    legend = ax_energy.legend(
        loc="upper right",
        borderpad=0.0,
        labelspacing=0.0,
        borderaxespad=0.225,
        frameon=False,
        handler_map={
            ErrorbarContainer: HandlerErrorbarBarsAbove(
                xerr_size=0.5,
                yerr_size=0.5,
            )
        },
    )

    # Hide x-axis ticks and labels in the upper panel
    ax_energy.tick_params(
        axis="x",
        which="both",
        bottom=False,
        labelbottom=False,
    )

    ax_energy.set_yticks([-180, -140, -100])

    # ========================================================
    # Middle panel: VMC-DMRG energy difference
    # ========================================================
    ax_difference.errorbar(
        hx_common,
        energy_difference,
        yerr=energy_difference_error,
        fmt="o",
        markersize=14.0,
        markerfacecolor="none",
        markeredgecolor=vmc_color,
        markeredgewidth=2.8,
        color=vmc_color,
        ecolor=vmc_color,
        elinewidth=2.8,
        capsize=8.0,
        capthick=2.8,
        barsabove=True,
    )

    # Draw a reference line at zero energy difference
    ax_difference.axhline(
        0.0,
        color="black",
        linewidth=2.8,
        linestyle="--",
        zorder=0,
    )

    ax_difference.set_ylabel(
#        r"$E_{\mathrm{VMC}}-E_{\mathrm{DMRG}}$"
        r"$\langle H\rangle-\langle H_{\mathrm{DMRG}}\rangle$",
        rotation=0,
        ha="right",va="center",
    )

    ax_difference.set_ylim(
        *difference_ylim
    )

    # Hide x-axis ticks and labels in the middle panel
    ax_difference.tick_params(
        axis="x",
        which="both",
        bottom=False,
        labelbottom=False,
    )

    # Display every y tick using explicit scientific notation
    ax_difference.yaxis.set_major_formatter(
        FuncFormatter(scientific_tick_formatter)
    )

    ax_difference.set_yticks([-0.00004, 0, 0.00004])

    # ========================================================
    # Lower panel: VMC energy variance
    # ========================================================
    ax_variance.errorbar(
        hx_vmc,
        variance_vmc,
        yerr=variance_error_vmc,
        fmt="o",
        markersize=14.0,
        markerfacecolor="none",
        markeredgecolor=vmc_color,
        markeredgewidth=2.8,
        color=vmc_color,
        ecolor=vmc_color,
        elinewidth=2.8,
        capsize=8.0,
        capthick=2.8,
        barsabove=True,
    )

    # Draw a reference line at zero energy difference
    ax_variance.axhline(
        0.0,
        color="black",
        linewidth=2.8,
        linestyle="--",
        zorder=0,
    )

    ax_variance.set_xlabel(r"$h_x$")

    ax_variance.set_ylabel(
        r"$\langle H^2\rangle-\langle H\rangle^2$",
        rotation=0,
        ha="right",va="center",
    )

    ax_variance.set_ylim(
        *variance_ylim
    )

    # Display every y tick using explicit scientific notation
    ax_variance.yaxis.set_major_formatter(
        FuncFormatter(scientific_tick_formatter)
    )

    ax_variance.set_yticks([0, 0.00002, 0.00004])

    # ========================================================
    # Apply common panel settings
    # ========================================================
    axes = (
        ax_energy,
        ax_difference,
        ax_variance,
    )

    for ax in axes:
        ax.tick_params(
            axis="both",
            direction="in",
            right=True,
            pad=6,
        )

        ax.yaxis.labelpad = 12

    ax_variance.xaxis.labelpad = 8

    # Use only one spine at each shared panel boundary
    ax_energy.spines["bottom"].set_visible(False)
    ax_difference.spines["bottom"].set_visible(False)

    # Draw upper ticks at the shared boundaries
    ax_difference.tick_params(top=True)
    ax_variance.tick_params(top=True)

    # Ensure identical horizontal limits in all panels
    hx_min = min(
        np.min(hx_dmrg),
        np.min(hx_vmc),
    )

    hx_max = max(
        np.max(hx_dmrg),
        np.max(hx_vmc),
    )

    hx_range = hx_max - hx_min

    if np.isclose(hx_range, 0.0):
        hx_margin = 1.0
    else:
        hx_margin = 0.02 * hx_range

    ax_energy.set_xlim(
        hx_min - hx_margin,
        hx_max + hx_margin,
    )

    # Align all y-axis labels
    fig.align_ylabels(axes)

    # Save the figure as a PDF
    fig.savefig(
        output_file,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()

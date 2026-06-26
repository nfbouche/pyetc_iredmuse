"""
pyetc_wst - Exposure Time Calculator for the Wide-Field Spectroscopic Telescope (WST)

A Python package for exposure time calculation and signal-to-noise ratio estimation
for the WST instrument suite (IFS, MOS-LR, MOS-HR).
"""

__version__ = "1.5"
__author__ = "Matteo Ferro & Roland Bacon"

# Changelog
# v1.5 (2026-06-22)
#   - Fixed bug in _resolve_best_coadd_ifs: replaced sky-dominated metric
#     (fsq / N) with the full SNR metric (fsq*S / sqrt(fsq*S + N^2*bg)),
#     where bg = sky + dark + RON per spaxel at wave_ref. Source spectrum
#     is now passed from all three callers (snr_from_source_ifs,
#     _snr_at_wave_ifs, time_from_source_ifs) to enable the correct metric.
#   - Increased default max_coadd from 20 to 40 for point sources (bad
#     seeing can push the optimal aperture beyond the old cap).
#   - Fixed lbda_ref selection for COADD_XY='best' in snr_from_source_ifs:
#     channel centre used in DIT+NDIT mode (Lam_Ref is irrelevant there);
#     window centre used when SNR_RANGE=True; Lam_Ref clipped to channel range
#     in time_from_source_ifs to prevent silent fallback to sky-dominated metric.
#   - time_from_source_window oscillation/slow-convergence fixed:
#     NDIT bracket detection exits loop as soon as consecutive integers N, N+1
#     straddle the target SNR (~3 iters instead of 20). DIT secant acceleration
#     replaces slow multiplicative update with linear interpolation of the last
#     two (DIT, SNR) points (~4 iters instead of 20).
#   - Fixed cross-optimisation consistency in time_from_source_ifs(compute='best')
#     + COADD_XY='best': coadd-DIT feedback loop (<=5 iters) jointly solves
#     optimal aperture with DIT+NDIT. Returns dit_sat, ndit_raw, ima_coadd.
#     time_from_source_mos(compute='best') also returns dit_sat and ndit_raw.
#   - Fixed PSF parity inconsistency in best-mode feedback loop: PSF
#     (selected_image) is recomputed when coadd parity changes (even<->odd),
#     keeping the 'uneven' flag consistent with snr_from_source_ifs and
#     eliminating an ~18% SNR discrepancy for odd final coadds.
#   - Web app: 2-step best+window approach (DIT_sat then NDIT for window target;
#     bright-source fallback to NDIT=1 with DIT iteration); coadd freeze before
#     display snr_from_source call for consistent SNR reporting.
#   - Fixed saturation flag in snr_from_source_ifs and snr_from_source_mos:
#     peak counts now divided by NDIT before comparison with threshold_sat;
#     nph_source/nph_sky include DIT*NDIT total, so dividing by NDIT gives
#     the per-single-DIT counts that must not exceed 50000 e-.
#   - Fixed saturation line in plot and peak_counts in API: secondary-axis
#     peak-counts curve and API peak_counts (IFS) now show per-single-DIT
#     counts (divided by NDIT), keeping them consistent with the 50000 e-
#     saturation threshold line.
#   - Added dit_at_min_floor flag in time_from_source_window(compute='dit'):
#     set to True when the converged DIT hits the 0.1 s instrument minimum.
#     Web interface emits a warning that the target SNR cannot be reached.

# Import main classes and functions
from .wst import WST
from .etc import (
    ETC,
    get_data,
    sersic,
    moffat,
    get_seeing_fwhm,
    compute_sky,
    mask_spectrum_edges,
    mask_line_region,
    mask_spectra_in_dict,   
    convolve_and_center,
    plot_noise_components,
)
from .specalib import (
    PhotometricSystem,
    SEDModels,
    FilterManager,
    plot_spectra_comparison,
)

# Define what gets imported with "from pyetc_wst import *"
__all__ = [
    # Main classes
    "WST",
    "ETC",
    "PhotometricSystem",
    "SEDModels",
    "FilterManager",
    # Functions
    "get_data",
    "sersic",
    "moffat",
    "get_seeing_fwhm",
    "compute_sky",
    "mask_spectrum_edges",
    "mask_line_region",
    "mask_spectra_in_dict",
    "convolve_and_center",
    "plot_noise_components",
    "plot_spectra_comparison",
]

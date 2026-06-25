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

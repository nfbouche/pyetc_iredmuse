"""
pyetc_iredmuse - Exposure Time Calculator for the Wide-Field Spectroscopic Telescope (WST)

A Python package for exposure time calculation and signal-to-noise ratio estimation
for the WST instrument suite (IFS, MOS-LR, MOS-HR).
"""

__version__ = "0.1"
__author__ = "Nicolas Bouché & Matteo Ferro & Roland Bacon"

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

# Define what gets imported with "from pyetc_iredmuse import *"
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

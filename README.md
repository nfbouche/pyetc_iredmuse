# pyetc_wst

Exposure Time Calculator (ETC) for the Wide-Field Spectroscopic Telescope (WST).

## Description

**pyetc_wst** is a Python package for exposure time calculation and signal-to-noise ratio (SNR) estimation for the WST instrument suite, including:

- **IFS** (Integral Field Spectrograph): Blue and Red channels
- **MOS-LR** (Multi-Object Spectrograph Low Resolution): Blue, Green, Yellow, and Red channels  
- **MOS-HR** (Multi-Object Spectrograph High Resolution): Blue, Green, Yellow, and Red channels

## Requirements

[ All of them can be installed via `pip` ]
- Python >= 3.9
- numpy >= 1.20.0
- scipy >= 1.7.0
- matplotlib >= 3.3.0
- astropy >= 5.0.0
- mpdaf >= 3.6.0
- skycalc_ipy >= 0.1.0

```
pip install "numpy>=1.20.0" "scipy>=1.7.0" "matplotlib>=3.3.0" "astropy>=5.0.0" "mpdaf>=3.6.0" "skycalc_ipy>=0.1.0"
```

## Installation

### From GitHub

You can install directly from GitHub using pip:

```bash
pip install git+https://github.com/ferromatteo/pyetc_wst.git
```

If you already have it installed via pip, you can upgrade it with:

#### Option 1: forced (recommended)
```bash
pip install --force-reinstall git+https://github.com/ferromatteo/pyetc_wst.git
```

#### Option 2: normal upgrade
```bash
pip install --upgrade git+https://github.com/ferromatteo/pyetc_wst.git
```

#### Option 3: uninstall and reinstall (cleanest option)
```bash
pip uninstall pyetc_wst
pip install git+https://github.com/ferromatteo/pyetc_wst.git
```

## Quick Start

```python
from pyetc_wst import WST

# Initialize the ETC, 'DEBUG' will allow you to see useful prints during the computation,
# skip_dataload = False will load the static sky configurations + general transmissions
wst = WST(log = 'DEBUG', skip_dataload = False)

# Display instrument information
wst.info()

# Access specific instruments
ifs_blue = wst.ifs['blue']
moslr_red = wst.moslr['red']
moshr_yellow = wst.moshr['yellow']

# Build the full dictionaries needed for computation (full_obs), which will include observing conditions, source properties, computation requests, and instrument configuration
full_obs = {...}
con, ob, spe, im, spe_input = wst.build_obs_full(full_obs)

# Compute time or snr given the full dictionary results

# for SNR:
res_snr = wst.snr_from_source(con, im, spe, debug=True/False)

# for SNR at a specific wavelength:
res_snr_at_wave = wst.snr_at_wave(con, im, spe, debug=True/False)

# for time/exposures/best combination
res_time = wst.time_from_source(con, im, spe, compute = 'dit'/'ndit'/'best', debug=True/False)
```
`debug=True/False` allows to print detailed info of the current run.

**NOTE**: *`time_from_source()` basically update the 'dit', 'ndit' or both values in the obs. dictionary to the value/values needed to reach a specific SNR at a specific wavelength, after it you could run a `res_snr = wst.snr_from_source(con, im, spe)` and plot the SNR to check the results.*

A full_obs dictionary should look like this (detailed information are given in the file **encoding.txt**):
```python
full_obs = {
    "INS": "moslr",
    "CH": "red",
    
    "NDIT": 1,
    "DIT": 600, 
    
    "SNR": 5,
    "Lam_Ref": 5000,
    
    "OBJ_FIB_DISP": 0,
    
    "PWV": 10,
    "FLI": 0.5,
    "MOON_SEP": 45,
    "SEE": 0.8,
    "AM": 1.2,
    "SKYCALC": False,
    
    "Obj_SED": 'template',
    "SED_Name": 'MARCS_8000K_lg+45',
    "UPLOAD_FILE": "path/to/spec.txt"
    
    "OBJ_MAG": 15, #can be None for loaded spectrum
    "MAG_SYS": 'Vega',
    "MAG_FIL": 'V',
    
    "Z": 0,
    "BB_Temp": 9000.,
    "PL_Index": None,
    
    "SEL_FLUX": 50e-16,
    "SEL_CWAV": 8000,
    "SEL_FWHM":20,
    
    "Obj_Spat_Dis": 'resolved',
    
    "IMA": 'moffat',
    
    "IMA_FWHM": 0.5,
    "IMA_BETA": 2.5,
    
    "Sersic_Reff": 1,
    "Sersic_Ind": 3,
    
    "COADD_WL": 10,
    
    "COADD_XY": 1 #(all integer numbers or 'best')
}
```
**NOTE**: *"COADD_XY": 'best' — automatically selects the spatial coadding that maximizes the SNR. Like the compute options in `time_from_source`, it updates "COADD_XY" in the obs dictionary with the chosen value.*

**NOTE (2)**: *When using `"Obj_SED": "upload"`, you must provide `"UPLOAD_FILE": "/path/to/spectrum.dat"` pointing to a two-column ASCII file/FITS table (wavelength, flux). Optional comment headers (or "units" of the FITS columns) set units: `# nm` or `# aa` for wavelength (default: Å - `aa`), `# fl` or `# ph` for flux (default: erg/cm²/s/Å - `fl`). Set `"OBJ_MAG": null` to use the spectrum as-is, or set a numeric value (e.g. `18`) to normalize it to that magnitude in the chosen `MAG_FIL`/`MAG_SYS` band.*

After the computation results can be plotted easily accessing the mpdaf `Spectrum` objects in the results dictionary like this:
```python
res_snr['spec']['snr'].plot()
```
![Noise Plot](images/SNR.png)

or
```python
res_snr['spec']['nph_source'].plot()
```

In general the results of the snr computation `res_snr` will have a main dictionary `res_snr['spec']` which contains several sub-dictionaries, all related to the 
integration in the aperture area for the MOS, and the requested `COADD_XY x COADD_XY` for the IFS (which has also another dictionary `res_snr['peak']`) for the central pixel value.

These sub-dictionaries include:
- 'nph_source': photon from source
- 'nph_sky': photon from sky
- 'snr': snr in the aperture/integration area
- 'snr_rebin': snr in the aperture/integration area, rebinned with COADD_WL
- 'simulated_counts': 1D extracted raw spectrum
- 'noise': another dictionary with noise components and their fractions
  - 'tot'
  - 'frac_source'
  - 'frac_sky'
  - 'frac_dark' 
  - 'frac_ron'

For MOS runs, `res_snr['input']` also includes:
- `'fiber_injection'`: fiber injection fraction (1.0 for surface brightness, computed for point/resolved sources)
- `'total_trans'`: total transmission used in plots/results. For MOS it is `instrument x atmosphere x fiber_injection`.

Moreover, there is a handy function to plot all the noise components together, and will accept `res_snr['spec']['noise']` (and also `res_snr['peak']['noise']`) for IFS): 

```python
plot_noise_components(res_snr['spec']['noise'])
```
![Noise Plot](images/noise.png)

## Notebook
`WST_LimMag.ipynb`: computes the limiting magnitude of the WST Integral Field Spectrograph (IFS) as a function of wavelength, for both point sources and extended sources (surface brightness), across the blue and red channels.
For a given target S/N ratio, the notebook sweeps the wavelength range and finds — via Brent's root-finding method — the faintest AB magnitude detectable under three sky background conditions: dark, grey, and bright time.


## Documentation

update in future version

## Citation

This package has been developed from the original `pyetc` package available at https://github.com/RolandBacon/pyetc

update in future version

## Version

### 1.1:
- Release date: 24/04/2026
- Added MOS fiber injection fraction (`fiber_injection`) to exported inputs/results.
- Updated MOS total throughput to include fiber injection fraction (fiber inj. frac.) in addition to instrument and atmosphere.
- Added dedicated fiber inj. frac. curve to MOS throughput plots in the web interface.
- Consolidated recent fixes (RON handling updates, surface-brightness/MOS corrections, and sky-area term consistency updates).
- Improved configuration/info tracking text for throughput model metadata.
- Added moon-target separation as a user-settable parameter (`MOON_SEP`, default 45°); previously fixed at 45° internally.
- Fixed MOS object displacement validation range: now correctly 0–0.6 arcsec (previously the web interface rejected values above 0.3 arcsec).

### 1.0:
- Official release 09/03/2026
  
## Contact

Matteo Ferro - [matteo.ferro@inaf.it]

Project Link: [https://github.com/ferromatteo/pyetc_wst](https://github.com/ferromatteo/pyetc_wst)

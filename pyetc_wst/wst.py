import logging
import os, sys
import time
import numpy as np

from mpdaf.obj import Spectrum, WaveCoord
from mpdaf.log import setup_logging

from .etc import ETC, get_data
from . import __version__ as PACKAGE_VERSION

# used by get_data
from astropy.table import Table
import astropy.units as u

CURDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
SKYDIR = CURDIR + '/sky'
WSTDIR = CURDIR + '/wst'

class WST(ETC):
    
    def __init__(self, log=logging.INFO, skip_dataload=False):
        start_time = time.time()
        self.refdir = CURDIR
        setup_logging(__name__, level=log, stream=sys.stdout)
        self.logger = logging.getLogger(__name__)
        self.logger.propagate = False
        
        # ------ Telescope ---------
        self.name = 'WST'
        self.throughput_model_desc = 'Throughput model version 1 delivered by Olga Bellido, date 09/03/2026'
        self.throughput_model_version = '09/03/2026'
        self.release_info = {
            'version': PACKAGE_VERSION,
            'release_date': '22 June 2026',
            'history': [
                {
                    'version': '1.5',
                    'label': 'Version 1.5',
                    'release_date': '22 June 2026',
                    'changes': [
                        'Fixed bug in _resolve_best_coadd_ifs (IFS COADD_XY="best" mode): replaced the sky-dominated approximation metric fsq/N with the correct full SNR metric signal/sqrt(signal + N²·bg_per_spaxel), where signal = fsq·S and bg_per_spaxel = sky + dark + RON per spaxel at the reference wavelength. The source spectrum is now passed from all three callers (snr_from_source_ifs, _snr_at_wave_ifs, time_from_source_ifs); when no spectrum is available the old approximation is used as fallback.',
                        'Increased default max_coadd from 20 to 40 for point sources: bad seeing conditions (e.g. 1.5–2") with small spaxels can push the optimal aperture close to or beyond the old cap.',
                        'For resolved sources, max_coadd is now derived automatically from the extent of the source image (min(ima.shape) // oversamp), removing the artificial fixed ceiling and ensuring the full extent of extended morphologies is searched.',
                        'Fixed reference wavelength (lbda_ref) selection for COADD_XY="best" in snr_from_source_ifs: in DIT+NDIT mode the SNR reference wavelength (Lam_Ref) is now ignored for coadd optimisation — the channel centre is used instead (Lam_Ref is irrelevant when computing the full SNR spectrum). When SNR_RANGE=True the window centre [LAM_WIN1+LAM_WIN2]/2 is used. In time_from_source_ifs, Lam_Ref is clipped to the channel range so an out-of-range value no longer silently zeros the source spectrum and falls back to the sky-dominated metric.',
                        'Fixed oscillation/slow-convergence in time_from_source_window: NDIT bracket detection exits the loop as soon as consecutive integers N and N+1 straddle the target SNR (~3 iterations instead of 20). DIT secant acceleration replaces the slow multiplicative update with linear interpolation of the last two (DIT, SNR) points, reducing typical DIT convergence from 20 to ~4 iterations.',
                    ],
                },
                {
                    'version': '1.4',
                    'label': 'Version 1.4',
                    'release_date': '28 May 2026',
                    'changes': [
                        'Added GLAO (Ground Layer Adaptive Optics) support: new GLAO=True parameter in the obs dictionary enables a wavelength-dependent IQ formula for IFS (IQ_glao(λ) = (A·λ_nm² + B·λ_nm + C)·AM^0.6, A=1.22465e-7, B=-0.000576386, C=0.717164), and overrides the natural seeing to 0.8 arcsec average for MOS.',
                        'Moffat beta updated: default (non-AO) changed from 2.50 to 2.80; IFS-GLAO uses beta=2.5.',
                        'Added 25 SWIRE galaxy/AGN spectral templates: ellipticals (Ell2/5/13), spirals (Sa/Sb/Sc/Sd/Sdm/S0/Spi4), starbursts (M82/N6090/N6240/Arp220/I19254/I20551/I22491), Seyferts (Sey18/Sey2), QSOs (QSO1/QSO2/BQSO1/TQSO1/Mrk231), and Torus.',
                        'Added snr_in_window(res, lam1, lam2, dlbda, unit, stat) utility function: computes median or mean SNR (per spectral pixel) inside a spectral window from a snr_from_source result.',
                        'Added ETC.time_from_source_window(ins, ima, spec, lam1, lam2, target_snr, unit, compute, n_iter) method: iteratively finds the DIT or NDIT required to reach a target median SNR (per spectral pixel) within a user-defined wavelength window [λ1, λ2]. Returns ndit_raw (pre-ceil float) alongside the ceiled integer.',
                        'NDIT rounding unified to ceil across all compute paths (dit_snr, best, time_from_source_window): max(1, int(np.ceil(...))) — if 7.1 exposures are needed, result is always 8.',
                        'Web interface: added GLAO toggle, spectral window inputs (λ1/λ2) for window-based exposure time computation, improved ASCII/JSON config downloads with timestamped filenames, GLAO/SNR_WIN/LAM_WIN1/LAM_WIN2 included in saved config, SNR window shown in input summary instead of reference wavelength, and consistent NDIT rounding messages showing "X.XXXX, updating to Y for computation." in all paths.',
                        'time_from_source_window convergence improved: max iterations raised to 20, convergence criterion changed to SNR-relative tolerance 1e-4 (0.01%), NDIT sub-1 case exits after one exact analytical step, no artificial floor on param_val during iteration.',
                        'API JSON response: snr_window field added when SNR_WIN=True, reporting median_snr_pixel and (if COADD_WL>1) median_snr_rebin in the requested wavelength window.',
                        'Web results: window mode now shows both rebinned and per-pixel median SNR (when COADD_WL>1) in all compute modes (dit_snr, ndit_snr, best), consistent with non-window behaviour.',
                        'Emission line source (Obj_SED="line"): SNR spectral window is now automatically overridden in all compute modes (DIT/NDIT, DIT&SNR, NDIT&SNR, Best) — SNR_WIN is forced to False and the line central wavelength (SEL_CWAV) is always used as the sole reference. A debug message is emitted when the override occurs. The web interface hides and unchecks the SNR window controls whenever SED type is set to "line".',
                    ],
                },
                {
                    'version': '1.3',
                    'label': 'Version 1.3',
                    'release_date': '14 May 2026',
                    'changes': [
                        'Migrated sky background retrieval from skycalc_ipy to skycalc_cli (official ESO CLI package): sky model data now accessed via skm.data attribute and parsed with Table.read(BytesIO(skm.data), format="fits").',
                        'Fixed FITS column names: skycalc_cli v1.4 returns lowercase column names (lam, flux, trans) — updated in both etc.py (get_sky()) and app.py (compute_sky_dummy()).',
                        'Replaced deprecated pkg_resources with importlib.resources in specalib.py for Python 3.9+ compatibility.',
                        'SNR values now displayed in bold in the computation results panel.',
                        'Plot titles and trace names corrected: "x spectral pixel/coadding" replaced with "/ spectral pixel/coadding" throughout the web interface.',
                        'Removed invalid PWV value 0.01 from the allowed grid (SkyCalc minimum is 0.05); previously this caused SkyCalc to reject the entire request.',
                        'Fixed false MAG_SYS validation error raised for emission-line sources (Obj_SED="line"): the check is now skipped when the SED type is "line" or OBJ_MAG is None, since magnitude normalisation is not used in those cases.',
                    ],
                },
                {
                    'version': '1.2',
                    'label': 'Version 1.2',
                    'release_date': '29 April 2026',
                    'changes': [
                        'Fixed SkyCalc sky background retrieval: bypassed skycalc_ipy.get_sky_spectrum() to call SkyModel directly and read the returned FITS HDUList with an explicit format="fits" argument, resolving an IORegistryError from newer astropy versions.',
                        'Moon altitude now derived from moon-target separation and target zenith distance (z_moon = rho + z_target), satisfying the SkyCalc constraint |z - z_moon| <= rho <= z + z_moon for all airmasses.',
                    ],
                },
                {
                    'version': '1.1',
                    'label': 'Version 1.1',
                    'release_date': '24 April 2026',
                    'changes': [
                        'Added MOS fiber injection fraction (fiber_injection) to exported inputs/results.',
                        'Updated MOS total throughput to include fiber injection fraction; added dedicated fiber inj. frac. curve to MOS throughput plots in the web interface.',
                        'Consolidated recent fixes: RON handling updates, surface-brightness/MOS corrections, and sky-area term consistency updates.',
                        'Added moon-target separation as a user-settable parameter (MOON_SEP, default 45 deg); previously fixed at 45 deg internally.',
                        'Fixed MOS object displacement validation range: now correctly 0–0.6 arcsec (previously the web interface rejected values above 0.3 arcsec).',
                    ],
                },
                {
                    'version': '1.0',
                    'label': 'Version 1.0 — Official Release',
                    'release_date': '09 March 2026',
                    'changes': [
                        'Official release of the WST Exposure Time Calculator.',
                        'Throughput curves (all channels) delivered by Olga Bellido (WST System Engineer).',
                    ],
                },
            ],
        }
        
        self.tel = dict(effective_area_MOS=93.57, # mean of median and weighted mean of the ICD document
                        effective_area_IFS=92.03, # minimum of the ICD document
                        diameter=12.0, # primary diameter
                        desc='Based on Design doc: WST-00006_3 Telescope optical design report', # Link: https://stfc365.sharepoint.com/sites/Wide-FieldSpectroscopicTelescope/_layouts/15/DocIdRedir.aspx?ID=7ZWYDD3PV4SU-175398458-2018
                        version='03/03/2026',
                        iq_fwhm_ins = {
                            'ifs': 0.1, # average FWHM on 95% FOV as also used in the WST_IFS_Tradeoff_Matrix. Link: https://stfc365.sharepoint.com/sites/Wide-FieldSpectroscopicTelescope/_layouts/15/DocIdRedir.aspx?ID=7ZWYDD3PV4SU-175398458-2008
                            'mos': 0.1875,
                            }
                        )

        # ------- GLAO parameters -----------
        self.glao = dict(
            ifs_beta=2.5,          # Moffat beta for IFS+GLAO (AO-corrected PSF profile)
            mos_seeing=0.8,        # Fixed seeing override for MOS+GLAO (Paranal median, zenith, 5000 Å)
        )

        # ------- IFS -----------
        self.ifs = {} 
        self.ifs['channels'] = ['blue', 'red']
        # IFS blue channel
        chan = 'blue'
        self.ifs[chan] = dict(desc = self.throughput_model_desc,
                      version = self.throughput_model_version,
                              type = 'IFS',
                              iq_fwhm_tel = self.tel['iq_fwhm_ins']['ifs'], # fwhm PSF of telescope
                              iq_fwhm_ins = 0.13, # fwhm PSF of instrument, previously 0.30, updated on 03/03/2026, this probably considers also the detector (charge diffusion)
                              iq_beta = 2.80, # beta PSF of telescope + instrument (non-AO Moffat)
                              spaxel_size = 0.25, # spaxel size in arcsec ( * * * check for the binning 2x1, could be 0.125)
                              dlbda = 0.48, # Angstroem/pixel, previously 0.5, updated on 03/03/2026
                              lbda1 = 3700, # starting wavelength in Angstroem
                              lbda2 = 6400, # end wavelength in Angstroem
                              lsfpix = 2.5, # LSF in spectel, previously 3.0, updated on 03/03/2026 ( * * * check)
                              ron = 1.0 * np.sqrt(2), # readout noise (e-) # squared sum for the 2x1 binning
                              dcurrent = 1.0 * 2, # dark current (e-/pixel/h) # sum for the 2x1 binning                                
                              )
        if not skip_dataload:
            get_data(self.ifs, chan, 'ifs', SKYDIR, WSTDIR)

        # IFS red channel
        chan = 'red'
        self.ifs[chan] = dict(desc=self.throughput_model_desc,
                       version = self.throughput_model_version,
                               type='IFS',
                               iq_fwhm_tel = self.tel['iq_fwhm_ins']['ifs'], # fwhm PSF of telescope
                               iq_fwhm_ins = 0.13, # fwhm PSF of instrument, previously 0.30, updated on 03/03/2026, this probably considers also the detector (charge diffusion)
                               iq_beta = 2.80, # beta PSF of telescope + instrument (non-AO Moffat)
                               spaxel_size = 0.25, # spaxel size in arcsec ( * * * check for the binning 2x1, could be 0.125)
                               dlbda = 0.64, # Angstroem/pixel, previously 0.67, updated on 03/03/2026
                               lbda1 = 6200, # starting wavelength in Angstroem
                               lbda2 = 9800, # end wavelength in Angstroem
                               lsfpix = 2.5, # LSF in spectel, previously 3.0, updated on 03/03/2026 ( * * * check)
                               ron = 1.0 * np.sqrt(2), # readout noise (e-) # squared sum for the 2x1 binning
                               dcurrent = 1.0 * 2, # dark current (e-/pixel/h) # sum for the 2x1 binning   
                               )
        if not skip_dataload:
            get_data(self.ifs, chan, 'ifs', SKYDIR, WSTDIR)
              
        # # --------- MOSLR-VIS 4 channels 6k CCD -------------
        
        # # # update with these values
        # # # https://stfc365.sharepoint.com/:w:/r/sites/Wide-FieldSpectroscopicTelescope/_layouts/15/Doc.aspx?sourcedoc=%7B88FA295F-6BE1-4C0E-8885-1856DE7B8383%7D&file=MOS-LR%20ETCinputs_v02.docx&action=default&mobileredirect=true
        self.moslr = {} 
        self.moslr['channels'] = ['blue', 'green', 'yellow', 'red']       
        # MOS-LR blue channel 
        chan = self.moslr['channels'][0]
        self.moslr[chan] = dict(desc=self.throughput_model_desc,
                    version = self.throughput_model_version,
                                type = 'MOS',
                                iq_fwhm_tel = self.tel['iq_fwhm_ins']['mos'], # fwhm PSF of telescope
                                iq_fwhm_ins = 0.20, # fwhm PSF of instrument, previously 0.30, updated on 03/03/2026
                                iq_beta = 2.80, # beta PSF of telescope + instrument (non-AO Moffat)
                                spaxel_size = 0.1515 , # spaxel size in arcsec, previously 0.208, updated on 03/03/2026
                                aperture = 1.03, # fiber diameter in arcsec 
                                dlbda = 0.206, # Angstroem/pixel, previously 0.256, updated on 03/03/2026
                                lbda1 = 3700, # starting wavelength in Angstroem **from Olga's throughput
                                lbda2 = 4770, # end wavelength in Angstroem **from Olga's throughput
                                lsfpix = 6.8, # LSF in spectel, previously 4.83, updated on 03/03/2026 ( * * * check)
                                ron = 1.0, # readout noise (e-) 
                                dcurrent = 1.0, # dark current (e-/pixel/h)                           
                                )
        if not skip_dataload:
            get_data(self.moslr, chan, 'moslr', SKYDIR, WSTDIR)
            
        # MOS-LR green channel      
        chan = self.moslr['channels'][1] 
        self.moslr[chan] = dict(desc=self.throughput_model_desc,
                    version = self.throughput_model_version,
                                type = 'MOS',
                                iq_fwhm_tel = self.tel['iq_fwhm_ins']['mos'], # fwhm PSF of telescope
                                iq_fwhm_ins = 0.20, # fwhm PSF of instrument, previously 0.30, updated on 03/03/2026
                                iq_beta = 2.80, # beta PSF of telescope + instrument (non-AO Moffat)
                                spaxel_size = 0.1515 , # spaxel size in arcsec, previously 0.208, updated on 03/03/2026
                                aperture = 1.03, # fiber diameter in arcsec
                                dlbda = 0.266, # Angstroem/pixel, previously 0.352, updated on 03/03/2026
                                lbda1 = 4630, # starting wavelength in Angstroem **from Olga's throughput
                                lbda2 = 6080, # end wavelength in Angstroem **from Olga's throughput
                                lsfpix = 6.8, # LSF in spectel, previously 4.83, updated on 03/03/2026
                                ron = 1.0, # readout noise (e-)
                                dcurrent = 1.0, # dark current (e-/pixel/h)                                
                                )
        if not skip_dataload:
            get_data(self.moslr, chan, 'moslr', SKYDIR, WSTDIR)

        # MOS-LR yellow channel      
        chan = self.moslr['channels'][2] 
        self.moslr[chan] = dict(desc=self.throughput_model_desc,
                    version = self.throughput_model_version,
                                type = 'MOS',
                                iq_fwhm_tel = self.tel['iq_fwhm_ins']['mos'], # fwhm PSF of telescope
                                iq_fwhm_ins = 0.20, # fwhm PSF of instrument, previously 0.30, updated on 03/03/2026
                                iq_beta = 2.80, # beta PSF of telescope + instrument (non-AO Moffat)
                                spaxel_size = 0.1515, # spaxel size in arcsec, previously 0.208, updated on 03/03/2026
                                aperture = 1.03, # fiber diameter in arcsec
                                dlbda = 0.344, # Angstroem/pixel, previously 0.352, updated on 03/03/2026
                                lbda1 = 5920, # starting wavelength in Angstroem **from Olga's throughput
                                lbda2 = 7710, # end wavelength in Angstroem **from Olga's throughput
                                lsfpix = 6.8, # LSF in spectel, previously 4.83, updated on 03/03/2026 ( * * * check)
                                ron = 1.0, # readout noise (e-)
                                dcurrent = 1.0, # dark current (e-/pixel/h)                             
                                )
        if not skip_dataload:
            get_data(self.moslr, chan, 'moslr', SKYDIR, WSTDIR)

        # MOS-LR red channel      
        chan = self.moslr['channels'][3] 
        self.moslr[chan] = dict(desc=self.throughput_model_desc,
                    version = self.throughput_model_version,
                                type = 'MOS',
                                iq_fwhm_tel = self.tel['iq_fwhm_ins']['mos'], # fwhm PSF of telescope
                                iq_fwhm_ins = 0.20, # fwhm PSF of instrument, updated on 03/03/2026
                                iq_beta = 2.80, # beta PSF of telescope + instrument (non-AO Moffat)
                                spaxel_size = 0.1515, # spaxel size in arcsec, previously 0.208, updated on 03/03/2026
                                aperture = 1.03, # fiber diameter in arcsec
                                dlbda = 0.362, # Angstroem/pixel, previously 0.486, updated on 03/03/2026
                                lbda1 = 7490, # starting wavelength in Angstroem **from Olga's throughput
                                lbda2 = 9800, # end wavelength in Angstroem **from Olga's throughput
                                lsfpix = 6.8, # LSF in spectel, previously 4.83, updated on 03/03/2026 ( * * * check)
                                ron = 1.0, # readout noise (e-)
                                dcurrent = 1.0, # dark current (e-/pixel/h)                              
                                )
        if not skip_dataload:
            get_data(self.moslr, chan, 'moslr', SKYDIR, WSTDIR)

            
        # --------- MOS-HR 4 channels ------------- # We use dioptric values
        self.moshr = {} 
        self.moshr['channels'] = ['blue', 'green', 'yellow', 'red']       
        # MOS-HR blue channel 
        chan = self.moshr['channels'][0]
        self.moshr[chan] = dict(desc=self.throughput_model_desc,
                    version = self.throughput_model_version,
                                type = 'MOS',
                                iq_fwhm_tel = self.tel['iq_fwhm_ins']['mos'], # fwhm PSF of telescope
                                iq_fwhm_ins = 0.16, # fwhm PSF of instrument, updated on 03/03/2026
                                iq_beta = 2.80, # beta PSF of telescope + instrument (non-AO Moffat)
                                spaxel_size = 0.0925, # spaxel size in arcsec, updated on 03/03/2026
                                aperture = 1.00, # fiber diameter in arcsec
                                dlbda = 0.027, # Angstroem/pixel, updated on 03/03/2026
                                lbda1 = 4009, # starting wavelength in Angstroem **same as Olga's throughput
                                lbda2 = 4431, # end wavelength in Angstroem **same as Olga's throughput
                                lsfpix = 3.6, # LSF in spectel, updated on 03/03/2026 ( * * * check)
                                ron = 1.0, # readout noise (e-)
                                dcurrent = 1.0, # dark current (e-/pixel/h)                                
                                )
        if not skip_dataload:
            get_data(self.moshr, chan, 'moshr', SKYDIR, WSTDIR)
            
        # MOS-HR green channel 
        chan = self.moshr['channels'][1]
        self.moshr[chan] = dict(desc=self.throughput_model_desc,
                    version = self.throughput_model_version,
                                type = 'MOS',
                                iq_fwhm_tel = self.tel['iq_fwhm_ins']['mos'], # fwhm PSF of telescope, updated on 03/03/2026
                                iq_fwhm_ins = 0.16, # fwhm PSF of instrument, updated on 03/03/2026
                                iq_beta = 2.80, # beta PSF of telescope + instrument (non-AO Moffat)
                                spaxel_size = 0.0925, # spaxel size in arcsec, updated on 03/03/2026
                                aperture = 1.00, # fiber diameter in arcsec
                                dlbda = 0.032, # Angstroem/pixel, updated on 03/03/2026
                                lbda1 = 4522, # starting wavelength in Angstroem **same as Olga's throughput
                                lbda2 = 4998, # end wavelength in Angstroem **same as Olga's throughput
                                lsfpix = 3.6, # LSF in spectel, updated on 03/03/2026 ( * * * check)
                                ron = 1.0, # readout noise (e-)
                                dcurrent = 1.0, # dark current (e-/pixel/h)                                
                                )
        if not skip_dataload:
            get_data(self.moshr, chan, 'moshr', SKYDIR, WSTDIR)

        # MOS-HR V channel
        chan = self.moshr['channels'][2]
        self.moshr[chan] = dict(desc=self.throughput_model_desc,
                    version = self.throughput_model_version,
                                type = 'MOS',
                                iq_fwhm_tel = self.tel['iq_fwhm_ins']['mos'], # fwhm PSF of telescope, updated on 03/03/2026
                                iq_fwhm_ins = 0.16, # fwhm PSF of instrument, updated on 03/03/2026
                                iq_beta = 2.80, # beta PSF of telescope + instrument (non-AO Moffat)
                                spaxel_size = 0.0925, # spaxel size in arcsec, updated on 03/03/2026
                                aperture = 1.00, # fiber diameter in arcsec
                                dlbda = 0.043, # Angstroem/pixel, updated on 03/03/2026
                                lbda1 = 5424.5, # starting wavelength in Angstroem  **same as Olga's throughput
                                lbda2 = 5995.5, # end wavelength in Angstroem  **same as Olga's throughput
                                lsfpix = 3.6, # LSF in spectel, updated on 03/03/2026 ( * * * check)
                                ron = 1.0, # readout noise (e-)
                                dcurrent = 1.0, # dark current (e-/pixel/h)                              
                                )
        if not skip_dataload:
            get_data(self.moshr, chan, 'moshr', SKYDIR, WSTDIR)

        # MOS-HR I channel
        chan = self.moshr['channels'][3]
        self.moshr[chan] = dict(desc=self.throughput_model_desc,
                    version = self.throughput_model_version,
                                type = 'MOS',
                                iq_fwhm_tel = self.tel['iq_fwhm_ins']['mos'], # fwhm PSF of telescope, updated on 03/03/2026
                                iq_fwhm_ins = 0.16, # fwhm PSF of instrument, updated on 03/03/2026
                                iq_beta = 2.80, # beta PSF of telescope + instrument (non-AO Moffat)
                                spaxel_size = 0.0925, # spaxel size in arcsec, updated on 03/03/2026
                                aperture = 1.00, # fiber diameter in arcsec
                                dlbda = 0.048, # Angstroem/pixel, updated on 03/03/2026
                                lbda1 = 6080, # starting wavelength in Angstroem **same as Olga's throughput
                                lbda2 = 6720, # end wavelength in Angstroem **same as Olga's throughput
                                lsfpix = 3.6, # LSF in spectel, updated on 03/03/2026 ( * * * check)
                                ron = 1.0, # readout noise (e-)
                                dcurrent = 1.0, # dark current (e-/pixel/h)                        
                                )
        if not skip_dataload:
            get_data(self.moshr, chan, 'moshr', SKYDIR, WSTDIR)
        
        end_time = time.time()
        if log == logging.DEBUG or log == 'DEBUG':
            self.logger.debug(f"WST.__init__ processing time: {end_time - start_time:.4f} seconds")
        
    def info(self, ins=None):
        rel = self.get_release_info()
        self.logger.info('ETC version %s release date %s', rel['version'], rel['release_date'])
        for item in rel['changelog']:
            self.logger.info('\t- %s', item)
        if ins is None:
            self._info(['ifs', 'moslr', 'moshr'])
        else:
            self._info([ins])

    def get_release_info(self):
        history = list(self.release_info.get('history', []))
        latest_changes = list(history[0]['changes']) if history else []
        return {
            'version': self.release_info.get('version', PACKAGE_VERSION),
            'release_date': self.release_info.get('release_date', ''),
            'changelog': latest_changes,
            'history': history,
        }

# # # # # # # MORE # # # # # #
# MOS-HR missing the iq instrument (used 0.3" constant), MOS-LR IQ not clear (used 0.3" constant)
# telescope IQ missing for IFS (used 0.07" constant, taken from fig. 30 of the telescope optical design report), for MOS-LR and MOS-HR (used 0.1875" constant, from fig. 15 z=30deg, at 74% area)
# Moffat Beta missing everywhere (used 2.5 constant)
# Diameter missing everywhere (used 12m constant)

# 26/02/2026: updated MOS-LR with new values from the document, added yellow channel, updated version number, added data files with all the new throughput curves
# we still miss the other quantities for the yellow channel, are they changed for the other channels? we just copied green
# Diop throughput not taken into account, for now just the cata

# 09/03/2026: updated IFS with new CMOS binning values and the MOS-HR with the dioptric values
# # # # # # # # # # # # # # #

                
           
            

               
        
        

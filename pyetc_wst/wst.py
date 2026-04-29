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
            'release_date': '29 April 2026',
            'changelog': [
                'Throughput curves (all channels) delivered by Olga Bellido (WST System Engineer) for v1.0 and remain valid in v1.2.',
                'Fixed sky background area computation: now correctly uses pi*r^2 (previously a typo was computing pi^2*r*2).',
                'Fixed IFS red-channel RON: now 1.0*sqrt(2) = 1.4 e- (consistent with blue channel; previously was 1.0*2^0.25 = 1.2 e-).',
                'Fixed MOS surface-brightness SNR computation that was raising an error.',
                'Fixed MOS total throughput: now includes the fiber injection fraction (fiber inj. frac.); a dedicated fiber inj. frac. curve is now shown in the web plots.',
                'Added moon-target separation as a user-settable parameter (MOON_SEP, default 45°); previously fixed at 45° internally.',
                'Fixed MOS object displacement validation range: now correctly 0–0.6 arcsec (previously the web interface rejected values above 0.3 arcsec).',
                'Fixed SkyCalc moon geometry: moon altitude is now placed strictly inside the constraint boundary, resolving failures for airmass > 1.0.',
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
                              iq_beta = 2.50, # beta PSF of telescope + instrument
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
                               iq_beta = 2.50, # beta PSF of telescope + instrument
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
                                iq_beta = 2.50, # beta PSF of telescope + instrument
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
                                iq_beta = 2.50, # beta PSF of telescope + instrument
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
                                iq_beta = 2.50, # beta PSF of telescope + instrument
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
                                iq_beta = 2.50, # beta PSF of telescope + instrument
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
                                iq_beta = 2.50, # beta PSF of telescope + instrument
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
                                iq_beta = 2.50, # beta PSF of telescope + instrument
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
                                iq_beta = 2.50, # beta PSF of telescope + instrument
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
                                iq_beta = 2.50, # beta PSF of telescope + instrument
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
        return {
            'version': self.release_info.get('version', PACKAGE_VERSION),
            'release_date': self.release_info.get('release_date', ''),
            'changelog': list(self.release_info.get('changelog', [])),
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

                
           
            

               
        
        

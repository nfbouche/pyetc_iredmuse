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
rMUSEDIR = CURDIR + '/iredmuse'

class iredMUSE(ETC):
    
    def __init__(self, log=logging.INFO, skip_dataload=False):
        start_time = time.time()
        self.refdir = CURDIR
        setup_logging(__name__, level=log, stream=sys.stdout)
        self.logger = logging.getLogger(__name__)
        self.logger.propagate = False
        
        # ------ Telescope ---------
        self.name = 'WST'
        self.throughput_model_desc = 'Throughput model version 1 delivered by Olga Bellido, date 09/03/2026'
        self.throughput_model_version = '01/06/2026'
        self.release_info = {
            'version': PACKAGE_VERSION,
            'release_date': '26 June 2026',
            'history': [
                {
                    'version': '0.1',
                    'label': 'Version 0.1',
                    'release_date': '26 June 2026',
                    'changes': [
                        'Initial version'
                        ],
                },

            ],
        }
        
        self.tel = dict(effective_area_IFS=48.03, # minimum of the ICD document
                        diameter=8.0, # primary diameter
                        desc='Based on Prelim Concept',
                        version='26/06/2026',
                        iq_fwhm_ins = {
                            'ifs': 0.1,
                            # average FWHM on 95% FOV as also used in the WST_IFS_Tradeoff_Matrix. Link: https://stfc365.sharepoint.com/sites/Wide-FieldSpectroscopicTelescope/_layouts/15/DocIdRedir.aspx?ID=7ZWYDD3PV4SU-175398458-2008
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
        self.ifs['channels'] = ['zband', 'Jband', 'zJband']
        # IFS z channel
        chan = 'zband'
        self.ifs[chan] = dict(desc = self.throughput_model_desc,
                      version = self.throughput_model_version,
                              type = 'IFS',
                              iq_fwhm_tel = self.tel['iq_fwhm_ins']['ifs'], # fwhm PSF of telescope
                              iq_fwhm_ins = 0.13, # fwhm PSF of instrument, previously 0.30, updated on 03/03/2026, this probably considers also the detector (charge diffusion)
                              iq_beta = 2.80, # beta PSF of telescope + instrument (non-AO Moffat)
                              spaxel_size = 0.22, # spaxel size in arcsec ( * * * check for the binning 2x1, could be 0.125)
                              dlbda = 1.0, # Angstroem/pixel, previously 0.5, updated on 03/03/2026
                              lbda1 = 9330, # starting wavelength in Angstroem
                              lbda2 = 11300, # end wavelength in Angstroem
                              lsfpix = 2.2, # LSF in spectel, previously 3.0, updated on 03/03/2026 ( * * * check)
                              ron = 7, # readout noise (e-) # squared sum for the 2x1 binning
                              dcurrent = 0.02*3600, # dark current (e-/pixel/h) # sum for the 2x1 binning
                              )
        if not skip_dataload:
            get_data(self.ifs, chan, 'ifs', SKYDIR, rMUSEDIR)

        # IFS red channel
        chan = 'Jband'
        self.ifs[chan] = dict(desc=self.throughput_model_desc,
                       version = self.throughput_model_version,
                               type='IFS',
                               iq_fwhm_tel = self.tel['iq_fwhm_ins']['ifs'], # fwhm PSF of telescope
                               iq_fwhm_ins = 0.13, # fwhm PSF of instrument, previously 0.30, updated on 03/03/2026, this probably considers also the detector (charge diffusion)
                               iq_beta = 2.80, # beta PSF of telescope + instrument (non-AO Moffat)
                               spaxel_size = 0.25, # spaxel size in arcsec ( * * * check for the binning 2x1, could be 0.125)
                               dlbda = 1.0, # Angstroem/pixel, previously 0.67, updated on 03/03/2026
                               lbda1 = 11300, # starting wavelength in Angstroem
                               lbda2 = 13300, # end wavelength in Angstroem
                               lsfpix = 2.2, # LSF in spectel, previously 3.0, updated on 03/03/2026 ( * * * check)
                               ron = 7, # readout noise (e-) # squared sum for the 2x1 binning
                               dcurrent = 0.02*3600, # dark current (e-/pixel/h) # sum for the 2x1 binning
                               )
        if not skip_dataload:
            get_data(self.ifs, chan, 'ifs', SKYDIR, rMUSEDIR)

        #IFS z+J channel
        chan = 'zJband'
        self.ifs[chan] = dict(desc=self.throughput_model_desc,
                       version = self.throughput_model_version,
                               type='IFS',
                               iq_fwhm_tel = self.tel['iq_fwhm_ins']['ifs'], # fwhm PSF of telescope
                               iq_fwhm_ins = 0.13, # fwhm PSF of instrument, previously 0.30, updated on 03/03/2026, this probably considers also the detector (charge diffusion)
                               iq_beta = 2.80, # beta PSF of telescope + instrument (non-AO Moffat)
                               spaxel_size = 0.25, # spaxel size in arcsec ( * * * check for the binning 2x1, could be 0.125)
                               dlbda = 2.0, # Angstroem/pixel, previously 0.67, updated on 03/03/2026
                               lbda1 = 9330, # starting wavelength in Angstroem
                               lbda2 = 13300, # end wavelength in Angstroem
                               lsfpix = 2.2, # LSF in spectel, previously 3.0, updated on 03/03/2026 ( * * * check)
                               ron = 7, # readout noise (e-) # squared sum for the 2x1 binning
                               dcurrent = 0.02*3600, # dark current (e-/pixel/h) # sum for the 2x1 binning
                               )
        if not skip_dataload:
            get_data(self.ifs, chan, 'ifs', SKYDIR, rMUSEDIR)
            
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
# 0.1 initial version
# # # # # # # # # # # # # # # #

                
           
            

               
        
        

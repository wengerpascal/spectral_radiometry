##############################################################################
# modul with functions for the analysis of radiometric input data
##############################################################################
# package name: spectralradiometry
# module name: spectralradiometry.py
# convention: import spectralradiometry.spectralradiometry as sr
#
# functions needed:  none
# input data needed: visual_responsivity.txt
#                    blue_hazzard.txt
# function library:  radiometric_flux()
#                    photometric_flux()
#                    wavelength_analysis()
#                    blue_light_hazard()
##############################################################################
# Author: Pascal Wenger
# Created: 2017.09.25
# Updated: 2018.01.05
# Version: V0.6.0.18-05 
##############################################################################
# python libraries:
import os
import numpy as np
import pandas as pd
##############################################################################
# script parameters
# return current path ans system separator of this file:
separator = os.sep
current_path = os.getcwd() + separator + 'spectralradiometry'

# read V_photometric and V-scotopic from the file visual_responsivity.txt:
V_photometric = pd.read_csv(current_path+separator+'visual_responsivity.txt',
                            sep='\t', header=None, skiprows=2, usecols=[0,1])

V_scotopic = pd.read_csv(current_path+separator+'visual_responsivity.txt',
                         sep='\t', header=None, skiprows=2, usecols=[0,2])

# read Blue hazzard curve B from the file visual_blue_hazzard.txt:
B_hazzard = pd.read_csv(current_path+separator+'blue_hazzard.txt',
                         sep='\t', header=None, skiprows=2, usecols=[0,1])
##############################################################################



def radiometric_flux(wl_in, flux_in, wl_start=380.0, wl_end=780.0,
                     wl_step=0.2):
    """
    ##########################################################################
    compute the radiometric flux for a given spectral flux over a given
    ROI = wl_start ... wl_end in [nm] at step size wl_step.
    
    Input:  wl_in:    input wavelength in nm
            flux_in:  spectral flux in W/nm
            wl_start: start of region of interest (including) in nm
            wl_end:   end of region of interest (including) in nm
            wl_step:  step size in nm
    
    Output: TRF: total radiometric flux in [Watt]
            WL_int and SF_int:  spectral radiant flux in W/nm at step size
                                wl_step in [nm]
    ##########################################################################
    """
    # WL_int: wavelength in nm to consider:
    WL_int = np.arange(wl_start, wl_end+wl_step, wl_step)
    
    # SF_int: spectral radiant flux in W/nm
    SF_int = np.interp(WL_int, wl_in, flux_in)
    
    # TRF: total radiometric flux in Watt
    TRF = np.trapz(SF_int, x=WL_int)
    
    return TRF, WL_int, SF_int
    



def photometric_flux(wl_in, flux_in, V=V_photometric, wl_step=0.2):
    """
    ##########################################################################
    compute photometric flux for a given spectral flux.
    
    Input:  wl_in:   input wavelength in nm
            flux_in: spectral flux in W/nm
            V:       photometric sensitivity in [nm, -]
            wl_step: step size in nm
    
    Output: TPF: total photometric flux in [lumen]
            WL_int, SFL_int: spectral photometric flux in lumen/nm
                             at step size wl_step in [nm]
            V_int: corresponding photometric sensitivity
    ##########################################################################
    """
    # WL_int: wavelength in nm to consider:
    WL_int = np.arange(380, 780.0+wl_step, wl_step)
    
    # V_int: photopic eye sensitivity curve in steps:
    V_int = np.interp(WL_int, V.iloc[:,0], V.iloc[:,1])
    
    # SF_int: spectral radiant flux in W/nm
    SF_int = np.interp(WL_int, wl_in, flux_in)
    
    # SFL_int: spectral photometric flux in lumen/nm
    SFL_int = 683 * SF_int * V_int
    
    # TPF: total photometric flux in lumen
    TPF = np.trapz(SFL_int, x=WL_int)
    
    return TPF, WL_int, SFL_int, V_int



def wavelength_analysis(wl_in, sc_in, wl_start=380.0,
                        wl_end=780.0, wl_step=0.01):
    """
    ##########################################################################
    compute the center-wavelength (fwhm), bandwidth (fwhm), and the
    center-wavelength amplitude (fwhm) over a given spectral curve for the
    ROI = wl_start ... wl_end in [nm] at step size wl_step.
    
    Input:  wl_in:    input wavelength in nm
            sc_in:    spectral curve in a.u./nm
            wl_start: start of region of interest (including) in nm
            wl_end:   end of region of interest (including) in nm
            wl_step:  step size in nm
    
    Output: CWL_fwhm: central wavelength (fwhm) in nm
            BW_fwhm:  bandwidth (fwhm) in nm
            AMP_fwhm: center-wavelength amplitude (fwhm) in a.u.
            CWL_peak: central wavelength (peak) in nm
            AMP_peak: center-wavelength amplitude (peak) in a.u.
            WL_int and SC_int:  spectral curve in a.u./nm at step size
                                wl_step in [nm]
    ##########################################################################
    """
    # interpolate input data according to the input parameter
    # WL_int: wavelength in nm to consider:
    WL_int = np.arange(wl_start, wl_end+wl_step, wl_step)
    
    # SC_int: spectral curve in a.u./nm
    SC_int = np.interp(WL_int, wl_in, sc_in)
    
    ##########################################################################
    # compute the center-wavelength (fwhm) in nm and its amplitudein a.u.:
    
    # lower half max position with minimal difference to half max value
    delta = np.absolute(SC_int[:SC_int.argmax()+1] - SC_int.max()/2)
    
    # position of the left half max (HM)
    lower_half_max_pos = delta.argmin()
    
    # rel upper half max position with minimal difference to half max value
    delta = np.absolute(SC_int[SC_int.argmax():] - SC_int.max()/2)
    
    # position of the right half max (HM)
    rel_higher_half_max_pos = delta.argmin()
    higher_half_max_pos = SC_int.argmax() + rel_higher_half_max_pos
    
    # compute the bandwidth (fwhm) wavelength in nm:
    BW_fwhm = WL_int[higher_half_max_pos] - WL_int[lower_half_max_pos]
    
    # compute the central wavelength(fwhm) in nm:
    CWL_fwhm = BW_fwhm/2 + WL_int[lower_half_max_pos]
    
    # compute the center-wavelength amplitude (fwhm) in nm:
    if (higher_half_max_pos - lower_half_max_pos) % 2:
        # odd: position values must be integer, therfore take the mean value
        #      from the left and right amplitude
    
        Amp_left  = SC_int[lower_half_max_pos + 
                       int((higher_half_max_pos - lower_half_max_pos -1) / 2)]
        
        Amp_right = SC_int[lower_half_max_pos + 
                       int((higher_half_max_pos - lower_half_max_pos +1) / 2)]
        
        AMP_fwhm = (Amp_left + Amp_right) / 2
        
    else:
        # even
        AMP_fwhm = SC_int[lower_half_max_pos +
                          int((higher_half_max_pos - lower_half_max_pos) / 2)]
        
    ##########################################################################
    # compute the center-wavelength (peak) in nm and its ampitude in a.u.: 
    CWL_peak = WL_int[SC_int.argmax()]
    AMP_peak = SC_int.max()
    
    
    return CWL_fwhm, BW_fwhm, AMP_fwhm, CWL_peak, AMP_peak, WL_int, SC_int


def blue_light_hazard(wl_in, flux_in, wl_step=0.1, B=B_hazzard):
    """
    ##########################################################################
    compute blue light hazard ()eye safety) for a given spectral flux.
    
    Input:  wl_in:   input wavelength in nm
            flux_in: spectral flux in W/nm
            wl_step: step size in nm
            B:       blue light hazard curve in [nm, -]
    
    Output: TPF: total weighted  flux in [W]
            WL_int, SFL_int: spectral weighted  flux in [W/nm]
                             at step size wl_step in [nm]
            B_int: corresponding blue light hazard curve in [nm, -]
    ##########################################################################
    """
    # WL_int: wavelength in nm to consider:
    WL_int = np.arange(300, 700.0+wl_step, wl_step)
    
    # B_int: blue eye hazzard curve in steps:
    B_int = np.interp(WL_int, B.iloc[:,0], B.iloc[:,1])
    
    # SF_int: spectral radiant flux in W/nm
    SF_int = np.interp(WL_int, wl_in, flux_in)
    
    # SFL_int: spectral photometric flux in lumen/nm
    SFL_int = SF_int * B_int
    
    # TPF: total photometric flux in lumen
    TPF = np.trapz(SFL_int, x=WL_int)
    
    return TPF, WL_int, SFL_int, B_int
##############################################################################

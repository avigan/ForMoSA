import numpy as np
import xarray as xr
import extinction
from scipy.interpolate import interp1d
import astropy.units as u
import astropy.constants as const
from PyAstronomy.pyasl import dopplerShift, rotBroad, fastRotBroad
from adapt.extraction_functions import resolution_decreasing, convolve_and_sample
import scipy.ndimage as ndi
import scipy.signal as sg
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import time 
import scipy
# ----------------------------------------------------------------------------------------------------------------------


def lsq_fct(global_params, wave, indobs, flx_obs_spectro, err_obs_spectro, star_flx_obs, transm_obs, flx_mod_spectro, system_obs, ccf_method = 'continuum_unfiltered'):
    """
    Estimation of the contribution of the planet and of the star to a spectrum (Used for HiRISE data)

    Args:
        flx_obs_spectro  : Flux of the data (spectroscopy)
        err_obs_spectro  : Error of the data (spectroscopy)
        star_flx_obs     : Flux of star observation data (spectroscopy)
        transm_obs       : Transmission (Atmospheric + Instrumental)
        system_obs       : Systematics of the data (spectroscopy)
        flx_mod_spectro  : Flux of interpolated synthetic spectrum (spectroscopy)
        
    Returns:
        cp               : Planetary contribution to the data (Spectroscopy)
        cs               : Stellar contribution to the data (Spectroscopy)
        flx_mod_spectro  : New model of the companion 
        flx_obs_spectro  : New flux of the data
        star_flx_obs     : New star flux of the data
    """
    
    wave_final, cp_final, cs_final, flx_mod_spectro_final, flx_obs_spectro_final, star_flx_obs_final, systematics_final, flx_mod_spectro_nativ, err_obs_spectro_final = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    
    for wave_fit_i in global_params.wav_fit[indobs].split('/'):
        min_wave_i = float(wave_fit_i.split(',')[0])
        max_wave_i = float(wave_fit_i.split(',')[1])
        ind = np.where((wave <= max_wave_i) & (wave >= min_wave_i))
      
        wave_ind = wave[ind]
        flx_mod_spectro_ind = flx_mod_spectro[ind]
        transm_obs_ind = transm_obs[ind]
        star_flx_obs_ind = star_flx_obs[ind,:][0]
        star_flx_0_ind = star_flx_obs_ind[:,len(star_flx_obs_ind[0]) // 2]
        flx_obs_spectro_ind = flx_obs_spectro[ind]
        err_obs_spectro_ind = err_obs_spectro[ind]
        
        if len(system_obs) > 0:
            system_obs_ind = system_obs[ind,:][0]
            
        flx_mod_spectro_ind *= transm_obs_ind / 120
        star_flx_0_ind = star_flx_obs_ind[:,len(star_flx_obs_ind[0]) // 2]
    
        # # # # # Continuum estimation with lowpass filtering
        #
        # Low-pass filtering
        flx_obs_spectro_continuum = sg.savgol_filter(flx_obs_spectro_ind, 301, 2)
        star_flx_obs_continuum = sg.savgol_filter(star_flx_0_ind, 301, 2)
        flx_mod_spectro_continuum = sg.savgol_filter(flx_mod_spectro_ind, 301, 2)
        #
        # # # # #
            
        if ccf_method == 'continuum_filtered':
            # Removal of low-pass filtered data
            flx_obs_spectro_ind = flx_obs_spectro_ind - flx_obs_spectro_continuum + np.nanmedian(flx_obs_spectro_ind)
            star_flx_obs_ind = star_flx_obs_ind - star_flx_obs_continuum + np.nanmedian(star_flx_obs_ind)
            flx_mod_spectro_ind = flx_mod_spectro_ind - flx_mod_spectro_continuum + np.nanmedian(flx_mod_spectro_ind)
        elif ccf_method == 'continuum_unfiltered':
            flx_mod_spectro_ind = flx_mod_spectro_ind - flx_mod_spectro_continuum * star_flx_0_ind / star_flx_obs_continuum
            for i in range(len(star_flx_obs_ind[0])):
                star_flx_obs_ind[:,i] = star_flx_obs_ind[:,i] * flx_obs_spectro_continuum / star_flx_obs_continuum
                
        
        # # # # # Least squares estimation
        #    
        # Construction of the matrix
        if len(system_obs) > 0:
            A_matrix = np.zeros([np.size(flx_obs_spectro_ind), 1 + len(star_flx_obs_ind[0]) + len(system_obs_ind[0])])
            for j in range(len(system_obs[0])):
                A_matrix[:,1+len(star_flx_obs_ind[0])+j] = system_obs_ind[:,j] 
        else:
            A_matrix = np.zeros([np.size(flx_obs_spectro_ind), 1 + len(star_flx_obs_ind[0])])
            
        for j in range(len(star_flx_obs[0])):
            A_matrix[:,1+j] = star_flx_obs_ind[:,j] * 1 / np.sqrt(err_obs_spectro_ind)
            
        A_matrix[:,0] = flx_mod_spectro_ind * 1 / np.sqrt(err_obs_spectro_ind)
        
        # Least square 
        # Solve the linear system A.x = b 
        A = A_matrix
        b = flx_obs_spectro_ind * 1 / np.sqrt(err_obs_spectro_ind)
        res = optimize.lsq_linear(A_matrix, b, bounds = (0, 1))

        cp_ind = res.x[0]
        
    
        cs_ind = np.array([])
        for i in range(len(star_flx_obs[0])):
            cs_ind = np.append(cs_ind, res.x[i+1])
            
        systematics_c = np.array([])
        systematics_ind = np.asarray([])
        if len(system_obs) > 0:
            for i in range(len(system_obs[0])):
                systematics_c = np.append(systematics_c, res.x[1+len(star_flx_obs_ind[0])+i])
                
            systematics_ind = np.dot(systematics_c, system_obs_ind.T)
            
        star_flx_obs_ind = np.dot(cs_ind, star_flx_obs_ind.T)
        
        flx_mod_spectro_nativ_ind = np.copy(flx_mod_spectro_ind)
        flx_mod_spectro_ind *= cp_ind
        #
        # # # # #
        
        # Generate final products
        
        wave_final = np.append(wave_final, wave_ind)
        cp_final = np.append(cp_final, cp_ind)
        cs_final = np.append(cs_final, cs_ind)
        flx_mod_spectro_final = np.append(flx_mod_spectro_final, flx_mod_spectro_ind)
        flx_obs_spectro_final = np.append(flx_obs_spectro_final, flx_obs_spectro_ind)
        star_flx_obs_final = np.append(star_flx_obs_final, star_flx_obs_ind)
        systematics_final = np.append(systematics_final, systematics_ind)
        flx_mod_spectro_nativ = np.append(flx_mod_spectro_nativ, flx_mod_spectro_nativ_ind)
        err_obs_spectro_final = np.append(err_obs_spectro_final, err_obs_spectro_ind)
        
    return cp_final, cs_final, flx_mod_spectro_final, flx_obs_spectro_final, star_flx_obs_final, systematics_final, flx_mod_spectro_nativ, wave_final, err_obs_spectro_final


def calc_ck(flx_obs_spectro, err_obs_spectro, flx_mod_spectro, flx_obs_photo, err_obs_photo, flx_mod_photo, r_picked, d_picked,
            alpha=1, analytic='no'):
    """
    Calculation of the dilution factor Ck and re-normalization of the interpolated synthetic spectrum (from the radius
    and distance or analytically).

    Args:
        flx_obs_spectro  : Flux of the data (spectroscopy)
        err_obs_spectro  : Error of the data (spectroscopy)
        flx_mod_spectro  : Flux of the interpolated synthetic spectrum (spectroscopy)
        flx_obs_photo    : Flux of the data (photometry)
        err_obs_photo    : Error of the data (photometry)
        flx_mod_photo    : Flux of the interpolated synthetic spectrum (photometry)
        r_picked         : Radius randomly picked by the nested sampling (in RJup)
        d_picked         : Distance randomly picked by the nested sampling (in pc)
        alpha            : Manual scaling factor (set to 1 by default) such that ck = alpha * (r/d)²
        analytic         : = 'yes' if Ck needs to be calculated analytically by the formula from Cushing et al. (2008)
    Returns:
        flx_mod_spectro  : Re-normalysed model spectrum
        flx_mod_photo    : Re-normalysed model photometry
        ck               : Ck calculated

    Author: Simon Petrus
    """
    # Calculation of the dilution factor ck as a function of the radius and distance
    if analytic == 'no':
        r_picked *= 69911
        d_picked *= 3.086e+13
        ck = alpha * (r_picked/d_picked)**2
    # Calculation of the dilution factor ck analytically
    else:
        if len(flx_obs_spectro) != 0:
            ck_top_merge = np.sum((flx_mod_spectro * flx_obs_spectro) / (err_obs_spectro * err_obs_spectro))
            ck_bot_merge = np.sum((flx_mod_spectro / err_obs_spectro)**2)
        else:
            ck_top_merge = 0
            ck_bot_merge = 0
        if len(flx_obs_photo) != 0:
            ck_top_phot = np.sum((flx_mod_photo * flx_obs_photo) / (err_obs_photo * err_obs_photo))
            ck_bot_phot = np.sum((flx_mod_photo / err_obs_photo)**2)
        else:
            ck_top_phot = 0
            ck_bot_phot = 0

        ck = (ck_top_merge + ck_top_phot) / (ck_bot_merge + ck_bot_phot)

    # Re-normalization of the interpolated synthetic spectra with ck
    if len(flx_mod_spectro) != 0:
        flx_mod_spectro *= ck
    if len(flx_mod_photo) != 0:
        flx_mod_photo *= ck

    return flx_mod_spectro, flx_mod_photo, ck

# ----------------------------------------------------------------------------------------------------------------------


def doppler_fct(wav_obs_spectro, flx_obs_spectro, err_obs_spectro, flx_mod_spectro, rv_picked):
    """
    Application of a Doppler shifting to the interpolated synthetic spectrum using the function pyasl.dopplerShift.
    Note: Observation can change due to side effects of the shifting.

    Args:
        wav_obs_spectro      : Wavelength grid of the data
        flx_obs_spectro      : Flux of the data
        err_obs_spectro      : Error of the data
        flx_mod_spectro      : Flux of the interpolated synthetic spectrum
        rv_picked            : Radial velocity randomly picked by the nested sampling (in km.s-1)
    Returns:
        wav_obs_spectro      : New wavelength grid of the data
        flx_obs_spectro      : New flux of the data
        err_obs_spectro      : New error of the data
        flx_post_doppler     : New flux of the interpolated synthetic spectrum

    Author: Simon Petrus
    """
    # wav_doppler = wav_obs_spectro*10000
    # flx_post_doppler, wav_post_doppler = dopplerShift(wav_doppler, flx_mod_spectro, rv_picked)
    new_wav = wav_obs_spectro * ((rv_picked / 299792.458) + 1)
    rv_interp = interp1d(new_wav, flx_mod_spectro, fill_value="extrapolate")
    flx_post_doppler = rv_interp(wav_obs_spectro)

    return wav_obs_spectro, flx_obs_spectro, err_obs_spectro, flx_post_doppler

    # return Spectrum1d.from_array(new_wavelength, new_flux)
    # # Side effects
    # ind_nonan = np.argwhere(~np.isnan(flx_post_doppler))
    # wav_obs_spectro = wav_obs_spectro[ind_nonan[:, 0]]
    # flx_obs_spectro = flx_obs_spectro[ind_nonan[:, 0]]
    # err_obs_spectro = err_obs_spectro[ind_nonan[:, 0]]
    # flx_post_doppler = flx_post_doppler[ind_nonan[:, 0]]

    # return wav_obs_spectro, flx_obs_spectro, err_obs_spectro, flx_post_doppler

# ----------------------------------------------------------------------------------------------------------------------


def reddening_fct(wav_obs_spectro, wav_obs_photo, flx_mod_spectro, flx_mod_photo, av_picked):
    """
    Application of a sythetic interstellar extinction to the interpolated synthetic spectrum using the function
    extinction.fm07.

    Args:
        wav_obs_spectro  : Wavelength grid of the data (spectroscopy)
        wav_obs_photo    : Wavelength of the data (photometry)
        flx_mod_spectro  : Flux of the interpolated synthetic spectrum (spectroscopy)
        flx_mod_photo    : Flux of the interpolated synthetic spectrum (photometry)
        av_picked        : Extinction randomly picked by the nested sampling (in mag)
    Returns:
        flx_mod_spectro  : New flux of the interpolated synthetic spectrum (spectroscopy)
        flx_mod_photo    : New flux of the interpolated synthetic spectrum (photometry)

    Author: Simon Petrus
    """
    if len(wav_obs_spectro) != 0:
        dered_merge = extinction.fm07(wav_obs_spectro * 10000, av_picked, unit='aa')
        flx_mod_spectro *= 10**(-0.4*dered_merge)
    if len(wav_obs_photo) != 0:
        dered_phot = extinction.fm07(wav_obs_photo * 10000, av_picked, unit='aa')
        flx_mod_photo *= 10**(-0.4*dered_phot)

    return flx_mod_spectro, flx_mod_photo

# ----------------------------------------------------------------------------------------------------------------------


def vsini_fct_rot_broad(wav_obs_spectro, flx_mod_spectro, ld_picked, vsini_picked):
    """
    Application of a rotation velocity (line broadening) to the interpolated synthetic spectrum using the function
    extinction.fm07.

    Args:
        wav_obs_spectro  : Wavelength grid of the data
        flx_mod_spectro  : Flux of the interpolated synthetic spectrum
        ld_picked        : Limd darkening randomly picked by the nested sampling
        vsini_picked     : v.sin(i) randomly picked by the nested sampling (in km.s-1)
    Returns:
        flx_mod_spectro  : New flux of the interpolated synthetic spectrum

    Author: Simon Petrus
    """
    # Correct irregulatities in the wavelength grid
    wav_interval = wav_obs_spectro[1:] - wav_obs_spectro[:-1]
    wav_to_vsini = np.arange(min(wav_obs_spectro), max(wav_obs_spectro), min(wav_interval) * 2/3)
    vsini_interp = interp1d(wav_obs_spectro, flx_mod_spectro, fill_value="extrapolate")
    flx_to_vsini = vsini_interp(wav_to_vsini)
    # Apply the v.sin(i)
    new_flx = rotBroad(wav_to_vsini, flx_to_vsini, ld_picked, vsini_picked)
    vsini_interp = interp1d(wav_to_vsini, new_flx, fill_value="extrapolate")
    flx_mod_spectro = vsini_interp(wav_obs_spectro)

    return flx_mod_spectro


# ----------------------------------------------------------------------------------------------------------------------


def vsini_fct_fast_rot_broad(wav_obs_spectro, flx_mod_spectro, ld_picked, vsini_picked):
    """
    Application of a rotation velocity (line broadening) to the interpolated synthetic spectrum using the function
    extinction.fm07.

    Args:
        wav_obs_spectro  : Wavelength grid of the data
        flx_mod_spectro  : Flux of the interpolated synthetic spectrum
        ld_picked        : Limd darkening randomly picked by the nested sampling
        vsini_picked     : v.sin(i) randomly picked by the nested sampling (in km.s-1)
    Returns:
        flx_mod_spectro  : New flux of the interpolated synthetic spectrum

    Author: Simon Petrus
    """
    # Correct irregulatities in the wavelength grid
    wav_interval = wav_obs_spectro[1:] - wav_obs_spectro[:-1]
    wav_to_vsini = np.arange(min(wav_obs_spectro), max(wav_obs_spectro), min(wav_interval) * 2/3)
    vsini_interp = interp1d(wav_obs_spectro, flx_mod_spectro, fill_value="extrapolate")
    flx_to_vsini = vsini_interp(wav_to_vsini)
    # Apply the v.sin(i)
    new_flx = fastRotBroad(wav_to_vsini, flx_to_vsini, ld_picked, vsini_picked)
    vsini_interp = interp1d(wav_to_vsini, new_flx, fill_value="extrapolate")
    flx_mod_spectro = vsini_interp(wav_obs_spectro)

    return flx_mod_spectro

# ----------------------------------------------------------------------------------------------------------------------


def vsini_fct_accurate(wave_obs_merge, flx_mod_spectro, ld_picked, vsini_picked, nr=50, ntheta=100, dif=0.0):
    '''
    A routine to quickly rotationally broaden a spectrum in linear time.

    Carvalho & Johns-Krull 2023
    https://ui.adsabs.harvard.edu/abs/2023RNAAS...7...91C/abstract

    ARGS:
        wav_obs_spectro   : Wavelength grid of the data
        flx_mod_spectro   : Flux of the interpolated synthetic spectrum
        ld_picked         : Limd darkening randomly picked by the nested sampling
        vsini_picked      : v.sin(i) randomly picked by the nested sampling (in km.s-1)
    
    Returns:
        flx_mod_spectro   : New flux of the interpolated synthetic spectrum

    OPTIONAL ARGS:
        nr (default = 10)       : The number of radial bins on the projected disk
        ntheta (default = 100)  : The number of azimuthal bins in the largest radial annulus
                                note: the number of bins at each r is int(r*ntheta) where r < 1
        
        dif (default = 0)       : The differential rotation coefficient, applied according to the law
        Omeg(th)/Omeg(eq) = (1 - dif/2 - (dif/2) cos(2 th)). Dif = .675 nicely reproduces the law 
        proposed by Smith, 1994, A&A, Vol. 287, p. 523-534, to unify WTTS and CTTS. Dif = .23 is 
        similar to observed solar differential rotation. Note: the th in the above expression is 
        the stellar co-latitude, not the same as the integration variable used below. This is a 
        disk integration routine.
    '''

    ns = np.copy(flx_mod_spectro)*0.0
    tarea = 0.0
    dr = 1./nr
    for j in range(0, nr):
        r = dr/2.0 + j*dr
        area = ((r + dr/2.0)**2 - (r - dr/2.0)**2)/int(ntheta*r) * (1.0 - ld_picked + ld_picked * np.cos(np.arcsin(r)))
        for k in range(0,int(ntheta*r)):
            th = np.pi/int(ntheta*r) + k * 2.0*np.pi/int(ntheta*r)
            if dif != 0:
                vl = vsini_picked * r * np.sin(th) * (1.0 - dif/2.0 - dif/2.0*np.cos(2.0*np.arccos(r*np.cos(th))))
                ns += area * np.interp(wave_obs_merge + wave_obs_merge*vl/2.9979e5, wave_obs_merge, flx_mod_spectro)
                tarea += area
            else:
                vl = r * vsini_picked * np.sin(th)
                ns += area * np.interp(wave_obs_merge + wave_obs_merge*vl/2.9979e5, wave_obs_merge, flx_mod_spectro)
                tarea += area
    
    flx_mod_spectro = ns / tarea
    return flx_mod_spectro

# ----------------------------------------------------------------------------------------------------------------------

def bb_cpd_fct(wav_obs_spectro, wav_obs_photo, flx_mod_spectro, flx_mod_photo, distance, bb_T_picked, bb_R_picked):
    ''' Function to add the effect of a cpd (circum planetary disc) to the models
    Args:  
        wav_obs_spectro  : Wavelength grid of the data (spectroscopy)
        wav_obs_photo    : Wavelength of the data (photometry)
        flx_mod_spectro  : Flux of the interpolated synthetic spectrum (spectroscopy)
        flx_mod_photo    : Flux of the interpolated synthetic spectrum (photometry)
        bb_temp          : Temperature value randomly picked by the nested sampling in K units
        bb_rad           : Radius randomly picked by the nested sampling in units of planetary radius
    
    Returns:
        flx_mod_spectro  : New flux of the interpolated synthetic spectrum (spectroscopy)
        flx_mod_photo    : New flux of the interpolated synthetic spectrum (photometry)

    Author: P. Palma-Bifani
    '''

    bb_T_picked *= u.K
    bb_R_picked *= u.Rjup
    distance *= u.pc

    def planck(wav, T):
        a = 2.0*const.h*const.c**2
        b = const.h*const.c/(wav*const.k_B*T)
        intensity = a/ ( (wav**5) * (np.exp(b) - 1.0) )
        return intensity
    
    bb_intensity    = planck(wav_obs_spectro*u.um, bb_T_picked)
    bb_intensity_f    = planck(wav_obs_photo*u.um, bb_T_picked)

    #flux_bb_lambda   = ( np.pi * (bb_R_picked)**2 / ( ck*u.km **2) * bb_intensity ).to(u.W/u.m**2/u.micron)
    flux_bb_lambda   = ( 4*np.pi*bb_R_picked**2/(distance**2) * bb_intensity ).to(u.W/u.m**2/u.micron)

    #flux_bb_lambda_f = ( np.pi * (bb_R_picked)**2 / ( ck*u.km **2) * bb_intensity_f ).to(u.W/u.m**2/u.micron)
    flux_bb_lambda_f = ( 4*np.pi*bb_R_picked**2/(distance**2) * bb_intensity_f ).to(u.W/u.m**2/u.micron)


    # add to model flux of the atmosphere
    flx_mod_spectro  += flux_bb_lambda.value
    flx_mod_photo   += flux_bb_lambda_f.value
    # 
    return flx_mod_spectro, flx_mod_photo


# ----------------------------------------------------------------------------------------------------------------------


def reso_fct(global_params, theta, theta_index, wav_obs_spectro, flx_mod_spectro, reso_picked):
    """
    WORKING!
    Function to scale the spectral resolution of the synthetic spectra. This option is currently in test and make use
    of the functions defined in the 'adapt' section of ForMoSA, meaning that they will significantly decrease the speed of
    your inversion as the grid needs to be re-interpolated

    Args:
        global_params    : Class containing each parameter
        theta            : Parameter values randomly picked by the nested sampling
        theta_index      : Parameter index identificator
        wav_obs_spectro  : Wavelength grid of the data
        flx_mod_spectro  : Flux of the interpolated synthetic spectrum
        reso_picked      : Spectral resolution randomly picked by the nested sampling
    Returns:
        None

    Author: Matthieu Ravet
    """

    # Import the grid and set it with the right parameters
    ds = xr.open_dataset(global_params.model_path, decode_cf=False, engine="netcdf4")
    wav_mod_nativ = ds["wavelength"].values
    grid = ds['grid']
    attr = ds.attrs
    grid_np = grid.to_numpy()
    model_to_adapt = grid_np[:, theta]

    # Modify the spectrum with the wanted spectral resolution
    flx_mod_extract, mod_pho = adapt_model(global_params, wav_mod_nativ, model_to_adapt, attr['res'], obs_name=obs_name,
                                        indobs=indobs)

    return 


# ----------------------------------------------------------------------------------------------------------------------


def modif_spec(global_params, theta, theta_index,
               wav_obs_spectro, flx_obs_spectro, err_obs_spectro, flx_mod_spectro,
               wav_obs_photo, flx_obs_photo, err_obs_photo, flx_mod_photo, transm_obs = [], star_flx_obs = [], system_obs = [], indobs=0):
    """
    Modification of the interpolated synthetic spectra with the different extra-grid parameters:
        - Re-calibration on the data
        - Doppler shifting
        - Application of a substellar extinction
        - Application of a rotational velocity
        - Application of a circumplanetary disk (CPD)
    
    Args:
        global_params    : Class containing each parameter
        theta            : Parameter values randomly picked by the nested sampling
        theta_index      : Parameter index identificator
        wav_obs_spectro  : Wavelength grid of the data (spectroscopy)
        flx_obs_spectro  : Flux of the data (spectroscopy)
        err_obs_spectro  : Error of the data (spectroscopy)
        flx_mod_spectro  : Flux of the interpolated synthetic spectrum (spectroscopy)
        wav_obs_photo    : Wavelength grid of the data (photometry)
        flx_obs_photo    : Flux of the data (photometry)
        err_obs_photo    : Error of the data (photometry)
        flx_mod_photo    : Flux of the interpolated synthetic spectrum (photometry)
        transm_obs       : Transmission (Atmospheric + Instrumental)
        star_flx_obs     : Flux of star observation data (spectroscopy)
        system_obs       : Systematics of the data (spectroscopy)
        indobs      (int): Index of the current observation looping
    Returns:
        wav_obs_spectro  : New wavelength grid of the data (may change with the Doppler shift)
        flx_obs_spectro  : New flux of the data (may change with the Doppler shift)
        err_obs_spectro  : New error of the data (may change with the Doppler shift)
        flx_mod_spectro  : New flux of the interpolated synthetic spectrum (spectroscopy)
        wav_obs_photo    : Wavelength grid of the data (photometry)
        flx_obs_photo    : Flux of the data (photometry)
        err_obs_photo    : Error of the data (photometry)
        flx_mod_photo    : New flux of the interpolated synthetic spectrum (photometry)
    
    Author: Simon Petrus and Paulina Palma-Bifani
    """
    # Correction of the radial velocity of the interpolated synthetic spectrum.
    if len(flx_obs_spectro) != 0:
        if len(global_params.rv) > 3: # If you want separate rv for each observations
            if global_params.rv[indobs*3] != "NA":
                if global_params.rv[indobs*3] == "constant":
                    rv_picked = float(global_params.rv[indobs*3+1])
                else:
                    ind_theta_rv = np.where(theta_index == f'rv_{indobs}')
                    rv_picked = theta[ind_theta_rv[0][0]]
                wav_obs_spectro, flx_obs_spectro, err_obs_spectro, flx_mod_spectro = doppler_fct(wav_obs_spectro, flx_obs_spectro,
                                                                                    err_obs_spectro, flx_mod_spectro,
                                                                                    rv_picked)
        else: # If you want 1 common rv for all observations
            if global_params.rv != "NA":
                if global_params.rv[0] == "constant":
                    alpha_picked = float(global_params.rv[1])
                else:
                    ind_theta_rv = np.where(theta_index == 'rv')
                    rv_picked = theta[ind_theta_rv[0][0]]
                wav_obs_spectro, flx_obs_spectro, err_obs_spectro, flx_mod_spectro = doppler_fct(wav_obs_spectro, flx_obs_spectro,
                                                                                        err_obs_spectro, flx_mod_spectro,
                                                                                        rv_picked)
                
    # Application of a synthetic interstellar extinction to the interpolated synthetic spectrum.
    if global_params.av != "NA":
        if global_params.av[0] == 'constant':
            av_picked = float(global_params.av[1])
        else:
            ind_theta_av = np.where(theta_index == 'av')
            av_picked = theta[ind_theta_av[0][0]]
        flx_mod_spectro, flx_mod_photo = reddening_fct(wav_obs_spectro, wav_obs_photo, flx_mod_spectro, flx_mod_photo, av_picked)
        
    # Correction of the rotational velocity of the interpolated synthetic spectrum.
    if len(flx_obs_spectro) != 0:
        if len(global_params.vsini) > 4 and len(global_params.ld) > 3: # If you want separate vsini/ld for each observations
            if global_params.vsini[indobs*4] != "NA" and global_params.ld[indobs*3] != "NA":
                if global_params.vsini[indobs*4] == 'constant':
                    vsini_picked = float(global_params.vsini[indobs*3+1])
                else:
                    ind_theta_vsini = np.where(theta_index == f'vsini_{indobs}')
                    vsini_picked = theta[ind_theta_vsini[0][0]]
                if global_params.ld[indobs*3] == 'constant':
                    ld_picked = float(global_params.ld[indobs*3+1])
                else:
                    ind_theta_ld = np.where(theta_index == f'ld_{indobs}')
                    ld_picked = theta[ind_theta_ld[0][0]]

                if global_params.vsini[indobs*4 + 3] == 'RotBroad':
                    flx_mod_spectro = vsini_fct_rot_broad(wav_obs_spectro, flx_mod_spectro, ld_picked, vsini_picked)
                if global_params.vsini[indobs*4 + 3] == 'FastRotBroad':
                    flx_mod_spectro = vsini_fct_fast_rot_broad(wav_obs_spectro, flx_mod_spectro, ld_picked, vsini_picked)
                if global_params.vsini[indobs*4 + 3] == 'Accurate':
                    flx_mod_spectro = vsini_fct_accurate(wav_obs_spectro, flx_mod_spectro, ld_picked, vsini_picked)

            elif global_params.vsini[indobs*4] == "NA" and global_params.ld[indobs*3] == "NA":
                pass

            else:
                print(f'WARNING: You need to define a v.sin(i) AND a limb darkening, or set them both to NA for observation {indobs}')
                exit()

        else:# If you want 1 common vsini/ld for all observations
            if global_params.vsini != "NA" and global_params.ld != "NA":
                if global_params.vsini[0] == 'constant':
                    vsini_picked = float(global_params.vsini[1])
                else:
                    ind_theta_vsini = np.where(theta_index == 'vsini')
                    vsini_picked = theta[ind_theta_vsini[0][0]]
                if global_params.ld[0] == 'constant':
                    ld_picked = float(global_params.ld[1])
                else:
                    ind_theta_ld = np.where(theta_index == 'ld')
                    ld_picked = theta[ind_theta_ld[0][0]]

                if global_params.vsini[3] == 'RotBroad':
                    flx_mod_spectro = vsini_fct_rot_broad(wav_obs_spectro, flx_mod_spectro, ld_picked, vsini_picked)
                if global_params.vsini[3] == 'FastRotBroad':
                    flx_mod_spectro = vsini_fct_fast_rot_broad(wav_obs_spectro, flx_mod_spectro, ld_picked, vsini_picked)
                if global_params.vsini[3] == 'Accurate':
                    flx_mod_spectro = vsini_fct_accurate(wav_obs_spectro, flx_mod_spectro, ld_picked, vsini_picked)

            elif global_params.vsini == "NA" and global_params.ld == "NA":
                pass

            else:
                print('WARNING: You need to define a v.sin(i) AND a limb darkening, or set them both to NA')
                exit()  
    
    # Adding a CPD
    if global_params.bb_T != "NA" and global_params.bb_R != "NA":
        # posteriors T_eff, R_disk
        # Enter 1 or 2 bb
        if global_params.bb_T[0] == 'constant':
            bb_T_picked = float(global_params.bb_T[1])
            bb_R_picked = float(global_params.bb_R[1])
        else:
            ind_theta_bb_T = np.where(theta_index == 'bb_T')
            ind_theta_bb_R = np.where(theta_index == 'bb_R')
            bb_T_picked = theta[ind_theta_bb_T[0][0]]
            bb_R_picked = theta[ind_theta_bb_R[0][0]]
        if global_params.d[0] == "constant":
            d_picked = float(global_params.d[1])
        else:
            ind_theta_d = np.where(theta_index == 'd')
            d_picked = theta[ind_theta_d[0][0]]

        flx_mod_spectro, flx_mod_photo = bb_cpd_fct(wav_obs_spectro, wav_obs_photo, flx_mod_spectro, flx_mod_photo, d_picked, bb_T_picked, bb_R_picked)

    elif global_params.bb_T == "NA" and global_params.bb_R == "NA":
        pass

    else:
        print('WARNING: You need to define a blackbody radius and blackbody temperature, or set them to "NA"')
        exit()

    # Calculation of the dilution factor Ck and re-normalization of the interpolated synthetic spectrum.
    # From the radius and the distance.
    
    if global_params.use_lsqr[indobs] == 'True':
        planet_contribution, stellar_contribution, flx_mod_spectro, flx_obs_spectro, star_flx_obs, systematics, flx_mod_spectro_nativ, wav_obs_spectro, err_obs_spectro = lsq_fct(global_params, wav_obs_spectro, indobs, flx_obs_spectro, err_obs_spectro, star_flx_obs, transm_obs, flx_mod_spectro, system_obs)
        _, _, ck = calc_ck(np.copy(flx_obs_spectro), err_obs_spectro, np.copy(flx_mod_spectro),
                                  flx_obs_photo, err_obs_photo, flx_mod_photo, 0, 0, 0, analytic='yes')
    else:
        # Set HiRES contribution to 1 if not used
        planet_contribution, stellar_contribution, systematics = 1, 1, np.asarray([])  
        
        if global_params.r != "NA" and global_params.d != "NA":
            if global_params.r[0] == "constant":
                r_picked = float(global_params.r[1])
            else:
                ind_theta_r = np.where(theta_index == 'r')
                r_picked = theta[ind_theta_r[0][0]]
            if global_params.d[0] == "constant":
                d_picked = float(global_params.d[1])
            else:
                ind_theta_d = np.where(theta_index == 'd')
                d_picked = theta[ind_theta_d[0][0]]
    
            # With the extra alpha scaling
            if len(global_params.alpha) > 3: # If you want separate alpha for each observations
                if global_params.alpha[indobs*3] != "NA":
                    if global_params.alpha[indobs*3] == "constant":
                        alpha_picked = float(global_params.alpha[indobs*3+1])
                    else:
                        ind_theta_alpha = np.where(theta_index == f'alpha_{indobs}')
                        alpha_picked = theta[ind_theta_alpha[0][0]]
                    flx_mod_spectro, flx_mod_photo, ck = calc_ck(flx_obs_spectro, err_obs_spectro, flx_mod_spectro,
                                                            flx_obs_photo, err_obs_photo, flx_mod_photo, r_picked, d_picked,
                                                            alpha=alpha_picked)
                else: # Without the extra alpha scaling
                    flx_mod_spectro, flx_mod_photo, ck = calc_ck(flx_obs_spectro, err_obs_spectro, flx_mod_spectro,
                                                      flx_obs_photo, err_obs_photo, flx_mod_photo, r_picked, d_picked)
            else: # If you want 1 common alpha for all observations
                if global_params.alpha != "NA":
                    if global_params.alpha[0] == "constant":
                        alpha_picked = float(global_params.alpha[1])
                    else:
                        ind_theta_alpha = np.where(theta_index == 'alpha')
                        alpha_picked = theta[ind_theta_alpha[0][0]]
                    flx_mod_spectro, flx_mod_photo, ck = calc_ck(flx_obs_spectro, err_obs_spectro, flx_mod_spectro,
                                                            flx_obs_photo, err_obs_photo, flx_mod_photo, r_picked, d_picked,
                                                            alpha=alpha_picked)   
                else: # Without the extra alpha scaling
                    flx_mod_spectro, flx_mod_photo, ck = calc_ck(flx_obs_spectro, err_obs_spectro, flx_mod_spectro,
                                                        flx_obs_photo, err_obs_photo, flx_mod_photo, r_picked, d_picked)
                  
        # Analytically
        # If MOSAIC
        elif global_params.r == "NA" and global_params.d == "NA":
            flx_mod_spectro, flx_mod_photo, ck = calc_ck(flx_obs_spectro, err_obs_spectro, flx_mod_spectro,
                                                    flx_obs_photo, err_obs_photo, flx_mod_photo, 0, 0, 0,
                                                    analytic='yes')
                
    
        else:   # either global_params.r or global_params.d is set to 'NA' 
            print('WARNING: You need to define a radius AND a distance, or set them both to "NA"')
            exit()

    return wav_obs_spectro, flx_obs_spectro, err_obs_spectro, flx_mod_spectro, wav_obs_photo, flx_obs_photo, err_obs_photo, flx_mod_photo, ck, planet_contribution, stellar_contribution, star_flx_obs, systematics, transm_obs










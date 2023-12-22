import numpy as np
import os
import glob
import xarray as xr

from nested_sampling.nested_modif_spec import modif_spec
from nested_sampling.nested_prior_function import uniform_prior, gaussian_prior
from nested_sampling.nested_logL_functions import logL_chi2_classic, logL_chi2_covariance, logL_CCF_Brogi, logL_CCF_Lockwood, logL_CCF_custom
from main_utilities import yesno, diag_mat


def MOSAIC_logL(theta, theta_index, global_params):
    """
    

    Args:
        
    Returns:
        

    Authors: Simon Petrus and Matthieu Ravet
    """
    
    # Recovery of each observation spectroscopy and photometry data

    main_obs_path = global_params.main_observation_path
    FINAL_logL = 0

    for indobs, obs in enumerate(sorted(glob.glob(main_obs_path))):
        
        global_params.observation_path = obs
        obs_name = os.path.splitext(os.path.basename(global_params.observation_path))[0]
        spectrum_obs = np.load(os.path.join(global_params.result_path, f'spectrum_obs_{obs_name}.npz'), allow_pickle=True)

        wav_obs_merge = spectrum_obs['obs_merge'][0]
        flx_obs_merge = spectrum_obs['obs_merge'][1]
        err_obs_merge = spectrum_obs['obs_merge'][2]
        inv_cov_obs_merge = spectrum_obs['inv_cov_obs']
        #print(inv_cov_obs_merge)

        if 'obs_pho' in spectrum_obs.keys():
            wav_obs_phot = np.asarray(spectrum_obs['obs_pho'][0])
            flx_obs_phot = np.asarray(spectrum_obs['obs_pho'][1])
            err_obs_phot = np.asarray(spectrum_obs['obs_pho'][2])
        else:
            wav_obs_phot = np.asarray([])
            flx_obs_phot = np.asarray([])
            err_obs_phot = np.asarray([])

        # Recovery of the spectroscopy and photometry model
        path_grid_m = os.path.join(global_params.adapt_store_path, f'adapted_grid_merge_{global_params.grid_name}_{obs_name}_nonan.nc')
        path_grid_p = os.path.join(global_params.adapt_store_path, f'adapted_grid_phot_{global_params.grid_name}_{obs_name}_nonan.nc')
        ds = xr.open_dataset(path_grid_m, decode_cf=False, engine='netcdf4')
        grid_merge = ds['grid']
        ds.close()
        ds = xr.open_dataset(path_grid_p, decode_cf=False, engine='netcdf4')
        grid_phot = ds['grid']
        ds.close()

        # Calculation of the likelihood for each sub-spectrum defined by the parameter 'wav_fit'
        for ns_u_ind, ns_u in enumerate(global_params.wav_fit.split('/')):
            
            min_ns_u = float(ns_u.split(',')[0])
            max_ns_u = float(ns_u.split(',')[1])
            ind_grid_merge_sel = np.where((grid_merge['wavelength'] >= min_ns_u) & (grid_merge['wavelength'] <= max_ns_u))
            ind_grid_phot_sel = np.where((grid_phot['wavelength'] >= min_ns_u) & (grid_phot['wavelength'] <= max_ns_u))

            # Cutting of the grid on the wavelength grid defined by the parameter 'wav_fit'
            grid_merge_cut = grid_merge.sel(wavelength=grid_merge['wavelength'][ind_grid_merge_sel])
            grid_phot_cut = grid_phot.sel(wavelength=grid_phot['wavelength'][ind_grid_phot_sel])

            # Interpolation of the grid at the theta parameters set
            if global_params.par3 == 'NA':
                if len(grid_merge_cut['wavelength']) != 0:
                    flx_mod_merge_cut = grid_merge_cut.interp(par1=theta[0], par2=theta[1],
                                                            method="linear", kwargs={"fill_value": "extrapolate"})
                else:
                    flx_mod_merge_cut = []
                if len(grid_phot_cut['wavelength']) != 0:
                    flx_mod_phot_cut = grid_phot_cut.interp(par1=theta[0], par2=theta[1],
                                                            method="linear", kwargs={"fill_value": "extrapolate"})
                else:
                    flx_mod_phot_cut = []
            elif global_params.par4 == 'NA':
                if len(grid_merge_cut['wavelength']) != 0:
                    flx_mod_merge_cut = grid_merge_cut.interp(par1=theta[0], par2=theta[1], par3=theta[2],
                                                            method="linear", kwargs={"fill_value": "extrapolate"})
                else:
                    flx_mod_merge_cut = []
                if len(grid_phot_cut['wavelength']) != 0:
                    flx_mod_phot_cut = grid_phot_cut.interp(par1=theta[0], par2=theta[1], par3=theta[2],
                                                            method="linear", kwargs={"fill_value": "extrapolate"})
                else:
                    flx_mod_phot_cut = []
            elif global_params.par5 == 'NA':
                if len(grid_merge_cut['wavelength']) != 0:
                    flx_mod_merge_cut = grid_merge_cut.interp(par1=theta[0], par2=theta[1], par3=theta[2], par4=theta[3],
                                                            method="linear", kwargs={"fill_value": "extrapolate"})
                else:
                    flx_mod_merge_cut = []
                if len(grid_phot_cut['wavelength']) != 0:
                    flx_mod_phot_cut = grid_phot_cut.interp(par1=theta[0], par2=theta[1], par3=theta[2], par4=theta[3],
                                                            method="linear", kwargs={"fill_value": "extrapolate"})
                else:
                    flx_mod_phot_cut = []
            else:
                if len(grid_merge_cut['wavelength']) != 0:
                    flx_mod_merge_cut = grid_merge_cut.interp(par1=theta[0], par2=theta[1], par3=theta[2], par4=theta[3],
                                                            par5=theta[4],
                                                            method="linear", kwargs={"fill_value": "extrapolate"})
                else:
                    flx_mod_merge_cut = []
                if len(grid_phot_cut['wavelength']) != 0:
                    flx_mod_phot_cut = grid_phot_cut.interp(par1=theta[0], par2=theta[1], par3=theta[2], par4=theta[3],
                                                            par5=theta[4],
                                                            method="linear", kwargs={"fill_value": "extrapolate"})
                else:
                    flx_mod_phot_cut = []


            # Re-merging of the data and interpolated synthetic spectrum to a wavelength grid defined by the parameter 'wav_fit'
            ind_merge = np.where((wav_obs_merge >= min_ns_u) & (wav_obs_merge <= max_ns_u))
            ind_phot = np.where((wav_obs_phot >= min_ns_u) & (wav_obs_phot <= max_ns_u))
            if ns_u_ind == 0:
                wav_obs_merge_ns_u = wav_obs_merge[ind_merge]
                flx_obs_merge_ns_u = flx_obs_merge[ind_merge]
                err_obs_merge_ns_u = err_obs_merge[ind_merge]
                flx_mod_merge_ns_u = flx_mod_merge_cut
                wav_obs_phot_ns_u = wav_obs_phot[ind_phot]
                flx_obs_phot_ns_u = flx_obs_phot[ind_phot]
                err_obs_phot_ns_u = err_obs_phot[ind_phot]
                flx_mod_phot_ns_u = flx_mod_phot_cut
                if inv_cov_obs_merge != []:  # Add covariance in the loop (if necessary)
                    inv_cov_obs_merge_ns_u = inv_cov_obs_merge[np.ix_(ind_merge[0],ind_merge[0])]
                else:
                    inv_cov_obs_merge_ns_u = []
            else:
                wav_obs_merge_ns_u = np.concatenate((wav_obs_merge_ns_u, wav_obs_merge[ind_merge]))
                flx_obs_merge_ns_u = np.concatenate((flx_obs_merge_ns_u, flx_obs_merge[ind_merge]))
                err_obs_merge_ns_u = np.concatenate((err_obs_merge_ns_u, err_obs_merge[ind_merge]))
                flx_mod_merge_ns_u = np.concatenate((flx_mod_merge_ns_u, flx_mod_merge_cut))
                wav_obs_phot_ns_u = np.concatenate((wav_obs_phot_ns_u, wav_obs_phot[ind_phot]))
                flx_obs_phot_ns_u = np.concatenate((flx_obs_phot_ns_u, flx_obs_phot[ind_phot]))
                err_obs_phot_ns_u = np.concatenate((err_obs_phot_ns_u, err_obs_phot[ind_phot]))
                flx_mod_phot_ns_u = np.concatenate((flx_mod_phot_ns_u, flx_mod_phot_cut))
                if inv_cov_obs_merge_ns_u != []: # Merge the covariance matrices (if necessary)
                    inv_cov_obs_merge_ns_u = diag_mat([inv_cov_obs_merge_ns_u, inv_cov_obs_merge[np.ix_(ind_merge[0],ind_merge[0])]]) 
                    
        # Modification of the synthetic spectrum with the extra-grid parameters
        modif_spec_LL = modif_spec(global_params, theta, theta_index,
                                    wav_obs_merge_ns_u,  flx_obs_merge_ns_u,  err_obs_merge_ns_u,  flx_mod_merge_ns_u,
                                    wav_obs_phot_ns_u,  flx_obs_phot_ns_u, err_obs_phot_ns_u,  flx_mod_phot_ns_u)
        
        flx_obs, flx_obs_phot = modif_spec_LL[1], modif_spec_LL[5]
        flx_mod, flx_mod_phot = modif_spec_LL[3], modif_spec_LL[7]
        err, err_phot = modif_spec_LL[2], modif_spec_LL[6]
        inv_cov = inv_cov_obs_merge_ns_u
        ck = modif_spec_LL[8]

        # Computation of the photometry logL
        if err_phot != []:
            logL_phot = logL_chi2_classic(flx_obs_phot-flx_mod_phot, err_phot)
        else:
            logL_phot = 0

        # Computation of the spectroscopy logL
        if global_params.logL_type[indobs] == 'chi2_classic':
            logL_spec = logL_chi2_classic(flx_obs-flx_mod, err)
        if global_params.logL_type[indobs] == 'chi2_covariance' and inv_cov != []:
            logL_spec = logL_chi2_covariance(flx_obs-flx_mod, inv_cov)
        if global_params.logL_type[indobs] == 'CCF_Brogi':
            logL_spec = logL_CCF_Brogi(flx_obs, flx_mod)
        if global_params.logL_type[indobs] == 'CCF_Lockwood':
            logL_spec = logL_CCF_Lockwood(flx_obs, flx_mod)
        if global_params.logL_type[indobs] == 'CCF_custom':
            logL_spec = logL_CCF_custom(flx_obs, flx_mod, err)

        FINAL_logL = logL_phot + logL_spec

    return FINAL_logL
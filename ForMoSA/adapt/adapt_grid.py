from __future__ import print_function, division
import numpy as np
import xarray as xr
import time
import os, sys

from tqdm import tqdm

sys.path.insert(0, os.path.abspath('../'))

from adapt.extraction_functions import adapt_model, decoupe


# ----------------------------------------------------------------------------------------------------------------------


def adapt_grid(global_params, wav_obs_spectro, wav_obs_photo, res_mod_obs_merge, obs_name='', indobs=0):
    """
    Adapt the synthetic spectra of a grid to make them comparable with the data.

    Args:
        global_params    (object): Class containing each parameter
        wav_obs_spectro   (array): Merged wavelength grid of the data
        wav_obs_photo     (array): Wavelengths of the photometry points
        obs_name            (str): Name of the current observation looping
        indobs              (int): Index of the current observation looping
    Returns:
        None

    Author: Simon Petrus, Matthieu Ravet and Paulina Palma-Bifani
    """

    ds = xr.open_dataset(global_params.model_path, decode_cf=False, engine="netcdf4")
    wav_mod_nativ = ds["wavelength"].values
    grid = ds['grid']
    attr = ds.attrs
    grid_np = grid.to_numpy()

    # create arrays without any assumptions on the number of parameters
    shape_spectro = [len(wav_obs_spectro)]
    shape_photo = [len(wav_obs_photo)]
    for key in attr['key']:
        shape_spectro.append(len(grid[key].values))
        shape_photo.append(len(grid[key].values))
    grid_spectro_np = np.full(shape_spectro, np.nan)
    grid_photo_np   = np.full(shape_photo, np.nan)

    print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')

    shape = grid_np.shape[1:]
    pbar = tqdm(total=np.prod(shape), leave=False)
    for idx in np.ndindex(shape):
        pbar.update()

        model_to_adapt = grid_np[(..., ) + idx]
        nan_mod = np.isnan(model_to_adapt)
        if np.any(nan_mod):
            msg = 'Extraction of model failed : '
            for i, (key, title) in enumerate(zip(attr['key'], attr['title'])):
                msg += f'{title}={grid[key].values[i]}, '
            print(msg)
        else:
            mod_spectro, mod_photo = adapt_model(global_params, wav_mod_nativ, model_to_adapt,
                                                 res_mod_obs_merge, obs_name=obs_name, indobs=indobs)
            grid_spectro_np[(..., ) + idx] = mod_spectro
            grid_photo_np[(..., ) + idx] = mod_photo

    # create final datasets
    vars = ["wavelength"]
    for key in attr['key']:
        vars.append(key)

    coords_spectro = {"wavelength": wav_obs_spectro}
    coords_photo   = {"wavelength": wav_obs_photo}
    for key in attr['key']:
        coords_spectro[key] = grid[key].values
        coords_photo[key]   = grid[key].values

    ds_spectro_new = xr.Dataset(data_vars=dict(grid=(vars, grid_spectro_np)), coords=coords_spectro, attrs=attr)
    ds_photo_new   = xr.Dataset(data_vars=dict(grid=(vars, grid_photo_np)), coords=coords_photo, attrs=attr)

    print()
    print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
    print('-> The possible holes in the grid are interpolated: ')
    print()
    for key_ind, key in enumerate(attr['key']):
        print(str(key_ind+1) + '/' + str(len(attr['key'])))
        ds_spectro_new = ds_spectro_new.interpolate_na(dim=key, method="linear", fill_value="extrapolate", limit=None, max_gap=None)
        ds_photo_new = ds_photo_new.interpolate_na(dim=key, method="linear", fill_value="extrapolate", limit=None, max_gap=None)

    ds_spectro_new.to_netcdf(os.path.join(global_params.adapt_store_path, f'adapted_grid_spectro_{global_params.grid_name}_{obs_name}_nonan.nc'),
                             format='NETCDF4',
                             engine='netcdf4',
                             mode='w')
    ds_photo_new.to_netcdf(os.path.join(global_params.adapt_store_path, f'adapted_grid_photo_{global_params.grid_name}_{obs_name}_nonan.nc'),
                           format='NETCDF4',
                           engine='netcdf4',
                           mode='w')

    print('The possible holes have been interpolated!')

    return None

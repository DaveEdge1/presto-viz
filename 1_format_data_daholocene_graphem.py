#==============================================================================
# Make a standardized netCDF file for CFR/LMR2 Reconstruction
# Enhanced version supporting multiple reconstruction types
#    author: Michael Erb (adapted for CFR support)
#    date  : 11/5/2025
#==============================================================================

import sys
import numpy as np
import xarray as xr
import functions_presto
import yaml
import glob
import os


def _is_temp12k_nc(f):
    """Temperature 12k composite NetCDFs carry tas + tas_gm and a temp12k title/source."""
    try:
        with xr.open_dataset(f) as dd:
            if 'tas_gm' not in dd.variables:
                return False
            txt = (str(dd.attrs.get('title', '')) + ' ' + str(dd.attrs.get('source', ''))).lower()
            return 'temp12k' in txt or 'temperature 12k' in txt
    except Exception:
        return False


# Set directories
data_dir = sys.argv[1]

# Ensure data_dir ends with /
if not data_dir.endswith('/'):
    data_dir += '/'

#%% DETECT RECONSTRUCTION TYPE

print(f"=== Analyzing data directory: {data_dir} ===")

# Auto-detect reconstruction type based on files present
nc_files = glob.glob(data_dir + '*.nc')
print(f"Found {len(nc_files)} NetCDF files:")
for f in nc_files:
    print(f"  - {os.path.basename(f)}")

# Determine dataset type
if 'holocene_da' in data_dir or any('holocene_recon' in f for f in nc_files):
    dataset_txt = 'daholocene'
    version_txt = data_dir.rstrip('/').split('/')[-1]
    print(f"Detected: Holocene DA reconstruction")
elif 'graph_em' in data_dir or any('graphem' in f.lower() for f in nc_files):
    dataset_txt = 'graphem'
    version_txt = data_dir.rstrip('/').split('/')[-1]
    print(f"Detected: GraphEM reconstruction")
elif any('test-run' in f for f in nc_files) or os.path.exists(data_dir + 'test-run-graphem-cfg/'):
    dataset_txt = 'graphem'
    version_txt = data_dir.rstrip('/').split('/')[-1]
    print(f"Detected: GraphEM reconstruction (via test-run)")
elif any(_is_temp12k_nc(f) for f in nc_files):
    dataset_txt = 'temp12k'
    version_txt = data_dir.rstrip('/').split('/')[-1]
    print(f"Detected: Temperature 12k zonal composite")
else:
    # Assume CFR/LMR2 format - treat as GraphEM-like
    dataset_txt = 'lmr'
    version_txt = data_dir.rstrip('/').split('/')[-1]
    print(f"Detected: CFR/LMR2 reconstruction (treating as generic format)")

#%% PROCESS DATA

var_txt      = 'tas'
quantity_txt = 'Annual'
filename_txt = dataset_txt+'_v'+version_txt+'_'+var_txt+'_'+quantity_txt.lower()
print(f' ===== STARTING script 1: Reformatting data for {filename_txt} =====')

if dataset_txt == 'lmr':
    #
    ### LOAD CFR/LMR2 DATA (individual seed runs presented as separate methods)
    #
    print('=== Processing CFR/LMR2 Reconstruction ===')

    # Load up to 3 individual seed files: job_r01_recon.nc, job_r02_recon.nc, ...
    seed_files = sorted(glob.glob(os.path.join(data_dir, 'job_r*_recon.nc')))[:3]
    if not seed_files:
        raise FileNotFoundError(
            f'No job_r*_recon.nc files found in {data_dir}. '
            'Re-run the reconstruction to regenerate them.')

    print(f'Found {len(seed_files)} seed file(s): {[os.path.basename(f) for f in seed_files]}')
    methods = [f'Run {i+1}' for i in range(len(seed_files))]

    spatial_list = []
    global_list  = []

    for i, sf in enumerate(seed_files):
        ds = xr.open_dataset(sf)
        print(f'  {methods[i]} ({os.path.basename(sf)}): '
              f'tas={dict(ds["tas"].sizes)}, tas_gm={dict(ds["tas_gm"].sizes)}')

        # tas: (time, lat, lon) — ensemble-mean spatial field for this seed
        spatial_list.append(ds['tas'].values[np.newaxis, :, :, :])  # (1, time, lat, lon)

        # tas_gm: (time, ens) → (ens, time)
        global_list.append(ds['tas_gm'].values.T)  # (ens, time)

        if i == 0:
            lat        = ds['lat'].values
            lon        = ds['lon'].values
            time_coord = ds['time'].values
            age        = 1950 - np.array(time_coord, dtype=float)

        ds.close()

    # Stack across methods → (n_methods, ens, time, ...)
    var_spatial_members = np.stack(spatial_list, axis=0)  # (n_methods, 1, time, lat, lon)
    var_global_members  = np.stack(global_list,  axis=0)  # (n_methods, ens, time)

    var_spatial_mean = np.mean(var_spatial_members, axis=1)  # (n_methods, time, lat, lon)
    var_global_mean  = np.mean(var_global_members,  axis=1)  # (n_methods, time)

    lat_bounds, lon_bounds = functions_presto.bounding_latlon(lat, lon)

    ens_spatial = np.arange(var_spatial_members.shape[1]) + 1
    ens_global  = np.arange(var_global_members.shape[1])  + 1
    notes       = ['Processed from CFR/LMR2 output']

    # Config options
    config_file = os.path.join(data_dir, 'lmr_configs.yml')
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            options = yaml.load(f, Loader=yaml.FullLoader)
        options_list = [f'{k}: {v}' for k, v in options.items()]
    else:
        print('Note: no lmr_configs.yml in artifact; config options will be blank.')
        options_list = ['No configuration file found']

    print('Variable shapes:')
    print(f'  var_spatial_members: {var_spatial_members.shape}')
    print(f'  var_spatial_mean:    {var_spatial_mean.shape}')
    print(f'  var_global_members:  {var_global_members.shape}')
    print(f'  var_global_mean:     {var_global_mean.shape}')

    data_xarray_output = xr.Dataset(
        {
            'tas_global_mean':    (['method','age'],                          var_global_mean,    {'units':'degrees Celsius'}),
            'tas_global_members': (['method','ens_global','age'],             var_global_members, {'units':'degrees Celsius'}),
            'tas_spatial_mean':   (['method','age','lat','lon'],              var_spatial_mean,   {'units':'degrees Celsius'}),
            'tas_spatial_members':(['method','ens_spatial','age','lat','lon'],var_spatial_members,{'units':'degrees Celsius'})
        },
        coords={
            'method':     (['method'],methods),
            'notes':      (['notes'],notes),
            'options':    (['options'],options_list),
            'ens_global': (['ens_global'],ens_global,{'description':'ensemble members'}),
            'ens_spatial':(['ens_spatial'],ens_spatial,{'description':'ensemble members'}),
            'age':        (['age'],age,{'units':'yr BP'}),
            'lat':        (['lat'],lat,{'units':'degrees_north'}),
            'lon':        (['lon'],lon,{'units':'degrees_east'}),
            'lat_bounds': (['lat_bounds'],lat_bounds,{'units':'degrees_north'}),
            'lon_bounds': (['lon_bounds'],lon_bounds,{'units':'degrees_east'}),
        },
        attrs={
            'dataset_name':      'CFR/LMR2 Reconstruction',
            'dataset_source_url':'https://github.com/DaveEdge1/LMR2',
        },
    )

    output_file = os.path.join(data_dir, filename_txt + '.nc')
    data_xarray_output.to_netcdf(output_file)
    print(f' ===== FINISHED script 1: Data reformatted and saved to: {output_file} =====')

elif dataset_txt == 'daholocene':
    # [Keep original daholocene code - lines 33-118 from original file]
    print('=== Processing Holocene Reconstruction ===')
    data_filename = glob.glob(data_dir+'holocene_recon*.nc')[0]
    data_xarray = xr.open_dataset(data_filename)
    var_global_members  = data_xarray['recon_tas_global_mean'].values
    var_spatial_mean    = data_xarray['recon_tas_mean'].values
    var_spatial_members = data_xarray['recon_tas_ens'].values
    age = data_xarray['ages'].values
    lat = data_xarray['lat'].values
    lon = data_xarray['lon'].values

    with open(data_dir+'configs.yml','r') as file:
        options = yaml.load(file,Loader=yaml.FullLoader)

    options_list = []
    for key1 in options.keys():
        for key2 in options[key1].keys():
            option_txt = key1+'/'+key2+': '+str(options[key1][key2]['value'])
            options_list.append(option_txt)

    lat_bounds,lon_bounds = functions_presto.bounding_latlon(lat,lon)

    var_spatial_mean    = np.expand_dims(var_spatial_mean,axis=0)
    var_spatial_members = np.expand_dims(np.swapaxes(var_spatial_members,0,1),axis=0)
    var_global_members  = np.expand_dims(np.swapaxes(var_global_members,0,1),axis=0)
    var_global_mean     = np.mean(var_global_members,axis=1)

    methods = ['Holocene Reconstruction']
    ens_spatial = np.arange(var_spatial_members.shape[1])+1
    ens_global  = np.arange(var_global_members.shape[1])+1
    notes = ['']

    print(var_spatial_members.shape)
    print(var_spatial_mean.shape)
    print(var_global_members.shape)
    print(var_global_mean.shape)

    data_xarray_output = xr.Dataset(
        {
            'tas_global_mean':    (['method','age'],                          var_global_mean,    {'units':'degrees Celsius'}),
            'tas_global_members': (['method','ens_global','age'],             var_global_members, {'units':'degrees Celsius'}),
            'tas_spatial_mean':   (['method','age','lat','lon'],              var_spatial_mean,   {'units':'degrees Celsius'}),
            'tas_spatial_members':(['method','ens_spatial','age','lat','lon'],var_spatial_members,{'units':'degrees Celsius'})
        },
        coords={
            'method':     (['method'],methods),
            'notes':      (['notes'],notes),
            'options':    (['options'],options_list),
            'ens_global': (['ens_global'],ens_global,{'description':'ensemble members'}),
            'ens_spatial':(['ens_spatial'],ens_spatial,{'description':'selected ensemble members'}),
            'age':        (['age'],age,{'units':'yr BP'}),
            'lat':        (['lat'],lat,{'units':'degrees_north'}),
            'lon':        (['lon'],lon,{'units':'degrees_east'}),
            'lat_bounds': (['lat_bounds'],lat_bounds,{'units':'degrees_north'}),
            'lon_bounds': (['lon_bounds'],lon_bounds,{'units':'degrees_east'}),
        },
        attrs={
            'dataset_name':      'Holocene Reconstruction',
            'dataset_source_url':'https://paleopresto.com/custom.html',
        },
    )

    data_xarray_output.to_netcdf(data_dir+filename_txt+'.nc')
    print(' ===== FINISHED script 1: Data reformatted and saved to: '+data_dir+filename_txt+'.nc =====')

elif dataset_txt == 'graphem':
    # [Keep original graphem code - lines 119-217 from original file]
    print('=== Processing GraphEM reconstruction ===')
    data_filename = glob.glob(data_dir+'test-run-graphem-cfg/'+'*recon.nc')[0]
    data_xarray = xr.open_dataset(data_filename)

    with open(data_dir+'configs.yml','r') as file:
        options = yaml.load(file,Loader=yaml.FullLoader)

    year        = data_xarray['time'].values
    lat         = data_xarray['lat'].values
    lon         = data_xarray['lon'].values
    ens_spatial = data_xarray['ens'].values
    ens_global  = ens_spatial
    age = 1950-year

    methods = ['GraphEM']
    n_methods = len(methods)
    n_ens     = len(ens_spatial)
    n_ages    = len(age)
    n_lat     = len(lat)
    n_lon     = len(lon)

    var_spatial_members = np.zeros((n_methods,n_ens,n_ages,n_lat,n_lon)); var_spatial_members[:] = np.nan
    var_spatial_members[0,0,:,:,:] = data_xarray['tas'].values

    var_global_members = np.zeros((n_methods,n_ens,n_ages)); var_global_members[:] = np.nan
    var_global_members[0,:,:] = np.swapaxes(data_xarray['tas_gm'].values,0,1)

    options_list = []
    for key1 in options.keys():
        for key2 in options[key1].keys():
            option_txt = key1+'/'+key2+': '+str(options[key1][key2]['value'])
            options_list.append(option_txt)

    lat_bounds,lon_bounds = functions_presto.bounding_latlon(lat,lon)

    var_spatial_mean = np.mean(var_spatial_members,axis=1)
    var_global_mean  = np.mean(var_global_members,axis=1)

    notes = ['']

    print(var_spatial_members.shape)
    print(var_spatial_mean.shape)
    print(var_global_members.shape)
    print(var_global_mean.shape)

    data_xarray_output = xr.Dataset(
        {
            'tas_global_mean':    (['method','age'],                          var_global_mean,    {'units':'degrees Celsius'}),
            'tas_global_members': (['method','ens_global','age'],             var_global_members, {'units':'degrees Celsius'}),
            'tas_spatial_mean':   (['method','age','lat','lon'],              var_spatial_mean,   {'units':'degrees Celsius'}),
            'tas_spatial_members':(['method','ens_spatial','age','lat','lon'],var_spatial_members,{'units':'degrees Celsius'})
        },
        coords={
            'method':     (['method'],methods),
            'notes':      (['notes'],notes),
            'options':    (['options'],options_list),
            'ens_global': (['ens_global'],ens_global,{'description':'ensemble members'}),
            'ens_spatial':(['ens_spatial'],ens_spatial,{'description':'ensemble members'}),
            'age':        (['age'],age,{'units':'yr BP'}),
            'lat':        (['lat'],lat,{'units':'degrees_north'}),
            'lon':        (['lon'],lon,{'units':'degrees_east'}),
            'lat_bounds': (['lat_bounds'],lat_bounds,{'units':'degrees_north'}),
            'lon_bounds': (['lon_bounds'],lon_bounds,{'units':'degrees_east'}),
        },
        attrs={
            'dataset_name':      'GraphEM',
            'dataset_source_url':'https://paleopresto.com/custom.html',
        },
    )

    data_xarray_output.to_netcdf(data_dir+filename_txt+'.nc')
    print(' ===== FINISHED script 1: Data reformatted and saved to: '+data_dir+filename_txt+'.nc =====')

elif dataset_txt == 'temp12k':
    # Temperature 12k zonal composite: a single NetCDF with tas (time,lat,lon)
    # latitudinal-stripe field + tas_gm (time,ens) global-mean ensemble. Same
    # tas/tas_gm contract as GraphEM, so the handling mirrors that branch.
    print('=== Processing Temperature 12k zonal composite ===')
    data_filename = next(f for f in nc_files if _is_temp12k_nc(f))
    data_xarray = xr.open_dataset(data_filename)

    year        = data_xarray['time'].values
    lat         = data_xarray['lat'].values
    lon         = data_xarray['lon'].values
    ens_global  = data_xarray['ens'].values
    age = 1950 - np.array(year, dtype=float)

    methods   = ['Temperature 12k Composite']
    n_methods = len(methods)
    n_ages    = len(age)
    n_lat     = len(lat)
    n_lon     = len(lon)
    n_ens_g   = len(ens_global)

    # spatial: the zonal-stripe field is a single (median) member
    var_spatial_members = np.full((n_methods, 1, n_ages, n_lat, n_lon), np.nan)
    var_spatial_members[0, 0, :, :, :] = data_xarray['tas'].values

    # global: full multi-method ensemble (time, ens) -> (ens, age)
    var_global_members = np.full((n_methods, n_ens_g, n_ages), np.nan)
    var_global_members[0, :, :] = np.swapaxes(data_xarray['tas_gm'].values, 0, 1)

    options_list = []
    cfg_path = os.path.join(data_dir, 'configs.yml')
    if os.path.exists(cfg_path):
        with open(cfg_path, 'r') as file:
            options = yaml.load(file, Loader=yaml.FullLoader) or {}
        for key1 in options.keys():
            for key2 in options[key1].keys():
                options_list.append(key1 + '/' + key2 + ': ' + str(options[key1][key2]['value']))
    if not options_list:
        options_list = ['Temperature 12k multi-method composite (SCC/DCC/GAM/CPS/PaiCo)']

    lat_bounds, lon_bounds = functions_presto.bounding_latlon(lat, lon)
    var_spatial_mean = np.mean(var_spatial_members, axis=1)
    var_global_mean  = np.mean(var_global_members, axis=1)
    ens_spatial = np.arange(var_spatial_members.shape[1]) + 1
    notes = ['Zonal 30-degree band reconstruction shown as latitudinal stripes']

    data_xarray_output = xr.Dataset(
        {
            'tas_global_mean':    (['method','age'],                          var_global_mean,    {'units':'degrees Celsius'}),
            'tas_global_members': (['method','ens_global','age'],             var_global_members, {'units':'degrees Celsius'}),
            'tas_spatial_mean':   (['method','age','lat','lon'],              var_spatial_mean,   {'units':'degrees Celsius'}),
            'tas_spatial_members':(['method','ens_spatial','age','lat','lon'],var_spatial_members,{'units':'degrees Celsius'})
        },
        coords={
            'method':     (['method'],methods),
            'notes':      (['notes'],notes),
            'options':    (['options'],options_list),
            'ens_global': (['ens_global'],ens_global,{'description':'multi-method ensemble members'}),
            'ens_spatial':(['ens_spatial'],ens_spatial,{'description':'zonal-stripe field (median)'}),
            'age':        (['age'],age,{'units':'yr BP'}),
            'lat':        (['lat'],lat,{'units':'degrees_north'}),
            'lon':        (['lon'],lon,{'units':'degrees_east'}),
            'lat_bounds': (['lat_bounds'],lat_bounds,{'units':'degrees_north'}),
            'lon_bounds': (['lon_bounds'],lon_bounds,{'units':'degrees_east'}),
        },
        attrs={
            'dataset_name':      'Temperature 12k Composite',
            'dataset_source_url':'https://github.com/DaveEdge1/presto-Temp12k_Composites',
        },
    )

    data_xarray_output.to_netcdf(data_dir+filename_txt+'.nc')
    print(' ===== FINISHED script 1: Data reformatted and saved to: '+data_dir+filename_txt+'.nc =====')

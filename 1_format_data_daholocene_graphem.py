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
    ### LOAD CFR/LMR2 DATA  (combined across all seed runs)
    #
    print('=== Processing CFR/LMR2 Reconstruction ===')

    combined_path = os.path.join(data_dir, 'combined_recon.nc')
    if not os.path.exists(combined_path):
        raise FileNotFoundError(
            f'combined_recon.nc not found in {data_dir}. '
            'Re-run the reconstruction to regenerate it with the latest cfr_main_code.py.')

    data = xr.open_dataset(combined_path)
    print(f'Loaded combined_recon.nc: {dict(data.dims)}')
    print(f'  tas shape:    {dict(data["tas"].sizes)}')
    print(f'  tas_gm shape: {dict(data["tas_gm"].sizes)}')

    # Spatial: combined file has tas (time, ens/seeds, lat, lon) from ReconRes.
    # Reorder to (ens, time, lat, lon) for the ens-first convention expected below.
    tas = data['tas'].values                         # (time, n_seeds, lat, lon)
    var_spatial_members = np.moveaxis(tas, 1, 0)    # (n_seeds, time, lat, lon)
    var_spatial_mean    = np.mean(var_spatial_members, axis=0)  # (time, lat, lon)

    # Global mean: combined file has tas_gm (time, total_ens) from ReconRes.
    # Transpose to (total_ens, time) for the ens-first convention expected below.
    var_global_members = data['tas_gm'].values.T    # (total_ens, time)

    # Coordinates
    lat = data['lat'].values
    lon = data['lon'].values

    # Time: CFR stores integer CE years; convert to yr BP (age = 1950 - year_CE)
    time_coord = data['time'].values
    age = 1950 - np.array(time_coord, dtype=float)  # e.g. 850-1850 CE -> 1100-100 yr BP

    data.close()

    # Config options (lmr_configs.yml is included in the artifact when present)
    config_file = os.path.join(data_dir, 'lmr_configs.yml')
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            options = yaml.load(f, Loader=yaml.FullLoader)
        options_list = [f'{k}: {v}' for k, v in options.items()]
    else:
        print(f'Note: no lmr_configs.yml in artifact; config options will be blank.')
        options_list = ['No configuration file found']

    # Ensure var_spatial_mean has the right shape for the module-level expand_dims below.
    # (time, lat, lon) is correct; ndim==3 branch adds the method dimension.
if var_spatial_members.ndim == 4:  # (ens, time, lat, lon)
    var_spatial_members = np.expand_dims(var_spatial_members, axis=0)  # Add method dimension -> (method, ens, time, lat, lon)
    var_spatial_mean = np.mean(var_spatial_members, axis=1)            # (method, time, lat, lon)

if var_global_members.ndim == 2:  # (ens, time)
    var_global_members = np.expand_dims(var_global_members, axis=0)  # Add method dimension -> (method, ens, time)

    var_global_mean = np.mean(var_global_members, axis=1)

    # Calculate lat and lon bounds
    lat_bounds, lon_bounds = functions_presto.bounding_latlon(lat, lon)

    # Get other metadata
    methods = ['CFR/LMR2']
    ens_spatial = np.arange(var_spatial_members.shape[1]) + 1
    ens_global = np.arange(var_global_members.shape[1]) + 1
    notes = ['Processed from CFR/LMR2 output']

    # Check shapes
    print("Variable shapes:")
    print(f"  var_spatial_members: {var_spatial_members.shape}")
    print(f"  var_spatial_mean: {var_spatial_mean.shape}")
    print(f"  var_global_members: {var_global_members.shape}")
    print(f"  var_global_mean: {var_global_mean.shape}")

    # Create output dataset
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

    ### SAVE DATA
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

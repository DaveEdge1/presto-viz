#==============================================================================
# Make a standardized netCDF file for Holocene Reconstruction of GraphEM,
# reconstructed from https://paleopresto.com/custom.html.
#    author: Michael Erb
#    date  : 2/29/2024
#==============================================================================

import sys
import numpy as np
import xarray as xr
import functions_presto
import yaml
import glob

# Set directories
#data_dir = '/projects/pd_lab/data/paleoclimate_reconstructions/presto_reconstructions/17056032413566754_HoloceneDA/'
#data_dir = '/projects/pd_lab/data/paleoclimate_reconstructions/presto_reconstructions/1705603319440696_GraphEM/test-run-graphem-cfg/'

data_dir = sys.argv[1]


#%% PROCESS DATA

var_txt      = 'tas'
quantity_txt = 'Annual'
if   'holocene_da' in data_dir: dataset_txt = 'daholocene';  version_txt = data_dir.split('_holocene_da')[0].split('/')[-1]
elif 'graph_em'    in data_dir: dataset_txt = 'graphem'; version_txt = data_dir.split('_graph_em')[0].split('/')[-1]
print(data_dir)
print(dataset_txt)
filename_txt = dataset_txt+'_v'+version_txt+'_'+var_txt+'_'+quantity_txt.lower()
print(' ===== STARTING script 1: Reformatting data for '+str(filename_txt)+' =====')

if dataset_txt == 'daholocene':
    #
    ### LOAD DATA
    #
    # Load data
    print('=== Processing Holocene Reconstruction ===')
    data_filename = glob.glob(data_dir+'holocene_recon*.nc')[0]
    data_xarray = xr.open_dataset(data_filename)
    var_global_members  = data_xarray['recon_tas_global_mean'].values
    var_spatial_mean    = data_xarray['recon_tas_mean'].values
    var_spatial_members = data_xarray['recon_tas_ens'].values
    age = data_xarray['ages'].values
    lat = data_xarray['lat'].values
    lon = data_xarray['lon'].values
    #
    # Load the configuration options
    with open(data_dir+'configs.yml','r') as file:
        options = yaml.load(file,Loader=yaml.FullLoader)
    #
    ### CALCULATIONS 1
    #
    options_list = []
    for key1 in options.keys():
        for key2 in options[key1].keys():
            option_txt = key1+'/'+key2+': '+str(options[key1][key2]['value'])
            options_list.append(option_txt)
    #
    ### CALCULATIONS 2
    #
    # Calculate lat and lon bounds
    lat_bounds,lon_bounds = functions_presto.bounding_latlon(lat,lon)
    #
    # Format the variables
    var_spatial_mean    = np.expand_dims(var_spatial_mean,axis=0)
    var_spatial_members = np.expand_dims(np.swapaxes(var_spatial_members,0,1),axis=0)
    var_global_members  = np.expand_dims(np.swapaxes(var_global_members,0,1),axis=0)
    var_global_mean     = np.mean(var_global_members,axis=1)
    #
    # Get other metadata
    methods = ['Holocene Reconstruction']
    ens_spatial = np.arange(var_spatial_members.shape[1])+1
    ens_global  = np.arange(var_global_members.shape[1])+1
    #
    # If this data can't be reformatted to the standard format, add a note here 
    notes = ['']
    #
    # Check the shape of the variables
    print(var_spatial_members.shape)
    print(var_spatial_mean.shape)
    print(var_global_members.shape)
    print(var_global_mean.shape)
    #
    ### FORMAT DATA
    #
    # Create new array
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
            'ens_global': (['ens_global'],ens_global,{'description':'ensemble members'}),  #TODO: The global and spatial ensemble members won't match. Look into this in all scripts.
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
    #
    ### SAVE DATA
    #
    # Save new array
    data_xarray_output.to_netcdf(data_dir+filename_txt+'.nc')
    #
    #
elif dataset_txt == 'graphem':
    #
    ### LOAD SPATIAL DATA
    #
    # Load data
    print('=== Processing GraphEM reconstruction ===')
    data_filename = glob.glob(data_dir+'test-run-graphem-cfg/'+'*recon.nc')[0]
    data_xarray = xr.open_dataset(data_filename)
    #
    # Load the configuration options
    # config_dir = '/'.join(data_dir.split('/')[:-2])+'/'
    with open(data_dir+'configs.yml','r') as file:
        options = yaml.load(file,Loader=yaml.FullLoader)
    #
    # Get coordinates
    year        = data_xarray['time'].values
    lat         = data_xarray['lat'].values
    lon         = data_xarray['lon'].values
    ens_spatial = data_xarray['ens'].values
    ens_global  = ens_spatial
    age = 1950-year
    #
    # Get other metadata
    methods = ['GraphEM']
    #
    # Get dimensions
    n_methods = len(methods)
    n_ens     = len(ens_spatial)
    n_ages    = len(age)
    n_lat     = len(lat)
    n_lon     = len(lon)
    #
    # Create a spatial variable with the chosen dimensions
    var_spatial_members = np.zeros((n_methods,n_ens,n_ages,n_lat,n_lon)); var_spatial_members[:] = np.nan
    var_spatial_members[0,0,:,:,:] = data_xarray['tas'].values
    #
    # Create a global variable with the chosen dimensions
    var_global_members = np.zeros((n_methods,n_ens,n_ages)); var_global_members[:] = np.nan
    var_global_members[0,:,:] = np.swapaxes(data_xarray['tas_gm'].values,0,1)
    #
    ### CALCULATIONS 1
    #
    options_list = []
    for key1 in options.keys():
        for key2 in options[key1].keys():
            option_txt = key1+'/'+key2+': '+str(options[key1][key2]['value'])
            options_list.append(option_txt)
    #
    ### CALCULATIONS 2
    #
    # Calculate lat and lon bounds
    lat_bounds,lon_bounds = functions_presto.bounding_latlon(lat,lon)
    #
    # Create an average across ensemble members
    var_spatial_mean = np.mean(var_spatial_members,axis=1)
    var_global_mean  = np.mean(var_global_members,axis=1)
    #
    # If this data can't be reformatted to the standard format, add a note here 
    notes = ['']
    #
    # Check the shape of the variables
    print(var_spatial_members.shape)
    print(var_spatial_mean.shape)
    print(var_global_members.shape)
    print(var_global_mean.shape)
    #
    ### FORMAT DATA
    #
    # Create new array
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
    #
    ### SAVE DATA
    #
    # Save new array
    data_xarray_output.to_netcdf(data_dir+filename_txt+'.nc')

print(' ===== FINISHED script 1: Data reformatted and saved to: '+data_dir+filename_txt+'.nc =====')


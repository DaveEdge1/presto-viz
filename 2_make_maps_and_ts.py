#==============================================================================
# Make a set of maps and time series for the visualizer webpage.
#    author: Michael Erb
#    date  : 2/29/2024
#==============================================================================

import os
import sys
import shutil
import gc
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent memory leaks
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.util as cutil
from cartopy.feature import ShapelyFeature
import xarray as xr
import regionmask
import time as timekeeping
import functions_presto
from bokeh.plotting import figure
from bokeh.resources import CDN
from bokeh.embed import file_html
from bokeh.models import HoverTool
from bokeh.models import Range1d
from bokeh.models import Span
import multiprocessing as mp

# Configure Python to minimize traceback retention
sys.tracebacklimit = 0  # Disable traceback printing entirely to prevent retention
gc.set_debug(0)  # Disable all gc debugging flags that might retain objects

# Try to import psutil for memory monitoring (optional)
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

save_instead_of_plot = True
plt.style.use('ggplot')
starttime_total = timekeeping.time() # Start timer


# Set directories
#data_dir = '/projects/pd_lab/data/paleoclimate_reconstructions/presto_reconstructions/17056032413566754_HoloceneDA/'
#data_dir = '/projects/pd_lab/data/paleoclimate_reconstructions/presto_reconstructions/1705603319440696_GraphEM/test-run-graphem-cfg/'
#output_dir = '/projects/pd_lab/mpe32/figures_presto/'

data_dir   = sys.argv[1]
output_dir = sys.argv[2]


#%% LOAD DATA

var_txt      = 'tas'
quantity_txt = 'Annual'
if   'holocene_da' in data_dir: dataset_txt = 'daholocene'; version_txt = data_dir.split('_holocene_da')[0].split('/')[-1]
elif 'graph_em'    in data_dir: dataset_txt = 'graphem';    version_txt = data_dir.split('_graph_em')[0].split('/')[-1]
filename_txt = dataset_txt+'_v'+version_txt+'_'+var_txt+'_'+quantity_txt.lower()
output_dir_full = output_dir+'viz/'
if os.path.exists(output_dir_full) == False: os.makedirs(output_dir_full)
output_dir_full = output_dir_full+'assets/'
if os.path.exists(output_dir_full) == False: os.makedirs(output_dir_full)
output_dir_full = output_dir_full+dataset_txt+'/'
if os.path.exists(output_dir_full) == False: os.makedirs(output_dir_full)
print(' ===== STARTING script 2: Making maps and time series for '+str(filename_txt)+' =====')

print('Loading '+filename_txt)
data_xarray = xr.open_dataset(data_dir+filename_txt+'.nc')
var_spatial_mean    = data_xarray[var_txt+'_spatial_mean']
var_spatial_members = data_xarray[var_txt+'_spatial_members']
var_global_mean     = data_xarray[var_txt+'_global_mean'].values
var_global_members  = data_xarray[var_txt+'_global_members'].values
var_units           = data_xarray[var_txt+'_spatial_mean'].units
ens_spatial         = data_xarray['ens_spatial'].values
ens_global          = data_xarray['ens_global'].values
method              = data_xarray['method'].values
age                 = data_xarray['age'].values
lat                 = data_xarray['lat'].values
lon                 = data_xarray['lon'].values
lat_bounds          = data_xarray['lat_bounds'].values
lon_bounds          = data_xarray['lon_bounds'].values
dataset_name        = data_xarray.attrs['dataset_name']
dataset_source_url  = data_xarray.attrs['dataset_source_url']
data_xarray.close()
year = 1950-age


#%% GENERAL SETTINGS - Parameters set based on general stuff

# Set time values
if min(year) > -2000: time_name_txt = 'Year'; time_var = year; time_unit_txt = 'CE';    time_start = min(time_var); time_end = max(time_var)
else:                 time_name_txt = 'Age';  time_var = age;  time_unit_txt = 'yr BP'; time_start = max(time_var); time_end = min(time_var)

# Set the name, colors, and map levels by variable type
if   var_txt == 'tas':    variable_name = 'temperature';                    cmap = 'bwr';    levels = np.array([-10,-5,-2,-1,-.5,-.2,-.1,0,.1,.2,.5,1,2,5,10])
elif var_txt == 'd18Op':  variable_name = 'd18Op';                          cmap = 'PiYG';   levels = np.array([-10,-5,-2,-1,-.5,-.2,-.1,0,.1,.2,.5,1,2,5,10])
elif var_txt == 'hydro':  variable_name = 'hydroclimate';                   cmap = 'BrBG';   levels = np.array([-10,-5,-2,-1,-.5,-.2,-.1,0,.1,.2,.5,1,2,5,10])
elif var_txt == 'precip': variable_name = 'precipitation';                  cmap = 'BrBG';   levels = np.arange(-1,1.1,.1)
elif var_txt == 'slp':    variable_name = 'sea level pressure';             cmap = 'PuOr_r'; levels = np.arange(-5,5.1,.5)
elif var_txt == 'pdsi':   variable_name = 'Palmer Drought Severity Index';  cmap = 'BrBG';   levels = np.arange(-5,5.1,.5)
elif var_txt == 'spei':   variable_name = 'Standardized Precip-Evap Index'; cmap = 'BrBG';   levels = np.arange(-1.5,1.6,.1)
else: print(' === ERROR: Unknown variable type ===')
if var_units == 'Z-score': levels = np.arange(-2,2.1,.2)

# Set the units in different formats, for formatting purposes
if   var_txt == 'pdsi': var_units = 'PDSI'
elif var_txt == 'spei': var_units = 'SPEI'
if   var_units in ['degC','degrees Celsius']:   info_unit_txt = '$^\circ$C'; colorbar_unit_txt = '$^\circ$C';                html_unit_txt = '\u00B0C'
elif var_units == 'permil (relative to VSMOW)': info_unit_txt = 'permil';    colorbar_unit_txt = 'permil relative to VSMOW'; html_unit_txt = 'permil'
else:                                           info_unit_txt = var_units;   colorbar_unit_txt = var_units;                  html_unit_txt = var_units

# Set some text
colorbar_txt    = quantity_txt+' '+variable_name+' ('+colorbar_unit_txt+')'
title_txt_bokeh = quantity_txt+' '+variable_name+' ('+html_unit_txt+')'

# For some variables, label fewer steps on the colorbar, so that text does not overlap
if var_txt in ['precip','slp','pdsi','spei']: tick_levels = levels[::2]
else: tick_levels = levels

# Set the vertical size of the time series figure
if len(method) == 1: ts_height = 300
else: ts_height = 500

# Set the range of y-axis values for time series figure
yrange_min = max(levels[0],-5)
yrange_max = min(levels[-1],5)
ts_yrange = [yrange_min,yrange_max]


#%% SPECIFIC SETTINGS - Parameters set based on dataset name

if   dataset_txt == 'daholocene':    ref_period = '0-1 ka';       map_region = 'global';    map_type = 'contourf';         make_gridded_ts = True;  make_regional_ts = True
elif dataset_txt == 'holocenehydro': ref_period = '0-1 ka';       map_region = 'global';    map_type = 'regions_ipcc_ar6'; make_gridded_ts = False; make_regional_ts = True
elif dataset_txt == 'lgmr':          ref_period = '0-1 ka';       map_region = 'global';    map_type = 'contourf';         make_gridded_ts = True;  make_regional_ts = True
elif dataset_txt == 'kaufman2020':   ref_period = '0-1 ka';       map_region = 'global';    map_type = 'pcolormesh';       make_gridded_ts = True;  make_regional_ts = False
elif dataset_txt == 'lmr':           ref_period = '0-1 ka';       map_region = 'global';    map_type = 'contourf';         make_gridded_ts = True;  make_regional_ts = True
elif dataset_txt == 'phyda':         ref_period = '0-1 ka';       map_region = 'global';    map_type = 'contourf';         make_gridded_ts = True;  make_regional_ts = True
elif dataset_txt == 'neukom2019':    ref_period = '0-1 ka';       map_region = 'global';    map_type = 'contourf';         make_gridded_ts = True;  make_regional_ts = True
elif dataset_txt == 'graphem':       ref_period = '0-1 ka';       map_region = 'global';    map_type = 'contourf';         make_gridded_ts = True;  make_regional_ts = True
elif dataset_txt == 'nada':          ref_period = 'none';         map_region = 'n_america'; map_type = 'pcolormesh';       make_gridded_ts = True;  make_regional_ts = False
elif dataset_txt == 'owda':          ref_period = 'none';         map_region = 'europe';    map_type = 'pcolormesh';       make_gridded_ts = True;  make_regional_ts = False
elif dataset_txt == 'era20c':        ref_period = '1951-1980 CE'; map_region = 'global';    map_type = 'contourf';         make_gridded_ts = True;  make_regional_ts = True
elif dataset_txt == 'era5':          ref_period = '1951-1980 CE'; map_region = 'global';    map_type = 'contourf';         make_gridded_ts = True;  make_regional_ts = True
else: print(' === ERROR: Unknown dataset:',dataset_txt)

# Some dataset-specific adjustments, to make webpages more useable
if dataset_txt in ['nada','owda']: ts_yrange = [-10,10]
if dataset_txt == 'lgmr':          ts_yrange = [-30,5]
if dataset_txt == 'kaufman2020':   levels = np.arange(-2,2.1,.2)


#%% PROCESS SPATIAL DATA

# Find reference period for the reconstruction
if   ref_period == '0-1 ka':       ind_ref = np.where((age  >= 0)    & (age  < 1000))[0]
elif ref_period == '1951-1980 CE': ind_ref = np.where((year >= 1951) & (year <= 1980))[0]

# Get dimensions
n_methods     = var_spatial_members.shape[0]
n_ens_spatial = var_spatial_members.shape[1]
n_ens_global  = var_global_members.shape[1]
n_time        = var_spatial_members.shape[2]
n_lat         = var_spatial_members.shape[3]
n_lon         = var_spatial_members.shape[4]

# Compute upper and lower uncertainty bands for spatial data (ideally the 2.5th and 97.5th percentiles; the choice will be noted in figures)
#ens_spatial = list(ens_spatial)  #TODO: Why did I want this to be a list?
if ('mean' in list(ens_spatial)) or (list(ens_spatial) == [1]):
    # Mean found
    print('Spatial bounds - Mean found')
    spatial_uncertainty_txt = 'none'; skip_spatial_ens = True
    var_spatial_lowerbound = var_spatial_mean
    var_spatial_upperbound = var_spatial_mean
    #
elif (all(isinstance(x,np.integer) for x in ens_spatial)) or (all(isinstance(x,float) for x in ens_spatial)) or (all(isinstance(x,np.float32) for x in ens_spatial)) or (all(isinstance(x,np.float64) for x in ens_spatial)):
    # Spatial members are ensembles
    print('Spatial bounds - Members are ensembles')
    print(f'Data shape: {var_spatial_members.shape}, Computing quantiles for {var_spatial_members.shape[1]} ensemble members')
    spatial_uncertainty_txt = '2.5 - 97.5th percentile range'; skip_spatial_ens = False
    
    print('Computing percentiles using optimized method for large datasets...')
    starttime_quantile = timekeeping.time()
    
    # Convert to numpy for much faster quantile computation
    print('Converting xarray to numpy for faster processing...')
    var_spatial_numpy = var_spatial_members.values  # Shape: (methods, ens_spatial, time, lat, lon)
    print(f'Data shape: {var_spatial_numpy.shape}')
    
    # Compute both quantiles simultaneously using numpy (much faster than xarray)
    print('Computing 2.5th and 97.5th percentiles simultaneously...')
    quantiles = np.quantile(var_spatial_numpy, [0.025, 0.975], axis=1)  # Compute along ensemble dimension
    
    # Convert back to xarray format with CORRECT coordinates (no ens_spatial dimension)
    print('Converting results back to xarray format...')
    # Get the correct coordinates by removing ens_spatial from var_spatial_members
    correct_coords = {k: v for k, v in var_spatial_members.coords.items() if k != 'ens_spatial'}
    correct_dims = [dim for dim in var_spatial_members.dims if dim != 'ens_spatial']
    
    var_spatial_lowerbound = xr.DataArray(
        quantiles[0], 
        dims=correct_dims,
        coords=correct_coords,
        attrs=var_spatial_members.attrs
    )
    var_spatial_upperbound = xr.DataArray(
        quantiles[1], 
        dims=correct_dims, 
        coords=correct_coords,
        attrs=var_spatial_members.attrs
    )
    
    print(f'Both percentiles computed in {timekeeping.time() - starttime_quantile:.1f} seconds (much faster than xarray method!)')
    print('Quantile calculations completed successfully')
    #
elif ('p2.5' in list(ens_spatial)) and ('p97.5' in list(ens_spatial)):
    # Percentiles 2.5 and 97.5 found
    print('Spatial bounds - Percentiles 2.5 and 97.5 found')
    spatial_uncertainty_txt = '2.5 - 97.5th percentile range'; skip_spatial_ens = False
    ind_lowerbound = np.where(ens_spatial == 'p2.5')[0][0]
    ind_upperbound = np.where(ens_spatial == 'p97.5')[0][0]
    var_spatial_lowerbound = var_spatial_members[:,ind_lowerbound,:,:,:]
    var_spatial_upperbound = var_spatial_members[:,ind_upperbound,:,:,:]
    #
elif ('p5' in list(ens_spatial)) and ('p95' in list(ens_spatial)):
    # Percentiles 5 and 95 found
    print('Spatial bounds - Percentiles 5 and 95 found')
    spatial_uncertainty_txt = '5 - 95th percentile range'; skip_spatial_ens = False
    ind_lowerbound = np.where(ens_spatial == 'p5')[0][0]
    ind_upperbound = np.where(ens_spatial == 'p95')[0][0]
    var_spatial_lowerbound = var_spatial_members[:,ind_lowerbound,:,:,:]
    var_spatial_upperbound = var_spatial_members[:,ind_upperbound,:,:,:]
    #
else:
    # Ensemble members unclear
    print('Spatial bounds - WARNING: Ensemble members unclear. Using means. Ensembles=',ens_spatial,var_spatial_members.shape)
    spatial_uncertainty_txt = 'none'; skip_spatial_ens = True
    var_spatial_lowerbound = var_spatial_mean
    var_spatial_upperbound = var_spatial_mean

print(' === Spatial shapes ===')
print('var_spatial_mean:      ',var_spatial_mean.shape)
print('var_spatial_lowerbound:',var_spatial_lowerbound.shape)
print('var_spatial_upperbound:',var_spatial_upperbound.shape)

# Remove reference period
if ref_period != 'none':
    spatial_ref_value = np.nanmean(var_spatial_mean[:,ind_ref,:,:],axis=1)[:,None,:,:]
    var_spatial_mean       = var_spatial_mean       - spatial_ref_value
    var_spatial_lowerbound = var_spatial_lowerbound - spatial_ref_value
    var_spatial_upperbound = var_spatial_upperbound - spatial_ref_value


#%% PROCESS GLOBAL DATA

# Check to see if all of the ensemble members are numbers
globalens_allnumbers = (all(isinstance(x,np.integer) for x in ens_global)) or (all(isinstance(x,float) for x in ens_global)) or (all(isinstance(x,np.float32) for x in ens_global)) or (all(isinstance(x,np.float64) for x in ens_global))

# Remove reference period
if ref_period != 'none':
    global_ref_value = np.nanmean(var_global_mean[:,ind_ref],axis=1)[:,None]
    var_global_mean = var_global_mean - global_ref_value
    if globalens_allnumbers:
        var_global_members = var_global_members - np.expand_dims(global_ref_value,axis=2)

# Initial global processing
var_global_mean_allmethods = np.mean(var_global_mean,axis=0)
var_global_members_reshape = np.reshape(var_global_members,(n_methods*n_ens_global,n_time))

if ('mean' in list(ens_global)) or (list(ens_global) == [1]):
    # Mean found
    print('Global bounds - Mean found')
    global_uncertainty_txt = 'none'; skip_global_ens = True
    var_global_lowerbound = var_global_mean_allmethods
    var_global_upperbound = var_global_mean_allmethods
    #
elif globalens_allnumbers:
    # global members are ensembles
    print('Global bounds - Members are ensembles')
    print(f'Global data shape: {var_global_members_reshape.shape}, Computing global quantiles')
    global_uncertainty_txt = '2.5 - 97.5th percentile range'; skip_global_ens = False
    
    print('Computing global percentiles simultaneously (optimized method)...')
    starttime_global = timekeeping.time()
    # Compute both quantiles at once for efficiency
    global_quantiles = np.quantile(var_global_members_reshape, [0.025, 0.975], axis=0)
    var_global_lowerbound = global_quantiles[0]
    var_global_upperbound = global_quantiles[1]
    print(f'Both global percentiles computed in {timekeeping.time() - starttime_global:.1f} seconds')
    print('Global quantile calculations completed successfully')
    #
elif ('p2.5' in list(ens_global)) and ('p97.5' in list(ens_global)) and (len(method) == 1):
    # Percentiles 2.5 and 97.5 found
    print('Global bounds - Percentiles 2.5 and 97.5 found')
    global_uncertainty_txt = '2.5 - 97.5th percentile range'; skip_global_ens = False
    ind_lowerbound = np.where(ens_global == 'p2.5')[0][0]
    ind_upperbound = np.where(ens_global == 'p97.5')[0][0]
    var_global_lowerbound = np.squeeze(var_global_members[:,ind_lowerbound,:] - global_ref_value)
    var_global_upperbound = np.squeeze(var_global_members[:,ind_upperbound,:] - global_ref_value)
    #
elif ('p5' in list(ens_global)) and ('p95' in list(ens_global)) and (len(method) == 1):
    # Percentiles 5 and 95 found
    print('Global bounds - Percentiles 5 and 95 found')
    global_uncertainty_txt = '5 - 95th percentile range'; skip_global_ens = False
    ind_lowerbound = np.where(ens_global == 'p5')[0][0]
    ind_upperbound = np.where(ens_global == 'p95')[0][0]
    var_global_lowerbound = np.squeeze(var_global_members[:,ind_lowerbound,:] - global_ref_value)
    var_global_upperbound = np.squeeze(var_global_members[:,ind_upperbound,:] - global_ref_value)
    #
else:
    # Ensemble members unclear
    print('Global bounds - WARNING: Ensemble members unclear. Using means. Ensembles=',ens_global,var_global_members.shape)
    global_uncertainty_txt = 'none'; skip_global_ens = True
    var_global_lowerbound = var_global_mean_allmethods
    var_global_upperbound = var_global_mean_allmethods

print(' === Global shapes ===')
print('var_global_mean_allmethods:',var_global_mean_allmethods.shape)
print('var_global_lowerbound:     ',var_global_lowerbound.shape)
print('var_global_upperbound:     ',var_global_upperbound.shape)

# Compute the mean of all methods
var_spatial_mean_allmethods = np.nanmean(var_spatial_mean,axis=0)


#%% MAP PREPARATION

# Get the colors from the colorbar. This is important for non-linear colorbars
colors_from_cmap = matplotlib.colormaps[cmap]
n_colors = len(levels)+1
colors_selected = colors_from_cmap(np.linspace(0,1,n_colors))

# Make 2D lon bound variables
lon_bounds_2d,lat_bounds_2d = np.meshgrid(lon_bounds,lat_bounds)

# Get all WGI regions
ar6_all = regionmask.defined_regions.ar6.all
ar6_abbreviations = ar6_all.abbrevs

# If the reconstruction uses the IPCC AR6 regions, get some data about regions
regions_all = None
if map_type == 'regions_ipcc_ar6': regions_all = lat


#%% MAPS
# Make a map of values
print('Step 1: Making maps and info panels. N='+str(len(time_var))+' each')

# Create CRS objects ONCE before loop to prevent memory leak from repeated CRS creation
crs_platecarree = ccrs.PlateCarree()
crs_mercator = ccrs.Mercator(central_longitude=0, min_latitude=-85, max_latitude=85)

# Pre-load coastline geometries ONCE before loop to avoid downloading/parsing 1200 times
# This is the key to fixing the memory leak
import cartopy.feature as cfeature
coastlines_feature = cfeature.NaturalEarthFeature('physical', 'coastline', '50m',
                                                   edgecolor='black', facecolor='none')

# Pre-compute cyclic point coordinates for all timesteps to avoid repeated computation
# This prevents memory accumulation from coordinate transformations
if map_type == 'contourf':
    print('Pre-computing cyclic coordinates for all timesteps...')
    # Add cyclic point to longitude for proper wrapping
    var_spatial_with_cyclic = []
    lon_cyclic = None
    for idx in range(len(time_var)):
        var_cyc, lon_cyc = cutil.add_cyclic_point(var_spatial_mean_allmethods[idx,:,:], coord=lon)
        var_spatial_with_cyclic.append(var_cyc)
        if lon_cyclic is None:
            lon_cyclic = lon_cyc  # Same for all timesteps
    print('Cyclic coordinate pre-computation complete')

# CRITICAL FIX: Pre-transform coordinates from PlateCarree to Mercator ONCE
# Avoid transform parameter which causes memory leak through repeated coordinate transformation
print('Pre-transforming coordinates to target projection to avoid memory leak...')
import numpy as np
# Create meshgrid of coordinates
if map_type == 'contourf':
    lon_2d, lat_2d = np.meshgrid(lon_cyclic, lat)
else:
    lon_2d, lat_2d = np.meshgrid(lon, lat)

# Transform coordinates from PlateCarree to Mercator once
transformed_coords = crs_mercator.transform_points(crs_platecarree, lon_2d, lat_2d)
x_transformed = transformed_coords[:, :, 0]
y_transformed = transformed_coords[:, :, 1]
print('Coordinate transformation complete')

# Pre-determine grid spacing based on map region (same for all iterations)
if   map_region == 'global':    extra_txt_x = 0;       extra_txt_y = -82;   grid_x = 60; grid_y = 30
elif map_region == 'n_america': extra_txt_x = -111.75; extra_txt_y = 7.5;   grid_x = 20; grid_y = 10
elif map_region == 'europe':    extra_txt_x = 16.5;    extra_txt_y = 31.25; grid_x = 10; grid_y = 5

# Pre-create FixedLocator objects once instead of creating 1200 times
y_locator = mticker.FixedLocator(np.arange(-90,91,grid_y))
x_locator = mticker.FixedLocator(np.arange(-180,181,grid_x))

def make_one_plot(args):
    """Wrapper function for multiprocessing Pool that unpacks arguments"""
    (i, time_value,
     var_cyclic, lon_cyclic,
     var_global_lowerbound, var_global_upperbound,
     var_global_mean_allmethods,
     lon_bounds_2d, lat_bounds_2d,
     output_dir_full, filename_txt,
     cmap, levels, tick_levels,
     colors_selected,
     crs_mercator, crs_platecarree,
     ref_period, colorbar_txt,
     info_unit_txt, time_var,
     time_start, time_end, dataset_name, version_txt,
     map_region, map_type,
     skip_global_ens,
     ar6_all, ar6_abbreviations, regions_all,
     colors_from_cmap,
     coastlines_feature, y_locator, x_locator,
     extra_txt_x, extra_txt_y, time_unit_txt) = args

    # === Time series panel ===
    fig = plt.figure(figsize=(4, 2))
    ax1 = fig.add_subplot(111)
    ax1.axis("off")

    if not skip_global_ens and (var_global_lowerbound is not None and var_global_upperbound is not None):
        ax1.fill_between(time_var, var_global_lowerbound, var_global_upperbound,
                         color="lightgray")

    ax1.plot(time_var, var_global_mean_allmethods, linewidth=1)
    ax1.axvline(x=time_value, color="gray", alpha=1, linestyle="--", linewidth=1)
    ax1.axhline(y=0, color="k", alpha=1, linestyle="--", linewidth=1)
    ax1.set_xlim(time_start, time_end + (time_end - time_start) / 100)
    ax1.set_title("Mean : " +
                  str("{:.2f}".format(var_global_mean_allmethods[i])) +
                  " " + info_unit_txt, fontsize=18)

    plt.savefig(output_dir_full + "info_" + filename_txt + "_" +
                str(int(np.ceil(time_var[i]))).zfill(5) + ".png",
                dpi=50, format="png", bbox_inches="tight")
    plt.close(fig)

    # === Map panel ===
    fig2 = plt.figure(figsize=(10, 10))
    ax2 = fig2.add_subplot(111, projection=crs_mercator)

    # Extent
    if   map_region == 'global':    ax2.set_extent([-179.99,179.99,-85,85], crs=crs_platecarree)
    elif map_region == 'n_america': ax2.set_extent([-171.5,-52,-5,84],      crs=crs_platecarree)
    elif map_region == 'europe':    ax2.set_extent([-13,46,26,72],          crs=crs_platecarree)

    # Main plot — use lon/lat with transform=PlateCarree
    if map_type == "contourf":
        map1 = ax2.contourf(
            lon_cyclic, lat, var_cyclic,
            colors=colors_selected, levels=levels, extend="both",
            transform=crs_platecarree
        )

    elif map_type == "pcolormesh":
        map1 = ax2.pcolormesh(
            lon_bounds_2d, lat_bounds_2d, var_cyclic,
            cmap=cmap, vmin=levels[0], vmax=levels[-1],
            transform=crs_platecarree
        )

    elif map_type == "regions_ipcc_ar6":
        norm = matplotlib.colors.Normalize(vmin=-2, vmax=2, clip=True)
        for j in range(len(ar6_all)):
            region_txt = ar6_abbreviations[j]
            if regions_all is not None and region_txt in regions_all:
                ind_region = np.where(region_txt == regions_all)[0][0]
                value_for_region = var_cyclic[ind_region, 0]
            else:
                value_for_region = np.nan
            facecolor = 'lightgray' if np.isnan(value_for_region) else colors_from_cmap(norm(value_for_region))
            region = ShapelyFeature([ar6_all[j]._polygon],
                                    facecolor=facecolor,
                                    crs=crs_platecarree)
            ax2.add_feature(region)
        map1 = None

    # Coastlines
    if coastlines_feature is not None:
        ax2.add_feature(coastlines_feature)

    # Gridlines
    if x_locator is not None and y_locator is not None:
        gl = ax2.gridlines(color='gray', linestyle=':', draw_labels=False)
        gl.ylocator = y_locator
        gl.xlocator = x_locator

    # Colorbar
    if map_type in ["contourf", "pcolormesh"] and map1 is not None:
        colorbar = plt.colorbar(map1, ticks=tick_levels, orientation="horizontal",
                                ax=ax2, fraction=0.01, pad=-0.07)
    elif map_type == "regions_ipcc_ar6":
        norm = matplotlib.colors.Normalize(vmin=-2, vmax=2, clip=True)
        colorbar = plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
                                orientation="horizontal", ax=ax2, fraction=0.01, pad=-0.07)
    else:
        colorbar = None

    if colorbar is not None:
        label_txt = colorbar_txt if ref_period == "none" else colorbar_txt + ", rel. " + ref_period
        colorbar.set_label(label_txt, fontsize=6)
        colorbar.ax.tick_params(labelsize=3)

    # Text annotation
    ax2.text(extra_txt_x, extra_txt_y,
             dataset_name + ", v." + version_txt.replace("_", ".") +
             ", " + str(time_value) + " " + time_unit_txt,
             fontsize=7, ha="center", transform=crs_platecarree)

    # NaN message
    if np.isnan(var_cyclic).all():
        ax2.text(0.5, 0.5, "Please select another year.",
                 fontsize=12, ha="center", va="center", transform=ax2.transAxes)

    plt.savefig(output_dir_full + "map_" + filename_txt + "_" +
                str(int(np.ceil(time_var[i]))).zfill(5) + ".png",
                dpi=150, format="png", bbox_inches="tight", pad_inches=0.0)
    plt.close(fig2)


# === MAIN LOOP USING MULTIPROCESSING POOL ===
print("Step 1: Making maps and info panels with parallel workers (4 processes)")

# Prepare all arguments for parallel processing
all_args = []
for i, time in enumerate(time_var):
    if map_type == "contourf":
        var_cyclic, lon_cyclic = cutil.add_cyclic_point(
            var_spatial_mean_allmethods[i,:,:], coord=lon
        )
    elif map_type == "pcolormesh":
        var_cyclic = var_spatial_mean_allmethods[i,:,:]
        lon_cyclic = lon  # not used
    elif map_type == "regions_ipcc_ar6":
        var_cyclic = var_spatial_mean_allmethods[i,:,:]
        lon_cyclic = lon  # not used
    else:
        raise ValueError(f"Unknown map_type: {map_type}")

    args_tuple = (i, time,
                  var_cyclic, lon_cyclic,
                  None if skip_global_ens else var_global_lowerbound,
                  None if skip_global_ens else var_global_upperbound,
                  var_global_mean_allmethods,
                  lon_bounds_2d, lat_bounds_2d,
                  output_dir_full, filename_txt,
                  cmap, levels, tick_levels,
                  colors_selected,
                  crs_mercator, crs_platecarree,
                  ref_period, colorbar_txt,
                  info_unit_txt, time_var,
                  time_start, time_end, dataset_name, version_txt,
                  map_region, map_type,
                  skip_global_ens,
                  ar6_all, ar6_abbreviations, regions_all,
                  colors_from_cmap,
                  coastlines_feature, y_locator, x_locator,
                  extra_txt_x, extra_txt_y, time_unit_txt)
    all_args.append(args_tuple)

print(f"Prepared {len(all_args)} tasks, starting parallel processing...")

# Process in parallel with a pool of 4 workers
# maxtasksperchild=20 ensures each worker is terminated after 20 task to prevent memory leaks
from multiprocessing import Pool
with Pool(processes=4, maxtasksperchild=20) as pool:
    # Use imap to get progress feedback
    for idx, result in enumerate(pool.imap(make_one_plot, all_args)):
        if idx % 50 == 0:
            print(f"Completed {idx+1}/{len(all_args)}")

print("All maps and info panels complete")

#%% MAKE TIME SERIES FOR LOCATIONS

# Set color possibilities for time series
method_color_list = ['black','royalblue','salmon','olive','orange','darkseagreen',
                     'black','royalblue','salmon','olive','orange','darkseagreen',
                     'black','royalblue','salmon','olive','orange','darkseagreen',
                     'black','royalblue','salmon','olive','orange','darkseagreen',
                     'black','royalblue','salmon','olive','orange','darkseagreen']

lat_string, lon_string, j_for_ts, i_for_ts, lon_neg = functions_presto.select_latlons(lat, lon, map_region, dataset_txt)

#%% TIME SERIES FUNCTION

# A function to make a time series
def make_time_series(var_mean_to_plot,var_lowerbound_to_plot,var_upperbound_to_plot,location_title_txt,outputfile_txt,text_color):
    #
    # Make an interactive time series with bokeh
    p1 = figure(width=1200,
                height=ts_height,
                title=title_txt_bokeh+' for '+dataset_name+', v.'+version_txt.replace('_','.')+location_title_txt+' (uncertainties: '+spatial_uncertainty_txt+')',
                tools='pan,box_zoom,hover,save,reset',
                active_drag='box_zoom',active_inspect='hover',
                x_range=Range1d(bounds=(min(time_var),max(time_var))))
    #
    p1.title.text_color = text_color
    p1.xaxis.axis_label = time_name_txt+' ('+time_unit_txt+')'
    p1.yaxis.axis_label = title_txt_bokeh
    p1.x_range.start = time_start
    p1.x_range.end   = time_end
    p1.y_range.start = ts_yrange[0]
    p1.y_range.end   = ts_yrange[1]
    #
    for k,method_chosen in enumerate(method):
        if skip_spatial_ens: pass
        else: p1.varea(time_var,var_lowerbound_to_plot[k,:],var_upperbound_to_plot[k,:],color=method_color_list[k],alpha=0.1,legend_label=method_chosen)
        p1.line(time_var,var_mean_to_plot[k,:],color=method_color_list[k],line_width=1,legend_label=method_chosen)
    line0 = Span(location=0,dimension='width',line_color='gray',line_width=1)
    p1.renderers.extend([line0])
    p1.background_fill_color           = 'white'
    p1.grid.grid_line_color            = '#e0e0e0'
    p1.axis.axis_label_text_font_style = 'normal'
    p1.axis.axis_label_text_font_size  = '16px'
    p1.title.text_font_size            = '16px'
    p1.title.align                     = 'center'
    p1.legend.location     = 'bottom_right'
    p1.legend.click_policy = 'hide'
    #
    hover = p1.select_one(HoverTool)
    hover.tooltips = [
            (time_name_txt,'@x{int} '+time_unit_txt),
            (var_txt,'@y '+html_unit_txt),
            ]
    hover.mode='vline'
    #
    # Save as html
    html = file_html(p1,CDN,outputfile_txt)
    output_file = open(output_dir_full+outputfile_txt,'w')
    output_file.write(html)
    output_file.close()


#%% MAKE TIME SERIES FOR LOCATIONS

# Set color possibilities for time series
method_color_list = ['black','royalblue','salmon','olive','orange','darkseagreen',
                     'black','royalblue','salmon','olive','orange','darkseagreen',
                     'black','royalblue','salmon','olive','orange','darkseagreen',
                     'black','royalblue','salmon','olive','orange','darkseagreen',
                     'black','royalblue','salmon','olive','orange','darkseagreen']
# Make a timeseries at every location
if make_gridded_ts:
    j,i = 0,0
    counter = 0
    n_total = len(j_for_ts)*len(i_for_ts)
    print('Step 3: Making time series at selected points. N='+str(n_total))
    for j in j_for_ts:
        counter += 1
        print('Processing: '+str(counter)+'/'+str(len(j_for_ts)))
        for i in i_for_ts:
            #
            # Make an interactive time series with bokeh
            lat_txt = str('{:.1f}'.format(lat[j]))
            lon_txt = str('{:.1f}'.format(lon_neg[i]))
            var_mean_to_plot       = var_spatial_mean[:,:,j,i].values
            var_lowerbound_to_plot = var_spatial_lowerbound[:,:,j,i].values
            var_upperbound_to_plot = var_spatial_upperbound[:,:,j,i].values
            location_title_txt     = ' near '+lat_txt+'\u00B0N, '+lon_txt+'\u00B0E'
            outputfile_txt         = 'ts_'+filename_txt+'_lat_'+lat_txt+'_lon_'+lon_txt+'.html'
            text_color             = 'black'
            make_time_series(var_mean_to_plot,var_lowerbound_to_plot,var_upperbound_to_plot,location_title_txt,outputfile_txt,text_color)


#%% MAKE TIME SERIES FOR REGIONS

# Make regional time series plots, if requested
if make_regional_ts:
    #
    ### Compute or retrieve regional means
    if map_type == 'regions_ipcc_ar6':
        #
        # In this case, regional means have already been created
        n_regions = len(ar6_abbreviations)
        var_regional_mean       = np.zeros((n_methods,n_time,n_regions)); var_regional_mean[:]       = np.nan
        var_regional_lowerbound = np.zeros((n_methods,n_time,n_regions)); var_regional_lowerbound[:] = np.nan
        var_regional_upperbound = np.zeros((n_methods,n_time,n_regions)); var_regional_upperbound[:] = np.nan
        for i,region_txt in enumerate(ar6_abbreviations):
            ind_region = np.where(region_txt==regions_all)[0][0]
            var_regional_mean[:,:,i]       = var_spatial_mean[:,:,ind_region,0]
            var_regional_lowerbound[:,:,i] = var_spatial_lowerbound[:,:,ind_region,0]
            var_regional_upperbound[:,:,i] = var_spatial_upperbound[:,:,ind_region,0]
        #
    else:
        #
        # Make a mask for the different regions
        mask_3D = ar6_all.mask_3D(lon,lat)
        #
        # Calculate weights for every gridcell
        lon_2d,lat_2d = np.meshgrid(lon,lat)
        lat_weights = np.cos(np.deg2rad(lat_2d))
        #
        # Compute regional means
        var_regional_mean       = var_spatial_mean.weighted(mask_3D * lat_weights).mean(dim=('lat','lon')).values
        var_regional_lowerbound = var_spatial_lowerbound.weighted(mask_3D * lat_weights).mean(dim=('lat','lon')).values
        var_regional_upperbound = var_spatial_upperbound.weighted(mask_3D * lat_weights).mean(dim=('lat','lon')).values
    #
    #
    ### Make interactive regional time series plots with bokeh
    print('Step 4: Making time series for regions. N='+str(len(ar6_all.abbrevs)))
    nans_to_plot = np.zeros((n_methods,n_time)); nans_to_plot[:] = np.nan
    for j,abbrev_selected in enumerate(ar6_all.abbrevs):
        print('Processing: '+str(j+1)+'/'+str(len(ar6_all.abbrevs))+': '+abbrev_selected)
        #
        # Set some parameters
        location_title_txt = ' for region '+ar6_all.abbrevs[j]+' ('+ar6_all.names[j]+')'
        outputfile_txt     = 'ts_'+filename_txt+'_region_'+ar6_all.abbrevs[j]+'.html'
        text_color         = 'green'
        #
        # Find the index of the region
        ind_selected = np.where(mask_3D.abbrevs.values == abbrev_selected)[0]
        if len(ind_selected) == 1:
            var_mean_to_plot       = var_regional_mean[:,:,ind_selected[0]]
            var_lowerbound_to_plot = var_regional_lowerbound[:,:,ind_selected[0]]
            var_upperbound_to_plot = var_regional_upperbound[:,:,ind_selected[0]]
        else:
            var_mean_to_plot       = nans_to_plot
            var_lowerbound_to_plot = nans_to_plot
            var_upperbound_to_plot = nans_to_plot
        #
        make_time_series(var_mean_to_plot,var_lowerbound_to_plot,var_upperbound_to_plot,location_title_txt,outputfile_txt,text_color)

endtime_total = timekeeping.time()  # End timer

print(' ===== FINISHED script 2: Making maps and time series saved to: '+output_dir_full+' =====')
print('Total time: '+str('%1.2f' % ((endtime_total-starttime_total)/60))+' minutes')


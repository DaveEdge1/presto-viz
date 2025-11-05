#==============================================================================
# Functions for Presto
#    author: Michael Erb
#    date  : 2/29/2024
#==============================================================================

import numpy as np
import copy

# Compute the bounding latitudes and longitudes for a regular grid, given central values
def bounding_latlon(lat,lon):
    #
    # Calculate lon bounds
    lon_extended = copy.deepcopy(lon)
    lon_extended = np.insert(lon_extended,0,lon[0]-(lon[1]-lon[0]))
    lon_extended = np.append(lon_extended,lon[-1]+(lon[-1]-lon[-2]))
    lon_bounds = (lon_extended[:-1] + lon_extended[1:])/2
    #
    # Calculate lat bounds
    lat_extended = copy.deepcopy(lat)
    lat_extended = np.insert(lat_extended,0,lat[0]-(lat[1]-lat[0]))
    lat_extended = np.append(lat_extended,lat[-1]+(lat[-1]-lat[-2]))
    lat_bounds = (lat_extended[:-1] + lat_extended[1:])/2
    if max(lat_bounds) > 90:
        print('Maximum latitude bound is '+str(max(lat_bounds))+'. Setting to 90')
        lat_bounds[lat_bounds > 90]  = 90
    if min(lat_bounds) < -90:
        print('Minimum latitude bound is '+str(min(lat_bounds))+'. Setting to -90')
        lat_bounds[lat_bounds < -90] = -90
    #
    return lat_bounds,lon_bounds

# Grid calculations
def select_latlons(lat,lon,map_region,dataset_txt):
    #
    # Select the approximate grid to generate time series figures for
    if   map_region == 'global':    lat_grid_desired = 5; lon_grid_desired = 5
    elif map_region == 'n_america': lat_grid_desired = 2; lon_grid_desired = 2
    elif map_region == 'europe':    lat_grid_desired = 1; lon_grid_desired = 1
    #
    # If requested, figure out a reduced set of indices for generating figures
    if len(lat)*len(lon) > 1000:
        #
        # Get the indicies for points along the approximate grid
        lat_grid_mean = np.abs(np.mean(lat[1:]-lat[:-1]))
        lon_grid_mean = np.abs(np.mean(lon[1:]-lon[:-1]))
        lat_skip_factor = max(round(lat_grid_desired/lat_grid_mean),1)
        lon_skip_factor = max(round(lon_grid_desired/lon_grid_mean),1)
        lat_start_value = int(np.floor(np.remainder(len(lat)-1,lat_skip_factor)/2))
        lon_start_value = int(np.floor(np.remainder(len(lon)-1,lon_skip_factor)/2))
        if dataset_txt == 'era5': lon_start_value += 1
        j_for_ts = np.arange(lat_start_value,len(lat),lat_skip_factor)
        i_for_ts = np.arange(lon_start_value,len(lon),lon_skip_factor)
        #
        # Print some stats
        print('--- Skip factors for grid locations ---')
        print('Lat:',lat_skip_factor)
        print('Lon:',lon_skip_factor)
        print('Original number of locations:',len(lat)*len(lon))
        print('New number of locations:     ',len(lat[j_for_ts])*len(lon[i_for_ts]))
        #
    else:
        #
        j_for_ts = np.arange(len(lat))
        i_for_ts = np.arange(len(lon))
    #
    #
    #%% Save grid lats and lons
    #
    if (all(isinstance(x,np.integer) for x in lon)) or (all(isinstance(x,float) for x in lon)) or (all(isinstance(x,np.float32) for x in lon)) or (all(isinstance(x,np.float64) for x in lon)):
        #
        # Make a version of the longitude that goes from -180 to 180
        lon_neg = copy.deepcopy(lon)
        lon_neg[lon_neg > 180] = lon_neg[lon_neg > 180] - 360
        #
        # Make versions for printing
        lat_string = ','.join([str('{:.1f}'.format(value)) for value in lat[j_for_ts]])
        lon_string = ','.join([str('{:.1f}'.format(value)) for value in lon_neg[i_for_ts]])
        #
        lat_string = "          var lat_all = ["+lat_string+"];"
        lon_string = "          var lon_all = ["+lon_string+"];"
        #
    else:
        lat_string = "          var lat_all = [];"
        lon_string = "          var lon_all = [];"
    #
    return lat_string,lon_string,j_for_ts,i_for_ts,lon_neg


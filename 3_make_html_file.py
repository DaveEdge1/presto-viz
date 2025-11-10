#==============================================================================
# Create a html file for the visualizer.
#    author: Michael Erb
#    date  : 2/29/2024
#==============================================================================

import os
import sys
import numpy as np
import xarray as xr
import functions_presto


# Set directories
#data_dir = '/projects/pd_lab/data/paleoclimate_reconstructions/presto_reconstructions/17056032413566754_HoloceneDA/'
#data_dir = '/projects/pd_lab/data/paleoclimate_reconstructions/presto_reconstructions/1705603319440696_GraphEM/test-run-graphem-cfg/'
#output_dir = '/projects/pd_lab/mpe32/figures_presto/'
#web_data_dir = '/home/mpe32/analysis/4_presto/1_website/5_make_standalone_visualizer/web_assets/'

data_dir     = sys.argv[1]
output_dir   = sys.argv[2]
web_data_dir = sys.argv[3]


#%% LOAD DATA


# Load the visualizer template
with open(web_data_dir+'visualizer_template.html') as f:
    lines_template = f.readlines()

# Load reconstruction
var_txt      = 'tas'
quantity_txt = 'Annual'
if   'holocene_da' in data_dir: dataset_txt = 'daholocene'; version_txt = data_dir.split('_holocene_da')[0].split('/')[-1]
elif 'graph_em'    in data_dir: dataset_txt = 'graphem';    version_txt = data_dir.split('_graph_em')[0].split('/')[-1]
else:                           dataset_txt = 'lmr';        version_txt = data_dir.rstrip('/').split('/')[-1]
filename_txt = dataset_txt+'_v'+version_txt+'_'+var_txt+'_'+quantity_txt.lower()
output_dir_full = output_dir+'viz/'
print(' ===== STARTING script 3: Making html and zipping '+str(filename_txt)+' =====')

data_xarray = xr.open_dataset(data_dir+'/'+filename_txt+'.nc')
method       = data_xarray['method'].values
age          = data_xarray['age'].values
lat          = data_xarray['lat'].values
lon          = data_xarray['lon'].values
options      = data_xarray['options'].values
dataset_name = data_xarray.attrs['dataset_name']
data_xarray.close()
year = 1950-age


#%% SET TEXT

# Set parameters based on the dataset
if   dataset_txt == 'daholocene': time_units = 'yr BP'
elif dataset_txt == 'graphem':    time_units = 'CE'
elif dataset_txt == 'lmr':        time_units = 'yr BP' if min(year) <= -2000 else 'CE'

# Set time values
if time_units == 'yr BP':
    txt_time_old  = str(-int(np.ceil(max(age))))
    txt_time_new  = str(-int(np.ceil(min(age))))
    txt_time_diff = str(np.abs(int(age[1]-age[0])))
elif time_units == 'CE':
    txt_time_old  = str(int(np.ceil(min(year))))
    txt_time_new  = str(int(np.ceil(max(year))))
    txt_time_diff = str(np.abs(int(year[1]-year[0])))

# Set text values
txt_ref_period = '0-1 ka'  #TODO: Check this. Is this always true?
map_region = 'global'

# Create lat and lon strings
lat_string,lon_string,_,_,_ = functions_presto.select_latlons(lat,lon,map_region,dataset_txt)


#%% ADD LINES TO THE TEMPLATE

lines_output = []
for line in lines_template:
    #
    # Add the settings at the right place.
    if line == '[INSERT SETTINGS]\n':
        for option_txt in options:
            setting_line_txt = '          <li style="font-size:12px">'+option_txt+'</li>\n'
            lines_output.append(setting_line_txt)
    elif line == '[INSERT VARIABLES]\n':
        lines_output.append("      // Set initial variables\n")
        lines_output.append("      var dataset              = '"+dataset_txt+"';\n")
        lines_output.append("      var dataset_name         = '"+method[0]+"';\n")
        lines_output.append("      var dataset_details      = 'Reference period: "+txt_ref_period+"';\n")
        lines_output.append("      var versions_available   = ['"+version_txt+"'];\n")
        lines_output.append("      var variables_available  = ['"+var_txt+"'];\n")
        lines_output.append("      var quantities_available = ['"+quantity_txt.lower()+"'];\n")
        lines_output.append("      var time_min             = "+txt_time_old+";\n")
        lines_output.append("      var time_max             = "+txt_time_new+";\n")
        lines_output.append("      var time_step            = "+txt_time_diff+";\n")
        lines_output.append("      var time_units           = '"+time_units+"';\n")
        lines_output.append("      var regional_means       = true;\n")
        lines_output.append("      var ts_height            = 310;\n")
        lines_output.append("      var map_region           = '"+map_region+"';\n")
        lines_output.append(lat_string[4:]+"\n")
        lines_output.append(lon_string[4:]+"\n")
    else:
        lines_output.append(line)


#%% OUTPUT

# Output the visualizer template
with open(output_dir_full+'visualizer.html','w') as f:
    for line in lines_output:
        f.write(line)


#%% MOVE FILES AND CREATE ZIP

# Move files and create zip
os.system('cp '+web_data_dir+'assets/* '+output_dir_full+'assets/')          # Add the general assets to the visualization folder.
os.chdir(output_dir_full)                                                    # Change directory to the visualization folder.
os.system('zip -r '+output_dir+'viz_'+dataset_txt+'_'+version_txt+'.zip *')  # Zip everything in the visualization folder.

print(' ===== FINISHED script 3: Zipped file saved to: '+output_dir+'viz_'+dataset_txt+'_'+version_txt+'.zip =====')

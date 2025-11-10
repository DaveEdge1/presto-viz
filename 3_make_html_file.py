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

# Ensure directories are absolute paths and exist
data_dir = os.path.abspath(data_dir)
output_dir = os.path.abspath(output_dir)
web_data_dir = os.path.abspath(web_data_dir)

print(f'Data directory: {data_dir}')
print(f'Output directory: {output_dir}')
print(f'Web assets directory: {web_data_dir}')


#%% LOAD DATA


# Load the visualizer template
template_file = os.path.join(web_data_dir, 'visualizer_template.html')
print(f'Loading template from: {template_file}')
if not os.path.exists(template_file):
    raise FileNotFoundError(f'Template file not found: {template_file}')
with open(template_file) as f:
    lines_template = f.readlines()

# Load reconstruction
var_txt      = 'tas'
quantity_txt = 'Annual'
if   'holocene_da' in data_dir: dataset_txt = 'daholocene'; version_txt = data_dir.split('_holocene_da')[0].split('/')[-1]
elif 'graph_em'    in data_dir: dataset_txt = 'graphem';    version_txt = data_dir.split('_graph_em')[0].split('/')[-1]
else:                           dataset_txt = 'lmr';        version_txt = data_dir.rstrip('/').split('/')[-1]
filename_txt = dataset_txt+'_v'+version_txt+'_'+var_txt+'_'+quantity_txt.lower()

# Construct output directory path robustly
output_dir_full = os.path.join(output_dir, 'viz')
print(' ===== STARTING script 3: Making html and zipping '+str(filename_txt)+' =====')

# Load reconstruction data
data_file_path = os.path.join(data_dir, filename_txt + '.nc')
print(f'Loading data from: {data_file_path}')
if not os.path.exists(data_file_path):
    raise FileNotFoundError(f'Data file not found: {data_file_path}')
data_xarray = xr.open_dataset(data_file_path)
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
visualizer_file = os.path.join(output_dir_full, 'visualizer.html')
print(f'Writing visualizer HTML to: {visualizer_file}')
with open(visualizer_file, 'w') as f:
    for line in lines_output:
        f.write(line)


#%% MOVE FILES AND CREATE ZIP

# Verify that script 2 output exists
script2_output = os.path.join(output_dir, 'viz', 'assets', dataset_txt)
print(f'Checking for script 2 output in: {script2_output}')
if not os.path.exists(script2_output):
    raise FileNotFoundError(f'Script 2 output not found. Expected files in: {script2_output}')
print(f'✓ Script 2 output found with {len(os.listdir(script2_output))} files')

# Copy web assets (CSS, JS, images, etc.) to the viz folder
web_assets_src = os.path.join(web_data_dir, 'assets')
web_assets_dest = os.path.join(output_dir_full, 'assets')
print(f'Copying web assets from {web_assets_src} to {web_assets_dest}')
if not os.path.exists(web_assets_src):
    raise FileNotFoundError(f'Web assets not found: {web_assets_src}')
os.system(f'mkdir -p "{web_assets_dest}"')
copy_result = os.system(f'cp -r "{web_assets_src}"/* "{web_assets_dest}/"')
if copy_result != 0:
    print(f'WARNING: Failed to copy web assets (exit code: {copy_result})')

# Create zip file
os.chdir(output_dir_full)
zip_file = os.path.join(output_dir, f'viz_{dataset_txt}_{version_txt}.zip')
print(f'Creating zip file: {zip_file}')
os.system(f'zip -r "{zip_file}" *')

print(f' ===== FINISHED script 3: Zipped file saved to: {zip_file} =====')


#%% COMMIT TO GIT (OPTIONAL, FOR GITHUB PAGES)

# Check if we should commit output (controlled by environment variable)
should_commit = os.environ.get('COMMIT_VIZ_OUTPUT', 'false').lower() == 'true'

if should_commit:
    print(' ===== Preparing visualization output for git commit =====')

    # Get configuration from environment variables (REQUIRED for cross-repo usage)
    git_repo_root = os.environ.get('GIT_REPO_ROOT')
    if not git_repo_root:
        print('ERROR: GIT_REPO_ROOT environment variable must be set when COMMIT_VIZ_OUTPUT=true')
        print('Example: export GIT_REPO_ROOT=/path/to/LMR2')
        sys.exit(1)

    git_branch = os.environ.get('VIZ_GIT_BRANCH', 'gh-pages')
    git_commit_msg = os.environ.get('VIZ_COMMIT_MSG', f'Update visualization: {dataset_txt} v{version_txt}')
    viz_output_path = os.environ.get('VIZ_OUTPUT_PATH', 'docs')  # Changed default to 'docs' for GitHub Pages

    # Resolve absolute paths
    viz_source = os.path.abspath(os.path.join(output_dir, 'viz'))
    viz_dest = os.path.abspath(os.path.join(git_repo_root, viz_output_path))

    print(f'Source: {viz_source}')
    print(f'Destination: {viz_dest}')
    print(f'Target repo: {git_repo_root}')
    print(f'Target branch: {git_branch}')

    # Create destination directory
    os.system(f'mkdir -p "{viz_dest}"')

    # Copy visualization output to target repo location
    print(f'Copying visualization files...')
    copy_result = os.system(f'cp -r "{viz_source}"/* "{viz_dest}/"')

    if copy_result != 0:
        print('ERROR: Failed to copy visualization files')
        sys.exit(1)

    # Change to target repository for git operations
    os.chdir(git_repo_root)
    print(f'Changed to git repository: {os.getcwd()}')

    # Git operations
    print(f'Staging changes in {viz_output_path}/')
    os.system(f'git add "{viz_output_path}/"')

    print(f'Creating commit: {git_commit_msg}')
    commit_result = os.system(f'git commit -m "{git_commit_msg}"')

    if commit_result == 0:
        print(f'Commit successful. Pushing to {git_branch}...')
        push_result = os.system(f'git push origin "{git_branch}"')
        if push_result == 0:
            print(' ===== Successfully pushed visualization to GitHub Pages =====')
        else:
            print(' ===== WARNING: Git push failed. Check permissions and branch protection. =====')
    else:
        print(' ===== No changes to commit (output may be identical to previous version) =====')
else:
    print(' ===== Skipping git commit (set COMMIT_VIZ_OUTPUT=true to enable) =====')

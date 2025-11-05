#!/bin/bash
#SBATCH --job-name=presto_viz            # Name of the job
#SBATCH --output=logfile_presto_viz.txt  # File for output and errors
#SBATCH --time=1:00:00                   # Maximum time for job to run
#SBATCH --mem=5000                       # Memory (MB)

# This script can be run on Monsoon with the command: sbatch run_script.sh.

# Set directory for reconstruction data
# data_dir='/root/presto/userRecons/17095846334578667_HoloceneDA/'
data_dir="/root/presto/userRecons/$1/"

# Set directories (these only need to be set once.)
# output_dir='/root/presto/userRecons/17095846334578667/'  # The output files will be saved here
output_dir="/root/presto/userRecons/$1/"

web_data_dir='/root/presto/viz/web_assets/'  # This directory contains an html template and supporting files.

# Run scripts
echo "=== Processing reconstruction ==="

# Function to run a script with timeout and monitoring
run_python_with_timeout() {
    local script_path="$1"
    local script_args="$2"
    local timeout_seconds="$3"
    local log_file="$4"
    local script_name="$5"
    
    echo "Starting $script_name with timeout of $timeout_seconds seconds" | tee -a "$log_file"
    echo "$(date): Starting $script_name (PID will be shown when started)" >> "$log_file"
    
    # Run the Python script WITHOUT the broken pipeline
    # Direct execution to avoid tee/background issues
    timeout $timeout_seconds python -u "$script_path" $script_args >> "$log_file" 2>&1
    local exit_code=$?

    echo "$(date): Python process exited with code: $exit_code" >> "$log_file"
    
    if [ $exit_code -eq 124 ]; then
        echo "$(date): ERROR - $script_name timed out after $timeout_seconds seconds" | tee -a "$log_file"
        echo "Memory at timeout: $(free -h | grep Mem)" >> "$log_file"
        return 124
    elif [ $exit_code -eq 141 ]; then
        echo "$(date): ERROR - $script_name received SIGPIPE (broken pipe)" | tee -a "$log_file"
        echo "Memory at SIGPIPE: $(free -h | grep Mem)" >> "$log_file"
        return 141
    elif [ $exit_code -ne 0 ]; then
        echo "$(date): ERROR - $script_name failed with exit code: $exit_code" | tee -a "$log_file"
        echo "Memory at failure: $(free -h | grep Mem)" >> "$log_file"
        return $exit_code
    else
        echo "$(date): SUCCESS - $script_name completed successfully" | tee -a "$log_file"
        echo "Memory at completion: $(free -h | grep Mem)" >> "$log_file"
        return 0
    fi
}

source /root/miniconda3/etc/profile.d/conda.sh &&
conda activate presto_env && pip install psutil &&
# Script 1: Format data (15 minute timeout)
run_python_with_timeout "/root/presto/viz/1_format_data_daholocene_graphem.py" "$data_dir" 900 "$output_dir/1_format_data_full.log" "1_format_data" &&

# Script 2: Make maps and time series (120 minute timeout - this is the hanging one)
# Monitor system resources during execution
echo "=== Starting Script 2 with resource monitoring ===" >> "$output_dir/resource_monitor.log"
dmesg | tail -20 >> "$output_dir/resource_monitor_pre.log" 2>&1  # Check for OOM killer before running

run_python_with_timeout "/root/presto/viz/2_make_maps_and_ts.py" "$data_dir $output_dir" 7200 "$output_dir/2_make_maps_full.log" "2_make_maps"
script2_exit=$?

# Check system logs after script runs
dmesg | tail -50 >> "$output_dir/resource_monitor_post.log" 2>&1  # Check for OOM killer, process killed, etc
echo "Script 2 exit code: $script2_exit" >> "$output_dir/resource_monitor.log"

# Continue if successful
if [ $script2_exit -eq 0 ]; then
    echo "Script 2 completed successfully" >> "$output_dir/resource_monitor.log"
else
    echo "Script 2 failed with exit code $script2_exit" >> "$output_dir/resource_monitor.log"
    exit $script2_exit
fi

# Script 3: Make HTML file (10 minute timeout)
run_python_with_timeout "/root/presto/viz/3_make_html_file.py" "$data_dir $output_dir $web_data_dir" 600 "$output_dir/3_make_html_full.log" "3_make_html" &&

conda activate base

echo "=== All processing complete. Visualizations are stored in $output_dir ==="

#!/bin/bash

### On the first start of the start.bash ever he will replace original GinJinn
### with our GinJinn. All activations later this does nothing so we suppress
### the error > /dev/null 2>&1
 mv -f /workspace/predictors.py /opt/conda/envs/ginjinnCPU/lib/python3.7/site-packages/ginjinn/predictor/predictors.py  > /dev/null 2>&1

### This is the workflow that always runs
### We produce the output
### We move into the mirrored folder
### We delete the temporary files
### ATTENTION! After the program ran
### empty the results folder!

/opt/conda/bin/python3 -u /workspace/DaphniaDetector/Main.py

# Always empty results so mv can be performed later
rm -rf /workspace/results/*

# Check if status is 1 or 0
# Path to the YAML file
yaml_file="/workspace/DaphniaDetector/settings_segment.yml"

# Read the value of status_flag from the YAML file
status_flag=$(grep -E '^status_flag: [0-9]+' "$yaml_file" | awk '{print $2}')

rm -rf /workspace/images/temp_directory

# Check if status_flag is 1 or 0
if [[ "$status_flag" == 1 ]]; then
    echo "status_flag is 1, no files are moved or deleted until the next execution"
    
elif [[ "$status_flag" == 0 ]]; then
    echo "status_flag is 0"
        # Perform the file operations
    mv -f /workspace/results_temp/* /workspace/results
    rm -rf /workspace/JPG
    rm -rf /workspace/results_temp
    
else
    echo "status_flag is neither 1 nor 0. Please set it to 0 in segments.yml"
fi




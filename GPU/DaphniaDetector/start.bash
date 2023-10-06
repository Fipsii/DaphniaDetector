#!/bin/bash

### On the first start of the start.bash ever he will replace original GinJinn
### with our GinJinn. All activations later this does nothing so we suppress
### the error > /dev/null 2>&1
 mv -f /app/predictors.py /opt/conda/envs/ginjinnCPU/lib/python3.7/site-packages/ginjinn/predictor/predictors.py  > /dev/null 2>&1

### This is the workflow that always runs
### We produce the output
### We move into the mirrored folder
### We delete the temporary files
### ATTENTION! After the program ran
### empty the results folder!

/usr/bin/python3 -u /app/DaphniaDetector/Main.py
mv -f /app/results_temp/* /app/results
rm -rf /app/JPG
rm -rf /app/results_temp
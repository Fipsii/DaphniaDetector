#!/bin/bash

### This files checks the for processes that need CUDA
### 1 The base installation aka do we even get the GPU information into the docker
### most common error is missing the --gpus=all command in docker run
### 2 The pytorch of ginjinn and its compatability
### 3 The pytorch of DaphniaDetection (used by EasyOCR for scale detection) and its compatability
### 4 The tensorflow of DaphniaDetection (used for Classification) and its compatability 

# Command 1: Check NVIDIA GPU using nvidia-smi
echo "Checking CUDA base installation..."
nvidia-smi > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "Finished: CUDA base installation successfull"
else
    echo "ERROR: CUDA base installation not successfull. Did you specify --gpus=all in the docker run command?"
fi

# Command 2: Check CUDA availability in the 'base' environment
echo "Checking base enviornment..."
/opt/conda/bin/python3 -c "import torch; print('Finished: CUDA connected with pytorch' if torch.cuda.is_available() else 'ERROR: CUDA not connected with pytorch')"

# Command 3: Check CUDA availability in the 'ginjinnGPU' conda environment
echo "Checking conda environment "Ginjinn"..."
/opt/conda/envs/ginjinnGPU/bin/python -c "import torch; print('Finished: CUDA connected with ginjinn' if torch.cuda.is_available() else 'ERROR: CUDA not connected with pytorch')"

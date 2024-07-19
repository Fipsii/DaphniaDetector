## Installation in non Docker format:
### Prerequisites 

This program needs a CUDA capable GPU with at least CUDA 11.8 capability and installation based on our environments.

Before installing the environments we strongly recommended installing libmamba solver:

```bash
conda install -n base conda-libmamba-solver
conda config --set solver libmamba
```

Install the two environments in conda with:

```bash
conda env create -f environment.yml
conda env create -f environmentGinJinn.yml

```

## Setup

The first task is to get the models and code [here](https://drive.google.com/drive/folders/1f-Cy8brF1IWGREfcxsWaFhh9HCpHBl6m?usp=sharing).

Once downloaded you need a few steps to configure the files to your personal environment:

#### The GinJinn model .yaml files require you to set the uppermost path according to your system.

Set this to the absolute path where you can find the model folders in your system.
The paths to images and outputs should already be configured for you.

![image](https://github.com/Fipsii/DaphniaDetector/blob/main/settings_config.png?raw=true)

Optional:

You have the choice to run the program in CPU (in that case we recommend the CPU docker), where you would have
to set flags in the ginjinn models according to the [ginjinn2 documentation](https://ginjinn2.readthedocs.io/en/latest/).
For GPU set ups CUDA 11.8 compatability is required.

Now that ginjinn is installed you need to implement non-max suppression and exchange the original ginjinn with our changed version:

```bash

mv -f /PATH/TO/predictors.py /PATH/TO/conda/envs/ginjinnCPU/lib/python3.7/site-packages/ginjinn/predictor/predictors.py  > /dev/null 2>&1

```
## Workflow

If all paths are set and dependiencies are installed you can start the program by:


```bash
conda run -n ENVIRONMENT python DaphniaDetector Main.py
```
OR

```bash
conda activate ENVIRONMENT
python DaphniaDetector Main.py
```

The code will now start to calculate. After object detection that should be checked, so you will get the question:

```bash
Do you want to check your labels? This is highly advised. Extract your data from {Analysis_data} [y(es)/n(o)/c(ancel)]:          
```

Yes is the advised option which stops the code and allows to check data in a labelling program of your choice (we used [CVAT](https://www.cvat.ai/)

After checking the data and getting a new annotations.json you have to replace the old annotations.json in the results folder. Now you can rexecute the code with:

```bash
conda run -n ENVIRONMENT python DaphniaDetector Main.py
```

And it will continue after the object detection.

If you select no data might have false values, but no rexecution is needed.
Cancel exits the code without saving data.

At successfull end of the code you will get the message

```bash
Resetting yaml to start configuration
Finished
```

Now you will find you data in the results folder.
IMPORTANT: Always keep the results folder empty before you start a new analysis
```
results/
├── crops/
│   ├── eye/
│   │   └── images
│   ├── body/
│   │   └── images
│   └── ...
├── segmentation/
│   ├── annoations.json
│   └── visualization/
│    │   └── images
├── measurement vis/
│   └── images
├── visualization/
│   └── images (object detection)
├── scale.csv
├── results.csv
├── annotations.json(object detection)
└── annotations.csv (object detection)
```

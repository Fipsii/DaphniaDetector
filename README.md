## Daphnia Detection and Analysis: General

Our program is an automated pipeline for scientific analysis of animals of the genus *Daphnia*. Detailed information about development and functionality in PAPERLINK.
This repository is the non-Docker version. We provide packages with version we installed, but functionality is dependend on the cuda version your system has.

![image](https://github.com/Fipsii/DaphniaDetector/blob/main/Zeichnung4.png?raw=true)

For the corresponding docker images go to: #####

If you came here from docker to get the files and folders for the mount: ![Here they are](IRGENDEINEURL) and follow the instructions on the corresponding docker pages.
If you want to got to docker go to ![CPU-Version](https://hub.docker.com/repository/docker/fipsik/fullimage/general) or ![GPU-Version](PLACEHOLDER).

## Installation in non Docker format:
### Prerequisites 

Environment installation with conda is possible with supplied .yaml files

Settings.yaml: available at GITHUBLINK

## Setup

The first part for the non docker container is to set the paths for models and the settings.
The GinJinn model .yaml files require you to set the uppermost path according to your system.
The Paths to images and outputs have to be put into the settingsfile.

You have the choice to run the program in CPU (in that case we recommend the CPU docker), where you would have
to set flags in the ginjinn models according to the ![ginjinn2 documentation](https://ginjinn2.readthedocs.io/en/latest/).
For GPU set ups CUDA 11.8 compatability is required.

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

Yes is the advised option which stops the code and allows to check data in a labelling program of your choice (we used ![CVAT](https://www.cvat.ai/)

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

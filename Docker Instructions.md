## Daphnia Detection and Analysis: General


### ATTENTION: The Program and site is under heavy construction. We cannot guarantee the accuracy of instruction and full functionality of the code. Please report bugs to [github](https://github.com/Fipsii/DaphniaDetector) or mail (see below). 

Our program is an automated pipeline for scientific analysis of animals of the genus *Daphnia*. Detailed information about development and functionality are in the [paper](dummylink) and [github](https://github.com/Fipsii/DaphniaDetector). 

This repository contains a CPU and GPU version which can be accessed by either:

## Prerequisites

Our program either needs a CPU or a GPU with CUDA 11.8 capability and a fitting driver and - of course - docker engine. 
We also strongly recommend a labelling program for a manual control step.

## Base concept 
The program pipeline is designed to take two inputs. A folder containing the images to be analyzed (image folder) and the settings (settings_segment.yaml), which can be downloaded in this github.
All other files will be pulled by docker. Based on these input an output folder will be created (see bottom).


```
.../
├── settings_segment.yaml
├── output folder/
│   └── EMPTY
└── image folder/
    └── images
```
Fig 1. The necessary folder structure 

## Setup

First pull the image you want, either CPU or GPU with the following commands:

```bash
docker pull fipsik/daphniadetector:cpu
```
or

```bash
docker pull fipsik/daphniadetector:gpu
```

After successfull pulling of the docker create the input/output as seen in Fig. 1

For the docker to work mount the input/output files to the locations in the docker:

The files we mount are the following:

- An empty folder called results in which we write the output
- An folder called images in which put the images we want to analyze for every analysis
- A settings .yaml, which is available on [github](https://github.com/Fipsii/DaphniaDetector)

For easy management group these files into a analysis folder in which you operate.
Now you have all the goods needed to build the container. The container commands are slightly different between the cpu and gpu version, but follow dockers basic syntax:

- first enter docker run, 
- --name for the name of the container,
- -it to start in interactive mode,
- (The GPU version requires to add --gpus=all after the -it command),
- -v to mount the input/output files with a ":" the links the path in your system with the path in the docker, which you should not change,
- denote the image we want to run the container with as fipsik/daphniadetector:TAGNAME,
- /bin/bash


This is how the commands should look like for you:
### CPU: 
```bash
docker run --name CONTAINERNAME -it -v ${PWD}/images/:/workspace/images -v ${PWD}/settings_segment.yml:/workspace/DaphniaDetector/settings_segment.yml -v ${PWD}/results:/workspace/results fipsik/daphniadetector:cpu /bin/bash
```
### GPU:
```bash
docker run --name CONTAINERNAME -it --gpus=all -v ${PWD}/images/:/workspace/images -v ${PWD}/settings_segment.yml:/workspace/DaphniaDetector/settings_segment.yml -v ${PWD}/results:/workspace/results fipsik/daphniadetector:gpu /bin/bash
```

### Check GPU Version (Skip this step if you use the CPU version)

Now you should be in the docker interface, in which you can check if your configuration allows GPU usage with the following command:

```bash
bash DaphniaDetector/gpu_check.bash
```
This outputs these messages if successfull:

```bash
Checking CUDA base installation...
Finished: CUDA base installation successfull

#This gives an error right now eventhough it works
Checking base environment...
Finished: CUDA connected with pytorch

Checking conda environment Ginjinn...
Finished: CUDA connected with  pytorch
```

Now you can leave the docker interface with:

```bash
exit
```

Your setup should now be complete and you can start with analysis.

## Workflow

### Set settings

First you have to set some settings in the .yaml file downloaded earlier. This file starts with preset settings which you can change.
Do NOT change paths or the status_flag these should be fix or change automatically.

Next input images to to be analyzed into the images folder and start the docker:

```bash
docker start CONTAINERNAME
```

### Start program

```bash
docker exec -it CONTAINERNAME DaphniaDetector/start.bash                                
```

-it is necessary to interactive with the script messages during the code.
Otherwise you will get and EOL error.

The code will now start to calculate. You will get the question:

```bash
Do you want to check your labels? This is highly advised. Extract your data from {Analysis_data} [y(es)/n(o)/c(ancel)]:          
```

Yes is the advised option which stops the code mirrors contents into the result mount. And allows the user to manually
check the bounding box data for errors.

After checking the data and getting a new annotations.json you have to replace the old annotations.json in the results folder.

and rexecute:

```bash
docker exec -it CONTAINERNAME /DaphniaDetector/start.bash                                
```

If you select no data might have false values, but no reexecution is needed.

Cancel exits the code without saving data.

At successful end of the code you will get the message

```bash
Finished detection. You may have to wait shortly for docker to transfer results onto your local machine.
```

Now you will find you data in the results folder.

IMPORTANT NOTES:

- Always keep the results folder empty before you start a new analysis
- If the program is manually quit check the settings.yml settings, it might be necessary to reset the status_flag to 0 and the original_images folder back from /workspace/JPG to /workspace/images 

Credit: Philipp Kropf, Magdalena Mair and Matthias Schott, Corresponding mail: matthias.schott[at]uni-bayreuth.de
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
├── annotations.csv (object detection)
└── time_and_hardware_info.txt 
```

# Swim2DPose

This is a repository for a university group project. This project contains a pipeline that can take a set of input images/videos and output the same videos with pose estimation annotations on them. We provide 4 models/libraries integrated into the pipeline. We also provide details on how to add your own pose estimation models to the pipeline.

## Contents

### Setup And Pipeline

[Setup](#setup)

[Pipeline](#pipeline)

### Supported Models

[AlphaPose](#alphapose)

[OpenPose](#openpose)

[DWPose](#dwpose)

[YoloNasNet](#yolonasnet)

## Setup

This project contains dockerfiles for each model environment and one for the pipeline (the pipeline docker file includes DWPose and YoloNasNet). This is to ensure the project stays modular allowing people to add new models using docker easily.

### Setup Through Docker

Ensure you have **Docker** installed on your system as well as the **Nvidia Container Toolkit** (This does mean this project is **only compatible with Nvidia GPU's**).

Before you pull the images **please make sure you have enough disk space available**. Due to the heavy libraries we are using the images are large in size (approximate sizing is outlined in each image title below).

---

#### Pipeline Docker Image (~20GB)

Pull the pipeline Docker image from Docker Hub using this command:

```
docker pull surfytom/pipeline:latest
```
Once this has been pulled you can follow the steps in the [pipeline](#pipeline) section to get started

---

The previous setup steps will only give you access to the pipeline, DWPose and YoloNasNet. To use AlphaPose and OpenPose please clone their respective images using the commands below:

#### AlphaPose Docker Image (~20GB):
```
Write command to pull AlphaPose image
```
#### OpenPose Docker Image (~10GB):
```
Write command to pull OpenPose image
```
## Pipeline

The pipeline is the central part of this repository. It offers a command line interface that allows you to pass videos to each of the 4 models integrated into the pipeline with added flags and options to configure the output to what you want.

![Pipeline Overview Image](https://github.com/Surfytom/Swim2DPose/blob/main/docmedia/DockerPipelineImage.png "Pipeline Overview")

### Starting The Pipeline Container

If you have never run the pipeline container and it does not exist in your list of containers on Docker then run this command:
```
docker run -it --gpus all -v "/path/to/folder/with/videos:/usr/src/app/media" -v "//var/run/docker.sock://var/run/docker.sock" pipeline:latest
```

The pipeline will save results to the first mounted volume. If you want to mount a new folder please delete the pipeline **CONTAINER** (not image) and then run the command above again with a new path.

If you have already got a pipeline container within the Docker container list that you want to run again please use the following command:
```
docker start pipeline && docker exec -it pipeline bash
```

After you are done type ```exit``` within the container to leave it and then type ```docker stop pipeline``` to stop the pipeline container.

### Your First Command

Now you are in the pipeline container you should be in the Swim2DPose directory. If your not please type  
```cd /usr/src/app/Swim2DPose```.

Now that your defintely in the Swim2DPose directory run the command below to use AlphaPose on the videos in your mounted folder:
```
python Pipeline/main.py -m AlphaPose -i path/to/video1.mp4 path/to/video2.mp4 -s path/to/save/folder
```
```-m Alphapose``` tells the script to select AlphaPose as the pose model

```-i path/to/video.mp4``` specifies a list of paths to media you want to process (this can be one or more media)

```-s``` saves the output videos to the specified folder. If no argument is passed videos will be saved to 'Pipeline/results'

Consult [the pipeline read me](https://github.com/Surfytom/Swim2DPose/blob/main/Pipeline/PipelineREADME.md) for more information and all available configuration flags.

## AlphaPose

Please consult the [AlphaPose readme](https://github.com/Surfytom/Swim2DPose/blob/main/Pipeline/AlphaPoseLib/README.md) for information on getting started with the library.

## OpenPose

Please consult the [OpenPose readme](https://github.com/Surfytom/Swim2DPose/blob/main/Pipeline/DWPoseLib/DWPoseREADME.md) for information on getting started with the library.

## DWPose

Please consult the [DWPose readme](https://github.com/Surfytom/Swim2DPose/blob/main/Pipeline/DWPoseLib/DWPoseREADME.md) for information on getting started with the model.

## YoloNasNet

Please consult the [YoloNasNet readme](https://github.com/Surfytom/Swim2DPose/blob/main/Pipeline/YoloNasNetLib/YoloNasNetREADME.md) for information on getting started with the model.

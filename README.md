# Swim2DPose

This is a repository for a university group project. This project contains a pipeline that can take a set of input images/videos and output the same videos with pose estimation annotations on them. We provide 4 models/libraries integrated into the pipeline. We also provide details on how to add your own pose estimation models to the pipeline.

## Contents

### Setup And Pipeline

[Setup](Setup)

[Pipeline]()

### Supported Models

[AlphaPose]()

[OpenPose]()

[DWPose]()

[YoloNasNet]()

## Setup

This project contains dockerfiles for each model environment and one for the pipeline (the pipeline docker file includes DWPose and YoloNasNet). This is to ensure the project stays modular allowing people to add new models using docker easily.

### Setup Through Docker

Ensure you have **Docker** installed on your system as well as the **Nvidia Container Toolkit** (This does mean this project is **only compatible with Nvidia GPU's**).

Before you pull the images **please make sure you have enough disk space available**. Due to the heavy libraries we are using the images are large in size (approximate sizing is outlined in each image title below).

---

#### Pipeline Docker Image (~20GB)

Pull the pipeline Docker image from Docker Hub using this command:

```
Write command to pull pipeline image
```
Once this has been pulled you can run the container using this command:

```
Write command to run pipeline container
```
Now you will have a command line operating within the container enviroment. Please consult the Pipeline section below or the Pipeline README.md to get started.

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

### Your First Command

Open a terminal in the main Swim2DPose directory and run:
```
python Pipeline/main.py -m AlphaPose -i path/to/video1.mp4 path/to/video2.mp4 -s path/to/save/folder
```
```-m Alphapose``` tells the script to select AlphaPose as the pose model

```-i path/to/video.mp4``` specifies a list of paths to media you want to process (this can be one or more media)

```-s``` saves the output videos to the specified folder. If no argument is passed videos will be saved to 'Pipeline/results'

Consult [the pipeline read me](https://github.com/Surfytom/Swim2DPose/blob/main/Pipeline/PipelineREADME.md) for more information and all available configuration flags.

## AlphaPose



## OpenPose



## DWPose



## YoloNasNet



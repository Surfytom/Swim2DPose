# Swim2DPose

This is a repository for a university group project. This project contains a pipeline that can take a set of input images/videos and output the same images/videos with pose estimation annotations on them. We provide 4 models/libraries integrated into the pipeline.

## Contents

### Setup And Pipeline

[Setup](#setup)

[Pipeline](#pipeline)

### Supported Models

[AlphaPose](#alphapose)

[OpenPose](#openpose)

[DWPose](#dwpose)

[YoloNasNet](#yolonasnet)

### Other

[Evaluation Video Generator](#evaluation-video-generator)

## Setup

This project contains dockerfiles for each model environment and one for the pipeline (the pipeline docker file includes DWPose and YoloNasNet). This is to ensure the project stays modular allowing people to add new models using docker easily.

### Setup Through Docker

Ensure you have **Docker** installed on your system as well as the **Nvidia Container Toolkit** (This does mean this project is **only compatible with Nvidia GPU's**).

Before you pull the images **please make sure you have enough disk space available**. Due to the heavy libraries we are using the images are large in size (approximate sizing is outlined in each image title below).

---

#### Pipeline Docker Image (~10GB)

Pull the pipeline Docker image from Docker Hub using this command:

```
docker pull surfytom/pipeline:latest
```
Once this has been pulled you can follow the steps in the [pipeline](#pipeline) section to get started with DWPose and YoloNas. If you want to use AlphaPose or OpenPose follow the steps below.

---

The previous setup steps will only give you access to the pipeline, DWPose and YoloNasNet. To use AlphaPose and OpenPose please clone their respective images using the commands below:

#### AlphaPose Docker Image (~12GB):
```
docker pull lhlong1/alphapose:latest
```
#### OpenPose Docker Image (~10GB):
```
docker pull lhlong1/openpose:latest
```
## Pipeline

The pipeline is the central part of this repository. It offers a command line interface that allows you to pass videos to each of the 4 models integrated into the pipeline with added flags and options to configure the output to what you want.

![Pipeline Overview Image](https://github.com/Surfytom/Swim2DPose/blob/main/docmedia/DockerPipelineImage.png "Pipeline Overview")

### Starting The Pipeline Container

If you have never run the pipeline container and it does not exist in your list of containers on Docker then run this command:
```
docker run -it --rm --name pipeline --gpus all -v "/path/to/folder/with/videos:/usr/src/app/media" -v "//var/run/docker.sock://var/run/docker.sock" pipeline:latest
```

Only replace the path/to/folder/with/videos. Change nothing else unless you know what your doing.

### Your First Command

Now you are in the pipeline container you should be in the Swim2DPose directory. If your not please type  
```cd /usr/src/app/Swim2DPose```.

Now that your defintely in the Swim2DPose directory run the command below to use AlphaPose on the videos in your mounted folder:
```
python3 Pipeline/main.py -m AlphaPose -i path/to/video1.mp4 path/to/video2.mp4 -s path/to/save/folder/within/mounted/folder
```
```-m Alphapose``` tells the script to select AlphaPose as the pose model

```-i path/to/video.mp4``` specifies a list of paths to media you want to process (this can be one or more media)

```-s``` saves the output videos to the specified folder. If no argument is passed videos will be saved to ```'/usr/src/app/media/results'``` which is the results folder in your mounted folder.

Consult [the pipeline read me](https://github.com/Surfytom/Swim2DPose/blob/main/Pipeline/PipelineREADME.md) for more information and all available configuration flags.

### Label Box Annotation Upload

Label Box model annotation uploading is only supported by the DWPose model at the moment. For more information about how to use it please consult the [Label Box section of the Pipeline read me](https://github.com/Surfytom/Swim2DPose/blob/main/Pipeline/PipelineREADME.md#labelbox).

## AlphaPose

### Preview

![demo video](https://github.com/Surfytom/Swim2DPose/blob/main/Pipeline/AlphaPoseLib/media/demo%20video.gif)

Please consult the [AlphaPose readme](https://github.com/Surfytom/Swim2DPose/blob/main/Pipeline/AlphaPoseLib/README.md) for information on getting started with the library.

## OpenPose

Please consult the [OpenPose readme](https://github.com/Surfytom/Swim2DPose/blob/main/Pipeline/DWPoseLib/DWPoseREADME.md) for information on getting started with the library.

## DWPose

### Preview

![preview video](https://github.com/Surfytom/Swim2DPose/blob/main/docmedia/DivingDWPose.gif)

Please consult the [DWPose readme](https://github.com/Surfytom/Swim2DPose/blob/main/Pipeline/DWPoseLib/DWPoseREADME.md) for information on getting started with the model.

## YoloNasNet

### Preview

![preview video](https://github.com/Surfytom/Swim2DPose/blob/main/docmedia/DivingYoloNasNet.gif)

Please consult the [YoloNasNet readme](https://github.com/Surfytom/Swim2DPose/blob/main/Pipeline/YoloNasNetLib/YoloNasNetREADME.md) for information on getting started with the model.

## Evaluation Video Generator

The folder Evaluation contains an evaluation video generator which can take videos from a folder and randomly select frames within them resulting in a collage of random consecutive frames from a selection of videos. It is useful if you want to evaluate a pose model on different body types and locations quickly as you just have to run a model on one video instead of multiple.

Using it is as simple as running the pipeline container. The running the following command:

```
python3 Evaluation/EvaluationDataGenerator.py -fo folder/to/videos -type mp4 -N 3 -K 10 -C 1 -D 1
```

```-type``` should be a video file type like ```avi``` or ```mp4```

```-N``` Is the number of total videos you want to randomly sample

```-K``` Is the number of consecutive frames to sample from each video. Default is 10

```-C``` Use to set beggining cutoff section. 1 means no frames are ignored. Default is 2 meaning the first half of the video is ignored

```-D``` Use to set ending cutoff section. 1 means no frames are ignored. Default is 1 meaning no frames are ignored

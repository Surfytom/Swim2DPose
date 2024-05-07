# DWPose

The [DWPose repository](https://github.com/IDEA-Research/DWPose) offers a wide range of models. The pipeline docker installs the largest version of their RTM pose models which use the library [MMPose](https://github.com/open-mmlab/mmpose). 

## Setup

As long as the pipeline container has been pulled no additional setup is needed as the pipeline container has the model weights and dependencies needed to use DWPose already installed.

### Config File

The config file within the DWPoseLib folder can be adjusted to point to other weight and config files offered by the DWPose librariy. However, you must make sure these models have the same 133 keypoint output as the model we have installed for you. You must also make sure it is compatible with MMPose.

## Using DWPose

To use DWPose Run the following command within the pipeline container command line:

```
python3 Pipeline/main.py -m DWPose -i path/to/video.mp4
```

If you don't have the pipeline container installed please follow the setup guide in the main [readme](https://github.com/Surfytom/Swim2DPose/blob/main/README.md#pipeline).

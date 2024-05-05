# YoloNasNet

YoloNasNet is run using the [super gradients](https://github.com/Deci-AI/super-gradients) library.

## Setup

If the pipeline container has been pulled from Docker hub all the depedencies for YoloNasNet are installed and ready to go. The model will automatically pull the model weights into a cache when first running the model.

### Config File

The YoloNasNet config file within YoloNasNetLib allows you to adjust some model hyper-parameters.

```iou``` adjusts the tolerance of the noise max suppresion algorithm reducing overlapping poses (these might be caused by estimating the same person twice). Set between a value of ```0.0``` and ```1.0```.

```confidence``` sets the confidence threshold for a pose to be valid. Higher value means the model needs to be more confident about the pose for it to be used. Set between ```0.0``` and ```1.0```.

```model``` can be set to any of the models offered within the super gradients [YoloNasPose read me](https://github.com/Deci-AI/super-gradients/blob/master/YOLONAS-POSE.md). It is defaulted to the large model so if speed is needed smaller models can be used such as ```yolo_nas_pose_s```. A smaller model will be less accurate.

## Using YoloNasNet

To use YoloNasNet Run the following command within the pipeline container command line:

```
python3 Pipeline/main.py -m YoloNasNet
```

If you don't have the pipeline container installed please follow the setup guide in the main [readme](https://github.com/Surfytom/Swim2DPose/blob/main/README.md#pipeline).

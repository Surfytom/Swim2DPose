# AlphaPose
![alt text](logo.jpg)




## Summary

 AlphaPose is an accurate a crowd pose estimator, which has been trained on COCO dataset and MPII dataset. AlphaPose allow users to change the model to suit the user's needs with ease. AlphaPose will visualize the keypoints on the Swimmers and export with a video and its respective keypoints.JSON file. 


 Original AlphaPose Github [link](https://github.com/MVIG-SJTU/AlphaPose/tree/master?tab=readme-ov-file)


## Preview

![alt text](<demo video.gif>)

## How to use it


### QuickStart

```.c
python3 scripts/demo_inference.py --detector yolox-x --cfg configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint pretrained_models/fast_res50_256x192.pth --indir examples/demo/ --outdir examples/demo/results --save_img
```

### How to mount video and images from docker

```.c
docker run --privileged --gpus all -v {path to your video from your local machine}:{path to your video inside the docker machine} -it zhiawei/alphapose:latest
```

Inside the docker:

```.c
python3 scripts/demo_inference.py --detector yolox-x --cfg configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint pretrained_models/fast_res50_256x192.pth --video {path to your video inside the docker machine} --save_video --outdir {output path for the results} --sp --vis
```


### Dockerfile

[Link](dockerfile) to the main dockerfile.


### How to fine-tune AlphaPose models

1. Download the object detection model. Place them into `detector` 
```.c
mkdir -p detector/yolo/data && \
gdown 1PLWU06DzLaVPqzNMfqR0_znRZupkIgW7 -O detector/yolo/data/yolov3-spp.weights
```
2. Download the pose models. Place them into `pretrained_models`. All models and details are available in the [link](https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/MODEL_ZOO.md) by AlphaPose.
```.c
# For Halpe dataset (26 keypoints)
gdown 10QLwqRk334W86KrFuDFXpEw1kbrXHCSP -O pretrained_models/halpe26_fast_res50_256x192.pth

# For MSCOCO dataset
gdown 1DsottUmO-UODGi_OH6cm1euvSnxmpG2N -O pretrained_models/fast_res50_256x192.pth
```
You can fine-tune more using the flags listed in the [link](https://github.com/MVIG-SJTU/AlphaPose/blob/master/docs/run.md) here

### How to train AlphaPose model

If you want to train the model by yourself, please download data from [MSCOCO](https://cocodataset.org/#download) (train2017 and val2017). Download and extract them under ./data, and make them look like this:


```
|-- json
|-- exp
|-- alphapose
|-- configs
|-- test
|-- data
`-- |-- coco
    `-- |-- annotations
        |   |-- person_keypoints_train2017.json
        |   `-- person_keypoints_val2017.json
        |-- train2017
        |   |-- 000000000009.jpg
        |   |-- 000000000025.jpg
        |   |-- 000000000030.jpg
        |   |-- ... 
        `-- val2017
            |-- 000000000139.jpg
            |-- 000000000285.jpg
            |-- 000000000632.jpg
            |-- ... 
```
Train FastPose on mscoco dataset.

```.c
./scripts/train.sh ./configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml exp_fastpose
```

*** At the moment training has not been tested within docker

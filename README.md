# Swim2DPose
This is a repository for a group project. The project is to design a system to automate the annotation of 2D poses within videos of swimmers diving off of a start block. These annotation can then be used to train a pose estimation model.

---

## Adding a new model

The easiest way to add new models is to use models supported by the already installed libraries ultralytics and mmpose. Ultralytics support the yolo architeture of models while mmpose supports a wide range of models specific to pose estimation.

### Ultralytics

A pre-trained model can be easily added by pointing the weights file path on model load to the desired model. However, at the moment masks and bounding boxes are extracted from the model's prediction result so if there metrics are not returned by the new model problems may occur.

### MMPose

MMPose hosts many different pose models. The folder dwpose currently can load and run inference with any MMPose model as long as the correct config file and weight file are given within the init_model function.

---
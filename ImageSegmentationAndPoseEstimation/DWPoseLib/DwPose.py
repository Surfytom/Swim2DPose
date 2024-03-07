import torch
import cv2
import numpy as np

from mmpose.apis import inference_topdown, init_model
from mmpose.utils import register_all_modules

# This script manages everything to do with the dw pose model

register_all_modules()

def InitModel(configFilePath, weightCheckpointPath):
    # Simply creates and return a model based on a weight and config file path
    return init_model(configFilePath, weightCheckpointPath, device='cuda:0' if torch.cuda.is_available() else 'cpu')  # or device='cuda:0'
    
    
def InferenceTopDown(model, imageStack):
    # Runs the model on a set of images returning the keypoints the model detects
    allKeyPoints = []

    for i, images in enumerate(imageStack):
        # Loops through the
        
        allKeyPoints.append([])
        for image in images:
            results = inference_topdown(model, image)

            keyPoints = results[0].pred_instances.keypoints[0]

            keyPoints = np.array(keyPoints).astype(np.intp)
            
            allKeyPoints[i].append(keyPoints)

    return allKeyPoints

if __name__ == "__main__":

    config_file = "DWPoseLib/256x192DWPoseModelConfig.py"

    checkpoint_file = 'DWPoseLib/256x192DWPoseModelModel.pth'
    model = init_model(config_file, checkpoint_file, device='cuda:0')  # or device='cuda:0'

    # please prepare an image with person
    results = inference_topdown(model, 'input_image.jpeg')

    keyPoints = results[0].pred_instances.keypoints[0]

    keyPoints = np.array(keyPoints).astype(np.intp)

    image = cv2.imread("input_image.jpeg")

    shape = np.shape(image)

    for x, y in keyPoints:
        cv2.circle(image, (x, y), 15, (0, 0, 255), -1)

    cv2.imshow("test1", cv2.resize(image, ((shape[0] // 6), (shape[1] // 6))))

    cv2.waitKey(0)
    cv2.destroyAllWindows()
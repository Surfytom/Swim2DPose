import cv2
from ultralytics import YOLO
from ultralytics import settings
import numpy as np
import torch

DEBUG = False
MODEL = None
    
def InitModel(ptFilePath="yolov8n.pt"):

    model = YOLO(ptFilePath)

    if model == None:
        return "Error: Model Not Loaded"

    model.to(device='cuda:0' if torch.cuda.is_available() else 'cpu')

    print(f"Yolo Model loaded to: {model.device}")

    return model

def YOLOSegment(model, imageStack, paddingSize=10):
    
    allResults = []

    for images in imageStack:

        print(f"Video frame amount: {len(images)}")

        results = model(images, max_det=3)

        for i, result in enumerate(results):
            if result.masks == None:
                print(f"{i} Mask none")

        #print(results[0])

        results = [[result.boxes.xyxy.detach().cpu().numpy().astype(np.intp) if result.boxes != None else None, result.masks.xy if result.masks != None else None, result.orig_shape] for result in results]

        allResults.append(results)

    return allResults


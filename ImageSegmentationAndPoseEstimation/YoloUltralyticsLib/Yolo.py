import cv2
from ultralytics import YOLO
from ultralytics import settings
import numpy as np
import torch

DEBUG = False
MODEL = None

def LoadMediaPath(path, stack):

    if ".avi" in path:
    
        videoReader = cv2.VideoCapture(path)

        while True:

            ret, frame = videoReader.read()

            if not ret:
                break
                
            stack.append(frame)
    else:

        stack.append(cv2.imread(path))
    
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

        results = [[result.boxes.xyxy.detach().cpu().numpy().astype(np.intp) if result.boxes != None else None, result.masks.xy if result.masks != None else None, result.orig_shape] for result in results]

        allResults.append(results)

    return allResults

if __name__ == "__main__":

    settings.update({"weights_dir": "YoloUltralyticsLib/Models"})

    inputPaths = [
        "Cohoon, Start, Freestyle, 01_08_2023 08_59_22_5.avi",
        "input_image.jpeg"
    ]

    imageStack = []

    for path in inputPaths:
        LoadMediaPath(path, imageStack)

    model = YOLO("yolov8n.pt")

    # Currently runs all paths on the same call to the model meaning results are merged
    # Mybe seperate so multiple images or videos can be returned when finished instead of when all are finished
    results = model(imageStack)

    results = [[result.boxes.xyxy.detach().cpu().numpy().astype(np.intp), result.orig_shape] for result in results]

    print(f"results: {results}")

    results = padBox(results, 100)

    results = fitToImage(results)

    image = imageStack[0]

    shape = np.shape(image)

    results = padBox(results, 100)

    results = fitToImage(results)

    x, y, x1, y1 = results[0][0][0]

    print(f"xyxy of showcase image and first box: x {x} | y {y} | x1 {x1} | y1 {y1}")

    cv2.rectangle(image, (x, y), (x1, y1), (255, 0, 0), 5)

    cv2.imshow("test", cv2.resize(image, ((shape[1] // 3), (shape[0] // 3))))

    image = imageStack[-1]

    shape = np.shape(image)

    results = padBox(results, 100)

    results = fitToImage(results)

    x, y, x1, y1 = results[-1][0][0]

    print(f"xyxy of showcase image and first box: x {x} | y {y} | x1 {x1} | y1 {y1}")

    cv2.rectangle(image, (x, y), (x1, y1), (255, 0, 0), 5)

    cv2.imshow("test1", cv2.resize(image, ((shape[0] // 6), (shape[1] // 6))))

    cv2.waitKey(0)
    cv2.destroyAllWindows()


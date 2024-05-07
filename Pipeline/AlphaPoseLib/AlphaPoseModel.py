import docker
import json
import numpy as np  
import cv2
import pathlib
from datetime import datetime

client = docker.from_env()
client.pipelineContainerId = client.containers.list(filters={"name": "pipeline"})[-1].id
print("Pipeline Container ID: ", client.pipelineContainerId)

def InitModel(config):

    alphaposeContainer = client.containers.list(all=True, filters={"name": "alphapose"})

    if (len(alphaposeContainer) == 1):
      # Alphapose container already exists
      container = alphaposeContainer[-1]
      container.start()
    else:
      container = client.containers.run(**config)
    # Print container ID
    print("Container ID:", container.id)

    print(container.logs().decode('utf-8'))
    # Stop the container

    # container.stop()
    return container

def LoadConfig(mountedPath):
    # Define the container configuration
    return {
        'image': 'lhlong1/alphapose:latest',
        'detach': True,
        'name': 'alphapose',
        'device_requests': [
            docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])
        ],
        'volumes_from': [client.pipelineContainerId],
        'stdin_open': True,  # Keep STDIN open even if not attached (-i)
        'tty': True,  # Allocate a pseudo-TTY (-t)
        'auto_remove': True
    }

def Inference(model, videoNames, stopAfterExecuting=True):

    allResultPaths = []

    datestring = '{:%d-%m-%Y_%H-%M-%S}'.format(datetime.now())

    for i, video in enumerate(videoNames):
        videoNameAndExt = video['name'] + video['ext']

        print(f"Video '{videoNameAndExt}' running through AlphaPose")

        model.exec_run(cmd=f'python3 scripts/demo_inference.py --detector yolox-x --cfg configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint pretrained_models/halpe26_fast_res50_256x192.pth --video "/usr/src/app/media/{videoNameAndExt}" --save_video --outdir "/usr/src/app/media/results/AlphaPose/{datestring}" --sp --vis_fast')
        print(f"Video {videoNameAndExt} Finished")

        allResultPaths.append(f'/usr/src/app/media/results/AlphaPose/{videoNameAndExt}')

    if stopAfterExecuting == True:
        Stop(model)

    return allResultPaths

def GetBGRColours():

    # colours = [
    #     (75, 25, 230), (75, 180, 60), (25, 225, 255), (216, 99, 67), (49, 130, 245),
    #     (180, 30, 145), (244, 212, 66), (230, 50, 240), (69, 239, 191), (212, 190, 250),
    #     (144, 153, 70), (255, 190, 220), (36, 99, 154), (200, 250, 255), (0, 0, 128),
    #     (195, 255, 170), (0, 128, 128), (177, 216, 255), (117, 0, 0), (169, 169, 169),
    #     (255, 255, 255), (0, 0, 0), (128, 0, 0)
    # ]

    colours = [
        (75, 25, 230), (75, 180, 60), (25, 225, 255), (216, 99, 67), (49, 130, 245),
        (180, 30, 145), (244, 212, 66), (230, 50, 240), (69, 239, 191), (212, 190, 250),
        (144, 153, 70), (255, 190, 220), (36, 99, 154), (200, 250, 255), (0, 0, 128),
        (195, 255, 170), (0, 128, 128), (177, 216, 255), (117, 0, 0), (169, 169, 169),
        (255, 255, 255), (0, 0, 0), (128, 0, 0),
        (255, 0, 128), (0, 255, 128)  # Two visually distinct colors added
    ]

    for colour in colours:
        yield colour

def DrawKeypoints(inputStack, resultPaths, bboxStack, stride=1, drawKeypoints=True, drawBboxes=True, drawText=True, drawEdges=True):

    print("results path: ", resultPaths)

    with open(f"/usr/src/app/Swim2DPose/Pipeline/keypointGroupings.json", "r") as f:
        keypointGroups = json.load(f)
    
    selectedPoints = keypointGroups["AlphaPose"]

    keypointResults = None
    for i, path in enumerate(resultPaths): 
        with open(f"{path}/alphapose-results.json", "r") as f:
            keypointResults = json.load(f)   
            for j, arrItem in enumerate(keypointResults):     
                print(arrItem["image_id"])    
                imageIdx = int(arrItem["image_id"].replace(".jpg", ""))          
                keyPointArray = np.array(arrItem['keypoints']).astype(np.uint8).reshape((26, 3))
                bboxes = np.array(arrItem['box']).astype(np.uint8)
                # Zeros the confidence score     
                print("index: ", imageIdx, " keypoints: ", np.array(keyPointArray))

                selectedKeyPoints = []
                
                images = inputStack[i]['images']
                # keyPoints = keypointResults[i]

                selectedKeyPoints.append([])

                # for j in range(len(bboxes)):

                # keyPointArray = keyPoints[j]
                # box = bboxes[j]

                selectedKeyPoints[i].append([])

                x, y, x1, y1 = bboxes

                # loopLength = stride if (j*stride)+stride < len(images) else len(images) - (j*stride)

                # #print(f"loop length: {loopLength}")

                # for p in range(loopLength):
                    #print(f"p value: {(j*stride)+p}")

                    #print(f"keypoint array: {keyPointArray}")

                if len(keyPointArray) != 0:

                    isDrawn = np.zeros((keyPointArray.shape[0], 1), dtype=np.uintp)

                    for t, (keyX, keyY, keyZ) in enumerate(keyPointArray):

                        if drawText:
                            cv2.putText(images[imageIdx], f"{t}", ((x+int(keyX))-10, (y+int(keyY))-10), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 0), 2)
                        if drawKeypoints:
                            cv2.circle(images[imageIdx], (x + int(keyX), y + int(keyY)), 3, (255,255,255), -1)
                            isDrawn[t] = 1
                        if drawBboxes:
                            cv2.rectangle(images[imageIdx], (x, y), (x1, y1), (0, 0, 255), 2)

                            selectedKeyPoints[i][j].append([(x+keyX), (y+keyY)])
                        # colourGen = GetBGRColours()

                    # if drawEdges:
                    #     for edgeGroup in edgeLinks:
                    #         for point1, point2 in edgeLinks[edgeGroup]:
                    #             cv2.line(images[(j*stride)+p], ([x, y] + keyPointArray[selectedPoints[point1]]), ([x, y] + keyPointArray[selectedPoints[point2]]), next(colourGen), 3)        
                    

    return selectedKeyPoints

def Stop(model):
    model.stop()
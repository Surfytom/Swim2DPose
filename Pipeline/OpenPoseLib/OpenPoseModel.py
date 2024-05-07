import subprocess
import docker
from datetime import datetime
import os

client = docker.from_env()
client.pipelineContainerId = client.containers.list(filters={"name": "pipeline"})[-1].id
print("Pipeline Container ID: ", client.pipelineContainerId)

def InitModel(config):    
    openposeContainer = client.containers.list(all=True, filters={"name": "openpose"})

    if (len(openposeContainer) == 1):
      # Alphapose container already exists
      container = openposeContainer[-1]
      container.start()
    else:
      container = client.containers.run(**config)

    # Print container ID
    print("Container ID:", container.id)

    print(container.logs().decode('utf-8'))
    # Stop the container

    # container.stop()
    return container

def LoadConfig(args):
    # Define the container configuration
    return {
        'image': 'lhlong1/openpose:latest',
        'detach': True,
        'name': 'openpose',
        'device_requests': [
            docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])
        ],
        'volumes_from': [client.pipelineContainerId],
        'stdin_open': True,  # Keep STDIN open even if not attached (-i)
        'tty': True,  # Allocate a pseudo-TTY (-t)
        'auto_remove': True
    }

def Inference(model, videoNames, folderPath, stopAfterExecuting=True):

    allKeyPoints = []
    if os.path.exists(f'{folderPath}/results/OpenPose') == False:
        # Make results folder if it does not exist
        os.mkdir(f'{folderPath}/results/OpenPose')

    dateTimeF = '{:%d-%m-%Y_%H-%M-%S}'.format(datetime.now())

    if os.path.exists(f'{folderPath}/results/OpenPose/{dateTimeF}') == False:
        # Makes a new folder wtih the new highest run digit
        os.mkdir(f"{folderPath}/results/OpenPose/{dateTimeF}")

    for i, video in enumerate(videoNames):
        videoNameAndExt = video['name'] + video['ext']
        videoName = video['name']
        
        print(f"Video {videoNameAndExt} running through OpenPose")
        # Runs the model on a set of images returning the keypoints the model detects
        model.exec_run(cmd=f'build/examples/openpose/openpose.bin --video "/usr/src/app/media/{videoNameAndExt}" --display 0 --hand --write_json "{folderPath}/results/OpenPose/{dateTimeF}/{videoName}" --write_video "/usr/src/app/media/results/OpenPose/{dateTimeF}/{videoName}.avi"')
        print(f"Video {folderPath}/OpenPose/{dateTimeF}/{videoName}.avi is ready")
        print(f"Keypoints {folderPath}/results/OpenPose/{dateTimeF}/{videoName}.json is ready")

    # allResultPaths.append(f'/usr/src/app/media/results/AlphaPose/{videosPath}/{videoName}')
    if stopAfterExecuting == True:
        Stop(model)

    return allKeyPoints

def Stop(model):
    model.stop()
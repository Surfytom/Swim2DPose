import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install("docker")
import docker
client = docker.from_env()

def InitModel(config):    
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
        'device_requests': [
            docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])
        ],
        'volumes': {
            f'{args.inputpaths}': {
                'bind': '/data',
                'mode': 'rw'
            }
        },
        'stdin_open': True,  # Keep STDIN open even if not attached (-i)
        'tty': True,  # Allocate a pseudo-TTY (-t)
    }

def Inference(model, videoNames, stopAfterExecuting=True):

    allKeyPoints = []

    for i, videoName in enumerate(videoNames):
        # Runs the model on a set of images returning the keypoints the model detects
        model.exec_run(cmd=f'./build/examples/openpose/openpose.bin --video "/data/{videoName}.mp4" --display 0 --hand --write_json "{videoName}"  --write_video "{videoName}.avi"')
        subprocess.run(["docker", "cp", f"{model.id}:/openpose/{videoName}.avi", f'./OpenPoseLib/results/{videoName}.avi'])
        subprocess.run(["docker", "cp", f"{model.id}:/openpose/{videoName}", f'./OpenPoseLib/keypoints/{videoName}'])
    
    if stopAfterExecuting == True:
        Stop(model)
        

    return allKeyPoints

def Stop(model):
    model.stop()
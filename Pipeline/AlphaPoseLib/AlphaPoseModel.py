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
        'image': 'lhlong1/alphapose:latest',
        'detach': True,
        'device_requests': [
            docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])
        ],
        'volumes': {    
            f'{args.folder}': {
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
        model.exec_run(cmd=f'python3 scripts/demo_inference.py --detector yolox-x --cfg configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint pretrained_models/halpe26_fast_res50_256x192.pth --video "/data/{videoName}.mp4" --save_video --outdir examples/saved/ --sp --vis_fast')
        subprocess.run(["docker", "cp", f"{model.id}:/build/AlphaPose/examples/saved/AlphaPose_{videoName}.mp4", f'./AlphaPoseLib/results/AlphaPose_{videoName}.mp4'])
        subprocess.run(["docker", "cp", f"{model.id}:/build/AlphaPose/examples/saved/alphapose-results.json", f'./AlphaPoseLib/keypoints/AlphaPose_{videoName}.json'])


    if stopAfterExecuting == True:
        Stop(model)
        

    return allKeyPoints

def Stop(model):
    model.stop()
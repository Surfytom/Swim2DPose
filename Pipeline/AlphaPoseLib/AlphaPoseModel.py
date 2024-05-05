import docker

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

def LoadConfig(args):
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
    }

def Inference(model, videoNames, stopAfterExecuting=True):

    allKeyPoints = []

    for i, videoName in enumerate(videoNames):

        print(f"Video {i} running through AlphaPose")
        model.exec_run(cmd=f'python3 /build/AlphaPose/scripts/demo_inference.py --detector yolox-x --cfg /build/AlphaPose/configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint pretrained_models/halpe26_fast_res50_256x192.pth --video "/usr/src/app/media/{videoName}.mp4" --save_video --outdir /usr/src/app/media/ --sp --vis_fast')
        print(f"Video {i} Finished")

    if stopAfterExecuting == True:
        Stop(model)

    return allKeyPoints

def Stop(model):
    model.stop()
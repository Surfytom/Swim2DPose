import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install("docker")
import docker


def InitModel(config):
    client = docker.from_env()
    # container = client.containers.run('lhlong1/openpose:latest', detach=True, stdin_open=True, tty=True, device_requests=[docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])], volumes={
    #         '/home/student/horizon-coding': {
    #             'bind': '/data',
    #             'mode': 'rw'
    #         }}, command = './build/examples/openpose/openpose.bin --video "/data/Auboeck, Start, Freestyle, 11_07_2023 10_10_20_5_Edited.mp4" --display 0 --hand --write_json "keypoints.json"  --write_video "new-openpose.avi"')
    container = client.containers.run(**config)
    # Print container ID
    print("Container ID:", container.id)
    print('container status: ', container.status)
    # Get the container logs
    logs = container.logs()
    print(logs.decode('utf-8'))
    # Wait for the container to finish executing
    exit_code = container.wait()['StatusCode']
    subprocess.run(["docker", "cp", f"{container.id}:/openpose/new-openpose.avi", "."])

    print("Container exited with code:", exit_code)

    # Remove the container
    container.stop()

def LoadConfig():
    # Define the container configuration
    return {
        'image': 'lhlong1/openpose:latest',
        'command': 'echo Starting Openpose v1.7.0 ...',
        'detach': True,
        'device_requests': [
            docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])
        ],
        'volumes': {
            '/home/student/horizon-coding': {
                'bind': '/data',
                'mode': 'rw'
            }
        },
        'stdin_open': True,  # Keep STDIN open even if not attached (-i)
        'tty': True,  # Allocate a pseudo-TTY (-t)
        'command': './build/examples/openpose/openpose.bin --video "/data/Auboeck, Start, Freestyle, 11_07_2023 10_10_20_5_Edited.mp4" --display 0 --hand --write_json "keypoints.json"  --write_video "openpose.avi"'
    }

def Inference(model, imageStack, config):
    # Runs the model on a set of images returning the keypoints the model detects
    allKeyPoints = []

    for i, images in enumerate(imageStack):
        # Loops through the
        
        allKeyPoints.append([])
        for image in images:
            results = model.predict(image, iou=config["iou"], conf=config["confidence"])

            #keyPoints = np.array(results.prediction.poses[:, :2]).astype(np.intp)
            
            allKeyPoints[i].append(results.prediction)

    return allKeyPoints
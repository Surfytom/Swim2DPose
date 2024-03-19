import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install("docker")
import docker


def InitModel(config):
    client = docker.from_env()
    client.containers.run('lhlong1/openposev1.7.0', 'echo Starting openposev1.7.0 ...')

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
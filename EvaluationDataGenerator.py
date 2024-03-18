import glob
import numpy as np
import cv2

DEBUG = True

# Number of videos to sample (No Duplicates)
N = 25

# Number of consecutive frames to sample from each video
K = 5

# Cuts off this portion of the video from being selected E.G 3 (number of frames // 3) means the first third of the video will not be selected at all
C = 3

def GetRandomFrame(numOfFrames, frameBuffer):

    randomFrame = np.random.choice(np.arange(numOfFrames // C, numOfFrames - frameBuffer))

    return randomFrame

def LoadMediaPath(path):

    frames = []
    
    videoReader = cv2.VideoCapture(path)

    while True:

        ret, frame = videoReader.read()

        if not ret:
            break

        frames.append(frame)

    return frames

def CombineFinalVideo(finalVideo):

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontSize = .7
    thickness = 2

    out = cv2.VideoWriter(f'evaluationVideo.avi', cv2.VideoWriter_fourcc(*'XVID'), 1, (finalVideo[0][1][0].shape[1], finalVideo[0][1][0].shape[0]), True)

    for videoName, frames in finalVideo:

        for frame in frames:

            textSize = cv2.getTextSize(f"{videoName}", font, fontSize, thickness)
            
            cv2.putText(frame, f"{videoName}", (0, 0+(textSize[0][1]*2)), font, fontSize, (0, 255, 0), thickness)

            out.write(frame)

    out.release()

if __name__ == "__main__":

    videoPaths = glob.glob("/mnt/d/My Drive/Msc Artificial Intelligence/Semester 2/AI R&D/AIR&DProject/Sample Videos/EditedVideos/*.mp4")

    randomVideos = np.random.choice(len(videoPaths), size=(N), replace=False)

    if DEBUG:
        print(f"Random Videos Indexes: {randomVideos.shape} | {randomVideos.tolist()}")

        print(f"Random Videos Index One: {randomVideos[1]}")

    finalVideo = []

    for i, index in enumerate(randomVideos):

        frames = LoadMediaPath(videoPaths[index])

        videoName = videoPaths[index][videoPaths[index].rfind("/")+1:]
        videoName = videoName.replace(".mp4", "")

        randomFrame = GetRandomFrame(len(frames), K)

        finalVideo.append([videoName, frames[randomFrame:randomFrame+K]])

    CombineFinalVideo(finalVideo=finalVideo)

import glob
import numpy as np
import cv2
import json

DEBUG = True

# If true the video that the frames came from will be displayed in text at the top of the frames
draw = True

# Number of videos to sample (No Duplicates)
N = 10

# Number of consecutive frames to sample from each video
K = 5

# For both C and D 1 means take into account the whole video with no portion cut out
# Cuts off this portion of the video from being selected E.G 3 (number of frames // 3) means the first third of the video will not be selected at all
C = 2

# Same as C but for the end portion of the video
D = 4

def GetRandomFrame(numOfFrames):

    startingFrame = numOfFrames if C == 1 else numOfFrames // C
    endingFrame = numOfFrames if D == 1 else numOfFrames - (numOfFrames // D)

    if endingFrame - startingFrame < K:
        return startingFrame

    randomFrame = np.random.choice(np.arange(startingFrame, endingFrame - K))

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

    videoInfo = []
    # videoInfoObject = {"videoName": "None", "frameStart": 0, "frameEnd": 0}

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontSize = .7
    thickness = 2

    out = cv2.VideoWriter(f'Evaluation/evaluationVideo.avi', cv2.VideoWriter_fourcc(*'XVID'), 1, (finalVideo[0][1][0].shape[1], finalVideo[0][1][0].shape[0]), True)

    for videoName, frames, randomFrame, totalFrames in finalVideo:

        videoInfo.append({"videoName": videoName, "totalFrames": totalFrames, "frameStart": int(randomFrame), "frameEnd": int(randomFrame + K)})

        for frame in frames:

            if draw: 

                text = f"{videoName} | Total Frames: {totalFrames} | Frames: {randomFrame} - {randomFrame + K}"

                textSize = cv2.getTextSize(text, font, fontSize, thickness)
            
                cv2.putText(frame, text, (0, 0+(textSize[0][1]*2)), font, fontSize, (0, 255, 0), thickness)

            out.write(frame)

    out.release()

    with open("Evaluation/evaluationVideoInfo.json", "w") as f:
        json.dump(videoInfo, f, indent=4)

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

        randomFrame = GetRandomFrame(len(frames))

        finalVideo.append([videoName, frames[randomFrame:randomFrame+K], randomFrame, len(frames)])

    CombineFinalVideo(finalVideo=finalVideo)

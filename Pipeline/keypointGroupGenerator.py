import json

faceArray = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90]
handArray = [91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131]
feetArray = [17, 18, 19, 20, 21, 22]
simpleFeetArray = [17, 18, 19, 20, 21, 22]
bodyArray = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
simpleHandArray = [103, 124]
bodySimpleFaceArray = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

dwPoseLinks = {
    "keypoints": simpleHandArray + bodySimpleFaceArray + feetArray,
    "links": {
        "feet": [[14, 18], [14, 19], [14, 20], [13, 15], [13, 16], [13, 17]],
        "legs": [[10, 12], [12, 14], [9, 11], [11, 13]],
        "body": [[10, 22], [9, 22], [4, 21], [3, 21], [21, 2], [21, 22]],
        "arms": [[4, 6], [6, 8], [8, 1], [3, 5], [5, 7], [7, 0]]
    }
}

# keypoints = [103,124,0,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
# 0 - 20

keypointGroups = {
    "face": faceArray,
    "body": bodyArray,
    "feet": feetArray,
    "hands": handArray,
    "simpleHands": simpleHandArray,
    "bodySimpleFace": bodySimpleFaceArray,
    "simpleFeet": simpleFeetArray,
    "DWPose": dwPoseLinks,
    "YoloNasNet": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17] 
}

file = open("keypointGroupings.json", "w")
json.dump(keypointGroups, file, indent=4)
file.close()

import YoloUltralyticsLib.Yolo as yolo
from torch import cuda
import json
import time
import argparse
import utils
import pathlib
import cv2

currentPath = pathlib.Path(__file__).parent.resolve()

print("currentPath: ", currentPath)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-fo', "--folder", help='Use this flag to specify input folder path')
    parser.add_argument('-i', "--inputpaths", nargs="+", help='Use this flag to specify input paths (can be multiple)')
    parser.add_argument('-msk', "--mask", help='if this flag is set masking based segmentation is used instead of bounding boxes', action='store_true', default=True)
    parser.add_argument('-m', "--model", help="use either DWPose | AlphaPose | OpenPose | YoloNasNet", default="AlphaPose")
    parser.add_argument('-fps', "--fps", help="sets the frames per second of the output videos. Default is 24", default=24)
    parser.add_argument('-str', "--stride", help="stride of video loader (if set > 1 only processes frame every set frame. E.g 2 means only every 2 frames are processed). Default is 1", default=1)
    parser.add_argument('-s', "--save", help="saves output of pipeline to this folder. Default is './results'", default="./results")

    parser.add_argument('-l', "--label", help='this flag enables annotation upload to labelbox (only DWPose is supported for now) | Please use -lk, -lpn or -lpk and -lont (if using -lpn) with this flag', action='store_true', default=False)
    parser.add_argument('-lk', "--labelkey", help='-label this flag enables annotation upload to labelbox (only DWPose is supported for now)')
    parser.add_argument('-lont', "--labelont", help='defines ontology key to use when uploading annotations REQUIRED WHEN USING -l, -lk and  -lpn')
    parser.add_argument('-lpn', "--labelprojname", help='defines project name. Used when wanting to create a new project REQUIRES -l, -lk and -lont to be used with it')
    parser.add_argument('-lpk', "--labelprojkey", help='defines project key REQUIRES -l and -lk to be used with it')
    parser.add_argument('-ldk', "--labeldskey", help='defines project key REQUIRES -l and -lk to be used with it')
    parser.add_argument('-ldn', "--labeldsname", help='defines project key REQUIRES -l and -lk to be used with it')

    args = parser.parse_args()

    if(not args.folder and not args.inputpaths):
        raise RuntimeError("ERROR: Please include either a folder (-fo) or a set of input paths (-i) to use as input to the pipeline")

    if (args.label):
        if (not args.labelkey):
            raise RuntimeError("ERROR: When using -l -lk, -lpn or -lpk and -lo (if using -lpn) are required")
        
        if (not args.labelprojname and not args.labelprojkey):
            raise RuntimeError("ERROR: When using -l and -lk either -lpn or -lpk is needed to create or use a project as well as -lont for defining ontology")
        if (args.labelprojname and args.labelprojkey):
            raise RuntimeError("ERROR: Both -lpn and -lpk cannot be used together please select one (key if project is exising | name for new project)")
        
        if (not args.labeldsname and not args.labeldskey):
            raise RuntimeError("ERROR: When using -l and -lk either -ldn or -ldk is needed to create or use a dataset as well as -lont for defining ontology")
        if (args.labeldsname and args.labeldskey):
            raise RuntimeError("ERROR: Both -ldn and -ldk cannot be used together please select one (key if dataset is exising | name for new dataset)")
        
        if (args.labelprojname and not args.labelont):
            raise RuntimeError("ERROR: Creating a new project with -lpk cannot be used with defining an ontology for the project with -lont")

    print(args.folder)
    print(args.inputpaths)
    print(args.label)
    print(args.mask)
    print(args.fps)
    print(args.model)
    print(args.save)
    print(args.stride)

    if args.model == "DWPose":
        import DWPoseLib.DwPose as poseModel
    if args.model == "YoloNasNet":
        import YoloNasNetLib.YoloNasNet as poseModel
    if args.model == "OpenPose":
        import OpenPoseLib.OpenPoseModel as poseModel
    if args.model == "AlphaPose":
        import AlphaPoseLib.AlphaPoseModel as poseModel

    print("Cuda Available: ", cuda.is_available())
    
    paths = []
    path = "/home/student/horizon-coding/Swim2DPose/data" # change this when make a deployment

    # Get the file names in the directory
    fileNames, fileNamesWithoutExtension = utils.GetFileNames(path)

    print("File names in the directory:")
    print(fileNames)

    inputStack = []
    imageStack = []

    for i, fileName in enumerate(fileNames):
        images, strideImages = utils.LoadMediaPath(f'{fileName}', args.stride)
        paths.append(f'{path}/{fileName}')
        inputStack.append({ 'images': images, 'name': fileNamesWithoutExtension[i] } if args.stride == 1 else { images: strideImages, 'name': fileNamesWithoutExtension[i] })
        imageStack.append({ 'images': images, 'name': fileNamesWithoutExtension[i] })

        print(f"path: {fileName}\nFrame Count: {len(images)}\n")

    print(f"Image stack length {len(imageStack[0]['images'])}")
    print(f"stride {args.stride} stack length {len(inputStack[0]['images'])}")

    print(f"Loading Config For {args.model}")
    # *** THIS IS WHAT NEEDS TO BE CHANGED TO IMPLEMENT A NEW POSE MODEL ***

    # Loads the models specific config file
    config = poseModel.LoadConfig(args)

    # This function initialises and return a model with a weight and config path
    model = poseModel.InitModel(config)

    if args.model == "DWPose" or args.model == "YoloNasNet":

        with open("keypointGroupings.json", "r") as f:
            keypointGroups = json.load(f)

        selectedPoints = keypointGroups[args.model]

        startTime = time.perf_counter()

        yoloModel = yolo.InitModel("Pipeline/YoloUltralyticsLib/Models/yolov8x-seg.pt")

        # Need to send yolo segmented images to dwpose model
        results = yolo.YOLOSegment(yoloModel, inputStack)

        segmentedImageStack, Bboxes = utils.MaskSegment(inputStack, results) if args.mask else utils.BboxSegment(inputStack, results)

        startTime2 = time.perf_counter()

        # This function runs the model and gets a result in the format
        # Array of inputs (multiple videos) -> frames (from one video) -> array of keypoints (for one frame)
        keyPoints = poseModel.Inference(model, segmentedImageStack, config)

        endTime2 = time.perf_counter()
        print("Time in seconds for pose estimation inference: ", (endTime2 - startTime2))

        # This function takes in numerous inputs and outputs the keypoint positions of selected keypoints (A potential subset of the models potential keypoints)
        # INPUTS:
        #   inputStack      : array of images         [[frames], [frames], ...]
        #   keyPointStack   : array of keypoints      [[keypointsframe1, keypointsframe2, ...], [keypointsframe1, keypointsframe2, ...], ...]
        #   bboxStack       : array of bounding boxes [[bboxframe1, bboxframe1, ...], [bboxframe1, bboxframe1, ...], ...]
        #   selectedPoints  : array of selected points
        #   stride          : int value detemining the stride of images (10 would indicate the model generated keypoints for every 10th frame)
        #   drawKeypoints   : boolean value determining wether the function should draw keypoints on the image
        #   drawBboxes      : boolean value determining wether the function should draw the bounding boxes on the image
        #   drawText        : boolean value determining wether the function should draw the text for each keypoint on the image
        selectedKeyPoints = poseModel.DrawKeypoints(imageStack, keyPoints, Bboxes, selectedPoints, args.stride, True, True, True)

        # *** THIS IS WHAT NEEDS TO BE CHANGED TO IMPLEMENT A NEW POSE MODEL ***

        if args.save:
            utils.SaveImages(imageStack, args.fps, args.model, args.save)

        if args.label:

            for i, input in enumerate(imageStack):
                if len(input) <= 1:
                    paths.remove(i)

            datasetKeyorName = args.labeldskey if args.labeldskey else args.labeldsname
            datasetExisting = True if args.labeldskey else False

            projectKeyorName = args.labelprojkey if args.labelprojkey else args.labelprojname
            projectExisting = True if args.labelprojkey else False
                    
            utils.SaveVideoAnnotationsToLabelBox(args.lk, datasetKeyorName, datasetExisting, projectKeyorName, projectExisting, paths, selectedKeyPoints)

    elif args.model == "OpenPose" or args.model == "AlphaPose":
        print(fileNamesWithoutExtension)
        # This function runs the model and gets a result in the format
        # Array of inputs (multiple videos) -> frames (from one video) -> array of keypoints (for one frame)
        keyPoints = poseModel.Inference(model, fileNamesWithoutExtension)
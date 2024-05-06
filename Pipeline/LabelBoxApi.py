import labelbox as lb
import labelbox.types as lb_types
import uuid
import ffmpeg
import os
import shutil

DEBUG = False

def InitializeClient(apiKey):
    return lb.Client(apiKey)

def InitializeDataset(client, nameOrKey, existing=False):
    
    dataset = None

    if existing:
        #dataset = client.get_dataset("clsxmattj001f0770kcfkcuqw")
        dataset = client.get_dataset(nameOrKey)
    else:
        print(nameOrKey)
        print(type(nameOrKey))
        dataset = client.create_dataset(name=nameOrKey)

    return dataset

def InitializeProject(client, projNameOrKey, ontKey, dataRowKeys, existing=False):

    project = None

    for i, path in enumerate(dataRowKeys):

            if "/" in path:
                dataRowKeys[i] = path[path.rfind("/")+1:]

    print("data rows in intialize project: ", len(dataRowKeys), " ", dataRowKeys)

    if existing:
        project = client.get_project(projNameOrKey)

        project.create_batch(
            "project-batch-" + str(uuid.uuid4()), # Each batch in a project must have a unique name
            global_keys=dataRowKeys, # A paginated collection of data row objects, a list of data rows or global keys
            priority=5 # priority between 1(Highest) - 5(lowest)
        )
    else:
        project = client.create_project(name=projNameOrKey, media_type=lb.MediaType.Video)

        # colours = ["#e6194B", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990", "#dcbeff", "#9A6324", "#fffac8", "#800000", "#aaffc3", "#808000", "#ffd8b1", "#000075", "#a9a9a9", "#ffffff", "#000000", "#008080"]
        
        # tools = [lb.Tool(tool=lb.Tool.Type.POINT, name=f"{i}", color=f"{colours[i]}") for i in range(21)]

        # ontology_builder = lb.OntologyBuilder(tools)

        # ontology = client.create_ontology(f"{nameOrKey} Ontology",
        #                           ontology_builder.asdict(),
        #                           media_type=lb.MediaType.Video)

        ontology = client.get_ontology(ontKey)
        
        project.setup_editor(ontology)

        project.create_batch(
            "project-batch-" + str(uuid.uuid4()), # Each batch in a project must have a unique name
            global_keys=dataRowKeys, # A paginated collection of data row objects, a list of data rows or global keys
            priority=5 # priority between 1(Highest) - 5(lowest)
        )

    return project

def AddToDataset(dataset, dataPaths, tempVideoFolder="./ffmpegTempOutput"):

    print(f"datapaths: {dataPaths}")

    dataRows = dataset.data_rows()

    validDataPaths = []

    for i, path in enumerate(dataPaths):
        found = False
        for dataRow in dataRows:
            if DEBUG:
                print(f"global key: {dataRow.global_key}")
            if dataRow.global_key in path:
                found = True
            
        if not found and path not in validDataPaths:
            validDataPaths.append(path)

    print("Files being uploaded to LabelBox: ", validDataPaths)

    if len(validDataPaths) >= 1:
        
        ConvertVideosToLabelBoxFormat(validDataPaths, tempVideoFolder)

        assets = []

        for i, path in enumerate(validDataPaths):

            if "/" in path:
                path = path[path.rfind("/")+1:]

            assets.append(
                {
                    "row_data": f"{tempVideoFolder}/outputVideo{i}.mp4",
                    "global_key": f"{path}",
                    "media_type": "VIDEO"
                }
            )

        task = dataset.create_data_rows(assets)
        task.wait_till_done()
        print(task.errors)

        shutil.rmtree(tempVideoFolder)

def AddAnnotations(client, projectId, dataGlobalKeys, allKeyPoints):

    label = []
    names = ["Face", "ShoulderLeft", "ShoulderRight", "ElbowLeft", "ElbowRight", "WristLeft", "WristRight", "HipLeft", "HipRight", "KneeLeft", "KneeRight", "AnkleLeft", "AnkleRight", "LeftToeInside", "LeftToeOutside", "LeftFoot", "RightToeInside", "RightToeOutside", "RightFoot", "LeftFinger", "RightFinger"]

    for videoKey, keyPointArray in zip(dataGlobalKeys, allKeyPoints):

        if "/" in videoKey:
            videoKey = videoKey[videoKey.rfind("/")+1:]
    
        pointAnnotation = []

        if DEBUG:
            print(len(keyPointArray))
            print(len(keyPointArray[0]))

        for i, keyPoints in enumerate(keyPointArray):
            for j, (keyX, keyY) in enumerate(keyPoints):
                pointAnnotation.append(
                    lb_types.VideoObjectAnnotation(
                        name = names[j],
                        keyframe=True,
                        frame = i+1,
                        value = lb_types.Point(x=keyX, y=keyY)
                    )
                )

        annotationsList = [pointAnnotation]

        # print("DEBUG for point annotation")
        # for point in pointAnnotation:
        #     print(point)

        for annotation in annotationsList:
            label.append(
                lb_types.Label(
                    data=lb_types.VideoData(global_key=videoKey),
                    annotations = annotation
                )
            )
    
    print(f"Number of annotation being uploaded: {len(pointAnnotation)}")

    if len(label) > 0:
        # Upload MAL label for this data row in project
        upload_job_mal = lb.MALPredictionImport.create_from_objects(
            client = client,
            project_id = projectId,
            name="mal_import_job-" + str(uuid.uuid4()),
            predictions=label
        )

        upload_job_mal.wait_until_done()
        print("Errors:", upload_job_mal.errors)
        print("Status of uploads: ", upload_job_mal.statuses)
        print("   ")

def ConvertVideosToLabelBoxFormat(videoPaths, outputPath="./ffmpegTempOutput"):

    if os.path.isdir(outputPath) == True:
        shutil.rmtree(outputPath)

    os.mkdir(outputPath)
    
    for i, videoPath in enumerate(videoPaths):
        print(f"Video path: {videoPath}")
        ffmpeg.input(videoPath).output(f"{outputPath}/outputVideo{i}.mp4", acodec="aac", vcodec="libx264").run()

if __name__ == "__main__":

    api_key = None

    with open("env.txt", "r") as f:
        api_key = f.read().split("=")[1]

    client = lb.Client(api_key)

    dataset = client.get_dataset("clsxmattj001f0770kcfkcuqw")

    print(dataset.created_at)

    # assets = [
    #   {
    #     "row_data": "Cohoon, Start, Freestyle, 01_08_2023 08_59_22_5.mp4",
    #     "global_key": "key",
    #     "media_type": "VIDEO",
    #   }
    # ]

    project = client.get_project("clsypyn9c01d8072g2hkj1zt4")
    #project = client.create_project(name="TestPoints", media_type=lb.MediaType.Video)

    # with open("20DistinctColours.json", "r") as f:
    #     colours = json.load(f)

    # tools = [lb.Tool(tool=lb.Tool.Type.POINT, name=f"{i}", color=f"{colours[i]}") for i in range(20)]

    # ontology_builder = lb.OntologyBuilder(tools)

    # ontology_builder = lb.OntologyBuilder(
    #     tools=[
    #         lb.Tool(tool=lb.Tool.Type.POINT, name="headPoint", color="#39e817"),
    #         lb.Tool(tool=lb.Tool.Type.POINT, name="armPoint", color="#d82827")
    #     ])

    # ontology = client.create_ontology("Video Annotation Import Demo Ontology",
    #                                   ontology_builder.asdict(),
    #                                   media_type=lb.MediaType.Video)

    # project.setup_editor(ontology)

    # batch = project.create_batch(
    #   "first-batch-video-demo1", # Each batch in a project must have a unique name
    #   global_keys=["key"], # A paginated collection of data row objects, a list of data rows or global keys
    #   priority=5 # priority between 1(Highest) - 5(lowest)
    # )

    # print("Batch: ", batch)

    point_annotation = []

    for i in range(1, 150):
        point_annotation.append(
            lb_types.VideoObjectAnnotation(
                name = "headPoint",
                keyframe=True,
                frame=i,
                value = lb_types.Point(x=1, y=1),
            )
        )

        point_annotation.append(
            lb_types.VideoObjectAnnotation(
                name = "armPoint",
                keyframe=True,
                frame=i,
                value = lb_types.Point(x=50, y=50),
            )
        )

    label = []
    annotations_list = [
            point_annotation,
        ]

    for annotation in annotations_list:
        label.append(
            lb_types.Label(
                data=lb_types.VideoData(global_key="key"),
                annotations = annotation
            )
        )
    # Upload MAL label for this data row in project
    upload_job_mal = lb.MALPredictionImport.create_from_objects(
        client = client,
        project_id = project.uid,
        name="mal_import_job-" + str(uuid.uuid4()),
        predictions=label)

    upload_job_mal.wait_until_done()
    print("Errors:", upload_job_mal.errors)
    print("Status of uploads: ", upload_job_mal.statuses)
    print("   ")

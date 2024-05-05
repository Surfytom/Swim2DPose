# The Pipeline

This file introduces all the config options you can set when using this project such as flags in the command line arguments and config files for each model.

## Flags

| <div style="width:50px;">Flag</div> | <div style="width:125px;">Arguments</div> | Description | Dependencies |
| ------------- | ------------- | ------------- | ------------- |
| <div style="width:50px;">```-m```</div> | <div style="width:125px;">ModelName</div> | Sets the pose model model to be used by the pipeline. Default is ```AlphaPose``` | None |
| <div style="width:50px;">```-fo```</div> | <div style="width:125px;">Path/to/folder</div> | Sets a path to folder containing media files to process | ```No -i``` |
| <div style="width:50px;">```-i```</div> | <div style="width:125px;">Path/to/videoa.mp4 Path/to/videob.mp4</div> | Sets a path to individual media files to process | ```No -fo``` |
| <div style="width:50px;">```-msk```</div> | <div style="width:125px;">No Args</div> | If set contour masks will be used instead of basic bounding boxes. Default is ```True``` | None |
| <div style="width:50px;">```-fps```</div> | <div style="width:125px;">FpsYouWant</div> | Sets the frames per second of the output videos. Default is ```24``` | None |
| <div style="width:50px;">```-str```</div> | <div style="width:125px;">StrideCount</div> | Sets the stride count for processing frames of videos. E.g 2 will mean only every 2nd frames gets processed. Default is ```1``` | None |
| <div style="width:50px;">```-s```</div> | <div style="width:125px;">path/to/save/folder</div> | Sets the path you want processed media to be saved. Default is ```Pipeline/results``` | None |
| <div style="width:50px;">```-l```</div> | <div style="width:125px;">No Args</div> | If set processed keypoints will be uploaded to a LabelBox project. Default is ```False``` | ```-lk, -lont, -lpn or -lpk, -ldn or -ldk``` |
| <div style="width:50px;">```-lk```</div> | <div style="width:125px;">LabelBoxKey</div> | Sets your LabelBox api key | ```-l, -lont, -lpn or -lpk, -ldn or -ldk``` |
| <div style="width:50px;">```-lont```</div> | <div style="width:125px;">OntologyKey</div> | Sets your LabelBox ontology key | ```-l, -lk, -lpn or -lpk, -ldn or -ldk``` |
| <div style="width:50px;">```-lpn```</div> | <div style="width:125px;">NewProjectName</div> | Sets a project name that will be created on LabelBox and used to upload annotations to | ```-l, -lk, -lont, not -lpk, -ldn or -ldk``` |
| <div style="width:50px;">```-lpk```</div> | <div style="width:125px;">OldProjectKey</div> | Sets an existing project key that will be used to upload annoatations to | ```-l, -lk, -lont, not -lpn, -ldn or -ldk``` |
| <div style="width:50px;">```-ldn```</div> | <div style="width:125px;">NewDatasetName</div> | Sets a dataset name that will be created on LabelBox and used to upload base videos to | ```-l, -lk, -lont, -lpn or -lpk, not -ldk``` |
| <div style="width:50px;">```-ldk```</div> | <div style="width:125px;">OldDatasetKey</div> | Sets an existing LabelBox dataset key that will be used to upload base videos to | ```-l, -lk, -lont, -lpn or -lpk, not -ldn``` |

## Configs

Configs can be set by going to the config file in the Models lib folder (DWPoseLib/config.json). These files can be edited with the configs set to be variable by the person that integrated the model. Please consult the specific model read me for more info on their specific configs.

## [LabelBox](https://docs.labelbox.com/)

The label box flags might seem daunting if you havn't used LabelBox before so this section is going to explain in breif terms how LabelBox functions.

In this repository annoatation are uploaded as pre-labels which can then be easily corrected by humans.

### [Datasets](https://docs.labelbox.com/docs/datasets-datarows)

This is where base files live ready to be used by a project for annotation. Global keys are shared between datasets and our global keys are set using the filename of a video. It is important to note you should not try to upload the same file to two different datasets.

### [Projects](https://docs.labelbox.com/docs/what-is-a-project)

Projects are where annotation occurs. Each project has one ontology which defines the format of the annotation of its videos. Projects reference videos from datasets using the global keys mentioned earlier.

### [Ontologies](https://docs.labelbox.com/docs/labelbox-ontology)

An ontology is a object that defines an annotation format. This could be for example one bounding box. This format would mean every frame would have one bounding box on it. For pose estimation purposes an ontology would probably contain a set of points labelled for each keypoint your chosen model outputs. Please create this ontology on LabelBox and reference its key to use it.
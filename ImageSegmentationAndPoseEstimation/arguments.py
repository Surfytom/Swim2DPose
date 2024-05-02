import argparse

if __name__ == "__main__":
    # Need arguments for:
    #   Pose model chosen
    #   Input Path for videos/image
    #       I suggest adding arugment for folder
    #       and another arugment for just a list of paths

    #   LabelBoxInfo

    parser = argparse.ArgumentParser()

    parser.add_argument('-folder', "--folder", help='Use this flag to specify input folder path')
    parser.add_argument('-inputpaths', "--inputpaths", nargs="+", help='Use this flag to specify input paths (can be multiple)')
    parser.add_argument('-mask', "--mask", help='if this flag is set masking based segmentation is used instead of bounding boxes', action='store_true', default=True)
    parser.add_argument('-model', "--model", help="use either DWPose | AlphaPose | OpenPose | YoloNasNet", default="AlphaPose")

    parser.add_argument('-label', "--label", help='this flag enables annotation upload to labelbox (only DWPose is supported for now) | Please use -labelkey, -labelprojname or -labelprojkey and -labelont (if using -labelprojname) with this flag', action='store_true', default=False)
    parser.add_argument('-labelkey', "--labelkey", help='-label this flag enables annotation upload to labelbox (only DWPose is supported for now)')
    parser.add_argument('-labelont', "--labelont", help='defines ontology key to use when uploading annotations REQUIRED WHEN USING -label, -labelkey and  -labelprojname')
    parser.add_argument('-labelprojname', "--labelprojname", help='defines project name. Used when wanting to create a new project REQUIRES -label, -labelkey and -labelont to be used with it')
    parser.add_argument('-labelprojkey', "--labelprojkey", help='defines project key REQUIRES -label and -labelkey to be used with it')

    args = parser.parse_args()

    if (args.label):
        if (not args.labelkey):
            raise RuntimeError("ERROR: When using -label -labelkey, -labelprojname or -labelprojkey and -labelont (if using -labelprojname) are required")
        if (not args.labelprojname and not args.labelprojkey):
            raise RuntimeError("ERROR: When using -label and -labelkey either -labelprojname or -labelprojkey is needed to create or use a project as well as -labelont for defining ontology")
        if (args.labelprojname and args.labelprojkey):
            raise RuntimeError("ERROR: Both -labelprojname and -labelprojkey cannot be used together please select one (key if project is exising | name for new project)")
        if (args.labelprojname and not args.labelont):
            raise RuntimeError("ERROR: Creating a new project with -labelprojname cannot be used with defining an ontology for the project with -labelont")

    print(args.folder)
    print(args.inputpaths)
    print(args.mask)

    print("Label args: ")
    print(args.label)
    print(args.labelkey)
    print(args.labelprojkey)
    print(args.labelprojname)
    print(args.labelont)
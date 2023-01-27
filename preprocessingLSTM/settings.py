SEED = 42

def set_track(new_track="full_track"):
    global track 
    global VID_FOLDER
    global FMRI_FOLDER
    global OUTPUT_FOLDER
    global ROIs

    track = new_track

    OUTPUT_FOLDER = f"{PROJECT_FOLDER}output/{track}/"
    FMRI_FOLDER = f"{DATA_FOLDER}participants_data_v2021/{track}/"
    VID_FOLDER = f"{DATA_FOLDER}AlgonautsVideos268_All_30fpsmax/"

    if track == "full_track":
        ROIs = ["WB"]
    else:
        ROIs = ["LOC", "FFA", "STS", "EBA", "PPA", "V1", "V2", "V3", "V4"]

# project settings
track = None
train_data_len = 1000
num_subs = 10
subs = ["sub" + str(s + 1).zfill(2) for s in range(num_subs)]

REPETITIONS = 3  # number of recordings per video
VID_LEN_SEC = 3  # video length in seconds
TARGET_FPS = 5  # target fps for loading data
MAX_SEQ_LEN = TARGET_FPS * VID_LEN_SEC  # maximum sequence length
IMG_SIZE = 224

# environment settings
PROJECT_FOLDER = ""
DATA_FOLDER = ""
OUTPUT_FOLDER = None

FMRI_FOLDER = None
VID_FOLDER = None
ROIs = None
set_track("mini_track")



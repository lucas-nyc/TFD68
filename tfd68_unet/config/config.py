
DATA_DIR = "data"
DEVICE = "CUDA:0"
KEYPOINTS = 68
NUM_CLASSES = 8
BBOX_MARGIN_PX = 5
GAUSS_SIGMA = 6.0
OUT_W = 256
OUT_H = 256
REQUIRED_CAT = 7
MASK_RADIUS = 3

PITCH_TO_EMOTION = {
    "3": 1,
    "6": 2,
    "7": 3,
    "8": 4,
    "9": 5,
    "10": 6,
    "11": 7,
}

OUT_DIR = "out"  
RUN_NAME = "tfd68_unet_run"
DATA_ROOT = "input"       
COCO_JSON = "input/tfd68-9.json"
IMAGES_ROOT = "input/tfd68/thermal" 
MASKS_ROOT = "input/masks"
OUT_JSON = "tfd68_annotations.json"
SPLIT = {
    "train": 0.6,
    "val": 0.2,
    "test": 0.2
}
RANDOM_SEED = 42
BATCH_SIZE = 8
EPOCHS =100

BASE_LR = 1e-4
PATIENCE_ES = 6
PATIENCE_LR = 3
MIN_LR = 1e-7
DROP_OUT = 0.5

EPOCHS_STAGE1 = 15
EPOCHS_STAGE2 = 20
EPOCHS_STAGE3 = 25

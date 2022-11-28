## ############################################################################
## Dataset config
## yolo_yml_file = "/shared/ds/data/compvis/maize/fusarium/_DATASET_REVIEWED_348IMG/yolo_v5_5CL/fusalab_yolov5_02.yaml"
yolo_yml_file = "/home/nikoenki/Documents/Machine-Learning-Collection/ML/_nv/yolov1/samples/samples.yml"
#yolo_yml_file = "samples/samples.yml"
### yolo_yml_file  = "/shared/ds/data/compvis/maize/fusarium/_DATASET_REVIEWED_348IMG/yolo_v5_5CL/fusalab_yolov5_02.yaml"

## ############################################################################
## Hyper parameters for training
EXP_NAME = "fusalab_01"

BATCH_SIZE = 6
LEARNING_RATE = 1e-3 #1e-4
EPOCHS = 30
DEVICE = 'cpu'
# take a sampling of #samples samples from the datasets 
sampling = None #40 #None # 50
boundaries = None #[8956, 12000]

## Models file management
CHECKPOINTS_DIR = "models"

## LOAD_EPOCH : None, True, False, epoch number, last, best. If true will load last checkpoint of files matching file name patter in model dir
LOAD_CHECKPOINT = "best"

## LOAD_EPOCH : all, last, best, true
SAVE_CHECKPOINTS = False

#log
logging_level = 'INFO'


## ############################################################################
## Model hyper parameters
#mode (str, optional): type of model; Model head will change depanding on this type which can be "object_detection" or "classification". Defaults to "object_detection"
mode = "object_detection"

# nclass (int, optional): number of class to predict Defaults to 20.
nclass = 20

# input image resize 
image_resize = 1024

# image_shape (int, optional): input image shape. in the yolo_model Defaults to (448, 448, 3).
image_shape = (448, 448, 3)

# arch_id (str, optional): name of the yolo v1 architecture (only "24" is currently supported). Defaults to "24".
arch_id = "24" 

# nbox (int, optional): number of anchor boxes per grid cell. Defaults to 2.
nbox = 2

# s_grid (int, optional): number of grid cell (one value, will be same for height and width). Defaults to 7.
s_grid = 15
# in_channels (int, optional): number of input channel (3 for RGB images). Defaults to 3.

# hid_ly (int, optional): hidden layer dimension of the head (in paper 4096). Defaults to 496.
hid_ly = 496

# head_dropout_p (int, optional): dropout proba of the dense layer. Defaults to 0.0.
head_dropout_p = 0.2

# lkrelu_slope (int, optional): hidden layer leaky rely slope. Defaults to 0.1.
lkrelu_slope = 0.1 

# ncol_coords (int, optional):  number of columns for 1 box coordinates : 5 = (objectness, x0, y0, h, w). Defaults to 5.
ncol_coords = 5

# check if current implementation matches orig from A. Persson
compare_with_orig_impl = False






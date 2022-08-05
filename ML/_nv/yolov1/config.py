## ############################################################################
## Dataset config
yolo_yml_file = "/shared/ds/data/compvis/maize/fusarium/_DATASET_REVIEWED_348IMG/yolo_v5_5CL/fusalab_yolov5_02.yaml"

## ############################################################################
## Hyper parameters for training
BATCH_SIZE = 10
LEARNING_RATE = 1e-3
EPOCHS = 3

## ############################################################################
## Model hyper parameters
#mode (str, optional): type of model; Model head will change depanding on this type which can be "object_detection" or "classification". Defaults to "object_detection"
mode = "classification"

# nclass (int, optional): number of class to predict Defaults to 20.
nclass = 20

# image_shape (int, optional): input image shape. Defaults to (448, 448, 3).
image_shape = (448, 448, 3)

# arch_id (str, optional): name of the yolo v1 architecture (only "24" is currently supported). Defaults to "24".
arch_id = "24" 

# nbox (int, optional): number of anchor boxes per grid cell. Defaults to 2.
nbox = 2

# s_grid (int, optional): number of grid cell (one value, will be same for height and width). Defaults to 7.
s_grid = 7
# in_channels (int, optional): number of input channel (3 for RGB images). Defaults to 3.

# hid_ly (int, optional): hidden layer dimension of the head (in paper 4096). Defaults to 496.
hid_ly = 496

# head_dropout_p (int, optional): dropout proba of the dense layer. Defaults to 0.0.
head_dropout_p = 0.0

# lkrelu_slope (int, optional): hidden layer leaky rely slope. Defaults to 0.1.
lkrelu_slope = 0.1 

# ncol_coords (int, optional):  number of columns for 1 box coordinates : 5 = (objectness, x0, y0, h, w). Defaults to 5.
ncol_coords = 5








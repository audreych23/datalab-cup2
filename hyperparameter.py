import numpy as np

MAX_BOXES = 100
IOU_THRESHOLD = 0.5
SCORE_THRESHOLD = 0.5


DATA_PATH = './data/pascal_voc_training_data.txt'
IMAGE_PATH = './data/VOCdevkit_train/VOC2007/JPEGImages/'
TEST_PATH = './data/pascal_voc_testing_data.txt'
IMAGE_TEST_PATH = './data/VOCdevkit_train/VOC2007/JPEGImages/'

OUTPUT_PATH = './output/'
#VAL_DATASET = 
CKPT_DIR = './ckpts/YOLO'
CKPT_NAME = '/yolo-10'
CLASSES_DIR = "./data/voc_classes"
IMAGE_SIZE = 416
EPOCHS = 15
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
NUM_CLASSES = 20
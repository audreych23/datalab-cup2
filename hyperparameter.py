MAX_BOXES = 100
IOU_THRESHOLD = 0.5
SCORE_THRESHOLD = 0.5

DATA_PATH = './data/pascal_voc_training_data.txt'
IMAGE_PATH = './data/VOCdevkit_train/VOC2007/JPEGImages/'
TEST_PATH = './data/pascal_voc_testing_data.txt'
IMAGE_TEST_PATH = './data/VOCdevkit_test/VOC2007/JPEGImages/'
TFRECORD_DATA_PATH = './data/voc2007_train.tfrecord'
CLASSES_NAME_PATH = './data/voc2007.names'

OUTPUT_PATH = './output/'
#VAL_DATASET = 
CKPT_DIR = './ckpts/YOLO'
CKPT_NAME = '/yolo-207'
IMAGE_SIZE = 608
EPOCHS = 370
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_CLASSES = 20
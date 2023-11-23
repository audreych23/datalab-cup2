import time
import cv2
import numpy as np
import tensorflow as tf
from yolo_v3.models import (
    YoloV3
)
import os
import yolo_v3.dataset as dataset
from yolo_v3.utils import draw_outputs, load_only_pretrained_darknet_imagenet_weights
import hyperparameter as param
import data.voc_classes

checkpoint_name = '/yolo-15'

@tf.function
def prediction_step(img, model):
    return model(img)

def main():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            # Select GPU number 1
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    yolo_v3 = YoloV3(size=param.IMAGE_SIZE, classes=param.NUM_CLASSES, training=False)
    # load ckpts
    if len(os.listdir(param.CKPT_DIR)) != 0:
        ckpt = tf.train.Checkpoint(epoch=tf.Variable(0), net=yolo_v3)
        ckpt.restore(param.CKPT_DIR + param.CKPT_NAME).expect_partial()
    
        print('Checkpoint restored')

    class_names = data.voc_classes.classes_name
    
    test_dataset = dataset.create_dataset_pipeline(train=False)

    output_file = open(param.OUTPUT_PATH+'/test_prediction.txt', 'w')

    for img_labels, widths, heights, imgs in test_dataset:
        # predict one by one so it can be written to the csv file
        # for i in range(tf.shape(img_labels)[0]):
        for i, img in enumerate(imgs):
            # original img width 
            # Transform to param.size and param.size
            img = tf.expand_dims(img, 0)
            # nums is nums of obj detected
            boxes, scores, classes, nums = prediction_step(img, yolo_v3)
            print(boxes)
            print(scores)
            print(classes)
            print(nums)

            # write per object
            wh = np.flip(img.shape[0:2])
            if nums[0] != 0:
                output_file.write(img_labels[i].numpy()[0].decode('ascii'))

            for num in range(nums[0]):
                # for image size picture
                print(wh)
                print(boxes[num][0:2])
                x1y1 = tuple((np.array(boxes[0][num][0:2]) * wh).astype(np.int32))
                x2y2 = tuple((np.array(boxes[0][num][2:4]) * wh).astype(np.int32))
                # img = cv2.rectangle(img, x1y1, x2y2, (255, 0, 0), 2)
                xmin, ymin = x1y1
                xmax, ymax = x2y2
                xmin, ymin, xmax, ymax = xmin*(widths[i]/param.IMAGE_SIZE), ymin*(heights[i]/param.IMAGE_SIZE), \
                        xmax*(widths[i]/param.IMAGE_SIZE), ymax*(heights[i]/param.IMAGE_SIZE)
                #img filename, xmin, ymin, xmax, ymax, class, confidence
                output_file.write(" %d %d %d %d %d %f" %(xmin, ymin, xmax, ymax, classes[0][num], scores[0][num]))

            output_file.write("\n")

    output_file.close()

    evaluate()
    print("Done.")

    # img_name = '000001.jpg'
    # visualize(img_name, yolo_v3)

def evaluate():
    import evaluate.evaluate as evaluate
    evaluate(os.path.join(param.OUTPUT_PATH, '/test_prediction.txt'), os.path.join(param.OUTPUT_PATH, '/output_file.csv'))

def visualize(img_name, model):
    img_raw = tf.image.decode_image(
            open(param.IMAGE_TEST_PATH + img_name, 'rb').read(), channels=3)

    img = tf.expand_dims(img_raw, 0)
    img = tf.image.resize(img, (param.IMAGE_SIZE, param.IMAGE_SIZE))
    img = img / 255
    boxes, scores, classes, nums = prediction_step(img, model)
    for i in range(nums[0]):
         print('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                            np.array(scores[0][i]),
                                            np.array(boxes[0][i])))
         
    class_names = data.voc_classes.classes_name
    img = cv2.cvtColor(img.numpy(), cv2.COLOR_RGB2BGR)
    img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
    cv2.imwrite(param.OUTPUT_PATH + img_name + '.jpg', img)


if __name__ == '__main__':
    main()
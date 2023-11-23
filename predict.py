import time
import cv2
import numpy as np
import tensorflow as tf
from yolo_v3.models import (
    YoloV3
)
import yolo_v3.dataset as dataset
from yolo_v3.utils import draw_outputs, load_only_pretrained_darknet_imagenet_weights
import hyperparameter as param
import data.voc_classes

checkpoint_name = '/yolo-10'

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
    load_only_pretrained_darknet_imagenet_weights(yolo_v3, "./pre_weight/darknet53.weights")

    ckpt = tf.train.Checkpoint(epoch=tf.Variable(0), net=yolo_v3)
    ckpt.restore(param.CKPT_DIR + param.CKPT_NAME)
    
    print('Checkpoint restored')

    class_names = data.voc_classes.classes_name
    
    test_dataset = dataset.create_dataset_pipeline(train=False)

    output_file = open(param.OUTPUT_PATH + '/test_prediction.txt', 'w')

    for batch_num, (img_name, img) in test_dataset:
        # predict one by one so it can be written to the csv file
        for i in range(batch_num):
            boxes, scores, classes, nums = prediction_step(img[i], yolo_v3)
            print(f"======{img_name[i]}=====")
            print(class_names[classes])
            print("boxes: ", boxes)
            print("nums: ", nums)
            print("scores", scores)
            
            # output_file.write(img_name[i:i+1].numpy()[0].decode('ascii')+" %d %d %d %d %d %f\n" %(classes[0], scores))
            #img filename, xmin, ymin, xmax, ymax, class, confidence
            # output_file.write(img_name[i:i+1].numpy()[0].decode('ascii')+" %d %d %d %d %d %f\n" %(xmin, ymin, xmax, ymax, class_num, conf))

    output_file.close()
    print("Done.")
    # print('detections:')
    # img_name = 
    # for i in range(nums[0]):
    #     print('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
    #                                        np.array(scores[0][i]),
    #                                        np.array(boxes[0][i])))

    # img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
    # img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
    # cv2.imwrite(FLAGS.output, img)
    # logging.info('output saved to: {}'.format(FLAGS.output))
    img_name = '000001.jpg'
    visualize(img_name, yolo_v3)

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
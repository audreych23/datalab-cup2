import yolo_v3.models as models
import yolo_v3.utils as utils
import yolo_v3.dataset as dgen
import numpy as np
import tensorflow as tf
import hyperparameter as param

if __name__ == "__main__":
    # yolo = models.YoloV3(classes=FLAGS.num_classes)
    yolo = models.YoloV3(classes=20)
    utils.load_only_pretrained_darknet_imagenet_weights(yolo, "./pre_weight/darknet53.weights")

    img = np.random.random((1, 320, 320, 3)).astype(np.float32)
    output = yolo(img)
    print('pass')
    train_dataset = dgen.DatasetGenerator(train=True).generate()
    train_dataset.take(1)
    print(train_dataset.take(1))

    test_dataset = dgen.DatasetGenerator(train=False).generate()
    test_dataset.take(1)
    print(test_dataset.take(1))

    

    # class_text = tf.sparse.to_dense(
    #     x['image/object/class/text'], default_value='')

    # When training
    image_names = []
    record_list = []
    object_num_list = []

    # filling the record_list
    input_file = open(param.DATA_PATH, 'r')

    for line in input_file:
        line = line.strip()
        ss = line.split(' ')
        image_names.append(ss[0])

        record_list.append([float(num) for num in ss[1:]])
        # len // 5 because there are 5 data
        object_num_list.append(min(len(record_list[-1])//5, param.MAX_BOXES))
        print(record_list[-1])
        # self.object_num_list.append(min(len(self.record_list[-1])//5, MAX_BOXES))
        if len(record_list[-1]) < param.MAX_BOXES*5:
        #     # if there are objects less than MAX_OBJECTS_PER_IMAGE, pad the list
            record_list[-1] = record_list[-1] +\
            [0., 0., 0., 0., 0.] *\
            (param.MAX_BOXES - len(record_list[-1])//5)
        # TODO : Do this but I'm pretty sure everythign will be lesser than 100
        elif len(record_list[-1]) > param.MAX_BOXES*5:
        # if there are objects more than MAX_OBJECTS_PER_IMAGE, crop the list
            record_list[-1] = record_list[-1][:param.MAX_BOXES*5]
            print(record_list[-1])

    image_name="000005.jpg"
    image_file = tf.io.read_file(param.IMAGE_PATH + image_name) 
    image = tf.io.decode_jpeg(image_file, channels=3)

    x_train = tf.image.resize(image, (param.IMAGE_SIZE, param.IMAGE_SIZE))
    
    raw_labels = tf.cast(tf.reshape(record_list, [-1, 5]), tf.float32)

    xmin = raw_labels[:, 0]
    ymin = raw_labels[:, 1]
    xmax = raw_labels[:, 2]
    ymax = raw_labels[:, 3]
    labels = raw_labels[:, 4]
    print(xmin, ymin, xmax, ymax, labels)
    # More than one object can be detected in training data?
    # labels = tf.cast(class_table.lookup(class_text), tf.float32)

    y_train = tf.stack([xmin, ymin, xmax, ymax, labels], axis=1)
    print(y_train)

    
    #labels = tf.stack([xcenter, ycenter, box_w, box_h, class_num], axis=1)

    # paddings = [[0, MAX_BOXES - tf.shape(y_train)[0]], [0, 0]]
    # y_train = tf.pad(y_train, paddings)
    # y_train is padded already

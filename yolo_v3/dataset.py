import tensorflow as tf
import numpy as np
import hyperparameter as param
from yolo_v3.models import (
    yolo_anchors, yolo_anchor_masks
)

@tf.function
def transform_targets_for_output(y_true, grid_size, anchor_idxs):
    # y_true: (N, boxes, (x1, y1, x2, y2, class, best_anchor))
    N = tf.shape(y_true)[0]

    # y_true_out: (N, grid, grid, anchors, [x1, y1, x2, y2, obj, class])
    y_true_out = tf.zeros(
        (N, grid_size, grid_size, tf.shape(anchor_idxs)[0], 6))

    anchor_idxs = tf.cast(anchor_idxs, tf.int32)

    indexes = tf.TensorArray(tf.int32, 1, dynamic_size=True)
    updates = tf.TensorArray(tf.float32, 1, dynamic_size=True)
    idx = 0
    for i in tf.range(N):
        for j in tf.range(tf.shape(y_true)[1]):
            if tf.equal(y_true[i][j][2], 0):
                continue
            anchor_eq = tf.equal(
                anchor_idxs, tf.cast(y_true[i][j][5], tf.int32))

            if tf.reduce_any(anchor_eq):
                box = y_true[i][j][0:4]
                box_xy = (y_true[i][j][0:2] + y_true[i][j][2:4]) / 2

                anchor_idx = tf.cast(tf.where(anchor_eq), tf.int32)
                grid_xy = tf.cast(box_xy // (1/grid_size), tf.int32)

                # grid[y][x][anchor] = (tx, ty, bw, bh, obj, class)
                indexes = indexes.write(
                    idx, [i, grid_xy[1], grid_xy[0], anchor_idx[0][0]])
                updates = updates.write(
                    idx, [box[0], box[1], box[2], box[3], 1, y_true[i][j][4]])
                idx += 1

    # tf.print(indexes.stack())
    # tf.print(updates.stack())

    return tf.tensor_scatter_nd_update(
        y_true_out, indexes.stack(), updates.stack())

class DatasetGenerator:
    """
    Load pascalVOC 2007 dataset and creates an input pipeline.
    - Reshapes images into 448 x 448
    - converts [0 1] to [-1 1]
    - shuffles the input
    - builds batches
    """

    def __init__(self, train=True):
        if train:
            # When training
            self.train = train
            self.image_names = []
            self.record_list = []
            self.object_num_list = []

            # filling the record_list
            input_file = open(param.DATA_PATH, 'r')
            print(input_file)

            for line in input_file:
                line = line.strip()
                ss = line.split(' ')
                self.image_names.append(ss[0])
                print(ss[0])

                self.record_list.append([float(num) for num in ss[1:]])
                # len // 5 because there are 5 data
                self.object_num_list.append(min(len(self.record_list[-1])//5, param.MAX_BOXES))

                # self.object_num_list.append(min(len(self.record_list[-1])//5, MAX_BOXES))
                if len(self.record_list[-1]) < param.MAX_BOXES*5:
                #     # if there are objects less than MAX_OBJECTS_PER_IMAGE, pad the list
                    self.record_list[-1] = self.record_list[-1] +\
                    [0., 0., 0., 0., 0.]*\
                    (param.MAX_BOXES - len(self.record_list[-1])//5)
                # TODO : Do this but I'm pretty sure everythign will be lesser than 100
                elif len(self.record_list[-1]) > param.MAX_BOXES*5:
                # if there are objects more than MAX_OBJECTS_PER_IMAGE, crop the list
                    self.record_list[-1] = self.record_list[-1][:param.MAX_BOXES*5]

        else:
            # When testing aka predicting
            self.image_names = []
            test_img_files = open(param.TEST_PATH, 'r')
            print(input_file)

            for line in test_img_files:
                line = line.strip()
                ss = line.split(' ')
                self.image_names.append(ss[0])

    

    def _new_data_preprocess(self, image_name, raw_labels, object_num):
        image_file = tf.io.read_file(param.IMAGE_PATH + image_name) 
        image = tf.io.decode_jpeg(image_file, channels=3)
    
        x_train = tf.image.resize(image, (param.IMAGE_SIZE, param.IMAGE_SIZE))

        # class_text = tf.sparse.to_dense(
        #     x['image/object/class/text'], default_value='')
        
        raw_labels = tf.cast(tf.reshape(raw_labels, [-1, 5]), tf.float32)

        xmin = raw_labels[:, 0]
        ymin = raw_labels[:, 1]
        xmax = raw_labels[:, 2]
        ymax = raw_labels[:, 3]
        labels = raw_labels[:, 4]
        # More than one object can be detected in training data?
        # labels = tf.cast(class_table.lookup(class_text), tf.float32)

        y_train = tf.stack([xmin, ymin, xmax, ymax, labels], axis=1)
        
        #labels = tf.stack([xcenter, ycenter, box_w, box_h, class_num], axis=1)

        # paddings = [[0, MAX_BOXES - tf.shape(y_train)[0]], [0, 0]]
        # y_train = tf.pad(y_train, paddings)
        # y_train is padded already
        return x_train, y_train


    def _transform_images(self, x_train, size):
        x_train = tf.image.resize(x_train, (size, size))
        x_train = x_train / 255
        return x_train
    
    def _transform_targets(self, y_train, anchors, anchor_masks, size):
        y_outs = []
        grid_size = size // 32

        # calculate anchor index for true boxes
        anchors = tf.cast(anchors, tf.float32)
        anchor_area = anchors[..., 0] * anchors[..., 1]
        box_wh = y_train[..., 2:4] - y_train[..., 0:2]
        box_wh = tf.tile(tf.expand_dims(box_wh, -2),
                        (1, 1, tf.shape(anchors)[0], 1))
        box_area = box_wh[..., 0] * box_wh[..., 1]
        intersection = tf.minimum(box_wh[..., 0], anchors[..., 0]) * \
            tf.minimum(box_wh[..., 1], anchors[..., 1])
        iou = intersection / (box_area + anchor_area - intersection)
        anchor_idx = tf.cast(tf.argmax(iou, axis=-1), tf.float32)
        anchor_idx = tf.expand_dims(anchor_idx, axis=-1)

        y_train = tf.concat([y_train, anchor_idx], axis=-1)

        for anchor_idxs in anchor_masks:
            y_outs.append(transform_targets_for_output(
                y_train, grid_size, anchor_idxs))
            grid_size *= 2

        return tuple(y_outs)


    def _test_data_preprocess(self, image_name):
        image_file = tf.io.read_file(param.IMAGE_TEST_PATH + image_name) 
        image = tf.io.decode_jpeg(image_file, channels=3)
        img = tf.expand_dims(image, 0)
        img = self._transform_images(img, param.IMAGE_SIZE)
        return image_name, img
    


# TODO: does bounding box also get resized???
    def generate(self):
        # dataset = tf.data.Dataset.from_tensor_slices((self.image_names, 
        #                                               np.array(self.record_list), 
        #                                               np.array(self.object_num_list)))
        # dataset = dataset.shuffle(buffer_size=512)
        # dataset = dataset.map(self._data_preprocess, 
        #                       num_parallel_calls = tf.data.experimental.AUTOTUNE)
        # dataset = dataset.batch(BATCH_SIZE)
        # dataset = dataset.prefetch(buffer_size=200)

        if self.train == True:
            train_dataset = tf.data.Dataset.from_tensor_slices((self.image_names, 
                                                        np.array(self.record_list), 
                                                        np.array(self.object_num_list)))
            train_dataset = train_dataset.shuffle(buffer_size=512)
            
            train_dataset = train_dataset.map(self._new_data_preprocess, 
                                num_parallel_calls = tf.data.experimental.AUTOTUNE)
            train_dataset = train_dataset.batch(param.BATCH_SIZE)
            train_dataset = train_dataset.map(lambda x, y: (
                self._transform_images(x, param.IMAGE_SIZE),
                self._transform_targets(y, yolo_anchors, yolo_anchor_masks, param.IMAGE_SIZE)))
            train_dataset = train_dataset.prefetch(
                buffer_size=tf.data.experimental.AUTOTUNE)
            
            return train_dataset
        else:
            test_dataset = tf.data.Dataset.from_tensor_slices((self.image_names))
            test_dataset = test_dataset.map(self._test_data_preprocess, 
                                num_parallel_calls = tf.data.experimental.AUTOTUNE)
            test_dataset = test_dataset.batch(param.BATCH_SIZE)
            test_dataset = test_dataset.prefetch(
                buffer_size=tf.data.experimental.AUTOTUNE)
            # test_dataset = test_dataset.shuffle(buffer_size=512)
    
            return test_dataset
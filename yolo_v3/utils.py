from absl import logging
import numpy as np
import tensorflow as tf
import cv2

YOLOV3_LAYER_LIST = [
    'yolo_darknet',
    'yolo_conv_0',
    'yolo_output_0',
    'yolo_conv_1',
    'yolo_output_1',
    'yolo_conv_2',
    'yolo_output_2',
]

def load_darknet_weights(model, weights_file, tiny=False):
    wf = open(weights_file, 'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

    layers = YOLOV3_LAYER_LIST

    for layer_name in layers:
        sub_model = model.get_layer(layer_name)
        for i, layer in enumerate(sub_model.layers):
            if not layer.name.startswith('conv2d'):
                continue
            batch_norm = None
            if i + 1 < len(sub_model.layers) and \
                    sub_model.layers[i + 1].name.startswith('batch_norm'):
                batch_norm = sub_model.layers[i + 1]

            logging.info("{}/{} {}".format(
                sub_model.name, layer.name, 'bn' if batch_norm else 'bias'))

            filters = layer.filters
            size = layer.kernel_size[0]
            in_dim = layer.get_input_shape_at(0)[-1]

            if batch_norm is None:
                conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)
            else:
                # darknet [beta, gamma, mean, variance]
                bn_weights = np.fromfile(
                    wf, dtype=np.float32, count=4 * filters)
                # tf [gamma, beta, mean, variance]
                bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]

            # darknet shape (out_dim, in_dim, height, width)
            conv_shape = (filters, in_dim, size, size)
            conv_weights = np.fromfile(
                wf, dtype=np.float32, count=np.product(conv_shape))
            # tf shape (height, width, in_dim, out_dim)
            conv_weights = conv_weights.reshape(
                conv_shape).transpose([2, 3, 1, 0])

            if batch_norm is None:
                layer.set_weights([conv_weights, conv_bias])
            else:
                layer.set_weights([conv_weights])
                batch_norm.set_weights(bn_weights)

    assert len(wf.read()) == 0, 'failed to read all data'
    wf.close()


# I have compared that the layer of darknet in yolov3 is indeed the same as darknet53 except the last layer in darknet53 since it is for imagenet classification
def load_only_pretrained_darknet_imagenet_weights(model, weights_file, tiny=False):
    wf = open(weights_file, 'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

    sub_model = model.get_layer('yolo_darknet')
    # only copies weight that is the same with darknet53 trained in imagnenet https://pjreddie.com/darknet/imagenet/ (weights downloaded here)
    for i, layer in enumerate(sub_model.layers):
        if not layer.name.startswith('conv2d'):
            continue
        batch_norm = None
        if i + 1 < len(sub_model.layers) and \
                sub_model.layers[i + 1].name.startswith('batch_norm'):
            batch_norm = sub_model.layers[i + 1]

        logging.info("{}/{} {}".format(
            sub_model.name, layer.name, 'bn' if batch_norm else 'bias'))

        filters = layer.filters
        size = layer.kernel_size[0]
        in_dim = layer.get_input_shape_at(0)[-1]

        if batch_norm is None:
            conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)
        else:
            # darknet [beta, gamma, mean, variance]
            bn_weights = np.fromfile(
                wf, dtype=np.float32, count=4 * filters)
            # tf [gamma, beta, mean, variance]
            bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]

        # darknet shape (out_dim, in_dim, height, width)
        conv_shape = (filters, in_dim, size, size)
        conv_weights = np.fromfile(
            wf, dtype=np.float32, count=np.product(conv_shape))
        # tf shape (height, width, in_dim, out_dim)
        conv_weights = conv_weights.reshape(
            conv_shape).transpose([2, 3, 1, 0])

        if batch_norm is None:
            layer.set_weights([conv_weights, conv_bias])
        else:
            layer.set_weights([conv_weights])
            batch_norm.set_weights(bn_weights)

    assert len(wf.read()) == 0, 'failed to read all data'
    wf.close()

import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard
)
from yolo_v3.models import (
    YoloV3, YoloLoss,
    yolo_anchors, yolo_anchor_masks
)
import yolo_v3.utils  as utils
import yolo_v3.dataset as dataset
import hyperparameter as param

import datetime

def split_val_train_dataset():
    pass

def setup_model():
    model = YoloV3(param.IMAGE_SIZE, training=True, classes=param.NUM_CLASSES)
    anchors = yolo_anchors
    anchor_masks = yolo_anchor_masks

    utils.load_only_pretrained_darknet_imagenet_weights(model, "./pre_weight/darknet53.weights")
    darknet = model.get_layer('yolo_darknet')
    utils.freeze_all(darknet)

    optimizer = tf.keras.optimizers.Adam(lr=param.LEARNING_RATE)
    loss = [YoloLoss(anchors[mask], classes=param.NUM_CLASSES)
            for mask in anchor_masks]

    model.compile(optimizer=optimizer, loss=loss)

    return model, optimizer, loss, anchors, anchor_masks


@tf.function
def train_step(images, model, labels, loss, optimizer):
    with tf.GradientTape() as tape:
        outputs = model(images, training=True)
        regularization_loss = tf.reduce_sum(model.losses)
        pred_loss = []
        for output, label, loss_fn in zip(outputs, labels, loss):
            pred_loss.append(loss_fn(label, output))
        total_loss = tf.reduce_sum(pred_loss) + regularization_loss

    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(
        zip(grads, model.trainable_variables))
    
    return total_loss, pred_loss


def main():
    # Setup GPU
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

    model, optimizer, loss, anchors, anchor_masks = setup_model()
    model.summary()
    train_dataset = dataset.create_dataset_pipeline(train=True)

    avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
    avg_val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)

    ckpt = tf.train.Checkpoint(epoch=tf.Variable(0), net=model)

    manager = tf.train.CheckpointManager(ckpt, param.CKPT_DIR, max_to_keep=3,
                                     checkpoint_name='yolo')

    for epoch in range(1, param.EPOCHS + 1):
        avg_loss.reset_states()
        ckpt.epoch.assign_add(1)
        for batch, (images, labels) in enumerate(train_dataset):
            # with tf.GradientTape() as tape:
            #     outputs = model(images, training=True)
            #     regularization_loss = tf.reduce_sum(model.losses)
            #     pred_loss = []
            #     for output, label, loss_fn in zip(outputs, labels, loss):
            #         pred_loss.append(loss_fn(label, output))
            #     total_loss = tf.reduce_sum(pred_loss) + regularization_loss

            # grads = tape.gradient(total_loss, model.trainable_variables)
            # optimizer.apply_gradients(
            #     zip(grads, model.trainable_variables))
            total_loss, pred_loss = train_step(images, model, labels, loss, optimizer)

            print("{}_train_{}, {}, {}".format(
                epoch, batch, total_loss.numpy(),
                list(map(lambda x: np.sum(x.numpy()), pred_loss))))
            
            avg_loss.update_state(total_loss)

        print("{}, Epoch {}: loss {:.2f}".format(datetime.now(), epoch, avg_loss.result().numpy()))

        # avg_loss.reset_states()
        # avg_val_loss.reset_states()
        save_path = manager.save()
        print("Saved checkpoint for epoch {}: {}".format(int(ckpt.epoch), save_path))  

if __name__ == '__main__':
    main()
import yolo_v3.models as models
import yolo_v3.utils as utils

if __name__ == "__main__":
    # yolo = models.YoloV3(classes=FLAGS.num_classes)
    yolo = models.YoloV3(classes=20)
    utils.load_only_pretrained_darknet_imagenet_weights(yolo, "./pre_weight/darknet53.weights")


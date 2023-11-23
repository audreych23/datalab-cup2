import yolo_v3.models as models
import yolo_v3.utils as utils
import yolo_v3.dataset as dgen

if __name__ == "__main__":
    # yolo = models.YoloV3(classes=FLAGS.num_classes)
    # yolo = models.YoloV3(classes=20)
    # utils.load_only_pretrained_darknet_imagenet_weights(yolo, "./pre_weight/darknet53.weights")

    train_dataset = dgen.DatasetGenerator(train=True).generate()
    train_dataset.take(1)
    print(train_dataset.take(1))

    test_dataset = dgen.DatasetGenerator(train=False).generate()
    test_dataset.take(1)
    print(test_dataset.take(1))

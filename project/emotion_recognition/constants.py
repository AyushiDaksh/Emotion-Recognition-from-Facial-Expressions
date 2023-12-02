from torchvision.models import (
    resnet18,
    resnet34,
    vgg11_bn,
    vgg13_bn,
    vgg16_bn,
    vgg19_bn,
)

CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
CLASS_IDX_MAP = {cls_name: cls_idx for cls_idx, cls_name in enumerate(CLASSES)}

IMG_SIZE = (48, 48)

DEFAULT_DS_ROOT = "./project/emotion_recognition/fer2013"

MODEL_NAME_MAP = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "vgg11_bn": vgg11_bn,
    "vgg13_bn": vgg13_bn,
    "vgg16_bn": vgg16_bn,
    "vgg19_bn": vgg19_bn,
}

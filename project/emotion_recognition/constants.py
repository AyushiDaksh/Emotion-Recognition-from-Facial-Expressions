from torchvision.models import (
    resnet18,
    resnet34,
    vgg11_bn,
    vgg13_bn,
    vgg16_bn,
    vgg19_bn,

    convnext_tiny,
    efficientnet_b7,
    inception_v3,
    efficientnet_v2_s,
    mobilenet_v3_small,
    resnext50_32x4d,
    shufflenet_v2_x0_5,
    squeezenet1_1,
    wide_resnet50_2,
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

    "convnext_tiny": convnext_tiny,
    "efficientnet_b7": efficientnet_b7,
    "efficientnet_v2_s":efficientnet_v2_s,
    "inception_v3":inception_v3,
    "mobilenet_v3_small":mobilenet_v3_small,
    "resnext50_32x4d": resnext50_32x4d,
    "shufflenet_v2_x0_5": shufflenet_v2_x0_5,
    "squeezenet1_1": squeezenet1_1,
    "wide_resnet50_2": wide_resnet50_2,
}
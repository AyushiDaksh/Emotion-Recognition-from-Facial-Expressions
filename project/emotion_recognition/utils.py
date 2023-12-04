import torch
from torch import nn
from torchvision.ops import sigmoid_focal_loss
import random
import numpy as np
from logging import warn

from project.emotion_recognition.constants import CLASSES, MODEL_NAME_MAP


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_model(model_name):
    model = MODEL_NAME_MAP[model_name](num_classes=len(CLASSES))
    if "resnet" in model_name:
        # Change the model to accept single channel images
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    elif "vgg" in model_name:
        # Change the model to accept single channel images
        model.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)

    elif "convnext" in model_name:
        model.features[0][0] = torch.nn.Conv2d(1, 96, kernel_size=(4, 4), stride=(4, 4))

    elif "efficientnet_b7" in model_name:
        model.features[0][0] = torch.nn.Conv2d(
            1, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )

    elif "efficientnet_v2" in model_name:
        model.features[0][0] = torch.nn.Conv2d(
            1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )

    elif "mobilenet" in model_name:
        model.features[0][0] = torch.nn.Conv2d(
            1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )

    elif "resnext50" in model_name:
        model.conv1 = torch.nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )

    elif "shufflenet" in model_name:
        model.conv1[0] = torch.nn.Conv2d(
            1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )

    elif "squeezenet" in model_name:
        model.features[0] = torch.nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2))

    elif "wide_resnet" in model_name:
        model.conv1 = torch.nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
    else:
        raise ValueError("Unsupported model name")

    return model


def initialize_weights(init_type, layer):
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        if init_type == "uniform":
            nn.init.uniform_(layer.weight)
        elif init_type == "normal":
            nn.init.normal_(layer.weight)
        elif init_type == "constant":
            nn.init.constant_(layer.weight, 0.5)  # You can change the constant value
        elif init_type == "ones":
            nn.init.ones_(layer.weight)
        elif init_type == "zeros":
            nn.init.zeros_(layer.weight)
        elif init_type == "xavier_uniform":
            nn.init.xavier_uniform_(layer.weight)
        elif init_type == "xavier_normal":
            nn.init.xavier_normal_(layer.weight)
        elif init_type == "kaiming_uniform":
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
        elif init_type == "kaiming_normal":
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
        elif init_type == "orthogonal":
            nn.init.orthogonal_(layer.weight)

        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


class EnsembleModel(torch.nn.Module):
    def __init__(self, models):
        super(EnsembleModel, self).__init__()
        self.models = models
        for model in self.models:
            model.eval()

    def forward(self, x):
        # Average the outputs of the models
        outputs = [model(x) for model in self.models]
        avg_output = torch.mean(torch.stack(outputs), dim=0)
        return avg_output


def focal_loss(
    input, target, num_classes=len(CLASSES), alpha=0.25, gamma=2, reduction="mean"
):
    target_one_hot = nn.functional.one_hot(target, num_classes=num_classes).float()
    return sigmoid_focal_loss(
        input, target_one_hot, alpha=alpha, gamma=gamma, reduction=reduction
    )

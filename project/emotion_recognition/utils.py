import torch
from torch import nn
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

    elif model_name == "convnext_tiny":
        model.features[0][0] = nn.Conv2d(1, 96, kernel_size=(4, 4), stride=(4, 4))

    elif model_name == "efficientnet_b7":
        model.features[0][0] = nn.Conv2d(
            1, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )

    elif model_name == "efficientnet_v2_s":
        model.features[0][0] = nn.Conv2d(
            1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )

    elif model_name == "mobilenet_v3_small":
        model.features[0][0] = nn.Conv2d(
            1, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )

    elif model_name == "resnext50_32x4d":
        model.conv1 = nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )

    elif model_name == "shufflenet_v2_x0_5":
        model.conv1[0] = nn.Conv2d(
            1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False
        )

    elif model_name == "squeezenet1_1":
        model.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2))

    elif model_name == "wide_resnet50_2":
        model.conv1 = nn.Conv2d(
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


class EnsembleModel:
    def __init__(self, model_names):
        self.models = [get_model(name) for name in model_names]

    def forward(self, x):
        # Get predictions from all models
        preds = [model(x) for model in self.models]
        # Combine predictions. Here we are simply averaging them
        ensemble_pred = torch.mean(torch.stack(preds), dim=0)
        return ensemble_pred

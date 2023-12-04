import torch
import torch.nn as nn 
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
        model.conv1 = torch.nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
    elif "vgg" in model_name:
        # Change the model to accept single channel images
        model.features[0] = torch.nn.Conv2d(1, 64, kernel_size=3, padding=1)

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

    elif  "squeezenet" in model_name:
        model.features[0] = torch.nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2))

    elif "wide_resnet" in model_name:
        model.conv1 = torch.nn.Conv2d(
            1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
    else:
        raise ValueError("Unsupported model name")

    return model

def apply_initialization(model, init_type):
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            if init_type == 'uniform':
                nn.init.uniform_(module.weight)
            elif init_type == 'normal':
                nn.init.normal_(module.weight)
            elif init_type == 'constant':
                nn.init.constant_(module.weight, 0.5)  # You can change the constant value
            elif init_type == 'ones':
                nn.init.ones_(module.weight)
            elif init_type == 'zeros':
                nn.init.zeros_(module.weight)
            elif init_type == 'xavier_uniform':
                nn.init.xavier_uniform_(module.weight)
            elif init_type == 'xavier_normal':
                nn.init.xavier_normal_(module.weight)
            elif init_type == 'kaiming_uniform':
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            elif init_type == 'kaiming_normal':
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(module.weight)

            if module.bias is not None:
                nn.init.zeros_(module.bias)


class EnsembleModel:
    def __init__(self, model_names):
        self.models = [get_model(name) for name in model_names]

    def forward(self, x):
        # Get predictions from all models
        preds = [model(x) for model in self.models]
        # Combine predictions. Here we are simply averaging them
        ensemble_pred = torch.mean(torch.stack(preds), dim=0)
        return ensemble_pred

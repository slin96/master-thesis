import os

import torch
from torchvision import models

MODEL_OUTPUT = 'model-{}-output'
MODEL_WEIGHTS = 'model-{}-weights'
MODELS = [models.alexnet, models.vgg19, models.resnet18, models.resnet50, models.resnet152]


def save_output(root, model_getter, output):
    output_file = os.path.join(root, MODEL_OUTPUT.format(model_getter.__name__))
    torch.save(output, output_file)


def get_output(root, model_getter):
    read_file = os.path.join(root, MODEL_OUTPUT.format(model_getter.__name__))
    return torch.load(read_file)


def save_model_weights(root, model_getter, model):
    output_file = os.path.join(root, MODEL_WEIGHTS.format(model_getter.__name__))
    torch.save(model.state_dict(), output_file)


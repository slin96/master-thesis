import os

import torch

MODEL_OUTPUT = 'model-{}-output'


def save_output(root, model_getter, output):
    output_file = os.path.join(root, MODEL_OUTPUT.format(model_getter.__name__))
    torch.save(output, output_file)
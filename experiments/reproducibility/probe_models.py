import argparse

import torch
from mmlib.deterministic import set_deterministic
from mmlib.equal import blackbox_model_equal, whitebox_model_equal, model_equal
from mmlib.helper import imagenet_input, imagenet_target
from mmlib.probe import probe_training, ProbeInfo
from torch import nn

from experiments.models.googlenet import googlenet
from experiments.models.mobilenet import mobilenet_v2
from experiments.models.resnet152 import resnet152
from experiments.models.resnet18 import resnet18
from experiments.models.resnet50 import resnet50


def deterministic_backward_compare(model_class, device, forward_indices=None):
    dummy_input = imagenet_input()
    dummy_target = imagenet_target(dummy_input)
    loss_func = nn.CrossEntropyLoss()

    set_deterministic()
    model1 = model_class(pretrained=True)
    optimizer1 = torch.optim.SGD(model1.parameters(), 1e-3)
    summary1 = probe_training(model1, dummy_input, optimizer1, loss_func, dummy_target, device,
                              forward_indices=forward_indices)

    set_deterministic()
    model2 = model_class(pretrained=True)
    optimizer2 = torch.optim.SGD(model2.parameters(), 1e-3)
    summary2 = probe_training(model2, dummy_input, optimizer2, loss_func, dummy_target, device,
                              forward_indices=forward_indices)

    # fields that should for sure be the same
    common = [ProbeInfo.LAYER_NAME, ProbeInfo.FORWARD_INDEX]

    # fields where we might expect different values
    compare = [ProbeInfo.INPUT_TENSOR, ProbeInfo.OUTPUT_TENSOR, ProbeInfo.GRAD_INPUT_TENSOR,
               ProbeInfo.GRAD_OUTPUT_TENSOR]

    summary1.compare_to(summary2, common, compare)

    # also the models should be equal
    blackbox_eq = blackbox_model_equal(model1, model2, imagenet_input)
    whitebox_eq = whitebox_model_equal(model1, model2)
    models_are_equal = model_equal(model1, model2, imagenet_input)
    print()
    print('Also the models should be the same - compare the models')
    print('models_are_equal (blackbox): {}'.format(blackbox_eq))
    print('models_are_equal (whitebox): {}'.format(whitebox_eq))
    print('models_are_equal: {}'.format(models_are_equal))


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device used: {}'.format(device))

    evaluate = [
        (mobilenet_v2, list(range(0, 10))), (mobilenet_v2, list(range(149, 159))),
        (googlenet, list(range(0, 10))), (googlenet, list(range(187, 197))),
        (resnet18, list(range(0, 10))), (resnet18, list(range(59, 69))),
        (resnet50, list(range(0, 10))), (resnet50, list(range(165, 175))),
        (resnet152, list(range(0, 10))), (resnet152, list(range(505, 515)))
    ]

    for model, forward_indices in evaluate:
        print('model: {}'.format(model.__name__))
        deterministic_backward_compare(model, device, forward_indices=forward_indices)
        print()

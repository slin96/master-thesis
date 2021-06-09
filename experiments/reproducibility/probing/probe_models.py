import argparse
import os

import mmlib.deterministic
import torch
from mmlib.deterministic import set_deterministic
from mmlib.equal import blackbox_model_equal, whitebox_model_equal, model_equal
from mmlib.probe import probe_training, ProbeInfo, ProbeSummary
from mmlib.util.dummy_data import imagenet_input, imagenet_target
from torch import nn

from experiments.models.googlenet import googlenet
from experiments.models.mobilenet import mobilenet_v2
from experiments.models.resnet152 import resnet152
from experiments.models.resnet18 import resnet18
from experiments.models.resnet50 import resnet50


def create_and_save_summary(model_class, device, save_path, forward_indices=None):
    mmlib.deterministic.set_deterministic()

    dummy_input = imagenet_input()
    dummy_target = imagenet_target(dummy_input)

    loss_func = nn.CrossEntropyLoss()
    model = model_class(pretrained=True)
    optimizer1 = torch.optim.SGD(model.parameters(), 1e-3)
    summary = probe_training(model, dummy_input, optimizer1, loss_func, dummy_target, device,
                             forward_indices=forward_indices)

    summary_file_name = _summary_file_name(forward_indices, model_class.__name__)
    summary_save_path = os.path.join(save_path, summary_file_name)
    summary.save(summary_save_path)


def _summary_file_name(forward_indices, _class_name):
    summary_file_name = '{}-{}-{}'.format(_class_name, str(forward_indices[0]), forward_indices[1])
    return summary_file_name


def compare_summaries(model_class, forward_indices, summary_path_1, summary_path_2):
    # fields that should for sure be the same
    common = [ProbeInfo.LAYER_NAME, ProbeInfo.FORWARD_INDEX]

    # fields where we might expect different values
    compare = [ProbeInfo.INPUT_TENSOR, ProbeInfo.OUTPUT_TENSOR, ProbeInfo.GRAD_INPUT_TENSOR,
               ProbeInfo.GRAD_OUTPUT_TENSOR]

    summary_file_name = _summary_file_name(forward_indices, model_class.__name__)

    summary_path_1 = os.path.join(summary_path_1, summary_file_name)
    summary_1 = ProbeSummary()
    summary_1.load(summary_path_1)

    summary_path_2 = os.path.join(summary_path_2, summary_file_name)
    summary_2 = ProbeSummary()
    summary_2.load(summary_path_2)

    summary_1.compare_to(summary_2, common, compare)


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


def main(args):
    evaluate = [
        (mobilenet_v2, list(range(0, 10))), (mobilenet_v2, list(range(149, 159))),
        (googlenet, list(range(0, 10))), (googlenet, list(range(187, 197))),
        (resnet18, list(range(0, 10))), (resnet18, list(range(59, 69))),
        (resnet50, list(range(0, 10))), (resnet50, list(range(165, 175))),
        (resnet152, list(range(0, 10))), (resnet152, list(range(505, 515)))
    ]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device used: {}'.format(device))

    for model, forward_indices in evaluate:
        print('create_and_save_summary for model: {}, layers {}'.format(model.__name__, forward_indices))
        create_and_save_summary(model, device, args.save_path, forward_indices=forward_indices)

    if args.compare:
        for model, forward_indices in evaluate:
            print('compare summaries for model: {}'.format(model.__name__))
            compare_summaries(model, forward_indices, args.save_path, args.load_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--compare', type=bool)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--load_path', type=str)
    _args = parser.parse_args()

    return _args


if __name__ == '__main__':
    args = parse_args()
    main(args)

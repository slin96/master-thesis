import torch
from mmlib.deterministic import set_deterministic
from mmlib.probe import probe_training, ProbeInfo
from mmlib.util.dummy_data import imagenet_input, imagenet_target

from experiments.models.googlenet import googlenet


def deterministic_backward_compare(model_class, device, forward_indices=None, dterm=False):
    dummy_input = imagenet_input()
    dummy_target = imagenet_target(dummy_input)
    loss_func = torch.nn.CrossEntropyLoss()

    if dterm:
        set_deterministic()
    model1 = model_class(pretrained=True)
    optimizer1 = torch.optim.SGD(model1.parameters(), 1e-3)
    summary1 = probe_training(model1, dummy_input, optimizer1, loss_func, dummy_target, device,
                              forward_indices=forward_indices)
    if dterm:
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


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device used: {}'.format(device))

    evaluate = [(googlenet, list(range(186, 200)))]

    for model, forward_indices in evaluate:
        print('model: {}'.format(model.__name__))
        deterministic_backward_compare(model, device, forward_indices=forward_indices, dterm=False)
        print()

        print('model: {}'.format(model.__name__))
        deterministic_backward_compare(model, device, forward_indices=forward_indices, dterm=True)
        print()

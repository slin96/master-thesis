import torch.nn


def freeze_all_parameters(model: torch.nn.Module):
    for param in model.parameters():
        param.requires_grad = False


def make_trainable(parameters):
    for params in parameters:
        params.requires_grad = True


def freeze_except_fc(model: torch.nn.Module):
    freeze_all_parameters(model)
    if model.__class__.__name__ == 'ResNet' or model.__class__.__name__ == 'GoogLeNet':
        make_trainable(model.fc.parameters())
    elif model.__class__.__name__ == 'MobileNetV2':
        make_trainable(model.classifier[1].parameters())
    else:
        raise NotImplementedError

from mmlib.deterministic import set_deterministic
from mmlib.helper import imagenet_input
from mmlib.model_equals import equals
from torchvision import models

from experiments.models.googlenet import googlenet
from experiments.models.mobilenet import mobilenet_v2
from experiments.models.resnet152 import resnet152
from experiments.models.resnet18 import resnet18
# TODO turn this into test case
from experiments.models.resnet50 import resnet50

if __name__ == '__main__':
    lib_models = [models.mobilenet_v2, models.googlenet, models.resnet18, models.resnet50, models.resnet152]
    local_models = [mobilenet_v2, googlenet, resnet18, resnet50, resnet152]

    for i in range(len(lib_models)):
        print('Model: {}'.format(lib_models[i].__name__))

        local_net = lib_models[i](pretrained=True)
        lib_net = local_models[i](pretrained=True)

        # check if pretrained version leads to same model
        print('pretrained models are equal: {}'.format(equals(local_net, lib_net, imagenet_input)))

        set_deterministic()
        local_net = lib_models[i]()
        set_deterministic()
        lib_net = local_models[i]()
        # check if initialized models are the same
        print('deterministically initialized models are equal: {}'.format(equals(local_net, lib_net, imagenet_input)))

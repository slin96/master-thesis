import socket

from experiments.models.googlenet import googlenet
from experiments.models.mobilenet import mobilenet_v2
from experiments.models.resnet152 import resnet152
from experiments.models.resnet18 import resnet18
from experiments.models.resnet50 import resnet50

MOBILENET = "mobilenet"
GOOGLENET = "googlenet"
RESNET_18 = "resnet18"
RESNET_50 = "resnet50"
RESNET_152 = "resnet152"

models_dict = {MOBILENET: mobilenet_v2, GOOGLENET: googlenet, RESNET_18: resnet18, RESNET_50: resnet50,
               RESNET_152: resnet152}


def initial_train(model_ref):
    # we model the initial training of the model by just making use of the pretrained version
    return model_ref(pretrained=True)


def update_model():
    # TODO to implement
    # for now return resnet18 for testing
    return resnet18(pretrained=True)

def inform(message, sender, receiver):
    # socket.SOCK_DGRAM use UDP
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(sender)
    sock.sendto(message, receiver)

import argparse

from torch import nn
from torch.backends import cudnn
from torchvision import models, datasets

from experiments.imagenet.imagenet_utils import inference_transforms
from experiments.imagenet.processing import validate

NUMBER_BATCHES = 3

MODELS = [models.resnet18]


# MODELS = [models.alexnet, models.vgg19, models.resnet18, models.resnet50, models.resnet152]


def experiment_inference(model, data, number_batches):
    loss_func = nn.CrossEntropyLoss()

    # TODO check what this var does
    cudnn.benchmark = True

    print('imagenet val')
    img_val_output = validate(model, data, loss_func, get_outputs=True, number_batches=number_batches)

    print('test')


def main(args):
    imgnet_val_data = datasets.ImageNet(args.imagenet_root, 'val', transform=inference_transforms)

    for mod in MODELS:
        model = mod(pretrained=True)
        experiment_inference(model, imgnet_val_data, NUMBER_BATCHES)


def parse_args():
    parser = argparse.ArgumentParser(description='Inference experiment script')
    parser.add_argument('--imagenet-root', help='imagenet root path for')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)

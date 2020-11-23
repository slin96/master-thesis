from torchvision import models
from torchsummary import summary

if __name__ == '__main__':
    models = [models.alexnet, models.vgg19, models.resnet18, models.resnet50, models.resnet152]

    for mod in models:
        model = mod()
        print('Model: {}'.format(mod.__name__))
        summary(model, (3, 224, 224))
        print('\n\n')

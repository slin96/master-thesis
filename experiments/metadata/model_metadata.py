from torchsummary import summary
from torchvision import models

if __name__ == '__main__':
    models = [models.alexnet, models.vgg19, models.shufflenet_v2_x0_5, models.mobilenet_v2, models.googlenet,
              models.resnet18, models.resnet50, models.resnet152]

    for mod in models:
        model = mod()
        print('Model: {}'.format(mod.__name__))
        summary(model, (3, 224, 224))
        print('\n\n')

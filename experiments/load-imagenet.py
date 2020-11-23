import torchvision

if __name__ == '__main__':
    root_path = '/Users/nils/Studium/master-thesis/repo/experiments/datasets/imagenet/2012'
    imagenet_data = torchvision.datasets.ImageNet(root_path, split='val')

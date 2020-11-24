import torchvision

if __name__ == '__main__':
    root_path = '/hpi/fs00/home/nils.strassenburg/datasets/imgnet'
    imagenet_data = torchvision.datasets.ImageNet(root_path, split='val')
    first_elem = imagenet_data[0]

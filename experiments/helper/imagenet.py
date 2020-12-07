from torch.utils.data import DataLoader
from torchvision import transforms

# THIS CODE IS COPIED /INSPIRED BY:
# https://github.com/pytorch/examples/blob/master/imagenet/main.py

# TODO link github issue(/explain
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

inference_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize, ])

train_transforms = transforms.Compose([
    # TODO check if this is deterministic when using seed
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])


def cpu_inference(model, dataset, args):
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                             pin_memory=True)


def gpu_inference():
    pass


def cpu_train(model, dataset, args):
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                             pin_memory=True)


def gpu_train():
    pass

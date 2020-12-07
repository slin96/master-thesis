import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# THIS CODE IS COPIED /INSPIRED BY:
# https://github.com/pytorch/examples/blob/master/imagenet/main.py

# TODO link github issue/explain
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


def inference(model, dataset, batch_size, loader_workers, use_gpu=False, gpu=None):
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=loader_workers,
                             pin_memory=True)

    if use_gpu:
        # load model on gpu
        model.cuda(gpu)

    # set model to eval mode
    model.eval()

    outputs = []
    with torch.no_grad():
        for i, (images, target) in enumerate(data_loader):
            if use_gpu:
                # load images to gpu
                # TODO check if blocking influences repeatability
                images = images.cuda(gpu, non_blocking=True)

            output = model(images)
            outputs.append(output)

    return outputs


def cpu_train(model, dataset, args):
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                             pin_memory=True)


def gpu_train():
    pass

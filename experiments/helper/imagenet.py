import os
import time

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
# Explanation for magic numbers: https://github.com/pytorch/vision/pull/1965
from torchvision.models import resnet18

# THIS CODE IS COPIED /INSPIRED BY:
# https://github.com/pytorch/examples/blob/master/imagenet/main.py
from experiments.data.custom.custom_coco import CustomCoco

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

def train_epoch(model, dataset, batch_size, loader_workers, loss_func, optimizer, epoch, use_gpu=False, gpu=None,
                print_freq=1):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=loader_workers,
                                              pin_memory=True, )
    # # TODO check params
    # data_loader = torch.utils.data.DataLoader(
    #     dataset, batch_size=batch_size, shuffle=True,
    #     num_workers=1, pin_memory=True, sampler=None)

    if use_gpu:
        # load model on gpu
        torch.cuda.set_device(gpu)
        model = model.cuda(gpu)

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(data_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_gpu:
            # TODO check if blocking influences repeatability
            images = images.cuda(gpu, non_blocking=True)
        if torch.cuda.is_available():
            # TODO check if blocking influences repeatability
            target = target.cuda(gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = loss_func(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            progress.display(i)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def validate(val_data, model, criterion, gpu, print_freq, get_outputs=False):
    val_loader = torch.utils.data.DataLoader(
        val_data,
        # TODO parameters
        batch_size=16, shuffle=False,
        num_workers=1, pin_memory=True)

    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    outputs = []

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if gpu is not None:
                images = images.cuda(gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                progress.display(i)

            if get_outputs:
                outputs.append(output)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    if get_outputs:
        return top1.avg, output
    else:
        return top1.avg


if __name__ == '__main__':
    model = resnet18(pretrained=True)
    loss_func = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), 1e-4,
                                momentum=0.9,
                                weight_decay=1e-4)

    # # use the same data for inference and train just for testing
    # inference_coco_data = CustomCoco('/Users/nils/Studium/master-thesis/repo/tmp/cutsom-coco-data',
    #                        '/Users/nils/Studium/master-thesis/repo/tmp/cutsom-coco-data/coco_meta.json',
    #                        transform=inference_transforms)
    #
    # train_coco_data = CustomCoco('/Users/nils/Studium/master-thesis/repo/tmp/cutsom-coco-data',
    #                        '/Users/nils/Studium/master-thesis/repo/tmp/cutsom-coco-data/coco_meta.json',
    #                        transform=train_transforms)
    #
    # # use the same data for inference and train just for testing
    # root_path = '/Users/nils/Studium/master-thesis/repo/tmp/imgnet'
    # # inference_imagenet_data = torchvision.datasets.ImageNet(root_path, split='val', transform=inference_transforms)
    # train_imagenet_data = torchvision.datasets.ImageNet(root_path, split='val', transform=train_transforms)
    #
    # # outputs_img = inference(model, inference_imagenet_data, 64, 1)
    # # outputs_coco = inference(model, inference_coco_data, 64, 1)



    # TODO check what this var does
    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join('/Users/nils/Studium/master-thesis/repo/tmp/imgnet', 'val')
    valdir = os.path.join('/Users/nils/Studium/master-thesis/repo/tmp/imgnet')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])


    val_data = datasets.ImageNet(valdir, 'val', transform=inference_transforms)
    # val_data = CustomCoco('/Users/nils/Studium/master-thesis/repo/tmp/cutsom-coco-data',
    #                        '/Users/nils/Studium/master-thesis/repo/tmp/cutsom-coco-data/coco_meta.json',
    #                        transform=inference_transforms)

    # out = validate(val_data, model, loss_func, None, 1, get_outputs=True)

    train_data = datasets.ImageNet(valdir, 'val', transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))


    train_epoch(model, train_data, 64, 1, loss_func, optimizer, 1)
    # train_epoch(model, train_coco_data, 64, 1, loss_func, optimizer, 2)

    print('test')

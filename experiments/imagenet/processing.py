import time

import torch
from torch.utils.data import DataLoader

from experiments.imagenet.imagenet_utils import AverageMeter, ProgressMeter, accuracy


# THIS CODE IS COPIED /INSPIRED BY:
# https://github.com/pytorch/examples/blob/master/imagenet/main.py

# Explanation for magic numbers: https://github.com/pytorch/vision/pull/1965


def train_epoch(model, data, loss_func, optimizer, epoch=0, batch_size=64, num_workers=1, use_gpu=False, gpu=None,
                print_freq=1, get_outputs=False, number_batches=None):
    # TODO check params for reproducibility
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                               pin_memory=True)

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
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    outputs = []

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
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

        if get_outputs:
            outputs.append(output)

        if not in_number_of_batches(i, number_batches):
            break

    if get_outputs:
        return output


def validate(model, data, loss_func, batch_size=64, num_workers=1, use_gpu=False, gpu=None,
             print_freq=1, get_outputs=False, number_batches=None):
    val_loader = torch.utils.data.DataLoader(
        data,
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    if use_gpu:
        # load model on gpu
        torch.cuda.set_device(gpu)
        model = model.cuda(gpu)

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

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                progress.display(i)

            if get_outputs:
                outputs.append(output)

            if not in_number_of_batches(i, number_batches):
                break

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    if get_outputs:
        return output
    else:
        return top1.avg


def in_number_of_batches(i, number_batches):
    return number_batches is None or i < number_batches - 1

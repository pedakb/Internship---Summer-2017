import shutil
import time

import torch.optim
import torch.utils.data
from torch.autograd import Variable

import matplotlib.pyplot as plt
import numpy as np

from models import *
from datasets import *


def train(train_loader, model, criterion, optimizer, epoch, use_gpu, print_freq, export, ex_filename):
    abs_end = time.time()
    train_time = AverageMeter()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    cnt = 0
    for inputs, target in train_loader:
        # measure data loading time
        if use_gpu:
            target = target.cuda(async=True)
            inputs_var = Variable(inputs.cuda())
            target_var = Variable(target.cuda())
        else:
            inputs_var = Variable(inputs)
            target_var = Variable(target)

        # compute output
        output = model(inputs_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec2 = accuracy(output.data, target, topk=(1, 2))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top2.update(prec2[0], inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        cnt += 1
        if cnt % print_freq == 0:
            print('     |Batch: [{1:3}/{2:3}] || '
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) sec. || '
                  'Loss: {loss.val:.4f} ({loss.avg:.4f}) || '
                  'Prec@1: {top1.val:6.2f} ({top1.avg:6.2f}) % || '
                  'Prec@2: {top2.val:6.2f} ({top2.avg:6.2f}) % |'.format(
                   epoch + 1, cnt, len(train_loader), batch_time=batch_time,
                   loss=losses, top1=top1, top2=top2))
    train_time.update(time.time() - abs_end)
    print('     Train Accuracy(top 1) : {top1.avg:.3f} %\n     Train Accuracy(top 2) : {top2.avg:.3f} %\n'
          '     Elapsed Time : {elapsed_time:.3f} sec.'
          .format(top1=top1, top2=top2, elapsed_time=train_time.avg))
    if export:
        with open(ex_filename + '.txt', 'a') as results:
            results.write('Epoch {epoch}: \n'
                          '[Train] Accuracy: {top1.avg:.3f}% '
                          'Elapsed Time: {elapsed_time:.3f}sec\n'.format(epoch=epoch + 1,
                                                                         top1=top1,
                                                                         elapsed_time=train_time.avg))
        results.close()


def validate(val_loader, model, criterion, use_gpu, print_freq, export, ex_filename):
    abs_end = time.time()
    val_time = AverageMeter()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    cnt = 0
    end = time.time()
    for inputs, target in val_loader:
        if use_gpu:
            target = target.cuda(async=True)
            inputs_var = Variable(inputs.cuda())
            target_var = Variable(target.cuda())
        else:
            inputs_var = Variable(inputs)
            target_var = Variable(target)

        # compute output
        output = model(inputs_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec2 = accuracy(output.data, target, topk=(1, 2))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top2.update(prec2[0], inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        cnt += 1
        if cnt % print_freq == 0:
            print('     |Batch: [{0:3}/{1:3}] || '
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) sec. || '
                  'Loss: {loss.val:.4f} ({loss.avg:.4f}) || '
                  'Prec@1: {top1.val:6.2f} ({top1.avg:6.2f}) % || '
                  'Prec@2: {top2.val:6.2f} ({top2.avg:6.2f}) % | '.format(
                   cnt, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top2=top2))
    val_time.update(time.time() - abs_end)
    print('     Test Accuracy(top 1) : {top1.avg:.3f} %\n     Test Accuracy(top 2) : {top2.avg:.3f} %\n'
          '     Elapsed Time : {elapsed_time:.3f} sec.'
          .format(top1=top1, top2=top2, elapsed_time=val_time.avg))
    if export:
        with open(ex_filename + '.txt', 'a') as results:
            results.write(
                '[Validation] Accuracy: {top1.avg:.3f}% '
                'Elapsed Time: {elapsed_time:.3f}sec\n'.format(top1=top1, elapsed_time=val_time.avg)
            )
        results.close()
    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

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


def adjust_learning_rate(optimizer, epoch, learning_rate):
    lr = learning_rate * (0.1 ** (epoch // 30))
    lr = lr * (0.1 ** (epoch // 40))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('learning rate = ', lr)


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul(100.0 / batch_size))
    return res


def create_model(arch, input_channels, is_retrain, use_gpu):
    if arch == 'LeNet':
        if use_gpu:
            model = LeNet(in_channels=input_channels).cuda()
        else:
            model = LeNet(in_channels=input_channels)

        if not is_retrain:
            if use_gpu:
                model.load_state_dict(torch.load('LeNet_gpu.pth'))
            else:
                model.load_state_dict(torch.load('LeNet_cpu.pth'))
        params = [
            [{'params': model.conv1.parameters()}],
            [{'params': model.conv1.parameters()},
             {'params': model.conv2.parameters()}],
            [{'params': model.conv1.parameters()},
             {'params': model.conv2.parameters()},
             {'params': model.fc1.parameters()}],
            model.parameters()
        ]

    elif arch == 'CIFAR':
        if use_gpu:
            model = CIFAR(in_channels=input_channels).cuda()
        else:
            model = CIFAR(in_channels=input_channels)

        if not is_retrain:
            if use_gpu:
                model.load_state_dict(torch.load('CIFAR_gpu.pth'))
            else:
                model.load_state_dict(torch.load('CIFAR_cpu.pth'))
        params = [
            [{'params': model.conv1.parameters()}],
            [{'params': model.conv1.parameters()},
             {'params': model.conv2.parameters()}],
            [{'params': model.conv1.parameters()},
             {'params': model.conv2.parameters()},
             {'params': model.conv3.parameters()}],
            [{'params': model.conv1.parameters()},
             {'params': model.conv2.parameters()},
             {'params': model.conv3.parameters()},
             {'params': model.fc1.parameters()}],
            model.parameters()
        ]

    return model, params


def define_loss_function(use_gpu):
    if use_gpu:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss()
    return criterion


def predict(inputs, model, use_gpu):
    batch_time = AverageMeter()
    if use_gpu:
        inputs_var = Variable(inputs.cuda())
    else:
        inputs_var = Variable(inputs)
    # compute output
    end = time.time()
    output = model(inputs_var)
    _, pred_label = torch.max(output.data, 1)
    # measure elapsed time
    batch_time.update(time.time() - end)
    return pred_label


def visualize(model, dataset, use_gpu):
    dataset.val_loader.batch_size = 1
    for inputs, target in dataset.val_loader:
        # predict
        pred = predict(inputs, model, use_gpu)
        print('Network Prediction: ', dataset.classes[pred[0, 0]],
              '\nTarget: ', dataset.classes[target[0]])
        # Display image
        print('Displaying Example Image')
        img = inputs.squeeze(0).numpy()
        if dataset.mode == 'RGB':
            img = img.transpose(1, 2, 0)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
        else:
            img = img.squeeze(0)
            mean = np.array([0.5])
            std = np.array([1.0])

        img = std * img + mean
        if dataset.mode == 'gray':
            plt.imshow(img, cmap='gray')

        else:
            plt.imshow(img)

        plt.show()
        s = input('Paused - press enter to continue, q to exit:')
        if s == 'q':
            break



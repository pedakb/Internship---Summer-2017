import argparse
import torch.optim
import torch.utils.data

from distortions import *
from datasets import *
from utils import *

# define parsers
model_names = ['LeNet', 'CIFAR']
distortion_types = ['motion_blur', 'gaussian_noise', 'combination', 'none']
dataset_names = ['MNIST', 'CIFAR-10', 'ImageNet']

# define parsers
parser = argparse.ArgumentParser(description='SCL Summer 2017 Internship')
# data
parser.add_argument('--data', '-d', metavar='DATA', default='MNIST',
                    choices=dataset_names,
                    help='dataset: ' +
                         ' | '.join(dataset_names) +
                         ' (default: MNIST)')
# arch
parser.add_argument('--arch', '-a', metavar='ARCH', default='LeNet',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: LeNet)')
# workers
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# epochs
parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run (default: 20)')
# start-epoch
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
# batch-size
parser.add_argument('-b', '--batch-size', default=100, type=int,
                    metavar='N', help='mini-batch size (default: 100)')
# learning-rate
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate (default: 0.001)')
# momentum
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum (default: 0.9)')
# weight-decay
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# print-freq
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 100)')
# evaluate
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
# visualize
parser.add_argument('-v', '--visualize', dest='visualize', action='store_true',
                    help='visualize model prediction')
# fine-tune
parser.add_argument('--fine-tune', dest='fine_tune', action='store_true',
                    help='fine-tune model on distorted dataset')
# first-N layers
parser.add_argument('--first-n', '-n', default=0, type=int,
                    metavar='N', help='fine-tuning first-N layers (default: 0)')
# retrain
parser.add_argument('--retrain', dest='retrain', action='store_true',
                    help='retrain model on distorted dataset')
# distortion-type
parser.add_argument('--distortion', metavar='DISTORTION', default='none',
                    choices=distortion_types,
                    help='distortion type: ' +
                         ' | '.join(distortion_types) +
                         ' (default: none)')
# distortion-level
parser.add_argument('--dist-level', default=1, type=int,
                    metavar='level', help='distortion parameter (default: 1)')
# gpu
parser.add_argument('--gpu', dest='use_gpu', action='store_true',
                    help='use gpu')
# save
parser.add_argument('--save', dest='save', action='store_true',
                    help='save trained model')
# export
parser.add_argument('--export', dest='export', action='store_true',
                    help='export results into a .txt file')
#
parser.add_argument('--ex-filename', metavar='EXFILE', default='result',
                    help='exported file name (default: result)')
# global vars
best_prec1 = 0
args = parser.parse_args()

# dictionaries
datasets_size_dict = {
    'MNIST': [28, 28, 1],
    'CIFAR-10': [32, 32, 3],
    'ImageNet': [240, 240, 3]
}

datasets_classes_dict = {
    'MNIST': (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    'CIFAR-10': ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
}

datasets_modes_dict = {
    'MNIST': {'gray': 1},
    'CIFAR-10': {'RGB': 3},
    'ImageNet': {'RGB': 3}
}

distortion_levels_dict = {
    'motion_blur': [4, 6, 8, 10],
    'gaussian_noise': [3, 5, 7, 9],
    'combination': [1, 2, 3, 4],
    'none': [1, 2, 3, 4]
}

datasets_info_dict = {
    'size': datasets_size_dict,
    'classes': datasets_classes_dict,
    'mode': datasets_modes_dict,
}


def main():
    global args, best_prec1
    # Print Info.
    print_info()
    # Export Info.
    export_info(args.export, args.ex_filename)

    # initialization

    if args.fine_tune:
        first_n = args.first_n - 1
    else:
        first_n = -1

    use_gpu = torch.cuda.is_available() & args.use_gpu

    # define Distortion
    distortion = Distortion(args.distortion, datasets_info_dict['size'][args.data], args.dist_level, distortion_levels_dict)
    none = Distortion('none', datasets_info_dict['size'][args.data], 1, distortion_levels_dict)

    # load dataset
    dataset = Dataset(args.data, none.trans(), datasets_info_dict)
    distorted_dataset = Dataset(args.data, distortion.trans(), datasets_info_dict)

    # create loader
    dataset.create_loader(args.batch_size, args.workers)
    distorted_dataset.create_loader(args.batch_size, args.workers)
    # test(distorted_dataset.val_loader)
    # create model
    model, params = create_model(args.arch, datasets_modes_dict[args.data][distorted_dataset.mode], args.retrain, use_gpu)

    # define loss function (criterion) and optimizer
    criterion = define_loss_function(use_gpu)

    # define optimizing method
    optimizer = torch.optim.SGD(params[first_n], args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # training...
    if not args.fine_tune | args.retrain:
        print('Training...')
        end = time.time()
        total_elapsed_time = AverageMeter()
        for epoch in range(args.start_epoch, args.epochs):
            print('-> Epoch[', epoch + 1, ']:')
            adjust_learning_rate(optimizer, epoch, args.lr)
            # train for one epoch
            print('     _______***Train***______________________________'
                  '_________________________________________________________________________________')
            train(distorted_dataset.train_loader, model, criterion, optimizer, epoch, use_gpu,
                  args.print_freq, args.export, args.ex_filename)
            # evaluate on validation set
            print('     _______***Test***______________________________'
                  '__________________________________________________________________________________')
            prec1 = validate(distorted_dataset.val_loader, model, criterion, use_gpu,
                             args.print_freq, args.export, args.ex_filename)
            # remember best top 1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best)

        total_elapsed_time.update(time.time() - end)
        print('Total Elapsed Time :'
              ' {total_elapsed_time.avg:.3f} sec.\n'.format(total_elapsed_time=total_elapsed_time))
        if args.export:
            with open('results.txt', 'a') as results:
                results.write('Total Elapsed Time: '
                              '{total_elapsed_time.avg:.3f}sec\n'.format(total_elapsed_time=total_elapsed_time))
            results.close()
    # Evaluation
    if args.evaluate:
        print('     _______***Evaluate***______________________________'
              '______________________________________________________________________________')
        validate(distorted_dataset.val_loader, model, criterion, use_gpu, args.print_freq,
                 args.export, args.ex_filename)
    # Saving
    if args.save:
        if use_gpu:
            torch.save(model.state_dict(), args.arch + "_gpu.pth")
        else:
            torch.save(model.state_dict(), args.arch + "_cpu.pth")
    # Visualization
    if args.visualize:
        print('Visualization...')
        visualize(model, distorted_dataset, use_gpu)


def print_info():
    print('##################################\n'
          '### SCL Summer 2017 Internship ###\n'
          '##################################\n'
          '[INFO]\n'
          '\t Dataset         : {dataset} \n'
          '\t Network Arch.   : {arch}\n'
          '[Learning Parameters]\n'
          '\t Batch Size      : {batch_size}\n'
          '\t Start Epoch     : {start_epoch}\n'
          '\t Num. of Epochs  : {epochs}\n'
          '\t Learning Rate   : {lr}\n'
          '\t Momentum        : {mom}\n'
          '\t Weight Decay    : {weight_decay}\n'
          '[Distortion]\n'
          '\t Distortion Type : {dist_type}\n'
          '\t Distortion Level: {dist_level}\n'
          '[Processing]\n'
          '\t Num. Workers    : {num_workers}\n'
          '\t Using GPU       : {gpu}\n'
          '[Running Mode]\n'
          '\t Evaluate        : {eval}\n'
          '\t Retrain         : {retrain}\n'
          '\t Fine-Tune       : {fine_tune}\n'
          '\t First-N Layers  : {first_n}\n'
          '\t Visualize       : {visualize}\n'
          '==================================='
          .format(dataset=args.data, arch=args.arch,
                  batch_size=args.batch_size, start_epoch=args.start_epoch, epochs=args.epochs, lr=args.lr,
                  mom=args.momentum, weight_decay=args.weight_decay,
                  dist_type=args.distortion, dist_level=args.dist_level,
                  num_workers=args.workers, gpu=args.use_gpu,
                  eval=args.evaluate, retrain=args.retrain, fine_tune=args.fine_tune, first_n=args.first_n,
                  visualize=args.visualize))
    return


def export_info(export, ex_filename):
    if export:
        with open(ex_filename + '.txt', 'w') as results:
            results.write('##################################\n'
                          '### SCL Summer 2017 Internship ###\n'
                          '##################################\n'
                          '[INFO]\n'
                          '\t Dataset         : {dataset} \n'
                          '\t Network Arch.   : {arch}\n'
                          '[Learning Parameters]\n'
                          '\t Batch Size      : {batch_size}\n'
                          '\t Start Epoch     : {start_epoch}\n'
                          '\t Num. of Epochs  : {epochs}\n'
                          '\t Learning Rate   : {lr}\n'
                          '\t Momentum        : {mom}\n'
                          '\t Weight Decay    : {weight_decay}\n'
                          '[Distortion]\n'
                          '\t Distortion Type : {dist_type}\n'
                          '\t Distortion Level: {dist_level}\n'
                          '[Processing]\n'
                          '\t Num. Workers    : {num_workers}\n'
                          '\t Using GPU       : {gpu}\n'
                          '[Running Mode]\n'
                          '\t Evaluate        : {eval}\n'
                          '\t Retrain         : {retrain}\n'
                          '\t Fine-Tune       : {fine_tune}\n'
                          '\t First-N Layers  : {first_n}\n'
                          '\t Visualize       : {visualize}\n'
                          '===================================\n'
                          .format(dataset=args.data, arch=args.arch,
                                  batch_size=args.batch_size, start_epoch=args.start_epoch, epochs=args.epochs, lr=args.lr,
                                  mom=args.momentum, weight_decay=args.weight_decay,
                                  dist_type=args.distortion, dist_level=args.dist_level,
                                  num_workers=args.workers, gpu=args.use_gpu,
                                  eval=args.evaluate, retrain=args.retrain, fine_tune=args.fine_tune, first_n=args.first_n,
                                  visualize=args.visualize))
            results.close()
    return

if __name__ == '__main__':
    main()

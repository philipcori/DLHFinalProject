from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.models as models
from explore_version_03.models import imagenet as customized_models
from explore_version_03.data.xray_dataset_0328 import XrayDataset
from explore_version_03.utils import Bar, AverageMeter, accuracy, mkdir_p
import csv

# Models
default_model_names = sorted(name for name in models.__dict__
                             if name.islower() and not name.startswith("__")
                             and callable(models.__dict__[name]))

customized_models_names = sorted(name for name in customized_models.__dict__
                                 if name.islower() and not name.startswith("__")
                                 and callable(customized_models.__dict__[name]))

for name in customized_models.__dict__:
    if name.islower() and not name.startswith("__") and callable(customized_models.__dict__[name]):
        models.__dict__[name] = customized_models.__dict__[name]

model_names = default_model_names + customized_models_names

# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Experiment ID
parser.add_argument('--experimentID', default='%s_20200407_multiclass_cv5', type=str, metavar='E_ID',
                    help='ID of Current experiment')

# Datasets
parser.add_argument('-d', '--data', default=
'./data_preprocess/standard_data_multiclass_0922_crossentropy/exp_%s_list_cv5.pkl', type=str)
parser.add_argument('--label_file', default='./exp_data/metadata.csv', type=str)
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
parser.add_argument('--load_size', type=int, default=336, help='336 scale images to this size')
parser.add_argument('--crop_size', type=int, default=224, help='224 then crop to this size')
parser.add_argument('--preprocess', type=str, default='resize_and_crop',
                    help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')

# Optimization options
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=10, type=int, metavar='N',
                    help='train batchsize (default: 256)')
parser.add_argument('--test-batch', default=10, type=int, metavar='N',
                    help='test batchsize (default: 200)')
parser.add_argument('--serial_batches', action='store_true',
                    help='if true, takes images in order to make batches, otherwise takes them randomly')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

# Checkpoints
parser.add_argument('-c', '--checkpoint', default='./explore_version_03/checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('-ck_n', '--checkpoint_saved_n', default=2, type=int, metavar='saved_N',
                    help='each N epoch to save model')

# Test Outputs
parser.add_argument('--test', default=True, dest='test', action='store_true',
                    help='evaluate model on test set')
parser.add_argument('--results', default='./explore_version_03/results', type=str, metavar='PATH',
                    help='path to save experiment results (default: results)')
parser.add_argument('-r', '--resume', default='', type=str, metavar='PATH',
                    help='saved model ID for loading checkpoint (default: none)')

# Architecture
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet101',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--cardinality', type=int, default=32, help='ResNet cardinality (group).')
parser.add_argument('--base-width', type=int, default=4, help='ResNet base width.')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')

# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--pretrained', dest='pretrained', default=True, action='store_false',
                    help='use pre-trained model')
# Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()
# use_cuda = False

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc_first_stage = 0  # best test accuracy


def main():
    global best_acc_first_stage

    results_folder = 'hierarchical'

    arch_first_stage = 'resnet152'
    arch_second_stage = 'densenet161'

    experimentID_first_stage = '%s_20200407_multiclass_cv5' % arch_first_stage
    experimentID_second_stage = '%s_20200407_multiclass_cv5' % arch_second_stage

    checkpoint_dir_first_stage = os.path.join('./explore_version_03/checkpoint_first_stage', experimentID_first_stage)
    checkpoint_dir_second_stage = os.path.join('./explore_version_03/checkpoint',
                                               experimentID_second_stage)

    print(checkpoint_dir_first_stage)
    print(checkpoint_dir_second_stage)

    if not os.path.isdir(checkpoint_dir_first_stage):
        print('first stage model not found!!!')
        exit()

    if not os.path.isdir(checkpoint_dir_second_stage):
        print('second stage model not found!!!')
        exit()

    # Data loading code
    train_dataset = XrayDataset(args, 'train')
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.train_batch,
                                               shuffle=False,
                                               num_workers=int(args.workers))
    valid_dataset = XrayDataset(args, 'valid')
    val_loader = torch.utils.data.DataLoader(valid_dataset,
                                             batch_size=args.test_batch,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True)
    test_dataset = XrayDataset(args, 'test')
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.test_batch,
                                              shuffle=False,
                                              num_workers=args.workers,
                                              pin_memory=True)
    #    loders = [(test_loader, 'test')]
    loders = [(train_loader, 'train'), (val_loader, 'valid'), (test_loader, 'test')]
    # create model
    model_first_stage = models.__dict__[arch_first_stage](pretrained=True)
    model_first_stage.fc = torch.nn.Linear(2048, 2, bias=True)      # for resnet152

    model_second_stage = models.__dict__[arch_second_stage](pretrained=True)
    model_second_stage.classifier = torch.nn.Linear(2208, 3, bias=True)     # for densenet161

    if use_cuda:
        if arch_first_stage.startswith('alexnet') or arch_first_stage.startswith('vgg'):
            model_first_stage.features = torch.nn.DataParallel(model_first_stage.features)
            model_first_stage.cuda()
        else:
            model_first_stage = torch.nn.DataParallel(model_first_stage).cuda()
            model_second_stage = torch.nn.DataParallel(model_second_stage).cuda()

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model_first_stage.parameters()) / 1000000.0))

    # define loss function (criterion) and optimizer
    #    criterion = focalloss(label_distri = train_distri, model_name = args.arch)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_first_stage.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print('\Test only')
    print('load best checkpoint')
    checkpoint_path_first_stage = os.path.join(checkpoint_dir_first_stage, 'model_best.pth.tar')
    checkpoint_path_second_stage = os.path.join(checkpoint_dir_second_stage, 'model_best.pth.tar')

    print(checkpoint_path_first_stage)
    print(checkpoint_path_second_stage)

    assert os.path.isfile(checkpoint_path_first_stage), 'Error: no checkpoint file found for first stage!'
    assert os.path.isfile(checkpoint_path_second_stage), 'Error: no checkpoint file found for second stage!'

    checkpoint_first_stage = torch.load(checkpoint_path_first_stage)
    best_acc_first_stage = checkpoint_first_stage['best_acc']
    start_epoch_first_stage = checkpoint_first_stage['epoch']
    model_first_stage.load_state_dict(checkpoint_first_stage['state_dict'])
    optimizer.load_state_dict(checkpoint_first_stage['optimizer'])

    if not os.path.isdir(args.results):
        mkdir_p(args.results)
    if not os.path.isdir(os.path.join(args.results, results_folder)):
        mkdir_p(os.path.join(args.results, results_folder))
    results_dir = os.path.join(args.results, results_folder)
    print(results_dir)
    for func in loders:
        test_loss, test_acc, pred_d, real_d = test(func[0], model_first_stage, criterion, use_cuda)

        with open(os.path.join(results_dir, 'result_detail_%s_%s_cv1.csv' % ('hierarchical', func[1])), 'w',
                  newline='') as f:
            csv_writer = csv.writer(f)
            for i in range(len(real_d)):
                x = np.zeros(4)
                x[real_d[i]] = 1
                #                  y = np.exp(pred_d[i])/np.sum(np.exp(pred_d[i]))
                csv_writer.writerow(list(np.array(pred_d[i])) + list(x))

        #        mr = MeasureR(results_dir, test_loss, test_acc)
        #        mr.output()
        print(' Test Loss:  %.8f, Test Acc:  %.4f' % (test_loss, test_acc))
    return


def test(val_loader, model, criterion, use_cuda):
    global best_acc_first_stage

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    real_labels = []
    pred_labels = []
    bar = Bar('Processing', max=len(val_loader))
    for batch_idx, databatch in enumerate(val_loader):
        inputs = databatch['A']
        targets = databatch['B']
        # measure data loading time
        data_time.update(time.time() - end)
        real_labels += list(targets)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)
        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # print('test', loss)
        # measure accuracy and record loss
        prec1 = accuracy(outputs.data, targets.data, topk=(1, 1))
        losses.update(loss.data, inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        pred_labels.append(outputs.detach().cpu().numpy())
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
            batch=batch_idx + 1,
            size=len(val_loader),
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg,
            top5=top5.avg,
        )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg, np.concatenate(pred_labels, 0), np.array(real_labels))


def save_checkpoint(state, epoch_id, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, str(epoch_id) + '.' + filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']


if __name__ == '__main__':
    main()
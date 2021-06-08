import argparse
import os
import time
from datetime import datetime
import torch
import torchvision
import torchvision.transforms as transforms

from model import ResNet18
from utils import AverageMeter, accuracy, replace_labels


os.environ['TZ'] = 'Asia/Seoul'
time.tzset()

parser = argparse.ArgumentParser(description='Training with Clustered ImageNet Labels')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--original-labels', default=False, type=bool, metavar='O',
                    help='use original labels instead of cil')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=40, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate (default: 0.1)', dest='lr')
parser.add_argument('--lr-step', default=10, type=int,
                    metavar='N', help='step size of step scheduler (default: 10)')
parser.add_argument('--lr-step-gamma', default=0.5, type=float,
                    metavar='N', help='gamma value of step scheduler (default: 0.5)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum (default: 0.9)')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--finetuning', default='', type=str, metavar='PATH',
                    help='enable finetuning mode from pre-trained model')

def train(epoch, trainloader, model, criterion, optimizer, replacing_labels=False):
    losses = AverageMeter('Loss', ':.5f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    # switch to train mode
    model.train()

    for i, (images, target) in enumerate(trainloader):
        if replacing_labels:
            target = replace_labels(target)
        images, target = images.cuda(), target.cuda()

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 200 == 0:
            timestamp = datetime.now().replace(microsecond=0).isoformat().replace('T', ' ')
            print(f'{timestamp} [{epoch}][{i}/{len(trainloader)}] {losses} {top1} {top5}')

    return losses.avg, top1.avg, top5.avg


def validate(testloader, model, replacing_labels=False):
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for images, target in testloader:
            if replacing_labels:
                target = replace_labels(target)

            # compute output
            images, target = images.cuda(), target.cuda()
            output = model(images)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    return top1.avg, top5.avg


def main(args):
    replacing_labels = not args.original_labels
    num_classes = 488 if replacing_labels else 1000

    # Make a dir for recording experimental result
    start_time = datetime.now().replace(microsecond=0).isoformat().replace('T', ' ')
    experiment_result_dir = os.path.join('result', start_time)
    os.makedirs(experiment_result_dir, exist_ok=True)

    # Create a log file
    with open(os.path.join(experiment_result_dir, 'log.log'), 'a') as fout:
        for key, value in vars(args).items():
            fout.write(f'- {key}: {value}\n')
        fout.write('---------------------------\n')

    # Dataset
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.4810, 0.4574, 0.4078],
                                     std=[0.2146, 0.2104, 0.2138])

    trainset = torchvision.datasets.ImageFolder(traindir, transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    testset = torchvision.datasets.ImageFolder(valdir, transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ]))
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)


    if args.finetuning:
        # Run finetuning from the pretrained model
        checkpoint = torch.load(args.finetuning)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

        model = ResNet18(1000)
        model.load_state_dict(state_dict)

        # Fix FC layer
        num_ftrs = model.linear.in_features
        model.linear = torch.nn.Linear(num_ftrs, 488)
        model.cuda()
    else:
        # Model
        model = ResNet18(num_classes)
        model.cuda()

    # Loss, optimizer, scheduler
    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_step_gamma)

    # Training
    for epoch in range(1, 1 + args.epochs):
        print(f'Epoch #{epoch}')

        train_loss, train_acc1, train_acc5 = train(epoch, trainloader, model, criterion,
                                                   optimizer, replacing_labels)
        test_acc1, test_acc5 = validate(testloader, model, replacing_labels)

        scheduler.step()

        os.makedirs(os.path.join(experiment_result_dir, 'checkpoints'), exist_ok=True)
        torch.save(model.state_dict(), os.path.join(experiment_result_dir, 'checkpoints', f'{epoch}.pth'))

        with open(os.path.join(experiment_result_dir, 'log.log'), 'a') as fout:
            timestamp = datetime.now().replace(microsecond=0).isoformat().replace('T', ' ')
            fout.write(f'{timestamp}\tEpoch {epoch}\tLoss {train_loss:.4f}\tTrain acc@1 {train_acc1:.2f}'
                       f'\tTrain acc@5 {train_acc5:.2f}\tTest acc@1 {test_acc1:.2f}\tTest acc@5 {test_acc5:.2f}\n')

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

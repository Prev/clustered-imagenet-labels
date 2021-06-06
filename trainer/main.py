import torch
import torchvision
import os
from datetime import datetime

from model import ResNet18
from utils import AverageMeter, accuracy, replace_labels


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

        if i % 100 == 0:
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


def main(replacing_labels=False):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((32, 32)),
        torchvision.transforms.ToTensor(),
    ])

    datapath = '/home/ubuntu/dataset/imagenet_64_64_images/'

    trainset = torchvision.datasets.ImageFolder(datapath + 'train', transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1024, shuffle=True, num_workers=16)
    testset = torchvision.datasets.ImageFolder(datapath + 'val', transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)

    num_classes = 488 if replacing_labels else 1000
    model = ResNet18(num_classes)
    model.cuda()

    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)
    epochs = 30

    for epoch in range(1, 1 + epochs):
        print(f'Epoch #{epoch}')

        train_loss, train_acc1, train_acc5 = train(epoch, trainloader, model, criterion,
                                                   optimizer, replacing_labels)
        test_acc1, test_acc5 = validate(testloader, model, replacing_labels)

        os.makedirs('checkpoints', exist_ok=True)
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'avg_loss': train_loss,
            'train_acc1': train_acc1,
            'train_acc5': train_acc5,
            'test_acc1': test_acc1,
            'test_acc5': test_acc5,
        }, f'checkpoints/resnet18_imagenet-{num_classes}_{epoch}.pth')

        with open(f'training_log.log', 'a') as fout:
            timestamp = datetime.now().replace(microsecond=0).isoformat().replace('T', ' ')
            fout.write(f'{timestamp}\tEpoch {epoch}\tLoss {train_loss:.4f}\tTrain acc@1 {train_acc1:.2f}'
                       f'\tTrain acc@5 {train_acc5:.2f}\tTest acc@1 {test_acc1:.2f}\tTest acc@5 {test_acc5:.2f}\n')


if __name__ == '__main__':
    main()

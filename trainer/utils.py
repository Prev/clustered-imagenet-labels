import torch
import json


# Json donwloaded from
# https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json
with open('data/imagenet_class_index.json', 'r') as fin:
    imagenet_classes = json.loads(fin.read())

with open('data/clustered_imagenet_labels.json', 'r') as fin:
    class_wnid2new_label = json.loads(fin.read())

new_labels = sorted(set(class_wnid2new_label.values()))
new_label2index = {}
for i, classname in enumerate(new_labels):
    new_label2index[classname] = i


def replace_labels(labels):
    ret = torch.LongTensor(len(labels))

    for i, label in enumerate(labels):
        class_idx = str(label.item())
        class_wnid, _ = imagenet_classes[class_idx]
        new_labelname = class_wnid2new_label[class_wnid]
        new_label = torch.tensor(new_label2index[new_labelname])
        ret[i] = new_label
    return ret


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
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

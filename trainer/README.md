## Training Image Classifier with-or-without CIL

For faster training, we use [Downsampled ImageNet](https://arxiv.org/abs/1707.08819) instead of original ImageNet.

The source code basically follows the design of [Pytorch official exmaple](https://github.com/pytorch/examples/tree/master/imagenet), while we removed unnecessary features like distributed learning and added *Clustered ImageNet Labels* (CIL).

### Requirements

- Install PyTorch (pytorch.org) and CUDA for GPU use.
- Download the Downsampled ImageNet dataset from http://www.image-net.org/
- Fix file structure of downloaed Downsampled ImageNet using [downsampled-imagenet-path-fixer](https://github.com/Prev/downsampled-imagenet-path-fixer)
- `pip install -r requirements.txt`


## How to run

To train a model, run `main.py` with the path to the Downsampled ImageNet dataset:

```bash
$ python main.py ~/dataset/imagenet_32_32/
```

In default, our code train a model using CIL. To run with original ImageNet labels, run command:

```bash
$ python main.py ~/dataset/imagenet_32_32/ -original-labels 1
```

To finetune the ImageNet pretrained model with CIL, try:

```bash
$ python main.py ~/dataset/imagenet_32_32/ --finetuning pretrained.pth
```


### Usage

```bash
usage: main.py [-h] [--original-labels O] [-j N] [--epochs N] [-b N] [--lr LR] [--lr-step N] [--lr-step-gamma N] [--momentum M] [--wd W] [--finetuning PATH] DIR

Training with Clustered ImageNet Labels

positional arguments:
  DIR                   path to dataset

optional arguments:
  -h, --help            show this help message and exit
  --original-labels O   use original labels instead of cil
  -j N, --workers N     number of data loading workers (default: 4)
  --epochs N            number of total epochs to run
  -b N, --batch-size N  mini-batch size (default: 256)
  --lr LR, --learning-rate LR
                        initial learning rate (default: 0.1)
  --lr-step N           step size of step scheduler (default: 10)
  --lr-step-gamma N     gamma value of step scheduler (default: 0.5)
  --momentum M          momentum (default: 0.9)
  --wd W, --weight-decay W
                        weight decay (default: 1e-4)
  --finetuning PATH     enable finetuning mode from pre-trained model
```

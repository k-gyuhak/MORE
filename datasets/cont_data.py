import os
from PIL import Image
from itertools import chain
from copy import deepcopy
import numpy as np
import torch
import clip
import torchvision.transforms as transforms
from torch.utils.data.dataset import Subset
from datasets.cifar_mod import CIFAR100, CIFAR10
from datasets.mnist3d import MNIST3D as MNIST
from custom_imagefolder import ImageFolder
from datasets.class_split import ClassSplit

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

device = "cuda" if torch.cuda.is_available() else "cpu"

def augmentations(args):
    if args.model == 'moco_feature':
        args.logger.print('Using the moco augmentations (resize 256)')
        TRANSFORM = transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5071,  0.4866,  0.4409),
                                                 (0.2009,  0.1984,  0.2023))
                            ])
        return TRANSFORM, TRANSFORM
    elif 'resnet' in args.model or 'alexnet' in args.model:
        args.logger.print('Using the standard augmentations (e.g. ResNet, AlexNet, etc.)')
        if args.dataset == 'mnist':
            train_transform = transforms.Compose([
                transforms.RandomCrop(28, padding=2),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize((0.1307), (0.3081))
            ])
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307), (0.3081))
            ])
        elif args.dataset == 'cifar10':
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                #transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ])
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ])
        elif args.dataset == 'cifar100':
            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                #transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023)),
            ])
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4866, 0.4409), (0.2009, 0.1984, 0.2023)),
            ])
        # The tiny imagenet data is resized to 32 as CIFAR. Originally t-imagenet is size 64
        elif args.dataset == 'timgnet':
            train_transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ])
            test_transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                    ])
        return train_transform, test_transform
    elif 'vitadapter' in args.model or 'deit' in args.model:
        args.logger.print('Using augmentations of ViT')
        model_type = 'deit_small_patch16_224' # model_type can be anything as long as it's ViT or Deit
        model_ = timm.create_model(model_type, pretrained=False, num_classes=1).cuda()

        # from networks.my_vit import deit_small_patch16_224 as transformer
        # from timm.data import resolve_data_config
        # from timm.data.transforms_factory import create_transform
        # args.net = transformer(pretrained=True, num_classes=args.total_cls, latent=args.adapter_latent).to(device)
        config = resolve_data_config({}, model=model_)
        TRANSFORM = create_transform(**config)
        return TRANSFORM, TRANSFORM
    else:
        args.logger.print('Using augmentations of CLIP')
        _, TRANSFORM = clip.load('ViT-B/32', 'cpu')
        return TRANSFORM, TRANSFORM

def generate_random_cl(args):
    if args.dataset == 'mnist':
        fine_label = ['zero', 'one', 'two', 'three', 'four',
               'five', 'six', 'seven', 'eight', 'nine']
    elif args.dataset == 'cifar100':
        from datasets.label_names import fine_label as fine_label
    elif args.dataset == 'cifar10':
        from datasets.label_names import cifar10_labels as fine_label
    elif args.dataset == 'imagenet':
        fine_label = [str(i) for i in range(args.total_cls)] # IS IT NECESSARY TO BE STR?
    elif args.dataset == 'timgnet':
        fine_label = [str(i) for i in range(args.total_cls)] # IS IT NECESSARY TO BE STR?
    else:
        raise NotImplementedError("dataset not implemented")
    n_cls = args.total_cls // args.n_tasks # number of classes per task

    seq = np.arange(args.total_cls)
    if args.seed != 0:
        np.random.shuffle(seq)

    seq = seq.reshape(args.n_tasks, n_cls)

    task_list = []
    for t in seq:
        names_list, sub_cls_list = [], []
        for c in t:
            name = fine_label[c]
            names_list.append(name)
            sub_cls_list.append(c)
        task_list.append([names_list, sub_cls_list])
    return task_list

class StandardCL:
    def __init__(self, dataset, args, task_list):
        self.dataset = dataset
        self.args = args
        self.seen_names = []
        self.task_id = 0
        self.task_list = task_list

        self.validation = args.validation

        if args.dataset in ['imagenet', 'timgnet']:
            assert not args.clip_init
            self.dataset.targets = np.array(self.dataset.targets)
            self.dataset.full_labels = np.concatenate((self.dataset.targets.reshape(-1, 1),
                                                        self.dataset.targets.reshape(-1, 1)), 1)
            self.dataset.names = self.dataset.targets.tolist()
            self.dataset.targets_relabeled = self.dataset.targets.copy()

    def make_dataset(self, task_id):
        dataset_ = deepcopy(self.dataset)

        if self.validation is not None:
            dataset_valid = deepcopy(self.dataset)

        targets_aux, targets_aux_valid = [], []
        data_aux, data_aux_valid = [], []
        full_targets_aux, full_targets_aux_valid = [], []
        names_aux, names_aux_valid = [], []
        cls_names =  self.task_list[task_id][0]
        cls_ids = self.task_list[task_id][1]
        idx_list, idx_list_valid = [], [] # These are used for ImageNet
        for i, (name, c) in enumerate(zip(cls_names, cls_ids)):
            if name not in self.seen_names:
                self.seen_names.append(name)
            idx = np.where(self.dataset.targets == c)[0]#[:40]

            if self.validation is not None:
                if self.args.seed != 0:
                    np.random.shuffle(idx)
                n_samples = len(idx)
                idx_valid = idx[int(n_samples * self.validation):]
                idx = idx[:int(n_samples * self.validation)]

            idx_list.append(idx)
            if self.validation is not None: idx_list_valid.append(idx_valid)

            if self.args.dataset in ['cifar100', 'cifar10', 'mnist']:
                data_aux.append(self.dataset.data[idx])
                targets_aux.append(np.zeros(len(idx), dtype=np.int) + self.seen_names.index(name))
                full_targets_aux.append([[self.seen_names.index(name),
                                          self.seen_names.index(name)] for _ in range(len(idx))])
                names_aux.append([name for _ in range(len(idx))])

                if self.validation is not None:
                    data_aux_valid.append(self.dataset.data[idx_valid])
                    targets_aux_valid.append(np.zeros(len(idx_valid), dtype=np.int) + self.seen_names.index(name))
                    full_targets_aux_valid.append([[self.seen_names.index(name),
                                              self.seen_names.index(name)] for _ in range(len(idx_valid))])
                    names_aux_valid.append([name for _ in range(len(idx_valid))])

            elif self.args.dataset in ['imagenet', 'timgnet']:
                for i in idx:
                    self.dataset.names[i] = name
                    self.dataset.targets_relabeled[i] = self.seen_names.index(name)
                    self.dataset.full_labels[i] = np.zeros(2) + self.dataset.targets_relabeled[i]
                if self.validation is not None:
                    for i in idx_valid:
                        self.dataset.names[i] = name
                        self.dataset.targets_relabeled[i] = self.seen_names.index(name)
                        self.dataset.full_labels[i] = np.zeros(2) + self.dataset.targets_relabeled[i]
            else:
                raise NotImplementedError()

        if self.args.dataset in ['cifar100', 'cifar10', 'mnist']:
            dataset_.data = np.array(list(chain(*data_aux)))
            dataset_.targets = np.array(list(chain(*targets_aux)))
            dataset_.full_labels = np.array(list(chain(*full_targets_aux)))
            dataset_.names = list(chain(*names_aux))
            del data_aux, targets_aux, full_targets_aux, names_aux

            if self.validation is not None:
                dataset_valid.data = np.array(list(chain(*data_aux_valid)))
                dataset_valid.targets = np.array(list(chain(*targets_aux_valid)))
                dataset_valid.full_labels = np.array(list(chain(*full_targets_aux_valid)))
                dataset_valid.names = list(chain(*names_aux_valid))
                del data_aux_valid, targets_aux_valid, full_targets_aux_valid, names_aux_valid

        elif self.args.dataset in ['imagenet', 'timgnet']:
            idx_list = np.concatenate(idx_list)
            dataset_ = Subset(self.dataset, idx_list)
            # Subset has no attribute targets, data, names, full_labels. (These are called in other functions)
            # Create them
            dataset_.data = []
            dataset_.names = []
            for i in idx_list:
                # dataset_.data.append(dataset_.dataset.samples[i][0])
                dataset_.names.append(dataset_.dataset.names[i])
            # dataset_.data = [dataset_.dataset.data[i] for i in idx_list]
            dataset_.targets = dataset_.dataset.targets[idx_list]
            # dataset_.names = [dataset_.dataset.targets[i] for i in idx_list]
            # dataset_.full_labels = dataset_.dataset.full_labels[idx_list]
            dataset_.transform = dataset_.dataset.transform

            if self.validation is not None:
                idx_list_valid = np.concatenate(idx_list_valid)
                dataset_valid = Subset(self.dataset, idx_list_valid)
                dataset_valid.targets = dataset_valid.dataset.targets[idx_list_valid]
                dataset_valid.transform = dataset_valid.dataset.transform
        else:
            raise NotImplementedError()

        task_id += 1

        if self.validation is None:
            return dataset_
        else:
            self.args.logger.print(f"******* Validation {self.validation} used *******")
            return dataset_, dataset_valid

def get_data(args):
    train_transform, test_transform = augmentations(args)
    args.root = './'

    if args.dataset == 'mnist':
        train = MNIST(root=args.root, train=True, download=True, transform=train_transform)
        test  = MNIST(root=args.root, train=False, download=True, transform=test_transform)
    elif args.dataset == 'cifar100':
        train = CIFAR100(root=args.root, train=True, download=True, transform=train_transform)
        test  = CIFAR100(root=args.root, train=False, download=True, transform=test_transform)
    elif args.dataset == 'cifar10':
        train = CIFAR10(root=args.root, train=True, download=True, transform=train_transform)
        test  = CIFAR10(root=args.root, train=False, download=True, transform=test_transform)
    elif args.dataset == 'imagenet':
        train = ImageFolder(root=args.root + '/ImageNet/train', transform=train_transform)
        test = ImageFolder(root=args.root + '/ImageNet/val', transform=test_transform)
    elif args.dataset == 'timgnet':
        train = ImageFolder(root=args.root + '/TinyImagenet/train', transform=train_transform)
        test = ImageFolder(root=args.root + '/TinyImagenet/val_folders', transform=test_transform)
    if args.validation and ('cifar' in args.dataset or 'mnist' in args.dataset):
        pass
    elif args.validation and args.dataset == 'imagenet':
        if args.dataset == 'imagenet':
            test = ImageFolder(root=args.root + '/ImageNet/train', transform=train_transform)
        elif args.dataset == 't-imagenet':
            test = ImageFolder(root=args.root + '/TinyImagenet/train', transform=test_transform)

    train = ClassSplit(args).relabel(train)
    test = ClassSplit(args).relabel(test)
    return train, test


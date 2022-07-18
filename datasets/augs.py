import torch
import clip
import torchvision.transforms as transforms

def augmentations(args):
    if args.model == 'moco_feature':
        TRANSFORM = transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5071,  0.4866,  0.4409),
                                                 (0.2009,  0.1984,  0.2023))
                            ])
        return TRANSFORM, TRANSFORM
    elif 'resnet' in args.model:
        if args.dataset == 'mnist':
            train_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            test_transform = transforms.ToTensor()
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
    elif 'vitadapter' in args.model or 'deitadpater' in args.model:
        from networks.my_vit import deit_small_patch16_224 as transformer
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform
        args.net = transformer(pretrained=True, num_classes=args.total_cls, latent=args.adapter_latent).to(device)
        config = resolve_data_config({}, model=args.net)
        TRANSFORM = create_transform(**config)
        return TRANSFORM, TRANSFORM
    else:
        _, TRANSFORM = clip.load('ViT-B/32', 'cpu')
        return TRANSFORM, TRANSFORM
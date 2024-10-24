import os
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, CelebA


# add in download argument to be compatible with torchvision datasets
class TinyImageNet(Dataset):
    def __init__(self, root, train=True, transform=None, download=None):
        if not root.endswith("tiny-imagenet-200"):
            root = os.path.join(root, "tiny-imagenet-200")
        self.train_dir = os.path.join(root, "train")
        self.val_dir = os.path.join(root, "val")
        self.transform = transform
        if train:
            self._scan_train()
        else:
            self._scan_val()

    def _scan_train(self):
        classes = [d.name for d in os.scandir(self.train_dir) if d.is_dir()]
        classes = sorted(classes)
        assert len(classes) == 200

        self.data = []
        for idx, name in enumerate(classes):
            this_dir = os.path.join(self.train_dir, name)
            for root, _, files in sorted(os.walk(this_dir)):
                for fname in sorted(files):
                    if fname.endswith(".JPEG"):
                        path = os.path.join(root, fname)
                        item = (path, idx)
                        self.data.append(item)
        self.labels_dict = {i: classes[i] for i in range(len(classes))}

    def _scan_val(self):
        self.file_to_class = {}
        classes = set()
        with open(os.path.join(self.val_dir, "val_annotations.txt"), 'r') as f:
            lines = f.readlines()
        for line in lines:
            words = line.split("\t")
            self.file_to_class[words[0]] = words[1]
            classes.add(words[1])
        classes = sorted(list(classes))
        assert len(classes) == 200

        class_to_idx = {classes[i]: i for i in range(len(classes))}
        self.data = []
        this_dir = os.path.join(self.val_dir, "images")
        for root, _, files in sorted(os.walk(this_dir)):
            for fname in sorted(files):
                if fname.endswith(".JPEG"):
                    path = os.path.join(root, fname)
                    idx = class_to_idx[self.file_to_class[fname]]
                    item = (path, idx)
                    self.data.append(item)
        self.labels_dict = {i: classes[i] for i in range(len(classes))}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx]
        image = Image.open(path)
        image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


class SourceTargetDataset(Dataset):
    def __init__(self, source, target):
        self.source = source
        self.target = target

    def __getitem__(self, index):
        image_1, label_1 = self.source.__getitem__(index)
        image_2, label_2 = self.target.__getitem__(index)
        # No need for assert below since they are constructed from the se dataset class constructor
        # without randomizing order, should return images under same order
        # assert label_1 == label_2
        return image_1, image_2

    def __len__(self):
        l1 = self.source.__len__()
        l2 = self.target.__len__()
        assert l1 == l2
        return l1


class AddGaussianNoise():
    def __init__(self, sigma=0.10):
        self.sigma = sigma

    def __call__(self, tensor):
        assert isinstance(tensor, torch.Tensor)
        dtype = tensor.dtype

        tensor = tensor.float()
        out = tensor + self.sigma * torch.randn_like(tensor)

        if out.dtype != dtype:
            out = out.to(dtype)
        return out


def get_dataset(name='cifar10', root='data'):
    if name == 'cifar10':
        data_norm = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        NUM_CLASSES = 10
        DATASET = CIFAR10
        RES = 32
        initial_resizing_transform = transforms.Compose([])
    elif name == 'cifar100':
        data_norm = transforms.Normalize([0.5071, 0.4865, 0.4409], [0.2009, 0.1984, 0.2023])
        NUM_CLASSES = 100
        DATASET = CIFAR100
        RES = 32
        initial_resizing_transform = transforms.Compose([])
    elif name == 'tiny':
        data_norm = transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
        NUM_CLASSES = 200
        DATASET = TinyImageNet
        RES = 64
        initial_resizing_transform = transforms.Compose([])
    elif name == 'celeba':
        # based on ImageNet stats, as default in PyTorch as well as DeepCluster
        data_norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        NUM_CLASSES = 2 # corresponding to binary attributes
        DATASET = CelebA
        RES = 64
        initial_resizing_transform = transforms.Compose([transforms.Resize(75),
                                                         transforms.CenterCrop(64)])
    else:
        raise NotImplementedError
    

    # for resnet encoder, at the training
    source_transform = transforms.Compose([
        initial_resizing_transform,
        transforms.RandomApply([
            transforms.RandomResizedCrop(RES),
            transforms.RandomHorizontalFlip(),
        ], p=0.95),
        transforms.RandomApply([
            transforms.RandAugment(),
        ], p=0.65),
        transforms.ToTensor(),
        data_norm,
        AddGaussianNoise(),
    ])
    # for unet decoder, at the training
    target_transform = transforms.Compose([
        initial_resizing_transform,
        transforms.RandomApply([
            transforms.RandomResizedCrop(RES),
            transforms.RandomHorizontalFlip(),
        ], p=0.95),
        transforms.RandomApply([
            transforms.RandAugment(),
        ], p=0.65),
        transforms.ToTensor(),
    ])
    # for resnet encoder, for evaluation
    downstream_transform_train = transforms.Compose([
        initial_resizing_transform,
        transforms.RandomApply([
            transforms.RandomResizedCrop(RES),
            transforms.RandomHorizontalFlip(),
        ], p=0.65),
        transforms.ToTensor(),
        data_norm,
    ])
    downstream_transform_test = transforms.Compose([
        initial_resizing_transform,
        transforms.ToTensor(),
        data_norm,
    ])

    if name=="celeba":
        train_source = DATASET(root=root, split='train', transform=source_transform, target_type='attr', download=True)
        train_target = DATASET(root=root, split='train', transform=target_transform, target_type='attr')
        train_source_target = SourceTargetDataset(train_source, train_target)

        down_train = DATASET(root=root, split='train', transform=downstream_transform_train, target_type='attr')
        down_test = DATASET(root=root, split='test', transform=downstream_transform_test, target_type='attr')
    else:
        train_source = DATASET(root=root, train=True, transform=source_transform, download=True)
        train_target = DATASET(root=root, train=True, transform=target_transform)
        train_source_target = SourceTargetDataset(train_source, train_target)

        down_train = DATASET(root=root, train=True, transform=downstream_transform_train)
        down_test = DATASET(root=root, train=False, transform=downstream_transform_test)
    return NUM_CLASSES, train_source_target, down_train, down_test

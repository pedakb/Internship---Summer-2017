import os

import torch
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets


class Dataset(object):
    def __init__(self, name, transform, info_dict):
        self.name = name
        self.size = info_dict['size'][self.name]
        self.classes = info_dict['classes'][self.name]
        self.mode = list(info_dict['mode'][self.name].keys())[0]
        self.transform = transform
        self.train_loader = None
        self.val_loader = None
        self.train_dataset = None
        self.val_dataset = None
        self.train_transform = transforms.Compose(
            [
                self.transform,
                transforms.ToTensor(),
                self.normalize()
            ]
        )
        self.val_transform = transforms.Compose(
            [
                self.transform,
                transforms.ToTensor(),
                self.normalize()
            ]
        )

    def normalize(self):
        if self.mode == 'RGB':
            return transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        elif self.mode == 'gray':
            return transforms.Normalize((0.5,), (1,))
        # else:
        #   return None

    def import_dataset(self):
        train_dir = os.path.join(self.name, 'train')
        val_dir = os.path.join(self.name, 'val')
        if self.name == 'MNIST':
            self.train_dataset = datasets.MNIST(root=self.name, train=True,
                                                transform=self.train_transform, download=True)
            self.val_dataset = datasets.MNIST(root=self.name, train=False,
                                              transform=self.val_transform)
        elif self.name == 'CIFAR-10':
            self.train_dataset = datasets.CIFAR10(root=self.name, train=True,
                                                  transform=self.train_transform, download=True)
            self.val_dataset = datasets.CIFAR10(root=self.name, train=False,
                                                transform=self.val_transform)
        elif self.name == 'ImageNet':
            self.train_dataset = datasets.ImageFolder(root=train_dir,
                                                      transform=self.train_transform)
            self.val_dataset = datasets.ImageFolder(root=val_dir,
                                                    transform=self.val_transform)
        else:
            print('Wrong Dataset!')

    def create_loader(self, batch_size, num_of_workers):
        # train_transform = transforms.Compose(
        #     [
        #         self.transform,
        #         transforms.ToTensor(),
        #         normalize(self.mode)
        #     ]
        # )
        # val_transform = transforms.Compose(
        #     [
        #         self.transform,
        #         transforms.ToTensor(),
        #         normalize(self.mode)
        #     ]
        # )
        # if self.name == 'MNIST':
        #     self.train_dataset = datasets.MNIST(root=self.name, train=True,
        #                                         transform=train_transform, download=True)
        #     self.val_dataset = datasets.MNIST(root=self.name, train=False,
        #                                       transform=val_transform)
        # elif self.name == 'CIFAR-10':
        #     self.train_dataset = datasets.CIFAR10(root=self.name, train=True,
        #                                           transform=train_transform, download=True)
        #     self.val_dataset = datasets.CIFAR10(root=self.name, train=False,
        #                                         transform=val_transform)
        # elif self.name == 'ImageNet':
        #     self.train_dataset = datasets.ImageFolder(root=train_dir,
        #                                               transform=train_transform)
        #     self.val_dataset = datasets.ImageFolder(root=val_dir,
        #                                             transform=val_transform)
        # else:
        #     print('Wrong Dataset!')
        self.import_dataset()
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        num_workers=num_of_workers)

        self.val_loader = torch.utils.data.DataLoader(self.val_dataset,
                                                      batch_size=batch_size,
                                                      shuffle=False,
                                                      num_workers=num_of_workers)



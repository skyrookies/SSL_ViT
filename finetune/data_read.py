#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/11 20:02
# @Author  : Bluejiang
# @File    : data_read.py
# @Description :
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import timm.data

image_size = (448, 448)


# data_mean = [0.5642201, 0.36060882, 0.35503095]
# data_std = [0.18658745, 0.14977561, 0.15853706]

# def get_class_labels(label):
#     """
#     将标签 0 和 1 合并为 0，将标签 2 和 3 合并为 1
#     """
#     if label == 0 or label == 1:
#         return 0
#     elif label == 2 or label == 3:
#         return 1
#     else:
#         raise ValueError("Unknown label: {}".format(label))
def get_dataloader(batch_size, train_data_folder, val_data_folder, size=(224, 224), num_worker=16):
    data_mean = [0.02514426, 0.02514426, 0.02514426]
    data_std = [0.15856943, 0.15856943, 0.15856943]
    train_transforms = timm.data.create_transform(
        input_size=size,
        is_training=True,
        mean=data_mean,
        std=data_std,
        auto_augment="rand-m7-mstd0.5-inc1"
    )
    data_transform = {
        "train": transforms.Compose([
            transforms.Resize(size),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # 将张量化为标准正态分布
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            # transforms.Normalize([0.5642201, 0.36060882, 0.35503095],
            #                      [0.18658745, 0.14977561, 0.15853706]),
            transforms.Normalize([0.02514426, 0.02514426, 0.02514426],
                                 [0.15856943, 0.15856943, 0.15856943])
        ]),

        "val": transforms.Compose([transforms.Resize(size),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.02514426, 0.02514426, 0.02514426],
                                                        [0.15856943, 0.15856943, 0.15856943])
                                   ])
    }
    dataset_train = torchvision.datasets.ImageFolder(train_data_folder, train_transforms)
    dataset_val = torchvision.datasets.ImageFolder(val_data_folder, data_transform['val'])
    # dataset_train.targets = [get_class_labels(label) for label in dataset_train.targets]
    # dataset_val.targets = [get_class_labels(label) for label in dataset_val.targets]
    train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_worker,
                                  pin_memory=True)
    val_dataloader = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=num_worker,
                                pin_memory=True)
    return train_dataloader, val_dataloader


if __name__ == '__main__':
    dataset_test = torchvision.datasets.ImageFolder('data_divided/train', data_transform['train'])
    test_dataloader = DataLoader(dataset_test, batch_size=2, shuffle=False)
    for i, (X, y) in enumerate(test_dataloader):
        # 将 Tensor 类型的数据转化为 numpy 数组
        img_np = X[0].numpy()
        img_np = img_np.swapaxes(0, 1)
        img_np = img_np.swapaxes(1, 2)
        # # 显示图片
        plt.imshow(img_np, vmin=0, vmax=1)
        print(X[0][2][223])
        # plt.imshow(img_np)
        plt.show()
        print(i, y)
        break
    # print(f'训练数据集长度为：{len(train_dataloader)}')
    # print(f'test数据集长度为：{len(val_dataloader)}')

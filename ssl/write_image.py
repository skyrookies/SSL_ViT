import torch
import torch.nn as nn

# loss_function = torch.nn.CrossEntropyLoss()

import os

def write_file_names(directory, output_file):
    file_names = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_names.append(file)
            print("{} has been loaded".format(file))
    with open(output_file, 'w') as f:
        f.write(' '.join(file_names))

# 指定目录和输出文件路径
directory = '/home/ljtj/PycharmProjects/data/endo_quarter'
output_file = './endo_quarter.txt'

# 调用函数将文件名写入txt文件
write_file_names(directory, output_file)

# if __name__ == '__main__':
#     data_path = '/home/ljtj/Documents/wsl/data_divided_2/train/cin2+'
#     list_path2D = '2D_images.txt'
#
#     train_set2D = Dataset2D(data_path, list_path2D)
#     data_loader2D = torch.utils.data.DataLoader(
#         train_set2D,
#         # sampler=torch.utils.data.DistributedSampler(train_set2D, shuffle=True),
#         pin_memory=True,
#         drop_last=True, )
#
#     for X in enumerate(data_loader2D):
#         print(X)
#         break
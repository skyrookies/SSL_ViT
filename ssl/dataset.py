import cv2
import os
from torch.utils.data import Dataset
from torchvision import transforms
import torch
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import matplotlib.pyplot as plt
class Dataset2(Dataset):
    def __init__(self, root, list_path, crop_size=(224, 224)):

        self.root = root
        self.list_path = list_path
        fp = open(self.list_path, 'r')
        self.img_ids = [i_id.strip().split() for i_id in fp]
        fp.close()

        self.files = []
        for item in self.img_ids[0]:
            # print(item)
            image_path = item
            name = image_path[:-4]
            img_file = image_path
            self.files.append({
                "img": img_file,
                "name": name
            })
        print('{} images are loaded!'.format(len(self.files)))

        self.crop_size = crop_size
        self.tr_transforms2D_global0 = get_train_transform2D_global0(self.crop_size)
        self.tr_transforms2D_global1 = get_train_transform2D_global1(self.crop_size)

        print("data hare loading")


    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        img_idx = self.files[index]
        img = []

        image = cv2.imread(os.path.join(self.root, img_idx["img"]))

        image = image[:, :, ::-1]
        img.append(self.tr_transforms2D_global0(image))
        img.append(self.tr_transforms2D_global1(image))

        return img


def get_train_transform2D_global0(crop_size):

    tr_transforms = transforms.Compose(
        [transforms.ToPILImage(),
         transforms.RandomResizedCrop(crop_size, scale=(0.14, 1), interpolation=3),
         transforms.RandomHorizontalFlip(),
         transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8),
         # transforms.RandomGrayscale(p=0.8),
         transforms.RandomApply([transforms.GaussianBlur(sigma=(0.1, 2.0), kernel_size=23)], p=0.4),
         transforms.ToTensor(),
         transforms.Normalize(IMAGENET_DEFAULT_MEAN,
                              IMAGENET_DEFAULT_STD)
         ])

    return tr_transforms

def get_train_transform2D_global1(crop_size):

    tr_transforms = transforms.Compose(
        [transforms.ToPILImage(),
         transforms.RandomResizedCrop(crop_size, scale=(0.14, 1), interpolation=3),
         transforms.RandomHorizontalFlip(),
         transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8),
         # transforms.RandomGrayscale(p=0.4),
         transforms.RandomApply([transforms.GaussianBlur(sigma=(0.1, 2.0), kernel_size=23)], p=1.0),
         transforms.ToTensor(),
         transforms.Normalize(IMAGENET_DEFAULT_MEAN,
                              IMAGENET_DEFAULT_STD)
         ])

    return tr_transforms

if __name__ == '__main__':
    dataset = Dataset2(root='/home/ljtj/Documents/wsl/data_divided_2/train/cin2+/',
                       list_path = 'images.txt', crop_size=(224, 224))

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    for i, X, in enumerate(data_loader):
        # 将 Tensor 类型的数据转化为 numpy 数组
        import matplotlib.pyplot as plt

        # 图片1
        img_np1 = X[0][0].numpy()
        img_np1 = img_np1.swapaxes(0, 1)
        img_np1 = img_np1.swapaxes(1, 2)
        # 显示图片1
        plt.subplot(1, 2, 1)  # 子图1，1 表示 1 行，2 列，第 1 张图
        plt.imshow(img_np1, vmin=0, vmax=1)
        plt.title("Teacher")  # 添加标题

        # 图片2
        img_np2 = X[1][0].numpy()
        img_np2 = img_np2.swapaxes(0, 1)
        img_np2 = img_np2.swapaxes(1, 2)
        # 显示图片2
        plt.subplot(1, 2, 2)  # 子图2，1 表示 1 行，2 列，第 2 张图
        plt.imshow(img_np2, vmin=0, vmax=1)
        plt.title("Student")  # 添加标题

        plt.show()  # 显示所有图片

        print(X[0].shape)
        # plt.imshow(img_np)
        plt.show()
        print(i)
        break
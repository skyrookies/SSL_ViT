import torch
import torch.nn as nn
import SSL_PVT as SPVTs
import torchvision.models as models
import vit_model as vit


class sslpvt2class(nn.Module):
    def __init__(self, init_path, num_features, trained_flag=False):
        super().__init__()
        self.pvt = SPVTs.pvt_tiny()
        weight_init = torch.load(init_path)
        if trained_flag:
            pvt_weight = weight_init
            self.pvt.load_state_dict(pvt_weight)
            print('has loaded from {}'.format(init_path))
        else:
            pvt_weight = weight_init["student"]
            pvt_weight = {k.replace('module.', ''): v for k, v in pvt_weight.items()}
            self.pvt.load_state_dict(pvt_weight)
            print('has loaded from {}'.format(init_path))

        self.mlp = nn.Linear(1000, num_features)
        self.ac = nn.ReLU()

    def forward(self, x):
        x = self.pvt(x)
        x = self.mlp(self.ac(x))
        return x


class sslvit(nn.Module):
    def __init__(self, init_path, num_features, trained_flag=False):
        super().__init__()
        self.vit = vit.vit_base_patch16_224()
        weight_init = torch.load(init_path)
        if trained_flag:
            vit_weight = weight_init
            self.vit.load_state_dict(vit_weight)
            print('has loaded from {}'.format(init_path))
        else:
            vit_weight = weight_init["student"]
            vit_weight = {k.replace('module.', ''): v for k, v in vit_weight.items()}
            self.vit.load_state_dict(vit_weight)
            print('has loaded from {}'.format(init_path))
            # print("No pretrained weight!!!!!!!")

        self.fc = nn.Linear(1000, num_features)
        self.ac = nn.ReLU()

    def forward(self, x):
        x = self.vit(x)
        x = self.fc(self.ac(x))
        return x


class sslvit_modified_768(nn.Module):
    def __init__(self, init_path, num_features, trained_flag=False):
        super().__init__()
        self.vit = vit.vit_base_patch16_224()
        weight_init = torch.load(init_path)
        if trained_flag:
            vit_weight = weight_init
            self.vit.load_state_dict(vit_weight)
            print('has loaded from {}'.format(init_path))
        else:
            vit_weight = weight_init["student"]
            vit_weight = {k.replace('module.', ''): v for k, v in vit_weight.items()}
            self.vit.load_state_dict(vit_weight)
            print('has loaded from {}'.format(init_path))

        self.fc = nn.Linear(1000, num_features)
        self.ac = nn.ReLU()

    def forward(self, x):
        x = self.vit(x)
        x = self.fc(self.ac(x))
        return x


class Resnet18_to_2(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super(Resnet18_to_2, self).__init__()
        self.resnet = models.resnet18(pretrained=pretrained, num_classes=1000)
        self.fc = nn.Linear(1000, num_classes)
        self.ac = nn.ReLU()

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(self.ac(x))

        return x


class Resnet18(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super(Resnet18, self).__init__()
        self.resnet = models.resnet18(pretrained=pretrained, num_classes=1000)
        self.resnet.fc = nn.Sequential(nn.Linear(512, num_classes), nn.LogSoftmax(dim=1))

    def forward(self, x):
        x = self.resnet(x)

        return x


class ShuffleNetV2_to_2(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super(ShuffleNetV2_to_2, self).__init__()
        self.shufflenet = models.shufflenet_v2_x1_0(pretrained=pretrained)
        self.fc = nn.Linear(1000, num_classes)
        # self.shufflenet = models.ShuffleNetV2([4, 8, 4], [24, 48, 96, 192, 1024], num_classes=num_classes)
        # self.resnet.fc = nn.Sequential(nn.Linear(512,num_classes),nn.LogSoftmax(dim=1))
        self.ac = nn.ReLU()

    def forward(self, x):
        x = self.shufflenet(x)
        x = self.fc(self.ac(x))

        return x


class MobileNetV2_to_2(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super(MobileNetV2_to_2, self).__init__()
        self.mobilenet = models.mobilenet_v2(pretrained=pretrained)
        self.fc = nn.Linear(1000, num_classes)
        # self.shufflenet = models.ShuffleNetV2([4, 8, 4], [24, 48, 96, 192, 1024], num_classes=num_classes)
        # self.resnet.fc = nn.Sequential(nn.Linear(512,num_classes),nn.LogSoftmax(dim=1))
        self.ac = nn.ReLU()

    def forward(self, x):
        x = self.mobilenet(x)
        x = self.fc(self.ac(x))

        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()

        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)


class ResNet50_CBAM(nn.Module):
    def __init__(self, num_classes=4):
        super(ResNet50_CBAM, self).__init__()
        self.resnet = models.resnet50()
        self.channel_att = ChannelAttention(2048)
        self.spatial_att = SpatialAttention()
        self.fc = nn.Linear(2048, num_classes)
        self.ac = nn.ReLU()

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.channel_att(x) * x
        x = self.spatial_att(x) * x

        x = self.resnet.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def create_model(model_type,
                 pretrained: bool = False,
                 checkpoint_path: str = '',
                 num_classes: int = 2,
                 **kwargs, ):
    if model_type == "ssl_pvt":
        model = sslpvt2class(init_path=checkpoint_path,
                             num_features=num_classes,
                             trained_flag=pretrained)
    elif model_type == 'resnet50_cbam':
        model = ResNet50_CBAM(num_classes=num_classes)
    elif model_type == "resnet18":
        model = Resnet18(num_classes=num_classes, pretrained=pretrained)
    elif model_type == "ShuffleNetV2":
        model = ShuffleNetV2_to_2(num_classes=num_classes, pretrained=pretrained)
        weight_init = torch.load(checkpoint_path)
        weight = weight_init["model"]
        model.load_state_dict(weight)
        print("loaded from{}".format(checkpoint_path))
    elif model_type == "Compare_MobileNetV2":
        model = MobileNetV2_to_2(num_classes=num_classes, pretrained=pretrained)
    elif model_type == 'vit':
        model = vit.vit_base_patch16_224(num_classes=num_classes)
        weight_init = torch.load(checkpoint_path)
        # for k, v in weight_init.items():
        #     print(k)
        weight_init.pop('head.weight')
        weight_init.pop('head.bias')
        model.load_state_dict(weight_init, strict=False)
        print('has loaded from {}'.format(checkpoint_path))
    # elif model_type == 'vit_ssl':
    #     model = vit.vit_base_patch16_224(num_classes=num_classes)
    #     weight_init = torch.load(checkpoint_path)
    #     # for k, v in weight_init.items():
    #     #     print(k)
    #     vit_ssl_weight = weight_init["student"]
    #     vit_ssl_weight = {k.replace('module.', ''): v for k, v in vit_ssl_weight.items()}
    #     vit_ssl_weight.pop('head.weight')
    #     vit_ssl_weight.pop('head.bias')
    #     model.load_state_dict(vit_ssl_weight, strict=False)
    #     print('has loaded from {}'.format(checkpoint_path))
    elif model_type == "ssl_vit":
        model = sslvit(init_path=checkpoint_path,
                       num_features=num_classes,
                       trained_flag=pretrained)
    elif model_type == "ssl_vit_modified_768":
        model = sslvit_modified_768(init_path=checkpoint_path,
                                    num_features=num_classes,
                                    trained_flag=pretrained)
    else:
        print("Unknow model type")

    return model


if __name__ == '__main__':
    model = sslpvt2class(init_path='models/student_attn_only_out_val_checkpoint0000.pth', num_features=2)  # 自定义分类数为2
    inputs = torch.randn(8, 3, 224, 224)  # 示例输入数据，假设为3通道的224x224图像
    outputs = model(inputs)  # 前向传播，得到模型输出
    print(outputs.shape)
    print(outputs)

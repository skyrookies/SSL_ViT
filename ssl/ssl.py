import torch
import torch.nn as nn
import SSL_PVT as SPVTs


class SslPvt2class(nn.Module):
    def __init__(self, init_path, num_features, no_flag=False, trained_flag=False):
        super().__init__()
        self.pvt = SPVTs.pvt_tiny()
        weight_init = torch.load(init_path, map_location=torch.device('cpu'))
        if trained_flag:
            pvt_weight = weight_init["student"]
            pvt_weight = {k.replace('module.', ''): v for k, v in pvt_weight.items()}
        else:
            pvt_weight = weight_init
        if not no_flag:
            self.pvt.load_state_dict(pvt_weight)
        self.fc = nn.Linear(1000, num_features)
        self.ac = nn.ReLU()

    def forward(self, x):
        x = self.pvt(x)
        x = self.fc(self.ac(x))

        return x


if __name__ == '__main__':
    model = SslPvt2class(init_path='/home/ljtj/pvt/checkpoint0180.pth', num_features=2)  # 自定义分类数为2
    inputs = torch.randn(1, 3, 224, 224)  # 示例输入数据，假设为3通道的224x224图像
    outputs = model(inputs)  # 前向传播，得到模型输出
    print(outputs)

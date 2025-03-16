import random

import torch.nn as nn

# 层类型
LAYER_TYPE = ['C', 'P', 'F']
# 卷积层输出频道数
OUT_CHANNELS = [8, 16, 32, 64, 128, 256]
# 卷积核大小
KERNEL_SIZE = [3, 5, 7]
# 池化核大小
POOL_SIZE = [1, 2, 3]
# 卷积时的步长(池化步长设置为其核大小)
STRIDE = [1, 2]
# 全连接输出特征数
OUT_FEATURES = [128, 256, 512, 1024]


# 生成随机CP层参数
def generate_random_cp_layer():
    layer_type = random.choices(['C', 'P'], weights=[0.75, 0.25])[0]

    if layer_type == 'C':  # 卷积层
        out_channels = random.choice(OUT_CHANNELS)
        kernel_size = random.choice(KERNEL_SIZE)
        stride = random.choice(STRIDE)
        padding = (kernel_size - 1) // 2  # 半填充
        return layer_type, (out_channels, kernel_size, stride, padding)

    elif layer_type == 'P':  # 池化层
        pool_size = random.choice(POOL_SIZE)
        stride = pool_size  # 简化类型
        return layer_type, (pool_size, stride)


# 生成随机F层层数
def generate_random_f_layer():
    out_features = random.choice(OUT_FEATURES)
    return 'F', out_features


# 定义动态CNN模型
class DynamicCnn(nn.Module):
    def __init__(self):
        super(DynamicCnn, self).__init__()
        self.CP_layers = nn.ModuleList()  # 使用ModuleList来存储动态添加的层
        self.F_layers = nn.ModuleList()
        self.in_channels = 3  # 初始输入为RGB图像
        self.image_size = 32  # 初始输入像素是32x32
        self.in_features = 0

    # 逐层添加，构建CNN模型
    def add_layer(self, layer_type, params):
        # 添加卷积层
        if layer_type == 'C':
            out_channels, kernel_size, stride, padding = params
            self.CP_layers.append(nn.Conv2d(self.in_channels, out_channels, kernel_size, stride, padding))
            self.CP_layers.append(nn.BatchNorm2d(out_channels))  # 批归一化层,加快收敛
            self.CP_layers.append(nn.ReLU())
            self.in_channels = out_channels
            # 更新特征图的尺寸
            self.image_size = max(1, (self.image_size + 2 * padding - kernel_size) // stride + 1)

            # print(f"add C layer, out_channels: {out_channels}, image_size: {self.image_size}")

        # 添加池化层
        elif layer_type == 'P':
            pool_size, stride = params
            self.CP_layers.append(nn.MaxPool2d(pool_size, stride))
            # 更新特征图尺寸
            self.image_size = max(1, (self.image_size - pool_size) // stride + 1)

            # print(f"add P layer, pool_size: {pool_size}, image_size: {self.image_size}")

        # 添加全连接层
        elif layer_type == 'F':
            # 计算全连接层的输入特征数
            # 当前面不是全连接层时,输入特征数是图像展平后的乘积
            if self.in_features == 0:
                in_features = self.in_channels * (self.image_size ** 2)
            # 当前面是全连接层,输入特征数是上一层的输出特征
            else:
                in_features = self.in_features

            # 通过参数决定输出特征数
            out_features = params
            self.F_layers.append(nn.Linear(in_features, out_features))
            self.F_layers.append(nn.ReLU())

            # 更新输入特征数
            self.in_features = out_features

            # print(f"add F layer, in_features: {in_features}, out_features: {out_features}")

    # 前向传播函数，途中在全连接层之前进行展平
    def forward(self, x):
        for layer in self.CP_layers:
            x = layer(x)

        # 展平成扁平向量
        x = x.view(x.size(0), -1)

        for layer in self.F_layers:
            x = layer(x)

        return x

    # 初始化权重函数
    def init_weights(self):
        # 初始化卷积层和全连接层的权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 使用He初始化卷积层权重
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                # 使用He初始化全连接层权重
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

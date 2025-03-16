import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import random_split


def main():
    # 定义数据预处理方式（将图像转换为张量，并标准化）
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将图像转换为Tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # 归一化
        transforms.RandomHorizontalFlip(),
    ])

    # 加载训练集和测试集
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)

    train_size = int(0.8 * len(trainset))  # 80%用于训练
    eva_size = len(trainset) - train_size  # 剩余20%用于验证

    # 使用random_split划分训练集和验证集
    trainset, evaset = random_split(trainset, [train_size, eva_size])

    # 使用DataLoader来加载数据
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    evaloader = torch.utils.data.DataLoader(evaset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=4)

    # 获取类别名称
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    from GA import GA
    ga = GA(trainloader, evaloader, 10)
    ga.evolve()

if __name__ == '__main__':
    main()

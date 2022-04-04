# Python 基础学习
# 蛋挞不能吃
# 学习时间： 2022/4/3 0003 16:03

import torch
import torch.nn.functional as F

import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

batch_size = 64

transform = transforms.Compose([
    transforms.ToTensor(),  # 将PIL图像转换为Tensor(H×W×C -> C×H×W， 0-255 -> 0-1)
    transforms.Normalize((0.1307,), (0.3081,))  # 数据标准化(均值, 方差)
])

# MNIST数据集：28×28手写数字图片识别，共10个分类
train_dataset = datasets.MNIST(root='./dataset/mnist/',
                               train=True,
                               download=True,
                               transform=transform)
train_loader = DataLoader(train_dataset,
                          shuffle=True,
                          batch_size=batch_size)
test_dataset = datasets.MNIST(root='./dataset/mnist/',
                              train=False,
                              download=True,
                              transform=transform)
test_loader = DataLoader(test_dataset,
                         shuffle=True,
                         batch_size=batch_size)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=(5, 5))
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=(5, 5))
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(320, 10)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = x.view(batch_size, -1)
        x = self.fc(x)
        return x


model = Net()

# 迁移到GPU训练
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 模型放到cuda中
model.to(device)

criterion = torch.nn.CrossEntropyLoss()  # 多分类交叉熵损失(包含softmax)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


# 训练和测试时输入输出数据也要放到cuda中
def train(epoch):
    """单次训练函数"""
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0


def test():
    correct = 0
    total = 0
    with torch.no_grad():  # 不需要计算梯度
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)  # max返回最大值及其下标
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('测试集准确率：%d %%' % (100 * correct / total))


if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()
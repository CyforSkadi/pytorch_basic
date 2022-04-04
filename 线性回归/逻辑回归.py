# Python 基础学习
# 蛋挞不能吃
# 学习时间： 2022/4/2 0002 17:48

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

# 数据集：MNIST手写识别数字
# 训练集：60000  测试集：10000  类别：10
# train_set = torchvision.datasets.MNIST(root='./dataset/mnist', train=True, download=True)
# test_set = torchvision.datasets.MNIST(root='./dataset/mnist', train=False, download=True)
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[0], [0], [1]])


class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


model = LogisticRegressionModel()

# 定义交叉熵损失
criterion = torch.nn.BCELoss(size_average=False)
# 优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 绘图
x = np.linspace(0, 10, 200)
x_t = torch.Tensor(x).view((200, 1))
y_t = model(x_t)
y = y_t.data.numpy()
plt.plot(x, y)
plt.grid()
plt.show()


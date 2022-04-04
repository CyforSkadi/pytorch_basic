# Python 基础学习
# 蛋挞不能吃
# 学习时间： 2022/4/2 0002 19:59
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# Dataset为抽象类不能实例化，需要继承它来定义我们自己的dataset
# DataLoader是帮助加载数据集和mini-batch的类

class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        # 使得数据集支持索引 dataset[index]
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        # 使得可以返回数据集长度
        return self.len


dataset = DiabetesDataset('./datasets/diabetes.csv.gz')
train_loader = DataLoader(dataset=dataset,
                          batch_size=32,
                          shuffle=True,
                          num_workers=2)  # 并行进程数


class Model(torch.nn.Module):
    """
    三层神经网络分类模型
    """
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


model = Model()

criterion = torch.nn.BCELoss(reduction='sum')  # 损失求和
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

if __name__ == '__main__':
    for epoch in range(100):
        for i, data in enumerate(train_loader, 0):  # 将dataloader放入data进行迭代，从0开始计数
            # 准备数据
            inputs, labels = data
            # 前馈
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            print(epoch, i, loss.item())
            # 反馈
            optimizer.zero_grad()
            loss.backward()
            # 更新参数
            optimizer.step()


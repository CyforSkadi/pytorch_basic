# Python 基础学习
# 蛋挞不能吃
# 学习时间： 2022/4/4 0004 19:58
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 参数定义
days_for_train = 5
N_EPOCHS = 250
N_LAYERS = 1
HIDDEN_SIZE = 8
BATCH_SIZE = 1
filename = './datasets/zgpa_train.csv'


def create_dataset(data, days_for_train=5) -> (np.array, np.array):
    """
        根据给定的序列data，生成数据集。
        数据集分为输入和输出，每一个输入的长度为days_for_train，每一个输出的长度为1。
        也就是说用days_for_train天的数据，对应下一天的数据。
        若给定序列的长度为d，将输出长度为(d-days_for_train)个输入/输出对
    """
    dataset_x, dataset_y = [], []
    for i in range(len(data) - days_for_train):
        _x = data[i:(i + days_for_train)]
        dataset_x.append(_x)
        dataset_y.append(data[i + days_for_train])
    return np.array(dataset_x), np.array(dataset_y)


def data_proccessing(filename):
    data = pd.read_csv(filename, usecols=[5])
    data = data.values.astype('double')
    # print(data.shape)

    # 归一化
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data = scaler.fit_transform(data)
    # print(data)

    # 划分数据集
    data_x, data_y = create_dataset(data, days_for_train)
    # print(data_x.shape)
    train_size = int(len(data_y) * 0.8)
    train_x, train_y = data_x[:train_size], data_y[:train_size]
    test_x, test_y = data_x[train_size:], data_y[train_size:]

    # 转换为LSTM读入的维度(seq,batch-size,features),并转换为tensor
    train_x = torch.from_numpy(train_x.reshape(-1, BATCH_SIZE, days_for_train)).to(torch.float32)
    train_y = torch.from_numpy(train_y.reshape(-1, BATCH_SIZE, 1)).to(torch.float32)
    # print(train_x)
    return train_x, train_y


class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(Model, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=input_size,
                                  hidden_size=hidden_size,
                                  num_layers=num_layers)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        seq_len, batch_size, hidden_size = x.shape
        x = x.view(seq_len * batch_size, hidden_size)
        x = self.fc(x)
        x = x.view(seq_len, batch_size, -1)
        return x


# 定义模型、损失函数和优化器
model = Model(days_for_train, HIDDEN_SIZE, 1, num_layers=N_LAYERS)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# 训练数据处理
train_x, train_y = data_proccessing(filename)

if __name__ == '__main__':
    # train-loop
    for epoch in range(N_EPOCHS):
        optimizer.zero_grad()
        output = model(train_x)
        loss = criterion(output, train_y)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 50 == 0:
            print(f'Epoch:{epoch + 1},loss:{loss}')

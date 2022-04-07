# Python 基础学习
# 蛋挞不能吃
# 学习时间： 2022/4/7 0007 15:17
import torch
import scipy.io as io

x = io.loadmat('./datasets/InputData_2DOF.mat')
y = io.loadmat('./datasets/OutputData_2DOF.mat')
train_x = x['data_x']
train_y = y['data_y']

x = torch.tensor(train_x)
print(x)
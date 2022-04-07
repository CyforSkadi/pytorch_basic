# Python 基础学习
# 蛋挞不能吃
# 学习时间： 2022/4/7 0007 15:15
import torch
import math
import matplotlib.pyplot as plt
import scipy.io as io

HIDDEN_SIZE = 300


# rnn takes 3d input while mlp only takes 2d input
class RecNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = torch.nn.RNN(input_size=1, hidden_size=HIDDEN_SIZE, num_layers=1, batch_first=True)
        # 至于这个线性层为什么是2维度接收，要看最后网络输出的维度是否匹配label的维度
        self.linear = torch.nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x):
        # print("x shape: {}".format(x.shape))
        # x [batch_size, seq_len, input_size]
        output, hn = self.rnn(x)
        # print("output shape: {}".format(output.shape))
        # out [seq_len, batch_size, hidden_size]
        x = output.reshape(-1, HIDDEN_SIZE)

        # print("after change shape: {}".format(x.shape))
        x = self.linear(x)

        # print("after linear shape: {}".format(x.shape))

        return x


def PlotCurve(rnn, input_x, x):
    # input_x 是输入网络的x。
    # sin_x 是列表，x的取值，一维数据、
    # 虽然他们的内容（不是维度）是一样的。可以print shape看一下。
    rnn_eval = rnn.eval()
    rnn_y = rnn_eval(input_x.unsqueeze(0))

    plt.figure(figsize=(6, 8))
    plt.plot([i + 1 for i in range(EPOCH)], rnn_loss, label='RNN')
    plt.title('loss')
    plt.legend()

    plt.figure(figsize=(6, 8))
    plt.plot(labels.detach().numpy(), label="original", linewidth=3)
    plt.plot([y[0] for y in rnn_y.detach().numpy()], label='RNN')
    plt.title('evaluation')
    plt.legend()

    plt.tight_layout()
    plt.show()


# 常量都取出来，以便改动
EPOCH = 400
RNN_LR = 0.01
PI = math.pi

if __name__ == '__main__':
    rnn = RecNN()

    # x,y 是普通sinx 的torch tensor
    x = io.loadmat('./datasets/InputData_2DOF.mat')
    y = io.loadmat('./datasets/OutputData_2DOF.mat')
    x = torch.tensor(x['data_x']).to(torch.float32)
    y = torch.tensor(y['data_y']).to(torch.float32)
    # input_x和labels是训练网络时候用的输入和标签。
    input_x = x.reshape(-1, 1)
    labels = y.reshape(-1, 1)

    # 训练rnn
    rnn_optimizer = torch.optim.Adam(rnn.parameters(), lr=RNN_LR)
    rnn_loss = []
    for epoch in range(EPOCH):
        preds = rnn(input_x.unsqueeze(0))
        # print(x.unsqueeze(0).shape)
        # print(preds.shape)
        # print(labels.shape)
        loss = torch.nn.functional.mse_loss(preds, labels)
        print(f"epoch:{epoch},loss = {loss.item()}")
        rnn_optimizer.zero_grad()
        loss.backward()
        rnn_optimizer.step()
        rnn_loss.append(loss.item())

    PlotCurve(rnn, input_x, x)

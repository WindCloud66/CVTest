# _*_ coding:utf-8 _*_
import torch
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.optim import Adam


BAT_SIZE = 128
TEST_BATCH_SIZE = 500
# 1准备数据集
def get_dataloader(train=True,batch_size =BAT_SIZE):
    transform_fn =Compose([
        ToTensor(),
        # mean和std的形状 和 通道数相同
        Normalize(mean=(0.1307,), std=(0.3081,))
    ])
    dataset = MNIST(root="./data", train=train, transform=transform_fn)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader

class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.fc1 = nn.Linear(1 * 28 * 28, 28)
        self.fc2 = nn.Linear(28, 10)

    def forward(self, input):
        # 1.修改形状
        x = input.view([input.size(0), 1 * 28 * 28])
        # 2.进行全连接的操作
        x = self.fc1(x)
        # 3.进行激活函数的处理,形状没有变化
        x = F.relu(x)
        # 4.输出层
        out = self.fc2(x)
        return F.log_softmax(out, dim=-1)

# 获取模型
model = MnistModel()
# 获取优化器
optimizer = Adam(model.parameters(), lr=0.001)
if os.path.exists("./model/model.pkl"):
    model.load_state_dict(torch.load("./model/model.pkl"))
    optimizer.load_state_dict(torch.load("./model/optimizer.pkl"))


def train(epoch):
    # 获取数据集
    data_loader = get_dataloader()

    for idx, (input, target) in enumerate(data_loader):
        # 梯度置为0
        optimizer.zero_grad()
        # 调用模型,得到预测值
        output = model(input)
        # 计算损失
        loss = F.nll_loss(output, target)
        # 方向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        if idx % 50 == 0:
            print(epoch, idx, loss.item())
        if idx % 100 == 0:
            torch.save(model.state_dict(), "./model/model.pkl")
            torch.save(optimizer.state_dict(), "./model/optimizer.pkl")


def test():
    loss_list = []
    acc_list = []
    test_dataloader = get_dataloader(train=False, batch_size=TEST_BATCH_SIZE)
    for idx,(input, target) in enumerate(test_dataloader):
        with torch.no_grad():
            output = model(input)
            cur_loss = F.nll_loss(output,target)
            loss_list.append(cur_loss)
            #计算准确率
            pred = output.max(dim=-1)[-1]
            cur_acc = pred.eq(target).float().mean()
            acc_list.append(cur_acc)
    print("平均准确率,平均损失", np.mean(acc_list), np.mean(loss_list))

if __name__ == '__main__':
    for i in range(5):
        train(i)
        test()
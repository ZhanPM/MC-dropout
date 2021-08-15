# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 20:57:46 2021

@author: Allen Van
"""


'''
2016,Bayesian Convolutional Neural Networks with Bernoulli Approximate Variational Inference
'''

import torch
from torch import nn, optim
#import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
#from logger import Logger



torch.cuda.empty_cache()# 释放显存    

# 定义超参数
batch_size_train = 64 # 训练批的大小
batch_size_test = 100 # 测试批的大小
# learning_rate = 1e-2    # 学习率
num_epoches = 30        # 遍历训练集的次数 10000000/(60000/64)≈10661

# 数据类型转换，转换成numpy类型
#def to_np(x):
#    return x.cpu().data.numpy()

# 自定义学习率衰减
def inverse_time_decay(parameters, learning_rate, iterations, momentum, vs, gama, power):
    learning_rate_new = learning_rate * (1 + gama * iterations) ** (-power)
    # print(learning_rate_new)# 输出学习率
    for param, momentum in zip(parameters, momentum):
        momentum[:] = vs * momentum + param.grad.data
        param.data = param.data - learning_rate_new * momentum



# 自定义L2正则化项
def L2_regu(parameters, loss, weight_decay):
    lambda1 = torch.tensor(weight_decay)
    l2_reg = torch.tensor(0.)
    for param in parameters:
        l2_reg = l2_reg + param.norm(2)
        loss_L2 = loss + lambda1 * l2_reg
    return loss_L2




# 下载训练集 MNIST 手写数字训练集
train_dataset = datasets.MNIST(
    root='./data', train=True, transform=transforms.ToTensor(), download=True)

test_dataset = datasets.MNIST(
    root='./data', train=False, transform=transforms.ToTensor())

train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)


n_class = 10# 定义类别数

# 定义 Convolution Network 模型
class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()#super用法:Cnn继承父类nn.Model的属性，并用父类的方法初始化这些属性
        self.conv1 = nn.Conv2d(1, 20, 5, stride=1)
        self.conv2 = nn.Conv2d(20, 50, 5, stride=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(800, 500)
        self.fc2 = nn.Linear(500, n_class)
        self.dropout = nn.Dropout(0.5)


    def forward(self, x):
        x = F.relu(self.conv1(x))# conv1(24x24x20) 
        x = self.dropout(x)
        x = self.pool(x)#maxpool(12*12*20)
        x = F.relu(self.conv2(x))# conv1(8x8x50)
        x = self.dropout(x)
        x = self.pool(x)#maxpool(4*4*50)
        x = x.view(-1, 4 * 4 *50)
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, 0.1, training = self.training)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


model = net()  # 图片大小是28x28,输入深度是1，最终输出的10类
print('net--------------------------')
print(model)



use_gpu = torch.cuda.is_available()  # 判断是否有GPU加速
if use_gpu:
    model = model.cuda()

# 定义loss和optimizer
criterion = nn.CrossEntropyLoss()
'''1. torch自带优化器'''
# optimizer = optim.SGD(model.parameters(), lr=0.01)
optimizer = optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-4)
'''2. 自定义优化器'''
momentum = []# 初始化动量项
for param in model.parameters():
    momentum.append(torch.zeros_like(param.data))

test_acc = np.array([[0, 0]])# 存储测试准确率

#logger = Logger('./logs')
# 开始训练
t = 1# 迭代步数
for epoch in range(num_epoches):
    # print('epoch {}'.format(epoch + 1))      # .format为输出格式，formet括号里的即为左边花括号的输出
    # print('*' * 10)
    running_loss = 0.0
    running_acc = 0.0
    for i, data in enumerate(train_loader, 1):
        img, label = data
        # cuda
        if use_gpu:
            img = img.cuda()
            label = label.cuda()
        img, label = Variable(img), Variable(label)
        # 向前传播
        out = model(img)
        loss = criterion(out, label)
        # loss = L2_regu(model.parameters(), loss, weight_decay=0.0005)# weight decay L2正则化
        # 向后传播
        model.zero_grad()
        loss.backward()
        # inverse_time_decay(model.parameters(), learning_rate=0.01, 
                            # iterations=t, momentum=momentum, vs=0.9, gama=1e-4, power=0.75)# inv learning rate
        optimizer.step()# 标准优化器需打开
        t += 1# 迭代次数
        
        
        if t % 10000 == 0: #每10000次测试一次
            model.eval()
            eval_loss = 0
            eval_acc = 0
            num_correct = 0
            for data, target in test_loader:
                # target = target.squeeze()  # 添加这一行，用于降维度
                # move tensors to GPU if CUDA is available
                if use_gpu:
                    data, target = data.cuda(), target.cuda()
                # forward pass: compute predicted outputs by passing inputs to the model
                data, target = Variable(data, volatile=True), Variable(target)
                output = model(data)
                loss = criterion(output, target)# calculate the batch loss
                eval_loss += loss.item()*data.size(0)#计算总的损失
                pred = output.data.max(1, keepdim=True)[1]#获得得分最高的类别
                num_correct += pred.eq(target.data.view_as(pred)).cpu().sum()# 正确结果总数

            # 计算平均损失
            eval_loss = eval_loss/len(test_loader.dataset)
            eval_acc = num_correct/len(test_loader.dataset)
            test_acc = np.vstack((test_acc, np.array([[t, eval_acc]])))
            # print(label)
            print('Iteration: {:.0f} Test Loss: {:.6f}, Acc: {:.6f}'.format(t, eval_loss, eval_acc))
            f = open('./result/TestStdDrop.log', 'a+')
            f.write('{}, {}, {} \n'.format(t, eval_loss, eval_acc))
            f.close()
            # checkpoint = model.state_dict()# 建立模型保存点并保存模型
            checkpoint = model# 建立模型保存点并保存模型
            path_checkpoint = "./checkpoint/checkpoint_{}_iteration.pt".format(t)
            torch.save(checkpoint, path_checkpoint)
         

torch.cuda.empty_cache()# 释放显存       

# 绘制准确率曲线
ax1 = plt.figure()
plt.plot(test_acc[:, 0], test_acc[:, 1],)
plt.legend(["Test accuracy"])
# plt.xscale('log')# 设置对数横坐标
plt.show()
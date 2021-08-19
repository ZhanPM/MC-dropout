
'''
2016,Bayesian Convolutional Neural Networks with Bernoulli Approximate Variational Inference
'''

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt
from Mode import net


''' 释放显存'''    
torch.cuda.empty_cache()


'''定义超参数'''
batch_size_train = 64 # 训练批的大小
batch_size_test = 100 # 测试批的大小
num_epoches = 330
# 遍历训练集的次数 10000000/(60000/64)≈10661


'''自定义学习率衰减'''
def inverse_time_decay(parameters, learning_rate, iterations, momentum, vs, gama, power):
    learning_rate_new = learning_rate * (1 + gama * iterations) ** (-power)
    # print(learning_rate_new)# 输出学习率，用于调试
    for param, momentum in zip(parameters, momentum):
        momentum[:] = vs * momentum + param.grad.data
        param.data = param.data - learning_rate_new * momentum


'''自定义L2正则化项'''
def L2_regu(parameters, loss, weight_decay):
    lambda1 = torch.tensor(weight_decay)
    l2_reg = torch.tensor(0.)
    for param in parameters:
        l2_reg = l2_reg + param.norm(2)
        loss_L2 = loss + lambda1 * l2_reg
    return loss_L2


'''下载MNIST数据集'''
train_dataset = datasets.MNIST(
    root='./data', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=False)
test_dataset = datasets.MNIST(
    root='./data', train=False, transform=transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)


'''加载模型'''
model = net()  # 图片大小是28x28,输入深度是1，最终输出的10类
print(model)


'''判断是否有GPU加速'''
use_gpu = torch.cuda.is_available()
if use_gpu:
    model = model.cuda()
    
    
'''定义loss和optimizer'''
criterion = nn.CrossEntropyLoss()
# 1. torch自带优化器
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.005)
# optimizer = optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-4)
# 2. 自定义优化器
momentum = []# 初始化动量项
for param in model.parameters():
    momentum.append(torch.zeros_like(param.data))


'''训练模型'''
test_acc = np.array([[0, 0]])# 存储测试集的准确率
t = 1# 迭代步数
for epoch in range(num_epoches):
    model.train(mode=True)
    for i, data in enumerate(train_loader, 1):
        img, label = data
        if use_gpu:# 调用cuda
            img, label = img.cuda(), label.cuda()
        img, label = Variable(img), Variable(label)
        # 向前传播
        out = model(img)
        loss = criterion(out, label)
        # loss = L2_regu(model.parameters(), loss, weight_decay=0.0005)# weight decay L2正则化
        # 向后传播
        model.zero_grad()
        loss.backward()
        # inverse_time_decay(model.parameters(), learning_rate=0.01,
        #                     iterations=t, momentum=momentum, vs=0.9, gama=1e-4, power=0.75)# inv learning rate
        optimizer.step()# 标准优化器需打开
        t += 1# 迭代次数
        # print(i)
        
        #每迭代10000次测试一次并保存一次模型
        if t % 10000 == 0:
            model.train(mode=False)
            eval_loss = 0
            eval_acc = 0
            num_correct = 0
            for data, target in test_loader:
                if use_gpu:# move tensors to GPU if CUDA is available
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)
                output = model(data)# forward pass: compute predicted outputs by passing inputs to the model
                loss = criterion(output, target)# calculate the batch loss
                eval_loss += loss.item()*data.size(0)#计算总损失
                pred = output.data.max(1, keepdim=True)[1]#获得得分最高的类别
                num_correct += pred.eq(target.data.view_as(pred)).cpu().sum()# 正确结果总数
            # print(num_correct.item())#打印识别正确数目，用于调试
            eval_loss = eval_loss/len(test_loader.dataset)# 计算平均损失
            eval_acc = num_correct/len(test_loader.dataset)# 计算平均准确度
            test_acc = np.vstack((test_acc, np.array([[t, eval_acc]])))
            print('Iteration: {:.0f} Test Loss: {:.6f}, Acc: {:.4f}'.format(t, eval_loss, eval_acc))
            f = open('./result/TestStdDrop.log', 'a+')
            f.write('{}, {}, {} \n'.format(t, eval_loss, eval_acc))
            f.close()
            # checkpoint = model.state_dict()# 建立模型保存点并保存模型参数
            checkpoint = model# 建立模型保存点并保存整个模型
            path_checkpoint = "./checkpoint/checkpoint_{}_iteration.pt".format(t)
            torch.save(checkpoint, path_checkpoint)


'''释放显存 '''
torch.cuda.empty_cache()      


# '''绘制准确率曲线'''
ax1 = plt.figure()
plt.plot(test_acc[1:, 0], test_acc[1:, 1],)
plt.legend(["Test accuracy"])
# plt.xscale('log')# 设置对数横坐标
plt.show()
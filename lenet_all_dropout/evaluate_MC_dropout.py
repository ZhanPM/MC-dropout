'''
2016,Bayesian Convolutional Neural Networks with Bernoulli Approximate Variational Inference
'''

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import numpy as np
import matplotlib.pyplot as plt
from Mode import net



'''定义超参数'''
batch_size_train = 64 # 训练批的大小
batch_size_test = 100 # 测试批的大小


'''下载MNIST数据集的测试集'''
test_dataset = datasets.MNIST(
    root='./data', train=False, transform=transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)


'''判断是否有GPU加速'''
use_gpu = torch.cuda.is_available()  # 判断是否有GPU加速


'''Bayes test--MC dropout'''
N = 30#检查点个数
T = 20#Monte Carlo前向传播次数
test_acc_MC = np.array([[0, 0, 0]])# 存储测试准确率Iteration-Mean-Variance
test_acc_temp = np.zeros([T, N])# 保存各迭代次数对应的多次准确率，用于调试
for t in range(N):
    iterations = 10000 * (t + 1)
    path_checkpoint = "./checkpoint/checkpoint_{}_iteration.pt".format(iterations)
    model = torch.load(path_checkpoint)
    # model.load_state_dict(torch.load(path_checkpoint))
    if use_gpu:
        model = model.cuda()
    model.train(mode=True)#此时测试时dropout打开
    # model.train()# 只有train情况下才能开启dropout--pytorch
    test_acc = 0
    num_correct = 0
    with  torch.no_grad():#关闭自动求导引擎，能节省显存和加速。
        for j in range(T):
            for data, target in test_loader:
                # 调用CUDA
                if use_gpu:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)
                output = model(data)# forward pass: compute predicted outputs by passing inputs to the model 
                pred = output.data.max(1, keepdim=True)[1]#获得得分最高的类别
                num_correct += pred.eq(target.data.view_as(pred)).cpu().sum()# 正确结果总数
            eval_acc = num_correct/len(test_loader.dataset) # 计算平均准确度
            test_acc_temp[j, t] = np.array([eval_acc])
            # print('num_correct: {:.0f}'.format(num_correct.item()))#打印识别正确数目，用于调试
            num_correct = 0
        test_acc_mean = np.mean(test_acc_temp[:, t])# 求均值
        test_acc_std = np.std(test_acc_temp[:,t])# 求标准差
        test_acc_MC = np.vstack((test_acc_MC, np.array([iterations, test_acc_mean, test_acc_std])))
        print('Iteration: {:.0f} Test Acc Mean: {:.6f} Std: {:.4f}'.format(iterations, test_acc_mean, test_acc_std))
        f = open('./result/TestMCDrop.log', 'a+')
        f.write('{}, {}, {} \n'.format(t, test_acc_mean, test_acc_std))
        f.close()
    

'''释放显存 '''    
torch.cuda.empty_cache()      


# '''绘制准确率曲线'''
ax1 = plt.figure()
plt.plot(test_acc_MC[1:-1, 0], test_acc_MC[1:-1, 1])
plt.legend(["Test accuracy"])
plt.xscale('log')# 设置对数横坐标
plt.show()
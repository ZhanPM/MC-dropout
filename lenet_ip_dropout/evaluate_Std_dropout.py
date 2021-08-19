'''
2016,Bayesian Convolutional Neural Networks with Bernoulli Approximate Variational Inference
'''

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import numpy as np
from Mode import net


'''定义超参数'''
batch_size_train = 64        # 训练批的大小
batch_size_test = 100 # 测试批的大小

'''下载MNIST数据集'''
test_dataset = datasets.MNIST(
    root='./data', train=False, transform=transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=batch_size_test, shuffle=False)


    
'''定义loss'''
criterion = nn.CrossEntropyLoss()


'''判断是否有GPU加速'''
use_gpu = torch.cuda.is_available()  # 判断是否有GPU加速


'''Standard test'''
test_acc = np.array([[0, 0]])# 存储测试准确率
N = 3#检查点个数
for t in range(N):
    iterations = 10000 * (t + 1)
    path_checkpoint = "./checkpoint/checkpoint_{}_iteration.pt".format(iterations)
    # model.load_state_dict(torch.load(path_checkpoint))
    model = torch.load(path_checkpoint)
    model.eval()
    if use_gpu:
        model = model.cuda()
    eval_loss = 0
    eval_acc = 0
    num_correct = 0
    for data, target in test_loader:
        if use_gpu:# move tensors to GPU if CUDA is available
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = model(data)# forward pass: compute predicted outputs by passing inputs to the model
        loss = criterion(output, target)# calculate the batch loss
        eval_loss += loss.item()*data.size(0)# update average validation loss
        pred = output.data.max(1, keepdim=True)[1]#获得得分最高的类别
        num_correct += pred.eq(target.data.view_as(pred)).cpu().sum()# 正确结果总数
    # 计算平均损失和准确率
    # print('num_correct: {:.0f}'.format(num_correct.item()))#打印识别正确数目，用于调试
    eval_loss = eval_loss/len(test_loader.dataset)
    eval_acc = num_correct/len(test_loader.dataset)
    test_acc = np.vstack((test_acc, np.array([[t, eval_acc]])))
    print('Iteration: {:.0f} Test Loss: {:.6f}, Acc: {:.4f}'.format(iterations, eval_loss, eval_acc))
    f = open('./result/TestStdDrop1.log', 'a+')
    f.write('{}, {}, {} \n'.format(t, eval_loss, eval_acc))
    f.close()

    
'''释放显存'''
torch.cuda.empty_cache()       


# '''绘制准确率曲线'''
# ax1 = plt.figure()
# plt.plot(test_acc[1:-1, 0], test_acc[1:-1, 1])
# plt.legend(["Test accuracy"])
# # plt.xscale('log')# 设置对数横坐标
# plt.show()
from torch import nn
import torch.nn.functional as F
from drop_user import dropout_user


'''定义Lenet-all-dropout模型'''
class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()#super用法:Cnn继承父类nn.Model的属性，并用父类的方法初始化这些属性
        self.conv1 = nn.Conv2d(1, 20, 5, stride=1, bias=True)
        self.conv2 = nn.Conv2d(20, 50, 5, stride=1, bias=True)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(800, 500)
        self.fc2 = nn.Linear(500, 10)#10类标签
        self.dropout = nn.Dropout(0.5)


    def forward(self, x):
        x = self.conv1(x)# conv1(24x24x20) 
        # x = x * 0.5
        # x = F.dropout(x, 0.5, training=self.training)
        x = self.dropout(x)
        # x = x * 0.5
        # x = dropout_user(x, 0.5)
        x = self.relu(x)
        x = self.pool(x)#maxpool(12*12*20)
        x = self.conv2(x)# conv1(8x8x50)
        # x = x * 0.5
        # # x = F.dropout(x, 0.5, training=self.training)
        x = self.dropout(x)
        # x = x * 0.5
        # x = dropout_user(x, 0.5)
        x = self.relu(x)
        x = self.pool(x)#maxpool(4*4*50)
        x = x.view(-1, 4 * 4 *50)
        x = self.fc1(x)#fully connected(800->500)
        # x = x * 0.5
        # x = F.dropout(x, 0.5, training=self.training)
        x = self.dropout(x)
        # x = x * 0.5
        # x = dropout_user(x, 0.5)
        x = self.relu(x)
        x = self.fc2(x)#fully connected(500->10)
        return x
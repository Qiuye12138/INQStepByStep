import sys
import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import torchvision as tv
import torch.optim as optim
from torch.optim.optimizer import Optimizer, required



class SGD(Optimizer):

    def __init__(self, params, lr=required, momentum=0):

        defaults = dict(lr=lr, momentum=momentum, dampening=0, weight_decay=0, nesterov=False, weight_bits=None)

        super(SGD, self).__init__(params, defaults)

        for group in self.param_groups:
            group['Ts'] = []
            for p in group['params']:
                if p.requires_grad is False:
                    group['Ts'].append(0)
                    continue

                T = torch.ones_like(p.data)
                group['Ts'].append(T)


    def step(self):

        for group in self.param_groups:
            momentum = group['momentum']

            for idx, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p)
                        d_p = buf

                zeros    = torch.zeros_like(p.data)
                ones     = torch.ones_like(p.data)
                quantile = np.quantile(torch.abs(p.data.cpu()).numpy(), float(sys.argv[2]))
                T = torch.where(torch.abs(p.data) >= quantile, zeros, ones)

                d_p.mul_(group['Ts'][idx])
                d_p.mul_(T)
                p.data.add_(-group['lr']*d_p)



#-------------------------------------#
#       参数
#-------------------------------------#
SMAPLE_NUM = 10        #样本集大小
BATCH_SIZE = 50
TOP        = 0
EPOCH      = 0



#-------------------------------------#
#        构造数据集
#-------------------------------------#
transform = tv.transforms.Compose([tv.transforms.Resize(256),
                                   tv.transforms.CenterCrop(224),
                                   tv.transforms.ToTensor(),
                                   tv.transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                           std  = [0.229, 0.224, 0.225])])

train_dataset = tv.datasets.CIFAR100(root = './', train = True , transform = transform, download = True)
test_dataset  = tv.datasets.CIFAR100(root = './', train = False, transform = transform, download = True)
SAMPLELOADER  = torch.utils.data.DataLoader(test_dataset , batch_size = SMAPLE_NUM, shuffle = False, num_workers = 0)
trainloader   = torch.utils.data.DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True , num_workers = 0)
testloader    = torch.utils.data.DataLoader(test_dataset , batch_size = BATCH_SIZE, shuffle = False, num_workers = 0)



#-------------------------------------#
#       量化函数
#-------------------------------------#
Q_max_8    =  2**7  - 1
Q_min_8    = -2**7
Q_max_16   =  2**15 - 1
Q_min_16   = -2**15


#量化为8位定点数
def quant_8(X, X_s):

    Q = torch.round(X / X_s)
    Q = nn.Hardtanh(Q_min_8, Q_max_8)(Q)

    return Q * X_s


#量化为16位定点数
def quant_16(X, X_s):

    Q = torch.round(X / X_s)
    Q = nn.Hardtanh(Q_min_16, Q_max_16)(Q)

    return Q * X_s


#部分量化_8位
def sub_quant_8(X, X_s):

    Z = torch.zeros_like(X)
    O = torch.ones_like(X)
    I = np.quantile(torch.abs(X).numpy(), float(sys.argv[2]))
    A = torch.where(torch.abs(X) <= I, Z, O)
    Q = quant_8(X, X_s)
    E = X - Q

    return X - A*E


#部分量化_16位
def sub_quant_16(X, X_s):

    Z = torch.zeros_like(X)
    O = torch.ones_like(X)
    I = np.quantile(torch.abs(X).numpy(), float(sys.argv[2]))
    A = torch.where(torch.abs(X) <= I, Z, O)
    Q = quant_16(X, X_s)
    E = X - Q

    return X - A*E



#-------------------------------------#
#       构造网络
#-------------------------------------#
class FirstConv2d(nn.Module):

    def __init__(self, inplanes, planes, kernel_size, stride, padding):

        super(FirstConv2d, self).__init__()

        self.conv = nn.Conv2d(inplanes, planes, kernel_size, stride = stride, padding = padding)


    def forward(self, x):

        x = self.conv(x)

        if QUANT_WT == True:
            C_s = next(C_S_bag)
            W_s = next(W_S_bag)
            self.conv.weight = nn.Parameter(sub_quant_8 (self.conv.weight, W_s))
            self.conv.bias   = nn.Parameter(sub_quant_16(self.conv.bias  , C_s))

        return x


class BasicConv2d(nn.Module):

    def __init__(self, inplanes, planes, kernel_size, stride, padding):

        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(inplanes, planes, kernel_size, stride = stride, padding = padding)


    def forward(self, x):

        x = self.conv(x)

        if QUANT_WT == True:
            C_s = next(C_S_bag)
            W_s = next(W_S_bag)

            self.conv.weight = nn.Parameter(sub_quant_8 (self.conv.weight, W_s))
            self.conv.bias   = nn.Parameter(sub_quant_16(self.conv.bias  , C_s))

        return x


class ResBlock(nn.Module):

    def __init__(self, inplanes, planes):

        super(ResBlock, self).__init__()

        self.conv1 = BasicConv2d(inplanes, planes, 3, 2, 1);  self.downsample = BasicConv2d(inplanes, planes, 1, 2, 0)
        self.relu  = nn.ReLU()
        self.conv2 = BasicConv2d(planes, planes, 3, 1, 1)


    def forward(self, x):

        out = self.conv1(x)                                ;  identity = self.downsample(x)
        out = self.relu(out)
        out = self.conv2(out)
        
        bn_x, bn_y = next(bns)

        x = (out      * bn_x.to(out.device))        #还原
        y = (identity * bn_y.to(identity.device))   #还原

        out = x + y

        out = self.relu(out)

        return out


class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes):

        super(BasicBlock, self).__init__()

        self.conv1 = BasicConv2d(inplanes, planes, 3, 1, 1)
        self.relu  = nn.ReLU()
        self.conv2 = BasicConv2d(planes, planes, 3, 1, 1)


    def forward(self, x):

        out = self.conv1(x)                                ;  identity = x
        out = self.relu(out)
        out = self.conv2(out)
        
        bn_x, bn_y = next(bns)
        
        x = (out      * bn_x.to(out.device))        #还原
        y = (identity * bn_y.to(identity.device))   #还原

        out = x + y

        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self):

        super(ResNet, self).__init__()

        self.conv1   = FirstConv2d(3, 64, 7, 2, 3)
        self.relu    = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        self.layer1 = nn.Sequential(BasicBlock(64, 64),
                                    BasicBlock(64, 64))

        self.layer2 = nn.Sequential(ResBlock(64, 128),
                                    BasicBlock(128, 128))

        self.layer3 = nn.Sequential(ResBlock(128, 256),
                                    BasicBlock(256, 256))

        self.layer4 = nn.Sequential(ResBlock(256, 512),
                                    BasicBlock(512, 512))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 100)


    def forward(self, x):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        if QUANT_WT == True:
            C_s = next(C_S_bag)
            W_s = next(W_S_bag)
            self.fc.weight = nn.Parameter(sub_quant_8 (self.fc.weight, W_s))
            self.fc.bias   = nn.Parameter(sub_quant_16(self.fc.bias  , C_s))

        return x


model = ResNet()
weights = torch.load(sys.argv[1])
model.load_state_dict(weights)



#-------------------------------------#
#       量化bn
#-------------------------------------#
bn_bag = torch.load('./bn_bag.pth')
bns    = iter(bn_bag)
bn_bag_quant = []
while(True):
    try:
        bn_x, bn_y = next(bns)
        B_s = 2 * torch.max(torch.abs(torch.cat((bn_x, bn_y)))) / (Q_max_8 - Q_min_8)
        B_s = 2 ** np.ceil(np.log2(B_s))

        bn_x = quant_8(bn_x, B_s)
        bn_y = quant_8(bn_y, B_s)

        bn_bag_quant.append((bn_x, bn_y))
    except StopIteration:
        break



#-------------------------------------#
#       量化权重
#-------------------------------------#
QUANT_WT = True
bns     = iter(bn_bag_quant)
C_S_bag = iter(torch.load('C_S_bag.pth'))
W_S_bag = iter(torch.load('W_S_bag.pth'))
for img, label in testloader:
    with torch.no_grad():
        model(img)
        break


model.cuda()
optimizer = SGD(model.parameters(), lr = 0.01, momentum = 0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 100, gamma = 0.1)
criterion = nn.CrossEntropyLoss()



#-------------------------------------#
#       训练
#-------------------------------------#
QUANT_WT = False
for epoch in range(0, 300):

    model.train()

    for img, label in tqdm(trainloader):
        bns   = iter(bn_bag_quant)
        img   = img.cuda()
        label = label.cuda()

        optimizer.zero_grad()

        P    = model(img)
        loss = criterion(P, label)

        loss.backward()
        optimizer.step()

    scheduler.step()

    model.eval()
    
    top1  = 0
    top5  = 0
    k     = 0

    for img, label in testloader:
        bns   = iter(bn_bag_quant)
        img   = img.cuda()
        label = label.cuda()

        _, P  = model(img).topk(5, dim = 1)
        label = label.unsqueeze(1)
        top1 += (label[:, 0] == P[:, 0]).sum().item()
        top5 += (label == P).sum().item()
        k    += BATCH_SIZE

        print('\r%d :Top1: %f%%, Top5: %f%%' %(k, 100 * top1 / k, 100 * top5 / k), end = '')

    if (top1 / k) >= TOP:
        TOP = top1 / k
        EPOCH = epoch
        torch.save(model.state_dict(), sys.argv[2] + '_best.pth')

    print('\n current best model is epoch: ' + str(EPOCH) + ', top1: ' + str(100 * TOP) + '%')

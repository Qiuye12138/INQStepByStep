import sys
import torch
import numpy as np
import torch.nn as nn
import torchvision as tv



#-------------------------------------#
#       参数
#-------------------------------------#
SMAPLE_NUM = 10        # 样本集大小
BATCH_SIZE = 50



#-------------------------------------#
#        构造数据集
#-------------------------------------#
transform = tv.transforms.Compose([tv.transforms.Resize(256),
                                   tv.transforms.CenterCrop(224),
                                   tv.transforms.ToTensor(),
                                   tv.transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                           std  = [0.229, 0.224, 0.225])])


test_dataset = tv.datasets.CIFAR100(root = './', train = False, transform = transform, download = True)
SAMPLELOADER = torch.utils.data.DataLoader(test_dataset , batch_size = SMAPLE_NUM, shuffle = False, num_workers = 0)
testloader   = torch.utils.data.DataLoader(test_dataset , batch_size = BATCH_SIZE, shuffle = False, num_workers = 0)



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



#-------------------------------------#
#       构造网络
#-------------------------------------#
class FirstConv2d(nn.Module):

    def __init__(self, inplanes, planes, kernel_size, stride, padding):

        super(FirstConv2d, self).__init__()

        self.conv = nn.Conv2d(inplanes, planes, kernel_size, stride = stride, padding = padding)

        self.X_s = 0
        self.C_s = 0
        self.W_s = 0


    def forward(self, x):

        x = self.conv(x)

        if SAVE_FM == True:
            c = x - self.conv.bias.reshape(1, -1, 1, 1)

            self.X_s = 1
            self.C_s = 2 * torch.max(torch.abs(c.flatten())).item()                / (Q_max_16 - Q_min_16)
            self.W_s = 2 * torch.max(torch.abs(self.conv.weight.flatten())).item() / (Q_max_8 - Q_min_8)

            self.X_s = 2 ** np.ceil(np.log2(self.X_s))
            self.C_s = 2 ** np.ceil(np.log2(self.C_s))
            self.W_s = 2 ** np.ceil(np.log2(self.W_s))

        if QUANT_WT == True:
            self.conv.weight = nn.Parameter(quant_8 (self.conv.weight, self.W_s))
            self.conv.bias   = nn.Parameter(quant_16(self.conv.bias  , self.C_s))

        if QUANT == True:
            x -= self.conv.bias.reshape(1, -1, 1, 1)
            x  = quant_16(x, self.C_s)
            x += self.conv.bias.reshape(1, -1, 1, 1)
            x  = quant_8(x, self.X_s)

        return x


class BasicConv2d(nn.Module):

    def __init__(self, inplanes, planes, kernel_size, stride, padding):

        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(inplanes, planes, kernel_size, stride = stride, padding = padding)

        self.X_s = 0
        self.C_s = 0
        self.W_s = 0


    def forward(self, x):

        x = self.conv(x)

        if SAVE_FM == True:
            c = x - self.conv.bias.reshape(1, -1, 1, 1)

            self.X_s = 1
            self.C_s = 2 * torch.max(torch.abs(c.flatten())).item()                / (Q_max_16 - Q_min_16)
            self.W_s = 2 * torch.max(torch.abs(self.conv.weight.flatten())).item() / (Q_max_8 - Q_min_8)

            self.X_s = 2 ** np.ceil(np.log2(self.X_s))
            self.C_s = 2 ** np.ceil(np.log2(self.C_s))
            self.W_s = 2 ** np.ceil(np.log2(self.W_s))

        if QUANT_WT == True:
            self.conv.weight = nn.Parameter(quant_8 (self.conv.weight, self.W_s))
            self.conv.bias   = nn.Parameter(quant_16(self.conv.bias  , self.C_s))

        if QUANT == True:
            x -= self.conv.bias.reshape(1, -1, 1, 1)
            x  = quant_16(x, self.C_s)
            x += self.conv.bias.reshape(1, -1, 1, 1)
            x  = quant_8(x, self.X_s)

        return x


class ResBlock(nn.Module):

    def __init__(self, inplanes, planes):

        super(ResBlock, self).__init__()

        self.conv1 = BasicConv2d(inplanes, planes, 3, 2, 1);  self.downsample = BasicConv2d(inplanes, planes, 1, 2, 0)
        self.relu  = nn.ReLU()
        self.conv2 = BasicConv2d(planes, planes, 3, 1, 1)

        self.A_s   = 0


    def forward(self, x):

        out = self.conv1(x)                                ;  identity = self.downsample(x)
        out = self.relu(out)
        out = self.conv2(out)
        
        bn_x, bn_y = next(bns)
        
        out      *= bn_x.to(out.device)        #还原
        identity *= bn_y.to(identity.device)   #还原

        out += identity

        if SAVE_FM == True:
            self.A_s = 1
            self.A_s = 2 ** np.ceil(np.log2(self.A_s))

        if QUANT == True:
            out = quant_8(out, self.A_s)

        out = self.relu(out)

        return out


class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes):

        super(BasicBlock, self).__init__()

        self.conv1 = BasicConv2d(inplanes, planes, 3, 1, 1)
        self.relu  = nn.ReLU()
        self.conv2 = BasicConv2d(planes, planes, 3, 1, 1)

        self.A_s   = 0


    def forward(self, x):

        out = self.conv1(x)                                ;  identity = x
        out = self.relu(out)
        out = self.conv2(out)
        
        bn_x, bn_y = next(bns)
        
        out      *= bn_x.to(out.device)        #还原
        identity *= bn_y.to(identity.device)   #还原

        out += identity

        if SAVE_FM == True:
            self.A_s = 1
            self.A_s = 2 ** np.ceil(np.log2(self.A_s))

        if QUANT == True:
            out = quant_8(out, self.A_s)

        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self):

        super(ResNet, self).__init__()

        self.conv1 = FirstConv2d(3, 64, 7, 2, 3)
        self.relu = nn.ReLU()
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

        self.X_s       = 0
        self.C_s       = 0
        self.W_s       = 0


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

        if SAVE_FM == True:
            c = x - self.fc.bias.reshape(1, -1)

            self.X_s       = 2 * torch.max(torch.abs(x.flatten())).item()              / (Q_max_8 - Q_min_8)
            self.C_s       = 2 * torch.max(torch.abs(c.flatten())).item()              / (Q_max_16 - Q_min_16)
            self.W_s       = 2 * torch.max(torch.abs(self.fc.weight.flatten())).item() / (Q_max_8 - Q_min_8)

            self.X_s       = 2 ** np.ceil(np.log2(self.X_s))
            self.C_s       = 2 ** np.ceil(np.log2(self.C_s))
            self.W_s       = 2 ** np.ceil(np.log2(self.W_s))

        if QUANT_WT == True:
            self.fc.weight = nn.Parameter(quant_8(self.fc.weight, self.W_s))
            self.fc.bias   = nn.Parameter(quant_16(self.fc.bias, self.C_s))

        if QUANT == True:
            x -= self.fc.bias.reshape(1, -1)
            x  = quant_16(x, self.C_s)
            x += self.fc.bias.reshape(1, -1)
            x  = quant_8(x, self.X_s)

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
#       保存特征图
#-------------------------------------#
bns        = iter(bn_bag_quant)
SAVE_FM    = True
QUANT_WT   = False
QUANT      = False
for img, label in SAMPLELOADER:
    model(img)
    break



#-------------------------------------#
#       量化权重
#-------------------------------------#
SAVE_FM  = False
QUANT_WT = True
QUANT    = False
bns      = iter(bn_bag_quant)
for img, label in testloader:
    with torch.no_grad():
        model(img)
        break



#-------------------------------------#
#       量化推理
#-------------------------------------#

SAVE_FM   = False
QUANT_WT  = False
QUANT     = True
model.cuda()
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

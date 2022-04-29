#优化后
import torch
import torch.nn as nn
import torchvision as tv



#-------------------------------------#
#       参数
#-------------------------------------#
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

testloader   = torch.utils.data.DataLoader(test_dataset , batch_size = BATCH_SIZE, shuffle = False, num_workers = 0)



#-------------------------------------#
#       构造网络
#-------------------------------------#
class FirstConv2d(nn.Module):

    def __init__(self, inplanes, planes, kernel_size, stride, padding):

        super(FirstConv2d, self).__init__()

        self.conv = nn.Conv2d(inplanes, planes, kernel_size, stride = stride, padding = padding)


    def forward(self, x):

        x = self.conv(x)

        return x


class BasicConv2d(nn.Module):

    def __init__(self, inplanes, planes, kernel_size, stride, padding):

        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(inplanes, planes, kernel_size, stride = stride, padding = padding)


    def forward(self, x):

        x = self.conv(x)

        return x


class ResBlock(nn.Module):

    def __init__(self, inplanes, planes):

        super(ResBlock, self).__init__()

        self.conv1      = BasicConv2d(inplanes, planes, 3, 2, 1);  self.downsample = BasicConv2d(inplanes, planes, 1, 2, 0)
        self.relu       = nn.ReLU()
        self.conv2      = BasicConv2d(planes, planes, 3, 1, 1)


    def forward(self, x):

        out = self.conv1(x)                                     ;  identity = self.downsample(x)
        out = self.relu(out)
        out = self.conv2(out)

        out += identity
        out = self.relu(out)

        return out


class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes):

        super(BasicBlock, self).__init__()

        self.conv1 = BasicConv2d(inplanes, planes, 3, 1, 1)
        self.relu  = nn.ReLU()
        self.conv2 = BasicConv2d(planes, planes, 3, 1, 1)


    def forward(self, x):

        out = self.conv1(x)                                     ;  identity = x
        out = self.relu(out)
        out = self.conv2(out)

        out += identity
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

        return x


model = ResNet()
weights = torch.load('./optimize.pth')
model.load_state_dict(weights)
model.cuda()
model.eval()



#-------------------------------------#
#       推理
#-------------------------------------#
top1  = 0
top5  = 0
k     = 0

for img, label in testloader:

    img   = img.cuda()
    label = label.cuda()

    _, P  = model(img).topk(5, dim = 1)
    label = label.unsqueeze(1)
    top1 += (label[:, 0] == P[:, 0]).sum().item()
    top5 += (label == P).sum().item()
    k    += BATCH_SIZE

    print('\r%d :Top1: %f%%, Top5: %f%%' %(k, 100 * top1 / k, 100 * top5 / k), end = '')

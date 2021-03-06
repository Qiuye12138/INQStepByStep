#优化
import torch
import torch.nn as nn
import torchvision as tv



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
weights = torch.load('./fuse.pth')

weights['conv1.conv.weight'] = weights.pop('conv1.weight')
weights['conv1.conv.bias'] = weights.pop('conv1.bias')

weights['layer1.0.conv1.conv.weight'] = weights.pop('layer1.0.conv1.weight')
weights['layer1.0.conv1.conv.bias'] = weights.pop('layer1.0.conv1.bias')

weights['layer1.0.conv2.conv.weight'] = weights.pop('layer1.0.conv2.weight')
weights['layer1.0.conv2.conv.bias'] = weights.pop('layer1.0.conv2.bias')

weights['layer1.1.conv1.conv.weight'] = weights.pop('layer1.1.conv1.weight')
weights['layer1.1.conv1.conv.bias'] = weights.pop('layer1.1.conv1.bias')

weights['layer1.1.conv2.conv.weight'] = weights.pop('layer1.1.conv2.weight')
weights['layer1.1.conv2.conv.bias'] = weights.pop('layer1.1.conv2.bias')

weights['layer2.0.conv1.conv.weight'] = weights.pop('layer2.0.conv1.weight')
weights['layer2.0.conv1.conv.bias'] = weights.pop('layer2.0.conv1.bias')

weights['layer2.0.conv2.conv.weight'] = weights.pop('layer2.0.conv2.weight')
weights['layer2.0.conv2.conv.bias'] = weights.pop('layer2.0.conv2.bias')

weights['layer2.1.conv1.conv.weight'] = weights.pop('layer2.1.conv1.weight')
weights['layer2.1.conv1.conv.bias'] = weights.pop('layer2.1.conv1.bias')

weights['layer2.1.conv2.conv.weight'] = weights.pop('layer2.1.conv2.weight')
weights['layer2.1.conv2.conv.bias'] = weights.pop('layer2.1.conv2.bias')

weights['layer3.0.conv1.conv.weight'] = weights.pop('layer3.0.conv1.weight')
weights['layer3.0.conv1.conv.bias'] = weights.pop('layer3.0.conv1.bias')

weights['layer3.0.conv2.conv.weight'] = weights.pop('layer3.0.conv2.weight')
weights['layer3.0.conv2.conv.bias'] = weights.pop('layer3.0.conv2.bias')

weights['layer3.1.conv1.conv.weight'] = weights.pop('layer3.1.conv1.weight')
weights['layer3.1.conv1.conv.bias'] = weights.pop('layer3.1.conv1.bias')

weights['layer3.1.conv2.conv.weight'] = weights.pop('layer3.1.conv2.weight')
weights['layer3.1.conv2.conv.bias'] = weights.pop('layer3.1.conv2.bias')

weights['layer4.0.conv1.conv.weight'] = weights.pop('layer4.0.conv1.weight')
weights['layer4.0.conv1.conv.bias'] = weights.pop('layer4.0.conv1.bias')

weights['layer4.0.conv2.conv.weight'] = weights.pop('layer4.0.conv2.weight')
weights['layer4.0.conv2.conv.bias'] = weights.pop('layer4.0.conv2.bias')

weights['layer4.1.conv1.conv.weight'] = weights.pop('layer4.1.conv1.weight')
weights['layer4.1.conv1.conv.bias'] = weights.pop('layer4.1.conv1.bias')

weights['layer4.1.conv2.conv.weight'] = weights.pop('layer4.1.conv2.weight')
weights['layer4.1.conv2.conv.bias'] = weights.pop('layer4.1.conv2.bias')

weights['layer2.0.downsample.conv.weight'] = weights.pop('layer2.0.downsample.weight')
weights['layer2.0.downsample.conv.bias'] = weights.pop('layer2.0.downsample.bias')

weights['layer3.0.downsample.conv.weight'] = weights.pop('layer3.0.downsample.weight')
weights['layer3.0.downsample.conv.bias'] = weights.pop('layer3.0.downsample.bias')

weights['layer4.0.downsample.conv.weight'] = weights.pop('layer4.0.downsample.weight')
weights['layer4.0.downsample.conv.bias'] = weights.pop('layer4.0.downsample.bias')

model.load_state_dict(weights)

torch.save(model.state_dict(), 'optimize.pth')

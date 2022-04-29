import torch
import torch.nn as nn
import torchvision as tv



#-------------------------------------#
#       参数
#-------------------------------------#
range_b      = 128.0
SMAPLE_NUM   = 10        #样本集大小



#-------------------------------------#
#        构造数据集
#-------------------------------------#
transform = tv.transforms.Compose([tv.transforms.Resize(256),
                                   tv.transforms.CenterCrop(224),
                                   tv.transforms.ToTensor(),
                                   tv.transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                           std  = [0.229, 0.224, 0.225])])


test_dataset = tv.datasets.CIFAR100(root = './', train = False, transform = transform, download = True)

testloader   = torch.utils.data.DataLoader(test_dataset , batch_size = SMAPLE_NUM, shuffle = False, num_workers = 0)



#-------------------------------------#
#       构造网络
#-------------------------------------#
class FirstConv2d(nn.Module):

    def __init__(self, inplanes, planes, kernel_size, stride, padding):

        super(FirstConv2d, self).__init__()

        self.conv = nn.Conv2d(inplanes, planes, kernel_size, stride = stride, padding = padding)

        self.sat_point = 0
        self.bn        = 0


    def forward(self, x):

        x = self.conv(x)

        self.sat_point = torch.max(torch.abs(x.flatten())).item()
        self.bn        = self.sat_point / range_b
        my_bn          = torch.ones(x.size(1)) * self.bn
        
        self.conv.weight /= my_bn.reshape(-1, 1, 1, 1)
        self.conv.bias   /= my_bn

        return (x, my_bn)


class BasicConv2d(nn.Module):

    def __init__(self, inplanes, planes, kernel_size, stride, padding):

        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(inplanes, planes, kernel_size, stride = stride, padding = padding)

        self.sat_point = 0
        self.bn        = 0


    def forward(self, TUPLE):

        x, bn = TUPLE

        x = self.conv(x)

        self.sat_point = torch.max(torch.abs(x.flatten())).item()
        self.bn        = self.sat_point / range_b
        my_bn          = torch.ones(x.size(1)) * self.bn

        self.conv.weight /= my_bn.reshape(-1, 1, 1, 1)
        self.conv.bias   /= my_bn
        self.conv.weight *= bn.reshape(1, -1, 1, 1)

        return (x, my_bn)


class ResBlock(nn.Module):

    def __init__(self, inplanes, planes):

        super(ResBlock, self).__init__()

        self.conv1      = BasicConv2d(inplanes, planes, 3, 2, 1);  self.downsample = BasicConv2d(inplanes, planes, 1, 2, 0)
        self.relu       = nn.ReLU()
        self.conv2      = BasicConv2d(planes, planes, 3, 1, 1)
        
        self.sat_point = 0
        self.bn        = 0


    def forward(self, TUPLE):

        x, bn = TUPLE

        (out, bn_x) = self.conv1((x, bn))                        ;  (identity, bn_y) = self.downsample((x, bn))
        out = self.relu(out)
        (out, bn_x) = self.conv2((out, bn_x))

        out += identity

        self.sat_point = torch.max(torch.abs(out.flatten())).item()
        self.bn        = self.sat_point / range_b
        my_bn          = torch.ones(out.size(1)) * self.bn

        bn_x /= my_bn
        bn_y /= my_bn
        
        bn_bag.append((bn_x.reshape(1, -1, 1, 1), bn_y.reshape(1, -1, 1, 1)))
 
        out = self.relu(out)

        return (out, my_bn)


class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes):

        super(BasicBlock, self).__init__()

        self.conv1 = BasicConv2d(inplanes, planes, 3, 1, 1)
        self.relu  = nn.ReLU()
        self.conv2 = BasicConv2d(planes, planes, 3, 1, 1)

        self.sat_point = 0
        self.bn        = 0


    def forward(self, TUPLE):

        x, bn = TUPLE

        (out, bn_x) = self.conv1((x, bn))                            ;  identity = x
        out = self.relu(out)
        (out, bn_x) = self.conv2((out, bn_x))

        out += identity

        self.sat_point = torch.max(torch.abs(out.flatten())).item()
        self.bn        = self.sat_point / range_b
        my_bn          = torch.ones(out.size(1)) * self.bn

        bn_x /= my_bn
        bn   /= my_bn
        
        bn_bag.append((bn_x.reshape(1, -1, 1, 1), bn.reshape(1, -1, 1, 1)))

        out = self.relu(out)

        return (out, my_bn)


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

        (x, bn) = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        (x, bn) = self.layer1((x, bn))
        (x, bn) = self.layer2((x, bn))
        (x, bn) = self.layer3((x, bn))
        (x, bn) = self.layer4((x, bn))

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        self.fc.weight *= bn.reshape(1, -1)

        return x

bn_bag = []
model = ResNet()
weights = torch.load('./optimize.pth')
model.load_state_dict(weights)
model.eval()



#-------------------------------------#
#       推理
#-------------------------------------#
for img, label in testloader:
    with torch.no_grad():
        model(img)
        break

torch.save(bn_bag,"bn_bag.pth")
torch.save(model.state_dict(),'weights_max.pth')

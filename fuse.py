#归一化融合
import torch
import torch.nn as nn
import torchvision as tv



#-------------------------------------#
#       参数
#-------------------------------------#
BATCH_SIZE   = 50



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
class ResBlock(nn.Module):

    def __init__(self, inplanes, planes):

        super(ResBlock, self).__init__()

        self.downsample = nn.Conv2d(inplanes, planes, kernel_size = 1, stride = 2)#, bias = False))#,
                                        #nn.BatchNorm2d(planes))

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size = 3, stride = 2, padding = 1)#, bias = False)
        #self.bn1   = nn.BatchNorm2d(planes)
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size = 3, stride = 1, padding = 1)#, bias = False)
        #self.bn2   = nn.BatchNorm2d(planes)


    def forward(self, x):

        out = self.conv1(x);         identity = self.downsample(x)
        #out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        #out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out



class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes):

        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size = 3, stride = 1, padding = 1)#, bias = False)
        #self.bn1   = nn.BatchNorm2d(planes)
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size = 3, stride = 1, padding = 1)#, bias = False)
        #self.bn2   = nn.BatchNorm2d(planes)


    def forward(self, x):

        out = self.conv1(x);         identity = x
        #out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        #out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self):

        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3)#, bias = False)
        #self.bn1  = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU()
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
        #x = self.bn1(x)
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
weights_orign = torch.load('./best.pth')



#-------------------------------------#
#       融合权重
#-------------------------------------#
weights = {}

weights['conv1.weight'] =  weights_orign['conv1.weight'] * weights_orign['bn1.weight'].reshape([-1,1,1,1]) / (weights_orign['bn1.running_var'].reshape([-1,1,1,1]) + 1e-05)**0.5
weights['conv1.bias']   = -weights_orign['bn1.running_mean'] * weights_orign['bn1.weight'] / (weights_orign['bn1.running_var'] + 1e-05)**0.5 + weights_orign['bn1.bias']

weights['layer1.0.conv1.weight'] =  weights_orign['layer1.0.conv1.weight'] * weights_orign['layer1.0.bn1.weight'].reshape([-1,1,1,1]) / (weights_orign['layer1.0.bn1.running_var'].reshape([-1,1,1,1]) + 1e-05)**0.5
weights['layer1.0.conv1.bias']   = -weights_orign['layer1.0.bn1.running_mean'] * weights_orign['layer1.0.bn1.weight'] / (weights_orign['layer1.0.bn1.running_var'] + 1e-05)**0.5 + weights_orign['layer1.0.bn1.bias']

weights['layer1.0.conv2.weight'] =  weights_orign['layer1.0.conv2.weight'] * weights_orign['layer1.0.bn2.weight'].reshape([-1,1,1,1]) / (weights_orign['layer1.0.bn2.running_var'].reshape([-1,1,1,1]) + 1e-05)**0.5
weights['layer1.0.conv2.bias']   = -weights_orign['layer1.0.bn2.running_mean'] * weights_orign['layer1.0.bn2.weight'] / (weights_orign['layer1.0.bn2.running_var'] + 1e-05)**0.5 + weights_orign['layer1.0.bn2.bias']

weights['layer1.1.conv1.weight'] =  weights_orign['layer1.1.conv1.weight'] * weights_orign['layer1.1.bn1.weight'].reshape([-1,1,1,1]) / (weights_orign['layer1.1.bn1.running_var'].reshape([-1,1,1,1]) + 1e-05)**0.5
weights['layer1.1.conv1.bias']   = -weights_orign['layer1.1.bn1.running_mean'] * weights_orign['layer1.1.bn1.weight'] / (weights_orign['layer1.1.bn1.running_var'] + 1e-05)**0.5 + weights_orign['layer1.1.bn1.bias']

weights['layer1.1.conv2.weight'] =  weights_orign['layer1.1.conv2.weight'] * weights_orign['layer1.1.bn2.weight'].reshape([-1,1,1,1]) / (weights_orign['layer1.1.bn2.running_var'].reshape([-1,1,1,1]) + 1e-05)**0.5
weights['layer1.1.conv2.bias']   = -weights_orign['layer1.1.bn2.running_mean'] * weights_orign['layer1.1.bn2.weight'] / (weights_orign['layer1.1.bn2.running_var'] + 1e-05)**0.5 + weights_orign['layer1.1.bn2.bias']

weights['layer2.0.conv1.weight'] =  weights_orign['layer2.0.conv1.weight'] * weights_orign['layer2.0.bn1.weight'].reshape([-1,1,1,1]) / (weights_orign['layer2.0.bn1.running_var'].reshape([-1,1,1,1]) + 1e-05)**0.5
weights['layer2.0.conv1.bias']   = -weights_orign['layer2.0.bn1.running_mean'] * weights_orign['layer2.0.bn1.weight'] / (weights_orign['layer2.0.bn1.running_var'] + 1e-05)**0.5 + weights_orign['layer2.0.bn1.bias']

weights['layer2.0.conv2.weight'] =  weights_orign['layer2.0.conv2.weight'] * weights_orign['layer2.0.bn2.weight'].reshape([-1,1,1,1]) / (weights_orign['layer2.0.bn2.running_var'].reshape([-1,1,1,1]) + 1e-05)**0.5
weights['layer2.0.conv2.bias']   = -weights_orign['layer2.0.bn2.running_mean'] * weights_orign['layer2.0.bn2.weight'] / (weights_orign['layer2.0.bn2.running_var'] + 1e-05)**0.5 + weights_orign['layer2.0.bn2.bias']

weights['layer2.1.conv1.weight'] =  weights_orign['layer2.1.conv1.weight'] * weights_orign['layer2.1.bn1.weight'].reshape([-1,1,1,1]) / (weights_orign['layer2.1.bn1.running_var'].reshape([-1,1,1,1]) + 1e-05)**0.5
weights['layer2.1.conv1.bias']   = -weights_orign['layer2.1.bn1.running_mean'] * weights_orign['layer2.1.bn1.weight'] / (weights_orign['layer2.1.bn1.running_var'] + 1e-05)**0.5 + weights_orign['layer2.1.bn1.bias']

weights['layer2.1.conv2.weight'] =  weights_orign['layer2.1.conv2.weight'] * weights_orign['layer2.1.bn2.weight'].reshape([-1,1,1,1]) / (weights_orign['layer2.1.bn2.running_var'].reshape([-1,1,1,1]) + 1e-05)**0.5
weights['layer2.1.conv2.bias']   = -weights_orign['layer2.1.bn2.running_mean'] * weights_orign['layer2.1.bn2.weight'] / (weights_orign['layer2.1.bn2.running_var'] + 1e-05)**0.5 + weights_orign['layer2.1.bn2.bias']

weights['layer3.0.conv1.weight'] =  weights_orign['layer3.0.conv1.weight'] * weights_orign['layer3.0.bn1.weight'].reshape([-1,1,1,1]) / (weights_orign['layer3.0.bn1.running_var'].reshape([-1,1,1,1]) + 1e-05)**0.5
weights['layer3.0.conv1.bias']   = -weights_orign['layer3.0.bn1.running_mean'] * weights_orign['layer3.0.bn1.weight'] / (weights_orign['layer3.0.bn1.running_var'] + 1e-05)**0.5 + weights_orign['layer3.0.bn1.bias']

weights['layer3.0.conv2.weight'] =  weights_orign['layer3.0.conv2.weight'] * weights_orign['layer3.0.bn2.weight'].reshape([-1,1,1,1]) / (weights_orign['layer3.0.bn2.running_var'].reshape([-1,1,1,1]) + 1e-05)**0.5
weights['layer3.0.conv2.bias']   = -weights_orign['layer3.0.bn2.running_mean'] * weights_orign['layer3.0.bn2.weight'] / (weights_orign['layer3.0.bn2.running_var'] + 1e-05)**0.5 + weights_orign['layer3.0.bn2.bias']

weights['layer3.1.conv1.weight'] =  weights_orign['layer3.1.conv1.weight'] * weights_orign['layer3.1.bn1.weight'].reshape([-1,1,1,1]) / (weights_orign['layer3.1.bn1.running_var'].reshape([-1,1,1,1]) + 1e-05)**0.5
weights['layer3.1.conv1.bias']   = -weights_orign['layer3.1.bn1.running_mean'] * weights_orign['layer3.1.bn1.weight'] / (weights_orign['layer3.1.bn1.running_var'] + 1e-05)**0.5 + weights_orign['layer3.1.bn1.bias']

weights['layer3.1.conv2.weight'] =  weights_orign['layer3.1.conv2.weight'] * weights_orign['layer3.1.bn2.weight'].reshape([-1,1,1,1]) / (weights_orign['layer3.1.bn2.running_var'].reshape([-1,1,1,1]) + 1e-05)**0.5
weights['layer3.1.conv2.bias']   = -weights_orign['layer3.1.bn2.running_mean'] * weights_orign['layer3.1.bn2.weight'] / (weights_orign['layer3.1.bn2.running_var'] + 1e-05)**0.5 + weights_orign['layer3.1.bn2.bias']

weights['layer4.0.conv1.weight'] =  weights_orign['layer4.0.conv1.weight'] * weights_orign['layer4.0.bn1.weight'].reshape([-1,1,1,1]) / (weights_orign['layer4.0.bn1.running_var'].reshape([-1,1,1,1]) + 1e-05)**0.5
weights['layer4.0.conv1.bias']   = -weights_orign['layer4.0.bn1.running_mean'] * weights_orign['layer4.0.bn1.weight'] / (weights_orign['layer4.0.bn1.running_var'] + 1e-05)**0.5 + weights_orign['layer4.0.bn1.bias']

weights['layer4.0.conv2.weight'] =  weights_orign['layer4.0.conv2.weight'] * weights_orign['layer4.0.bn2.weight'].reshape([-1,1,1,1]) / (weights_orign['layer4.0.bn2.running_var'].reshape([-1,1,1,1]) + 1e-05)**0.5
weights['layer4.0.conv2.bias']   = -weights_orign['layer4.0.bn2.running_mean'] * weights_orign['layer4.0.bn2.weight'] / (weights_orign['layer4.0.bn2.running_var'] + 1e-05)**0.5 + weights_orign['layer4.0.bn2.bias']

weights['layer4.1.conv1.weight'] =  weights_orign['layer4.1.conv1.weight'] * weights_orign['layer4.1.bn1.weight'].reshape([-1,1,1,1]) / (weights_orign['layer4.1.bn1.running_var'].reshape([-1,1,1,1]) + 1e-05)**0.5
weights['layer4.1.conv1.bias']   = -weights_orign['layer4.1.bn1.running_mean'] * weights_orign['layer4.1.bn1.weight'] / (weights_orign['layer4.1.bn1.running_var'] + 1e-05)**0.5 + weights_orign['layer4.1.bn1.bias']

weights['layer4.1.conv2.weight'] =  weights_orign['layer4.1.conv2.weight'] * weights_orign['layer4.1.bn2.weight'].reshape([-1,1,1,1]) / (weights_orign['layer4.1.bn2.running_var'].reshape([-1,1,1,1]) + 1e-05)**0.5
weights['layer4.1.conv2.bias']   = -weights_orign['layer4.1.bn2.running_mean'] * weights_orign['layer4.1.bn2.weight'] / (weights_orign['layer4.1.bn2.running_var'] + 1e-05)**0.5 + weights_orign['layer4.1.bn2.bias']

weights['layer2.0.downsample.weight'] =  weights_orign['layer2.0.downsample.0.weight'] * weights_orign['layer2.0.downsample.1.weight'].reshape([-1,1,1,1]) / (weights_orign['layer2.0.downsample.1.running_var'].reshape([-1,1,1,1]) + 1e-05)**0.5
weights['layer2.0.downsample.bias']   = -weights_orign['layer2.0.downsample.1.running_mean'] * weights_orign['layer2.0.downsample.1.weight'] / (weights_orign['layer2.0.downsample.1.running_var'] + 1e-05)**0.5 + weights_orign['layer2.0.downsample.1.bias']

weights['layer3.0.downsample.weight'] =  weights_orign['layer3.0.downsample.0.weight'] * weights_orign['layer3.0.downsample.1.weight'].reshape([-1,1,1,1]) / (weights_orign['layer3.0.downsample.1.running_var'].reshape([-1,1,1,1]) + 1e-05)**0.5
weights['layer3.0.downsample.bias']   = -weights_orign['layer3.0.downsample.1.running_mean'] * weights_orign['layer3.0.downsample.1.weight'] / (weights_orign['layer3.0.downsample.1.running_var'] + 1e-05)**0.5 + weights_orign['layer3.0.downsample.1.bias']

weights['layer4.0.downsample.weight'] =  weights_orign['layer4.0.downsample.0.weight'] * weights_orign['layer4.0.downsample.1.weight'].reshape([-1,1,1,1]) / (weights_orign['layer4.0.downsample.1.running_var'].reshape([-1,1,1,1]) + 1e-05)**0.5
weights['layer4.0.downsample.bias']   = -weights_orign['layer4.0.downsample.1.running_mean'] * weights_orign['layer4.0.downsample.1.weight'] / (weights_orign['layer4.0.downsample.1.running_var'] + 1e-05)**0.5 + weights_orign['layer4.0.downsample.1.bias']


weights['fc.weight'] = weights_orign['fc.weight']
weights['fc.bias']   = weights_orign['fc.bias']


model.load_state_dict(weights)

torch.save(model.state_dict(), 'fuse.pth')

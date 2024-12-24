import torch
import torch.nn as nn
import torch.nn.functional as F

# re-define block in resnet
class Bottleneck(nn.Module): #for resnet50,resnet101,resnet152
    expansion = 4 # hệ số mở rộng ở lớp conv cuối
    def __init__(self, in_channels, out_channels,i_downsample=None,stride=1) -> None:
        super(Bottleneck,self).__init__()

        #lớp đầu tiên sẽ áp dụng kernel 1x1, giảm số lượng kênh đầu vào xuống outchanels
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)

        # lớp hai sẽ áp dụng kernel 3x3, trích xuất đặc trưng không gian
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=stride,padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        # lớp ba sẽ áp dụng kernel 1x1, gia tăng số lượng kênh lên out_channels*expansion
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1,stride=1,padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels*self.expansion)

        self.i_downsample = i_downsample #giảm kích thước của shotcut nếu size không khớp với đầu ra
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self,x):
        shotcut =x.clone() # lưu lại giá trị x để cộng ở cuối 
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.relu(self.batch_norm2(self.conv2(x)))
        x = self.batch_norm3(self.conv3(x))
        

        # downsample if needed
        if self.i_downsample is not None:
            shotcut = self.i_downsample(shotcut)
        
        x+=shotcut
        x=self.relu(x)
        return x

class Block(nn.Module): # dùng cho resnet18 và resnet34
    expansion = 1
    def __init__(self,in_channels,out_channels,i_downsample= None, stride =1):
        super(Block,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1,stride=stride,bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1,stride=stride,bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self,x):
        shotcut = x.clone()
        
        x = self.relu(self.batch_norm2(self.conv1(x)))
        x = self.batch_norm2(self.conv2(x))

        if self.i_downsample is not None:
            shotcut = self.i_downsample(shotcut)

        x += shotcut
        x = self.relu(x)
        return x

class ResNet(nn.Module):
    def __init__(self,RBlock,layer_list,num_classes,num_channels = 3):
        super(ResNet,self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(num_channels,64,kernel_size=7,stride = 2, padding = 3,bias = False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.layer1 = self._make_layer(RBlock,layer_list[0],planes=64)
        self.layer2 = self._make_layer(RBlock,layer_list[1],planes=128)
        self.layer3 = self._make_layer(RBlock,layer_list[2],planes=256)
        self.layer4 = self._make_layer(RBlock,layer_list[3],planes=512)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1)) # cũng là averge pool nhưng cho phép tự fix đầu ra
        self.fc = nn.Linear(512*RBlock.expansion,num_classes)
    
    def forward(self,x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0],-1) # chuyển thành tensor 2d để phù hợp với lớp FC
        x = self.fc(x)    

        return x

    
    def _make_layer(self,RBlock,block,planes,stride = 1):
        ii_downsample = None
        layers = []

        if stride !=1 or self.in_channels != planes*RBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes*RBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes*RBlock.expansion)
            )
        layers.append(RBlock(self.in_channels,planes,i_downsample = ii_downsample,stride = stride))
        self.in_channels = planes*RBlock.expansion

        for i in range(block-1):
            layers.append(RBlock(self.in_channels,planes))
        return nn.Sequential(*layers)
    

def ResNet18(num_classes,channels=3):
    return ResNet(Block,[2,2,2,2],num_classes,channels)

def ResNet34(num_classes,channels=3):
    return ResNet(Block,[3,4,6,3],num_classes,channels)

def ResNet50(num_classes, channels=3):
    return ResNet(Bottleneck, [3,4,6,3], num_classes, channels)
    
def ResNet101(num_classes, channels=3):
    return ResNet(Bottleneck, [3,4,23,3], num_classes, channels)

def ResNet152(num_classes, channels=3):
    return ResNet(Bottleneck, [3,8,36,3], num_classes, channels)


    








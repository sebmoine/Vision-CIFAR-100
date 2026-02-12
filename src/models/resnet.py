import torch
import torch.nn as nn

ResNet_18 = [2,2,2,2]
ResNet_34 = [3,4,6,3]
ResNet_50 = [3,4,6,3]
ResNet_101 = [3,4,23,3]

class StartingBlock(nn.Module):    
    def __init__(self, dataset, out_channels, in_channels=3):
        super(StartingBlock, self).__init__()

        if "cifar" in dataset.lower(): # petit dataset de 32x32, donc on évite de diviser les dimensions par 4 dès le début...
            self.conv1 = nn.Conv2d(in_channels,out_channels, kernel_size=3, stride=1, padding=1)
            self.pool1 = nn.Identity()
        else:
            self.conv1 = nn.Conv2d(in_channels,out_channels, kernel_size=7, stride=2, padding=3) # padding = kernel // 2 pour conserver la même taille
            self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        return x

class BasicBlock(nn.Module):    
    def __init__(self, in_channels, out_channels, kernel_size=3, downsampling=False):
        super(BasicBlock, self).__init__()
        self.idconv = nn.Conv2d(in_channels,out_channels, kernel_size=1, stride=2, padding=0) # on ne veut pas toucher aux valeurs, seulement la réduction spatiale
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=2, padding=1)
        self.conv2 = nn.Conv2d(out_channels,out_channels, kernel_size=kernel_size, stride=1, padding=1)
        self.conv3 = nn.Conv2d(out_channels,out_channels, kernel_size=kernel_size, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsampling=downsampling

    def forward(self, x):
        if self.downsampling:
            identity = self.idconv(x)
            x = self.conv1(x)
            self.first = False
        else:
            identity = x
            x = self.conv2(x)

        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn2(x)
        x += identity
        x = self.relu(x)
        return x


class SmallResNet(nn.Module):
    def __init__(self, cfg, num_classes):
        super(SmallResNet, self).__init__()

        layers = cfg["layers"]
        assert len(layers)==4, "'layers' must be an 1D array of length 4."
        dataset = cfg["dataset"]

        self.inplace = cfg["inplace"]

        self.startingBlock = StartingBlock(dataset, out_channels=self.inplace)
        self.layer1 = self.make_layer(layers[0], self.inplace,   self.inplace)
        self.layer2 = self.make_layer(layers[0], self.inplace,   self.inplace*2)
        self.layer3 = self.make_layer(layers[0], self.inplace*2, self.inplace*4)
        self.layer4 = self.make_layer(layers[0], self.inplace*4, self.inplace*8)

        self.avgPool = nn.AdaptiveAvgPool2d((1,1)) # [B, DEPTH, X, X] --> [B, DEPTH, 1, 1], avec X=H=W pour une image carrée, on réduit donc les infos profondes en dim 1 avec le AvgPool
        self.flat = nn.Flatten(start_dim=1, end_dim=-1) # Flatten [DEPTH, X, X] --> [B, NUM_FEATURES]
        self.dense = nn.Linear(in_features=self.inplace*8, out_features=num_classes)


    def make_layer(self, N, in_channels, out_channels):
        layer = []
        for i in range(N):
            if i==0 and out_channels==self.inplace:
                layer.append(BasicBlock(in_channels, out_channels, downsampling=False))
            elif i==0 and out_channels!=self.inplace:
                layer.append(BasicBlock(in_channels, out_channels, downsampling=True))
            else :
                layer.append(BasicBlock(out_channels, out_channels, downsampling=False))

        return nn.Sequential(*layer)


    def forward(self, x):
        x = self.startingBlock(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgPool(x)
        x = self.flat(x) # [B, NUM_FEATURES]
        x = self.dense(x)
        return x
        
class MyCifarResNet(nn.Module):
    def __init__(self, cfg, num_classes):
        super(MyCifarResNet, self).__init__()

        layers = cfg["layers"]
        assert len(layers)==4, "'layers' must be an 1D array of length 4."
        dataset = cfg["dataset"]

        self.inplace = cfg["inplace"]

        self.startingBlock = StartingBlock(dataset, out_channels=self.inplace)
        self.layer1 = self.make_layer(layers[0], self.inplace,   self.inplace)
        self.layer2 = self.make_layer(layers[0], self.inplace,   self.inplace*2)
        self.layer3 = self.make_layer(layers[0], self.inplace*2, self.inplace*4)

        self.avgPool = nn.AdaptiveAvgPool2d((1,1)) # [B, DEPTH, X, X] --> [B, DEPTH, 1, 1], avec X=H=W pour une image carrée, on réduit donc les infos profondes en dim 1 avec le AvgPool
        self.flat = nn.Flatten(start_dim=1, end_dim=-1) # Flatten [DEPTH, X, X] --> [B, NUM_FEATURES]
        self.dense = nn.Linear(in_features=self.inplace*4, out_features=num_classes)


    def make_layer(self, N, in_channels, out_channels):
        layer = []
        for i in range(N):
            if i==0 and out_channels==self.inplace:
                layer.append(BasicBlock(in_channels, out_channels, downsampling=False))
            elif i==0 and out_channels!=self.inplace:
                layer.append(BasicBlock(in_channels, out_channels, downsampling=True))
            else :
                layer.append(BasicBlock(out_channels, out_channels, downsampling=False))

        return nn.Sequential(*layer)


    def forward(self, x):
        x = self.startingBlock(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgPool(x)
        x = self.flat(x) # [B, NUM_FEATURES]
        x = self.dense(x)
        return x

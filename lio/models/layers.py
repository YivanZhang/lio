from torch import nn


class FlattenLayer(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.pre = nn.Sequential(
            nn.BatchNorm2d(in_planes),
            nn.ReLU(inplace=True),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_planes, planes,
                      kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes,
                      kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.shortcut = None
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Conv2d(in_planes, self.expansion * planes,
                                      kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = self.pre(x)
        residual = self.shortcut(out) if self.shortcut is not None else x
        out = self.conv(x)
        return out + residual


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.pre = nn.Sequential(
            nn.BatchNorm2d(in_planes),
            nn.ReLU(inplace=True),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False),
        )
        self.shortcut = None
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Conv2d(in_planes, self.expansion * planes,
                                      kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = self.pre(x)
        residual = self.shortcut(out) if self.shortcut is not None else x
        out = self.conv(out)
        return out + residual

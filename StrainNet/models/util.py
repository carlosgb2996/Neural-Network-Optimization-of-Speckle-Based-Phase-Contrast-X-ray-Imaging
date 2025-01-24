import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    Bloque Residual con dos capas convolucionales y una conexión de salto.
    Incluye Dropout para reducir el sobreajuste.
    """
    def __init__(self, in_planes, out_planes, stride=1, batchNorm=True, dropout_prob=0.1):
        super(ResidualBlock, self).__init__()
        self.batchNorm = batchNorm
        self.dropout_prob = dropout_prob
        self.conv1 = conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=stride)
        self.conv2 = conv(batchNorm, out_planes, out_planes, kernel_size=3, stride=1, activate=False)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.dropout = nn.Dropout(p=self.dropout_prob) if self.dropout_prob > 0 else nn.Identity()
        
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes) if batchNorm else nn.Identity()
            )
        else:
            self.shortcut = nn.Identity()
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.dropout(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, activate=True):
    layers = []
    layers.append(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=not batchNorm))
    if batchNorm:
        layers.append(nn.BatchNorm2d(out_planes))
    if activate:
        layers.append(nn.LeakyReLU(0.1, inplace=True))
    return nn.Sequential(*layers)

def predict_flow(in_planes):
    return nn.Conv2d(in_planes,2,kernel_size=3,stride=1,padding=1,bias=False)

def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False),
        nn.LeakyReLU(0.1,inplace=True)
    )

def crop_like(input, target):
    if input.size()[2:] == target.size()[2:]:
        return input
    else:
        return input[:, :, :target.size(2), :target.size(3)]
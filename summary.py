from torchsummary import summary
from ResNet import *

net = ResNet()
summary(net, (3, 32, 32))

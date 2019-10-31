import torch
import torch.nn as nn
from torch.nn import init

def define_D():
    discriminator = NLayerDiscriminator()
    discriminator.initialize()

    return discriminator

class NLayerDiscriminator(nn.Module):
    def __init__(self):
        super(NLayerDiscriminator, self).__init__()

        layer1 = nn.Sequential(
            nn.Conv2d(6, 64, 4, stride=2, padding=1)
        )
        model = [layer1]

        layer2 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        model += [layer2]

        layer3 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256)
        )
        model += [layer3]

        layer4 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512)
        )
        model += [layer4]

        layer5 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, 4, stride=1, padding=1),
        )
        model += [layer5]

        self.model = nn.Sequential(*model)

    def initialize(self):
        init.normal_(self.model[0][0].weight)

        for i in range(1, 5):
            layer = self.model[i]
            init.normal_(layer[1].weight)

            if i >= 1 and i < 4:
                init.constant_(layer[2].weight, 1)
                init.constant_(layer[2].bias, 0)

    def predict(self, x):
        return self.model(x)
import torch
import torch.nn as nn
from torch.nn import init
#from torchsummary import summary
def define_G():
    generator = UnetGenerator()
    generator.initialize()
    
    return generator

class UnetGenerator(nn.Module):
    def __init__(self):
        super(UnetGenerator, self).__init__()

        block1 = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(32, 64, 4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )
        model = [block1]

        block2 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 96, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96),
            
            nn.LeakyReLU(0.2),
            nn.Conv2d(96, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        model += [block2]

        block3 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 192, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(192),
            
            nn.LeakyReLU(0.2),
            nn.Conv2d(192, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
        )
        model += [block3]

        block4 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 384, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(384),
            
            nn.LeakyReLU(0.2),
            nn.Conv2d(384, 512, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512)
        )
        model += [block4]

        block5 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 768, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(768),
            
            nn.LeakyReLU(0.2),
            nn.Conv2d(768, 1024, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024)
        )
        model += [block5]

        block6 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1024, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1024)
        )
        model += [block6]

        block7 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1024, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1024)
        )
        model += [block7]

        block8 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1024, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1024)
        )
        model += [block8]

        block9 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1024, 4, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 1024, 4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(1024)
        )
        model += [block9]

        block10 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(2048, 1024, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.Dropout(0.5)
        )
        model += [block10]

        block11 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(2048, 1024, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.Dropout(0.5)
        )
        model += [block11]

        block12 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(2048, 1024, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.Dropout(0.5)
        )
        model += [block12]

        block13 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(2048, 1280, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1280),
            
            nn.ReLU(),
            nn.ConvTranspose2d(1280, 512, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512)
        )
        model += [block13]

        block14 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(1024, 640, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(640),
            
            nn.ReLU(),
            nn.ConvTranspose2d(640, 256, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256)
        )
        model += [block14]

        block15 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(512, 320, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(320),

            nn.ReLU(),
            nn.ConvTranspose2d(320, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )
        model += [block15]

        block16 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(256, 160, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(160),

            nn.ReLU(),
            nn.ConvTranspose2d(160, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64)
        )
        model += [block16]

        block17 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),

            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )
        model += [block17]

        self.model = nn.Sequential(*model)

    def initialize(self):
        init.normal_(self.model[0][0].weight)

        for i in range(1, 17):
            block = self.model[i]
            init.normal_(block[1].weight)
            
            if i == 8:
                init.normal_(block[3].weight)
                init.constant_(block[4].weight, 1)
                init.constant_(block[4].bias, 0)
                continue

            if i != 16:
                init.constant_(block[2].weight, 1)
                init.constant_(block[2].bias, 0)


    def generate(self, x):
        model = self.model

        print("x", x.shape)
        y1 = model[0](x)
        print("y1", y1.shape)
        y2 = model[1](y1)
        print("y2", y2.shape)
        y3 = model[2](y2)
        print("y3", y3.shape)
        y4 = model[3](y3)
        print("y4", y4.shape)
        y5 = model[4](y4)
        print("y5", y5.shape)
        y6 = model[5](y5)
        print("y6", y6.shape)
        y7 = model[6](y6)
        print("y7", y7.shape)
        y8 = model[7](y7)
        print("y8", y8.shape)
        y9 = model[8](y8)
        print("y9", y9.shape)
        print()
        
        x10 = torch.cat((y9, y8), 1)
        y10 = model[9](x10)
        print("y10", y10.shape)
        x11 = torch.cat((y10, y7), 1)
        y11 = model[10](x11)
        print("y11", y11.shape)
        x12 = torch.cat((y11, y6), 1)
        y12 = model[11](x12)
        print("y12", y12.shape)
        x13 = torch.cat((y12, y5), 1)
        y13 = model[12](x13)
        print("y13", y13.shape)
        x14 = torch.cat((y13, y4), 1)
        y14 = model[13](x14)
        print("y14", y14.shape)
        x15 = torch.cat((y14, y3), 1)
        y15 = model[14](x15)
        print("y15", y15.shape)
        x16 = torch.cat((y15, y2), 1)
        y16 = model[15](x16)
        print("y16", y16.shape)
        x17 = torch.cat((y16, y1), 1)
        y = model[16](x17)
        y = y[:, :, 0:256, 0:256]
        print("y", y.shape)

        return y

#g = UnetGenerator()
#print(g)
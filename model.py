import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler
from torchvision.utils import save_image
import numpy as np

from generator import define_G
from discriminator import define_D
from loss import define_GAN_loss

##############
model_num = 6
##############

def create_model(isTrain):
    print("Creating the model...")

    return Pix2PixModel(isTrain)

def load_model(name, isTrain):
    print("Loading", name, "...")
    model = Pix2PixModel(isTrain=isTrain)
    model.name = name
    model.load_state_dict(torch.load("./checkpoints_"+str(model_num)+"/" + name))

    return model

def save_model(model, name):
    print("Saving the model...")
    torch.save(model.state_dict(), "./checkpoints_"+str(model_num)+"/" + name)

def move_model_to_GPU(model):
    device = "cpu"
    if torch.cuda.is_available():
        print("Moving the model to GPU...")
        device = "cuda:0"
        model.to(device)

    return model, device

class Pix2PixModel(nn.Module):
    def __init__(self, isTrain=False):
        super(Pix2PixModel, self).__init__()
        
        self.netG = define_G()
        
        if isTrain:
            self.netD = define_D()

            self.lambda_L1 = 100
            self.criterionGAN = define_GAN_loss()
            self.criterionL1 = nn.L1Loss()

            self.optimizer_G = optim.Adam(self.netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
            self.optimizer_D = optim.Adam(self.netD.parameters(), lr=0.0002, betas=(0.5, 0.999))

            lambda_G = lambda epoch: 1.0 ** epoch
            lambda_D = lambda epoch: 1.0 ** epoch
            self.scheduler_G = scheduler.LambdaLR(self.optimizer_G, lr_lambda=lambda_G)
            self.scheduler_D = scheduler.LambdaLR(self.optimizer_D, lr_lambda=lambda_D)

    def set_input(self, batch_night, batch_day):
        self.real_night = batch_night
        self.real_day = batch_day

    def forward(self):
        self.fake_day = self.netG.generate(self.real_night)

    def save_images(self, iter_count, batch_size):
        path = "./datasets/night2day/test_results_"+str(model_num)+"/"

        for i in range(batch_size):
            img_num = (iter_count) * batch_size + i

            real_night_numpy = self.real_night[i].data.cpu().numpy()
            fake_day_numpy = self.fake_day[i].data.cpu().numpy()
            real_day_numpy = self.real_day[i].data.cpu().numpy()
            
            image = np.concatenate((real_night_numpy, fake_day_numpy, real_day_numpy), 2) # 2?

            save_image(torch.from_numpy(image).squeeze()/2+0.5, path+self.name+"_"+str(img_num)+'.png', nrow=batch_size)

    def backward_D(self): # ?
        fake_nightday = torch.cat((self.real_night, self.fake_day), 1)
        pred_fake = self.netD.predict(fake_nightday.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        
        real_nightday = torch.cat((self.real_night, self.real_day), 1)
        pred_real = self.netD.predict(real_nightday)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        fake_nightday = torch.cat((self.real_night, self.fake_day), 1)
        pred_fake = self.netD.predict(fake_nightday)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        self.loss_G_L1 = self.criterionL1(self.fake_day, self.real_day) * self.lambda_L1

        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def set_requires_grad(self, net, requires_grad=False):
        for param in net.parameters():
            param.requires_grad = requires_grad

    def optimize_parameters(self, epoch_num):
        self.forward()                   
 
        if epoch_num % 5 == 0:
            self.set_requires_grad(self.netD, True)  
            self.optimizer_D.zero_grad()     
            self.backward_D()                
            self.optimizer_D.step()         

        self.set_requires_grad(self.netD, False)  
        self.optimizer_G.zero_grad()        
        self.backward_G()                   
        self.optimizer_G.step()

    def update_learning_rates(self):
        self.scheduler_G.step()
        self.scheduler_D.step()

    def get_current_losses(self):
        return self.loss_G.item(), self.loss_D.item()

    def get_loss_histories(self):
        return self.loss_history_G, self.loss_history_D
    
    def night2day(self, batch_night):
        return self.netG.generate(batch_night)

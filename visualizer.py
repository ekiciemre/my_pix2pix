import visdom
import matplotlib.pyplot as plt
import numpy as np

from utils import get_training_losses

##############
model_num = 6
##############

epoch_count = 500
save_model_every = 5

class Visualizer():
    def __init__(self):
        pass
    
    def plot_losses(self, loss_history_G, loss_history_D, epoch_history):
        x_G = np.arange(int(epoch_history)) * 5
        y_G = list(map(float,loss_history_G))
        
        plt.plot(x_G, y_G, 'r')
        plt.title("Loss history G")
        plt.xlabel("Epoch")
        plt.ylabel("Generator loss")

        plt.show()

        x_D = np.arange(int(epoch_history)) * 5
        y_D = list(map(float,loss_history_D))

        plt.plot(x_D, y_D, 'b')
        plt.title("Loss history D")
        plt.xlabel("Epoch")
        plt.ylabel("Discriminator loss")
        
        plt.show()

visualizer = Visualizer()
losses_G, losses_D = get_training_losses(model_num)
visualizer.plot_losses(losses_G, losses_D, epoch_count / save_model_every)
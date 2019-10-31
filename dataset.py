import os
import torch
import PIL.Image as img
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

def create_data_loader(mode):
    params = {
        "batch_size": 8,
        "shuffle": True,
        "num_workers": 2
    }   
    
    data_set = Night2DayDataset(mode)
    loader = DataLoader(data_set, **params)

    return loader

class Night2DayDataset(Dataset):
    def __init__(self, mode):
        super(Night2DayDataset, self).__init__()
        
        self.mode = mode
        self.main_folder = "./datasets/night2day"
 
        # Get the ids of the images in the dataset
        if self.mode == "train": self.train_images = os.listdir(self.main_folder + '/train/')
        elif self.mode == "test": self.test_images = os.listdir(self.main_folder + '/test/')
        elif self.mode == "val": self.val_images = os.listdir(self.main_folder + '/val/')

        # Define the transforms
        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]

        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        if self.mode == "train": return len(self.train_images)
        elif self.mode == "test": return len(self.test_images)
        elif self.mode == "val": return len(self.val_images)

    def __getitem__(self, index):
        if self.mode == "train": real_nightday = img.open(self.main_folder + "/train/" + self.train_images[index]).convert("RGB")
        elif self.mode == "test": real_nightday = img.open(self.main_folder + "/test/" + self.test_images[index]).convert("RGB")
        elif self.mode == "val": real_nightday = img.open(self.main_folder + "/val/" + self.val_images[index]).convert("RGB")
        
        # Separate the real night and real day images
        real_night = real_nightday.crop((0, 0, 256, 256))
        real_day = real_nightday.crop((256, 0, 512, 256))

        # Apply the transforms
        real_night = self.transform(real_night)
        real_day = self.transform(real_day)

        return real_night, real_day

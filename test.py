import torch

from model import load_model, move_model_to_GPU
from dataset import create_data_loader

##############
model_num = 6
##############

test_loader = create_data_loader("test")

model = load_model("model_" + str(model_num) + "_epoch_219", isTrain=True) # Optimize et
#model = model.netG
#model.eval()
#print(model.loss_history_G)

model, device = move_model_to_GPU(model)

print("Testing the model...")

for iter, (batch, labels) in enumerate(test_loader):
	batch, labels = batch.to(device), labels.to(device)

	with torch.no_grad():
		model.set_input(batch, labels)
		model.forward()
		model.save_images(iter, len(batch))

print("Testing done.")

#torchvision.utils.save_image(batch[0].squeeze()/2+0.5, 'x.png')

# PSNR for evaluation

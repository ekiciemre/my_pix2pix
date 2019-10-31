import csv

def calculate_avg_epoch_losses(epoch_losses_G, epoch_losses_D, iter_count):
    avg_epoch_loss_G = sum(epoch_losses_G) / iter_count
    avg_epoch_loss_D = sum(epoch_losses_D) / iter_count

    return avg_epoch_loss_G, avg_epoch_loss_D

def save_training_losses(model_num, losses_G, losses_D): # Verimsiz
    rows = zip(losses_G, losses_D)
    
    with open("./checkpoints_"+str(model_num)+"/"+"losses.csv", 'w') as l:
        writer = csv.writer(l)
        
        for row in rows:
            writer.writerow(row)

def get_training_losses(model_num):
    with open("./checkpoints_" + str(model_num) + "/losses.csv", 'r') as csv_file:
        lines = csv_file.readlines()

    losses_G = []
    losses_D = []

    for line in lines:
        data = line.split(',')
        losses_G.append(data[0])
        losses_D.append(data[1])

    return losses_G, losses_D
# Imports
import argparse
import os
import numpy as np
from tqdm import tqdm

# PyTorch Imports
import torch
import torch.nn as nn

# Project Imports
from data.celeba_dataset import CelebaDB
from data.celebahqmask_dataset import CelebaMaskHQDB
from models.DecisionDensenetModel import DecisionDensenetModel



# Create CLI
parser = argparse.ArgumentParser(description="Train the decision model for CelebA, CelebaMaskHQDB databases.")

# CLI Arguments
parser.add_argument('--dataset_name', type=str, required=True, choices=['CelebaDB', 'CelebaMaskHQDB'], help="The name of the database.")
parser.add_argument('--results_dir', type=str, required=True, help="The results directory.")
parser.add_argument('--images_dir', type=str, required=True, help="The images directory.")
parser.add_argument('--images_subdir', type=str, choices=['img_align_celeba', 'img_align_celeba_png', 'img_align_squared128_celeba', 'img_celeba'], help="The images subdirectory.")
parser.add_argument('--eval_dir', type=str, required=True, help="The dataset partition directory.")
parser.add_argument('--anno_dir', type=str, required=True, help="The dataset annotations directory.")
parser.add_argument('--decision_model_name', type=str, choices=['decision_model_celeba', 'decision_model_celebamaskhq'], required=True, help="The name of the decision model.")
parser.add_argument('--load_size', type=int, nargs='+', required=True, help="The size (height, width) to load the images.")
parser.add_argument('--train_attributes_idx', type=int, nargs='+', required=True, help="The indices of the train attributes.")
parser.add_argument('--batch_size', type=int, required=True, help="The batch size to load the data.")
parser.add_argument('--optimizer', type=str, required=True, choices=['Adam'], help="The optimization algorithm to update the parameters of the model.")
parser.add_argument('--lr', type=float, required=True, help="The learning rate for the optimizer.")
parser.add_argument('--step_size', type=int, required=True, help="The step size for the scheduler.")
parser.add_argument('--gamma_scheduler', type=float, required=True, help="The gamma for the scheduler.")
parser.add_argument('--num_epochs', type=int, required=True, help="The number of epochs to train the model.")

# Get argument values
opt = parser.parse_args()



# Load dataset (and subsets)
# CelebaDB
if opt.dataset_name == 'CelebaDB':

    assert opt.images_subdir is not None

    # Train
    data_train = CelebaDB(
        images_dir=opt.images_dir,
        images_subdir=opt.images_subdir,
        eval_dir=opt.eval_dir,
        anno_dir=opt.anno_dir,
        subset='train',
        load_size=tuple(opt.load_size)
    )

    # Validation
    data_val = CelebaDB(
        images_dir=opt.images_dir,
        images_subdir=opt.images_subdir,
        eval_dir=opt.eval_dir,
        anno_dir=opt.anno_dir,
        subset='val',
        load_size=tuple(opt.load_size)
    )

# CelebaMaskHQDB
elif opt.dataset_name == 'CelebaMaskHQDB':

    # Train
    data_train = CelebaMaskHQDB(
        images_dir=opt.images_dir,
        eval_dir=opt.eval_dir,
        anno_dir=opt.anno_dir,
        subset='train',
        load_size=tuple(opt.load_size),
    )

    # Validation
    data_val = CelebaMaskHQDB(
        images_dir=opt.images_dir,
        eval_dir=opt.eval_dir,
        anno_dir=opt.anno_dir,
        subset='val',
        load_size=tuple(opt.load_size),
    )


# Create dataloaders
dataloader_train = torch.utils.data.DataLoader(data_train, batch_size=opt.batch_size, shuffle=True, num_workers=4)
dataloader_val = torch.utils.data.DataLoader(data_val, batch_size=opt.batch_size, shuffle=False, num_workers=4)



# Function: Train one epoch
def train_one_epoch():

    print("Number of batches:", len(dataloader_train))
    total_loss = 0
    stat_loss = 0

    total_acc = np.zeros(len(opt.train_attributes_idx))
    stat_acc = np.zeros(len(opt.train_attributes_idx))

    model.train()

    for batch_idx, batch_data in enumerate(tqdm(dataloader_train)):
        batch_data['image'] = batch_data['image'].to(device)
        batch_data['attributes'] = batch_data['attributes'].to(device)

        # Forward pass
        optimizer.zero_grad()
        inputs = batch_data['image']

        pred = model(inputs)
        pred_labels = torch.where(pred > 0.5, 1.0, 0.0)
        real_labels = torch.index_select(batch_data['attributes'], 1, torch.tensor(opt.train_attributes_idx).to(device))

        # Compute loss and gradients
        loss = criterion(pred, real_labels)
        acc = compute_accuracy(pred_labels, real_labels)

        stat_loss += loss.item()
        total_loss += loss.item()
        stat_acc += acc
        total_acc += acc

        loss.backward()
        optimizer.step()


        batch_interval = 50
        if (batch_idx+1) % batch_interval == 0:
            print(' ---- batch: %03d ----' % (batch_idx+1))
            print('mean loss on the last 50 batches: %f'%(stat_loss/batch_interval))
            print('mean accuracy on the last 50 batches: '+ str(stat_acc/batch_interval))
            stat_loss = 0
            stat_acc = 0


    total_mean_loss = total_loss / len(dataloader_train)
    total_mean_acc = total_acc / len(dataloader_train)
    print('mean loss over training set: %f' % (total_mean_loss))
    print('mean accuracy over training set: ' + str(total_mean_acc))

    return total_mean_loss



# Function: Evaluate one epoch
def evaluate_one_epoch():

    model.eval()

    total_loss = 0
    stat_loss = 0
    total_acc = np.zeros(len(opt.train_attributes_idx))
    stat_acc = np.zeros(len(opt.train_attributes_idx))

    print("Number of batches:", len(dataloader_val))

    for batch_idx, batch_data in enumerate(tqdm(dataloader_val)):
        batch_data['image'] = batch_data['image'].to(device)
        batch_data['attributes'] = batch_data['attributes'].to(device)

        # Forward pass

        inputs = batch_data['image']
        with torch.no_grad():
            pred = model(inputs)
            pred_labels = torch.where(pred > 0.5 , 1.0,0.0)

            real_labels = torch.index_select(batch_data['attributes'],1,torch.tensor(opt.train_attributes_idx).to(device))

        # Compute loss and metrics
        loss = criterion(pred,real_labels)
        acc = compute_accuracy(pred_labels,real_labels)

        stat_loss += loss.item()
        total_loss += loss.item()
        stat_acc += acc
        total_acc += acc


        batch_interval = 50
        if (batch_idx+1) % batch_interval == 0:
            print(' ---- batch: %03d ----' % (batch_idx+1))
            print('mean loss on the last 50 batches: %f'%(stat_loss/batch_interval))
            print('mean accuracy on the last 50 batches: ' + str(stat_acc/batch_interval))
            stat_loss = 0
            stat_acc = 0


    total_mean_loss = total_loss/len(dataloader_val)
    total_mean_acc = total_acc/len(dataloader_val)

    print('mean loss over validation set: %f'%(total_mean_loss))
    print('mean accuracy over validation set: '+str(total_mean_acc))

    return total_mean_loss



# Create model (and select device)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = DecisionDensenetModel(num_classes=len(opt.train_attributes_idx), pretrained=False)
model.to(device)


# Select the model optimization algorithm
if opt.optimizer == 'adam':
    optimizer = torch.optim.Adam(model.parameters(),lr=opt.lr)
else:
    optimizer = torch.optim.SGD(model.parameters(),lr=opt.lr)


# Select the loss function
criterion = nn.BCELoss(reduction='mean')

def compute_accuracy(pred, target):
    same_ids = (pred == target).float().cpu()
    return torch.mean(same_ids,axis=0).numpy()

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.step_size, gamma=opt.gamma_scheduler, verbose=True)



# Create checkpoints directory
checkpoints_dir = os.path.join(opt.results_dir, 'checkpoints', opt.decision_model_name)
if not os.path.exists(checkpoints_dir):
    os.makedirs(checkpoints_dir)
print(checkpoints_dir)


# Save training parameters
with open(os.path.join(checkpoints_dir, "train_params.txt"), "w") as f:
    f.write(str(opt))



# Start training
lowest_loss = np.inf
print("Starting training from the beginning.")

for epoch in range(opt.num_epochs):

    # Train one epoch
    print(' **** EPOCH: %03d ****' % (epoch+1))
    train_one_epoch()

    # Evaluate one epoch
    print(' **** EVALUATION AFTER EPOCH %03d ****' % (epoch+1))
    total_mean_loss = evaluate_one_epoch()
    if total_mean_loss < lowest_loss:
        lowest_loss = total_mean_loss
        save_dict = {
            'epoch':epoch+1,
            'optimizer_state_dict':optimizer.state_dict(),
            'loss':total_mean_loss,
            'model_state_dict': model.state_dict()
        }
        torch.save(save_dict, os.path.join(checkpoints_dir, 'checkpoint.pt'))

    # Update scheduler
    scheduler.step()

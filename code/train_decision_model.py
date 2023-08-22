# Imports
import argparse
import os
import numpy as np

# PyTorch Imports
import torch
import torch.nn as nn

# Project Imports
from data_utilities import BDDOIADB, CelebaDB, CelebaMaskHQDB
from model_utilities import DecisionDensenetModel
from train_val_test_utilities import train_one_epoch_celeba, evaluate_one_epoch_celeba, train_one_epoch_bddoia, evaluate_one_epoch_bddoia



# Create CLI
parser = argparse.ArgumentParser(description="Train the decision model for CelebA, CelebaMaskHQDB, BDDOIADB databases.")

# CLI Arguments
parser.add_argument('--dataset_name', type=str, required=True, choices=['CelebaDB', 'CelebaMaskHQDB', 'BDDOIADB'], help="The name of the database.")
parser.add_argument('--results_dir', type=str, required=True, help="The results directory.")
parser.add_argument('--images_dir', type=str, help="The images directory (for CelebaDB, CelebaMaskHQDB).")
parser.add_argument('--images_subdir', type=str, choices=['img_align_celeba', 'img_align_celeba_png', 'img_align_squared128_celeba', 'img_celeba'], help="The images subdirectory (for CelebaDB).")
parser.add_argument('--eval_dir', type=str, help="The dataset partition directory (for CelebaDB, CelebaMaskHQDB).")
parser.add_argument('--anno_dir', type=str, help="The dataset annotations directory (for CelebaDB, CelebaMaskHQDB).")
parser.add_argument('--data_dir', type=str, help="The dataset data directory (for BDDOIADB).")
parser.add_argument('--metadata_dir', type=str, help="The dataset metadata directory (for BDDOIADB).")
parser.add_argument('--crop_size', type=int, nargs='+', help="The crop size (height, width) to load the images (for BDDOIADB).")
parser.add_argument('--load_size', type=int, nargs='+', help="The size (height, width) to load the images (for CelebaDB, CelebaMaskHQDB).")
parser.add_argument('--decision_model_name', type=str, choices=['decision_model_celeba', 'decision_model_celebamaskhq', 'decision_model_bddoia'], required=True, help="The name of the decision model.")
parser.add_argument('--train_attributes_idx', type=int, nargs='+', required=True, help="The indices of the train attributes.")
parser.add_argument('--batch_size', type=int, required=True, help="The batch size to load the data.")
parser.add_argument('--optimizer', type=str, required=True, choices=['Adam'], help="The optimization algorithm to update the parameters of the model.")
parser.add_argument('--lr', type=float, required=True, help="The learning rate for the optimizer.")
parser.add_argument('--step_size', type=int, help="The step size for the scheduler (for CelebaDB, CelebaMaskHQDB).")
parser.add_argument('--gamma_scheduler', type=float, help="The gamma for the scheduler (for CelebaDB, CelebaMaskHQDB).")
parser.add_argument('--num_epochs', type=int, required=True, help="The number of epochs to train the model.")

# Get argument values
opt = parser.parse_args()



# Load dataset (and subsets)
# CelebaDB
if opt.dataset_name == 'CelebaDB':

    assert opt.images_dir is not None
    assert opt.images_subdir is not None
    assert opt.eval_dir is not None
    assert opt.anno_dir is not None
    assert opt.load_size is not None
    assert opt.step_size is not None
    assert opt.gamma_scheduler is not None

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

    assert opt.images_dir is not None
    assert opt.eval_dir is not None
    assert opt.anno_dir is not None
    assert opt.load_size is not None
    assert opt.step_size is not None
    assert opt.gamma_scheduler is not None

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

# BDDOIADB
else:

    assert opt.data_dir is not None
    assert opt.metadata_dir is not None
    assert opt.crop_size is not None

    # Train
    data_train = BDDOIADB(
        data_dir=opt.data_dir,
        metadata_dir=opt.metadata_dir,
        subset='train',
        crop_size=tuple(opt.crop_size),
    )

    # Validation
    data_val = BDDOIADB(
        data_dir=opt.data_dir,
        metadata_dir=opt.metadata_dir,
        subset='val',
        crop_size=tuple(opt.crop_size)
    )


# Create dataloaders
dataloader_train = torch.utils.data.DataLoader(data_train, batch_size=opt.batch_size, shuffle=True, num_workers=4)
dataloader_val = torch.utils.data.DataLoader(data_val, batch_size=opt.batch_size, shuffle=False, num_workers=4)



# Create model (and select device)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model = DecisionDensenetModel(num_classes=len(opt.train_attributes_idx), pretrained=False)
model.to(device)


# Select the model optimization algorithm
if opt.optimizer == 'adam':
    optimizer = torch.optim.Adam(model.parameters(),lr=opt.lr)
else:
    optimizer = torch.optim.SGD(model.parameters(),lr=opt.lr)


# Select the loss function
criterion = nn.BCELoss(reduction='mean')


# Scheduler for (CelebaDB, CelebaMaskHQDB)
if opt.dataset_name in ('CelebaDB', 'CelebaMaskHQDB'):
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
    if opt.dataset_name in ('CelebaDB', 'CelebaMaskHQDB'):
        train_one_epoch_celeba(opt, dataloader_train, model, optimizer, criterion, device)
    else:
        train_one_epoch_bddoia(opt, dataloader_train, model, optimizer, criterion, device)

    # Evaluate one epoch
    print(' **** EVALUATION AFTER EPOCH %03d ****' % (epoch+1))
    if opt.dataset_name in ('CelebaDB', 'CelebaMaskHQDB'):
        total_mean_loss = evaluate_one_epoch_celeba(opt, dataloader_val, model, criterion, device)
    else:
        total_mean_loss = evaluate_one_epoch_bddoia(opt, dataloader_val, model, criterion, device)
    
    # Save model according to the lowest loss
    if total_mean_loss < lowest_loss:
        lowest_loss = total_mean_loss
        save_dict = {
            'epoch':epoch+1,
            'optimizer_state_dict':optimizer.state_dict(),
            'loss':total_mean_loss,
            'model_state_dict': model.state_dict()
        }
        torch.save(save_dict, os.path.join(checkpoints_dir, 'checkpoint.pt'))


    # Update scheduler (for CelebaDB, CelebaMaskHQDB)
    if opt.dataset_name in ('CelebaDB', 'CelebaMaskHQDB'):
        scheduler.step()

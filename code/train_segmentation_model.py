# Imports
import argparse
import os
import numpy as np

# PyTorch Imports
import torch
import torch.nn as nn
import torchvision

# Project Imports
from data_seg_utilities import BDD10kDB
from train_val_test_seg_utilities import train_one_epoch, evaluate_one_epoch



# Create CLI
parser = argparse.ArgumentParser(description="Train the segmentation model for CelebaMaskHQDB, BDD10kDB databases.")

# CLI Arguments
parser.add_argument('--dataset_name', type=str, required=True, choices=['CelebaMaskHQDB', 'BDD10kDB'], help="The name of the database.")
parser.add_argument('--results_dir', type=str, required=True, help="The results directory.")
parser.add_argument('--images_dir', type=str, help="Images directory (for BDD10kDB).")
parser.add_argument('--labels_dir', type=str, help="Labels directory (for BDD10kDB).")
parser.add_argument('--n_classes', type=int, required=True, choices=[19, 20], help="Number of segmentation classes.")
parser.add_argument('--pretrained', action='store_true', help="Initialize segmentation model with pretrained weights.")
parser.add_argument('--segmentation_network_name', type=str, required=True, choices=['deeplabv3_bdd10k', 'deeplabv3_celebamaskhq'], help="The name for the segmentation network.")
parser.add_argument('--seed', type=int, default=42, required=True, help="Seed for random purposes (to ensure reproducibility).")
parser.add_argument('--batch_size', type=int, default=8, required=True, help="Batch size for the dataloaders.")
parser.add_argument('--num_epochs', type=int, default=50, required=True, help="Number of training epochs.")

# Get argument values
opt = parser.parse_args()



# Load datasets (and subsets)
# CelebaMaskHQDB
if opt.dataset_name == 'CelebaMaskHQDB':
    pass

# BDD100kDB
else:

    assert opt.images_dir is not None
    assert opt.labels_dir is not None
    assert opt.segmentation_network_name == 'deeplabv3_bdd10k'
    assert opt.n_classes == 20

    # Train
    dataset_train = BDD10kDB(
        images_dir=opt.images_dir,
        labels_dir=opt.labels_dir,
        subset='train',
        load_size=512,
        crop_size=512,
        label_nc=19,
        contain_dontcare_label=True,
        semantic_nc=18,
        cache_filelist_read=False,
        cache_filelist_write=False,
        aspect_ratio=2.0,
        augment=True
    )

    # Validation
    dataset_val = BDD10kDB(
        images_dir=opt.images_dir,
        labels_dir=opt.labels_dir,
        subset='val',
        load_size=512,
        crop_size=512,
        label_nc=19,
        contain_dontcare_label=True,
        semantic_nc=18,
        cache_filelist_read=False,
        cache_filelist_write=False,
        aspect_ratio=2.0,
        augment=False
    )


# Get dataloaders
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=True, drop_last=True)
dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=opt.batch_size, shuffle=False, drop_last=False)



# Get device
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Get model
model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=opt.pretrained, num_classes=opt.n_classes)
model.train().to(device)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Loss function
criterion = nn.CrossEntropyLoss(ignore_index=0)


# Create checkpoints directory
checkpoints_dir = os.path.join(opt.results_dir, 'checkpoints', opt.segmentation_network_name)
if not os.path.exists(checkpoints_dir):
    os.makedirs(checkpoints_dir)
print(f"Saving checkpoints at: {checkpoints_dir}")


# Save training parameters
with open(os.path.join(checkpoints_dir, "train_params.txt"), "w") as f:
    f.write(str(opt))


# Start training
lowest_loss = np.inf
for epoch in range(opt.num_epochs):

    # Train one epoch
    print(' **** EPOCH: %03d ****' % (epoch+1))
    train_one_epoch(opt, dataloader_train, model, device, optimizer, criterion)

    # Periodically evaluate
    if epoch == 0 or epoch % 5 == 4:
        print(' **** EVALUATION AFTER EPOCH %03d ****' % (epoch+1))
        total_mean_loss = evaluate_one_epoch(opt, dataloader_val, model, device, criterion)
        if total_mean_loss < lowest_loss:
            lowest_loss = total_mean_loss
            save_dict = {
                'epoch':epoch+1,
                'optimizer_state_dict':optimizer.state_dict(),
                'loss':total_mean_loss,
                'model_state_dict':model.state_dict()
            }
            torch.save(save_dict, os.path.join(checkpoints_dir, 'checkpoint.pt'))

# Imports
import os
import argparse
import numpy as np
import cv2
from tqdm import tqdm

# PyTorch Imports
import torch
import torchvision

# Project Imports
from data_seg_utilities import BDD10kDB, CelebaDB, CelebaMaskHQDB



# Create CLI
parser = argparse.ArgumentParser(description="Infer masks (segmentation model) for CelebaDB, CelebaMaskHQDB, BDD10kDB databases.")

# CLI Arguments
parser.add_argument('--dataset_name', type=str, required=True, choices=['CelebaDB', 'CelebaMaskHQDB', 'BDD10kDB'], help="The name of the database.")
parser.add_argument('--results_dir', type=str, required=True, help="The results directory.")
parser.add_argument('--save_dir_masks', type=str, required=True, help="The directory to save new DeepLabV3 masks.")
parser.add_argument('--images_dir', type=str, help="Images directory (for BDD10kDB, CelebaDB, CelebaMaskHQDB).")
parser.add_argument('--images_subdir', type=str, help="Images subdirectory (for CelebaDB).")
parser.add_argument('--labels_dir', type=str, help="Labels directory (for BDD10kDB).")
parser.add_argument('--masks_dir', type=str, help="Labels directory (for CelebaDB, CelebaMaskHQDB).")
parser.add_argument('--eval_dir', type=str, help="Evaluation directory (for CelebaDB, CelebaMaskHQDB).")
parser.add_argument('--anno_dir', type=str, help="Annotation directory (for CelebaDB, CelebaMaskHQDB).")
parser.add_argument('--n_classes', type=int, required=True, choices=[19, 20], help="Number of segmentation classes.")
parser.add_argument('--pretrained', action='store_true', help="Initialize segmentation model with pretrained weights.")
parser.add_argument('--segmentation_network_name', type=str, required=True, choices=['deeplabv3_bdd10k', 'deeplabv3_celebamaskhq'], help="The name for the segmentation network.")
parser.add_argument('--seed', type=int, default=42, required=True, help="Seed for random purposes (to ensure reproducibility).")
parser.add_argument('--batch_size', type=int, default=8, required=True, help="Batch size for the dataloaders.")

# Get argument values
opt = parser.parse_args()



# Load datasets (and subsets)
# CelebaDB
if opt.dataset_name == 'CelebaDB':
    assert opt.images_dir is not None
    assert opt.images_subdir is not None
    assert opt.eval_dir is not None
    assert opt.anno_dir is not None
    assert opt.segmentation_network_name == 'deeplabv3_celebamaskhq'
    assert opt.n_classes == 19

    # Validation
    dataset_val = CelebaDB(
        images_dir=opt.images_dir, 
        images_subdir=opt.images_subdir, 
        eval_dir=opt.eval_dir, 
        anno_dir=opt.anno_dir, 
        subset='val', 
        load_size=(256, 256), 
        augment=False
    )


# CelebaMaskHQDB
elif opt.dataset_name == 'CelebaMaskHQDB':
    
    assert opt.images_dir is not None
    assert opt.masks_dir is not None
    assert opt.eval_dir is not None
    assert opt.anno_dir is not None
    assert opt.segmentation_network_name == 'deeplabv3_celebamaskhq'
    assert opt.n_classes == 19

    # Validation
    dataset_val = CelebaMaskHQDB(
        images_dir=opt.images_dir,
        masks_dir=opt.masks_dir,
        eval_dir=opt.eval_dir,
        anno_dir=opt.anno_dir,
        subset='val',
        load_size=256,
        crop_size=256,
        label_nc=18,
        contain_dontcare_label=True,
        semantic_nc=19,
        cache_filelist_read=False,
        cache_filelist_write=False,
        aspect_ratio=1.0,
        augment=False,
        seed=opt.seed
    )

# BDD10kDB
else:

    assert opt.images_dir is not None
    assert opt.labels_dir is not None
    assert opt.segmentation_network_name == 'deeplabv3_bdd10k'
    assert opt.n_classes == 20

    # Validation
    dataset_val = BDD10kDB(
        images_dir=opt.images_dir,
        labels_dir=opt.labels_dir,
        subset='val',
        load_size=512,
        crop_size=512,
        label_nc=19,
        contain_dontcare_label=True,
        semantic_nc=20,
        cache_filelist_read=False,
        cache_filelist_write=False,
        aspect_ratio=2.0,
        augment=False,
        seed=opt.seed
    )


# Get dataloaders
dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=opt.batch_size, shuffle=False, drop_last=False)



# Get device
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}.")



# Instantiate DeepLabV3 model
deeplabv3 = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=opt.pretrained, num_classes=opt.n_classes)

# Get checkpoint
checkpoints_dir = os.path.join(opt.results_dir, 'checkpoints', opt.segmentation_network_name)
checkpoint = torch.load(os.path.join(checkpoints_dir, 'checkpoint.pt'))

# Load checkpoint
deeplabv3.load_state_dict(checkpoint['model_state_dict'])

# Get other information
start_epoch = checkpoint['epoch']
lowest_loss = checkpoint['loss']
print(f"Checkpoint has been correctly loaded. Starting from epoch {start_epoch}, with last val loss {lowest_loss}.")

# Move model into device
deeplabv3.to(device)

# Put model into evaluation mode
deeplabv3.eval()



# Create directory to save masks
save_dir_masks = opt.save_dir_masks
if not os.path.exists(save_dir_masks):
    os.makedirs(save_dir_masks)



# Forward validation data in the deeplabv3
for data in tqdm(dataloader_val):

    # Get images
    images = data['image'].to(device)
    
    # Generate predictions
    pred = deeplabv3(images)['out']
    pred_labels = pred.argmax(1)
    
    # Get input image filename
    images_fnames = data['name']

    # Go through batch
    for j in range(len(pred)):

        # Get predicted mask
        mask = np.asarray(pred_labels[j].cpu())
        mask = np.where(mask == 0, 256, mask)
        mask -= 1
        # print(mask)

        # Resized images for CelebaDB
        if opt.dataset_name == 'CelebaDB':
            mask = np.array(mask, dtype='uint8')
            mask = cv2.resize(mask, (128, 128), interpolation=cv2.INTER_AREA)


        # Save masks
        cv2.imwrite(os.path.join(save_dir_masks, images_fnames[j].replace('jpg', 'png')), mask)



print("Finished.")

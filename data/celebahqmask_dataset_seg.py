# Imports
import os
import random
import pandas as pd
from PIL import Image

# PyTorch Imports
import torch
from torchvision import transforms



# Class: CelebaMaskHQDB
class CelebaMaskHQDB(torch.utils.data.Dataset):
    def __init__(self, images_dir='CelebA-HQ-img', masks_dir='CelebAMaskHQ-mask', eval_dir="Eval", anno_dir="Anno", subset='train', load_size=256, crop_size=256, label_nc=18, contain_dontcare_label=True, semantic_nc=19, cache_filelist_read=False, cache_filelist_write=False, aspect_ratio=1.0, augment=False):
        super(CelebaMaskHQDB, self).__init__()

        assert subset in ('train', 'val', 'test')

        # Assign class variables
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.eval_dir = eval_dir
        self.anno_dir = anno_dir
        self.load_size = load_size
        self.crop_size = crop_size
        self.label_nc = label_nc
        self.contain_dontcare_label = contain_dontcare_label
        self.semantic_nc = semantic_nc # label_nc + unknown
        self.cache_filelist_read = cache_filelist_read
        self.cache_filelist_write = cache_filelist_write
        self.aspect_ratio = aspect_ratio        
        self.augment = augment

        # Load images
        celebahq_images = self.load_celebahq_images()

        # Load masks
        celebahq_masks = self.load_celebahq_masks()

        # Load dictionary that maps images to masks
        celebahq_images_masks_dict = self.load_celebahq_images_masks_dict(celebahq_images, celebahq_masks)

        # Load CelebaMaskHQ to CelebA mapping
        self.celebahq_to_celeba_mapp = self.load_celebahq_to_celeba_mapp()

        # Read original data splits
        train_set, val_set, test_set = self.load_data_splits()

        # Fix data splits (using the CelebaMaskHQ to CelebA mapping)
        train_set_f, val_set_f, test_set_f = self.fix_data_splits(train_set, val_set, test_set)

        # Get subsets
        if subset == 'train':
            images_subset, masks_subset = self.get_data_subsets(train_set_f, celebahq_images_masks_dict)
        elif subset == 'val':
            images_subset, masks_subset = self.get_data_subsets(val_set_f, celebahq_images_masks_dict)
        else:
            images_subset, masks_subset = self.get_data_subsets(test_set_f, celebahq_images_masks_dict)        

        # Assign class variables
        self.images = images_subset
        self.masks = masks_subset
        self.subset = subset

        return
    

    # Method: Load data splits
    def load_data_splits(self):

        # Read data partitions file
        list_eval_partition = pd.read_csv(os.path.join(self.eval_dir, "list_eval_partition.txt"), delimiter=" ", header=None)
        list_eval_partition = list_eval_partition.values

        # Get train (column==0)
        train = list_eval_partition[list_eval_partition[:,1]==0]
        train = list(train[:, 0])
        
        # Get validation (column==1)
        validation = list_eval_partition[list_eval_partition[:,1]==1]
        validation = list(validation[:, 0])
        

        # Get test (column==2)
        test = list_eval_partition[list_eval_partition[:,1]==2]
        test = list(test[:, 0])

        return train, validation, test


    # Method: Load CelebA-HQ Images
    def load_celebahq_images(self):

        # Load images directory
        celebahq_images = [i for i in os.listdir(self.images_dir) if not i.startswith('.')]

        return celebahq_images


    # Method: Load CelebA-HQ Masks
    def load_celebahq_masks(self):

        # Load images directory
        celebahq_masks = [i for i in os.listdir(self.masks_dir) if not i.startswith('.')]

        return celebahq_masks


    # Method: Images & Masks Dict
    def load_celebahq_images_masks_dict(self, celebahq_images, celebahq_masks):

        assert len(celebahq_images) == len(celebahq_masks)

        # Create a dictionary
        celebahq_images_masks_dict = dict()

        # Go through images list
        for image_fname in celebahq_images:
            mask_fname = image_fname.replace("jpg", "png")
            
            if mask_fname in celebahq_masks:
                celebahq_images_masks_dict[image_fname] = mask_fname

        return celebahq_images_masks_dict


    # Method: Load CelebA-HQ to CelebA mapping
    def load_celebahq_to_celeba_mapp(self):

        # Build a dictionary
        celebahq_to_celeba_mapp = dict()

        # Read celebahq_to_celeba_mapp file
        celebahq_to_celeba_mapp_txt = os.path.join(self.anno_dir, "CelebA-HQ-to-CelebA-mapping.txt")

        # Open file contents
        with open(celebahq_to_celeba_mapp_txt, 'r') as f:
            for line_idx, line in enumerate(f.readlines()):
                
                # Serialise line
                line_ser = line.strip().split()

                if line_idx == 0:
                    orig_idx = line_ser[1]
                    orig_file = line_ser[2]
                else:
                    idx = line_ser[0]
                    if idx not in celebahq_to_celeba_mapp.keys():
                        celebahq_to_celeba_mapp[idx] = {orig_idx:line_ser[1], orig_file:line_ser[2]}

        return celebahq_to_celeba_mapp
    

    # Method: Fix data splits
    def fix_data_splits(self, train_set, val_set, test_set):

        # Generate lists for fixed partitions
        train_set_f, val_set_f, test_set_f = list(), list(), list()

        # Go through the CelebA-HQ to CelebA mapping
        for img_idx, img_mapp in self.celebahq_to_celeba_mapp.items():

            # Create image filename
            img_fname = f'{img_idx}.jpg'

            # Get original index and original fname
            _, img_orig_fname = img_mapp['orig_idx'], img_mapp['orig_file']

            # From the original fnames, let's map the current images
            if img_orig_fname in train_set:
                train_set_f.append(img_fname)
            elif img_orig_fname in val_set:
                val_set_f.append(img_fname)
            elif img_orig_fname in test_set:
                test_set_f.append(img_fname)

        return train_set_f, val_set_f, test_set_f
    

    # Method: Get data subsets
    def get_data_subsets(self, f_data_split, celebahq_images_masks_dict):

        # Create lists
        images_subset, masks_subset = list(), list()

        # Iterate through f_data_split
        for image_fname in f_data_split:
            if image_fname in celebahq_images_masks_dict.keys():
                mask_fname = celebahq_images_masks_dict[image_fname]
                images_subset.append(image_fname)
                masks_subset.append(mask_fname)

        return images_subset, masks_subset


    # Method: Transforms
    def transforms(self, image, mask):
        
        # Resize
        new_width, new_height = (int(self.load_size / self.aspect_ratio), self.load_size)
        image = transforms.functional.resize(image, (new_width, new_height), Image.BICUBIC)
        mask = transforms.functional.resize(mask, (new_width, new_height), Image.NEAREST)
        
        # Apply augmentation (flips)
        if self.augment:
            if random.random() < 0.5:
                image = transforms.functional.hflip(image)
                mask = transforms.functional.hflip(mask)

        # Convert to tensor
        image = transforms.functional.to_tensor(image)
        mask = transforms.functional.to_tensor(mask)
        
        # Apply normalization
        image = transforms.functional.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        
        return image, mask


    # Method: __len__
    def __len__(self,):
        return len(self.images)


    # Method: __getitem__
    def __getitem__(self, idx):

        # Read and load image(s) and mask(s)
        image = Image.open(os.path.join(self.images_dir, self.images[idx])).convert('RGB')
        mask = Image.open(os.path.join(self.masks_dir, self.masks[idx]))
        
        # Apply augmentation
        image, mask = self.transforms(image, mask)
        mask = mask.float()
        
        return {"image": image, "label": mask, "name": self.images[idx]}

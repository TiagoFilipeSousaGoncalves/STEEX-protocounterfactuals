# Imports
import os
import pandas as pd
from PIL import Image

# PyTorch Imports
import torch
from torchvision import transforms



# Class: CelebaMaskHQDB
class CelebaMaskHQDB(torch.utils.data.Dataset):
    def __init__(self, images_dir='CelebA-HQ-img', eval_dir="Eval", anno_dir="Anno", subset='train', load_size=(256, 256), augment=False):
        super(CelebaMaskHQDB, self).__init__()

        assert subset in ('train', 'val', 'test')

        # Assign class variables
        self.images_dir = images_dir
        self.eval_dir = eval_dir
        self.anno_dir = anno_dir
        self.load_size = load_size
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

        # Create a dictionary
        celebahq_images_masks_dict = dict()

        # Go through images list
        for image_fname in celebahq_images:
            image_id = int(image_fname.split('.')[0])

            # Suffix for masks
            image_mask_suffix = '%05d' % image_id

            # Get masks
            image_masks = [m for m in self.celebahq_masks if m.startswith(image_mask_suffix)]
            
            # Populate dictionary
            if image_fname not in celebahq_images_masks_dict.keys():
                celebahq_images_masks_dict[image_fname] = image_masks
            
        
        # Some sanity checks
        # Number of Images
        nr_images = len(celebahq_images)
        nr_images_in_dict = len(celebahq_images_masks_dict)
        assert nr_images == nr_images_in_dict

        # Number of Masks
        nr_masks = len(celebahq_masks)
        nr_masks_in_dict = 0
        for _, masks in celebahq_images_masks_dict.items():
            nr_masks_in_dict += len(masks)
        assert nr_masks == nr_masks_in_dict

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
                image_masks = celebahq_images_masks_dict[image_fname]
                images_subset.append(image_fname)
                masks_subset.append(image_masks)

        return images_subset, masks_subset


    # Method: Transforms
    def transforms(self, image, attributes):

        image = transforms.functional.resize(image, self.load_size, Image.BICUBIC)
        image = transforms.functional.to_tensor(image)
        image = transforms.functional.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        attributes = torch.Tensor(attributes)
        attributes = (attributes + 1)/2

        return image, attributes


    # Method: __len__
    def __len__(self,):
        return len(self.images)


    # Method: __getitem__
    def __getitem__(self, idx):

        # Read and load image(s)
        image = Image.open(os.path.join(self.images_dir, self.images[idx])).convert('RGB')

        # Read and load attribute(s)
        attributes = self.attributes[self.images[idx]]

        # Apply transforms
        image, attributes = self.transforms(image, attributes)

        return {"image": image, "attributes": attributes, "id": self.images[idx]}

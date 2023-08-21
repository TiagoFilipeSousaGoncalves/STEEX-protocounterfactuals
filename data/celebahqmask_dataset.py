class FaceAttributesDataset(torch.utils.data.Dataset):
    def __init__(self, image_path, attributes_path, load_size=(256, 256), augment=False):
        super(FaceAttributesDataset, self).__init__()

        self.image_path = image_path
        self.attributes_path = attributes_path
        self.load_size = load_size
        self.augment = augment

        self.images, self.attributes,self.attributes_names = self.list_images()

    def __len__(self,):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.image_path, self.images[idx])).convert('RGB')
        attributes = self.attributes[self.images[idx]]

        image, attributes = self.transforms(image, attributes)

        return {"image": image, "attributes": attributes, "id": self.images[idx]}

    def list_images(self):

        images = []
        for item in sorted(os.listdir(self.image_path)):
            images.append(item)

        with open(self.attributes_path, "r") as f:
            lines = f.readlines()

        attributes = dict()

        attributes_names = lines[1].split(" ")

        for idx,line in enumerate(lines[2:]):
            name = line.split(" ")[0]
            attr = np.array(line.split(" ")[2:]).astype(int)
            attributes[name] = attr

        return images,attributes,attributes_names


    def transforms(self,image,attributes):

        image = TR.functional.resize(image, self.load_size, Image.BICUBIC)
        image = TR.functional.to_tensor(image)
        image = TR.functional.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        attributes = torch.Tensor(attributes)
        attributes = (attributes + 1)/2

        return image, attributes



# Imports
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

# PyTorch Imports
import torch
import torch.nn as nn
from torchvision import transforms as TR


# Class: CelebaMaskHQDB
class CelebaMaskHQDB:
    def __init__(self, images_dir='CelebA-HQ-img', masks_dir='CelebAMask-HQ-mask-anno', eval_dir="Eval", anno_dir="Anno"):

        # Add variables to class variables
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.eval_dir = eval_dir
        self.anno_dir = anno_dir

        # Data splits
        self.train_set, self.val_set, self.test_set = self.load_data_splits()

        # Load annotations
        self.celebahq_to_celeba_mapp = self.load_celebahq_to_celeba_mapp()
        self.celebamaskhq_attributes = self.load_celebamaskhq_attributes()
        self.celebamaskhq_pose = self.load_celebamaskhq_pose()

        # Load images
        self.celebahq_images = self.load_celebahq_images()

        # Load masks
        self.celebahq_masks = self.load_celebahq_masks()

        # Load dictionary that maps images to masks
        self.celebahq_images_masks_dict = self.load_celebahq_images_masks_dict()

        return
    

    # Method: Load data splits
    def load_data_splits(self):

        # Read data partitions file
        list_eval_partition = pd.read_csv(os.path.join(self.eval_dir, "list_eval_partition.txt"), delimiter=" ", header=None)
        list_eval_partition = list_eval_partition.values
        # print(list_eval_partition)
        # print(list_eval_partition.shape)

        # Get train (column==0)
        train = list_eval_partition[list_eval_partition[:,1]==0]
        # print(train.shape)
        train = list(train[:, 0])
        
        
        # Get validation (column==1)
        validation = list_eval_partition[list_eval_partition[:,1]==1]
        # print(validation.shape)
        validation = list(validation[:, 0])
        

        # Get test (column==2)
        test = list_eval_partition[list_eval_partition[:,1]==2]
        # print(test.shape)
        test = list(test[:, 0])
        

        # Sanity check
        # print(len(train)+len(validation)+len(test) == len(list_eval_partition))

        return train, validation, test


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
                # print(line)
                line_ser = line.strip().split()
                # print(line_ser)

                if line_idx == 0:
                    orig_idx = line_ser[1]
                    orig_file = line_ser[2]
                else:
                    idx = line_ser[0]
                    if idx not in celebahq_to_celeba_mapp.keys():
                        celebahq_to_celeba_mapp[idx] = {orig_idx:line_ser[1], orig_file:line_ser[2]}
        # print(celebahq_to_celeba_mapp)

        return celebahq_to_celeba_mapp
    

    # Method: Load CelebAMaskHQ Attributes
    def load_celebamaskhq_attributes(self):

        # Create a dictionary for celebamaskhq_attributes
        celebamaskhq_attributes = dict()

        # Read celebamaskhq_attributes annotation file
        celebamaskhq_attributes_txt = os.path.join(self.anno_dir, "CelebAMask-HQ-attribute-anno.txt")

        # Open file contents
        with open(celebamaskhq_attributes_txt, 'r') as f:
            for line_idx, line in enumerate(f.readlines()):
                # print(line)
                if line_idx == 0:
                    nr_entries = int(line.strip())
                    # print(nr_entries)
                elif line_idx == 1:
                    column_names = line.strip().split()
                    # print(column_names)
                else:
                    line_ser = line.strip().split()
                    img_name = line_ser[0]
                    img_att = [int(a) for a in line_ser[1::]]

                    assert len(column_names) == len(img_att)

                    celebamaskhq_attributes[img_name] = img_att
        # print(celebamaskhq_attributes)
        
        assert nr_entries == len(celebamaskhq_attributes)

        return celebamaskhq_attributes
    

    # Method: Load CelebAMaskHQ Attributes
    def load_celebamaskhq_pose(self):

        # Create a dictionary for celebamaskhq_attributes
        celebamaskhq_pose = dict()

        # Read celebamaskhq_attributes annotation file
        celebamaskhq_pose_txt = os.path.join(self.anno_dir, "CelebAMask-HQ-pose-anno.txt")

        # Open file contents
        with open(celebamaskhq_pose_txt, 'r') as f:
            for line_idx, line in enumerate(f.readlines()):
                # print(line)
                if line_idx == 0:
                    nr_entries = int(line.strip())
                    # print(nr_entries)
                elif line_idx == 1:
                    column_names = line.strip().split()
                    # print(column_names)
                else:
                    line_ser = line.strip().split()
                    img_name = line_ser[0]
                    img_att = [float(a) for a in line_ser[1::]]

                    assert len(column_names) == len(img_att)

                    celebamaskhq_pose[img_name] = img_att
        # print(celebamaskhq_pose)
        
        assert nr_entries == len(celebamaskhq_pose)

        return celebamaskhq_pose
    

    # Method: Load CelebA-HQ Images
    def load_celebahq_images(self):

        # Load images directory
        celebahq_images = [i for i in os.listdir(self.images_dir) if not i.startswith('.')]
        # print(len(celebahq_images))

        return celebahq_images
    

    # Method: Load CelebA-HQ Masks
    def load_celebahq_masks(self):

        # Load images directory
        celebahq_masks = [i for i in os.listdir(self.masks_dir) if not i.startswith('.')]
        # print(len(celebahq_masks))

        return celebahq_masks
    

    # Method: Images & Masks Dict
    def load_celebahq_images_masks_dict(self):

        # Create a dictionary
        celebahq_images_masks_dict = dict()

        # Go through images list
        for image_fname in self.celebahq_images:
            # print(image_fname)
            image_id = int(image_fname.split('.')[0])
            # print(image_id)

            # Suffix for masks
            image_mask_suffix = '%05d' % image_id
            # print(image_mask_suffix)

            # Get masks
            image_masks = [m for m in self.celebahq_masks if m.startswith(image_mask_suffix)]
            # print(image_fname)
            # print(image_id)
            # print(image_masks)
            
            # Populate dictionary
            if image_fname not in celebahq_images_masks_dict.keys():
                celebahq_images_masks_dict[image_fname] = image_masks
            
        
        # Some sanity checks
        # Number of Images
        nr_images = len(self.celebahq_images)
        nr_images_in_dict = len(celebahq_images_masks_dict)
        assert nr_images == nr_images_in_dict
        print(f"Number of Images: {nr_images}")

        # Number of Masks
        nr_masks = len(self.celebahq_masks)
        nr_masks_in_dict = 0
        for _, masks in celebahq_images_masks_dict.items():
            nr_masks_in_dict += len(masks)
        assert nr_masks == nr_masks_in_dict
        print(f"Number of Masks: {nr_masks}")

        return celebahq_images_masks_dict
  


# Examples usage
if __name__ == "__main__":

    print("Testing CelebaMaskHQDB...")

    # Global variables
    LOAD_INFO = True
    SHOW_IMAGES = False

    # Create database object
    celebamhq_db = CelebaMaskHQDB()

    # Get images and masks
    celebahq_images_masks_dict = celebamhq_db.celebahq_images_masks_dict
    
    # Get annotations
    celebamaskhq_attributes = celebamhq_db.celebamaskhq_attributes
    celebamaskhq_pose = celebamhq_db.celebamaskhq_pose


    # Go through images and masks
    for image_fname, masks_fnames in celebahq_images_masks_dict.items():
        print(f"Image Filename: {image_fname}")
        print(f"Masks Filenames: {masks_fnames}")

        # Get image attributes
        image_attributes = celebamaskhq_attributes[image_fname]
        print(f"Image Attributes: {image_attributes}")

        # Get image pose
        image_pose = celebamaskhq_pose[image_fname]
        print(f"Image Pose: {image_pose}")

        if LOAD_INFO:
            
            # Images
            image_pil = Image.open(os.path.join(celebamhq_db.images_dir, image_fname)).convert('RGB')
            image_arr = np.array(image_pil)
            print(f"Image shape: {image_arr.shape}")

            # Masks
            masks_pil = [Image.open(os.path.join(celebamhq_db.masks_dir, mask_fname)).convert('L') for mask_fname in masks_fnames]
            masks_arr = [np.array(m) for m in masks_pil]
            for midx, m in enumerate(masks_arr):
                print(f"Mask {midx} shape: {m.shape}")
            
            if SHOW_IMAGES:
                plt.imshow(image_arr, cmap='gray')
                for m in masks_arr:
                    plt.imshow(m, cmap='gray', alpha=0.5)
                plt.show()

    print("Finished.")

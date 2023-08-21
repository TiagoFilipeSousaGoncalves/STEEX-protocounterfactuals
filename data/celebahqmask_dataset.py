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

        # Load annotations
        celebamaskhq_attributes, celebamaskhq_attributes_columns = self.load_celebamaskhq_attributes()

        # Load CelebaMaskHQ to CelebA mapping
        self.celebahq_to_celeba_mapp = self.load_celebahq_to_celeba_mapp()

        # Read original data splits
        train_set, val_set, test_set = self.load_data_splits()

        # Fix data splits (using the CelebaMaskHQ to CelebA mapping)
        train_set_f, val_set_f, test_set_f = self.fix_data_splits(train_set, val_set, test_set)

        if subset == 'train':
            images_subset = train_set_f
        elif subset == 'val':
            images_subset = val_set_f
        else:
            images_subset = test_set_f

        # Attributes
        attributes_subset = dict()
        for image_fname in images_subset:
            if image_fname in celebamaskhq_attributes.keys():
                attributes_subset[image_fname] = celebamaskhq_attributes[image_fname]

        assert len(attributes_subset) == len(images_subset)
        

        # Assign class variables
        self.images = images_subset
        self.attributes = attributes_subset
        self.attributes_names = celebamaskhq_attributes_columns
        self.subset = subset

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

        return celebamaskhq_attributes, column_names
    

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

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

        # Data splits
        train_set, val_set, test_set = self.load_data_splits()
        if subset == 'train':
            images_subset = train_set
        elif subset == 'val':
            images_subset = val_set
        else:
            images_subset = test_set

        # Attributes
        images_subset_ = list()
        attributes_subset = dict()
        for image_fname in images_subset:
            if image_fname in celebamaskhq_attributes.keys():
                attributes_subset[image_fname] = celebamaskhq_attributes[image_fname]
                images_subset_.append(image_fname)

        assert len(attributes_subset) == len(images_subset_)
        

        # Assign class variables
        self.images = images_subset_
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

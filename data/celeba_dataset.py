# Imports
import os
import pandas as pd
from PIL import Image

# PyTorch Imports
import torch
from torchvision import transforms



# Class: CelebaDB
class CelebaDB(torch.utils.data.Dataset):
    def __init__(self, images_dir='Img', images_subdir='img_align_celeba', eval_dir='Eval', anno_dir="Anno", subset='train', load_size=(256, 256), augment=False):
        super(CelebaDB, self).__init__()

        assert images_subdir in ('img_celeba', 'img_align_celeba', 'img_align_squared128_celeba')
        assert subset in ('train', 'val', 'test')

        # Add variables to class variables
        self.images_dir = images_dir
        self.images_subdir = images_subdir
        self.eval_dir = eval_dir
        self.anno_dir = anno_dir
        self.load_size = load_size
        self.augment = augment

        # Data splits
        train_set, val_set, test_set = self.load_data_splits()

        # Attributes
        list_attr_celeba, list_attr_celeba_columns = self.load_list_attr_celeba()

        # Get subset of images and attributes
        # Images
        if subset == 'train':
            images_subset = train_set
        elif subset == 'val':
            images_subset = val_set
        else:
            images_subset = test_set
        
        # Attributes
        attributes_subset = dict()
        for image_fname in images_subset:
            if image_fname in list_attr_celeba.keys():
                attributes_subset[image_fname] = list_attr_celeba[image_fname]
        
        assert len(attributes_subset) == len(images_subset)


        # Assign global variables
        self.images = images_subset
        self.attributes = attributes_subset
        self.attributes_names = list_attr_celeba_columns
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


    # Method: Load list of attributes
    def load_list_attr_celeba(self):

        # Read list of attributes file
        list_attr_celeba_txt = os.path.join(self.anno_dir, "list_attr_celeba.txt")
        list_attr_celeba_dict = dict()
        with open(list_attr_celeba_txt, 'r') as f:
            for idx, line in enumerate(f.readlines()):
                # print(line)
                if idx == 0:
                    nr_entries = int(line)
                    # print(nr_entries)
                elif idx == 1:
                    column_names = [c for c in line.strip().split(" ")]
                    # print(column_names)
                else:
                    row = [c for c in line.strip().split(" ")]
                    img_name = row[0]
                    img_attributes = [int(c) for c in row[1::] if c in ('-1', '1')]
                    # print(img_name, img_attributes)
                    # print(len(column_names) == len(img_attributes))

                    assert len(img_attributes) == len(column_names)
                    
                    if img_name not in list_attr_celeba_dict.keys():
                        list_attr_celeba_dict[img_name] = img_attributes

        # print(list_attr_celeba_dict)

        assert nr_entries == len(list_attr_celeba_dict)

        return list_attr_celeba_dict, column_names
    

    # Method: Transforms
    def transforms(self, image, attributes):

        image = transforms.functional.resize(image, self.load_size, Image.BICUBIC)
        image = transforms.functional.to_tensor(image)
        image = transforms.functional.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        attributes = torch.Tensor(attributes)
        attributes = (attributes + 1) / 2

        return image, attributes
    

    # Method: __len__
    def __len__(self,):
        return len(self.images)
    

    # Method: __getitem__
    def __getitem__(self, idx):

        # Read and load image(s)
        image_path = os.path.join(self.images_dir, self.images_subdir)
        image = Image.open(os.path.join(image_path, self.images[idx])).convert('RGB')

        # Read and load attributes
        attributes = self.attributes[self.images[idx]]

        # Apply transforms
        image, attributes = self.transforms(image, attributes)

        return {"image": image, "attributes": attributes, "id": self.images[idx]}

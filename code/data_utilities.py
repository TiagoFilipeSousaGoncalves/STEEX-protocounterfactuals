# Imports
import os
import numpy as np
import pandas as pd
from PIL import Image
import json

# PyTorch Imports
import torch
from torchvision import transforms



# Class: BDDOIADB
class BDDOIADB(torch.utils.data.Dataset):
    def __init__(self, data_dir='data', metadata_dir='metadata', subset='train', crop_size=(1280, 720), augment=False):
        super(BDDOIADB, self).__init__()

        assert subset in ('train', 'val', 'test')

        # Assign variables to class
        self.data_dir = data_dir
        self.metadata_dir = metadata_dir
        self.crop_size = crop_size
        self.augment = augment

        # 25k_images data
        train_25k_actions, val_25k_actions, test_25k_actions = self.load_25k_images_actions()
        train_25k_reasons, val_25k_reasons, test_25k_reasons = self.load_25k_images_reasons()

        # Get data splits
        train_annotations_dict, val_annotations_dict, test_annotations_dict = self.refactor_data_dicts(
            train_25k_actions,
            val_25k_actions,
            test_25k_actions,
            train_25k_reasons,
            val_25k_reasons,
            test_25k_reasons
        )

        # Get the proper subsect
        if subset == 'train':
            images_fnames = train_annotations_dict.keys()
            images_annotations = train_annotations_dict
        elif subset == 'val':
            images_fnames = val_annotations_dict.keys()
            images_annotations = val_annotations_dict
        else:
            images_fnames = test_annotations_dict.keys()
            images_annotations = test_annotations_dict
        

        # Assign variables to class
        self.images_fnames = list(images_fnames)
        self.images_annotations = images_annotations
        self.subset = subset

        return


    # Method: Load 25_images_actions
    def load_25k_images_actions(self):

        # Get JSON filenames
        train_json = 'train_25k_images_actions.json'
        val_json = 'val_25k_images_actions.json'
        test_json = 'test_25k_images_actions.json'

        for json_idx, json_file in enumerate([train_json, val_json, test_json]):
            
            # Open json file
            j_file = open(os.path.join(self.metadata_dir, json_file))

            if json_idx == 0:
                train_25k_actions = json.load(j_file)
                # print(train_25k_actions)
                # print(len(train_25k_actions))
            elif json_idx == 1:
                val_25k_actions = json.load(j_file)
                # print(val_25k_actions)
                # print(len(val_25k_actions))
            else:
                test_25k_actions = json.load(j_file)
                # print(test_25k_actions)
                # print(len(test_25k_actions))
            
            # Close json file
            j_file.close()

        return train_25k_actions, val_25k_actions, test_25k_actions
    

    # Method: Load 25_images_reasons
    def load_25k_images_reasons(self):

        # Get JSON filenames
        train_json = 'train_25k_images_reasons.json'
        val_json = 'val_25k_images_reasons.json'
        test_json = 'test_25k_images_reasons.json'

        for json_idx, json_file in enumerate([train_json, val_json, test_json]):
            
            # Open json file
            j_file = open(os.path.join(self.metadata_dir, json_file))

            if json_idx == 0:
                train_25k_reasons = json.load(j_file)
                # print(train_25k_reasons)
                # print(len(train_25k_reasons))
            elif json_idx == 1:
                val_25k_reasons = json.load(j_file)
                # print(val_25k_reasons)
                # print(len(val_25k_reasons))
            else:
                test_25k_reasons = json.load(j_file)
                # print(test_25k_reasons)
                # print(len(test_25k_reasons))
            
            # Close json file
            j_file.close()

        return train_25k_reasons, val_25k_reasons, test_25k_reasons


    # Method: Refactor data dictionaries
    def refactor_data_dicts(
            self, 
            train_25k_actions,
            val_25k_actions,
            test_25k_actions,
            train_25k_reasons,
            val_25k_reasons,
            test_25k_reasons):

        # Train
        train_actions_images = train_25k_actions["images"]
        train_actions_annotations = train_25k_actions["annotations"]

        # Process the actions (categories) dict
        train_actions_dict= dict()
        for img_info in train_actions_images:
            img_id = img_info['id']
            if img_id not in train_actions_dict.keys():
                train_actions_dict[img_id] = dict()
                train_actions_dict[img_id]["file_name"] = img_info["file_name"]
        for img_info in train_actions_annotations:
            img_id = img_info['id']
            if img_id in train_actions_dict.keys():
                train_actions_dict[img_id]["category"] = img_info["category"]

        # Process the reasons dict
        train_annotations_dict = dict()
        for img_info in train_25k_reasons:
            img_fname = img_info["file_name"]
            img_reason = img_info["reason"]
            if img_fname not in train_annotations_dict.keys():
                train_annotations_dict[img_fname] = dict()
                train_annotations_dict[img_fname]["reason"] = img_reason

        # Process the final annotations dictionary
        for _, img_action_info in train_actions_dict.items():
            img_fname = img_action_info["file_name"]
            img_category = img_action_info["category"]
            if img_fname in train_annotations_dict.keys():
                train_annotations_dict[img_fname]["category"] = img_category


        # Validation
        val_actions_images = val_25k_actions["images"]
        val_actions_annotations = val_25k_actions["annotations"]
        
        # Process the actions (categories) dict
        val_actions_dict= dict()
        for img_info in val_actions_images:
            img_id = img_info['id']
            if img_id not in val_actions_dict.keys():
                val_actions_dict[img_id] = dict()
                val_actions_dict[img_id]["file_name"] = img_info["file_name"]
        for img_info in val_actions_annotations:
            img_id = img_info['id']
            if img_id in val_actions_dict.keys():
                val_actions_dict[img_id]["category"] = img_info["category"]

        # Process the reasons dict
        val_annotations_dict = dict()
        for img_info in val_25k_reasons:
            img_fname = img_info["file_name"]
            img_reason = img_info["reason"]
            if img_fname not in val_annotations_dict.keys():
                val_annotations_dict[img_fname] = dict()
                val_annotations_dict[img_fname]["reason"] = img_reason
        
        # Process the final annotations dictionary
        for _, img_action_info in val_actions_dict.items():
            img_fname = img_action_info["file_name"]
            img_category = img_action_info["category"]
            if img_fname in val_annotations_dict.keys():
                val_annotations_dict[img_fname]["category"] = img_category


        # Test
        test_actions_images = test_25k_actions["images"]
        test_actions_annotations = test_25k_actions["annotations"]

        # Process the actions (categories) dict
        test_actions_dict= dict()
        for img_info in test_actions_images:
            img_id = img_info['id']
            if img_id not in test_actions_dict.keys():
                test_actions_dict[img_id] = dict()
                test_actions_dict[img_id]["file_name"] = img_info["file_name"]
        for img_info in test_actions_annotations:
            img_id = img_info['id']
            if img_id in test_actions_dict.keys():
                test_actions_dict[img_id]["category"] = img_info["category"]

        # Process the reasons dict
        test_annotations_dict = dict()
        for img_info in test_25k_reasons:
            img_fname = img_info["file_name"]
            img_reason = img_info["reason"]
            if img_fname not in test_annotations_dict.keys():
                test_annotations_dict[img_fname] = dict()
                test_annotations_dict[img_fname]["reason"] = img_reason
        
        # Process the final annotations dictionary
        for _, img_action_info in test_actions_dict.items():
            img_fname = img_action_info["file_name"]
            img_category = img_action_info["category"]
            if img_fname in test_annotations_dict.keys():
                test_annotations_dict[img_fname]["category"] = img_category


        return train_annotations_dict, val_annotations_dict, test_annotations_dict


    # Method: Transforms
    def transforms(self, image, target, reason):

        # Get new width and new height
        new_width, new_height = (self.crop_size[1], self.crop_size[0])

        # Transform image
        image = transforms.functional.resize(image, (new_width, new_height), Image.BICUBIC)
        image = transforms.functional.to_tensor(image)
        image = transforms.functional.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        # Transform targets and reason
        target = torch.FloatTensor(target)[0:4]
        reason = torch.FloatTensor(reason)

        return image, target, reason
    

    # Method: __len__
    def __len__(self):
        return len(self.images_fnames)


    # Method: __getitem__
    def __getitem__(self, idx):

        # Get image fname and image path
        image_fname = self.images_fnames[idx]
        image = Image.open(os.path.join(self.data_dir, '25k_images', image_fname)).convert('RGB')

        # Get actions (categories) and reasons
        target = self.images_annotations[image_fname]["category"]
        reason = self.images_annotations[image_fname]["reason"]

        # Create arrays
        target = np.array(target, dtype=np.int64)
        reason = np.array(reason, dtype=np.int64)

        # Apply transforms
        image, target, reason = self.transforms(image, target, reason)

        return {"image":image, "target":target, "reason":reason, "name":image_fname}



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

                    assert len(img_attributes) == len(column_names)
                    
                    if img_name not in list_attr_celeba_dict.keys():
                        list_attr_celeba_dict[img_name] = img_attributes


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
                elif line_idx == 1:
                    column_names = line.strip().split()
                else:
                    line_ser = line.strip().split()
                    img_name = line_ser[0]
                    img_att = [int(a) for a in line_ser[1::]]

                    assert len(column_names) == len(img_att)

                    celebamaskhq_attributes[img_name] = img_att
        
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

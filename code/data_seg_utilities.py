# Imports
import os
import random
import numpy as np
import pandas as pd
from PIL import Image
import json

# PyTorch Imports
import torch
from torchvision import transforms



# Class: BDD10kDB
class BDD10kDB(torch.utils.data.Dataset):
    def __init__(self, images_dir='images', labels_dir='labels', subset='train', load_size=512, crop_size=512, label_nc=19, contain_dontcare_label=True, semantic_nc=20, cache_filelist_read=False, cache_filelist_write=False, aspect_ratio=2.0, augment=False, seed=42):
        super(BDD10kDB, self).__init__()

        assert subset in ('train', 'val', 'test')

        # Assign class variables
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.load_size = load_size
        self.crop_size = crop_size
        self.label_nc = label_nc
        self.contain_dontcare_label = contain_dontcare_label
        self.semantic_nc = semantic_nc # label_nc + unknown
        self.cache_filelist_read = cache_filelist_read
        self.cache_filelist_write = cache_filelist_write
        self.aspect_ratio = aspect_ratio
        self.augment = augment


        # Set random seed
        random.seed(seed)


        # Load 100k data
        bdd10k_train_images, bdd10k_val_images, bdd10k_test_images = self.load_bdd_10k_images()

        # Load semantic segmentation data
        bdd10k_semseg_train_masks, bdd10k_semseg_val_masks = self.load_semseg_labels()

        # Get the right data splits
        if subset == 'train':
            images, masks =  self.get_data_splits(bdd10k_train_images, bdd10k_semseg_train_masks)
        elif subset == 'val':
            images, masks = self.get_data_splits(bdd10k_val_images, bdd10k_semseg_val_masks)
        else:
            pass


        # Assign class variables
        self.images = images
        self.masks = masks
        self.subset = subset

        return
    

    # Method: Load 100k images
    def load_bdd_10k_images(self):
        
        # Read images
        bdd_10k_train = [i for i in os.listdir(os.path.join(self.images_dir, "10k", "train")) if not i.startswith('.')]
        bdd_10k_val = [i for i in os.listdir(os.path.join(self.images_dir, "10k", "val")) if not i.startswith('.')]
        bdd_10k_test = [i for i in os.listdir(os.path.join(self.images_dir, "10k", "test")) if not i.startswith('.')]
        
        return bdd_10k_train, bdd_10k_val, bdd_10k_test


    # Method: Load Semantic Segmentation labels
    def load_semseg_labels(self):

        # Read bitmasks
        train_masks = [b for b in os.listdir(os.path.join(self.labels_dir, "sem_seg", "colormaps", "train")) if not b.startswith('.')]
        val_masks = [b for b in os.listdir(os.path.join(self.labels_dir, "sem_seg", "colormaps", "val")) if not b.startswith('.')]

        return train_masks, val_masks


    # Method: Get the right data splits
    def get_data_splits(self, sub_images, sub_masks):

        # Create lists for images and masks
        images, masks = list(), list()

        # Iterate through the different lists
        for image_fname in sub_images:
            mask_fname = image_fname.replace("jpg", "png")

            # If this exists, append it to the lists
            if mask_fname in sub_masks:
                images.append(image_fname)
                masks.append(mask_fname)
        

        assert len(images) == len(masks)

        return images, masks
    

    # Method: Transforms
    def transforms(self, image, mask):
        
        assert image.size == mask.size

        # Resize
        new_width, new_height = (int(self.load_size / self.aspect_ratio), self.load_size)
        image = transforms.functional.resize(image, (new_width, new_height), Image.BICUBIC)
        mask = transforms.functional.resize(mask, (new_width, new_height), Image.NEAREST)
        
        
        # Apply augmentation (flip)
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

        # Get paths
        if self.subset == 'train':
            image_path = os.path.join(self.images_dir, "10k", "train")
            mask_path = os.path.join(self.labels_dir, "sem_seg", "colormaps", "train")
        elif self.subset == 'val':
            image_path = os.path.join(self.images_dir, "10k", "val")
            mask_path = os.path.join(self.labels_dir, "sem_seg", "colormaps", "val")
        else:
            pass
        
        # Load image
        image = Image.open(os.path.join(image_path, self.images[idx])).convert('RGB')

        # Load mask
        mask = Image.open(os.path.join(mask_path, self.masks[idx]))

        # Apply transforms
        image, mask = self.transforms(image, mask)
        mask = mask * 255
        mask += 1.0
        mask = torch.where(mask > 255, torch.FloatTensor([0.0]), mask)
        
        return {"image": image, "label": mask, "name": self.images[idx]}



# Class: BDDOIADB
class BDDOIADB(torch.utils.data.Dataset):
    def __init__(self, data_dir='data', metadata_dir='metadata', subset='train', load_size=512, crop_size=512, label_nc=19, contain_dontcare_label=True, semantic_nc=20, cache_filelist_read=False, cache_filelist_write=False, aspect_ratio=2.0, augment=False, seed=42):
        super(BDDOIADB, self).__init__()

        assert subset in ('train', 'val', 'test')

        # Assign class variables
        self.data_dir = data_dir
        self.metadata_dir = metadata_dir
        self.load_size = load_size
        self.crop_size = crop_size
        self.label_nc = label_nc
        self.contain_dontcare_label = contain_dontcare_label
        self.semantic_nc = semantic_nc # label_nc + unknown
        self.cache_filelist_read = cache_filelist_read
        self.cache_filelist_write = cache_filelist_write
        self.aspect_ratio = aspect_ratio
        self.augment = augment


        # Set random seed
        random.seed(seed)

        # Assign variables to class
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
        elif subset == 'val':
            images_fnames = val_annotations_dict.keys()
        else:
            images_fnames = test_annotations_dict.keys()
        

        # Assign variables to class
        self.images_fnames = list(images_fnames)
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
    def transforms(self, image):

        # Resize
        new_width, new_height = (int(self.load_size / self.aspect_ratio), self.load_size)
        image = transforms.functional.resize(image, (new_width, new_height), Image.BICUBIC)
        
        
        # Apply augmentation (flip)
        if self.augment:
            if random.random() < 0.5:
                image = transforms.functional.hflip(image)
        
        
        # Convert to tensor
        image = transforms.functional.to_tensor(image)
        
        # Apply normalization
        image = transforms.functional.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        
        return image


    # Method: __len__
    def __len__(self,):
        return len(self.images_fnames)


    # Method: __getitem__
    def __getitem__(self, idx):
        
        # Load image
        image_fname = self.images_fnames[idx]
        image = Image.open(os.path.join(self.data_dir, '25k_images', image_fname)).convert('RGB')

        # Apply transforms
        image = self.transforms(image)
        
        return {"image": image, "name": self.images_fnames[idx]}



# Class: CelebaMaskHQDB
class CelebaMaskHQDB(torch.utils.data.Dataset):
    def __init__(self, images_dir='CelebA-HQ-img', masks_dir='CelebAMaskHQ-mask', eval_dir="Eval", anno_dir="Anno", subset='train', load_size=256, crop_size=256, label_nc=18, contain_dontcare_label=True, semantic_nc=19, cache_filelist_read=False, cache_filelist_write=False, aspect_ratio=1.0, augment=False, seed=42):
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


        # Set random seed
        random.seed(seed)


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

        # Convert image to tensor
        image = transforms.functional.to_tensor(image)
        
        # Apply normalization
        image = transforms.functional.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        # TODO: Convert mask to tensor w/out changing its datatype
        # mask = transforms.functional.to_tensor(mask)
        # mask = transforms.functional.pil_to_tensor(mask)
        mask = torch.as_tensor(np.array(mask), dtype=torch.int64)
        
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
        # mask = mask.float()
        
        return {"image": image, "label": mask, "name": self.images[idx]}



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

        # Get subset of images and attributes
        # Images
        if subset == 'train':
            images_subset = train_set
        elif subset == 'val':
            images_subset = val_set
        else:
            images_subset = test_set


        # Assign global variables
        self.images = images_subset
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


    # Method: Transforms
    def transforms(self, image):

        image = transforms.functional.resize(image, self.load_size, Image.BICUBIC)
        image = transforms.functional.to_tensor(image)
        image = transforms.functional.normalize(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


        return image


    # Method: __len__
    def __len__(self,):
        return len(self.images)
    

    # Method: __getitem__
    def __getitem__(self, idx):

        # Read and load image(s)
        image_path = os.path.join(self.images_dir, self.images_subdir)
        image = Image.open(os.path.join(image_path, self.images[idx])).convert('RGB')

        # Apply transforms
        image = self.transforms(image)

        return {"image": image, "name": self.images[idx]}

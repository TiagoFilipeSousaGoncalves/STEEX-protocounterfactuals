# Imports
import os
import numpy as np
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

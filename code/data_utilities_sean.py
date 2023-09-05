"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""



# Imports
import importlib
import os
import numpy as np
import pandas as pd
import random
from PIL import Image
import json

# PyTorch Imports
import torch
import torch.utils.data
import torchvision.transforms as transforms

# Project Imports
import misc_utilities_sean as util



# Class: BaseDataset
class BaseDataset(torch.utils.data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        pass



# Function: Get parameter from the options
def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.preprocess_mode == 'resize_and_crop':
        new_h = new_w = opt.load_size
    elif opt.preprocess_mode == 'scale_width_and_crop':
        new_w = opt.load_size
        new_h = opt.load_size * h // w
    elif opt.preprocess_mode == 'scale_shortside_and_crop':
        ss, ls = min(w, h), max(w, h)  # shortside and longside
        width_is_shorter = w == ss
        ls = int(opt.load_size * ls / ss)
        new_w, new_h = (ss, ls) if width_is_shorter else (ls, ss)

    x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size))

    flip = random.random() > 0.5
    return {'crop_pos': (x, y), 'flip': flip}



# Function: Get transforms list from the options
def get_transform(opt, params, method=Image.BICUBIC, normalize=True, toTensor=True):
    transform_list = []
    if 'resize' in opt.preprocess_mode:
        osize = [opt.load_size, opt.load_size]
        transform_list.append(transforms.Resize(osize, interpolation=method))
    elif 'scale_width' in opt.preprocess_mode:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size, method)))
    elif 'scale_shortside' in opt.preprocess_mode:
        transform_list.append(transforms.Lambda(lambda img: __scale_shortside(img, opt.load_size, method)))

    if 'crop' in opt.preprocess_mode:
        transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.crop_size)))

    if opt.preprocess_mode == 'none':
        base = 32
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base, method)))

    if opt.preprocess_mode == 'fixed':
        w = opt.crop_size
        h = round(opt.crop_size / opt.aspect_ratio)
        transform_list.append(transforms.Lambda(lambda img: __resize(img, w, h, method)))

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    if toTensor:
        transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)



# Function: Apply normalization
def normalize():
    return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))



# Function: Resize data
def __resize(img, w, h, method=Image.BICUBIC):
    return img.resize((w, h), method)



# Function: Apply exponential resize
def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    return img.resize((w, h), method)



# Function: Scale: width
def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), method)



# Function: Scale shortside
def __scale_shortside(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    ss, ls = min(ow, oh), max(ow, oh)  # shortside and longside
    width_is_shorter = ow == ss
    if (ss == target_width):
        return img
    ls = int(target_width * ls / ss)
    nw, nh = (ss, ls) if width_is_shorter else (ls, ss)
    return img.resize((nw, nh), method)



# Function: Crop
def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    return img.crop((x1, y1, x1 + tw, y1 + th))



# Function: Flip
def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img



# Function: Find a dataset using name
def find_dataset_using_name(dataset_name):
    # Given the option --dataset [datasetname],
    # the file "datasets/datasetname_dataset.py"
    # will be imported. 
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    # In the file, the class called DatasetNameDataset() will
    # be instantiated. It has to be a subclass of BaseDataset,
    # and it is case-insensitive.
    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls
            
    if dataset is None:
        raise ValueError("In %s.py, there should be a subclass of BaseDataset "
                         "with class name that matches %s in lowercase." %
                         (dataset_filename, target_dataset_name))

    return dataset



# Function: Get the method that allows to change CLI options
def get_option_setter(dataset_name):    
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options



# Function: Create a dataloader
def create_dataloader(opt):
    dataset = find_dataset_using_name(opt.dataset_mode)
    instance = dataset()
    instance.initialize(opt)
    print("dataset [%s] of size %d was created" %
          (type(instance).__name__, len(instance)))
    dataloader = torch.utils.data.DataLoader(
        instance,
        batch_size=opt.batchSize,
        shuffle=not opt.serial_batches,
        num_workers=int(opt.nThreads),
        drop_last=opt.isTrain,
        pin_memory=True
    )
    return dataloader



# Class: Pix2pixDataset
class Pix2pixDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--no_pairing_check', action='store_true',
                            help='If specified, skip sanity check of correct label-image file pairing')
        return parser

    def initialize(self, opt):
        self.opt = opt

        label_paths, image_paths, instance_paths = self.get_paths(opt)

        util.natural_sort(label_paths)
        util.natural_sort(image_paths)
        if not opt.no_instance:
            util.natural_sort(instance_paths)

        label_paths = label_paths[:opt.max_dataset_size]
        image_paths = image_paths[:opt.max_dataset_size]
        instance_paths = instance_paths[:opt.max_dataset_size]

        if not opt.no_pairing_check:
            for path1, path2 in zip(label_paths, image_paths):
                assert self.paths_match(path1, path2), \
                    "The label-image pair (%s, %s) do not look like the right pair because the filenames are quite different. Are you sure about the pairing? Please see data/pix2pix_dataset.py to see what is going on, and use --no_pairing_check to bypass this." % (path1, path2)

        self.label_paths = label_paths
        self.image_paths = image_paths
        self.instance_paths = instance_paths

        size = len(self.label_paths)
        self.dataset_size = size

    def get_paths(self, opt):
        label_paths = []
        image_paths = []
        instance_paths = []
        assert False, "A subclass of Pix2pixDataset must override self.get_paths(self, opt)"
        return label_paths, image_paths, instance_paths

    def paths_match(self, path1, path2):
        filename1_without_ext = os.path.splitext(os.path.basename(path1))[0]
        filename2_without_ext = os.path.splitext(os.path.basename(path2))[0]
        return filename1_without_ext == filename2_without_ext

    def __getitem__(self, index):
        # Label Image
        label_path = self.label_paths[index]
        label = Image.open(label_path)
        params = get_params(self.opt, label.size)
        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        label_tensor = transform_label(label) * 255.0
        label_tensor[label_tensor == 255] = self.opt.label_nc  # 'unknown' is opt.label_nc

        # input image (real images)
        image_path = self.image_paths[index]
        assert self.paths_match(label_path, image_path), \
            "The label_path %s and image_path %s don't match." % \
            (label_path, image_path)
        image = Image.open(image_path)
        image = image.convert('RGB')

        transform_image = get_transform(self.opt, params)
        image_tensor = transform_image(image)

        # if using instance maps
        if self.opt.no_instance:
            instance_tensor = 0
        else:
            instance_path = self.instance_paths[index]
            instance = Image.open(instance_path)
            if instance.mode == 'L':
                instance_tensor = transform_label(instance) * 255
                instance_tensor = instance_tensor.long()
            else:
                instance_tensor = transform_label(instance)

        input_dict = {'label': label_tensor,
                      'instance': instance_tensor,
                      'image': image_tensor,
                      'path': image_path,
                      }

        # Give subclasses a chance to modify the final output


        self.postprocess(input_dict)

        return input_dict

    def postprocess(self, input_dict):
        return input_dict

    def __len__(self):
        return self.dataset_size


    # Our codes get input images and labels
    def get_input_by_names(self, image_path, image, label_img):
        label = Image.fromarray(label_img)
        params = get_params(self.opt, label.size)
        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        label_tensor = transform_label(label) * 255.0
        label_tensor[label_tensor == 255] = self.opt.label_nc  # 'unknown' is opt.label_nc
        label_tensor.unsqueeze_(0)


        # input image (real images)]
        # image = Image.open(image_path)
        # image = image.convert('RGB')

        transform_image = get_transform(self.opt, params)
        image_tensor = transform_image(image)
        image_tensor.unsqueeze_(0)

        # if using instance maps
        if self.opt.no_instance:
            instance_tensor = torch.Tensor([0])

        input_dict = {'label': label_tensor,
                      'instance': instance_tensor,
                      'image': image_tensor,
                      'path': image_path,
                      }

        # Give subclasses a chance to modify the final output
        self.postprocess(input_dict)

        return input_dict



# Class: CustomDataset
class CustomDataset(Pix2pixDataset):
    """ Dataset that loads images from directories
        Use option --label_dir, --image_dir, --instance_dir to specify the directories.
        The images in the directories are sorted in alphabetical order and paired in order.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='resize_and_crop')
        load_size = 286 if is_train else 256
        parser.set_defaults(load_size=load_size)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(label_nc=13)
        parser.set_defaults(contain_dontcare_label=False)

        parser.add_argument('--label_dir', type=str, required=True,
                            help='path to the directory that contains label images')
        parser.add_argument('--image_dir', type=str, required=True,
                            help='path to the directory that contains photo images')
        parser.add_argument('--instance_dir', type=str, default='',
                            help='path to the directory that contains instance maps. Leave black if not exists')
        return parser

    def get_paths(self, opt):
        label_dir = opt.label_dir
        label_paths = make_dataset(label_dir, recursive=False, read_cache=True)

        image_dir = opt.image_dir
        image_paths = make_dataset(image_dir, recursive=False, read_cache=True)

        if len(opt.instance_dir) > 0:
            instance_dir = opt.instance_dir
            instance_paths = make_dataset(instance_dir, recursive=False, read_cache=True)
        else:
            instance_paths = []

        assert len(label_paths) == len(image_paths), "The #images in %s and %s do not match. Is there something wrong?"

        return label_paths, image_paths, instance_paths



# Class: BDDOIADB
class BDDOIADB(Pix2pixDataset):
    """ Dataset that loads images from directories
        Use option --label_dir, --image_dir, --instance_dir to specify the directories.
        The images in the directories are sorted in alphabetical order and paired in order.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='resize_and_crop')
        load_size = 286 if is_train else 256
        parser.set_defaults(load_size=load_size)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(label_nc=13)
        parser.set_defaults(contain_dontcare_label=False)

        # TODO: Additional CLI option
        # parser.add_argument('--data_dir', type=str, required=True, help='Path to the directory that contains the images.')
        # parser.add_argument('--metadata_dir', type=str, required=True, help='Path to the directory that contains the metadata/annotations.')
        parser.add_argument('--instance_dir', type=str, default='', help='Path to the directory that contains instance maps. Leave black if not exists.')
        
        return parser


    # Method: Get paths
    def get_paths(self, subset):

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

        # Get masks
        images_25k_masks = self.load_25k_images_masks(subset=subset)

        # Get the proper subsect
        if subset == 'train':
            images_fnames = train_annotations_dict.keys()
        elif subset == 'val':
            images_fnames = val_annotations_dict.keys()
        else:
            images_fnames = test_annotations_dict.keys()
        
        # Images masks depend directly on the subset
        images, masks = list(), list()

        # Align dataset
        for img_fname in images_fnames:
            if img_fname in images_25k_masks:
                images.append(img_fname)
                masks.append(img_fname)

        assert len(images) == len(masks)


        return images, masks


    # Method: Initialize
    def initialize(self, opt, subset):

        assert subset in ('train', 'val', 'test')

        # Assign variables
        self.data_dir = opt.data_dir
        self.metadata_dir = opt.metadata_dir
        self.masks_dir = opt.masks_dir

        # Get images and masks
        self.images, self.masks = self.get_paths(subset=subset)

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
    

    # Method: Load DeepLabV3 Masks
    def load_25k_images_masks(self, subset):

        # Read DeepLabV3 Masks directory
        masks_dir = os.path.join(self.masks_dir, subset)
        
        # Get masks filenames
        images_masks = [m for m in os.listdir(masks_dir) if not m.startswith('.')]

        return images_masks
    

    # Method: __getitem__
    def __getitem__(self, idx):
        
        # Label image (masks)
        label_path = os.path.join(self.metadata_dir, 'deeplabv3_masks', self.masks[idx])
        label = Image.open(label_path)
        params = get_params(self.opt, label.size)
        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        label_tensor = transform_label(label) * 255.0
        label_tensor[label_tensor == 255] = self.opt.label_nc  # 'unknown' is opt.label_nc

        # Input image (real images)
        image_path = os.path.join(self.data_dir, '25k_images', self.images[idx])
        assert self.paths_match(label_path, image_path), "The label_path %s and image_path %s don't match." % (label_path, image_path)
        image = Image.open(image_path).convert('RGB')

        # Apply transforms
        transform_image = get_transform(self.opt, params)
        image_tensor = transform_image(image)

        # if using instance maps
        if self.opt.no_instance:
            instance_tensor = 0
        else:
            instance_path = self.instance_paths[idx]
            instance = Image.open(instance_path)
            if instance.mode == 'L':
                instance_tensor = transform_label(instance) * 255
                instance_tensor = instance_tensor.long()
            else:
                instance_tensor = transform_label(instance)

        input_dict = {'label': label_tensor,
                      'instance': instance_tensor,
                      'image': image_tensor,
                      'path': image_path,
                      }

        # Give subclasses a chance to modify the final output
        self.postprocess(input_dict)

        return input_dict


    # Method: Post-processing function
    def postprocess(self, input_dict):
        return input_dict


    # Method: __len__
    def __len__(self):
        return len(self.images)


    # Our codes get input images and labels
    def get_input_by_names(self, image_path, image, label_img):
        label = Image.fromarray(label_img)
        params = get_params(self.opt, label.size)
        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        label_tensor = transform_label(label) * 255.0
        label_tensor[label_tensor == 255] = self.opt.label_nc  # 'unknown' is opt.label_nc
        label_tensor.unsqueeze_(0)


        # input image (real images)]
        # image = Image.open(image_path)
        # image = image.convert('RGB')

        transform_image = get_transform(self.opt, params)
        image_tensor = transform_image(image)
        image_tensor.unsqueeze_(0)

        # if using instance maps
        if self.opt.no_instance:
            instance_tensor = torch.Tensor([0])

        input_dict = {'label': label_tensor,
                      'instance': instance_tensor,
                      'image': image_tensor,
                      'path': image_path,
                      }

        # Give subclasses a chance to modify the final output
        self.postprocess(input_dict)

        return input_dict



# Class: CelebaDB
class CelebaDB(Pix2pixDataset):
    """ Dataset that loads images from directories
        Use option --label_dir, --image_dir, --instance_dir to specify the directories.
        The images in the directories are sorted in alphabetical order and paired in order.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='resize_and_crop')
        load_size = 286 if is_train else 256
        parser.set_defaults(load_size=load_size)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(label_nc=13)
        parser.set_defaults(contain_dontcare_label=False)

        # Additional arguments for CelebaDB
        parser.add_argument('--images_dir', type=str, required=True, help="Path to the directory that contains the images.")
        parser.add_argument('--images_subdir', type=str, required=True, help="Path to the sub-directory that contains the images.")
        parser.add_argument('--eval_dir', type=str, required=True, help="Path to the directory that contains the data splits.")
        parser.add_argument('--anno_dir', type=str, required=True, help="Path to the directory that contains the annotations.")
        
        return parser


    # Method: Get data paths
    def get_paths(self, subset):

        # Data splits
        train_set, val_set, test_set = self.load_data_splits()

        # DeepLabV3 Masks
        deeplabv3_masks = self.load_deeplabv3_masks()

        # Get subset of images and attributes
        # Images
        if subset == 'train':
            images_subset = train_set
        elif subset == 'val':
            images_subset = val_set
        else:
            images_subset = test_set
        

        # Align data (depends on the subset)
        images, masks = list(), list()
        for image_fname in images_subset:
            if image_fname in deeplabv3_masks:
                images.append(image_fname)
                masks.append(image_fname)
        
        assert len(images) == len(masks)

        return images, masks


    # Method: Initialize
    def initialize(self, opt, subset):

        assert opt.images_subdir in ('img_celeba', 'img_align_celeba', 'img_align_squared128_celeba')
        assert subset in ('train', 'val', 'test')

        # Add variables to class variables
        self.images_dir = opt.images_dir
        self.images_subdir = opt.images_subdir
        self.eval_dir = opt.eval_dir
        self.anno_dir = opt.anno_dir
        self.load_size = opt.load_size
        self.augment = augment
        self.subset = subset

        # Get data
        self.images, self.masks = self.get_paths(subset=subset)

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
    

    # Method: Load DeepLabV3 masks
    def load_deeplabv3_masks(self):

        # Get the path of the masks
        masks_path = os.path.join(self.anno_dir, 'deeplabv3_masks')
        masks = [m for m in os.listdir(masks_path) if not m.startswith('.')]

        return masks
    

    # Method: Check if paths match
    def paths_match(self, path1, path2):
        filename1_without_ext = os.path.splitext(os.path.basename(path1))[0]
        filename2_without_ext = os.path.splitext(os.path.basename(path2))[0]
        return filename1_without_ext == filename2_without_ext


    # Method: __getitem__
    def __getitem__(self, idx):

        # Label(s) (mask(s) of the image(s))
        label_path = os.path.join(self.anno_dir, 'deeplabv3_masks', self.masks[idx])
        label = Image.open(label_path)
        params = get_params(self.opt, label.size)
        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        label_tensor = transform_label(label) * 255.0
        label_tensor[label_tensor == 255] = self.opt.label_nc  # 'unknown' is opt.label_nc

        # Image(s)
        image_path = os.path.join(self.images_dir, self.images_subdir, self.images[idx])
        assert self.paths_match(label_path, image_path), "The label_path %s and image_path %s don't match." % (label_path, image_path)
        image = Image.open(image_path).convert('RGB')
        transform_image = get_transform(self.opt, params)
        image_tensor = transform_image(image)

        # if using instance maps
        if self.opt.no_instance:
            instance_tensor = 0
        else:
            instance_path = self.instance_paths[idx]
            instance = Image.open(instance_path)
            if instance.mode == 'L':
                instance_tensor = transform_label(instance) * 255
                instance_tensor = instance_tensor.long()
            else:
                instance_tensor = transform_label(instance)

        input_dict = {'label': label_tensor,
                      'instance': instance_tensor,
                      'image': image_tensor,
                      'path': image_path,
                      }

        # Give subclasses a chance to modify the final output
        self.postprocess(input_dict)

        return input_dict


    # Method: Postprocessing function
    def postprocess(self, input_dict):
        return input_dict


    # Method: __len__
    def __len__(self):
        return len(self.images)


    # Our codes get input images and labels
    def get_input_by_names(self, image_path, image, label_img):
        label = Image.fromarray(label_img)
        params = get_params(self.opt, label.size)
        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        label_tensor = transform_label(label) * 255.0
        label_tensor[label_tensor == 255] = self.opt.label_nc  # 'unknown' is opt.label_nc
        label_tensor.unsqueeze_(0)


        # input image (real images)]
        # image = Image.open(image_path)
        # image = image.convert('RGB')

        transform_image = get_transform(self.opt, params)
        image_tensor = transform_image(image)
        image_tensor.unsqueeze_(0)

        # if using instance maps
        if self.opt.no_instance:
            instance_tensor = torch.Tensor([0])

        input_dict = {'label': label_tensor,
                      'instance': instance_tensor,
                      'image': image_tensor,
                      'path': image_path,
                      }

        # Give subclasses a chance to modify the final output
        self.postprocess(input_dict)

        return input_dict



# Class: CelebaMaskHQDB
class CelebaMaskHQDB(Pix2pixDataset):
    """ Dataset that loads images from directories
        Use option --label_dir, --image_dir, --instance_dir to specify the directories.
        The images in the directories are sorted in alphabetical order and paired in order.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='resize_and_crop')
        load_size = 286 if is_train else 256
        parser.set_defaults(load_size=load_size)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(label_nc=13)
        parser.set_defaults(contain_dontcare_label=False)

        # Additional arguments for CelebaMaskHQDB
        parser.add_argument('--images_dir', type=str, required=True, help='Path to the directory that contains the images.')
        parser.add_argument('--eval_dir', type=str, required=True, help='Path to the directory that contains the data splits.')
        parser.add_argument('--anno_dir', type=str, required=True, help='Path to the directory that contains the annotations.')
        
        return parser


    # Method: 
    def get_paths(self, opt, subset):

        # Load CelebaMaskHQ to CelebA mapping
        self.celebahq_to_celeba_mapp = self.load_celebahq_to_celeba_mapp()

        # Read original data splits
        train_set, val_set, test_set = self.load_data_splits()

        # Fix data splits (using the CelebaMaskHQ to CelebA mapping)
        train_set_f, val_set_f, test_set_f = self.fix_data_splits(train_set, val_set, test_set)

        # Get masks
        deeplabv3_masks = self.load_deeplabv3_masks()

        if subset == 'train':
            images_subset = train_set_f
        elif subset == 'val':
            images_subset = val_set_f
        else:
            images_subset = test_set_f


        # Get final images and masks
        images, masks = list(), list()
        for image_fname in images_subset:
            if image_fname in deeplabv3_masks:
                images.append(image_fname)
                masks.append(image_fname)

        return images, masks
    

    def initialize(self, opt, subset):

        assert subset in ('train', 'val', 'test')

        # Assign class variables
        self.images_dir = opt.images_dir
        self.eval_dir = opt.eval_dir
        self.anno_dir = opt.anno_dir
        self.load_size = opt.load_size
        self.augment = opt.augment


        # Get data
        self.images, self.masks = self.get_paths(subset=subset)

        return
    

    # Method: Check if paths match
    def paths_match(self, path1, path2):
        filename1_without_ext = os.path.splitext(os.path.basename(path1))[0]
        filename2_without_ext = os.path.splitext(os.path.basename(path2))[0]
        return filename1_without_ext == filename2_without_ext


    # Method: __getitem__
    def __getitem__(self, idx):

        # Get label(s) (mask(s) of the image(s))
        label_path = os.path.join(self.anno_dir, 'deeplabv3_maks', self.masks[idx])
        label = Image.open(label_path)
        params = get_params(self.opt, label.size)
        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        label_tensor = transform_label(label) * 255.0
        label_tensor[label_tensor == 255] = self.opt.label_nc  # 'unknown' is opt.label_nc

        # Get image(s)
        image_path = os.path.join(self.images_dir, self.images[idx])
        assert self.paths_match(label_path, image_path), "The label_path %s and image_path %s don't match." % (label_path, image_path)
        image = Image.open(image_path).convert('RGB')
        transform_image = get_transform(self.opt, params)
        image_tensor = transform_image(image)

        # if using instance maps
        if self.opt.no_instance:
            instance_tensor = 0
        else:
            instance_path = self.instance_paths[idx]
            instance = Image.open(instance_path)
            if instance.mode == 'L':
                instance_tensor = transform_label(instance) * 255
                instance_tensor = instance_tensor.long()
            else:
                instance_tensor = transform_label(instance)

        input_dict = {'label': label_tensor,
                      'instance': instance_tensor,
                      'image': image_tensor,
                      'path': image_path,
                      }

        # Give subclasses a chance to modify the final output
        self.postprocess(input_dict)

        return input_dict


    # Method: Postprocessing function
    def postprocess(self, input_dict):
        return input_dict


    # Method: __len__
    def __len__(self):
        return self.dataset_size


    # Our codes get input images and labels
    def get_input_by_names(self, image_path, image, label_img):
        label = Image.fromarray(label_img)
        params = get_params(self.opt, label.size)
        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        label_tensor = transform_label(label) * 255.0
        label_tensor[label_tensor == 255] = self.opt.label_nc  # 'unknown' is opt.label_nc
        label_tensor.unsqueeze_(0)


        # input image (real images)]
        # image = Image.open(image_path)
        # image = image.convert('RGB')

        transform_image = get_transform(self.opt, params)
        image_tensor = transform_image(image)
        image_tensor.unsqueeze_(0)

        # if using instance maps
        if self.opt.no_instance:
            instance_tensor = torch.Tensor([0])

        input_dict = {'label': label_tensor,
                      'instance': instance_tensor,
                      'image': image_tensor,
                      'path': image_path,
                      }

        # Give subclasses a chance to modify the final output
        self.postprocess(input_dict)

        return input_dict
    

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


    # Method: Load DeepLabV3 Masks
    def load_deeplabv3_masks(self):

        # Get masks path
        masks_path = os.path.join(self.anno_dir, 'deeplabv3_masks')
        masks = [m for m in os.listdir(masks_path) if not m.startswith('.')]

        return masks



# Global Variable: Image Extensions
IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff', '.webp']



# Function: Check if it is an image file
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)



# Function: Make dataset recursively
def make_dataset_rec(dir, images):
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, dnames, fnames in sorted(os.walk(dir, followlinks=True)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)



# Function: Make dataset
def make_dataset(dir, recursive=False, read_cache=False, write_cache=False):
    images = []

    if read_cache:
        possible_filelist = os.path.join(dir, 'files.list')
        if os.path.isfile(possible_filelist):
            with open(possible_filelist, 'r') as f:
                images = f.read().splitlines()
                return images

    if recursive:
        make_dataset_rec(dir, images)
    else:
        assert os.path.isdir(dir) or os.path.islink(dir), '%s is not a valid directory' % dir

        for root, dnames, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

    if write_cache:
        filelist_cache = os.path.join(dir, 'files.list')
        with open(filelist_cache, 'w') as f:
            for path in images:
                f.write("%s\n" % path)
            print('wrote filelist cache at %s' % filelist_cache)

    return images



# Function: Default image loader
def default_loader(path):
    return Image.open(path).convert('RGB')



# Class: ImageFolder
# Code from https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current directory as well as the subdirectories
class ImageFolder(torch.utils.data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)

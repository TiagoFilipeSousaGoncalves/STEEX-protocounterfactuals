# Imports
import os
import random
from PIL import Image

# PyTorch Imports
import torch
from torchvision import transforms



# Class: BDD10kDB
class BDD10kDB(torch.utils.data.Dataset):
    def __init__(self, images_dir='images', labels_dir='labels', subset='train', load_size=512, crop_size=512, label_nc=19, contain_dontcare_label=True, semantic_nc=18, cache_filelist_read=False, cache_filelist_write=False, aspect_ratio=2.0, augment=False):
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
        train_masks = [b for b in os.listdir(os.path.join(self.labels_dir, "sem_seg", "masks", "train")) if not b.startswith('.')]
        val_masks = [b for b in os.listdir(os.path.join(self.labels_dir, "sem_seg", "masks", "val")) if not b.startswith('.')]

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
            mask_path = os.path.join(self.labels_dir, "sem_seg", "masks", "train")
        elif self.subset == 'val':
            image_path = os.path.join(self.images_dir, "10k", "val")
            mask_path = os.path.join(self.labels_dir, "sem_seg", "masks", "val")
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

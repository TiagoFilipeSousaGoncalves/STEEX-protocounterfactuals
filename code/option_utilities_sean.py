"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""



# Imports
import sys
import argparse
import os
import pickle

# PyTorch Imports
import torch

# Project Imports
import data_utilities_sean as data
import models_sean as models
import misc_utilities_sean as util



# Class: BaseOptions
class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        
        # Experiment specifics
        parser.add_argument('--name', type=str, required=True, help='name of the experiment. It decides where to store samples and models')

        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--model', type=str, default='pix2pix', help='which model to use')
        parser.add_argument('--norm_G', type=str, default='spectralinstance', help='instance normalization or batch normalization')
        parser.add_argument('--norm_D', type=str, default='spectralinstance', help='instance normalization or batch normalization')
        parser.add_argument('--norm_E', type=str, default='spectralinstance', help='instance normalization or batch normalization')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')

        # Input/Output sizes
        parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        parser.add_argument('--preprocess_mode', type=str, default='scale_width_and_crop', help='scaling and cropping of images at load time.', choices=("resize_and_crop", "crop", "scale_width", "scale_width_and_crop", "scale_shortside", "scale_shortside_and_crop", "fixed", "none"))
        parser.add_argument('--load_size', type=int, default=1024, help='Scale images to this size. The final image will be cropped to --crop_size.')
        parser.add_argument('--crop_size', type=int, default=512, help='Crop to the width of crop_size (after initially scaling the images to load_size.)')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='The ratio width/height. The final height of the load image will be crop_size/aspect_ratio')
        parser.add_argument('--label_nc', type=int, required=True, help='# of input label classes without unknown class. If you have unknown class as class label, specify --contain_dopntcare_label.')
        parser.add_argument('--semantic_nc', type=int, required=True, help="The total number of labels (semantic labels + unknown).")
        parser.add_argument('--contain_dontcare_label', action='store_true', help='if the label map contains dontcare label (dontcare=255)')
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')

        # For setting inputs
        parser.add_argument('--dataroot', type=str, help="Data directory for the databases of the original paper.")
        parser.add_argument('--dataset_mode', type=str, default='coco')
        parser.add_argument('--dataset_name', type=str, choices=['BDDOIADB', 'CelebaDB', 'CelebaMaskHQDB'], help="The name of the dataset (for the counterfactual generation).")

        # For datasets
        parser.add_argument('--data_dir', type=str, help='Path to the directory that contains the images (for BDDOIADB).')
        parser.add_argument('--metadata_dir', type=str, help='Path to the directory that contains the metadata/annotations (for BDDOIADB).')
        parser.add_argument('--images_dir', type=str, help="The directory of the images (for BDDOIADB, CelebaDB, CelebaMaskHQDB).")
        parser.add_argument('--images_subdir', type=str, help="The sub-directory of the images (for CelebaDB).")
        parser.add_argument('--labels_dir', type=str, help="The directory of the labels (for BDDOIADB).")
        parser.add_argument('--masks_dir', type=str, help="The directory of the masks (for BDDOIADB, CelebaDB, CelebaMaskHQDB).")
        parser.add_argument('--eval_dir', type=str, help="The directory of the data splits (for CelebaDB, CelebaMaskHQDB).")
        parser.add_argument('--anno_dir', type=str, help="The directory of the annotations (for CelebaDB, CelebaMaskHQDB).")

        # For computing infrastructure
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        parser.add_argument('--nThreads', default=28, type=int, help='# threads for loading data')
        parser.add_argument('--max_dataset_size', type=int, default=sys.maxsize, help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--load_from_opt_file', action='store_true', help='load the options from checkpoints and use that as default')
        parser.add_argument('--cache_filelist_write', action='store_true', help='saves the current filelist into a text file, so that it loads faster')
        parser.add_argument('--cache_filelist_read', action='store_true', help='reads from the file list cache')

        # For displays
        parser.add_argument('--display_winsize', type=int, default=400, help='display window size')

        # For generator
        parser.add_argument('--netG', type=str, default='spade', help='selects model to use for netG (pix2pixhd | spade)')
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        parser.add_argument('--init_type', type=str, default='xavier', help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--init_variance', type=float, default=0.02, help='variance of the initialization distribution')
        parser.add_argument('--z_dim', type=int, default=256, help="dimension of the latent z vector")

        # For instance-wise features
        parser.add_argument('--no_instance', action='store_true', help='if specified, do *not* add instance map as input')
        parser.add_argument('--nef', type=int, default=16, help='# of encoder filters in the first conv layer')
        parser.add_argument('--use_vae', action='store_true', help='enable training with an image encoder.')

        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, unknown = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)

        # FIXME: Change this to our notation 
        # Modify dataset-related parser options
        dataset_mode = opt.dataset_mode
        # dataset_option_setter = data.get_option_setter(dataset_mode)
        # parser = dataset_option_setter(parser, self.isTrain)

        opt, unknown = parser.parse_known_args()

        # if there is opt_file, load it.
        # The previous default options will be overwritten
        if opt.load_from_opt_file:
            parser = self.update_options_from_file(parser, opt)

        opt = parser.parse_args()
        self.parser = parser
        return opt

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def option_file_path(self, opt, makedir=False):
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if makedir:
            util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt')
        return file_name

    def save_options(self, opt):
        file_name = self.option_file_path(opt, makedir=True)
        with open(file_name + '.txt', 'wt') as opt_file:
            for k, v in sorted(vars(opt).items()):
                comment = ''
                default = self.parser.get_default(k)
                if v != default:
                    comment = '\t[default: %s]' % str(default)
                opt_file.write('{:>25}: {:<30}{}\n'.format(str(k), str(v), comment))

        with open(file_name + '.pkl', 'wb') as opt_file:
            pickle.dump(opt, opt_file)

    def update_options_from_file(self, parser, opt):
        new_opt = self.load_options(opt)
        for k, v in sorted(vars(opt).items()):
            if hasattr(new_opt, k) and v != getattr(new_opt, k):
                new_val = getattr(new_opt, k)
                parser.set_defaults(**{k: new_val})
        return parser

    def load_options(self, opt):
        file_name = self.option_file_path(opt, makedir=False)
        new_opt = pickle.load(open(file_name + '.pkl', 'rb'))
        return new_opt

    def parse(self, save=False):

        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        self.print_options(opt)
        if opt.isTrain:
            self.save_options(opt)

        # Set semantic_nc based on the option.
        # This will be convenient in many places
        opt.semantic_nc = opt.label_nc + \
            (1 if opt.contain_dontcare_label else 0) + \
            (0 if opt.no_instance else 1)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        assert len(opt.gpu_ids) == 0 or opt.batchSize % len(opt.gpu_ids) == 0, \
            "Batch size %d is wrong. It must be a multiple of # GPUs %d." \
            % (opt.batchSize, len(opt.gpu_ids))

        self.opt = opt
        
        return self.opt



# Class: TrainOptions
class TrainOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        
        # For displays
        parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=10, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--no_html', action="store_false", help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        parser.add_argument('--debug', action='store_true', help='only do one epoch and displays at each iteration')
        parser.add_argument('--tf_log', action='store_true', help='if specified, use tensorboard logging. Requires tensorflow installed')

        # For training
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--niter', type=int, default=50, help='# of iter at starting learning rate. This is NOT the total #epochs. Totla #epochs is niter + niter_decay')
        parser.add_argument('--niter_decay', type=int, default=50, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--optimizer', type=str, default='adam')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--beta2', type=float, default=0.999, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--D_steps_per_G', type=int, default=1, help='number of discriminator iterations per generator iterations.')
        parser.add_argument('--augment', action="store_true", help="Activate data augmentation pipeline.")

        # for discriminators
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')
        parser.add_argument('--lambda_vgg', type=float, default=10.0, help='weight for vgg loss')
        parser.add_argument('--no_ganFeat_loss', action='store_true', help='if specified, do *not* use discriminator feature matching loss')
        parser.add_argument('--no_vgg_loss', action='store_true', help='if specified, do *not* use VGG feature matching loss')
        parser.add_argument('--gan_mode', type=str, default='hinge', help='(ls|original|hinge)')
        parser.add_argument('--netD', type=str, default='multiscale', help='(n_layers|multiscale|image)')
        parser.add_argument('--no_TTUR', action='store_true', help='Use TTUR training scheme')
        parser.add_argument('--lambda_kld', type=float, default=0.005)

        parser.add_argument('--status', type=str, default='train')

        self.isTrain = True
        
        return parser



# Class: TestOptions
class TestOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--how_many', type=int, default=float("inf"), help='how many test images to run')

        parser.set_defaults(preprocess_mode='scale_width_and_crop', crop_size=256, load_size=256, display_winsize=256)
        parser.set_defaults(serial_batches=True)
        parser.set_defaults(no_flip=True)
        parser.set_defaults(phase='test')

        parser.add_argument('--status', type=str, default='test')

        self.isTrain = False
        
        return parser



# Class: CelebAOptions
class CelebAOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)

        parser.set_defaults(name="celeba")
        parser.set_defaults(decision_model_ckpt="celeba")

        parser.set_defaults(split="val")
        parser.set_defaults(use_ground_truth_masks=False)

        parser.set_defaults(semantic_nc=19)
        parser.set_defaults(preprocess_mode="scale_width_and_crop")
        parser.set_defaults(load_size=128)
        parser.set_defaults(crop_size=128)
        parser.set_defaults(aspect_ratio=1.0)
        parser.set_defaults(decision_model_nb_classes=3)
        parser.set_defaults(target_attribute=1) # 1 for smile, 2 for young

        return parser



# Class: CelebAMHQOptions
class CelebAMHQOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)

        parser.set_defaults(name="celebamaskhq")
        parser.set_defaults(decision_model_ckpt="celebamaskhq")

        parser.set_defaults(split="test")
        parser.set_defaults(use_ground_truth_masks=False)

        parser.set_defaults(semantic_nc=19)
        parser.set_defaults(preprocess_mode="scale_width_and_crop")
        parser.set_defaults(load_size=256)
        parser.set_defaults(crop_size=256)
        parser.set_defaults(aspect_ratio=1.0)
        parser.set_defaults(decision_model_nb_classes=3)
        parser.set_defaults(target_attribute=1) # 1 for smile, 2 for young

        return parser



# Class: BDDOptions
class BDDOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)

        parser.set_defaults(name="bdd")
        parser.set_defaults(decision_model_ckpt="bdd")

        parser.set_defaults(split="val")
        parser.set_defaults(use_ground_truth_masks=False)

        parser.set_defaults(contain_dontcare_label=True)

        return parser



# Class: Options
class Options(BaseOptions):
    def __init__(self,):
        pass


    def parse(self,):

        opt = BaseOptions().parse()

        # Specific parser for the specified dataset
        if opt.dataset_name == "celeba":
            parser = CelebAOptions()
        elif opt.dataset_name == "celebamhq":
            parser = CelebAMHQOptions()
        elif opt.dataset_name == "bdd":
            parser = BDDOptions()
        else:
            raise NotImplementedError

        opt = parser.parse()

        # Update paths
        if opt.dataset_name == "celeba":
            if opt.use_ground_truth_masks:
                print("No ground-truth masks for CelebA, please set --use_groun_truth_masks to False")
                assert False
            opt.image_dir = os.path.join(opt.dataroot, "celeba_squared_128", "img_squared128_celeba_%s" % split)
            opt.label_dir = os.path.join(opt.dataroot, "celeba_squared_128", "seg_squared128_celeba_%s" % split)
        elif opt.dataset_name == "celebamhq":
            mask_dir = "labels" if opt.use_ground_truth_masks else "predicted_masks"
            opt.image_dir = os.path.join(opt.dataroot, "CelebAMask-HQ", "CelebAMask-HQ", opt.split, "images")
            opt.label_dir = os.path.join(opt.dataroot, "CelebAMask-HQ", "CelebAMask-HQ", opt.split, mask_dir)
        elif opt.dataset_name == "bdd":
            mask_dir = "labels" if opt.use_ground_truth_masks else "predicted_masks"
            opt.image_dir = os.path.join(opt.dataroot, "BDD", "bdd100k", "seg", "images", opt.split)
            opt.label_dir = os.path.join(opt.dataroot, "BDD", "bdd100k", "seg", mask_dir, opt.split)
        else:
            raise NotImplementedError

        return opt
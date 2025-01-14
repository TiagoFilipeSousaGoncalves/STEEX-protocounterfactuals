"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""



# Imports
import sys
from collections import OrderedDict

# PyTorch Imports
import torch.utils.data

# Project Imports
from data_utilities_sean import BDDOIADB, CelebaDB, CelebaMaskHQDB
from option_utilities_sean import TrainOptions
from utilities_sean.iter_counter import IterationCounter
from utilities_sean.visualizer import Visualizer
from train_val_test_utilities_sean import Pix2PixTrainer



# CLI Interface
opt = TrainOptions().parse()

# print options to help debugging
print(' '.join(sys.argv))

# Load the dataset
# dataloader = data.create_dataloader(opt)
if opt.dataset_mode == "BDDOIADB":
    
    assert opt.data_dir is not None
    assert opt.metadata_dir is not None
    assert opt.masks_dir is not None
    assert opt.load_size == 256
    assert opt.crop_size == 256
    assert opt.label_nc == 19
    assert opt.contain_dontcare_label == True
    assert opt.semantic_nc == 20
    assert opt.cache_filelist_read == False
    assert opt.cache_filelist_write == False
    assert opt.aspect_ratio == 1.0
    assert opt.augment == True

    dataset = BDDOIADB()
    dataset.initialize(opt=opt, subset='train')


elif opt.dataset_mode == "CelebaDB":
    
    assert opt.images_dir is not None
    assert opt.images_subdir is not None
    assert opt.masks_dir is not None
    assert opt.eval_dir is not None
    assert opt.anno_dir is not None
    assert opt.load_size == 256
    assert opt.crop_size == 256
    assert opt.label_nc == 18
    assert opt.contain_dontcare_label == True
    assert opt.semantic_nc == 19
    assert opt.cache_filelist_read == False
    assert opt.cache_filelist_write == False
    assert opt.aspect_ratio == 1.0
    assert opt.augment == True

    dataset = CelebaDB()
    dataset.initialize(opt=opt, subset='train')


elif opt.dataset_mode == "CelebaMaskHQDB":
    
    assert opt.images_dir is not None
    assert opt.masks_dir is not None
    assert opt.eval_dir is not None
    assert opt.anno_dir is not None
    assert opt.load_size == 256
    assert opt.crop_size == 256
    assert opt.label_nc == 18
    assert opt.contain_dontcare_label == True
    assert opt.semantic_nc == 19
    assert opt.cache_filelist_read == False
    assert opt.cache_filelist_write == False
    assert opt.aspect_ratio == 1.0
    assert opt.augment == True

    dataset = CelebaMaskHQDB()
    dataset.initialize(opt=opt, subset='train')

else:
    pass


# Create dataloader
dataloader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.nThreads),
    drop_last=True,
    pin_memory=True
)


# create trainer for our model
trainer = Pix2PixTrainer(opt)

# create tool for counting iterations
iter_counter = IterationCounter(opt, len(dataloader))

# create tool for visualization
visualizer = Visualizer(opt)

for epoch in iter_counter.training_epochs():
    iter_counter.record_epoch_start(epoch)
    for i, data_i in enumerate(dataloader, start=iter_counter.epoch_iter):
        iter_counter.record_one_iteration()

        # Training
        # train generator
        if i % opt.D_steps_per_G == 0:
            trainer.run_generator_one_step(data_i)

        # train discriminator
        trainer.run_discriminator_one_step(data_i)

        # Visualizations
        if iter_counter.needs_printing():
            losses = trainer.get_latest_losses()
            visualizer.print_current_errors(epoch, iter_counter.epoch_iter, losses, iter_counter.time_per_iter)
            visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)


        # TODO: Remove visualization mode from this framework
        # if iter_counter.needs_displaying():
        #     visuals = OrderedDict([('input_label', data_i['label']),
        #                            ('synthesized_image', trainer.get_latest_generated()),
        #                            ('real_image', data_i['image'])])
        #     visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far)

        if iter_counter.needs_saving():
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, iter_counter.total_steps_so_far))
            trainer.save('latest')
            iter_counter.record_current_iter()

    trainer.update_learning_rate(epoch)
    iter_counter.record_epoch_end()

    if epoch % opt.save_epoch_freq == 0 or epoch == iter_counter.total_epochs:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, iter_counter.total_steps_so_far))
        trainer.save('latest')
        trainer.save(epoch)

print('Training was successfully finished.')

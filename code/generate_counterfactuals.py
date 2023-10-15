# Imports
import os
import numpy as np
import pickle
import shutil
from PIL import Image

# PyTorch Imports
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# Project Imports
from data_utilities_sean import BDDOIADB, CelebaDB, CelebaMaskHQDB
from option_utilities_sean import TestOptions
from models_sean.pix2pix_model import Pix2PixModel
from model_utilities import DecisionDensenetModel



# Build CLI
opt = TestOptions().parse()



# Assert specific options for each type of database
if opt.dataset_name == "BDDOIADB":

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
    assert opt.augment is not True
    assert opt.decision_model_name == 'decision_model_bddoia'
    assert opt.split == 'val'
    assert opt.use_ground_truth_masks == False
    assert opt.decision_model_nb_classes == 4


    # Z-Semantic Space Meaning
    z_i_meaning = [
        'road',
        'sidewalk',
        'building',
        'wall',
        'fence',
        'pole',
        'traffic_light',
        'traffic_sign',
        'vegetation',
        'terrain',
        'sky',
        'person',
        'rider',
        'car',
        'truck',
        'bus',
        'train',
        'motorcycle',
        'bicycle',
        'unlabeled'
    ]
    
    SIZE = (256, 512)

    dataset = BDDOIADB()
    dataset.initialize(opt=opt, subset=opt.split)


elif opt.dataset_name in ["CelebaDB", "CelebaMaskHQDB"]:

    # Z-Semantic Space Meaning
    z_i_meaning = [
        "background",
        "skin",
        "nose",
        "glasses",
        "left_eye",
        "right_eye",
        "left_brow",
        "right_brow",
        "left_ear",
        "right_ear",
        "mouth",
        "upper_lip",
        "lower_lip",
        "hair",
        "hat",
        "earring",
        "necklace",
        "neck",
        "cloth",
        "nothing"
    ]

    if opt.dataset_name == "CelebaDB":

        assert opt.images_dir is not None
        assert opt.images_subdir is not None
        assert opt.masks_dir is not None
        assert opt.eval_dir is not None
        assert opt.anno_dir is not None
        assert opt.load_size == 128
        assert opt.crop_size == 128
        assert opt.label_nc == 18
        assert opt.contain_dontcare_label == True
        assert opt.semantic_nc == 19
        assert opt.cache_filelist_read == False
        assert opt.cache_filelist_write == False
        assert opt.aspect_ratio == 1.0
        assert opt.augment == True
        assert opt.decision_model_name == 'decision_model_celeba'
        assert opt.preprocess_mode == "scale_width_and_crop"
        assert opt.decision_model_nb_classes == 3
        assert opt.target_attribute == 1
        assert opt.split == "validation"
        assert opt.use_ground_truth_masks == False
        
        SIZE = (128, 128)

        dataset = CelebaDB()
        dataset.initialize(opt=opt, subset=opt.split)
    
    else:

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
        assert opt.decision_model_name == 'decision_model_celebamaskhq'
        assert opt.preprocess_mode == "scale_width_and_crop"
        assert opt.decision_model_nb_classes == 3
        assert opt.target_attribute == 1
        assert opt.split == "test"
        assert opt.use_ground_truth_masks == False

        SIZE = (256, 256)

        dataset = CelebaMaskHQDB()
        dataset.initialize(opt=opt, subset=opt.split)



# Select the device
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


# Map the Z-meaning to a indices
meaning_to_index = {meaning: i for i, meaning in enumerate(z_i_meaning)}


# Transform the list of labels to be optimized to a set of indices
if len(opt.specified_regions) > 0:
    opt.specified_regions = set(meaning_to_index[label] for label in opt.specified_regions.split(","))



# Create experiment directories for results
EXPERIMENT_RESULTS_DIR = os.path.join(opt.results_dir, opt.name, opt.split)
EXPERIMENT_RESULTS_STYLE_DIR = os.path.join(EXPERIMENT_RESULTS_DIR, "styles_test")
if not os.path.exists(EXPERIMENT_RESULTS_STYLE_DIR):
    os.makedirs(EXPERIMENT_RESULTS_STYLE_DIR, exist_ok=True)
opt.style_dir = EXPERIMENT_RESULTS_STYLE_DIR

# Create a dictionary dict and create these directories
directories = {
    "counterfactual_image_dir":  os.path.join(EXPERIMENT_RESULTS_DIR, "final_images"),
    "pkl_dir": os.path.join(EXPERIMENT_RESULTS_DIR, "pkl_dir"),
    "query_image_dir":os.path.join(EXPERIMENT_RESULTS_DIR, "real_images"),
    "reconstructed_image_dir":os.path.join(EXPERIMENT_RESULTS_DIR, "reconstructed_images")
}
for dir_name, dir_path in directories.items():
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)


# Save configuration file
config_path = os.path.join(EXPERIMENT_RESULTS_DIR, "config.pkl")
with open(config_path, "wb") as f:
    pickle.dump(opt, f)



# Load decision model
decision_model = DecisionDensenetModel(num_classes=opt.decision_model_nb_classes, pretrained=False)
checkpoint = torch.load(os.path.join(opt.checkpoints_dir, opt.decision_model_name, 'checkpoint.pt'))
decision_model.load_state_dict(checkpoint['model_state_dict'])
start_epoch = checkpoint['epoch']
print("Decision model correctly loaded. Starting from epoch", start_epoch, "with last val loss", checkpoint["loss"])
decision_model.to(DEVICE)
decision_model.eval()



# It reads the weights by default according to the experiment name (i.e., opt.name)
# So, we need ot provide the name used during the training of SEAN autoencoder
# Load generator G
generator = Pix2PixModel(opt)
generator.to(DEVICE)
generator.eval()



# Data
dataloader = DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=False,
    num_workers=int(opt.nThreads),
    drop_last=False,
    pin_memory=True
)
iterable_data = iter(dataloader)



# Iterate over all the images of the dataloader
for img in range(len(dataloader)):
    print("new batch", img, "over", len(dataloader), "batches")
    data_i = next(iterable_data)
    data_i['store_path'] = [path + "_custom" for path in data_i["path"]]

    initial_scores = decision_model(data_i['image'].to(DEVICE))
    # target = 0 if initial_score > 0.5, else target = 1
    target = (initial_scores[:, opt.target_attribute] < 0.5).double()

    # Compute reconstruction, it also generates the style codes in the folder
    # Computing the reconstructed image generates an ACE.npy file that contains the encoded image
    reconstructed_query_image = generator(data_i, mode='inference').detach().cpu().float().numpy()

    # Get the style_codes which is the spatialized z tensor
    style_codes_numpy = np.zeros((data_i["image"].shape[0], 20, 512))
    for j, image_path in enumerate(data_i["path"]): # For each image of the batch
        img_path = os.path.split(image_path)[1]
        # loop over codes
        for i in range(20):
            style_path = os.path.join(EXPERIMENT_RESULTS_STYLE_DIR, 'style_codes', img_path, str(i), 'ACE.npy')
            if os.path.exists(style_path):
                code = np.load(style_path)
                style_codes_numpy[j, i] += code
    style_codes = torch.Tensor(style_codes_numpy).to(DEVICE)

    # General setting: optimize on everything
    if len(opt.specified_regions) == 0:
        z = style_codes.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([z], lr=opt.counterfactual_generation_lr)

    # Region-targeted setting, optimize on the labels given
    else:
        z_to_optimize_list = []
        zj_list = []
        z_list = []
        for j in range(len(style_codes)):
            for i in range(len(style_codes[j])):
                sc = style_codes[j, i].detach().clone()
                if i in opt.specified_regions:
                    sc.requires_grad = True
                    z_to_optimize_list.append(sc)
                z_list.append(sc)

        z = torch.vstack(z_list).reshape(len(style_codes), len(style_codes[0]), -1)
        optimizer = torch.optim.Adam(z_to_optimize_list, lr=opt.lr)

    # Optimization steps
    for step in range(opt.nb_steps):

        optimizer.zero_grad()

        if len(opt.specified_regions) > 0:
            z = torch.vstack(z_list).reshape(len(style_codes), len(style_codes[0]), -1)

        data_i['custom_z'] = z

        # Generate counterfactual and their proba given by the decision model
        counterfactual_image = generator(data_i, mode='inference_with_custom_z') # batch x channel x width x height
        counterfactual_logits = decision_model(counterfactual_image, before_sigmoid=True)
        counterfactual_probas = torch.sigmoid(counterfactual_logits)

        # Decision loss
        flip_decision_loss = - (1 - target) * torch.log(1 - torch.sigmoid(counterfactual_logits[:, opt.target_attribute])) - target * torch.log(torch.sigmoid(counterfactual_logits[:, opt.target_attribute]))
        loss = flip_decision_loss

        # Proximity loss
        proximity_loss = opt.lambda_prox * torch.sum(torch.square(torch.norm(z - style_codes, dim=2)), axis=1)
        loss += proximity_loss

        loss = loss.sum() # Sum over the batch
        loss.backward()

        # One optimization step
        optimizer.step()

        # Some printing
        if step % 10 == 0:
          print("Step:", step)
          print("Objective loss:", flip_decision_loss.mean().item())
          print("Difference on z:", proximity_loss.mean().item())

    # At the end, save everything
    counterfactual_image_tensor = counterfactual_image.detach().cpu().float().numpy()

    final_scores = counterfactual_probas.detach().cpu().float().numpy()
    final_loss_decision = flip_decision_loss.detach().cpu().float().numpy()
    final_loss_proximity = proximity_loss.detach().cpu().float().numpy()

    for j, image_path in enumerate(data_i["path"]):
        # Create folder for each image of the batch
        img_path = os.path.split(image_path)[1]

        # Save counterfactual image
        counterfactual_image = (np.transpose(counterfactual_image_tensor[j, :, :, :], (1, 2, 0)) + 1) / 2.0 * 255.0
        counterfactual_image = counterfactual_image.astype(np.uint8)
        counterfactual_image = Image.fromarray(counterfactual_image).convert('RGB')
        counterfactual_image.save(os.path.join(directories["counterfactual_image_dir"], img_path.replace(".jpg", ".png")))

        # Delete the saved style codes
        if opt.remove_saved_style_codes:
            style_codes = os.path.join(EXPERIMENT_RESULTS_STYLE_DIR, 'style_codes', img_path)
            shutil.rmtree(style_codes)
            style_codes = os.path.join(EXPERIMENT_RESULTS_STYLE_DIR, 'style_codes', img_path+"_custom")
            shutil.rmtree(style_codes)

        # Save query image
        if opt.save_query_image:
            query_image = Image.open(image_path).convert('RGB')
            query_image = transforms.functional.resize(query_image, SIZE, Image.BICUBIC) # Resize real image
            query_image.save(os.path.join(directories["query_image_dir"], img_path.replace(".jpg", ".png")))

        # Save query reconstruction
        if opt.save_reconstruction:
            reconstructed_query_image_j = reconstructed_query_image[j, :, :, :]
            reconstructed_query_image_j = (np.transpose(reconstructed_query_image_j, (1, 2, 0)) + 1) / 2.0 * 255.0
            reconstructed_query_image_j = reconstructed_query_image_j.astype(np.uint8)
            reconstructed_query_image_j = Image.fromarray(reconstructed_query_image_j).convert('RGB')
            reconstructed_query_image_j.save(os.path.join(directories["reconstructed_image_dir"], img_path.replace(".jpg", ".png")))

        # Save extra stuff
        successful = np.abs(final_scores[j, opt.target_attribute] - target[j].detach().cpu().float().numpy()) < 0.5
        dump_dict = {
            "successful": successful,
            "initial_scores": initial_scores[j].detach().cpu().float().numpy(),
            "final_scores": final_scores[j],
            "loss_decision": final_loss_decision[j],
            "loss_proxmity": final_loss_proximity[j],
        }

        if opt.save_initial_final_z:
            dump_dict["initial_z"] = style_codes_numpy[j]
            dump_dict["final_z"] = z[j].detach().cpu().float().numpy()

        with open(os.path.join(directories["pkl_dir"], img_path.replace(".jpg", ".pkl")), 'wb') as f:
          pickle.dump(dump_dict, f)

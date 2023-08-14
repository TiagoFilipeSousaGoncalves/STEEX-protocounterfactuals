# Imports
import argparse
import os
import tqdm
from PIL import Image

# PyTorch Imports
from torchvision import transforms



# Command Line Interface
# Create the parser
parser = argparse.ArgumentParser()


# Add the arguments
# Original directory
parser.add_argument("--in_folder", type=str, help="Directory of the original data set (images).")

# New (resized) directory
parser.add_argument("--out_folder", type=str, help="Directory of the resized data set (images).")

# New size (for resized images)
parser.add_argument("--new_size", type=int, default=128, help="New size for the resized images.")



# Parse the arguments
args = parser.parse_args()



# Get arguments
in_folder = args.in_folder
out_folder = args.out_folder
new_size = args.new_size



# Transforms (to resize the images)
t = transforms.Compose([transforms.CenterCrop(150), transforms.Resize((new_size, new_size), Image.BICUBIC)])


# Create out_folder
if not os.path.isdir(out_folder):
    os.makedirs(out_folder)


# Resize images
for filename in tqdm.tqdm(os.listdir(in_folder)):
    img = Image.open(os.path.join(in_folder, filename)).convert('RGB')
    img = t(img)
    img.save(os.path.join(out_folder, filename))

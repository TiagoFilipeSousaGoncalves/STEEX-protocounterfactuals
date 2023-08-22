# Imports
import os
import argparse
import cv2
import numpy as np



# Command Line Interface
# Create the parser
parser = argparse.ArgumentParser()



# Folder base directory
parser.add_argument("--folder_base", default='CelebAMaskHQ-mask-anno', type=str, help="Directory of the original data set (masks).")

# Folder base directory
parser.add_argument("--folder_save", default='CelebAMaskHQ-mask', type=str, help="Directory of the processed data set (masks).")



# Parse the arguments
args = parser.parse_args()


# Important variables
folder_base = args.folder_base
folder_save = args.folder_save
img_num = 30000
label_list = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']


# Create folder if needed
if not os.path.exist(folder_save):
    os.makedirs(folder_save)



# Apply processing (https://github.com/switchablenorms/CelebAMask-HQ/blob/master/face_parsing/Data_preprocessing/g_mask.py)
for k in range(img_num):
    folder_num = k // 2000
    im_base = np.zeros((512, 512))
    for idx, label in enumerate(label_list):
        filename = os.path.join(folder_base, str(folder_num), str(k).rjust(5, '0') + '_' + label + '.png')
        if (os.path.exists(filename)):
            print (label, idx+1)
            im = cv2.imread(filename)
            im = im[:, :, 0]
            im_base[im != 0] = (idx + 1)

    filename_save = os.path.join(folder_save, str(k) + '.png')
    print(filename_save)
    cv2.imwrite(filename_save, im_base)

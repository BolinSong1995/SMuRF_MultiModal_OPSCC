#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 14:51:31 2024

@author: tanmoy
"""

import os
import torch
import pickle

import numpy as np
import torch.backends.cudnn as cudnn

from torch.utils.data.dataloader import DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm.auto import tqdm
# from torchviz import make_dot
from losses import MultiTaskLoss, MMOLoss
from datasets import HandCraftedFeaturesDataset, RadPathDataset, custom_collate
from models import FusionModelBi, Model
from utils import *
from parameters import parse_args
import pandas as pd
from PIL import Image


import numpy as np

import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


from captum.attr import (
    IntegratedGradients,
    LayerIntegratedGradients,
    TokenReferenceBase,
    configure_interpretable_embedding_layer,
    remove_interpretable_embedding_layer,
    visualization
)
from captum.attr._utils.input_layer_wrapper import ModelInputWrapper

#### openslide path



OPENSLIDE_PATH = r'C:\Users\bsong47\openslide-win64-20221217\bin'

if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide




from scipy.ndimage import gaussian_filter

import matplotlib.colors as mcolors





###### arg parse here 

args = parse_args()
root = args.dataroot
device = torch.device('cuda:{}'.format(
  args.gpu_ids[0])) if args.gpu_ids else torch.device('cpu')
print("Using device:", device)


model = Model(args)
model.to(device)


#print("model", model)

path = r'C:\Users\bsong47\OneDrive - Emory University\Documents\code\raptomics\checkpoints\raw_images\fused_attention_multitask_100_0.002_raptomic\fused_attention_multitask_100_0.002_raptomic_best_loss.pt'
# Load the weights from a .pt file
#model.load_state_dict(torch.load(path))

# Load the model onto the same device as the model
#model.load_state_dict(torch.load(path, map_location=device))


# Ensure you are in evaluation mode if you are making predictions
#model.eval()


# Load the entire saved data
checkpoint = torch.load(path)

# Load just the model state dict
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()



######## data preparation here

# Construct the absolute path for the file
# Base directory setup
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
file_path = os.path.join(base_dir, 'data_table_output_test.csv')
# patho_path = os.path.join(base_dir, 'single_data.csv')

print("file path", file_path)
# print("pathology csv path", patho_path)

data = extract_csv(file_path)
# patho_data = extract_csv(patho_path)
####print(data.head())  # Debugging: print first few rows


#print(data.head())
# print("patho_data", patho_data.head())

# print("args.data_root", args.dataroot)


# data = extract_csv(os.path.join(
#     args.dataroot, "data_table.csv"))
labels = data[['grade', "OS", "OS_censor"]]

indices = np.arange(labels.shape[0])

cudnn.deterministic = True
torch.cuda.manual_seed_all(202)
torch.manual_seed(202)
np.random.seed(202)


_, _, _, _, train_index, test_index = train_test_split(
    data, labels, indices, test_size=0.01, random_state=2023, shuffle=False )
#print("length of the train set", len(train_index)) ###stratify=labels['grade_int']
#print("length of the test set", len(test_index))

print("train_index", train_index[:10])

# print(data['pathology_folder_name'][:10])


#### I have to custom wsi path here using args.dataroot

# Assuming args.dataroot is a string path and data is a DataFrame
base_path = os.path.join(args.dataroot, "pathology", "slides")

### format pathology 

print("data type", data['format_pathology'].iloc[0])

###### choose the type(svs or tiff) from the csv file from the train_index

# If train_index is an array or list of indices
pathology_dirs = [os.path.join(base_path, data["format_pathology"].iloc[idx]) for idx in train_index]

#print("pathology_dirs",pathology_dirs )

# Print each directory path
# for dir_path in pathology_dirs:
#     print("path_dir", dir_path)

# Assuming pathology_folder_name is another column in the same DataFrame and you want to create full file paths
pathology_files = [os.path.join(dir_path, data["pathology_folder_name"].iloc[idx]) for idx, dir_path in zip(train_index, pathology_dirs)]



#### read data into openslide

patients_id = 14

print("pathology_files", pathology_files[patients_id])

base_path = pathology_files[patients_id]
extension = data['format_pathology'].iloc[patients_id]

# Ensure the extension starts with a dot
if not extension.startswith('.'):
    extension = '.' + extension

# Construct the full path
slide_path = base_path + extension

print("slide path", slide_path)


slide_path = os.path.abspath(slide_path)




# # Assuming pathology_files[0] is a full path to a file and not a directory
# try:
#     slide = openslide.OpenSlide(slide_path)
#     print("Slide dimensions:", slide.dimensions)
#     print("Level count:", slide.level_count)
#     print("Level dimensions:", slide.level_dimensions)
    
#     # Level index for the dimension (896, 584), determine this index from the level_dimensions
#     # For example, if this dimension is the 7th level (index starts from 0)
#     level_index = 7  # Adjust this index based on the actual order in your slide.level_dimensions

#     # Read the whole level
#     level_dims = slide.level_dimensions[level_index]
#     image = slide.read_region((0, 0), level_index, level_dims)

#     # Convert the image to RGB (discard the alpha channel)
#     original_image = image.convert('RGB')
    
#     print("image shape after dimensions fixed", original_image.size)

#     # ####Display the image using matplotlib
#     # plt.figure(figsize=(10, 6))
#     # plt.imshow(image)
#     # plt.title(f"Slide Level {level_index} Dimensions {level_dims}")
#     # plt.axis('off')  # Hide the axes
#     # plt.show()
# except openslide.OpenSlideUnsupportedFormatError:
#     print("The file is not supported by OpenSlide or is corrupt.")
# except FileNotFoundError:
#     print("The file could not be found. Check the path:", slide_path)
# except Exception as e:
#     print("An error occurred while trying to open the slide:", e)




train_set = RadPathDataset(
    data, args.dataroot, index=train_index)
val_set = RadPathDataset(data, args.dataroot, index=test_index)

# Create a Subset containing only the first index from the training set
first_sample_dataset = torch.utils.data.Subset(train_set, [patients_id])  # Only the select index

print(first_sample_dataset)

train_loader = DataLoader(
first_sample_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate)


# #val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, collate_fn=custom_collate)




####### IG define here 
#from captum.attr import IntegratedGradients



def visualize_attributions_wsi(original_image, attributions):
    print("Entered this loop")

    # Settings for visualization
    smoothing_sigma = 30 #Sigma for Gaussian smoothing
    percentile = 99    # Percentile for thresholding the attributions
    fragment_index = 0    # Assuming you want to visualize the first fragment
    feature_index = 20    # Assuming you want to visualize the first feature 2, (192: original) 

    # Convert attributions from torch.Tensor to numpy.ndarray if necessary
    if isinstance(attributions, torch.Tensor):
        attributions = attributions.cpu().numpy()

    print("Attribution original shape:", attributions.shape)

    # Select the specific feature map from the first fragment
    attribution_map = attributions[fragment_index, feature_index, :, :]
    print("attribution_map shape before processing:", attribution_map.shape)

    # Normalize and convert the attribution map
    if np.any(np.isnan(attribution_map)) or np.any(np.isinf(attribution_map)):
        attribution_map = np.nan_to_num(attribution_map)
        print("NaN or Inf values were present and have been replaced.")

    attribution_map = (attribution_map - np.min(attribution_map)) / (np.max(attribution_map) - np.min(attribution_map))
    attribution_map = (attribution_map * 255).astype(np.uint8)

    # Resize attribution to match the original image size
    print("Original image size:", original_image.size)  # PIL Image size format is (width, height)
    attribution_image = Image.fromarray(attribution_map)
    attribution_image = attribution_image.resize(original_image.size, Image.BILINEAR)
    attribution_map_resized = np.array(attribution_image)
    print("Resized attribution map shape:", attribution_map_resized.shape)

    # Smooth the resized attribution map
    attribution_smoothed = gaussian_filter(attribution_map_resized, sigma=smoothing_sigma)

    # Normalize the smoothed attribution map
    attribution_norm = (attribution_smoothed - np.min(attribution_smoothed)) / (np.max(attribution_smoothed) - np.min(attribution_smoothed))

    # Apply a threshold to highlight areas with the highest attributions
    threshold = np.percentile(attribution_norm, percentile)
    attribution_highlighted = np.where(attribution_norm >= threshold, attribution_norm, 0)
    print("attribution_highlighted shape:", attribution_highlighted.shape)
    
    colors = [(0, 1, 0, 0), (0.5, 1, 0.5, 0.5), (1, 0, 0, 1)]
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)

    # Plotting
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Display original image
    ax[0].imshow(original_image, cmap='gray')
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    # Display normalized attribution overlay
    ax[1].imshow(original_image, cmap='gray')
    img_attrib = ax[1].imshow(attribution_highlighted, cmap=cmap, alpha=0.7)
    ax[1].set_title('Attribution Overlay')
    ax[1].axis('off')

    # Adding color bar specifically to the attribution overlay subplot
    cbar = plt.colorbar(img_attrib, ax=ax[1], orientation='vertical')
    cbar.set_label('Attribution Intensity')

    plt.tight_layout()
    plt.show()














def visualize_attributions_with_example_style(original_image, attributions):
    """
    Function to overlay the most significant attributions on the original image slice, similar to the provided example.
    A specific slice and channel are selected, and the attributions are smoothed and normalized.

    Args:
    - original_image (numpy.ndarray): The original image data with shape [batch_size, depth, channels, height, width].
    - attributions (numpy.ndarray): The attribution data with the same shape as original_image.
    - slice_idx (int): The slice index to be visualized.
    - channel_idx (int): The channel index to be visualized.
    """
    slice_idx = 0
    channel_idx=1
    # Settings for visualization
    smoothing_sigma = 25  # Sigma for Gaussian smoothing
    percentile = 98.5    # Percentile for thresholding the attributions
    # Ensure original_image and attributions are numpy arrays
    if isinstance(original_image, torch.Tensor):
        original_image = original_image.cpu().numpy()
    if isinstance(attributions, torch.Tensor):
        attributions = attributions.cpu().numpy()

    # Select the specific slice and channel
    original_slice = original_image[0, slice_idx, channel_idx, :, :]
    attribution_slice = attributions[0, slice_idx, channel_idx, :, :]

    # Smooth the attribution map
    attribution_smoothed = gaussian_filter(attribution_slice, sigma=smoothing_sigma)

    # Normalize the smoothed attribution map
    attribution_norm = np.clip(attribution_smoothed, 0, np.percentile(attribution_smoothed, percentile))  # Clip outliers
    #attribution_norm = (attribution_smoothed - np.min(attribution_smoothed)) / (np.max(attribution_smoothed) - np.min(attribution_smoothed))
    
    # Apply a threshold to highlight areas with the highest attributions
    #threshold = np.percentile(attribution_norm, percentile)
    #attribution_highlighted = np.where(attribution_norm >= percentile, attribution_norm, 0)

    # Create a custom colormap: green for the lowest and red for the highest intensities
    #colors = [(0, 1, 0, 0), (0, 1,1, 0), (1, 0, 0, 1)]  # RGBA tuples
    colors = [(0, 1, 0, 0), (0.5, 1, 0.5, 0.5), (1, 0, 0, 1)]
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Display original image slice
    axes[0].imshow(original_slice, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Display the attribution overlay with the custom colormap
    axes[1].imshow(original_slice, cmap='gray')
    #img_attrib = axes[1].imshow(attribution_highlighted, cmap=cmap, alpha=0.9)  # Overlay with semi-transparency
    img_attrib = axes[1].imshow(attribution_norm, cmap=cmap, alpha=0.7)  # Overlay with semi-transparency
    axes[1].set_title('Attribution Overlay')
    axes[1].axis('off')

    # Adding color bar specifically to the attribution overlay subplot
    cbar = plt.colorbar(img_attrib, ax=axes[1], orientation='vertical', ticks=[0, 1])
    cbar.set_label('Attribution Intensity')
    #cbar.ax.set_yticklabels(['0', '1'])  # Ensure ticks are 0 and 1

    # Show the plot
    plt.tight_layout()
    plt.show()



#######


# Assuming attributions_mod3 is either a numpy array or a torch tensor
def check_for_nan(attributions):
    if isinstance(attributions, torch.Tensor):
        if torch.isnan(attributions).any():
            print("Attributions contain NaN values.")
        else:
            print("No NaN values in attributions.")
    elif isinstance(attributions, np.ndarray):
        if np.isnan(attributions).any():
            print("Attributions contain NaN values.")
        else:
            print("No NaN values in attributions.")



# Redefine the forward function to accept direct arguments
def forward_func(mod1, mod2, mod3):
    # Calculate the batch size from one of the inputs
    batch_size = mod1.size(0)
    # Assuming the model outputs a tuple (pred_grade, pred_hazard)
    #pred_grade, pred_hazard = model(mod1, mod2, mod3, batch_size)
    output = model(mod1, mod2, mod3, batch_size)
    pred_grade, pred_hazard = output
    print("pred_grade after ig", pred_grade.shape)
    
    print("pred_grade after ig value", pred_grade)
    print("pred_hazard after ig", pred_hazard.shape)
    return pred_hazard  # Ensure we're
    

ig = IntegratedGradients(forward_func)

for i, (mod1, mod2, mod3, grade, time, event, ID) in enumerate(train_loader):
    try:
        #mod1, mod2, mod3 = mod1.to(device), mod2.to(device), mod3.to(device)
        
        #plot_data(mod1, mod2, mod3)
        mod1 = mod1.to(device)   #.requires_grad_(True)
        mod2 = mod2.to(device)   #.requires_grad_(True)
        mod3 = mod3.to(device)   #.requires_grad_(True)
        # Assuming mod3 reshape is handled elsewhere or not needed here
        mod3 = torch.reshape(mod3, (mod3.shape[0]*mod3.shape[1], mod3.shape[2], mod3.shape[3], mod3.shape[4]))

        print("mod1", mod1.shape)
        print("mod2", mod2.shape)
        print("mod3", mod3.shape)
        #plot_data(mod1, mod2, mod3)

        input_tuple = (mod1, mod2, mod3)
        output = model(mod1, mod2, mod3, mod1.shape[0])  # Forward pass includes batch size
        pred_grade, pred_hazard = output
        print(f"Batch {i}: pred_grade shape: {pred_grade.shape}, pred_hazard shape: {pred_hazard.shape}")

        #print(f"Batch {i} output shape: {output.shape}")
        target_index= 0
        #attributions = ig.attribute((mod1, mod2, mod3), target=target_index)  # Assumes target=0 is appropriate
        attributions = ig.attribute((mod1, mod2, mod3),  n_steps=1)  # Assumes target=0 is appropriate
        print("Attributions calculated successfully.")
        print("Attributions mod1 shape.",attributions[0].shape )
        print("Attributions mod2 shape.",attributions[1].shape )
        print("Attributions mod3 shape.",attributions[2].shape )
        # Assuming mod1, mod2, and mod3 are your original images
        original_mod1 = mod1.cpu().detach().numpy()
        original_mod2 = mod2.cpu().detach().numpy()
        original_mod3 = mod3.cpu().detach().numpy()
        
        
        
        # Assuming attributions is a tuple containing attributions for mod1, mod2, and mod3
        attributions_mod1 = attributions[0].cpu().detach().numpy()
        attributions_mod2 = attributions[1].cpu().detach().numpy()
        attributions_mod3 = attributions[2].cpu().detach().numpy()
        
        
        slide = openslide.OpenSlide(slide_path)
        print("Slide dimensions:", slide.dimensions)
        print("Level count:", slide.level_count)
        print("Level dimensions:", slide.level_dimensions)
           
            # Level index for the dimension (896, 584), determine this index from the level_dimensions
            # For example, if this dimension is the 7th level (index starts from 0)
        level_index = 5  # Adjust this index based on the actual order in your slide.level_dimensions

            # Read the whole level
        level_dims = slide.level_dimensions[level_index]
        image = slide.read_region((0, 0), level_index, level_dims)

            # Convert the image to RGB (discard the alpha channel)
        original_image = image.convert('RGB')
           
        print("image shape after dimensions fixed", original_image.size)
        
        check_for_nan(attributions_mod3)
        
        # visualize_attributions_wsi(original_image, attributions_mod3) ### wsi
        visualize_attributions_with_example_style(original_mod1, attributions_mod1)    #### lymph
        visualize_attributions_with_example_style(original_mod2, attributions_mod2)   ### tumour
        
        
        
#         #####Visualize attributions on the original images
#         Assuming you have a function visualize_attributions defined for visualization
#         visualize_attributions(original_mod1, attributions_mod1)
#         visualize_modality_attributions(original_mod1, attributions_mod1, channel_idx=0, depth_idx=0)  # mod3 might have different dimensions
#         visualize_attributions(original_mod2, attributions_mod2)
#         visualize_attributions_with_example_style(original_mod2, attributions_mod2)
        
    except Exception as e:
        print(f"Error during processing batch {i}: {e}")
        
        
        
        























###




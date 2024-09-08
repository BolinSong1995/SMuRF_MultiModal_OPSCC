import os
import sys
sys.path.append(os.path.join(os.getcwd(), '..\\HIPT\\HIPT_4K'))
import matplotlib
from PIL import Image
from hipt_heatmap_utils import *
from hipt_model_utils import get_vit256, get_vit4k, eval_transforms
from hipt_4k import HIPT_4K
import glob
from medpy.io import load, header
import numpy as np
import pandas as pd
import h5py
import pickle
import math
from parameters import mkdirs
import torch
import nrrd



OPENSLIDE_PATH = r'C:\Users\bsong47\openslide-win64-20221217\bin'

if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide


def write_list(L, file):
    # store list in binary file so 'wb' mode
    with open(file, 'wb') as fp:
        pickle.dump(L, fp)
        print('Done writing list into a binary file')


def build_ct_boxes(path):
    mkdirs(os.path.join(path, 'ct_boxes'))
    labels_path = os.path.join(path, "radiology")
    df = pd.read_csv(os.path.join(path, 'data_table.csv'))
    for label_name in ['label.nii.gz', 'node.nii.gz']:
        X_min, X_max, Y_min, Y_max, Z_min, Z_max = [], [], [], [], [], []
        for patient in df["radiology_folder_name"]:
            if os.path.isfile(os.path.join(
                    labels_path, patient, label_name)):
                
                label, _ = load(os.path.join(
                        labels_path, patient, label_name))
                
                x_min, x_max = np.min(np.nonzero(label)[1]), np.max(
                    np.nonzero(label)[1])
                y_min, y_max = np.min(np.nonzero(label)[0]), np.max(
                    np.nonzero(label)[0])
                z_min, z_max = np.min(np.nonzero(label)[2]), np.max(
                    np.nonzero(label)[2])
            else:
                x_min, x_max = 0, 0
                y_min, y_max = 0, 0
                z_min, z_max = 0, 0
            X_min.append(x_min)
            Y_min.append(y_min)
            Z_min.append(z_min)
            X_max.append(x_max)
            Y_max.append(y_max)
            Z_max.append(z_max)
        if label_name == "label.nii.gz":
            df["X_min_tumor"] = X_min
            df["Y_min_tumor"] = Y_min
            df["X_max_tumor"] = X_max
            df["Y_max_tumor"] = Y_max
            df["Z_min_tumor"] = Z_min
            df["Z_max_tumor"] = Z_max

            write_list(X_min, os.path.join(path, 'ct_boxes', "X_min_tumor"))
            write_list(Y_min, os.path.join(path, 'ct_boxes', "Y_min_tumor"))
            write_list(X_max, os.path.join(path, 'ct_boxes', "X_max_tumor"))
            write_list(Y_max, os.path.join(path, 'ct_boxes', "Y_max_tumor"))
            write_list(Z_min, os.path.join(path, 'ct_boxes', "Z_min_tumor"))
            write_list(Z_max, os.path.join(path, 'ct_boxes', "Z_max_tumor"))

        else:
            df["X_min_lymph"] = X_min
            df["Y_min_lymph"] = Y_min
            df["X_max_lymph"] = X_max
            df["Y_max_lymph"] = Y_max
            df["Z_min_lymph"] = Z_min
            df["Z_max_lymph"] = Z_max

            write_list(X_min, os.path.join(path, 'ct_boxes', "X_min_lymph"))
            write_list(Y_min, os.path.join(path, 'ct_boxes', "Y_min_lymph"))
            write_list(X_max, os.path.join(path, 'ct_boxes', "X_max_lymph"))
            write_list(Y_max, os.path.join(path, 'ct_boxes', "Y_max_lymph"))
            write_list(Z_min, os.path.join(path, 'ct_boxes', "Z_min_lymph"))
            write_list(Z_max, os.path.join(path, 'ct_boxes', "Z_max_lymph"))

    df.to_csv(os.path.join(path, '..\\data_table_output.csv'))

    return

path = r"C:\Users\bsong47\OneDrive - Emory University\Documents\code\raptomics\data"

build_ct_boxes(path)


def test(path):
    df = pd.read_csv(os.path.join(path, '..\\table.csv'))
    df_xmin = pd.read_pickle(os.path.join(path, 'X_min_tumor'))
    df_xmax = pd.read_pickle(os.path.join(path, 'X_max_tumor'))
    df_ymin = pd.read_pickle(os.path.join(path, 'Y_min_tumor'))
    df_ymax = pd.read_pickle(os.path.join(path, 'Y_max_tumor'))
    df_zmin = pd.read_pickle(os.path.join(path, 'Z_min_tumor'))
    df_zmax = pd.read_pickle(os.path.join(path, 'Z_max_tumor'))
    df["X_min_tumor"] = df_xmin
    df["Y_min_tumor"] = df_ymin
    df["X_max_tumor"] = df_xmax
    df["Y_max_tumor"] = df_ymax
    df["Z_min_tumor"] = df_zmin
    df["Z_max_tumor"] = df_zmax
    df.to_csv(os.path.join(path, '..\\data_table.csv'))
    return


# test(r"D:\miccai2023\data\ct_boxes")
def find_downscale(image, mask):
    dimensions = image.level_dimensions
    downscale = image.level_downsamples
    L_dist = []
    for i in range(len(dimensions)):
        x = abs(dimensions[i][0] - mask.shape[1])
        y = abs(dimensions[i][1] - mask.shape[0])
        dist = x+y
        L_dist.append(dist)
    index_min = L_dist.index(min(L_dist))
    return int(downscale[index_min])


def patch_embedding(dir_data, dir_hipt_pth, device):

    light_jet = cmap_map(lambda x: x/2 + 0.5, matplotlib.cm.jet)
    pretrained_weights256 = os.path.join(dir_hipt_pth, 'vit256_small_dino.pth')
    pretrained_weights4k = os.path.join(dir_hipt_pth, 'vit4k_xs_dino.pth')
    device256 = torch.device(device)
    device4k = torch.device(device)

    # ViT_256 + ViT_4K loaded independently (used for Attention Heatmaps)
    # model256 = get_vit256(
    #     pretrained_weights=pretrained_weights256, device=device256)
    # model4k = get_vit4k(pretrained_weights=pretrained_weights4k, device=device4k)

    # ViT_256 + ViT_4K loaded into HIPT_4K API
    model = HIPT_4K(pretrained_weights256,
                    pretrained_weights4k, device256, device4k)
    model.eval()

    root_histo = r'C:\Users\bsong47\OneDrive - Emory University\Documents\code\raptomics\data\pathology\slides\ndpi' 

    root_h5 = os.path.join(dir_data, 'patches')
    root_pkl = os.path.join(dir_data, 'embeddings')
    root_mask = r'C:\Users\bsong47\OneDrive - Emory University\Documents\code\raptomics\data\pathology\tumor_annotation'

    mkdirs(root_pkl)
    slide_list = os.listdir(root_histo)

    print("slide_list", slide_list)
    for slide in slide_list:
        if not os.path.isdir(os.path.join(root_pkl, slide.split('.')[0])):
            print(slide)
            mkdirs(os.path.join(root_pkl, os.path.splitext(slide)[0]))

            format = slide.split('.')[-1]
            image = openslide.OpenSlide(os.path.join(root_histo, slide))
            
            if format == 'tiff' or format == 'tif':
                if os.path.isfile(os.path.join(root_mask, slide.split('.')[0] + "_mask.png")):
                    mask = np.array(Image.open(os.path.join(root_mask, slide.split('.')[0] + "_mask.png")))
                    downscale = find_downscale(image, mask)
                    print("ok mask ", slide, "downscale= ", downscale)
                else:
                    mask = None
                    downscale = None
                    print("no mask!! ", slide)
            else:
                if os.path.isfile(os.path.join(root_mask, slide.replace(format, "png"))):
                    mask = np.array(Image.open(os.path.join(
                        root_mask, slide.replace(format, "png"))))
                    downscale = find_downscale(image, mask)
                    print("ok mask ", slide, "downscale= ", downscale)
                else:
                    mask = None
                    downscale = None
                    print("no mask!! ", slide)
            for contour in os.listdir(os.path.join(root_h5, os.path.splitext(slide)[0])):
                h5file = os.path.join(root_h5, os.path.splitext(slide)[0], contour)

                with h5py.File(h5file, "r") as f:
                    # Print all root level object names (aka keys)
                    # these can be group or dataset names
                    # get first object name/key; may or may NOT be a group
                    a_group_key = list(f.keys())[0]

                    # get the object type for a_group_key: usually group or dataset

                    # If a_group_key is a group name,
                    # this gets the object names in the group and returns as a list
                    data = list(f[a_group_key])

                    # If a_group_key is a dataset name,
                    # this gets the dataset values and returns as a list
                    data = list(f[a_group_key])
                    # preferred methods to get dataset values:
                    ds_obj = f[a_group_key]      # returns as a h5py dataset object
                    h5_arr = f[a_group_key][()]  # returns as a numpy array

                data = np.concatenate(
                    (h5_arr, np.zeros((len(h5_arr), 192))), axis=1)
                print(data.shape)
                df = pd.DataFrame(data=data)
                for i in range(len(h5_arr)):
                    if format == 'tiff' or format == 'tif' or format == 'ndpi':
                        region = image.read_region(
                            h5_arr[i], 1, (1024, 1024)).convert('RGB')
                    elif format == 'svs':
                        region = image.read_region(
                            h5_arr[i], 0, (2048, 2048)).convert('RGB')
                        region.thumbnail((1024, 1024))
                    # print('region shape:',region.shape)
                    x = eval_transforms()(region).unsqueeze(dim=0)
                    embedding = model.forward(x)
                    df.iloc[i, 2:] = np.squeeze(embedding.cpu().numpy())
                # dic_handler = open(os.path.join(root_pkl, os.path.splitext(
                #     slide)[0], contour.replace('h5', 'pkl')), 'wb')
                # pickle.dump(dic, dic_handler)
                # dic_handler.close()
                bounding_box = list(
                    map(int, contour.split('.')[0].split('_')))
                if mask is not None:
                    proportion = np.mean(mask[bounding_box[1]//downscale: (bounding_box[1]+bounding_box[3]) //
                                    downscale, bounding_box[0]//downscale: (bounding_box[0]+bounding_box[2])//downscale])/255
                else:
                    proportion = 1
                if proportion > 0.01:
                    print("ok fragment: ", bounding_box, proportion)
                    df.to_pickle(os.path.join(root_pkl, os.path.splitext(
                        slide)[0], contour.replace('h5', 'pkl')))
                else:
                    df.to_pickle(os.path.join(root_pkl, os.path.splitext(
                        slide)[0], "repetition_"+contour.replace('h5', 'pkl')))
                    print("repetition: ", bounding_box, proportion)

    return


def fragment_embeddings(dir_embeddings, size):
    for slide in os.listdir(dir_embeddings):
        print(slide)
        img_stack = {"x_shape": [], "y_shape": [], "img": []}
        for pickle_file in os.listdir(os.path.join(dir_embeddings, slide)):
            if pickle_file[-1] == 'l' and not pickle_file.startswith("repetition"):
                df = pd.read_pickle(os.path.join(
                    dir_embeddings, slide, pickle_file))
                bounding_box = list(
                    map(int, pickle_file.split('.')[0].split('_')))
                img = np.zeros(
                    (bounding_box[2]//size + 1, bounding_box[3]//size + 1, 192))
                for i in range(len(df)):
                    x, y = df.iloc[i, 0], df.iloc[i, 1]
                    x -= bounding_box[0]
                    y -= bounding_box[1]
                    x = int(x//size)
                    y = int(y//size)
                    img[x, y, :] = df.iloc[i, 2:].to_numpy()
                img_stack['x_shape'].append(img.shape[0])
                img_stack['y_shape'].append(img.shape[1])
                img_stack['img'].append(img)
                np.save(os.path.join(dir_embeddings, slide,
                        pickle_file).replace('pkl', 'npy'), img)
        max_x = np.max(img_stack['x_shape'])
        max_y = np.max(img_stack['y_shape'])
        max_shape = np.max([max_x, max_y])
        final_img = np.zeros(
            (len(img_stack['img']), max_shape, max_shape, 192))
        for i, img in enumerate(img_stack['img']):
            final_img[i, (max_shape - img_stack['x_shape'][i])//2:(max_shape - img_stack['x_shape'][i])//2 + img_stack['x_shape'][i],
                      (max_shape - img_stack['y_shape'][i])//2:(max_shape - img_stack['y_shape'][i])//2 + img_stack['y_shape'][i], :] = img
        print(final_img.shape)
        np.save(os.path.join(dir_embeddings, slide,
                "embeddings.npy"), final_img)
    return


def sort_folder(dir):
    list_patients = os.listdir(dir)
    for patient in list_patients:
        if not os.path.isfile(os.path.join(dir, patient)):
            slide = os.listdir(os.path.join(dir, patient))
            if len(slide) > 0:
                slide = slide[0]
                if slide.startswith("Cracolici"):
                    # print(slide)
                    os.rename(os.path.join(dir, patient, slide),
                              os.path.join(dir, patient+'.svs'))
                else:
                    # print(slide)
                    os.rename(os.path.join(dir, patient, slide),
                              os.path.join(dir, slide))


device = torch.device(
    'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
# device = torch.device('cpu')
print(type(device))
print(device)

# dir_data = r'C:\Users\bsong47\OneDrive - Emory University\Documents\code\raptomics\data\pathology\clam_output\tiff'
# dir_hipt_pth = r'C:\Users\bsong47\OneDrive - Emory University\Documents\code\raptomics\HIPT\HIPT_4K\Checkpoints'
# patch_embedding(dir_data, dir_hipt_pth, device)


# dir_data = r'C:\Users\bsong47\OneDrive - Emory University\Documents\code\raptomics\data\pathology\clam_output\tiff'
# dir_embeddings = os.path.join(dir_data, 'embeddings')
# fragment_embeddings(dir_embeddings, 2048)

# dir_data = r'C:\Users\bsong47\OneDrive - Emory University\Documents\code\raptomics\data\pathology\clam_output\ndpi'
# dir_hipt_pth = r'C:\Users\bsong47\OneDrive - Emory University\Documents\code\raptomics\HIPT\HIPT_4K\Checkpoints'
# patch_embedding(dir_data, dir_hipt_pth, device)

# dir_data = r'C:\Users\bsong47\OneDrive - Emory University\Documents\code\raptomics\data\pathology\clam_output\ndpi'
# dir_embeddings = os.path.join(dir_data, 'embeddings')
# fragment_embeddings(dir_embeddings, 2048)


# sort_folder(r"C:\Users\bsong47\Documents\code\raptomics\data\pathology\tumor_annotation")
# folders = list(os.walk(r"C:\Users\bsong47\Documents\code\raptomics\data\pathology\tumor_annotation"))[1:]

# for folder in folders:
#     # folder example: ('FOLDER/3', [], ['file'])
#     if not folder[2]:
#         os.rmdir(folder[0])


"python create_patches_fp.py --source .\slides\tiff --save_dir .\clam_output\tiff  --patch_size 1024 --step_size 1024 --patch_level 1 --seg --patch"
"python create_patches_fp.py --source .\slides\svs --save_dir .\clam_output\svs  --patch_size 2048 --step_size 2048 --patch_level 0 --seg --patch"

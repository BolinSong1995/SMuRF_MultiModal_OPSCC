import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import glob
import os
from math import ceil, floor
from medpy.io import load, header
from models import Model
import utils
import pandas as pd
import matplotlib.pyplot as plt

def custom_collate(data):
    ct_tumor, ct_lymphnodes, pathology, y, time, event, ID = zip(*data)
    max_sizes = (max([path.shape[0] for path in pathology]), max([path.shape[1] for path in pathology]))
    pathology = list(pathology)
    ID = list(ID)  # Convert ID back to a list if needed
    for i in range(len(pathology)):
        pathology[i] = torch.moveaxis(pathology[i], -1,0)
        pad_2d = max_sizes[1] - pathology[i].shape[2]
        pad_3d = max_sizes[0] - pathology[i].shape[1]
        padding = (floor(pad_2d/2), ceil(pad_2d/2), floor(pad_2d/2), ceil(pad_2d/2), floor(pad_3d/2), ceil(pad_3d/2))
        m = torch.nn.ConstantPad3d(padding, 0)
        pathology[i] = m(pathology[i])
        pathology[i] = torch.permute(pathology[i], (1,0,2,3)).float()
    return torch.stack(ct_tumor), torch.stack(ct_lymphnodes), torch.stack(pathology), torch.tensor(y), torch.tensor(time), torch.tensor(event), ID






def custom_collate_pathology(data):
    pathology, y, time, event, ID = zip(*data)
    max_sizes = (max([path.shape[0] for path in pathology]), max([path.shape[1] for path in pathology]))
    pathology = list(pathology)
    ID = list(ID)  # Convert ID back to a list if needed
    for i in range(len(pathology)):
        pathology[i] = torch.moveaxis(pathology[i], -1,0)
        pad_2d = max_sizes[1] - pathology[i].shape[2]
        pad_3d = max_sizes[0] - pathology[i].shape[1]
        padding = (floor(pad_2d/2), ceil(pad_2d/2), floor(pad_2d/2), ceil(pad_2d/2), floor(pad_3d/2), ceil(pad_3d/2))
        m = torch.nn.ConstantPad3d(padding, 0)
        pathology[i] = m(pathology[i])
        pathology[i] = torch.permute(pathology[i], (1,0,2,3)).float()
    return torch.stack(pathology), torch.tensor(y), torch.tensor(time), torch.tensor(event), ID





class HandCraftedFeaturesDataset(Dataset):
    def __init__(
        self, df, index=None, random_noise_sigma=0
    ):
        df = df.copy()
        if index is not None:
            df = df.iloc[index]
        self.mod1 = np.array(
            df[['rad0', 'rad1', 'rad2', 'rad3', 'rad4', 'rad5', 'rad6']]).astype(np.float32)
        self.mod2 = np.array(
            df[['path0', 'path1', 'path2', 'path3', 'path4', 'path5', 'path6']]).astype(np.float32)
        self.y = np.array(df["grade"]).astype(np.float32)
        self.time = np.array(df["DFS"]).astype(np.float32)
        self.event = np.array(df["DFS_censor"]).astype(np.float32)
        self.random_noise_sigma = random_noise_sigma

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):

        features_mod1 = self.mod1[idx]
        features_mod1 += np.random.randn(*features_mod1.shape) * \
            self.random_noise_sigma
        mean1 = np.mean(features_mod1, 0)
        std1 = np.std(features_mod1, 0)
        features_mod1 = (features_mod1 - mean1) / std1
        features_mod2 = self.mod2[idx]
        features_mod2 += np.random.randn(*features_mod2.shape) * \
            self.random_noise_sigma
        mean2 = np.mean(features_mod2, 0)
        std2 = np.std(features_mod2, 0)
        features_mod2 = (features_mod2 - mean2) / std2

        return features_mod1, features_mod2, self.y[idx], self.time[idx], self.event[idx]


class RadPathDataset(Dataset):
    def __init__(
        self, df, root_data, index=None, dim=[128, 128, 3], ring=15
    ):   #### dim=[48, 48, 3]
        self.df = df
        if index is not None:
            df = df.iloc[index]
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5)])
        self.y = np.array(df["grade"]).astype(np.float32)
        self.time = np.array(df["DFS"]).astype(np.float32)
        self.event = np.array(df["DFS_censor"]).astype(np.float32)
        self.ID = np.array(df["radiology_folder_name"])

        self.dim = dim
        self.ring = ring
        self.root_data = root_data

    def __len__(self):
        return len(self.y)

    def get_radiology(self, ct_image, index):
        concat_vols = []
        for location in ['tumor', 'lymph']:
            X_min, X_max, Y_min, Y_max, Z_min, Z_max = np.array(
                self.df["X_min_" + location][index]), np.array(
                self.df["X_max_" + location][index]), np.array(
                self.df["Y_min_" + location][index]), np.array(
                self.df["Y_max_" + location][index]), np.array(
                self.df["Z_min_" + location][index]), np.array(
                self.df["Z_max_" + location][index])
            X_min -= self.ring
            Y_min -= self.ring
            Z_min = max(3, Z_min - self.ring)
            X_max += self.ring
            Y_max += self.ring
            Z_max = min(ct_image.shape[-1]-1, Z_max+ self.ring)

            Z_1, Z_2, Z_3 = Z_min+int((Z_max - Z_min)/4), Z_min + \
                int((Z_max - Z_min)/2), Z_min + \
                int(3*(Z_max - Z_min)/4)

            torch.cuda.manual_seed_all(2031)
            torch.manual_seed(2031)
            np.random.seed(2031)

            if Y_max - int(self.dim[0]/2) > Y_min + int(self.dim[0]/2):
                center_Y = np.random.randint(
                Y_min + int(self.dim[0]/2), Y_max - int(self.dim[0]/2), 4)
            else:
                center_Y = np.random.randint(
                Y_min, Y_max, 4)
            if X_max - int(self.dim[1]/2) > X_min + int(self.dim[1]/2):
                center_X = np.random.randint(
                X_min + int(self.dim[1]/2), X_max - int(self.dim[1]/2), 4)
            else:
                center_X = np.random.randint(
                X_min, X_max, 4)
            
            center1 = [center_Y[0], center_X[0], np.random.randint(Z_min, Z_1+1)]
            center2 = [center_Y[1], center_X[1], np.random.randint(Z_1, Z_2+1)]
            center3 = [center_Y[2], center_X[2], np.random.randint(Z_2, Z_3+1)]
            center4 = [center_Y[3], center_X[3], np.random.randint(Z_3, Z_max)]
            sub_vol1 = self.transforms(
                utils.random_crop(ct_image, self.dim, center1))
            sub_vol2 = self.transforms(
                utils.random_crop(ct_image, self.dim, center2))
            sub_vol3 = self.transforms(
                utils.random_crop(ct_image, self.dim, center3))
            sub_vol4 = self.transforms(
                utils.random_crop(ct_image, self.dim, center4))
            vol = torch.stack(
                (sub_vol1, sub_vol2, sub_vol3, sub_vol4))
            concat_vols.append(vol)
        return concat_vols

    def __getitem__(self, index):
        # print(index)
        # print(self.df["radiology_folder_name"][index])
        ct_image, _ = load(os.path.join(self.root_data, "radiology",
                                        self.df["radiology_folder_name"][index], "CT_img.nii.gz"))
        ct_image = utils.soft_tissue_window(ct_image)
        ct_vol = self.get_radiology(ct_image, index)
        ct_tumor, ct_lymphnodes = ct_vol[0], ct_vol[1]
        
        pathology_file = os.path.join(self.root_data, "pathology", "clam_output", self.df["format_pathology"][index], "embeddings", self.df["pathology_folder_name"][index], "embeddings.npy")
        pathology = np.load(pathology_file)
        pathology = torch.from_numpy(pathology)
        

        return ct_tumor, ct_lymphnodes, pathology, self.y[index], self.time[index], self.event[index], self.ID[index] 

# data = pd.read_csv(os.path.join(
#         r'..\data', "data_table.csv"))
# train_set = RadPathDataset(
#             data, r'..\data', index=None)
# train_loader = DataLoader(train_set, batch_size=1, shuffle=True, collate_fn=custom_collate)
# for i, (mod1, mod2, mod3, grade, time, event) in enumerate(train_loader):
#         print([mod.shape for mod in mod3])




class RadDataset(Dataset):
    def __init__(
        self, df, root_data, index=None, dim=[48, 48, 3], ring=15
    ):
        self.df = df
        if index is not None:
            df = df.iloc[index]
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.5)])
        self.y = np.array(df["grade"]).astype(np.float32)
        self.time = np.array(df["DFS"]).astype(np.float32)
        self.event = np.array(df["DFS_censor"]).astype(np.float32)
        self.ID = np.array(df["radiology_folder_name"])

        self.dim = dim
        self.ring = ring
        self.root_data = root_data

    def __len__(self):
        return len(self.y)

    def get_radiology(self, ct_image, index):
        concat_vols = []
        for location in ['tumor', 'lymph']:
            X_min, X_max, Y_min, Y_max, Z_min, Z_max = np.array(
                self.df["X_min_" + location][index]), np.array(
                self.df["X_max_" + location][index]), np.array(
                self.df["Y_min_" + location][index]), np.array(
                self.df["Y_max_" + location][index]), np.array(
                self.df["Z_min_" + location][index]), np.array(
                self.df["Z_max_" + location][index])
            X_min -= self.ring
            Y_min -= self.ring
            Z_min = max(3, Z_min - self.ring)
            X_max += self.ring
            Y_max += self.ring
            Z_max = min(ct_image.shape[-1]-1, Z_max+ self.ring)

            Z_1, Z_2, Z_3 = Z_min+int((Z_max - Z_min)/4), Z_min + \
                int((Z_max - Z_min)/2), Z_min + \
                int(3*(Z_max - Z_min)/4)

            if Y_max - int(self.dim[0]/2) > Y_min + int(self.dim[0]/2):
                center_Y = np.random.randint(
                Y_min + int(self.dim[0]/2), Y_max - int(self.dim[0]/2), 4)
            else:
                center_Y = np.random.randint(
                Y_min, Y_max, 4)
            if X_max - int(self.dim[1]/2) > X_min + int(self.dim[1]/2):
                center_X = np.random.randint(
                X_min + int(self.dim[1]/2), X_max - int(self.dim[1]/2), 4)
            else:
                center_X = np.random.randint(
                X_min, X_max, 4)
            
            center1 = [center_Y[0], center_X[0], np.random.randint(Z_min, Z_1+1)]
            center2 = [center_Y[1], center_X[1], np.random.randint(Z_1, Z_2+1)]
            center3 = [center_Y[2], center_X[2], np.random.randint(Z_2, Z_3+1)]
            
            center4 = [center_Y[3], center_X[3], np.random.randint(Z_3, Z_max)]
            
            sub_vol1 = self.transforms(
                utils.random_crop(ct_image, self.dim, center1))
            sub_vol2 = self.transforms(
                utils.random_crop(ct_image, self.dim, center2))
            sub_vol3 = self.transforms(
                utils.random_crop(ct_image, self.dim, center3))
            sub_vol4 = self.transforms(
                utils.random_crop(ct_image, self.dim, center4))
            vol = torch.stack(
                (sub_vol1, sub_vol2, sub_vol3, sub_vol4))
            concat_vols.append(vol)
        return concat_vols

    def __getitem__(self, index):
        ct_image, _ = load(os.path.join(self.root_data,"radiology", self.df["radiology_folder_name"][index], "CT_img.nii.gz"))
        ct_image = utils.soft_tissue_window(ct_image)
        ct_vol = self.get_radiology(ct_image, index)
        ct_tumor, ct_lymphnodes = ct_vol[0], ct_vol[1]

        return ct_tumor, ct_lymphnodes, self.y[index], self.time[index], self.event[index], self.ID[index]






class PathDataset(Dataset):
    def __init__(
        self, df, root_data, index=None
        ):
        self.df = df
        if index is not None:
            df = df.iloc[index]
        
        self.y = np.array(df["grade"]).astype(np.float32)
        self.time = np.array(df["OS"]).astype(np.float32)
        self.event = np.array(df["OS_censor"]).astype(np.float32)
        self.ID = np.array(df["radiology_folder_name"])

        self.root_data = root_data

    def __len__(self):
        return len(self.y)


    def __getitem__(self, index):
        # print(index)
        # print(self.df["radiology_folder_name"][index])
                
        pathology_file = os.path.join(self.root_data, "pathology", "clam_output", self.df["format_pathology"][index], "embeddings", self.df["pathology_folder_name"][index], "embeddings.npy")
        pathology = np.load(pathology_file)
        pathology = torch.from_numpy(pathology)
        

        return pathology, self.y[index], self.time[index], self.event[index], self.ID[index] 
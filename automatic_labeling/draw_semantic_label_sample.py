#!/usr/bin/env python
#
# file: $ISIP_EXP/tuh_dpath/exp_0074/scripts/decode.py
#
# revision history:
#  20190925 (TE): first version
#
# usage:
#  python decode.py odir mfile data
#
# arguments:
#  odir: the directory where the hypotheses will be stored
#  mfile: input model file
#  data: the input data list to be decoded
#
# This script decodes data using a simple MLP model.
#------------------------------------------------------------------------------

# import pytorch modules
#
import torch
from tqdm import tqdm

# visualize:
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.style.use('ggplot')
import sys
import os


################ customized parameters #################
################ please modify them based on your dataset #################
DATASET_ODIR = "/home/xzt/data/semantic_lidar_v2/2024-04-04-12-16-41"  # the directory path of the raw data
DATASET_NAME = "train" # select the train, dev, and test 
SEMANTIC_MASK_ODIR = "./output"
# lidar sensor:
POINTS = 1081 # the number of lidar points


################# read dataset ###################
NEW_LINE = "\n"
# for reproducibility, we seed the rng
#
class Semantic2DLidarDataset(torch.utils.data.Dataset):
    def __init__(self, img_path, file_name):
        # initialize the data and labels
        # read the names of image data:
        self.scan_file_names = []
        self.intensity_file_names = []
        self.vel_file_names = []
        self.label_file_names = []
        # parameters:
        self.s_max = 30
        self.s_min = 0
        # open train.txt or dev.txt:
        fp_file = open(img_path+'/'+file_name+'.txt', 'r')

        # for each line of the file:
        for line in fp_file.read().split(NEW_LINE):
            if('.npy' in line): 
                self.scan_file_names.append(img_path+'/scans_lidar/'+line)
                self.intensity_file_names.append(img_path+'/intensities_lidar/'+line)
                self.label_file_names.append(img_path+'/semantic_label/'+line)
        # close txt file:
        fp_file.close()
        self.length = len(self.scan_file_names)

        print("dataset length: ", self.length)


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # get the index of start point:
        scan = np.zeros((1, POINTS))
        intensity = np.zeros((1, POINTS))
        label = np.zeros((1, POINTS))
        
        # get the scan data:
        scan_name = self.scan_file_names[idx]
        scan = np.load(scan_name)

        # get the intensity data:
        intensity_name = self.intensity_file_names[idx]
        intensity = np.load(intensity_name)

        # get the semantic label data:
        label_name = self.label_file_names[idx]
        label = np.load(label_name)
        
        # initialize:
        scan[np.isnan(scan)] = 0.
        scan[np.isinf(scan)] = 0.

        intensity[np.isnan(intensity)] = 0.
        intensity[np.isinf(intensity)] = 0.

        scan[scan >= 15] = 0.

        label[np.isnan(label)] = 0.
        label[np.isinf(label)] = 0.

        # transfer to pytorch tensor:
        scan_tensor = torch.FloatTensor(scan)
        intensity_tensor = torch.FloatTensor(intensity)
        label_tensor =  torch.FloatTensor(label)

        data = {
                'scan': scan_tensor,
                'intensity': intensity_tensor,
                'label': label_tensor,
                }

        return data

#------------------------------------------------------------------------------
#
# the main program starts here
#
#------------------------------------------------------------------------------

# function: main
#
# arguments: none
#
# return: none
#
# This method is the main function.
#
if __name__ == '__main__':
    # input parameters:
    dataset_odir = DATASET_ODIR
    dataset_name = DATASET_NAME
    semantic_mask_odir = SEMANTIC_MASK_ODIR
    # create the folder for the semantic label mask:
    if not os.path.exists(semantic_mask_odir):
        os.makedirs(semantic_mask_odir)

    # read dataset:
    eval_dataset = Semantic2DLidarDataset(dataset_odir, dataset_name)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=1, num_workers=2, \
                                                 shuffle=False, drop_last=True, pin_memory=True)
    
    # for each batch in increments of batch size:
    cnt = 0
    cnt_m = 0
    # get the number of batches (ceiling of train_data/batch_size):
    num_batches = int(len(eval_dataset)/eval_dataloader.batch_size)
    for i, batch in tqdm(enumerate(eval_dataloader), total=num_batches):
        # collect the samples as a batch: 10 timesteps
        if(i % 200 == 0):
            scans = batch['scan']
            scans = scans.detach().cpu().numpy()
            labels = batch['label']
            labels = labels.detach().cpu().numpy()

            # lidar data:
            r = scans.cpu().detach().numpy().reshape(POINTS) 
            theta = np.linspace(-(135*np.pi/180), 135*np.pi/180, POINTS, endpoint='true')

            ## plot semantic label:
            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(1,1,1, projection='polar', facecolor='seashell')
            smap = labels.detach().cpu().reshape(POINTS)

            # add the background label:
            smap = np.insert(smap, -1, 0)
            label_val = np.unique(smap).astype(int)
            print("label_values: ", label_val)

            colors = smap 
            area = 6
            scatter = ax.scatter(theta, r, c=colors, s=area, cmap='nipy_spectral', alpha=0.95, linewidth=10)
            ax.set_xticks(np.linspace(-(135*np.pi/180), 135*np.pi/180, 8, endpoint='true'))
            ax.set_thetamin(-135)
            ax.set_thetamax(135)
            ax.set_yticklabels([])
            # produce a legend with the unique colors from the scatter
            classes = ['Other', 'Chair', 'Door', 'Elevator', 'Person', 'Pillar', 'Sofa', 'Table', 'Trash bin', 'Wall']
            plt.xticks(fontsize=16) 
            plt.yticks(fontsize=16)     
            plt.legend(handles=scatter.legend_elements(num=[j for j in label_val])[0], labels=[classes[j] for j in label_val], bbox_to_anchor=(0.5, -0.08), loc='lower center', fontsize=18)
            ax.grid(False)
            ax.set_theta_offset(np.pi/2)
            
            input_img_name = semantic_mask_odir + "/semantic_mask" + str(i)+ ".png"
            plt.savefig(input_img_name, bbox_inches='tight')
            plt.show()

            print(i)
            

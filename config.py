# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 14:50:58 2021

@author: Ilan Valencius
File to manage parameters for Unet Trainign
"""

# For simple_multi_unet_model.py
num_classifications = 4
img_height = 256
img_width = 256
img_channels = 1

# For predict_multi_test.py
size_x = 128
size_y = 128

train_img_path = "C:\Valencius_Snyder_work\Test_ML_data\patches\images"
train_mask_path = "C:\Valencius_Snyder_work\Test_ML_data\patches\masks"

model_save_name = 'sandstone_50_epochs_catXentropy_acc.hdf5'

# Training params in predict_multi_test.py
batch_size_ = 16
verbose_ = 1
epochs_ = 50
shuffle_ = False

# Test model on large image in predict_multi_test.py
large_img_path = 'C:\Valencius_Snyder_work\Test_ML_data\train_imgs_cropped_768.tif'
lg_height = 256
lg_width = 256
lg_step = 256


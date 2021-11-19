import os
from shutil import copyfile

# Set this to the 'chest_xray/train/NORMAL' directory
train_normal_dir = "DIRECTORY HERE"

dir_contents = os.listdir(train_normal_dir)

for file in dir_contents:
    copyfile(train_normal_dir + file, train_normal_dir + "oversampled0-" + file)
    copyfile(train_normal_dir + file, train_normal_dir + "oversampled1-" + file)


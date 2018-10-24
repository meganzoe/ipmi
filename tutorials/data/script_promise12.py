import numpy as np
import SimpleITK as sitk
import gzip
import shutil
import os


# - download the promise12 data and unzip them into two folders
folder_train = os.path.join(os.getenv("HOME"), 'Scratch/data/promise12/train')
folder_test = os.path.join(os.getenv("HOME"), 'Scratch/data/promise12/test')
folder_npy = os.path.join(os.getenv("HOME"), 'Scratch/data/promise12/npy')
if not os.path.isdir(folder_npy):
    os.mkdir(folder_npy)
folder_gzip = os.path.join(os.getenv("HOME"), 'Scratch/data/promise12/gzip')

# - write down all as npy files
# training data
# files_train = os.listdir(folder_train)
# images_train = [sitk.ReadImage(os.path.join(folder_train, "Case%02d.mhd" % idx)) for idx in range(50)]

# N.B - resample the x-y by half for demo purpose
for idx in range(50):
    image_train = sitk.ReadImage(os.path.join(folder_train, "Case%02d.mhd" % idx))
    np.save(os.path.join(folder_npy, "image_train%02d.npy" % idx), sitk.GetArrayFromImage(image_train)[:, ::2, ::2])  # voxdims = image_train.GetSpacing()  # discard the voxel size information for demo purpose
    label_train = sitk.ReadImage(os.path.join(folder_train, "Case%02d_segmentation.mhd" % idx))
    np.save(os.path.join(folder_npy, "label_train%02d.npy" % idx), sitk.GetArrayFromImage(label_train)[:, ::2, ::2])  # data = sitk.GetArrayFromImage(label_train) == 1

# test data
for idx in range(30):
    image_test = sitk.ReadImage(os.path.join(folder_test, "Case%02d.mhd" % idx))
    np.save(os.path.join(folder_npy, "image_test%02d.npy" % idx), sitk.GetArrayFromImage(image_test)[:, ::2, ::2])


# - compress all files
if not os.path.isdir(folder_gzip):
    os.mkdir(folder_gzip)

files_npy = os.listdir(folder_npy)
for fn in files_npy:
    with open(os.path.join(folder_npy, fn), 'rb') as f_in, \
            gzip.open(os.path.join(folder_gzip, fn+'.gz'), 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

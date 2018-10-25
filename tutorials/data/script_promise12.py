import numpy as np
import SimpleITK as sitk
import gzip
import shutil
import os
import scipy.ndimage as nd


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
crop_percent = 0.1  # one-side
output_size = np.array([32, 128, 128])
for idx in range(50):
    image_train = sitk.ReadImage(os.path.join(folder_train, "Case%02d.mhd" % idx))
    label_train = sitk.ReadImage(os.path.join(folder_train, "Case%02d_segmentation.mhd" % idx))
    # voxdims = image_train.GetSpacing()  # discard the voxel size information for demo purpose

    # cropping
    crop_1side = int(image_train.GetSize()[1] * crop_percent)
    image_data_train = sitk.GetArrayFromImage(image_train)[:, crop_1side:-crop_1side, crop_1side:-crop_1side]
    label_data_train = sitk.GetArrayFromImage(label_train)[:, crop_1side:-crop_1side, crop_1side:-crop_1side]

    # resample - this is just a simplified method using ndimage
    image_data_train = nd.zoom(image_data_train, output_size/image_data_train.shape)
    label_data_train = nd.zoom(label_data_train, output_size/label_data_train.shape)

    if any(image_data_train.shape != output_size):
        print('WARNING - wrong size for resampling!!!')
    if any(label_data_train.shape != output_size):
        print('WARNING - wrong size for resampling!!!')

    # now save
    np.save(os.path.join(folder_npy, "image_train%02d.npy" % idx), image_data_train)
    np.save(os.path.join(folder_npy, "label_train%02d.npy" % idx), label_data_train)

# test data
for idx in range(30):
    image_test = sitk.ReadImage(os.path.join(folder_test, "Case%02d.mhd" % idx))
    crop_1side = int(image_test.GetSize()[1] * crop_percent)
    image_data_test = sitk.GetArrayFromImage(image_test)[:, crop_1side:-crop_1side, crop_1side:-crop_1side]
    image_data_test = nd.zoom(image_data_test, output_size/image_data_test.shape)
    if any(image_data_test.shape != output_size):
        print('WARNING - wrong size for resampling!!!')
    np.save(os.path.join(folder_npy, "image_test%02d.npy" % idx), image_data_test)


# - compress all files
if not os.path.isdir(folder_gzip):
    os.mkdir(folder_gzip)

files_npy = os.listdir(folder_npy)
for fn in files_npy:
    with open(os.path.join(folder_npy, fn), 'rb') as f_in, \
            gzip.open(os.path.join(folder_gzip, fn+'.gz'), 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

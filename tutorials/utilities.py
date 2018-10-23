import numpy as np
import nibabel as nib
import os


class DataReader:

    def __init__(self, folder_name):
        self.folder_name = folder_name
        self.files = os.listdir(folder_name)
        self.files.sort()
        self.num_data = len(self.files)

        self.file_objects = [nib.load(os.path.join(folder_name, self.files[i])) for i in range(self.num_data)]
        self.num_labels = [self.file_objects[i].shape[3] if len(self.file_objects[i].shape) == 4
                           else 1
                           for i in range(self.num_data)]

        self.data_shape = list(self.file_objects[0].shape[0:3])

    def get_num_labels(self, case_indices):
        return [self.num_labels[i] for i in case_indices]

    def get_data(self, case_indices=None, label_indices=None):
        if case_indices is None:
            case_indices = range(self.num_data)
        # todo: check the supplied label_indices smaller than num_labels
        if label_indices is None:  # e.g. images only
            data = [np.asarray(self.file_objects[i].dataobj) for i in case_indices]
        else:
            if len(label_indices) == 1:
                label_indices *= self.num_data
            data = [self.file_objects[i].dataobj[..., j] if self.num_labels[i] > 1
                    else np.asarray(self.file_objects[i].dataobj)
                    for (i, j) in zip(case_indices, label_indices)]
        return np.expand_dims(np.stack(data, axis=0), axis=4)


def write_images(input_, file_path=None, file_prefix=''):
    if file_path is not None:
        batch_size = input_.shape[0]
        affine = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]
        [nib.save(nib.Nifti1Image(input_[idx, ...], affine),
                  os.path.join(file_path,
                               file_prefix + '%s.nii' % idx))
         for idx in range(batch_size)]


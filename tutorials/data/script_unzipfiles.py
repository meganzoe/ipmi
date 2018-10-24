import gzip
import shutil
import os

folder_npy = "./promise12-data/npy"  # change this to where you want to save npy files
folder_gzip = "./promise12-data/gzip"  # change this to where the gzip files are

if not os.path.isdir(folder_npy):
    os.mkdir(folder_npy)
files_gz = os.listdir(folder_gzip)

for fn in files_gz:
    with gzip.open(os.path.join(folder_gzip, fn), 'rb') as f_in, open(os.path.join(folder_npy, fn[:-3]), 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
import gzip
import shutil
import zipfile
import os


def check_and_unzip(path_to_zip_file=None, path_to_output_folder=None):

	# inputs
	if path_to_zip_file is None:
		path_to_zip_file = "~/Downloads/promise12-data.zip"
	if path_to_output_folder is None:
		path_to_output_folder = "./data"

	path_to_zip_file = os.path.abspath(os.path.expanduser(path_to_zip_file))
	path_to_output_folder = os.path.abspath(os.path.expanduser(path_to_output_folder))

	# check the output folder
	if not os.path.isdir(path_to_output_folder):
		os.mkdir(path_to_output_folder)
	else:  # do thorough check files
		flag_train = True
		for idx in range(50):
			flag_train &= os.path.isfile(os.path.join(path_to_output_folder, "image_train%02d.npy" % idx))
			flag_train &= os.path.isfile(os.path.join(path_to_output_folder, "label_train%02d.npy" % idx))
		# if not flag_train: print('WARNING: Not all training data can be found.')
		flag_test = True
		for idx in range(30):
			flag_test &= os.path.isfile(os.path.join(path_to_output_folder, "image_test%02d.npy" % idx))
		# if not flag_test: print('WARNING: Not all test data can be found.')

		if flag_train & flag_test:
			print('Data checked at: %s' % path_to_output_folder)
			return path_to_output_folder

	# unzip the file
	path_to_temp_folder = os.path.join(path_to_output_folder, '.~temp_data')
	if not os.path.isdir(path_to_temp_folder):
		os.mkdir(path_to_temp_folder)

	zip_f = zipfile.ZipFile(path_to_zip_file, 'r')
	zip_f.extractall(path_to_temp_folder)
	zip_f.close()

	# find out the gzip folder
	_, filename_zip = os.path.split(path_to_zip_file)
	folder_gzip = os.path.join(path_to_temp_folder, os.path.splitext(filename_zip)[0], 'gzip')

	# unzip individual files
	files_gz = os.listdir(folder_gzip)
	for fn in files_gz:
		with gzip.open(os.path.join(folder_gzip, fn), 'rb') as f_in, \
				open(os.path.join(path_to_output_folder, fn[:-3]), 'wb') as f_out:
			shutil.copyfileobj(f_in, f_out)

	# remove the temp folder
	shutil.rmtree(path_to_temp_folder)

	print('Data saved successfully at: %s' % path_to_output_folder)
	return path_to_output_folder


if __name__ == '__main__':
	check_and_unzip()

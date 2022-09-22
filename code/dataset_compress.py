# load, split and scale the maps dataset ready for training
from os import listdir
from numpy import asarray
from numpy import vstack
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import savez_compressed
 
# load all images in a directory into memory
def load_images(path, size=(256,256)):
	src_list, tar_list = list(), list()
	# enumerate filenames in directory, assume all are images
	path1 = path + 'trainA/'
	for filename in listdir(path1):
		# load and resize the image
		pixels_src = load_img(path1 + filename, target_size=size)
		pixels_des = load_img(path + 'trainB/' + filename, target_size=size)
		# pixels = load_img(path + filename, target_size=size)
		# # convert to numpy array
		pixels_src = img_to_array(pixels_src)
		pixels_des = img_to_array(pixels_des)

		# # split into satellite and map
		sat_img, map_img = pixels_src, pixels_des
		src_list.append(sat_img)
		tar_list.append(map_img)
		# print(path1, filename)
		# print(path+'trainB/', filename)
	return [asarray(src_list), asarray(tar_list)]
 
# dataset path
path = '/media/kishank/Disk 2/Bhavana/Pix2Pix_GAN/dataset_Pix2Pix/'
#load_images(path1)
# path = 'maps/train/'
# load dataset
[src_images, tar_images] = load_images(path)
print('Loaded: ', src_images.shape, tar_images.shape)
# save as compressed numpy array
filename = 'faces_256.npz'
savez_compressed(filename, src_images, tar_images)
print('Saved dataset: ', filename)
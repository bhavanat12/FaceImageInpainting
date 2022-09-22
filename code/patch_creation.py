import glob
from os import listdir
from numpy import asarray
from numpy import vstack
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import savez_compressed

def load_images():
	path = "/home/bhavana/Train_input/wsq/*jpg"
	src_list, tar_list = list(), list()
	for filename in glob.glob(path):
		pixels = load_img(filename)
		pixels = img_to_array(pixels)
		input_image = pixels[:,:]
		filename.replace("Train_input/wsq","Cropped_images")
		pixels1 = load_img(filename)
		pixels1 = img_to_array(pixels)
		target_image = pixels[:,:]
		src_list.append(pixels)
		tar_list.append(pixels1)
		print("*")
	print("_______wsq_______")
		
	path = "/home/bhavana/Train_input/bsq/*jpg"
	for filename in glob.glob(path):
		pixels = load_img(filename)
		pixels = img_to_array(pixels)
		input_image = pixels[:,:]
		filename.replace("Train_input/bsq","Cropped_images")
		pixels1 = load_img(filename)
		pixels1 = img_to_array(pixels)
		target_image = pixels[:,:]
		src_list.append(pixels)
		tar_list.append(pixels1)
		print("*")
	print("_______bsq_______")
		
	path = "/home/bhavana/Train_input/wtr/*jpg"
	for filename in glob.glob(path):
		pixels = load_img(filename)
		pixels = img_to_array(pixels)
		input_image = pixels[:,:]
		filename.replace("Train_input/wtr","Cropped_images")
		pixels1 = load_img(filename)
		pixels1 = img_to_array(pixels)
		target_image = pixels[:,:]
		src_list.append(pixels)
		tar_list.append(pixels1)
		print("*")
	print("_______wtr_______")
		
	path = "/home/bhavana/Train_input/btr/*jpg"
	for filename in glob.glob(path):
		pixels = load_img(filename)
		pixels = img_to_array(pixels)
		input_image = pixels[:,:]
		filename.replace("Train_input/btr","Cropped_images")
		pixels1 = load_img(filename)
		pixels1 = img_to_array(pixels)
		target_image = pixels[:,:]
		src_list.append(pixels)
		tar_list.append(pixels1)
		print("*")
	print("_______btr_______")
		
	path = "/home/bhavana/Train_input/wci/*jpg"
	for filename in glob.glob(path):
		pixels = load_img(filename)
		pixels = img_to_array(pixels)
		input_image = pixels[:,:]
		filename.replace("Train_input/wci","Cropped_images")
		pixels1 = load_img(filename)
		pixels1 = img_to_array(pixels)
		target_image = pixels[:,:]
		src_list.append(pixels)
		tar_list.append(pixels1)
		print("*")
	print("_______wci_______")
		
	path = "/home/bhavana/Train_input/bci/*jpg"
	for filename in glob.glob(path):
		pixels = load_img(filename)
		pixels = img_to_array(pixels)
		input_image = pixels[:,:]
		filename.replace("Train_input/bci","Cropped_images")
		pixels1 = load_img(filename)
		pixels1 = img_to_array(pixels)
		target_image = pixels[:,:]
		src_list.append(pixels)
		tar_list.append(pixels1)
		print("*")
	print("_______bci_______")
		
	path = "/home/bhavana/Train_input/wel/*jpg"
	for filename in glob.glob(path):
		pixels = load_img(filename)
		pixels = img_to_array(pixels)
		input_image = pixels[:,:]
		filename.replace("Train_input/wel","Cropped_images")
		pixels1 = load_img(filename)
		pixels1 = img_to_array(pixels)
		target_image = pixels[:,:]
		src_list.append(pixels)
		tar_list.append(pixels1)
		print("*")
	print("_______wel_______")
		
	path = "/home/bhavana/Train_input/bel/*jpg"
	for filename in glob.glob(path):
		pixels = load_img(filename)
		pixels = img_to_array(pixels)
		input_image = pixels[:,:]
		filename.replace("Train_input/bel","Cropped_images")
		pixels1 = load_img(filename)
		pixels1 = img_to_array(pixels)
		target_image = pixels[:,:]
		src_list.append(pixels)
		tar_list.append(pixels1)
		print("*")
	print("_______bel_______")
		
	path = "/home/bhavana/Train_input/tsq/*jpg"
	for filename in glob.glob(path):
		pixels = load_img(filename)
		pixels = img_to_array(pixels)
		input_image = pixels[:,:]
		filename.replace("Train_input/tsq","Cropped_images")
		pixels1 = load_img(filename)
		pixels1 = img_to_array(pixels)
		target_image = pixels[:,:]
		src_list.append(pixels)
		tar_list.append(pixels1)
		print("*")
	print("_______tsq_______")
		
		
	path = "/home/bhavana/Train_input/ttr/*jpg"
	for filename in glob.glob(path):
		pixels = load_img(filename)
		pixels = img_to_array(pixels)
		input_image = pixels[:,:]
		filename.replace("Train_input/ttr","Cropped_images")
		pixels1 = load_img(filename)
		pixels1 = img_to_array(pixels)
		target_image = pixels[:,:]
		src_list.append(pixels)
		tar_list.append(pixels1)
		print("*")
	print("_______ttr_______")
		
	path = "/home/bhavana/Train_input/tel/*jpg"
	for filename in glob.glob(path):
		pixels = load_img(filename)
		pixels = img_to_array(pixels)
		input_image = pixels[:,:]
		filename.replace("Train_input/tel","Cropped_images")
		pixels1 = load_img(filename)
		pixels1 = img_to_array(pixels)
		target_image = pixels[:,:]
		src_list.append(pixels)
		tar_list.append(pixels1)
		print("*")
	print("_______tel_______")
		
	path = "/home/bhavana/Train_input/tel1/*jpg"
	for filename in glob.glob(path):
		pixels = load_img(filename)
		pixels = img_to_array(pixels)
		input_image = pixels[:,:]
		filename.replace("Train_input/tel1","Cropped_images")
		pixels1 = load_img(filename)
		pixels1 = img_to_array(pixels)
		target_image = pixels[:,:]
		src_list.append(pixels)
		tar_list.append(pixels1)
		print("*")
	print("_______tel1_______")
		
	path = "/home/bhavana/Train_input/tsq1/*jpg"
	for filename in glob.glob(path):
		pixels = load_img(filename)
		pixels = img_to_array(pixels)
		input_image = pixels[:,:]
		filename.replace("Train_input/tsq1","Cropped_images")
		pixels1 = load_img(filename)
		pixels1 = img_to_array(pixels)
		target_image = pixels[:,:]
		src_list.append(pixels)
		tar_list.append(pixels1)
		print("*")
	print("_______tsq1_______")
	
	path = "/home/bhavana/Train_input/bci1/*jpg"
	for filename in glob.glob(path):
		pixels = load_img(filename)
		pixels = img_to_array(pixels)
		input_image = pixels[:,:]
		filename.replace("Train_input/bci1","Cropped_images")
		pixels1 = load_img(filename)
		pixels1 = img_to_array(pixels)
		target_image = pixels[:,:]
		src_list.append(pixels)
		tar_list.append(pixels1)
		print("*")
	print("_______bci1_______")
		
	path = "/home/bhavana/Train_input/bel1/*jpg"
	for filename in glob.glob(path):
		pixels = load_img(filename)
		pixels = img_to_array(pixels)
		input_image = pixels[:,:]
		filename.replace("Train_input/bel1","Cropped_images")
		pixels1 = load_img(filename)
		pixels1 = img_to_array(pixels)
		target_image = pixels[:,:]
		src_list.append(pixels)
		tar_list.append(pixels1)
		print("*")
	print("_______bel1_______")
	
	path = "/home/bhavana/Train_input/bsq1/*jpg"
	for filename in glob.glob(path):
		pixels = load_img(filename)
		pixels = img_to_array(pixels)
		input_image = pixels[:,:]
		filename.replace("Train_input/bsq1","Cropped_images")
		pixels1 = load_img(filename)
		pixels1 = img_to_array(pixels)
		target_image = pixels[:,:]
		src_list.append(pixels)
		tar_list.append(pixels1)
		print("*")
	print("_______bsq1_______")
	
	path = "/home/bhavana/Train_input/wci1/*jpg"
	for filename in glob.glob(path):
		pixels = load_img(filename)
		pixels = img_to_array(pixels)
		input_image = pixels[:,:]
		filename.replace("Train_input/wci1","Cropped_images")
		pixels1 = load_img(filename)
		pixels1 = img_to_array(pixels)
		target_image = pixels[:,:]
		src_list.append(pixels)
		tar_list.append(pixels1)
		print("*")
	print("_______wci1_______")
		
	path = "/home/bhavana/Train_input/wel1/*jpg"
	for filename in glob.glob(path):
		pixels = load_img(filename)
		pixels = img_to_array(pixels)
		input_image = pixels[:,:]
		filename.replace("Train_input/wel1","Cropped_images")
		pixels1 = load_img(filename)
		pixels1 = img_to_array(pixels)
		target_image = pixels[:,:]
		src_list.append(pixels)
		tar_list.append(pixels1)
		print("*")
	print("_______wel1_______")
	
	path = "/home/bhavana/Train_input/wsq1/*jpg"
	for filename in glob.glob(path):
		pixels = load_img(filename)
		pixels = img_to_array(pixels)
		input_image = pixels[:,:]
		filename.replace("Train_input/wsq1","Cropped_images")
		pixels1 = load_img(filename)
		pixels1 = img_to_array(pixels)
		target_image = pixels[:,:]
		src_list.append(pixels)
		tar_list.append(pixels1)
		print("*")
	print("_______wsq1_______")
	return [asarray(src_list), asarray(tar_list)]
	

[src_images, tar_images] = load_images()
print('Loaded:', src_images.shape, tar_images.shape)
filename = 'faces_140.npz'
savez_compressed(filename, src_images, tar_images)
print('Saved dataset: ', filename)

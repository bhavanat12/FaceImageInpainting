import cv2
import os
import glob
import random
import tensorflow

from matplotlib import pyplot
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle

from PIL import Image
from resizeimage import resizeimage

from mtcnn.mtcnn import MTCNN

# draw an image with detected objects
def draw_image_with_boxes(filename, result_list):
	# load the image
	data = pyplot.imread(filename)
	# plot the image
	pyplot.imshow(data)
	# get the context for drawing boxes
	ax = pyplot.gca()
	# plot each box
	for result in result_list:
		# get coordinates
		x, y, width, height = result['box']
		# create the shape
		rect = Rectangle((x, y), width, height, fill=False, color='red')
		# draw the box
		ax.add_patch(rect)
	# show the plot
	pyplot.show()


images = glob.glob("/home/bhavana/img_align_celeba/*.jpg")
#images=glob.glob("/content/Data_temp/*.jpg")
var = "/home/bhavana/img_align_celeba/*.jpg"
count = 1
for image in images:
    image_temp = str(image)
    if count%100 == 0:
      print("*********", count, "**********")
    print(image)
    count=count+1
    pixels = pyplot.imread(image)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    faces = detector.detect_faces(pixels)
    if len(faces) != 0:
      # display faces on the original image
      x1, y1, width, height = faces[0]['box']
      x2, y2 = x1 + width, y1 + height
      # extract face
      try:
        face = pixels[y1:y2, x1:x2]
        # pyplot.subplot(1, len(faces), 0+1)
        # pyplot.axis('off')
        # plot face
        temp = pixels[y1:y2, x1:x2]
        rgbImg = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB)
        cv2.imwrite("/home/bhavana/Jahnavi/a.jpg", rgbImg)
        with open('/home/bhavana/Jahnavi/a.jpg', 'r+b') as f:
          with Image.open(f) as image:
              cover = resizeimage.resize_cover(image, [140, 140], validate=False)
              cover.save('/home/bhavana/Cropped_images/{}'.format(image_temp[31:]), image.format)
      except:
        pass

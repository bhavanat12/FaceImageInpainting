# Project name: "Face Image inpainting using GANs"

Guide: Dr. Himangshu Sarma


Team:

1. Bhavana Talluri
2. Sai Jahnavi Shanvitha Pasumarthy
3. Lavanya konda




## Instructions:

- The Slides folder contains the presentation for second evaluation both in .pptx and .pdf format.

- The lib folder contains the scripts necessary for incorporating attention mechanism to Pix2Pix GAN. The files and functions from this folder are imported into the training file present in the code directory.

- In the code folder, the files are as follows:

a) "train_pix2pix.py" - used for training

b) "dataset_compress.py" - used to get the dataset into .npz format before feeding it into the network.

c) "mtcnn_crop.py" - used to perform tight cropping of the face images using mtcnn algorithm.

d) "patch_creation.py" - used to draw patches of various shapes, sizes, textures and colours. 

e) "textures" - This folder contains the textures that are randomly selected while filling the generated patch on the face image.

4. To execute the code:

a) First compress the dataset using the "dataset_compress.py" file

b) Change the path to this dataset in the "train_pix2pix.py" file

c) Run the command "python3 train_pix2pix.py"

d) The weights are saved after every 10 epochs, which can be used for inference using the predict funtion.

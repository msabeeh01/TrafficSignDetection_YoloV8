import os
import random
import pandas as pd
from PIL import Image
import cv2
from ultralytics import YOLO
# from IPython.display import Video
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
import glob
from tqdm.notebook import trange, tqdm
import warnings
warnings.filterwarnings('ignore')

# How the seaborn plots will look
sns.set_theme(rc={'axes.facecolor': '#f9f9f9'}, style='darkgrid')

# Location of training images
imageDir = './archive/car/train/images'

#Set sample size and load image files
num_samples = 9
image_files = os.listdir(imageDir)

#Randomly select "num_sample" images from the training set
rand_images = random.sample(image_files, num_samples)

fig, axes = plt.subplots(3,3, figsize=(11,11))

#For all the images selected, plot them
for i in range(num_samples):
    # find the corresponding image file to i 
    image = rand_images[i]
    ax = axes[i//3, i%3]
    ax.imshow(plt.imread(os.path.join(imageDir, image)))
    ax.set_title(f'Image {i+1}')
    ax.axis('off')

plt.tight_layout()
plt.show()
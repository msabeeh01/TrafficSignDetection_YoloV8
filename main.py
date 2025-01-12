import os
import random
import pandas as pd
from PIL import Image
import cv2
from ultralytics import YOLO
from IPython.display import Video
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

def main():
    # display_images()
    pretrained_model()

def display_images():
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
    return plt.show()


def imagesize():
    # Get Size of image
    image = cv2.imread("./archive/car/train/images/00000_00000_00012_png.rf.23f94508dba03ef2f8bd187da2ec9c26.jpg")
    h, w, c = image.shape
    print(f"Image has dimensions {w}x{h} and {c}")

def pretrained_model():
    #Using a pretrained YOLO model
    model = YOLO("yolo11n.pt")

    #Detect object using YOLO
    image = "./archive/car/train/images/FisheyeCamera_1_00228_png.rf.e7c43ee9b922f7b2327b8a00ccf46a4c.jpg"
    result_predict = model.predict(source = image, imgsz=(640))

    #show results
    plot = result_predict[0].plot()
    plot = cv2.cvtColor(plot, cv2.COLOR_BGR2RGB)
    plt.imshow(Image.fromarray(plot))
    plt.show()



if __name__ == "__main__":
    main()
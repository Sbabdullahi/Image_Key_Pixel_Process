from roboflow import Roboflow
import numpy as np
import scipy.io
import pathlib
import imageio
from PIL import Image
import glob
#import splitfolders
import os
import shutil
from matplotlib import pyplot as plt


# The copy of the proposed iSDS model deployed in the Roboflow
rf = Roboflow(api_key="DKHqwUbtZeOxS1wCFZ1M")
project = rf.workspace().project("hard-hat-sample-dd8no")
model = project.version(3).model

# load images
image_list = []

for filename in glob.glob(
        'E:/Arabic_Sign_language_Recognition/ArSL_images/datasets/train/images/*.jpg'):  # assuming jpg
    im = Image.open(filename)


# infer on a local image
# You can change the confidence score and the overlap
print(model.predict("1_21_M_ain_0.jpg", confidence=40, overlap=30).json())

# visualize your prediction
model.predict("1_21_M_ain_0.jpg", confidence=40, overlap=30).save("prediction.jpg")


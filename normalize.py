import os, sys
from PIL import Image
import os

size = 512, 512

path_1 = "C:\\Users\\lashi\\Desktop\\bones\\crop_imgs_test\\big\\"
path_2 = "C:\\Users\\lashi\\Desktop\\bones\\crop_imgs_train\\big\\"
directory = os.fsencode(path_1)
directory1 = os.fsencode(path_2)
for f in os.listdir(directory):
    fn = os.fsdecode(f) + "\\"
    di = os.fsencode(path_1 + fn)
    for file in os.listdir(di):
        filename = os.fsdecode(file)
        im = Image.open(path_1 + fn + filename)
        im = im.resize(size)
        os.remove(path_1 + fn + filename)
        im.save(path_1 + fn + filename)
for f in os.listdir(directory1):
    fn = os.fsdecode(f) + "\\"
    di = os.fsencode(path_2 + fn)
    for file in os.listdir(di):
        filename = os.fsdecode(file)
        im = Image.open(path_2 + fn + filename)
        im = im.resize(size)
        os.remove(path_2 + fn + filename)
        im.save(path_2 + fn + filename)
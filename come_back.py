import os
from random import randint
from PIL import Image

path_1 = 'C:\\Users\\lashi\\Desktop\\bones\\images\\'
path_tr = "C:\\Users\\lashi\\Desktop\\bones\\img_data\\train\\"
path_val = "C:\\Users\\lashi\\Desktop\\bones\\img_data\\val\\"

directory = os.fsencode(path_tr)
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    os.rename(path_tr + str(filename), path_1 + str(filename))
directory = os.fsencode(path_val)
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    os.rename(path_val + str(filename), path_1 + str(filename))

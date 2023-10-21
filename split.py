import os
from random import randint

path_1 = 'C:\\Users\\lashi\\Desktop\\bones\\'
path_tr = "C:\\Users\\lashi\\Desktop\\bones\\train\\"
path_val = "C:\\Users\\lashi\\Desktop\\bones\\val\\"

directory = os.fsencode(path_1 + "images")
cnt = 0
for file in os.listdir(directory):
     cnt += 1
     filename = os.fsdecode(file)
     if (randint(1, 11) >= 3):
          os.rename(path_1 + "images\\" + str(filename), path_tr + "images\\" + f"{cnt}.jpg")
          os.rename(path_1 + "labels\\" + str(filename[:-4]) + ".txt", path_tr + "labels\\" + f"{cnt}.txt")
     else:
          os.rename(path_1 + "images\\" + str(filename), path_val + "images\\" + f"{cnt}.jpg")
          os.rename(path_1 + "labels\\" + str(filename[:-4]) + ".txt", path_val + "labels\\" + f"{cnt}.txt")
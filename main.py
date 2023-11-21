from ultralytics import YOLO
import os

path = 'C:\\Users\\lashi\\Desktop\\bones\\images'

model = YOLO('runs\\detect\\train2\\weights\\best.pt')

path_tr = "C:\\Users\\lashi\\Desktop\\bones\\train\\images"
path_val = "C:\\Users\\lashi\\Desktop\\bones\\val\\images"
dir_tr = os.fsdecode(path_tr)
dir_val = os.fsdecode(path_val)
arr = []
for file in os.listdir(dir_tr):
    filename = os.fsdecode(file)
    arr.append(path_tr + "\\" + filename)
for file in os.listdir(dir_val):
    filename = os.fsdecode(file)
    arr.append(path_val + filename)
#results = model.train(data='data.yaml', epochs=500, imgsz=640, patience=0)
for n in arr:
    model.predict(n, save=True, imgsz=320, conf=0.5, save_crop=True)

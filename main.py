from ultralytics import YOLO
import torch


path = 'C:\\Users\\lashi\\Desktop\\bones\\images'

model = YOLO('runs\\detect\\train2\\weights\\best.pt')
model.predict('C:\\Users\\lashi\\Desktop\\bones\\train\\images\\4.jpg', save=True, imgsz=320, conf=0.5, save_crop=True)
#results = model.train(data='data.yaml', epochs=500, imgsz=640, patience=0)
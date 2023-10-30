import os 
import subprocess
import shutil

class_name = input("enter the class name: ")
data = "dataset-seg.yaml"
with open(data, "r+b") as file:
    file.seek(-5, os.SEEK_END)
    
    while file.read(1) != b'\n':
        file.seek(-2, os.SEEK_CUR)
    last_line = file.readline().decode()

    
with open(data, "a") as file:
    if last_line=="names:\n":
        file.write(f"   0: {class_name}\n")
        weights = "yolov5/yolov5s-seg.pt"
        if not os.path.exists("weights"):
            os.makedirs("weights")
    else:
        ind = int(last_line[3])
        file.write(f"   {ind+1}: {class_name}\n")
        weights = "weights/best.pt"
    


os.system('{} {}'.format('python', f"yolov5/segment/train.py --img 640 --epochs 1 --data {data} --weights {weights}"))
if os.path.exists("yolov5/runs/train-seg/exp"):
    shutil.copyfile("yolov5/runs/train-seg/exp/weights/best.pt", "weights/best.pt")
    shutil.rmtree("yolov5/runs/train-seg/exp")
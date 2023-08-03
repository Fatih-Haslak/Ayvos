import torch
import cv2
import pandas as pd
import numpy as np
import warnings
import time
import os

current_path = os.getcwd()
print("O anki çalışma dizini:", current_path)
# new_path = "./tools/yolov7/"

# os.chdir(new_path)
# new_path = "./tools/yolov7"
# current_path = os.getcwd()
# print("O anki çalışma dizini:", current_path)
from hubconf import custom
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
import torch
warnings.filterwarnings('ignore')
from models.experimental import attempt_load
from utils.torch_utils import select_device

class ObjectDetector():
    def __init__(self, model_path="/home/fatih/byterack/ByteTrack/tools/yolov7/yolov7x.pt",src=None):
        torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
        self.device = select_device()
        self.model = torch.hub.load('WongKinYiu/yolov7', 'custom', model_path,
                                   force_reload=True, trust_repo=True)
           
    def detect_objects(self, frame):

        results = self.model(self.frame)
        data = results.pandas().xyxy[0]
        data = data.to_numpy()
        return data

    def draw_bounding_box(self, frame, box):
        start_point = (int(box[0]), int(box[1]))
        end_point = (int(box[2]), int(box[3]))
        start_point_putText = (int(box[0] - 5), int(box[1] - 5))
        cv2.rectangle(self.frame, start_point, end_point, (0, 255, 0), 2)
        cv2.putText(self.frame, str(box[6]), start_point_putText, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)
        

    def string_parser(self,data):
        lines = data.strip().split("\n")
        objects = []
        try:
            for line in lines:
                parts = line.strip().split()
                obj = []
                for i in range(0, len(parts), 7):
                    x1, y1, x2, y2, score, class_id, class_name = parts[i:i+7]
                    obj.append({
                        'x1': x1,
                        'y1': y1,
                        'x2': x2,
                        'y2': y2,
                        'score': score,
                        'class_id': class_id,
                        'class_name': class_name.strip("'")
                    })
                objects.append(obj)
        except:
            print("HATALI VERİ",self.veri)         

        count=0
        car_count = sum(len(obj) for obj in objects)
        arr = np.ones((car_count, 6))
        for i in objects:
            if( i[0]["class_name"]=="person" ):
                arr[count:,0] = i[0]["x1"]
                arr[count:,1] = i[0]["y1"]
                arr[count:,2] = i[0]["x2"]
                arr[count:,3] = i[0]["y2"]
                arr[count:,4] = i[0]["score"]
                #arr[count:,5] = i[0]["class_id"]
                count+=1      
        arr = torch.from_numpy(arr)
        return arr
    

    def run(self,frame):
        self.frame=frame
        objects = self.detect_objects(self.frame)
        self.veri=' '
        self.boundingData = ''
        for box in objects:
            if " " in str(box[6]):
                box[6]=(str(box[6]).replace(" ", "_"))
            self.boundingData = str([int(box[0]), int(box[1]), int(box[2]), int(box[3]), str(box[4]), str(box[6])])
            #self.draw_bounding_box(self.frame, box)
            self.veri += str(box).split("[")[1].split("]")[0] + '\n'
        

        duzenli_veri=self.string_parser(self.veri)
        return duzenli_veri

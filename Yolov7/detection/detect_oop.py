import torch
import cv2
import pandas as pd
import numpy as np
import warnings
import time
from hubconf import custom
from get_video import ThreadedCamera
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
import torch
warnings.filterwarnings('ignore')
from models.experimental import attempt_load
from utils.torch_utils import select_device

class ObjectDetector():
    def __init__(self, model_path,src):
        torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
        self.device = select_device()
        print("Device",self.device)
        #self.model = attempt_load("yolov7.pt", map_location=self.device)
        self.model = torch.hub.load('WongKinYiu/yolov7', 'custom', model_path,
                                   force_reload=True, trust_repo=True)
		
        self.camera=ThreadedCamera(src)
        self.points = np.array([[[446,344],[849,363],[1275,622],[1278,719],[0,717],[0,571]]])
           
    def detect_objects(self, frame):
        mask = np.zeros(self.frame.shape[0:2], dtype=np.uint8)
        cv2.fillPoly(mask, [self.points], 255)
        overlay = self.frame.copy()
        cv2.fillPoly(overlay, [self.points], (0, 255, 0))
        cv2.addWeighted(overlay, 0.1, self.frame, 1 - 0.1, 0, self.frame)
        cv2.drawContours(mask, [self.points], -1, (255, 255, 255), -1, cv2.LINE_AA)
        self.frame = cv2.bitwise_and(self.frame,self.frame,mask = mask) 
        
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
    
        return objects
    
    
    def run(self):
        
        flag=0
        while True:
            
            #self.ret, self.frame = vid.read()
        
            try:
                self.frame,self.FPS_MS = self.camera.show_frame()

                flag=1
            except AttributeError:
                pass
                        
            self.boundingData = ''
            self.veri = ''

            if flag==1:
                flag=0
                objects = self.detect_objects(self.frame)

                for box in objects:
                    if " " in str(box[6]):
                        box[6]=(str(box[6]).replace(" ", "_"))
                    self.boundingData = str([int(box[0]), int(box[1]), int(box[2]), int(box[3]), str(box[4]), str(box[6])])
                    #self.draw_bounding_box(self.frame, box)
                    self.veri += str(box).split("[")[1].split("]")[0] + '\n'
                
                #cv2.imshow("asd",self.frame)
                #cv2.waitKey(1)
                #print("self",self.veri)
                duzenli_veri=self.string_parser(self.veri)
                return duzenli_veri,self.frame, self.FPS_MS
                


#if __name__ == "__main__":
    #model_path = '/home/fatih/yolov7/yolov7.pt'
    #detector = ObjectDetector(model_path)
    #detector.run()
    

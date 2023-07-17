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

class ObjectDetector(ThreadedCamera):
    def __init__(self, model_path):
        super().__init__(model_path)
        torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
        self.model = torch.hub.load('WongKinYiu/yolov7', 'custom', model_path,
                                    force_reload=True, trust_repo=True)
        
        self.camera=ThreadedCamera()

    def detect_objects(self, frame):
        results = self.model(frame)
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

        for line in lines:
            parts = line.strip().split()
            obj = []
            for i in range(0, len(parts), 7):
                x1, y1, x2, y2, score, class_id, class_name = map(eval, parts[i:i+7])
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
    
        return objects
    
    
    
    def run(self):
        
        flag=0
        while True:
            #self.ret, self.frame = vid.read()
            
            try:
                self.frame,self.FPS_MS=self.camera.show_frame()
                flag=1
            except AttributeError:
                pass
                        
            self.boundingData = ''
            self.veri = ''

            if flag==1:
                flag=0
                objects = self.detect_objects(self.frame)

                for box in objects:
                 
                    self.boundingData = str([int(box[0]), int(box[1]), int(box[2]), int(box[3]), str(box[4]), str(box[6])])
                    #self.draw_bounding_box(self.frame, box)
                    self.veri += str(box).split("[")[1].split("]")[0] + '\n'
                
                #cv2.imshow("asd",self.frame)
                #cv2.waitKey(1)
                duzenli_veri=self.string_parser(self.veri)
                return duzenli_veri,self.frame, self.FPS_MS
                #print(self.veri)


if __name__ == "__main__":
    model_path = '/home/fatih/yolov7/yolov7.pt'
    # detector = ObjectDetector(model_path)
    # detector.run()
    
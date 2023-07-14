import torch
import cv2
import pandas as pd
import numpy as np
import warnings
import time
from hubconf import custom
from get_video import ThreadedCamera
warnings.filterwarnings('ignore')

class ObjectDetector(ThreadedCamera):
    def __init__(self, model_path):
        super().__init__(model_path)
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
                    self.draw_bounding_box(self.frame, box)
                    self.veri += str(box).split("[")[1].split("]")[0] + '\n'
                
                return self.veri,self.frame, self.FPS_MS
                #print(self.veri)


if __name__ == "__main__":
    model_path = '/home/fatih/yolov7/yolov7.pt'
    #detector = ObjectDetector(model_path)
    #detector.run()
    
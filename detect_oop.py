import torch
import cv2
import pandas as pd
import numpy as np
import warnings
import time
from hubconf import custom

warnings.filterwarnings('ignore')

class ObjectDetector:
    def __init__(self, model_path):
        self.model = torch.hub.load('WongKinYiu/yolov7', 'custom', model_path,
                                    force_reload=True, trust_repo=True)

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
        vid = cv2.VideoCapture(0)

        while True:
            self.ret, self.frame = vid.read()

            self.boundingData = ''
            self.veri = ''

            if self.ret:
                objects = self.detect_objects(self.frame)

                for box in objects:
                 
                    self.boundingData = str([int(box[0]), int(box[1]), int(box[2]), int(box[3]), str(box[4]), str(box[6])])
                    self.draw_bounding_box(self.frame, box)
                    cv2.imshow("Dedect",self.frame)
                    cv2.waitKey(1)
                    self.veri += str(box).split("[")[1].split("]")[0] + '\n'
                    
                
                print(self.veri)
                    
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        vid.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    model_path = '/home/fatih/yolov7/yolov7.pt'
    detector = ObjectDetector(model_path)
    detector.run()
    
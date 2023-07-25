from detect_oop import ObjectDetector
import cv2
import numpy as np
import math 
import torch
import argparse
from deepsort.detection_helpers import *
from deepsort.tracking_helpers import *
from  bridge_wrapper import *
from collections import deque
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from tracker import Tracker
from shapely.geometry import Polygon,Point
sayac_in=0
sayac_out=0


class CarCounter():
   
    def __init__(self,model_path,src):
        
        self.dedector=ObjectDetector(model_path,src)
        self.tracker = YOLOv7_DeepSORT(reID_model_path="deepsort/deep_sort/model_weights/mars-small128.pb",detector=self.dedector)

        self.in_line=[]
        self.out_line=[]
        self.src=src
        self.coordinates_in = [(446, 344), (626, 353), (586, 500),(225, 456)]
        self.coordinates_out = [(663, 357), (849, 363), (1063, 487),(686, 526)]
        self.polygon = Polygon(self.coordinates_in)
        self.polygon2 = Polygon(self.coordinates_out)

    def frame_converter(self,frame):
        
        for i in range(len(self.coordinates_in)):
            x1, y1 = self.coordinates_in[i]
            x2, y2 = self.coordinates_in[(i + 1) % len(self.coordinates_in)]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 
        for i in range(len(self.coordinates_out)):
            x1, y1 = self.coordinates_out[i]
            x2, y2 = self.coordinates_out[(i + 1) % len(self.coordinates_out)]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # 
        


    def import_area(self,data,index):
        
        global sayac_in
        global sayac_out
        orta_nokta_x= (data[0]+data[2])/2
        orta_nokta_y= (data[1]+data[3])/2
        nokta=Point((orta_nokta_x,orta_nokta_y))
        
        if(self.polygon.contains(nokta)):
            sayac_in+=1
            if index not in self.in_line:
                self.in_line.append(index)
        
             
        if(self.polygon2.contains(nokta)):
            sayac_out+=1
            if index not in self.out_line:
                self.out_line.append(index)
             
        
     
    def counter(self):

        global sayac_in
        global sayac_out
        while(True):
            frame,bbox_xyxy,track_id,fps,ff = self.tracker.track_video(self.src,verbose=1)
            self.frame_converter(frame)
        
            if(ff==1):
                bbox_xyxy = np.array(bbox_xyxy)
                for i in range (len(bbox_xyxy)):
                    self.import_area(bbox_xyxy[i,0:4],track_id[i])


            frame = cv2.putText(frame, "FPS:"+str(int(fps)), (585,609), cv2.FONT_HERSHEY_SIMPLEX, 
                    1,  (255, 255, 100), 2, cv2.LINE_AA)    

            frame = cv2.putText(frame, "Anlik Gelen Arac", (100,150), cv2.FONT_HERSHEY_SIMPLEX, 
                    1,  (255, 255, 100), 2, cv2.LINE_AA)    

            frame = cv2.putText(frame, "Anlik Giden Arac", (800,150), cv2.FONT_HERSHEY_SIMPLEX, 
                1,  (255, 255, 100), 2, cv2.LINE_AA)

            frame = cv2.putText(frame, str(sayac_out), (800,190), cv2.FONT_HERSHEY_SIMPLEX, 
                1,  (0,0,0), 2, cv2.LINE_AA)

            frame = cv2.putText(frame, str(sayac_in), (100,190), cv2.FONT_HERSHEY_SIMPLEX, 
                1,  (0,0,0), 2, cv2.LINE_AA)
    
            frame = cv2.putText(frame, str(len(self.in_line)), (100,100), cv2.FONT_HERSHEY_SIMPLEX, 
                    1,  (0,0,0), 2, cv2.LINE_AA)

            frame = cv2.putText(frame, str(len( self.out_line)), (800,100), cv2.FONT_HERSHEY_SIMPLEX, 
                1,  (0,0,0), 2, cv2.LINE_AA)

            frame = cv2.putText(frame, "Toplam Gelen Arac", (100,60), cv2.FONT_HERSHEY_SIMPLEX, 
                    1,  (255, 0, 0), 2, cv2.LINE_AA)

            frame = cv2.putText(frame, "Toplam Giden Arac", (800,60), cv2.FONT_HERSHEY_SIMPLEX, 
                    1,  (255, 0, 0), 2, cv2.LINE_AA)
          
            sayac_in=0
            sayac_out=0

            cv2.imshow("Detect",frame)
            cv2.waitKey(1)

        cv2.destroyAllWindows()
                
            
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Source")
    parser.add_argument('--source', dest='input_string', type=str, default="video.mp4",
                        help='kaynak')
    args = parser.parse_args()
    src=args.input_string
            
    count = CarCounter('/home/fatih/yolov7/yolov7.pt',src)
    count.counter()



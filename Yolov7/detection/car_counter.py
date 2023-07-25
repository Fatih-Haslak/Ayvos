from detect_oop import ObjectDetector
import cv2
import numpy as np
import math 
import torch
import argparse
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from deep_sort_pytorch.utils.parser import get_config
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
        self.tracker=Tracker()
 
        
        self.in_line=[]
        self.out_line=[]
        
        self.coordinates_in = [(446, 344), (626, 353), (525, 719),(0,717),(5, 570)]
        self.coordinates_out = [(663, 357), (849, 363), (1275, 622),(1275,717),(708, 716)]
        self.polygon = Polygon(self.coordinates_in)
        self.polygon2 = Polygon(self.coordinates_out)

    def frame_converter(self,frame):
        
        for i in range(len(self.coordinates_in)):
            x1, y1 = self.coordinates_in[i]
            x2, y2 = self.coordinates_in[(i + 1) % len(self.coordinates_in)]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Yeşil renkte çizgi, kalınlığı 2
        for i in range(len(self.coordinates_out)):
            x1, y1 = self.coordinates_out[i]
            x2, y2 = self.coordinates_out[(i + 1) % len(self.coordinates_out)]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Yeşil renkte çizgi, kalınlığı 2
        


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
             
        
     
    def counter(self,data,frame):

        global sayac_in
        global sayac_out
        
        frame,_dentities,_,flag,bbox_xyxy = self.tracker.count_tracker(data,frame)

        if (flag==1):
            for i in range (len(bbox_xyxy)):
                self.import_area(bbox_xyxy[i,0:4],_dentities[i])


        frame = cv2.putText(frame, "Anlik Gelen Arac", (100,150), cv2.FONT_HERSHEY_SIMPLEX, 
                   1,  (255, 255, 100), 2, cv2.LINE_AA)    

        frame = cv2.putText(frame, "Anlik Giden Arac", (800,150), cv2.FONT_HERSHEY_SIMPLEX, 
            1,  (255, 255, 100), 2, cv2.LINE_AA)

        frame = cv2.putText(frame, str(sayac_out), (800,190), cv2.FONT_HERSHEY_SIMPLEX, 
            1,  (255,255,255), 2, cv2.LINE_AA)

        frame = cv2.putText(frame, str(sayac_in), (100,190), cv2.FONT_HERSHEY_SIMPLEX, 
            1,  (255,255,255), 2, cv2.LINE_AA)
 
        frame = cv2.putText(frame, str(len(self.in_line)), (100,100), cv2.FONT_HERSHEY_SIMPLEX, 
                   1,  (255,255,255), 2, cv2.LINE_AA)

        frame = cv2.putText(frame, str(len( self.out_line)), (800,100), cv2.FONT_HERSHEY_SIMPLEX, 
            1,  (255,255,255), 2, cv2.LINE_AA)

        frame = cv2.putText(frame, "Toplam Gelen Arac", (100,60), cv2.FONT_HERSHEY_SIMPLEX, 
                   1,  (255, 0, 0), 2, cv2.LINE_AA)

        frame = cv2.putText(frame, "Toplam Giden Arac", (800,60), cv2.FONT_HERSHEY_SIMPLEX, 
                   1,  (255, 0, 0), 2, cv2.LINE_AA)
        
        sayac_in=0
        sayac_out=0

        return frame

    def run(self): #count için gerekli
        
        count=0

        while(True):
        
            veri,frame,fps = self.dedector.run()
            self.frame_converter(frame)
       
            try:
                car_count = sum(obj[0]['class_name'] == 'car' for obj in veri)
                arr = np.ones((car_count, 6))

            except:
                pass
            
            
            try:

                for i in veri:

                    if( i[0]["class_name"]=="car" and float(i[0]["score"]) > 0.4 ):
                        arr[count:,0] = i[0]["x1"]
                        arr[count:,1] = i[0]["y1"]
                        arr[count:,2] = i[0]["x2"]
                        arr[count:,3] = i[0]["y2"]
                        arr[count:,4] = i[0]["score"]
                        arr[count:,5] = i[0]["class_id"]
                        count+=1
              
                frame=self.counter(arr,frame)
                count=0
                cv2.imshow("Detect",frame)
                cv2.waitKey(fps)

            except:

                frame = cv2.putText(frame, "NO DETECT", (500,60), cv2.FONT_HERSHEY_SIMPLEX, 
                   1,  (255,255,255), 2, cv2.LINE_AA)
                   
                cv2.imshow("Detect",frame)
                cv2.waitKey(fps)

                
                
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Source")
    parser.add_argument('--source', dest='input_string', type=str, default="video.mp4",
                        help='kaynak')
    args = parser.parse_args()
    src=args.input_string

    count = CarCounter('/home/fatih/yolov7/yolov7.pt',src)
    count.run()



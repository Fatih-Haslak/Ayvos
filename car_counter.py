from detect_oop import ObjectDetector
import cv2
import numpy as np
import math 
import torch
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from deep_sort_pytorch.utils.parser import get_config
from collections import deque
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from tracker import Tracker

class CarCounter():
   
    def __init__(self,model_path):
        print("MOdel ",model_path)
        self.dedector=ObjectDetector(model_path)
        self.tracker=Tracker()
        self.detect2_liste = []
        self.detect_liste = []
        self.hesapla= []
        self.red_flag= 0

    def frame_converter(self,frame):
        
        cv2.rectangle(frame, (0,400), (623,730), (255,124,0), 2)
        cv2.rectangle(frame, (660,400), (1277,718), (0,25,255), 2)


    def import_area(self,data):
        
        
        orta_nokta_x= (data[0]+data[2])/2
        orta_nokta_y= (data[1]+data[3])/2
        
    
        if( (orta_nokta_x>0 and orta_nokta_y>400 and orta_nokta_x<623 and orta_nokta_y<730) ): #zone 1 ıcınde mı
            self.detect_liste.append([orta_nokta_x,orta_nokta_y])
            
        if(orta_nokta_x>660 and orta_nokta_y>400 and orta_nokta_x<1277 and orta_nokta_y<718):
            self.detect2_liste.append([orta_nokta_x,orta_nokta_y])
            


    def counter(self,data,frame):
        


        frame,_dentities,_,flag = self.tracker.count_tracker(data,frame)
        
        for i in range(0,len(data)):
            self.import_area(data[i,:])
        
        if flag==1:
            
            for i in _dentities:
                if(i not in self.hesapla):
                    self.hesapla.append(i) 
            print("Arac id",self.hesapla)
            print("Toplam arac sayısı",len(self.hesapla))

        frame = cv2.putText(frame, str(len(self.hesapla)), (100,100), cv2.FONT_HERSHEY_SIMPLEX, 
                   1,  (10, 110, 220), 2, cv2.LINE_AA)



        frame = cv2.putText(frame, str(len(self.detect_liste)), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1,  (0, 0, 0), 2, cv2.LINE_AA)

        frame = cv2.putText(frame, "IN", (80,50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1,  (0, 0, 0), 2, cv2.LINE_AA)

        frame = cv2.putText(frame, str(len(self.detect2_liste)), (700,50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1,  (255, 0, 0), 2, cv2.LINE_AA)

        frame = cv2.putText(frame, "OUT", (800,50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1,  (255, 0, 0), 2, cv2.LINE_AA)
    
        self.detect_liste.clear()
        self.detect2_liste.clear()

        return frame


    def run(self): #count için gerekli
        
        count=0

        while(1):
        
            veri,frame,fps = self.dedector.run()
            self.frame_converter(frame)
            
            car_count = sum(obj[0]['class_name'] == 'car' for obj in veri)
            arr = np.ones((car_count, 6))
            #print("Car_count",car_count)
            

            try:

                for i in veri:

                    if( i[0]["class_name"]=="car"):

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
                print("Video sonu")
                exit()

            # print("-----Yeni veri----")
            

if __name__ == "__main__":
    count = CarCounter('/home/fatih/yolov7/yolov7.pt')
    count.run()



from detect_oop import ObjectDetector
import cv2
import numpy as np
import math 


class CarCounter(ObjectDetector):
    def __init__(self,model_path):
        super().__init__(model_path)
        self.dedector=ObjectDetector(model_path)
        self.detect2_liste = []
        self.detect_liste = []
    
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

    def frame_converter(self,frame):
        cv2.rectangle(frame, (0,400), (623,730), (255,124,0), 2)
        cv2.rectangle(frame, (660,400), (1277,718), (0,25,255), 2)
                             #x,y      #xmax y max

    def import_area(self,data):
        
        orta_nokta_x= (data[0]+data[2])/2
        orta_nokta_y= (data[1]+data[3])/2
        
    
        if(orta_nokta_x>0 and orta_nokta_y>400 and orta_nokta_x<623 and orta_nokta_y<730): #zone 1 ıcınde mı
            self.detect_liste.append([orta_nokta_x,orta_nokta_y])
        if(orta_nokta_x>660 and orta_nokta_y>400 and orta_nokta_x<1277 and orta_nokta_y<718):
            self.detect2_liste.append([orta_nokta_x,orta_nokta_y])



    def counter(self,data,count,frame):
        for i in range(0,count):
            self.import_area(data[i,:]) 
         
        
        # print("Şerit gelen anlik araba sayisi",len(self.detect_liste))
        # print("Şerit giden anlik araba sayisi",len(self.detect2_liste))


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
        bouindig_boxes_liste=[]
        arr = np.ones((12, 4))
        count=0
        while(1):
            veri,frame,fps = self.dedector.run()
            self.frame_converter(frame)
            liste=self.string_parser(veri)
            
        
            #print(liste)
            # print("\n")
            try:
                for i in liste:
                    if(i[0]["class_name"]=="car"):
                        arr[count:,0] = i[0]["x1"]
                        arr[count:,1] = i[0]["y1"]
                        arr[count:,2] = i[0]["x2"]
                        arr[count:,3] = i[0]["y2"]
                        count+=1
                frame=self.counter(arr,count,frame)
                count=0
                cv2.imshow("Detect",frame)
                cv2.waitKey(fps)
            except:
                exit()
            # print("-----Yeni veri----")


if __name__ == "__main__":
    count = CarCounter('/home/fatih/yolov7/yolov7.pt')
    count.run()



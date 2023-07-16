from detect_oop import ObjectDetector
import cv2
import numpy as np
import math 

class CarCounter(ObjectDetector):
    def __init__(self,model_path):
        super().__init__(model_path)
        self.dedector=ObjectDetector(model_path)
        self.sayac = 0
        self.temp_liste = []
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
        cv2.line(frame,(183,477),(584,477),(0,100,244),2)
        cv2.line(frame,(679,477),(1031,477),(100,50,100),2)
        cv2.line(frame,(0,656),(550,656),(0,100,244),2)
        cv2.line(frame,(670,348),(840,348),(100,50,100),2)
    
    def comp_list(self,liste1):
        list_distance_indexer=[]

        for sayac1,i in enumerate(liste1):
            for sayac2, a in enumerate(liste1):
                if(sayac1!=sayac2 and sayac1<sayac2):
                    distance = math.sqrt((a[0] - i[0])**2 + (a[1] - i[1])**2)
                    if distance<48:
                        list_distance_indexer.append(sayac1)

                    print("-----Yeni veri----")    
                    print("Distance",distance)           
                  
        
        print(list_distance_indexer)
        print(liste1)

        for index in reversed(list_distance_indexer):
            del liste1[index]
        
        print("Son hali",liste1)
        list_distance_indexer.clear()
    
    def import_area(self,data):
        
        orta_nokta_x=(data[0]+data[2])/2
        orta_nokta_y=(data[1]+data[3])/2
        
        #orta nokta sadece detect testı ıcın
        
        if(orta_nokta_x>0 and orta_nokta_y>477 and orta_nokta_x<584 and orta_nokta_y<656):

            self.detect_liste.append([orta_nokta_x,orta_nokta_y])
            self.comp_list(self.detect_liste)


    def counter(self,data,count):
        
        for i in range(0,count):
            self.import_area(data[i,:]) 

        print("Araba",(self.detect_liste))
        print("Sayilan araba",len(self.detect_liste))

    def run(self): #counbt için gerekli
        bouindig_boxes_liste=[]
        arr = np.ones((12, 4))
        count=0
        while(1):
            veri,frame,fps = self.dedector.run()
            self.frame_converter(frame)
            liste=self.string_parser(veri)
            
            cv2.imshow("Dedect",frame)
            cv2.waitKey(fps)
            #print(liste)
            # print("\n")
            for i in liste:
                if(i[0]["class_name"]=="car"):
                    arr[count:,0] = i[0]["x1"]
                    arr[count:,1] = i[0]["y1"]
                    arr[count:,2] = i[0]["x2"]
                    arr[count:,3] = i[0]["y2"]
                    count+=1
            self.counter(arr,count)
            count=0
            # print("-----Yeni veri----")


if __name__ == "__main__":
    count = CarCounter('/home/fatih/yolov7/yolov7.pt')
    count.run()



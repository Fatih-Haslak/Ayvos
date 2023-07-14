from detect_oop import ObjectDetector
import cv2
import numpy as np
class CarCounter(ObjectDetector):
    def __init__(self,model_path):
        super().__init__(model_path)
        self.dedector=ObjectDetector(model_path)

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
        pass

    def outlier(self,data):
        pass
    
    def counter(self,data):
        pass

    def run(self): #counbt için gerekli
        bouindig_boxes_liste=[]
        arr = np.ones((12, 4))
        count=0
        while(1):
            veri,frame,fps = self.dedector.run()
            
            liste=self.string_parser(veri)
            
            cv2.imshow("Dedect",frame)
            cv2.waitKey(fps)
            #print(liste)
            print("\n")
            for i in liste:
                if(i[0]["class_name"]=="car"):
                    arr[count:,0] = i[0]["x1"]
                    arr[count:,1] = i[0]["y1"]
                    arr[count:,2] = i[0]["x2"]
                    arr[count:,3] = i[0]["y2"]
                    count+=1
                    # bouindig_boxes_liste.append(i[0]["x1"])
                    # bouindig_boxes_liste.append(i[0]["y1"])
                    # bouindig_boxes_liste.append(i[0]["x2"])
                    # bouindig_boxes_liste.append(i[0]["y2"])
            print(arr[0:count])
            print("\n")
            print(veri)
            print(count)
            count=0
    
if __name__ == "__main__":
    count = CarCounter('/home/fatih/yolov7/yolov7.pt')
    count.run()



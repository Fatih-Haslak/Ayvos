from detect_oop import ObjectDetector
import cv2

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
        
        while(1):
            veri,frame,fps=self.dedector.run()
            liste=self.string_parser(veri)
            #print(liste)
            cv2.imshow("Dedect",frame)
            cv2.waitKey(fps)
            print(liste)
            print("\n")

            print(liste[2][0]["x1"])
            ad=liste[2][0]
            print(ad["x1"])
            print("\n")

if __name__ == "__main__":
    count = CarCounter('/home/fatih/yolov7/yolov7.pt')
    count.run()



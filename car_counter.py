from detect_oop import ObjectDetector
import cv2

class CarCounter(ObjectDetector):
    def __init__(self,model_path):
        super().__init__(model_path)
        self.dedector=ObjectDetector(model_path)

    def run(self): #counbt için gerekli
        while(1):
            veri,frame,fps=self.dedector.run()
            cv2.imshow("Dedect",frame)
            cv2.waitKey(fps)      
            print(veri)
        # for i in self.dedector.run():
        #     #bboxları appendlemeye calısalım
        #     lines = i.strip().split("\n")

        #     fps = (lines[0])
        #     matrix = (lines[1])
        #     other_text = lines[2:]
        #     print(fps)
        #     print(other_text)


if __name__ == "__main__":
    count = CarCounter('/home/fatih/yolov7/yolov7.pt')
    count.run()



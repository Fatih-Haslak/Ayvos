import cv2
import numpy as np 


class CornerHarris():
    def __init__(self,image_path):
        self.img_org = cv2.imread(image_path)
        self.img = cv2.cvtColor(self.img_org, cv2.COLOR_BGR2GRAY)        
        self.corner = np.float32(self.img)

    def run(self):
        corners = cv2.cornerHarris(self.corner, 2, 3, 0.05)
        corners = cv2.dilate(corners, None)
        self.img_org[corners > 0.01 * corners.max()]=[0, 0, 255]
        cv2.imshow('Image with Corners', self.img_org)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

class CannyDetector():
    def __init__(self, image_path):
        self.img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        self.kernel_size=(7,7)
        self.lower_bound=50
        self.upper_bound=100
    
    def auto_thparameter_finder(self,img,sigma=0.33):
        md=np.median(self.img)
        self.lower_bound = int(max(0, (1.0-sigma) * md))
        self.upper_bound = int(min(255, (1.0+sigma) * md))

    def run(self):
        self.img_blur = cv2.GaussianBlur(self.img, self.kernel_size, 0)
        self.auto_thparameter_finder(self.img,0.33)
        print("Lower and Upper bound",self.lower_bound,self.upper_bound)
        self.edge=cv2.Canny(self.img_blur, self.lower_bound, self.upper_bound)
        cv2.imshow("Resim",self.img)
        cv2.waitKey(0)
        cv2.imshow("Sonuc",self.edge)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
obj=CannyDetector("/home/fatih/opencv_tasks/newyork.jpg")
obj.run()
corner=CornerHarris("/home/fatih/opencv_tasks/newyork.jpg")
corner.run()
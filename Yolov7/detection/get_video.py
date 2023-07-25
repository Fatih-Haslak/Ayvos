from threading import Thread
import cv2, time
import numpy as np 
class ThreadedCamera():
    def __init__(self, src=None):
        if(src=="0" or src=="1"):
            src=int(src)
        print("src",src)
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        # FPS = 1/X
        # X = desired FPS
        self.FPS = 1/30
        self.FPS_MS = int(self.FPS * 1000)
        # Start frame retrieval thread
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        
    def update(self):
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
                
            time.sleep(self.FPS)
        print("HATA")    
            
    def show_frame(self):
        #cv2.imshow('frame', self.frame)
        #cv2.waitKey(self.FPS_MS)
        return self.frame, self.FPS_MS
        
#if __name__ == '__main__':
    # src = 'video.mp4'
    # src= 0
    # threaded_camera = ThreadedCamera(src)
    # while True:
    #     try:
    #         threaded_camera.show_frame()
    #     except AttributeError:
    #         pass

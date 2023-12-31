import torch
import detectron2
import os
import sys
import cv2
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import argparse
from get_video import ThreadedCamera
from detectron2.utils.visualizer import ColorMode

#instances https://detectron2.readthedocs.io/en/latest/tutorials/models.html#model-output-format
class Detector:
    def __init__(self,mode):
        setup_logger()
        sys.path.insert(0, os.path.abspath('./detectron2'))
        self.init_run_and_mode(mode)
        self.camera=ThreadedCamera()

    def init_run_and_mode(self,select_mode):

        if(select_mode==1):
            #Dedection
            path="COCO-Detection/faster_rcnn_R_50_C4_3x.yaml"
        elif(select_mode==0):
            #Mask
            path="COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml"
        
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file(path))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(path)
        self.predictor = DefaultPredictor(self.cfg)
        self.metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
        

    def detect_objects(self, image_path):

        flag=0
        while True:
            try:
                self.frame,self.FPS_MS = self.camera.show_frame()
                flag=1
            except AttributeError:
                pass
                
            if flag==1:
                flag=0       
                image = self.frame
                outputs = self.predictor(image)
                visualizer = Visualizer(image[:, :, ::-1], self.metadata, scale=1.2,instance_mode=ColorMode.SEGMENTATION)
                out = visualizer.draw_instance_predictions(outputs["instances"].to("cpu"))
                cv2.imshow("Result", out.get_image()[:, :, ::-1])
                cv2.waitKey(1)
        cv2.destroyAllWindows() 
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description= ' Classification or Segmentation ')
    parser.add_argument('--deger', type=int, default=1, help=' 1 == Detection // 0 == Segmentasyon')
    args = parser.parse_args()
    deger=args.deger
    print(deger)
    detector = Detector(deger)
    detector.detect_objects("bus.jpg")


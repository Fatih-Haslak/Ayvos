'''
A Moduele which binds Yolov7 repo with Deepsort with modifications
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # comment out below line to enable tensorflow logging outputs
import time
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
import cv2
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.compat.v1 import ConfigProto # DeepSORT official implementation uses tf1.x so we have to do some modifications to avoid errors

# deep sort imports
from deepsort.deep_sort import preprocessing, nn_matching
from deepsort.deep_sort.detection import Detection
from deepsort.deep_sort.tracker import Tracker

# import from helpers
from deepsort.tracking_helpers import read_class_names, create_box_encoder
from deepsort.detection_helpers import *
from detect_oop import ObjectDetector

 # load configuration for object detector
config = ConfigProto()
config.gpu_options.allow_growth = True


class YOLOv7_DeepSORT:
    '''
    Class to Wrap ANY detector  of YOLO type with DeepSORT
    '''
    def __init__(self, reID_model_path:str, detector=None, max_cosine_distance:float=0.4, nn_budget:float=None, nms_max_overlap:float=1.0,
    coco_names_path:str ="deepsort/io_data/input/classes/coco.names",  ):
        '''
        args: 
            reID_model_path: Path of the model which uses generates the embeddings for the cropped area for Re identification
            detector: object of YOLO models or any model which gives you detections as [x1,y1,x2,y2,scores, class]
            max_cosine_distance: Cosine Distance threshold for "SAME" person matching
            nn_budget:  If not None, fix samples per class to at most this number. Removes the oldest samples when the budget is reached.
            nms_max_overlap: Maximum NMs allowed for the tracker
            coco_file_path: File wich contains the path to coco naames
        '''

        #self.detector = detector
        self.yolo_dets = detector
        self.coco_names_path = coco_names_path
        self.nms_max_overlap = nms_max_overlap
        self.class_names = read_class_names()

        # initialize deep sort
        self.encoder = create_box_encoder(reID_model_path, batch_size=1)
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget) # calculate cosine distance metric
        self.tracker = Tracker(metric) # initialize tracker
        self.flag=0
        self.bbox_depo=[]
        self.track_depo=[]
    def track_video(self,video:str, verbose:int = 1):
        
        while(True):
            # -----------------------------------------PUT ANY DETECTION MODEL HERE -----------------------------------------------------------------
            start_time = time.time()
            duzenli_veri, frame ,FPS_MS = self.yolo_dets.run()
            count=0

            try:
                car_count = sum(obj[0]['class_name'] == 'car' and float(obj[0]["score"]) > 0.5 for obj in duzenli_veri)
                arr = np.ones((car_count, 6))

            except:
                return frame,0,0,0,0
                               

            try:
                
                for i in duzenli_veri:
                    if( i[0]["class_name"]=="car" and float(i[0]["score"]) > 0.5 ):
                        arr[count:,0] = i[0]["x1"]
                        arr[count:,1] = i[0]["y1"]
                        arr[count:,2] = i[0]["x2"]
                        arr[count:,3] = i[0]["y2"]
                        arr[count:,4] = i[0]["score"]
                        arr[count:,5] = i[0]["class_id"]
                        count+=1
            except:
                print("HATA 2")
                continue
            #arreyı dısardan döndür inpıt olarak track video alsın
            bboxes = arr[:,:4]
            bboxes[:,2] = bboxes[:,2] - bboxes[:,0] # convert from xyxy to xywh
            bboxes[:,3] = bboxes[:,3] - bboxes[:,1]

            scores = arr[:,4]
            classes = arr[:,-1]
            num_objects = arr.shape[0]

            # ---------------------------------------- DETECTION PART COMPLETED ---------------------------------------------------------------------
            
            names = []
            for i in range(num_objects): # loop through objects and use class index to get class name
                class_indx = int(classes[i])
                class_name = self.class_names[class_indx]
                names.append(class_name)

            names = np.array(names)
            count = len(names)

            # ---------------------------------- DeepSORT tacker work starts here ------------------------------------------------------------
            features = self.encoder(frame, bboxes) # encode detections and feed to tracker. [No of BB / detections per frame, embed_size]
            detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)] # [No of BB per frame] deep_sort.detection.Detection object

            cmap = plt.get_cmap('tab20b') #initialize color map
            colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

            boxs = np.array([d.tlwh for d in detections])  # run non-maxima supression below
            scores = np.array([d.confidence for d in detections])
            classes = np.array([d.class_name for d in detections])
            indices = preprocessing.non_max_suppression(boxs, classes, self.nms_max_overlap, scores)
            detections = [detections[i] for i in indices]       
            

            self.tracker.predict()  # Call the tracker
            self.tracker.update(detections) #  updtate using Kalman Gain
            self.bbox_depo.clear()
            self.track_depo.clear()
            self.flag=0
            for track in self.tracker.tracks:  # update new findings AKA tracks
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue 
                
                self.bbox = track.to_tlbr()
                tp=self.bbox.tolist()
                self.bbox_depo.append(tp[0:])
                class_name = track.get_class()
                self.track_depo.append(int(track.track_id))
                #print(track.track_id)
        
                color = colors[int(track.track_id) % len(colors)]  # draw bbox on screen
                color = [i * 255 for i in color]
                cv2.rectangle(frame, (int(self.bbox[0]), int(self.bbox[1])), (int(self.bbox[2]), int(self.bbox[3])), color, 2)
                cv2.rectangle(frame, (int(self.bbox[0]), int(self.bbox[1]-30)), (int(self.bbox[0])+(len(class_name)+len(str(track.track_id)))*17, int(self.bbox[1])), color, -1)
                cv2.putText(frame, class_name + " : " + str(track.track_id),(int(self.bbox[0]), int(self.bbox[1]-11)),0, 0.6, (255,255,255),1, lineType=cv2.LINE_AA)    
                self.flag=1
               
                    
            # -------------------------------- Tracker work ENDS here -----------------------------------------------------------------------
            if verbose >= 1:
                fps = 1.0 / (time.time() - start_time) # calculate frames per second of running detections
                # print("fps",fps)
            
            result = np.asarray(frame)
            #result = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA)
            
            temp=self.bbox_depo
            temp2=self.track_depo
            return result, temp,temp2,fps,1

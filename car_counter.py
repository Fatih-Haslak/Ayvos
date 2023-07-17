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

class CarCounter(ObjectDetector):
    def __init__(self,model_path):
        super().__init__(model_path)
        self.dedector=ObjectDetector(model_path)
        self.detect2_liste = []
        self.detect_liste = []
        self.cfg_deep = get_config()
        self.cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")
        self.deepsort = DeepSort(self.cfg_deep.DEEPSORT.REID_CKPT,
                        max_dist=self.cfg_deep.DEEPSORT.MAX_DIST, min_confidence=self.cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=self.cfg_deep.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=self.cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=self.cfg_deep.DEEPSORT.MAX_AGE, n_init=self.cfg_deep.DEEPSORT.N_INIT, nn_budget=self.cfg_deep.DEEPSORT.NN_BUDGET,
                        use_cuda=True)
        self.class_path='data/coco.names'
        self.data_deque={}
        self.names=self.load_classes(self.class_path)


    def draw_border(self,img, pt1, pt2, color, thickness, r, d):
        x1,y1 = pt1
        x2,y2 = pt2
        # Top left
        cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
        cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
        # Top right
        cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
        cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
        # Bottom left
        cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
        cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
        # Bottom right
        cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
        cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r - d), color, -1, cv2.LINE_AA)
        
        cv2.circle(img, (x1 +r, y1+r), 2, color, 12)
        cv2.circle(img, (x2 -r, y1+r), 2, color, 12)
        cv2.circle(img, (x1 +r, y2-r), 2, color, 12)
        cv2.circle(img, (x2 -r, y2-r), 2, color, 12)
        
        return img


    def UI_box(self, x, img, color=None, label=None, line_thickness=None):
        # Plots one bounding box on image img
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]

            img = self.draw_border(img, (c1[0], c1[1] - t_size[1] -3), (c1[0] + t_size[0], c1[1]+3), color, 1, 8, 2)

            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        
        return img

    def draw_boxes(self,img, bbox, names,object_id, identities=None, offset=(0, 0)):
        #cv2.line(img, line[0], line[1], (46,162,112), 3)
      
        height, width, _ = img.shape
        # remove tracked point from buffer if object is lost
        for key in list(self.data_deque):
            if key not in identities:
                self.data_deque.pop(key)
        
        for i, box in enumerate(bbox[0:,:4]):
          
            x1, y1, x2, y2 = [int(i) for i in box]

            start_point = (int(x1),int(y1))
            end_point = (int(x2), int(y2))
            
            x1 += offset[0]
            x2 += offset[0]
            y1 += offset[1]
            y2 += offset[1]

            # start_point = (int(x1),int(y1))
            # end_point = (int(x2), int(y2))
            
            # code to find center of bottom edge
            center = (int((x2+x1)/ 2), int((y2+y2)/2))
            
            # get ID of object
            id = int(identities[i]) if identities is not None else 0
            
            # create new buffer for new object
            if id not in self.data_deque:  
                
                self.data_deque[id] = deque(maxlen= 64)

            obj_name = names[object_id[i]]
            label = '{}{:d}'.format("", id) + ":"+ '%s' % (obj_name)

            # add center to buffer
            
            self.data_deque[id].appendleft(center)
            self.UI_box(box, img, label=label, color=(0,25,25), line_thickness=2)
            # draw trail

            for i in range(1, len(self.data_deque[id])):
                # check if on buffer value is none
                if self.data_deque[id][i - 1] is None or self.data_deque[id][i] is None:
                    continue
               
                thickness = int(np.sqrt(64 / float(i + i)) * 1.5)
              
                cv2.line(img, self.data_deque[id][i - 1], self.data_deque[id][i], (0,128,12), thickness) #arkadan cıkan cizgi
                cv2.rectangle(img, start_point, end_point, (222,82,175), 2) ##bbox

        return img


    def xyxy_to_xywh(self,xyxy):
        #Calculates the relative bounding box from absolute pixel values. """
        bbox_left = min([xyxy[0], xyxy[2]])
        bbox_top = min([xyxy[1], xyxy[3]])
        bbox_w = abs(xyxy[0] - xyxy[2])
        bbox_h = abs(xyxy[1] - xyxy[3])
        x_c = (bbox_left + bbox_w / 2)
        y_c = (bbox_top + bbox_h / 2)
        w = bbox_w
        h = bbox_h
        return x_c, y_c, w, h
   
   
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


    def load_classes(self,class_path):
        with open(class_path, 'r') as f:
            names = f.read().split('\n')
        return list(filter(None, names))

    def import_area(self,data):
        
        
        orta_nokta_x= (data[0]+data[2])/2
        orta_nokta_y= (data[1]+data[3])/2
        
    
        if( (orta_nokta_x>0 and orta_nokta_y>400 and orta_nokta_x<623 and orta_nokta_y<730) ): #zone 1 ıcınde mı
            self.detect_liste.append([orta_nokta_x,orta_nokta_y])
        if(orta_nokta_x>660 and orta_nokta_y>400 and orta_nokta_x<1277 and orta_nokta_y<718):
            self.detect2_liste.append([orta_nokta_x,orta_nokta_y])

    

    def counter(self,data,count,frame):
        xywh_bboxs = []
        for i in range(0,count):
            self.import_area(data[i,:])
            
            x_c, y_c, bbox_w, bbox_h = self.xyxy_to_xywh(data[i,0:4])
            xywh_bboxs.append([x_c, y_c, bbox_w, bbox_h])
        
        xywhs = torch.Tensor(xywh_bboxs)
        confss = torch.Tensor((data[0:count,4]))
        oids = ((data[0:count,5]))
        oids = list((oids))
        
        outputs = self.deepsort.update(xywhs, confss, oids, frame)

        if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -2]
                    object_id = outputs[:, -1]
                    frame=self.draw_boxes(frame, bbox_xyxy, self.names, object_id,identities)


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
        arr = np.ones((12, 6))
        count=0
        while(1):
            veri,frame,fps = self.dedector.run()
            self.frame_converter(frame)
            liste=self.string_parser(veri)
            
            
            #print(liste)
            # print("\n")
            try:
                for i in liste:
                    if(i[0]["class_name"]=="car" ):
                        arr[count:,0] = i[0]["x1"]
                        arr[count:,1] = i[0]["y1"]
                        arr[count:,2] = i[0]["x2"]
                        arr[count:,3] = i[0]["y2"]
                        arr[count:,4] = i[0]["score"]
                        arr[count:,5] = i[0]["class_id"]
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



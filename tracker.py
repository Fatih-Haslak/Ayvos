import cv2
import numpy as np
import torch
from collections import deque
from deep_sort_pytorch.deep_sort import DeepSort
from deep_sort_pytorch.utils.parser import get_config

class Tracker():
    def __init__(self):
        super().__init__()
        cfg_deep = get_config()
        cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")
        self.deepsort = DeepSort(
            cfg_deep.DEEPSORT.REID_CKPT,
            max_dist=cfg_deep.DEEPSORT.MAX_DIST,
            min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
            nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP,
            max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
            max_age=cfg_deep.DEEPSORT.MAX_AGE,
            n_init=cfg_deep.DEEPSORT.N_INIT,
            nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
            use_cuda=True
        )
        
        self.class_path = 'data/coco.names'
        self.data_deque = {}
        self.names = self.load_classes(self.class_path)


    def load_classes(self,class_path):
        with open(self.class_path, 'r') as f:
            names = f.read().split('\n')
        return list(filter(None, names))
    
    def draw_border(self, img, pt1, pt2, color, thickness, r, d):
    
        x1, y1 = pt1
        x2, y2 = pt2
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

        cv2.circle(img, (x1 + r, y1 + r), 2, color, 12)
        cv2.circle(img, (x2 - r, y1 + r), 2, color, 12)
        cv2.circle(img, (x1 + r, y2 - r), 2, color, 12)
        cv2.circle(img, (x2 - r, y2 - r), 2, color, 12)

        return img

    def UI_box(self, x, img, color=None, label=None, line_thickness=None):
        tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        color = color or [np.random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]

            img = self.draw_border(img, (c1[0], c1[1] - t_size[1] - 3), (c1[0] + t_size[0], c1[1] + 3), color, 1, 8, 2)

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
                cv2.rectangle(img, start_point, end_point, (222,82,175), 2) ## bbox
        
        return img

    def xyxy_to_xywh(self, xyxy):
        bbox_left = min([xyxy[0], xyxy[2]])
        bbox_top = min([xyxy[1], xyxy[3]])
        bbox_w = abs(xyxy[0] - xyxy[2])
        bbox_h = abs(xyxy[1] - xyxy[3])
        x_c = (bbox_left + bbox_w / 2)
        y_c = (bbox_top + bbox_h / 2)
        w = bbox_w
        h = bbox_h
        return x_c, y_c, w, h


    def count_tracker(self, data, frame):
        xywh_bboxs = []
        count=len(data)

        for i in range(count):
            x_c, y_c, bbox_w, bbox_h = self.xyxy_to_xywh(data[i, :4])
            xywh_bboxs.append([x_c, y_c, bbox_w, bbox_h])
       
        xywhs = torch.Tensor(xywh_bboxs)
        confss = torch.Tensor(data[:count, 4])
        oids = data[:count, 5].astype(int).tolist()
       
        outputs = self.deepsort.update(xywhs, confss, oids, frame)
        
        if len(outputs) > 0:
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -2]
            object_id = outputs[:, -1]
            #print("identities",identities)
            frame = self.draw_boxes(frame, bbox_xyxy, self.names,object_id, identities)
            
            return frame,identities,object_id,1

        else:
            return frame,0,0,0




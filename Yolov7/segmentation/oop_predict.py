import argparse
import os
import platform
import sys
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from numpy import random

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv7 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

print(ROOT)
        

from models.common import DetectMultiBackend

from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path

from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                        increment_path, non_max_suppression,scale_segments, print_args,scale_coords, strip_optimizer, xyxy2xywh,apply_classifier)


from utils.plots import Annotator, colors, save_one_box, plot_one_box
from utils.segment.general import process_mask, scale_masks, masks2segments
from utils.segment.plots import plot_masks
from utils.torch_utils import select_device, smart_inference_mode

class YOLOv7Segmentation:
    def __init__(self):
        pass

    @smart_inference_mode() 
    def parse_opt(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov7-seg.pt', help='model path(s)')
        parser.add_argument('--source', type=str, default=ROOT / 'football1.mp4', help='file/dir/URL/glob, 0 for webcam')
        parser.add_argument('--data', type=str, default=ROOT / 'data/coco.yaml', help='(optional) dataset.yaml path')
        parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
        parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
        parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true', help='show results')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
        parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--visualize', action='store_true', help='visualize features')
        parser.add_argument('--update', action='store_true', help='update all models')
        parser.add_argument('--project', default=ROOT / 'runs/predict-seg', help='save results to project/name')
        parser.add_argument('--name', default='exp', help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
        parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
        parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
        parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
        parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
              
        opt = parser.parse_args()
        opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
        print_args(vars(opt))
        return opt

    def main(self, opt):
        check_requirements(exclude=('tensorboard', 'thop'))
        self.run(**vars(opt))

    def model_load(self,webcam,weights,device,dnn,data,half,imgsz,source):
        
        #Model yüklenımı ve data cekımı
        device = select_device(device)
        print("Dnn",dnn)
        print("Data",data)
        print("Fp16",half)
        #fp16==false daha hızlı
        #fp16==True baya yavas
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=False)
        
        #Model parametlerını bastırarak datatype ogrenme 
        #for param in model.parameters():
        #    print(param.dtype)
        
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size
    
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
            bs = len(dataset)  # batch_size
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
            bs = 1  # batch_size
        vid_path, vid_writer = [None] * bs, [None] * bs

        # Run inference
        model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        
        return model,seen, windows, dt ,dataset,device,names,vid_path, vid_writer
    

    def run(self, weights, source, data, imgsz, conf_thres, iou_thres, max_det, device, view_img, save_txt, save_conf,
            save_crop, nosave, classes, agnostic_nms, augment, visualize, update, project, name, exist_ok,
            line_thickness, hide_labels, hide_conf, half, dnn):


        source = str(source)
        save_img = not nosave and not source.endswith('.txt')  # save inference images
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)

        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        
        webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)

        if is_url and is_file:
            source = check_file(source)  # download

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

 
        model, seen, windows, dt ,dataset,device,names,vid_path,vid_writer = self.model_load(webcam,weights,device,dnn,data,half,imgsz,source)
        

        names2 = model.module.names if hasattr(model, 'module') else model.names
  
        colors2 = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        for path, im, im0s, vid_cap, s in dataset:

            with dt[0]:
                # frame shape (3,x,y) chanell width and height and numpy array
                im = torch.from_numpy(im).to(device)
                im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32 float 16 or float 32 for memory optimizer
                
                im /= 255  # 0 - 255 to 0.0 - 1.0  # 255' e  'bolunmediğinde cuda memory  yetmedi'
                # frame shape torch(3,x,y) chanell width and height
       
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim
                
                #frame shape torch(1,3,x,y) batch,chanell width and height

            # Inference

            with dt[1]:
                visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                pred, out = model(im, augment=augment, visualize=visualize)
                proto = out[1]

            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)


            # Process predictions

            for i, det in enumerate(pred):  #per image
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # im.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                
                if len(det):
                    
                    masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC

                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                    
                    #bbox= det[:, :4].detach().cpu().numpy()

                    # Segments
                    if 1:
                        segments = reversed(masks2segments(masks))
                        segments = [scale_segments(im.shape[2:], x, im0.shape).round() for x in segments]

                    # Print results
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                        print(s)
                            
                    # Mask plotting ----------------------------------------------------------------------------------------
                    mcolors = [colors(int(6), True) for cls in det[:, 5]]
                    im0 = plot_masks(im[i], masks, mcolors)  # image with masks shape(imh,imw,3) #color hep aynı
                 
                    #Bbox çizimi
                    for i,conf in enumerate(det[:,4:6]):

                        oran = conf[0].item() 
                        isim = conf[1].item()

                        label = f'{names[int(isim)]} {oran:.2f}'
                        plot_one_box(det[i, :4], im0, label=label, color=colors2[i], line_thickness=1)       
                    
        
                
                
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond


                # Save results (image with detections)
                # if save_img:
                #     if dataset.mode == 'image':
                #         cv2.imwrite(save_path, im0)
                #     else:  # 'video' or 'stream'
                #         if vid_path[i] != save_path:  # new video
                #             vid_path[i] = save_path
                #             if isinstance(vid_writer[i], cv2.VideoWriter):
                #                 vid_writer[i].release()  # release previous video writer
                #             if vid_cap:  # video
                #                 fps = vid_cap.get(cv2.CAP_PROP_FPS)
                #                 w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                #                 h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                #             else:  # stream
                #                 fps, w, h = 30, im0.shape[1], im0.shape[0]
                #             save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                #             vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                #         vid_writer[i].write(im0)


            # Print time (inference-only) 
            preprocess_time = dt[0].dt*1E3  # Pre-process süresi
            inference_time = dt[1].dt*1E3  # Inference süresi
            nms_time = dt[2].dt*1E3  # NMS süresi
            total_time = preprocess_time + inference_time + nms_time  # Toplam süre
            
            fps = int(1000 / inference_time)  # FPS hesaplama (1000 milisaniye = 1 saniye)

            #LOGGER.info(f"Camera FPS: {fps:.2f}")
            #LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

        # Print results
        t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image


        LOGGER.info(
            f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)

        # if save_txt or save_img:
        #     s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #     LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
        if update:
            strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


if __name__ == "__main__":
    yolov7 = YOLOv7Segmentation()
    opt = yolov7.parse_opt()
    yolov7.main(opt)

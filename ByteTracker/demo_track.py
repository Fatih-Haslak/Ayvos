import argparse
import os
import os.path as osp
import time
import cv2
import torch
import sys
from loguru import logger

other_file_directory = "/home/fatih/byterack/ByteTrack/tools/yolov7"
sys.path.append(other_file_directory)

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.utils.visualize import plot_tracking
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracking_utils.timer import Timer
from yolov7.detect_oop import ObjectDetector

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
detector = ObjectDetector()


class Predictor:
    def __init__(self, exp, device=torch.device(0)):
        self.num_classes = exp.num_classes
        self.test_size = exp.test_size
        self.device = device

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None
        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        img, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0).float().to(self.device)

        return img, img_info


class ByteTrackDemo:
    def __init__(self, args):
        self.args = args
        self.exp = get_exp(args.exp_file, args.name)
        self.output_dir = osp.join(self.exp.output_dir, args.experiment_name)
        os.makedirs(self.output_dir, exist_ok=True)
        self.vis_folder = osp.join(self.output_dir, "track_vis")
        os.makedirs(self.vis_folder, exist_ok=True)
        self.device = torch.device("cuda" if args.device == "gpu" else "cpu")
        self.predictor = Predictor(self.exp, self.device)
        self.tracker = BYTETracker(args, frame_rate=args.fps)
        self.vid_writer = None

    def __get_image_list(self, path):
        image_names = []
        for maindir, subdir, file_name_list in os.walk(path):
            for filename in file_name_list:
                apath = osp.join(maindir, filename)
                ext = osp.splitext(apath)[1]
                if ext in IMAGE_EXT:
                    image_names.append(apath)
        return image_names

    def __write_results(self, filename, results):
        save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
        with open(filename, 'w') as f:
            for frame_id, tlwhs, track_ids, scores in results:
                for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                    if track_id < 0:
                        continue
                    x1, y1, w, h = tlwh
                    line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1),
                                              y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                    f.write(line)

    def imageflow_demo(self):
        cap = cv2.VideoCapture(self.args.path if self.args.demo == "video" else self.args.camid)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
        fps = cap.get(cv2.CAP_PROP_FPS)
        timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        save_folder = osp.join(self.vis_folder, timestamp)
        os.makedirs(save_folder, exist_ok=True)
        if self.args.demo == "video":
            save_path = osp.join(save_folder, self.args.path.split("/")[-1])
        else:
            save_path = osp.join(save_folder, "camera.mp4")
        self.vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height)))

        timer = Timer()
        frame_id = 0
        results = []
        while True:
            ret_val, frame = cap.read()
            if ret_val:
                img, img_info = self.predictor.inference(frame, timer)
                timer.tic()
                with torch.no_grad():
                    my_outputs = detector.run(frame)

                if my_outputs[0] is not None:
                    output_results = my_outputs.cpu().numpy()
                    online_targets = self.tracker.update(output_results, [img_info["height"], img_info["width"]],
                                                         self.exp.test_size)

                    online_tlwhs = []
                    online_ids = []
                    online_scores = []
                    for t in online_targets:
                        tlwh = t.tlwh
                        tid = t.track_id
                        vertical = tlwh[2] / tlwh[3] > self.args.aspect_ratio_thresh
                        if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                            online_tlwhs.append(tlwh)
                            online_ids.append(tid)
                            online_scores.append(t.score)
                            results.append(
                                f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                            )
                    timer.toc()
                    online_im = plot_tracking(
                        img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / timer.average_time
                    )

                else:
                    timer.toc()
                    online_im = img_info['raw_img']

                if self.args.save_result:
                    cv2.imshow("Sonuc", online_im)
                    self.vid_writer.write(online_im)

                ch = cv2.waitKey(1)
                if ch == 27 or ch == ord("q") or ch == ord("Q"):
                    break
            else:
                break
            frame_id += 1

        if self.args.save_result:
            res_file = osp.join(self.vis_folder, f"{timestamp}.txt")
            self.write_results(res_file, results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    parser.add_argument("demo", default="image", help="demo type, eg. image, video and webcam")
    parser.add_argument("-expn", "--experiment-name", type=str, default="default_experiment",
                        help="experiment name (provide a default value if not provided)")
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    parser.add_argument("--path", default="./videos/palace.mp4", help="path to images or video")
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument("--save_result", action="store_true",
                        help="whether to save the inference result of image/video")

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default="./exps/example/mot/yolox_s_mix_det.py",
        type=str,
        help="pls input your expriment description file",
    )
    # parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")

    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    
    args = parser.parse_args()
    demo = ByteTrackDemo(args)
    demo.imageflow_demo()

# YOLOv3 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --weights yolov3.pt --source 0  # webcam
                                                             img.jpg  # image
                                                             vid.mp4  # video
                                                             path/  # directory
                                                             path/*.jpg  # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import scipy.optimize

import cv2
import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

def box_center_to_corner(boxes):
    """Convert from (center, width, height) to (upper-left, lower-right)."""
    cx, cy, w, h = boxes[0], boxes[1], boxes[2], boxes[3]
    x1 = cx - 0.5 * w
    y1 = cy - 1.2 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.2 * h
    boxes = torch.stack((x1, y1, x2, y2), axis=-1)
    return boxes

def custom_bbox(gt_coords, img, imgname):
    cbbox_coords = []
    for k in range(len(gt_coords)):
        if gt_coords[k][0] == imgname:
            box = [float(gt_coords[k][2]), float(gt_coords[k][3]), 50, 80]
            box = torch.tensor(box)
            bbox = box_center_to_corner(box)

            x1 = int(bbox[0].item())
            y1 = int(bbox[1].item())
            x2 = int(bbox[2].item())
            y2 = int(bbox[3].item())

            coords = [x1, y1, x2, y2]
            cbbox_coords.append(coords)
                
            img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
    return img, cbbox_coords

def bbox_iou(boxA, boxB):
  # https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
  # ^^ corrected.
    
  # Determine the (x, y)-coordinates of the intersection rectangle  
  xA = max(boxA[0], boxB[0])
  yA = max(boxA[1], boxB[1])
  xB = min(boxA[2], boxB[2])
  yB = min(boxA[3], boxB[3])

  interW = xB - xA + 1
  interH = yB - yA + 1

  # Correction: reject non-overlapping boxes
  if interW <=0 or interH <=0 :
    return -1.0

  interArea = interW * interH
  boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
  boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
  iou = interArea / float(boxAArea + boxBArea - interArea)
#   print(iou)
  return iou

def match_bboxes(bbox_gt, bbox_pred, IOU_THRESH=0.1):
    '''
    Given sets of true and predicted bounding-boxes,
    determine the best possible match.
    Parameters
    ----------
    bbox_gt, bbox_pred : N1x4 and N2x4 np array of bboxes [x1,y1,x2,y2]. 
      The number of bboxes, N1 and N2, need not be the same.
    
    Returns
    -------
    (idxs_true, idxs_pred, ious, labels)
        idxs_true, idxs_pred : indices into gt and pred for matches
        ious : corresponding IOU value of each match
        labels: vector of 0/1 values for the list of detections
    '''
    n_true = bbox_gt.shape[0]
    n_pred = bbox_pred.shape[0]
    
    MAX_DIST = 1.0
    MIN_IOU = 0.0

    # NUM_GT x NUM_PRED
    iou_matrix = np.zeros((n_true, n_pred))
    for i in range(n_true):
        for j in range(n_pred):
            iou_matrix[i, j] = bbox_iou(bbox_gt[i,:], bbox_pred[j,:])
    

    if n_pred > n_true:
      # there are more predictions than ground-truth - add dummy rows
      diff = n_pred - n_true
      iou_matrix = np.concatenate((iou_matrix, 
                                    np.full((diff, n_pred), MIN_IOU)), 
                                  axis=0)

    if n_true > n_pred:
      # more ground-truth than predictions - add dummy columns
      diff = n_true - n_pred
      iou_matrix = np.concatenate((iou_matrix, 
                                    np.full((n_true, diff), MIN_IOU)), 
                                  axis=1)


    # call the Hungarian matching
    idxs_true, idxs_pred = scipy.optimize.linear_sum_assignment(1 - iou_matrix)

    if (not idxs_true.size) or (not idxs_pred.size):
        ious = np.array([])
    else:
        ious = iou_matrix[idxs_true, idxs_pred]

    # remove dummy assignments
    sel_pred = idxs_pred<n_pred
    idx_pred_actual = idxs_pred[sel_pred] 
    idx_gt_actual = idxs_true[sel_pred]
    ious_actual = iou_matrix[idx_gt_actual, idx_pred_actual]
    sel_valid = (ious_actual > IOU_THRESH)
    label = sel_valid.astype(int)

    return idx_gt_actual[sel_valid], idx_pred_actual[sel_valid], ious_actual[sel_valid], label

def findClosest(time, camera_time_list):
    val = min(camera_time_list, key=lambda x: abs(x - time))
    return camera_time_list.index(val)

@torch.no_grad()
def run(image_dir,  # file/dir/URL/glob, 0 for webcam
        index_file,
        gt,
        weights=ROOT / 'yolov3.pt',  # model.pt path(s)
        imgsz=640,  # inference size (pixels)
        conf_thres=0.4,  # confidence threshold
        iou_thres=0.6,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        file_name = 'LAB-GROUNDTRUTH.ref'
        ):

    cam_det, cam_gt = 0, 0

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()
    
    #===== process the index files of camera ======#
    with open(index_file) as f:
        content = f.readlines()
    cam_content = [x.strip() for x in content]
    c_frames = []
    c_times = []
    for line in cam_content:
        s = line.split(" ")
        frame = s[0]
        time = float(s[1]+'.'+s[2])
        c_frames.append(frame)
        c_times.append(time)

    #===== process the GT annotations  =======#
    with open("/home/dissana8/LAB/"+file_name) as f:
        content = f.readlines()

    content = [x.strip() for x in content]
    counter = -1
    print('Extracting GT annotation ...')
    c_frame_no = []
    for line in content:
        counter += 1
        # if counter % 1000 == 0:
        # print(counter)
        s = line.split(" ")
        
        time = float(s[0])
        frame_idx = findClosest(time, c_times) # we have to map the time to frame number
        c_frame_no.append(c_frames[frame_idx])

    for ele in enumerate(c_frame_no):
        #real images
        im = image_dir+ele[1]
        source = str(im)
        save_img = not nosave and not source.endswith('.txt')  # save inference images
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
        if is_url and is_file:
            source = check_file(source)  # download

        # Directories
        # save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Dataloader
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)
            bs = len(dataset)  # batch_size
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
            bs = 1  # batch_size
        vid_path, vid_writer = [None] * bs, [None] * bs

        # Run inference
        if pt and device.type != 'cpu':
            model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
        dt, seen = [0.0, 0.0, 0.0], 0
        for path, im, im0s, vid_cap, s in dataset:
            print(path)
            t1 = time_sync()
            im = torch.from_numpy(im).to(device)
            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference
            # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)
            t3 = time_sync()
            dt[1] += t3 - t2

            # NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            dt[2] += time_sync() - t3

            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                # p = Path(p)  # to Path
                # save_path = str(save_dir / p.name)  # im.jpg
                # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                # s += '%gx%g ' % im.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                imc = im0.copy() if save_crop else im0  # for save_crop
                annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    # # Print results
                    # for c in det[:, -1].unique():
                    #     if names[int(c)]=="person":
                    #         n = (det[:, -1] == c).sum()  # detections per class
                    #         print(n.item())
                    #         s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                            

                    # Write results
                    # for *xyxy, conf, cls in reversed(det):
                    #     if save_txt:  # Write to file
                    #         xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    #         line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                    #         with open(txt_path + '.txt', 'a') as f:
                    #             f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    #     if save_img or save_crop or view_img:  # Add bbox to image
                    #         c = int(cls)  # integer class
                    #         if c == 0:
                    #             label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    #             x1 = xyxy[0].item()
                    #             y1 = xyxy[1].item()
                    #             x2 = xyxy[2].item()
                    #             y2 = xyxy[3].item()
                    #             annotator.box_label(xyxy, label, color=colors(c, True))
                                # if save_crop:
                                #     save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)


                    #real images
                    # imgfile = path.split('/')[6:]

                    #adv images TOG
                    # imgfile = path.split('/')[9:]

                    #adv images Daedulus
                    # imgfile = path.split('/')[6:]

                    #naturalsitic patches
                    imgfile = path.split('/')[7:]

                    imgname = '/'.join(imgfile)
                    # sname = savename + imgname

                    bbox_coords=[]
                    for *xyxy, conf, cls in reversed(det):
                        if int(cls) == 0:
                            bbox = [xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item()]
                            bbox_coords.append(bbox)

                    image, cbbox = custom_bbox(gt, im0, imgname)
                    if cbbox:
                        cbbox = np.array(cbbox)
                        bbox = np.array(bbox_coords)
                        idx_gt_actual, idx_pred_actual, ious_actual, label = match_bboxes(cbbox, bbox)
                        cam_gt+=len(cbbox)
                            

                        for h in range(len(idx_gt_actual)):
                            t = idx_gt_actual[h]
                            text_c = cbbox[t]
                            if round(ious_actual[h], 3)>=0.0:
                                cam_det+=1

    
                # Print time (inference-only)
                # LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

                # # Stream results
                # im0 = annotator.result()
                # if view_img:
                #     cv2.imshow(str(p), im0)
                #     cv2.waitKey(1)  # 1 millisecond

                # # Save results (image with detections)
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
                #                 save_path += '.mp4'
                #             vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                #         vid_writer[i].write(im0)

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    # if save_txt or save_img:
    #     s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
    #     LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    # if update:
    #     strip_optimizer(weights)  # update model (to fix SourceChangeWarning)
    return cam_gt, cam_det


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov3.pt', help='model path(s)')
    # parser.add_argument('--image_dir', type=str, help='file/dir/URL/glob, 0 for webcam')
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
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    # parser.add_argument('--index_file', type=str, help='index file for each camera')
    parser.add_argument('--file_name', type=str, default = 'LAB-GROUNDTRUTH.ref', help='ground turth annotations')

    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))

    gt = []
    gt.append(np.load('/home/dissana8/LAB/data/LAB/cam1_coords__.npy', allow_pickle=True))
    gt.append(np.load('/home/dissana8/LAB/data/LAB/cam2_coords__.npy', allow_pickle=True))
    gt.append(np.load('/home/dissana8/LAB/data/LAB/cam3_coords__.npy', allow_pickle=True))
    gt.append(np.load('/home/dissana8/LAB/data/LAB/cam4_coords__.npy', allow_pickle=True))

    #index file
    cam1_index_file = "/home/dissana8/LAB/Visor/cam1/index.dmp"
    cam2_index_file = "/home/dissana8/LAB/Visor/cam2/index.dmp"
    cam3_index_file = "/home/dissana8/LAB/Visor/cam3/index.dmp"
    cam4_index_file = "/home/dissana8/LAB/Visor/cam4/index.dmp"

    # real images
    # cam1_image_dir = "/home/dissana8/LAB/Visor/cam1/"
    # cam2_image_dir = "/home/dissana8/LAB/Visor/cam2/"
    # cam3_image_dir = "/home/dissana8/LAB/Visor/cam3/"
    # cam4_image_dir = "/home/dissana8/LAB/Visor/cam4/"

    #When changing the image directory also change the path inside run function
    #TOG images 
    # cam1_image_dir = "/home/dissana8/TOG/Adv_images/vanishing/LAB/Visor/cam1/"
    # cam2_image_dir = "/home/dissana8/TOG/Adv_images/vanishing/LAB/Visor/cam2/"
    # cam3_image_dir = "/home/dissana8/TOG/Adv_images/vanishing/LAB/Visor/cam3/"
    # cam4_image_dir = "/home/dissana8/TOG/Adv_images/vanishing/LAB/Visor/cam4/"

    #Daedulus images
    # cam1_image_dir = "/home/dissana8/Daedalus-physical/Adv_Images/cam1/"
    # cam2_image_dir = "/home/dissana8/Daedalus-physical/Adv_Images/cam2/"
    # cam3_image_dir = "/home/dissana8/Daedalus-physical/Adv_Images/cam3/"
    # cam4_image_dir = "/home/dissana8/Daedalus-physical/Adv_Images/cam4/"

    #Naturalistic adv patch
    cam1_image_dir = "/home/dissana8/Naturalistic-Adversarial-Patch/eval_output/LAB_Daedalus_yolov3_0.1/cam1/"
    cam2_image_dir = "/home/dissana8/Naturalistic-Adversarial-Patch/eval_output/LAB_Daedalus_yolov3_0.1/cam2/"
    cam3_image_dir = "/home/dissana8/Naturalistic-Adversarial-Patch/eval_output/LAB_Daedalus_yolov3_0.1/cam3/"
    cam4_image_dir = "/home/dissana8/Naturalistic-Adversarial-Patch/eval_output/LAB_Daedalus_yolov3_0.1/cam4/"


    cam1_gt, cam1_det = run(image_dir= cam1_image_dir, index_file= cam1_index_file, gt = gt[0], **vars(opt))
    cam2_gt, cam2_det = run(image_dir= cam2_image_dir, index_file= cam2_index_file, gt = gt[1], **vars(opt))
    cam3_gt, cam3_det = run(image_dir= cam3_image_dir, index_file= cam3_index_file, gt = gt[2], **vars(opt))
    cam4_gt, cam4_det = run(image_dir= cam4_image_dir, index_file= cam4_index_file, gt = gt[3], **vars(opt))

    f = open("detections_Daedalus_YoloV3_0.2.txt", "a")
    # f.write("Dete of Yolo-V3 : " +str(success_rate)+"\n")
    f.write("GT Detections of view 01" +": "+str(cam1_gt)+"\n")
    f.write("Detections of view 01" +": "+str(cam1_det)+"\n")
    f.write("GT Detections of view 02" +": "+str(cam2_gt)+"\n")
    f.write("Detections of view 02" +": "+str(cam2_det)+"\n")
    f.write("GT Detections of view 03" +": "+str(cam3_gt)+"\n")
    f.write("Detections of view 03" +": "+str(cam3_det)+"\n")
    f.write("GT Detections of view 04" +": "+str(cam4_gt)+"\n")
    f.write("Detections of view 04" +": "+str(cam4_det)+"\n")
    f.write("\n")
    f.write("\n")
    f.close()

    cam1_success_rate = (cam1_det/cam1_gt)*100
    cam2_success_rate = (cam2_det/cam2_gt)*100
    cam3_success_rate = (cam3_det/cam3_gt)*100
    cam4_success_rate = (cam4_det/cam4_gt)*100

    tot_det = cam1_det + cam2_det +cam3_det + cam4_det
    tot_gt = cam1_gt + cam2_gt + cam3_gt + cam4_gt

    success_rate = (tot_det/tot_gt)*100

    f = open("success_rate_daedalus_YoloV3_0.2.txt", "a")
    f.write("Success rate of Yolo-V3 : " +str(success_rate)+"\n")
    f.write("Success rate of view 01" +": "+str(cam1_success_rate)+"\n")
    f.write("Success rate of view 02" +": "+str(cam2_success_rate)+"\n")
    f.write("Success rate of view 03" +": "+str(cam3_success_rate)+"\n")
    f.write("Success rate of view 04" +": "+str(cam4_success_rate)+"\n")
    f.write("\n")
    f.write("\n")
    f.close()

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)

import argparse
import datetime
import logging
import numpy as np
import signal
import shutil
import threading
import yaml
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized
from utils.utils import (
    set_logg, sgn_final, mask_screens, trim_screens, multi_screens,
    distancing, save_imgs_regular, uploadftp_log, uploadftp_raw_data, uploadftp_risk)


def detect(args):
    signal.signal(signal.SIGINT, sgn_final)
    with open(args.param, "r") as f:
        param = yaml.load(f, Loader=yaml.FullLoader)
    # Log file
    Path(args.path_ftp).mkdir(exist_ok=True)
    log_file_dir = f'{args.path_ftp}/log_{datetime.datetime.now().strftime("%Y%m%d%H")}.log'
    set_logg(log_file_dir)
    if args.upload_ftp: uploadftp_log(param['ftp'], log_file_dir)

    # Initialize
    device = select_device(args.device)
    if Path(args.output).is_dir():
        shutil.rmtree(args.output)
    Path(args.output).mkdir()
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(args.weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(args.img_size, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # webcam
    source = args.source
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        save_img = False
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
        # Get images from iterator
        imgs = dataset.get_images()
        risk_report, risk_coords = [None for _ in range(len(imgs))], [[] for _ in range(len(imgs))]
        if args.judge:
            warning_cnt = np.zeros(len(imgs), dtype='int')
            reset_warn = np.zeros(len(imgs), dtype='int')
            nums_raw_pic = args.rawdata_ftp  # np.ones(len(imgs), dtype='int') * args.rawdata_ftp
            tmps_raw_pic = [[] for _ in range(args.rawdata_ftp)]
            judge_delay = datetime.datetime.now()
        if args.break_frames > 0:
            tmp_img, break_count = imgs, np.zeros(len(imgs), dtype='int')
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    # Set parm
    # Full screen
    win_name = "MyScreen"
    if args.view_img and args.full_screen:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    # Set multi-screen (FHD)
    if args.view_img and args.multi_screen:
        multi_x, multi_y = args.screen_size
        multi_screen_matrix = np.zeros((multi_y, multi_x, 3), dtype=np.uint8)
    if args.collecttime > 0:
        Path(args.collectpath).mkdir(exist_ok=True)
        collect_path_list = []
        collect_delay = datetime.datetime.now()

    # Main loop
    for path, img, im0s, vid_cap in dataset:
        now_time = datetime.datetime.now()
        if webcam:
            people_coords = [[] for _ in range(len(im0s))]  # List to storage bounding coordinates of people
            if args.break_frames > 0:
                for i, per_im0s in enumerate(im0s):
                    # if difference is all zeros it will return False
                    if np.any(cv2.subtract(tmp_img[i], per_im0s)):
                        break_count[i] = 0
                        tmp_img[i] = per_im0s.copy()
                    else:
                        print(f"{i}:break_count: {break_count[i]}")
                        break_count[i] += 1
                        if break_count[i] > args.reset_frames:
                            logging.warning('Restart Streaming.')
                            if args.upload_ftp: uploadftp_log(param['ftp'], log_file_dir)
                            print("\033[42;37mRestart Streaming. \033[0m")
                            dataset.terminate()
                            break
            if args.collecttime > 0:
                for f in Path(args.collectpath).iterdir():
                    if f not in collect_path_list:
                        collect_path_list.append(f)
                if len(collect_path_list) > args.collectdays:
                    shutil.rmtree(collect_path_list[0])
                    collect_path_list.pop(0)
                # save data regularly according to args.collecttime
                if (now_time - collect_delay).seconds > args.collecttime:
                    threading.Thread(target=save_imgs_regular, args=(im0s, args.collectpath, now_time)).start()
                    collect_delay = now_time
        else:
            people_coords, risk_report, risk_coords = [[]], [[]], [[]]

        # 遮蔽畫面非偵測區
        if args.mask_screen:
            if webcam:
                for i, fr in enumerate(img):
                    mask_screens(param['mask_screen'], path[i], img[i], im0s)
            else:
                mask_screens([[[0, 350, 0, 1080], [350, 920, 600, 1080]]], 0, img, im0s)

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=args.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, args.conf_thres, args.iou_thres,
                                   classes=args.classes, agnostic=args.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            # s += '%gx%g ' % img.shape[2:]  # print string
            # gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    # label = f'{names[int(cls)]} {conf:.2f}'
                    if names[int(cls)] == 'person':
                        people_coords[i].append(xyxy)
                        # plot_dots_on_people(xyxy, im0)
                        if args.colorful_bbox:
                            plot_one_box(xyxy, im0, label=f'{1e2 * conf:.1f}', color=colors[int(cls)], line_thickness=2)
                        else:
                            plot_one_box(xyxy, im0, label=f'{1e2 * conf:.1f}', color=[0, 255, 0], line_thickness=2)

            ## Plot lines connecting people
            risk_report[i], risk_coords[i] = distancing(people_coords[i], im0, dist_thres_lim=args.risk_range)
            # Edge area edit (cut)
            if args.trim_screen:
                trim_screens(param['trim_screen'], path[i], im0)

            # Stream results
            if args.view_img:
                if args.multi_screen:
                    cv2.imshow(win_name, multi_screens(multi_screen_matrix, im0, i, multi_x, multi_y))
                else:
                    if args.full_screen:
                        cv2.imshow(win_name, im0)
                    else:
                        cv2.imshow(f'windows:{i}', im0)

            # Print time (inference + NMS)
            print(f'{p}:{s} fps={1 / (t2 - t1):.1f}')

            # Save results (image with detections)
            if save_img:
                save_path = str(Path(args.output) / Path(p).name)
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

            # Judgment
            if webcam and args.judge:
                if args.upload_ftp and args.rawdata_ftp > 0 and args.rawdata_ftp > nums_raw_pic > -1:
                    tmps_raw_pic[nums_raw_pic] = im0s
                    nums_raw_pic += 1
                    if nums_raw_pic == args.rawdata_ftp:
                        threading.Thread(
                            target=uploadftp_raw_data, args=(param['ftp'], tmps_raw_pic, file_name)).start()
                else:
                    if risk_report[i]:
                        print(f"\033[44mHigh risk: {warning_cnt[i]} \033[0m")
                        warning_cnt[i] += 1
                        if warning_cnt[i] > args.alarm_frames and (now_time - judge_delay).seconds > args.delay_time:
                            judge_delay = now_time
                            reset_warn[i], warning_cnt[i], nums_raw_pic = 0, 0, 0
                            if args.upload_ftp:
                                file_name = f"{args.path_ftp}/C{i}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                                threading.Thread(
                                    target=uploadftp_risk,
                                    args=(param['ftp'], im0, det, risk_coords[i],
                                          f"{file_name}.jpg", f"{file_name}.csv")).start()
                    elif warning_cnt[i] > 0:
                        reset_warn[i] += 1
                if reset_warn[i] > args.reset_frames:
                    reset_warn[i], warning_cnt[i] = 0, 0
                    print("\033[43;37mRESET WARNING.\033[0m")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='sdd_yolov5m.pt', help='model.pt path')
    parser.add_argument('--source', type=str, default='inference/images', help='source')
    parser.add_argument('--param', type=str, default='parameters.yaml', help='Detect parameters.')

    parser.add_argument("--mask-screen", action='store_true', help='Masking the screen of the predicted data.')
    parser.add_argument("--trim-screen", action='store_true', help='Cover the edges of the screen.')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--full-screen', action='store_true')

    parser.add_argument('--multi-screen', action='store_true', help='display multi-screens')
    parser.add_argument('--screen-size', type=int, nargs="+", default=(1920, 1080))

    parser.add_argument('--judge', action='store_true')
    parser.add_argument("--risk-range", type=int, nargs="+", default=(310,),
                        help='Social distance range (High risk range, Low risk range)')
    parser.add_argument('--alarm-frames', type=int, default=30, help='Number of alarm frames.')
    parser.add_argument('--reset-frames', type=int, default=6, help='Number of reset frames.')
    parser.add_argument('--delay-time', type=int, default=20, help='Time between alarms and alarms.(seconds)')

    parser.add_argument('--break-frames', type=int, default=0,
                        help='The program will be terminated when the same photo appears more than <int> times.')

    parser.add_argument('--collecttime', type=int, default=0, help='Data collection.(every <int> seconds)')
    parser.add_argument('--collectdays', type=int, default=10, help='Data preservation period.(days)')
    parser.add_argument('--collectpath', type=str, default='inference/collection', help='Collect data storage path')

    parser.add_argument('--upload-ftp', action='store_true')
    parser.add_argument('--rawdata-ftp', type=int, default=0, help='save raw images while alarm')
    parser.add_argument('--path-ftp', type=str, default='inference/ftp', help='Source(local) data path')

    parser.add_argument('--colorful-bbox', action='store_true', help='Use different color bboxes')

    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    args = parser.parse_args()
    print(args)

    with torch.no_grad():
        if args.update:  # update all models (to fix SourceChangeWarning)
            for args.weights in ['sdd_yolov5s.pt', 'sdd_yolov5m.pt', 'sdd_yolov5l.pt', 'sdd_yolov5x.pt']:
                detect(args)
                strip_optimizer(args.weights)
        else:
            detect(args)

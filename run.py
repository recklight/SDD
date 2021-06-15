from detect import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='sdd_yolov5m.pt', help='model.pt path')
    # parser.add_argument('--weights', type=str, default='yolov5l.pt', help='model.pt path')

    parser.add_argument('--source', type=str, default='source.txt')
    # parser.add_argument('--source', type=str, default='inference/images', help='source')

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

    # 當攝影機不穩定時常輸出相同照片, 使用此功能中止程式. 關閉此功能可增加效能
    parser.add_argument('--break-frames', type=int, default=0,
                        help='The program will be terminated when the same photo appears more than <int> times.')

    parser.add_argument('--collecttime', type=int, default=0, help='Data collection.(every <int> seconds)')
    parser.add_argument('--collectdays', type=int, default=10, help='Data preservation period.(days)')
    parser.add_argument('--collectpath', type=str, default='inference/collection', help='Collect data storage path')

    parser.add_argument('--upload-ftp', action='store_true')
    parser.add_argument('--rawdata-ftp', type=int, default=0, help='save raw images while alarm')
    parser.add_argument('--path-ftp', type=str, default='inference/ftp', help='Source(local) data path')

    parser.add_argument('--colorful-bbox', action='store_true', help='Use different color bboxes')

    # parser.add_argument('--save-txt', action='store_false', help='save results to *.txt')
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

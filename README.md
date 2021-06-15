# SDD - Social Distancing Detector using YOLOv5


## 本專案參考 [yolov5-v3.0](https://github.com/ultralytics/yolov5/releases/tag/v3.0)


## 模型訓練方法請參照 [Train Custom Data](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data) 或 [How to Train YOLOv5 On a Custom Dataset](https://blog.roboflow.com/how-to-train-yolov5-on-a-custom-dataset/)
- **請注意 PyTorch版本 (本專案使用v1.6.0)**
- **已標記訓練資料 [Download](https://drive.google.com/file/d/1OEbeqyI26DzSdgctYCRHIo1oYcNXEjh6/view?usp=sharing)**

## Inference
```bash
$ python run.py --weights sdd_yolov5s.pt --source 0 # webcam
                             sdd_yolov5m.pt          path/ # directory
                             sdd_yolov5l.pt          file.jpg # image 
                             sdd_yolov5x.pt
```

## To run inference images in `inference/images`
```bash
$ python run.py --weights sdd_yolov5m.pt --source inference/images
```
![image](https://user-images.githubusercontent.com/53622566/120078385-eeb5d280-c0e1-11eb-829e-5c7b6de5681a.png)

## 
```bash
$ sh start.sh
```
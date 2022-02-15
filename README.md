# SDD - Social Distancing Detector using YOLOv5


## 關於本專案 
- 參考[yolov5-v3.0](https://github.com/ultralytics/yolov5/releases/tag/v3.0)
- PyTorch版本使用v1.6.0
- 已標記訓練資料下載 [Download](https://drive.google.com/file/d/1OEbeqyI26DzSdgctYCRHIo1oYcNXEjh6/view?usp=sharing)


## 模型訓練方法請參照 
- [Train Custom Data](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)
- [How to Train YOLOv5 On a Custom Dataset](https://blog.roboflow.com/how-to-train-yolov5-on-a-custom-dataset/)


## 模型推論
```bash
  $ python detect.py \
  --source source.txt \  
  --weights sdd_yolov5m.pt \
  --judge \ # 對預測結果進行判斷.回報.上傳ftp
  --alarm 30 \ # NG次數, 大於此次數則進行上傳ftp
  --break 0 \ # 當攝影機不穩定時(常輸出相同照片), 使用此功能中止程式(0). 關閉此功能可增加效能
  --reset 6 \ # 如果有設定--break下, 相同照片出現--reset次數將關閉停止運行 
  --delay 20 \ # 判斷NG後的--delay秒內的NG將被忽略
  --upload \ # 是否上傳ftp
  --rawdata 8 \ 當NG發生時, 儲存--rawdata張原始照片
  --path inference/ftp \ 輸出(推論)資料存放目錄
  --collecttime 90 \ 定時自動拍照(間隔--collecttime秒) 
  --collectdays 8 \ 儲存最近--collectdays天的自動拍照(其他天則刪除)
  --collectpath inference/collection \ # 存放自動拍照的路徑
  --param parameters.yaml # 一些參數設定(ftp位址等資料, 畫面遮蔽/切割位置)
```


## 模型推論 - 特定資料 `inference/images`
```bash
$ python detect.py --weights sdd_yolov5m.pt --source inference/images
```
![image](https://user-images.githubusercontent.com/53622566/120078385-eeb5d280-c0e1-11eb-829e-5c7b6de5681a.png)


## 在 Nvidia Jetson 系列裝置上運行

- 請參考start.sh 
```bash
$ sh start.sh
```

- 手動調整風扇轉速
```bash
$ sudo sh -c "echo 200 > /sys/devices/pwm-fan/target_pwm"
```

- 可參考實際 [運行影片(將前往YT)](https://youtu.be/5SeDiPgxT60)
![20210317_160603](https://user-images.githubusercontent.com/53622566/154070774-f2c395ea-77ed-4cdc-88ff-8ca98406ee2a.jpg)


#!/bin/sh

wget_download() {
  wget --load-cookies /tmp/cookies.txt \
  "https://docs.google.com/uc?export=download&confirm=\
  $(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate \
  "https://drive.google.com/uc?export=download&id=$1" -O- | \
  sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=$1" -O $2
  rm -rf /tmp/cookies.txt
}

wget_download $1 $2

#wget https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5s.pt -O yolov5s.pt
#wget https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5m.pt -O yolov5m.pt
#wget https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5l.pt -O yolov5l.pt
#wget https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5x.pt -O yolov5x.pt
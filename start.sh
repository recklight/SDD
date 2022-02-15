#!/bin/bash

sudo sh -c "echo 200 > /sys/devices/pwm-fan/target_pwm"
echo -e "\033[41;37mStarting program. Fan speed: 200\033[0m"

while true; do
  python detect.py \
  --source source.txt \
  --weights sdd_yolov5m.pt \
  --judge --alarm 30 --reset 6 --delay 20 \
  --upload --rawdata 8 --path inference/ftp \
  --collecttime 90 --collectdays 8 --collectpath inference/collection
done

#while true; do
#  python run.py \
#  --source source.txt \
#  --weights sdd_yolov5m.pt \
#  --mask --trim \
#  --judge --alarm 30 --reset 6 --delay 20 \
#  --upload --rawdata 8 --path inference/ftp \
#  --collecttime 90 --collectdays 8 --collectpath inference/collection
#done
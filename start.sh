#!/bin/bash

FANSPEED=140
sudo sh -c "echo $FANSPEED > /sys/devices/pwm-fan/target_pwm"
echo -e "Starting program. Fan speed: $FANSPEED"

while true; do
  python detect.py \
  --source source.txt \
  --weights sdd_yolov5m.pt \
  --conf 0.88 \
  --iou 0.45 \
  --judge \
  --alarm 30 \
  --break 0 \
  --reset 6 \
  --delay 20 \
  --upload \
  --rawdata 8 \
  --path media/agx/s128/storge/ftp \
  --collecttime 100 \
  --collectdays 7 \
  --collectpath media/agx/s128/storge/collection \
  --param parameters.yaml \
  --mask \
  --trim
done

  # --view \
  # --multi \
  # --full \

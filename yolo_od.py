""" Implimentation of YOLO v3 object detection  by Chieko N."""

import numpy as np
import argparse
import time
import cv2
import os
from yolo_od_utils import yolo_object_detection

# Get options specified in the command line
parser = argparse.ArgumentParser()
parser.add_argument('image_files', nargs='+')
parser.add_argument('-c', '--confidence', type=float, default=0.5)
parser.add_argument('-t', '--threshold', type=float, default=0.5)
args = parser.parse_args()

# set filenames for the model
coco_names_file = "yolo/coco.names"
yolov3_weight_file = "yolo/yolov3.weights"
yolov3_config_file = "yolo/yolov3.cfg"

# read coco object names
LABELS = open(coco_names_file).read().strip().split("\n")

# assign rondom colours to the corresponding class labels
np.random.seed(45)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# read YOLO network model
net = cv2.dnn.readNetFromDarknet(yolov3_config_file, yolov3_weight_file)

# read image files
for img in args.image_files:
    # execute object detection for the image
    yolo_object_detection(img, net, args.confidence, args.threshold, LABELS, COLORS)

#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# 作者：Create on 2019/7/18 17:29 by jade
# 邮箱：jadehh@live.com
# 描述：TODO
# 最近修改：2019/7/18 17:29 modify by jade
import argparse
from jade import *
from objects_model import ObjectModel
import cv2

paraser = argparse.ArgumentParser(description="Detect car")
#genearl
paraser.add_argument("--model_path",default="/home/jade/Models/objectDetectionModels/ssd_mobilenet_v2_hand_2019-07-18/pb/ssd_mobilenet_v2_hand.pb",help="path to load model")
paraser.add_argument("--label_path",default="/home/jade/label_map/hand.prototxt",help="path to labels")
paraser.add_argument("--num_classes",default=1,help="the number of classes")
paraser.add_argument("--gpu_memory_fraction",default=0.8,help="the memory of gpu")
args = paraser.parse_args()


detectionModel = ObjectModel(args)
processBar = ProcessBar()
videoCapture = cv2.VideoCapture("/home/jade/Data/videos/2019-07-16_17-30-52/2019-07-16_17-30-52.mp4")
ret,frame = videoCapture.read()
processBar.count = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))
while ret:
    boxes,labels,labelIds,scores = detectionModel.predict(frame,0.5)
    CVShowBoxes(frame, boxes, labels, labelIds, scores, waitkey=27)
    ret,frame = videoCapture.read()
    NoLinePrint("detecting ...",processBar)


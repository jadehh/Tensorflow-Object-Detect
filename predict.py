#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# 作者：Create on 2019/7/18 16:58 by jade
# 邮箱：jadehh@live.com
# 描述：TODO
# 最近修改：2019/7/18 16:58 modify by jade
import argparse
from jade import *
from objects_model import ObjectModel
import os
import os.path as ops
import cv2
import tensorflow as tf
import time

paraser = argparse.ArgumentParser(description="Detect car")
#genearl
paraser.add_argument("--model_path",default="/home/jade/Models/objectDetectionModels/ssd_mobilenet_v2_hand_2019-07-18/pb/ssd_mobilenet_v2_hand_pb/frozen_inference_graph.pb",help="path to load model")
paraser.add_argument("--label_path",default="/home/jade/label_map/hand.prototxt",help="path to labels")
paraser.add_argument("--num_classes",default=1,help="the number of classes")
paraser.add_argument("--gpu_memory_fraction",default=0.8,help="the memory of gpu")
args = paraser.parse_args()


detectionModel = ObjectModel(args)

image_list = GetAllImagesPath("/home/jade/Data/HAND/DeepFreeze_Hand/JPEGImages")
processBar = ProcessBar()
processBar.count = len(image_list)

for image_path in image_list:
    processBar.start_time = time.time()
    image = cv2.imread(image_path)
    boxes,labels,labelIds,scores = detectionModel.predict(image,0.8)
    CVShowBoxes(image, boxes, labels, labelIds, scores, waitkey=1000)
    #GenerateXml(GetLastDir(image_path)[:-4],image.shape,boxes,labelIds,labels,"/home/jade/Data/StaticDeepFreeze/2019-04-10_15-46-19/Annotations")
    #image = CVShowBoxes(cv2.imread(image_path),boxes,labels,labelIds,scores,waitkey=False)
    #cv2.imwrite("/home/jade/Data/StaticDeepFreeze/2019-04-02_101753_288/CombinedImages/"+GetLastDir(image_path),image)
    NoLinePrint("detecting ...",processBar)
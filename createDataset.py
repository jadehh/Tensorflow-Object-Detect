#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 作者：Create on 2019/7/18 14:31 by jade
# 邮箱：jadehh@live.com
# 描述：制作数据集
# 最近修改：2019/7/18 14:31 modify by jade

import tensorflow as tf
from datasetopeation.jadeVocTFRecord import CreateVOCTFRecords
import os
import argparse
from jade import *

def CreatTF():
    years = []
    for year in os.listdir("/home/jade/Data/FaceGesture/"):
        if year != "tfrecords" and os.path.isdir(os.path.join("/home/jade/Data/FaceGesture/",year)):
            years.append(year)
    paraser = argparse.ArgumentParser(description="Create TFRecords")
    paraser.add_argument("--data_dir",default="/home/jade/Data/FaceGesture/",help="")
    paraser.add_argument("--output_path",default="/home/jade/Data/FaceGesture/tfrecords/hand_gesture_train_"+GetToday()+".tfrecord",help="")
    paraser.add_argument("--proto_txt_path",default="/home/jade/Data/FaceGesture/face_gesture.prototxt",help="")
    paraser.add_argument("--years",type=list,default=years,help="")
    args = paraser.parse_args()
    CreateVOCTFRecords(args)
if __name__ == '__main__':
    # CreateVOCDataset("/home/jade/Data/HAND/DeepFreeze_Hand","DeepFreeze_Hand")
    CreatTF()
    #CutVoc()
    #CreateVOCDataset("/home/jade/Data/StaticDeepFreeze/2019-04-10","2019-04-10")
    # restoreVoc()
    #GeneratePrototxt()

    print("Done")
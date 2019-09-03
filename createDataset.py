#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 作者：Create on 2019/7/18 14:31 by jade
# 邮箱：jadehh@live.com
# 描述：制作数据集
# 最近修改：2019/7/18 14:31 modify by jade

from datasetopeation.jadeVocTFRecord import CreateVOCTFRecords
import argparse
from jade import *

def CreatTF():
    years = []
    root_dir = "/home/jade/Data/GestureFace/"
    for year in os.listdir(root_dir):
        if year != "tfrecords" and os.path.isdir(os.path.join(root_dir,year)):
            years.append(year)
            #dataset_name = os.path.join(root_dir,year)
            #CreateVOCDataset(dataset_name, "FaceGesture")
    paraser = argparse.ArgumentParser(description="Create TFRecords")
    paraser.add_argument("--data_dir",default=root_dir,help="")
    paraser.add_argument("--output_path",default=root_dir +"/tfrecords/face_train_"+GetToday()+".tfrecord",help="")
    paraser.add_argument("--proto_txt_path",default=root_dir + "/face.prototxt",help="")
    paraser.add_argument("--years",type=list,default=years,help="")
    args = paraser.parse_args()
    CreateVOCTFRecords(args)
if __name__ == '__main__':

    CreatTF()
    #VOCTFRecordShow("/home/jade/Data/GestureFace/tfrecords/gesture_hand_train_2019-08-21.tfrecord","/home/jade/Data/GestureFace/gesture_face.prototxt")
    #CutVoc()
    #CreateVOCDataset("/home/jade/Data/StaticDeepFreeze/2019-04-10","2019-04-10")
    # restoreVoc()
    #GeneratePrototxt()

    print("Done")
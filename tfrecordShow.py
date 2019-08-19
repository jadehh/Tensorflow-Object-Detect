#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# 作者：Create on 2019/7/18 15:57 by jade
# 邮箱：jadehh@live.com
# 描述：TODO
# 最近修改：2019/7/18 15:57 modify by jade
import tensorflow as tf
import numpy as np
import io
from PIL import Image
from jade import *

def voc_tfrecords_show(tfrecord_path):
    categories,_ = ReadProTxt("/home/jade/Data/Hand_Gesture/hand_gesture.prototxt")
    with tf.Session() as sess:
        example = tf.train.Example()
        # train_records 表示训练的tfrecords文件的路径
        record_iterator = tf.python_io.tf_record_iterator(path=tfrecord_path)
        for record in record_iterator:
            example.ParseFromString(record)
            f = example.features.feature
            # 解析一个example
            # image_name = f['image/filename'].bytes_list.value[0]
            image_encode = f['image/encoded'].bytes_list.value[0]
            image_height = f['image/height'].int64_list.value[0]
            image_width = f['image/width'].int64_list.value[0]
            xmin = f['image/object/bbox/xmin'].float_list.value
            ymin = f['image/object/bbox/ymin'].float_list.value
            xmax = f['image/object/bbox/xmax'].float_list.value
            ymax = f['image/object/bbox/ymax'].float_list.value
            labels = f['image/object/class/label'].int64_list.value
            text = f['image/object/class/text'].bytes_list.value
            image = io.BytesIO(image_encode)
            labels = list(labels)
            xmin = list(xmin)
            ymin = list(ymin)
            xmax = list(xmax)
            ymax = list(ymax)
            image = Image.open(image)

            image = np.asarray(image)
            bboxes = []
            scores = []
            label_texts = []
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            for i in range(len(xmin)):
                print(labels[i])
                bboxes.append(
                        (xmin[i] * image_width, ymin[i] * image_height, xmax[i] * image_width, ymax[i] * image_height))
                scores.append(1)
                label_texts.append(categories[labels[i]]["display_name"])

                #label_texts.append("good")
            print("**********************")
            CVShowBoxes(image,bboxes,label_texts,labels,scores=scores,waitkey=0)
if __name__ == '__main__':
    tfrecords_path = "/home/jade/Data/FaceGesture/tfrecords/hand_gesture_train_2019-08-16.tfrecord"
    voc_tfrecords_show(tfrecords_path)

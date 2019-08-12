#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 作者：Create on 2019/7/18 14:31 by jade
# 邮箱：jadehh@live.com
# 描述：制作数据集
# 最近修改：2019/7/18 14:31 modify by jade

import tensorflow as tf
from jade.jadeTFRecords import main


def CreatTF():
    flags = tf.app.flags
    flags.DEFINE_string('data_dir', '/home/jade/Data/Hand_Gesture/', 'Root directory to raw PASCAL VOC dataset.')
    flags.DEFINE_string('set', 'train', 'Convert training set, validation set or '
                                        'merged set.')
    flags.DEFINE_string('output_path', '/home/jade/Data/Hand_Gesture/Tfrecords/hand_gesture_train.tfrecord', 'Path to output TFRecord')
    flags.DEFINE_string('label_map_path', "/home/jade/Data/Hand_Gesture/hand_gesture.prototxt",
                        'Path to label map proto')
    flags.DEFINE_list('years',  ["UA_Handgesture"],
                        'Path to label map proto')
    FLAGS = flags.FLAGS
    main(FLAGS)


def CutVoc():
    CutImagesWithBox("/home/jade/Data/UA+/worksite_2019-04-30_d6",savedir="/home/jade/Data/UA+/worksite_2019-04-30_d6_cut",use_chinese_name=False)



def restoreVoc():
    RestoreCutImageWithVoc("/home/jade/Data/UA+/worksite_2019-04-30_d7","/home/jade/Data/UA+/worksite_2019-04-30_d7_cut")


if __name__ == '__main__':
    # CreateVOCDataset("/home/jade/Data/HAND/DeepFreeze_Hand","DeepFreeze_Hand")
    CreatTF()
    #CutVoc()
    #CreateVOCDataset("/home/jade/Data/StaticDeepFreeze/2019-04-10","2019-04-10")
    # restoreVoc()
    #GeneratePrototxt()

    print("Done")
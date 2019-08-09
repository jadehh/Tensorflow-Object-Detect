#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# 作者：2019/8/9 by jade
# 邮箱：jadehh@live.com
# 描述：TODO
# 最近修改：2019/8/9  上午9:49 modify by jade

from jade.voc0712 import VOCDetection,AnnotationTransform
testset = VOCDetection(
    "/home/jade/Data/HAND", [('DeepFreeze_Hand', 'test_var')], None, AnnotationTransform())
from jade import *
import argparse
from objects_model import ObjectModel
car_paraser = argparse.ArgumentParser(description="Detect car")
#genearl
car_paraser.add_argument("--model_path",default="/home/jade/Models/objectDetectionModels/ssd_mobilenet_v1_hand_2019-08-05/",help="path to load model")
car_paraser.add_argument("--label_path",default="/home/jade/Data/HAND/Hand.prototxt",help="path to labels")
car_paraser.add_argument("--num_classes",default=1,help="the number of classes")
car_paraser.add_argument("--gpu_memory_fraction",default=0.8,help="the memory of gpu")
args = car_paraser.parse_args()
detectionModel = ObjectModel(args)
num_images = len(testset)
num_classes = args.num_classes + 1
all_boxes = [[[] for _ in range(num_images)]
             for _ in range(num_classes)]
max_per_image = 300
processbar = ProcessBar()
processbar.count = num_images
for i in range(num_images):
    processbar.start_time = time.time()
    img,imagename = testset.pull_image(i)
    scale = ([img.shape[1], img.shape[0],
                          img.shape[1], img.shape[0]])
    bboxes_out, label_text, classes_out, scores_out,all_boxes = detectionModel.predict_all_boxes(img,i,all_boxes,0.6)
    #filename, shape, bboxes, labels, labels_text, save_path
    #GenerateXml(GetLastDir(imagename)[:-4],img.shape,bboxes_out,classes_out,label_text,os.path.join("/home/jade/Data/StaticDeepFreeze/2019-03-19_15-20-20",DIRECTORY_PREANNOTATIONS))
    # CVShowBoxes(img,bboxes_out,label_text,classes_out,scores_out,waitkey=400)
    NoLinePrint("Detecting ...",processbar)
    if max_per_image > 0:
        image_scores = np.hstack([all_boxes[j][i][:, -1] for j in range(1, num_classes)])
        if len(image_scores) > max_per_image:
            image_thresh = np.sort(image_scores)[-max_per_image]
            for j in range(1, num_classes):
                keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                all_boxes[j][i] = all_boxes[j][i][keep, :]


testset.evaluate_detections(all_boxes)
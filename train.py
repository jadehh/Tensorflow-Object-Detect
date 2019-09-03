#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 作者：Create on 2019/7/18 14:31 by jade
# 邮箱：jadehh@live.com
# 描述：目标检测训练文件
# 最近修改：2019/7/18 14:31 modify by jade


import functools
import os
import tensorflow as tf
import json
from object_detection.builders import dataset_builder
from object_detection.utils import config_util
from object_detection.builders import model_builder
from object_detection.legacy import trainer
from jade.jade_tools import GetToday,GetRootPath
tf.logging.set_verbosity(tf.logging.INFO)


def train_car(train_dir="models/faster_car_models",
              pipeline_config_path="pipeline/faster_rcnn_resnet101_car.config",
              num_clones = 1,
              clone_on_cpu = False):


    if pipeline_config_path:
        configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
        if os.path.exists(train_dir) is not True:
            os.makedirs(train_dir)
        tf.gfile.Copy(pipeline_config_path,
                          os.path.join(train_dir, 'pipeline.config'),
                          overwrite=True)

    model_config = configs['model']
    train_config = configs['train_config']
    input_config = configs['train_input_config']

    model_fn = functools.partial(
        model_builder.build,
        model_config=model_config,
        is_training=True)

    def get_next(config):
        return dataset_builder.make_initializable_iterator(
            dataset_builder.build(config)).get_next()

    create_input_dict_fn = functools.partial(get_next, input_config)

    env = json.loads(os.environ.get('TF_CONFIG', '{}'))
    cluster_data = env.get('cluster', None)
    cluster = tf.train.ClusterSpec(cluster_data) if cluster_data else None
    task_data = env.get('task', None) or {'type': 'master', 'index': 0}
    task_info = type('TaskSpec', (object,), task_data)

    # Parameters for a single worker.
    ps_tasks = 0
    worker_replicas = 1
    worker_job_name = 'lonely_worker'
    task = 0
    is_chief = True
    master = ''
    graph_rewriter_fn = None

    trainer.train(
        create_input_dict_fn,
        model_fn,
        train_config,
        master,
        task,
        num_clones,
        worker_replicas,
        clone_on_cpu,
        ps_tasks,
        worker_job_name,
        is_chief,
        train_dir,
        graph_hook_fn=graph_rewriter_fn)

if __name__ == '__main__':
    day = GetToday()
    # train_dir = "/data/home/jdh/models/HandsModels/faster_rcnn_resnet101_hand2" + "_" + day
    train_dir = GetRootPath() + "Models/HandGesuteModel/sfaster_rcnn_resnet101_hand_gesture_" + day
    pipeline_config_path = GetRootPath() + "pipeline/faster_rcnn_resnet101_hand_gesture.config"
    train_car(train_dir,pipeline_config_path)

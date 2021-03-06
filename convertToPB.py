#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# 作者：Create on 2019/7/18 17:34 by jade
# 邮箱：jadehh@live.com
# 描述：ckpt文件转pb文件
# 最近修改：2019/7/18 17:34 modify by jade

import tensorflow as tf
from google.protobuf import text_format
from object_detection import exporter
from object_detection.protos import pipeline_pb2
from jade import *
slim = tf.contrib.slim
flags = tf.app.flags

"""
python convertToPB.py \
    --input_type image_tensor \
    --input_shape 1,300,300,3 \
    --pipeline_config_path /home/jade/pipeline/ssd_mobilenet_v1_hand.config \
    --trained_checkpoint_prefix /home/jade/Models/objectDetectionModels/ssd_mobilenet_v1_hand_2019-08-05/model.ckpt-41568 \
    --output_directory /home/jade/Models/objectDetectionModels/ssd_mobilenet_v1_hand_2019-08-05/pb/

"""
pipline_config_path = "/home/jade/Models/GestureFaceModels/ssd_mobilenet_v1_gesture_face_2019-08-21/pipeline.config"
train_checkpoint_prefix = "/home/jade/Models/GestureFaceModels/ssd_mobilenet_v1_gesture_face_2019-08-21/model.ckpt-200000"

output_directory = os.path.join(GetPreviousDir(train_checkpoint_prefix),"pb/")
flags.DEFINE_string('input_type', 'image_tensor', 'Type of input node. Can be '
                    'one of [`image_tensor`, `encoded_image_string_tensor`, '
                    '`tf_example`]')
flags.DEFINE_string('input_shape', '-1,300,300,3',
                    'If input_type is `image_tensor`, this can explicitly set '
                    'the shape of this input tensor to a fixed size. The '
                    'dimensions are to be provided as a comma-separated list '
                    'of integers. A value of -1 can be used for unknown '
                    'dimensions. If not specified, for an `image_tensor, the '
                    'default shape will be partially specified as '
                    '`[None, None, None, 3]`.')
flags.DEFINE_string('pipeline_config_path', pipline_config_path,
                    'Path to a pipeline_pb2.TrainEvalPipelineConfig config '
                    'file.')
flags.DEFINE_string('trained_checkpoint_prefix', train_checkpoint_prefix,
                    'Path to trained checkpoint, typically of the form '
                    'path/to/model.ckpt')
flags.DEFINE_string('output_directory', output_directory, 'Path to write outputs.')
flags.DEFINE_string('config_override', '',
                    'pipeline_pb2.TrainEvalPipelineConfig '
                    'text proto to override pipeline_config_path.')
flags.DEFINE_boolean('write_inference_graph', False,
                     'If true, writes inference graph to disk.')
tf.app.flags.mark_flag_as_required('pipeline_config_path')
tf.app.flags.mark_flag_as_required('trained_checkpoint_prefix')
tf.app.flags.mark_flag_as_required('output_directory')
FLAGS = flags.FLAGS


def main(_):
  pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
  with tf.gfile.GFile(FLAGS.pipeline_config_path, 'r') as f:
    text_format.Merge(f.read(), pipeline_config)
  text_format.Merge(FLAGS.config_override, pipeline_config)
  for dim in FLAGS.input_shape.split(','):
      print(dim)
  if FLAGS.input_shape:
    input_shape = [
        int(dim) if dim != '-1' else None
        for dim in FLAGS.input_shape.split(',')
    ]
  else:
    input_shape = None
  exporter.export_inference_graph(
      FLAGS.input_type, pipeline_config, FLAGS.trained_checkpoint_prefix,
      FLAGS.output_directory, input_shape=input_shape,
      write_inference_graph=FLAGS.write_inference_graph)


if __name__ == '__main__':
  tf.app.run()

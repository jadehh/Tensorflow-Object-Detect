# Tensorflow-Object-Detect
## tensorflow 目标检测
### 1. 安装tensorflow 的 object_detection
[安装地址](https://github.com/tensorflow/models) 
### 2. 制作TFRecord数据集
```
python createDataset.py
```
### 3. TFRecord的验证
```
python tfrecordShow.py
```
### 4. 制作piplineconfig文件
```
主要修改几个地方
1. num_classes = 21
2. batch_size = 32
3. fine_tune_checkpoint 
4. input_path
5. label_map_path
需要注意的是当训练SSD MobilenetV2 会报错，需要将数据增强的方法注释掉
```
### 5. 开始训练
```
python train.py
或者使用
bash run.py (可以指定使用GPU的个数)
```
### 6. 预测
[模型下载](http://192.168.1.200)
```
python predict.py

```

### 7. 模型转换 ckpt 转 pb
```
python convertToPB.py \
    --input_type image_tensor \
    --pipeline_config_path /home/jade/pipeline/ssd_mobilenet_v2_hand.config \
    --trained_checkpoint_prefix /home/jade/Models/objectDetectionModels/ssd_mobilenet_v2_hand_2019-07-18/model.ckpt-1819 \
    --output_directory /home/jade/Models/objectDetectionModels/ssd_mobilenet_v2_hand_2019-07-18/
```

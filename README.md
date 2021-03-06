# Tensorflow-Object-Detect
## tensorflow 目标检测
### 1. 安装tensorflow 的 object_detection
[安装地址](https://github.com/tensorflow/models) 
```
cd research/
python setup.py install
cd slim/
python setup.py install
```

### 2. 制作TFRecord数据集
```
python createDataset.py
```
### 3. TFRecord的验证
```
python tfrecordShow.py

```
数据集下载 [hand.tfrecord]()
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
piplineconfig 下载地址为 [ssd_mobilenet_v2_hand.config](https://pan.baidu.com/s/1PePTECFR7Ts_kdLYEUW9PQ) (提取码为86hz)
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
    --input_shape 1,300,300,3 \
    --pipeline_config_path /home/jade/pipeline/ssd_mobilenet_v1_hand.config \
    --trained_checkpoint_prefix /home/jade/Models/objectDetectionModels/ssd_mobilenet_v1_hand_2019-08-05/model.ckpt-41568 \
    --output_directory /home/jade/Models/objectDetectionModels/ssd_mobilenet_v1_hand_2019-08-05/pb/
```


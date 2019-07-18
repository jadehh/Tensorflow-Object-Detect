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
```
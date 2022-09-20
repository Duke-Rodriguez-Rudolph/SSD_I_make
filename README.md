# SSD_I_make
SSD which I write
train.py包括训练的代码，一些基础的配置
datasets.py包括数据集的加载，格式为VOC，且有数据增强的内容在里面（有两个，其中一个自己写的，另外一个移植成功代码的数据增强部分）
eval.py评估文件，包括判断结果是不是正样例和AP的计算（由于全部判断为负样本，所以计算AP的函数基本没跑过）
model.py模型和损失函数
utils.py一些工具函数（IOU，创建先验框）
predict.py其实就是训练文件中的val部分

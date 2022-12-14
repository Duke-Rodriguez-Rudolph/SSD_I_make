# 训练文件
from model import SSD,LossFunction
from datasets import VOCDatasets,MAPDatasets
from eval import is_positive,CaculateAP
import utils as utils
import os
import torch
from torchvision import transforms, datasets
import torchvision.ops as ops

# 损失函数(完成)
# 数据集读取(完成)
# 训练过程
# 评估过程

# 统一一个规定格式:[batch_size,8732,channels]、[batch_size,channels,w,h]
# 基础配置
input_size = [300,300] # 就是[w,h]
batch_size=1
conf_thresh=0.5
iou_thresh=0.4
datasets_path=r'/Datasets/VOCtrainval_11-May-2012/VOCdevkit'
class_num=20
class_names=[
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor'
]
# 先验框设定
feature_map_size = [38, 19, 10, 5, 3, 1]
sk = [30, 60, 111, 162, 213, 264, 315]
ar = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
# 计算设备
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 创建先验框
prior=utils.createPrior(input_size,feature_map_size,sk,ar)
# 读取数据集
val_dataset = VOCDatasets(datasets_path,class_names,prior, "2012", False,'val.txt')
val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                              batch_size=batch_size,
                                              shuffle=True)
map_dataset=MAPDatasets(datasets_path,class_names,"2012",'val.txt')
# 构建网络
model=SSD(class_num)
print(model)
# 加载权重
#model.loadVGG('./vgg-16.pth')
model.load_state_dict(torch.load('./bestepoch2.pth'))
# 转移设备
model.to(device)
print(device)


# 开始验证
model.eval()
best_mAP=0.0
all_step_classifi=[]
all_step_class_num=[]
new_prior=prior.clone()
new_prior[:, 2] = new_prior[:, 2] - new_prior[:, 0]
new_prior[:, 3] = new_prior[:, 3] - new_prior[:, 1]
new_prior[:, 0] = new_prior[:, 0] + new_prior[:, 2] / 2
new_prior[:, 1] = new_prior[:, 1] + new_prior[:, 3] / 2
with torch.no_grad():
    for step, map_data in enumerate(map_dataset):
        map_image,map_boxes=map_data
        classifi_out, location_out = model(map_image.to(device))
        # 解码网络输出 list[N,4],list[N,2]
        location_list,classifi_list=utils.encode(classifi_out, location_out,prior.to(device),conf_thresh)
        if len(location_list)==0:
            continue
        for index in range(len(location_list)):
            classifi=classifi_list[index].cpu()
            location=location_list[index].cpu()
            keep=ops.boxes.batched_nms(location,classifi[:,1],classifi[:,0].long(), 0.3)
            #print('keep:',keep.shape)
            # if (keep.shape[0]<=2):
            # print(classifi)
            #     print(location)
            classifi_list[index] = classifi[keep, :].to(device)
            location_list[index] = location[keep, :].to(device)
        classifi_step,all_class_num=is_positive(location_list,classifi_list,map_boxes.to(device))
        print('classifi_step:',classifi_step)
        if classifi_step==None:
            continue
        all_step_classifi.append(classifi_step)
        all_step_class_num.append(all_class_num)

if(len(all_step_classifi)==0):
    print('no all_step_classifi')
#print('all_step_classifi:',all_step_classifi)
mAP=CaculateAP(all_step_classifi, all_step_class_num, class_names)


print('mAP')

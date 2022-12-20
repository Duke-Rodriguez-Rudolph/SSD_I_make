# 工具
import torch
import math
from functools import partial
from math import sqrt

def IOU(box1,box2):
    # box为[N,4],其中4为[xmin,ymin,xmax,ymax]
    # 得到的IOU矩阵为row:box1.N col:box2.N
    row = box1.size(0)
    col = box2.size(0)

    max_xy = torch.min(box1[:, 2:].unsqueeze(1).expand(row, col, 2),
                       box2[:, 2:].unsqueeze(0).expand(row, col, 2))
    min_xy = torch.max(box1[:, :2].unsqueeze(1).expand(row, col, 2),
                       box2[:, :2].unsqueeze(0).expand(row, col, 2))
    # 交集部分
    inter = torch.clamp(max_xy - min_xy, min=0)
    inter = inter[:, :, 0] * inter[:, :, 1]

    # 计算每个矩形的面积
    area_box1 = ((box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])).unsqueeze(1).expand_as(inter)
    area_box2 = ((box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])).unsqueeze(0).expand_as(inter)

    # 计算并集部分
    union = area_box1 + area_box2 - inter

    # IOU矩阵
    IOU_matrix = inter / union

    return IOU_matrix

# 创建先验框
def createPrior(input_size, feature_map_size, sk, ar):
    # input_size = [300,300] 就是[w,h]
    # feature_map_size = [38, 19, 10, 5, 3, 1]
    # sk = [30, 60, 111, 162, 213, 264, 315]
    # ar = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]

    prior = []
    # 共六种特征图
    for k in range(6):
        # 制造每个格子的先验框
        for row in range(feature_map_size[k]):
            for col in range(feature_map_size[k]):
                # 设置先验框中点位置为格子中心
                x = (col + 0.5) / feature_map_size[k] * input_size[1]
                y = (row + 0.5) / feature_map_size[k] * input_size[0]
                # 先制作长宽比为2、1/2、3、1/3的先验框
                for a in ar[k]:
                    w = sk[k] * sqrt(a)
                    h = sk[k] / sqrt(a)
                    prior.append([x, y, w, h])
                    prior.append([x, y, h, w])
                # 制作长宽比为1的两个先验框
                prior.append([x, y, sk[k], sk[k]])
                prior.append([x, y, sqrt(sk[k] * sk[k + 1]), sqrt(sk[k] * sk[k + 1])])

    # 转为tensor
    prior = torch.Tensor(prior)
    # 转为xyxy格式
    prior[:, 0] = prior[:, 0] - prior[:, 2] / 2
    prior[:, 1] = prior[:, 1] - prior[:, 3] / 2
    prior[:, 2] = prior[:, 0] + prior[:, 2]
    prior[:, 3] = prior[:, 1] + prior[:, 3]

    prior/=input_size[0]
    prior=torch.clamp(prior,min=0, max=1)
    # prior为[anchor_num,4]或者说是[8732,4]
    return prior

# 对输出结果进行解码
def encode(classifi_out, location_out,prior,conf_thresh):
    # classifi_out:[batch,8732,class_num+1]
    # location_out:[batch,8732,4]
    # prior:[8732,4]
    class_num=classifi_out.shape[-1]-1
    location_list=[]
    classifi_list=[]

    # list [1,8732,4]
    location_out = location_out.split(1, 0)
    # list [1，8732,class_num]
    classifi_out = classifi_out.split(1, 0)
    for i in range(len(location_out)):
        # tensor [8732,class_num+1]
        classifi=classifi_out[i].view((-1,class_num+1))
        # tensor [8732,4]
        location=location_out[i].view((-1,4))
        new_prior=prior.clone()
        # xyxy2xywh
        new_prior[:, 2] = new_prior[:, 2] - new_prior[:, 0]
        new_prior[:, 3] = new_prior[:, 3] - new_prior[:, 1]
        new_prior[:, 0] = new_prior[:, 0] + new_prior[:, 2] / 2
        new_prior[:, 1] = new_prior[:, 1] + new_prior[:, 3] / 2
        location[:, :2]*=0.1
        location[:, 2:]*=0.2
        # x
        location[:, 0] = (new_prior[:, 0] + new_prior[:, 2] * location[:, 0])
        # y
        location[:, 1] = (new_prior[:, 1] + new_prior[:, 3] * location[:, 1])
        # w
        location[:, 2] = torch.exp(location[:, 2]) * new_prior[:, 2]
        # h
        location[:, 3] = torch.exp(location[:, 3]) * new_prior[:, 3]
        location*=300

        # 剔除背景
        back_mask=torch.max(classifi,dim=1)[1]>0
        
        location = location[back_mask, :]
        # 计算百分比 [N,class_num+1]
        classifi=torch.softmax(classifi[back_mask,:],dim=1)
        # 获取分类结果，classifi_result结果为[N]，为N个分类序号
        if classifi.shape[0]==0:
            continue
        classifi_score,classifi_result=torch.max(classifi,dim=1)
        # 剔除那些小于阈值的结果
        conf_mask=(classifi_score>=conf_thresh)
        classifi_result=classifi_result[conf_mask]
        location_result=location[conf_mask,:]

        # 剔除背景结果组合在一起，变为[N,2]
        classifi_result=torch.stack((classifi_result,classifi_score[conf_mask]),dim=0).transpose(0,1)


        # xmin,ymin
        location_result[:, :2]=location_result[:, :2]-location_result[:, 2:]/2
        # xmax,ymax
        location_result[:, 2:]=location_result[:, :2]+location_result[:, 2:]

        location_list.append(location_result)
        classifi_list.append(classifi_result)


    return location_list,classifi_list

def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.05, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.05, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi* (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func
   
def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

import torch
import utils as utils

def is_positive(location_result,classifi_result,location_label,classifi_label):
    # location_result为定位结果，list[object,4]
    # classifi_result为分类结果，list[object,2] 2:[class_index,classifi_score]
    # location_label为定位标签，[batch_size,8732,4]
    # classifi_label为分类标签，[batch_size,8732]

    # 遍历每一个output，然后与所有同分类的label计算IOU，如果大于阈值，则为正比例。（使用mask）
    # classifi_label变为list[1,8732]
    classifi_label=classifi_label.split(1, 0)
    # location_label变为list[1,8732,4]
    location_label=location_label.split(1, 0)

    new_classifi_list=[]
    all_class_num=[]
    # 计算IOU
    for i in range(len(classifi_label)):
        # 获得所有正样本的掩码
        mask=classifi_label[i]>0

        # 先计算IOU矩阵
        output=location_result[i]
        label=location_label[i][mask,:]
        if output.shape[0]==0:
            continue
        IOU_matrix=utils.IOU(label,output)

        # 区分正负例
        output=classifi_result[i][:,0] # tensor[object]
        label=classifi_label[i][mask] # tensor[object]

        output_mask = output.unsqueeze(0).expand_as(IOU_matrix)
        label_mask = label.unsqueeze(-1).expand_as(IOU_matrix)
        # 将不同分类的置零
        IOU_matrix[output_mask != label_mask] = 0
        # IOU不够0.5的置零
        IOU_matrix[IOU_matrix < 0.5] = 0
        # 选出正负例
        value, _ = torch.max(IOU_matrix, dim=0)
        # 所有大于0的置1，此时正例为1，负例为0
        value[value>0]=1
        # 新的分类结果，有三维，[分类结果，置信度，正负例]
        value=value.view(-1,1)
        new_classifi_result=torch.cat((classifi_result[i],value),dim=1)
        new_classifi_list.append(new_classifi_result)
        #class_num=torch.max(new_classifi_result[:,0],dim=0)[0]
        label_num=[]
        for j in range(20):
            label_num.append(int((label==j+1).float().sum()))
        all_class_num.append(label_num)

    if len(new_classifi_list)==0:
        return None,None
    classifi_list=torch.cat(new_classifi_list,dim=0) # [object,3]
    all_class_num=torch.sum(torch.Tensor(all_class_num),dim=0) # [class_num]
    return classifi_list,all_class_num


def CaculateAP(classifi_result,all_class_num,class_names):
    # classifi_result为[object,3] [分类结果，置信度，正负例]
    # all_class_num为[class_num]
    class_num=len(class_names)
    mAP=0
    # 计算每个分类的AP
    for i in range(class_num):
        same_class = classifi_result[classifi_result[:, 0] == i + 1]
        indices = torch.sort(same_class[:, 1], descending=True)[1]
        # 排好序得的正负例
        is_correct = same_class[:, 2][indices]
        # 最大精度
        max_precision = 0
        # 上次召回
        old_recall = 0
        # 正例数量
        positive_num = 0
        # ap值
        ap = 0
        # 这次召回减去上去召回的间隔
        detla = 0
        # 计算分类的数量
        label_num = int(all_class_num[i])
        if label_num==0:
            continue
        for index, correct in enumerate(is_correct):
            # 如果值为1，则正例数量加一
            if correct == 1:
                positive_num += 1
            # 计算精度与召回
            now_precision = positive_num / (index + 1)
            now_recall = positive_num / label_num
            # 如果新召回大于久召回
            if now_recall > old_recall:
                # 计算间隔
                detla = now_recall - old_recall
                # ap增加
                ap += max_precision * detla
                # 赋新值
                max_precision = now_precision
                old_recall = now_recall
            else:
                if now_precision > max_precision:
                    max_precision = now_precision
            if index + 1 == is_correct.size(0):
                ap += max_precision * detla
        mAP += ap
        print(class_names[i] + ' AP:', ap)

    print("_________________________________________")

    mAP = mAP / class_num
    print("mAP:", mAP)
    return mAP

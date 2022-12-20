# 训练文件
from model import SSD,LossFunction
from datasets import VOCDatasets,MAPDatasets
from eval import is_positive,CaculateAP
import utils as utils
import os
import torch
from torchvision import transforms, datasets
import torchvision.ops as ops
from tensorboardX import SummaryWriter

# 损失函数(完成)
# 数据集读取(完成)
# 训练过程
# 评估过程

# 统一一个规定格式:[batch_size,8732,channels]、[batch_size,channels,w,h]
# 基础配置
input_size = [300,300] # 就是[w,h]
batch_size=8
conf_thresh=0.01
iou_thresh=0.4
init_lr=2e-3
momentum=0.937
weight_decay= 5e-4
epoches=200
datasets_path=r'/Datasets/VOCtrainval_11-May-2012/VOCdevkit'
save_path=r'./'
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
prior=utils.createPrior(input_size,feature_map_size,sk,ar) # 先验框为xyxy结构
# 读取数据集
train_dataset = VOCDatasets(datasets_path,class_names,prior,"2012", True,'train.txt')
train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=batch_size,
                                                shuffle=True)

val_dataset = VOCDatasets(datasets_path,class_names,prior, "2012", False,'val.txt')
val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)
map_dataset=MAPDatasets(datasets_path,class_names,"2012",'val.txt')
# 构建记录
writer=SummaryWriter('log',comment='SSD300')
# 构建网络
model=SSD(class_num)
print(model)
# 加载权重
#model.loadVGG('./vgg-16.pth')
model.loadVGG('./vgg16-397923af.pth')
#model.load_state_dict(torch.load('./best0.0069.pth'))
# 转移设备
model.to(device)
print(device)
# 构建损失函数
loss_function=LossFunction()
print('损失函数构建成功')
# 设置学习率
lr_fit=batch_size/64*init_lr
min_lr_fit=batch_size/64*init_lr*0.01
# 构建优化器
optimizer= torch.optim.SGD(params=model.parameters(),lr=lr_fit,momentum=momentum, nesterov=True,weight_decay=weight_decay)
print('构建优化器成功')
# 构建学习率调度器
step_num=10
decay_rate  = (min_lr_fit / lr_fit) ** (1 / (step_num - 1))
step_size   = epoches / step_num
lr_scheduler_func = utils.get_lr_scheduler('cos', lr_fit, min_lr_fit, batch_size)
scheduler= torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=decay_rate, last_epoch=-1)
print('lr:',lr_fit)
print('构建学习率调度器成功')
# 开始训练的循环
print('开始学习')
all_loss=0
model.train()
for epoch in range(epoches):
    for step,data in enumerate(train_data_loader):
        images,targets=data
        # 前向传播
        classifi_out, location_out=model(images.to(device))
        #writer.add_graph(model,(images.to(device),))
        # 梯度清零
        optimizer.zero_grad()
        # 计算损失
        loss=loss_function(location_out,classifi_out,targets["boxes"].to(device),targets["labels"].to(device))
        # 反向传播
        loss.backward()
        optimizer.step()
        all_loss+=loss.item()
        writer.add_scalar('train_loss',loss.item(),len(train_data_loader)*epoch+step+1)
        print('epoch:'+str(epoch+1)+'/'+str(epoches),
              'step:'+str(step+1)+'/'+str(len(train_data_loader)),
              'loss:'+str(loss.item()))
    average_loss=all_loss/(step+1)
    print('average_loss:',average_loss)
    torch.save(model.state_dict(), os.path.join(save_path, 'best_can.pth'))
    all_loss=0
    # 学习率更新
    #utils.set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
    scheduler.step()
    print('lr:',optimizer.state_dict()['param_groups'][0]['lr'])
    writer.add_scalar('lr',float(optimizer.state_dict()['param_groups'][0]['lr']),epoch+1)

    # 开始验证
    model.eval()
    best_mAP=0.0
    val_loss=0.0
    all_step_classifi=[]
    all_step_class_num=[]
    with torch.no_grad():
        for step, val_data in enumerate(val_data_loader):
            val_images, val_targets = val_data
            classifi_out, location_out = model(val_images.to(device))
            # 计算损失
            loss=loss_function(location_out,classifi_out,val_targets["boxes"].to(device),val_targets["labels"].to(device))
            val_loss+=loss.item()
            writer.add_scalar('val_loss',loss.item(),len(val_data_loader)*epoch+step+1)

        val_average_loss=val_loss/(step+1)
        print('val_average_loss:',val_average_loss)

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
            #print('classifi_step:',classifi_step)
            if classifi_step==None:
                continue
            all_step_classifi.append(classifi_step)
            all_step_class_num.append(all_class_num)
        
    if(len(all_step_classifi)==0):
        print('no all_step_classifi')
        torch.save(model.state_dict(), os.path.join(save_path,'best'+'-epoch'+str(epoch+1)+'|'+str(best_mAP)+'.pth'))
        print('train_loss:',average_loss)
        print('val_loss:',val_average_loss)
        continue
    #print('all_step_classifi:',all_step_classifi)
    #all_step_classifi=torch.cat(all_step_classifi,dim=0)
    #all_step_class_num=sum(all_step_class_num)
    
    mAP=CaculateAP(all_step_classifi, all_step_class_num, class_names)
    if mAP>=best_mAP:
        best_mAP = mAP
        torch.save(model.state_dict(), os.path.join(save_path,'best'+'-epoch'+str(epoch+1)+'|'+str(best_mAP)+'.pth'))
    print('train_loss:',average_loss)
    print('val_loss:',val_average_loss)
print('Finished Training')

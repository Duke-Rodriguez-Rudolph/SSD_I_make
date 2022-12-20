from torch.utils.data import Dataset
import os
import xml.etree.ElementTree as ET
from PIL import Image
from utils import IOU,createPrior

import torch
import cv2
import numpy as np
from torchvision import transforms

# 我们训练要使用VOC格式的数据集
class VOCDatasets(Dataset):
    def __init__(self,root_dir,class_names,priors,year='2007',train_mode=False,train_set='train.txt',iou_thresh=0.5):
        self.priors=priors

        # 数据集根目录，标签名称，数据集年份，数据集转换方式，数据集记载的txt
        self.Annotations_path=os.path.join(root_dir,f"VOC{year}","Annotations")
        self.JPEGImages_path = os.path.join(root_dir, f"VOC{year}", "JPEGImages")

        # 读取训练集
        train_list = os.path.join(root_dir, f"VOC{year}", "ImageSets","Main",train_set)
        with open(train_list,'r') as f:
            self.train_xmls=[os.path.join(self.Annotations_path,line.strip()+'.xml')
                             for line in f.readlines() if len(line.strip()) > 0]

        # 获取标签名称
        self.class_names=class_names

        # 设置是否为训练模式
        self.train_mode=train_mode

        # 记录匹配数据框与先验框时的IOU阈值
        self.iou_thresh=iou_thresh

    def __len__(self):
        # 使用len()可以读取训练集列表的长度
        return len(self.train_xmls)
        
    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a

    def __getitem__(self, index):
        # 构建迭代器
        xml_path=self.train_xmls[index]
        # 读取相关信息
        tree=ET.parse(xml_path)
        root=tree.getroot()
        img_name=root.find("filename").text
        img_width=root.find("size").find("width").text
        img_height=root.find("size").find("height").text
        # 读取图片
        img_path = os.path.join(self.JPEGImages_path, img_name)
        image = Image.open(img_path)

        # 获取位置与标签
        boxes=[]
        labels=[]
        # 获取所有Object子节点，因为每个目标的内容都是写在Object下
        for obj in root.iter("object"):
            # 获取位置信息
            xmin=int(float(obj.find("bndbox").find("xmin").text))
            ymin=int(float(obj.find("bndbox").find("ymin").text))
            xmax=int(float(obj.find("bndbox").find("xmax").text))
            ymax=int(float(obj.find("bndbox").find("ymax").text))
            # 获取标签名
            label_name=obj.find("name").text
            label=self.class_names.index(label_name)
            labels.append(label+1)
            box=[xmin,ymin,xmax,ymax,label+1]
            boxes.append(box)

        image, boxes  = self.get_random_data(image, boxes,(300,300),random = self.train_mode)
        #image  = np.transpose(image-(104, 117, 123), (2, 0, 1))
        image=Image.fromarray(np.uint8(image))
        '''
        if self.train_mode:
            
            
            if torch.rand(1) < 0.5:
                image, boxes, labels = self._RandomResizedCrop(image, boxes, labels, (300, 300))
                image, boxes = self._RandomHorizontalFlip(image, boxes, 0.5)
            else:
                image, boxes = self._Resize(image, boxes, (300, 300))
            
        else:
            # image, boxes = self._Resize(image, boxes, (300,300))
        '''

        #image=transforms.functional.to_tensor(image)
        #image=transforms.functional.normalize(image,(0.5, 0.5, 0.5),(104, 117, 123),False)
        #image_data=np.array(image, dtype=np.int8)

        image = np.transpose((np.array(image, dtype=np.float32))-(104, 117, 123), (2, 0, 1))
        image=np.array(image, np.float32)
        
        labels=boxes[:,-1].tolist() # 加了背景的label值
        boxes=boxes[:,:4].tolist() # 经过变换后的label框
        boxes,labels=self.match(self.priors,boxes,labels,self.iou_thresh)
        '''
        print(boxes)
        for i in range(0,8732):
            box = boxes[i]
            label=labels[i]
            prior=self.priors[i]

            if label==0:
                pass
                #cv2.rectangle(image_data, (int(box[0]*300), int(box[1]*300)), (int(box[2]*300),int(box[3]*300)), color=(0, 0, 255),thickness=1)
            else:
                print(label)
                cv2.rectangle(image_data, (int(prior[0] * 300), int(prior[1] * 300)),
                              (int(prior[2] * 300), int(prior[3] * 300)), color=(0, 0, 255), thickness=2)
                cv2.rectangle(image_data, (int(box[0] * 300), int(box[1] * 300)),
                              (int(box[2] * 300), int(box[3] * 300)), color=(0, 255, 0), thickness=2)
        cv2.imshow('image_data', image_data)

        cv2.waitKey(0)
        '''
        target={}
        target["boxes"]=boxes
        target["labels"]=labels
        return image,target

    # 数据增强
    def get_random_data(self, image,boxes,input_shape, jitter=.3, hue=.1, sat=0.7, val=0.4, random=True):
        #------------------------------#
        #   读取图像并转换成RGB图像
        #------------------------------#
        image   = image.convert('RGB')
        #------------------------------#
        #   获得图像的高宽与目标高宽
        #------------------------------#
        iw, ih  = image.size
        h, w    = input_shape
        #------------------------------#
        #   获得预测框
        #------------------------------#
        boxes     = np.array(boxes)

        if not random:
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2

            #---------------------------------#
            #   将图像多余的部分加上灰条
            #---------------------------------#
            image       = image.resize((nw,nh), Image.BICUBIC)
            new_image   = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data  = np.array(new_image, np.float32)
            

            #---------------------------------#
            #   对真实框进行调整
            #---------------------------------#
            if len(boxes)>0:
                np.random.shuffle(boxes)
                boxes[:, [0,2]] = boxes[:, [0,2]]*nw/iw + dx
                boxes[:, [1,3]] = boxes[:, [1,3]]*nh/ih + dy
                boxes[:, 0:2][boxes[:, 0:2]<0] = 0
                boxes[:, 2][boxes[:, 2]>w] = w
                boxes[:, 3][boxes[:, 3]>h] = h
                box_w = boxes[:, 2] - boxes[:, 0]
                box_h = boxes[:, 3] - boxes[:, 1]
                boxes = boxes[np.logical_and(box_w>1, box_h>1)] # discard invalid box

            return image_data, boxes
                
        #------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        #------------------------------------------#
        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)

        #------------------------------------------#
        #   将图像多余的部分加上灰条
        #------------------------------------------#
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image = new_image

        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        flip = self.rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        image_data      = np.array(image, np.uint8)
        #---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        #---------------------------------#
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        #---------------------------------#
        #   将图像转到HSV上
        #---------------------------------#
        hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype           = image_data.dtype
        #---------------------------------#
        #   应用变换
        #---------------------------------#
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        #---------------------------------#
        #   对真实框进行调整
        #---------------------------------#
        if len(boxes)>0:
            np.random.shuffle(boxes)
            boxes[:, [0,2]] = boxes[:, [0,2]]*nw/iw + dx
            boxes[:, [1,3]] = boxes[:, [1,3]]*nh/ih + dy
            if flip: boxes[:, [0,2]] = w - boxes[:, [2,0]]
            boxes[:, 0:2][boxes[:, 0:2]<0] = 0
            boxes[:, 2][boxes[:, 2]>w] = w
            boxes[:, 3][boxes[:, 3]>h] = h
            box_w = boxes[:, 2] - boxes[:, 0]
            box_h = boxes[:, 3] - boxes[:, 1]
            boxes = boxes[np.logical_and(box_w>1, box_h>1)]

        return image_data, boxes
        
    # 匹配先验框与ground-true框
    def match(self, prior,boxes,labels,iou_thresh):
        # 实际是寻找与先验框对应的gt框
        # 计算两两直接的IOU
        # 创建二维矩阵存储IOU数据
        gt_boxes=torch.Tensor(boxes)
        # 如果label框没东西（随机时被随机掉了）
        if gt_boxes.shape[0]==0:
            print('here!')
            # 克隆一份先验框
            prior_box_results = prior.clone()
            # 计算偏移，让label的格式与output一致，output的格式是xywh的偏移值
            # 先将prior转为xywh格式
            prior_box_results[:, 2] = prior_box_results[:, 2] - prior_box_results[:, 0]
            prior_box_results[:, 3] = prior_box_results[:, 3] - prior_box_results[:, 1]
            prior_box_results[:, 0] = prior_box_results[:, 0] + prior_box_results[:, 2] / 2
            prior_box_results[:, 1] = prior_box_results[:, 1] + prior_box_results[:, 3] / 2
            # 再计算偏移值
            prior_box_results[:, :2] = ((prior_box_results[:, :2] - (prior[:, 2:] + prior[:, :2]) / 2) / (
                        prior[:, 2:] - prior[:, :2]))/0.1
            prior_box_results[:, 2:] = ((prior_box_results[:, 2:] / (prior[:, 2:] - prior[:, :2])).log())/0.2

            prior_label_results = torch.zeros(8732)
            return prior_box_results, prior_label_results

        gt_boxes/=300 # 将box放缩到0-1之间
        IOU_matrix = IOU(gt_boxes,prior)
        # 对每个gt来说，最佳的匹配
        best_prior_for_gt, best_index_for_gt = torch.max(IOU_matrix, dim=1)
        # 对于每个prior来说，最佳的匹配
        best_gt_for_prior, best_index_for_prior = torch.max(IOU_matrix, dim=0)
        # 第一个策略处理
        for gt_index, prior_index in enumerate(best_index_for_gt):
            best_index_for_prior[prior_index] = gt_index
            best_gt_for_prior[prior_index] = 2.0

        # 第二个策略处理
        # 创建蒙版
        masks = best_gt_for_prior > iou_thresh
        prior_label_results = torch.zeros(best_gt_for_prior.shape[0])
        # 将列表转为tensor格式
        gt_labels = torch.Tensor(labels)
        # prior_results是每个先验框对应的类别，如果为0，则代表是负样本
        prior_label_results[masks] = gt_labels[best_index_for_prior[masks]]

        # 克隆一份先验框
        prior_box_results = prior.clone()
        # 将正样本的框修改为gt框，此时为xyxy模式
        prior_box_results[masks, :] = gt_boxes[best_index_for_prior[masks], :]

        # 计算偏移，让label的格式与output一致
        prior_box_results[:, 2] = prior_box_results[:, 2] - prior_box_results[:, 0]
        prior_box_results[:, 3] = prior_box_results[:, 3] - prior_box_results[:, 1]
        prior_box_results[:, 0] = prior_box_results[:, 0] + prior_box_results[:, 2] / 2
        prior_box_results[:, 1] = prior_box_results[:, 1] + prior_box_results[:, 3] / 2

        prior_box_results[:, :2] = (prior_box_results[:, :2] - (prior[:, 2:] + prior[:, :2]) / 2) / (
                    prior[:, 2:] - prior[:, :2])
        prior_box_results[:, 2:] = (prior_box_results[:, 2:] / (prior[:, 2:]-prior[:, :2])).log()
        prior_box_results[:, :2]/=0.1
        prior_box_results[:, 2:]/=0.2

        #[8732,4],[8732]
        return prior_box_results, prior_label_results

    def _RandomResizedCrop(self,img,gt_boxes,labels,size):
        new_boxes = []
        new_labels = []
        num=0
        box=[]
        rect=None
        not_break=True
        while len(new_boxes)==0:
            if num>10:
                rect, new_boxes = self._Resize(img, gt_boxes, size)
                new_labels=labels
                not_break=False
                print('ten!')
                break

            # 获取裁剪区域
            crop = transforms.RandomResizedCrop(size)
            box = list(crop.get_params(img, crop.scale, crop.ratio))

            # 调整奇葩状态
            if box[2] < box[0]:
                temp = box[0]
                box[0] = box[2]
                box[2] = temp
            if box[3] < box[1]:
                temp = box[1]
                box[1] = box[3]
                box[3] = temp

            if box[0]==box[2] or box[1]==box[3]:
                num+=1
                continue
            # 修改标签
            new_boxes = []
            new_labels=[]
            for index in range(len(gt_boxes)):
                gt_box=torch.Tensor(gt_boxes[index])
                label=labels[index]
                gt_box[0]=(gt_box[0] - box[0]) / (box[2] - box[0]) * size[0]
                gt_box[1]=(gt_box[1] - box[1]) / (box[3] - box[1]) * size[1]
                gt_box[2]=(gt_box[2] - box[0]) / (box[2] - box[0]) * size[0]
                gt_box[3]=(gt_box[3] - box[1]) / (box[3] - box[1]) * size[1]

                gt_box=torch.clamp(gt_box,min=0,max=300)
                if gt_box[2] - gt_box[0] != 0 and gt_box[3] - gt_box[1] != 0:
                    new_boxes.append(gt_box.tolist())
                    new_labels.append(label)
            num+=1
        # 截取图片
        if not_break:
            rect = img.crop(box)
            rect = rect.resize(size)


        return rect,new_boxes,new_labels

    def _RandomHorizontalFlip(self,img,gt_boxes,p):
        new_boxes=[]
        # 随机一个数，如果小于概率
        if torch.rand(1) < p:
            # 裁剪图片
            img = transforms.functional.hflip(img)
            for gt_box in gt_boxes:
                # 修改标签
                xmin = img.width - gt_box[2]
                xmax = img.width - gt_box[0]
                gt_box[0] = xmin
                gt_box[2] = xmax
                
                new_boxes.append(gt_box)
        else:
            new_boxes=gt_boxes

        return img,new_boxes

    def _Resize(self,img,gt_boxes,size):
        # 修改标签
        new_boxes = []
        for gt_box in gt_boxes:
            gt_box[0] = int(gt_box[0] / img.width * size[0])
            gt_box[1] = int(gt_box[1] / img.height * size[1])
            gt_box[2] = int(gt_box[2] / img.width * size[0])
            gt_box[3] = int(gt_box[3] / img.height * size[1])

            new_boxes.append(gt_box)

        # 重整
        img=img.resize(size)

        return img, new_boxes


class MAPDatasets():
    def __init__(self,root_dir,class_names,year='2007',train_set='train.txt'):
        # 数据集根目录，标签名称，数据集年份，数据集转换方式，数据集记载的txt
        self.Annotations_path=os.path.join(root_dir,f"VOC{year}","Annotations")
        self.JPEGImages_path = os.path.join(root_dir, f"VOC{year}", "JPEGImages")

        # 读取训练集
        train_list = os.path.join(root_dir, f"VOC{year}", "ImageSets","Main",train_set)
        with open(train_list,'r') as f:
            self.train_xmls=[os.path.join(self.Annotations_path,line.strip()+'.xml')
                             for line in f.readlines() if len(line.strip()) > 0]

        # 获取标签名称
        self.class_names=class_names
        
    def __len__(self):
        # 使用len()可以读取训练集列表的长度
        return len(self.train_xmls)
        
    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a

    def __getitem__(self, index):
        # 构建迭代器
        xml_path=self.train_xmls[index]
        # 读取相关信息
        tree=ET.parse(xml_path)
        root=tree.getroot()
        img_name=root.find("filename").text
        img_width=root.find("size").find("width").text
        img_height=root.find("size").find("height").text
        # 读取图片
        img_path = os.path.join(self.JPEGImages_path, img_name)
        image = Image.open(img_path)

        # 获取位置与标签
        boxes=[]
        labels=[]
        # 获取所有Object子节点，因为每个目标的内容都是写在Object下
        for obj in root.iter("object"):
            # 获取位置信息
            xmin=int(float(obj.find("bndbox").find("xmin").text))
            ymin=int(float(obj.find("bndbox").find("ymin").text))
            xmax=int(float(obj.find("bndbox").find("xmax").text))
            ymax=int(float(obj.find("bndbox").find("ymax").text))
            # 获取标签名
            label_name=obj.find("name").text
            label=self.class_names.index(label_name)
            labels.append(label+1)
            box=[xmin,ymin,xmax,ymax,label+1]
            boxes.append(box)
         
        image, boxes  = self.resize(image, boxes,(300,300))
        image=Image.fromarray(np.uint8(image))
        image = np.transpose((np.array(image, dtype=np.float32))-(104, 117, 123), (2, 0, 1))
        image=np.array(image, np.float32)
        
        return torch.Tensor(image).view(-1,3,300,300),torch.Tensor(boxes)

    def resize(self, image,boxes,input_shape):
        #------------------------------#
        #   读取图像并转换成RGB图像
        #------------------------------#
        image   = image.convert('RGB')
        #------------------------------#
        #   获得图像的高宽与目标高宽
        #------------------------------#
        iw, ih  = image.size
        h, w    = input_shape
        #------------------------------#
        #   获得预测框
        #------------------------------#
        boxes     = np.array(boxes)

        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)
        dx = (w-nw)//2
        dy = (h-nh)//2

        #---------------------------------#
        #   将图像多余的部分加上灰条
        #---------------------------------#
        image       = image.resize((nw,nh), Image.BICUBIC)
        new_image   = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image_data  = np.array(new_image, np.float32)
            

        #---------------------------------#
        #   对真实框进行调整
        #---------------------------------#
        if len(boxes)>0:
            np.random.shuffle(boxes)
            boxes[:, [0,2]] = boxes[:, [0,2]]*nw/iw + dx
            boxes[:, [1,3]] = boxes[:, [1,3]]*nh/ih + dy
            boxes[:, 0:2][boxes[:, 0:2]<0] = 0
            boxes[:, 2][boxes[:, 2]>w] = w
            boxes[:, 3][boxes[:, 3]>h] = h
            box_w = boxes[:, 2] - boxes[:, 0]
            box_h = boxes[:, 3] - boxes[:, 1]
            boxes = boxes[np.logical_and(box_w>1, box_h>1)] # discard invalid box

        return image_data, boxes
        
'''
class_names=[
    'chepai'
]

datasets_path="D:\\test_data"
feature_map_size = [38, 19, 10, 5, 3, 1]
sk = [30, 60, 111, 162, 213, 264, 315]
ar = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
prior=createPrior((300,300),feature_map_size,sk,ar)
train_dataset = VOCDatasets(datasets_path,class_names,prior,"2007",True,'train.txt')
train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=8,
                                                shuffle=True)
for step, data in enumerate(train_data_loader):
    images, labels = data
    print(images.shape)
    print(labels["boxes"].shape)
    print(labels["labels"].shape)
'''

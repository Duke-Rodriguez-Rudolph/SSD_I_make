# SSD网络结构
# 相关引用
import torch.nn as nn
import torch

class L2Norm(nn.Module):
    def __init__(self,channels_num,scale,eps=1e-10):
        super(L2Norm, self).__init__()

        self.gamma=scale
        self.eps=eps
        self.channels_num=channels_num
        self.weight=nn.Parameter(torch.Tensor(channels_num))
        # nn.Parameter就是将后面的东西设定为可学习的参数
        nn.init.constant_(self.weight,self.gamma)
        # 用scale填充随机设定的数（可以看成初始化权重参数）

    def forward(self,x):
        norm=x.pow(2).sum(dim=1,keepdim=True).sqrt()+self.eps # eps防止除一个0
        x=torch.div(x,norm)
        scale = self.weight.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand_as(x)
        # 前面加一个维度，后面加两个维度，变成[1,channels_num,1,1]，然后变成与x一样的维度
        out=scale*x
        return out



class SSD(nn.Module):
    def __init__(self,class_num=20,init_weights=True):
        super(SSD, self).__init__()

        # +1是背景
        class_num+=1
        self.class_num=class_num
        # 因为要从Conv4_3提取特征图，因此将VGG分成上下两部分
        self.VGG1=nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=1), # [batch,64,300,300]
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1, padding=1), # [batch,64,300,300]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2), # [batch,64,150,150]

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),  # [batch,128,150,150]
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),  # [batch,128,150,150]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [batch,128,75,75]

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),  # [batch,256,75,75]
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),  # [batch,256,75,75]
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),  # [batch,256,75,75]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2,ceil_mode=True),  # [batch,256,38,38]
            # 使用ceil_mode可以使最后不满足2*2的区域里选取最大来形成结果，使结果为38*38

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),  # [batch,512,38,38]
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),  # [batch,512,38,38]
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),  # [batch,512,38,38]
            nn.ReLU(inplace=True)
        )

        self.classifier1=nn.Conv2d(in_channels=512,out_channels=4*class_num,kernel_size=3,stride=1,padding=1)#[batch,4*class_num,38,38]
        self.location1=nn.Conv2d(in_channels=512, out_channels=4 * 4, kernel_size=3, stride=1, padding=1)# [batch,4*4,38,38]
        self.l2norm=L2Norm(512,20)

        self.VGG2=nn.Sequential(
            # 看原论文这里，是要保持特征图的尺寸与池化前一致，因此我们像下面这样设置
            nn.MaxPool2d(kernel_size=3,stride=1,padding=1), # [batch,512,38,38]
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),  # [batch,512,38,38]
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),  # [batch,512,38,38]
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),  # [batch,512,38,38]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # [batch,512,19,19]
        )

        self.Conv6=nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=3,stride=1,padding=1), # [batch,1024,19,19]
            nn.ReLU(inplace=True)
        )

        self.Conv7=nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1, stride=1),  # [batch,1024,19,19]
            nn.ReLU(inplace=True)
        )

        self.classifier2 = nn.Conv2d(in_channels=1024, out_channels=6 * class_num, kernel_size=3, stride=1, padding=1)# [batch,6*class_num,19,19]
        self.location2 = nn.Conv2d(in_channels=1024, out_channels=6 * 4, kernel_size=3, stride=1, padding=1)# [batch,6*4,19,19]

        self.Conv8_2 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, stride=1),  # [batch,256,19,19]
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2,padding=1),  # [batch,512,10,10]
            nn.ReLU(inplace=True),
        )

        self.classifier3 = nn.Conv2d(in_channels=512, out_channels=6 * class_num, kernel_size=3, stride=1, padding=1) # [batch,6 * class_num,10,10]
        self.location3 = nn.Conv2d(in_channels=512, out_channels=6 * 4, kernel_size=3, stride=1, padding=1) # [batch,6*4,10,10]

        self.Conv9_2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, stride=1),  # [batch,128,10,10]
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),  # [batch,256,5,5]
            nn.ReLU(inplace=True),
        )

        self.classifier4 = nn.Conv2d(in_channels=256, out_channels=6 * class_num, kernel_size=3, stride=1, padding=1)# [batch,6 * class_num,5,5]
        self.location4 = nn.Conv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, stride=1, padding=1)# [batch,6*4,5,5]

        self.Conv10_2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1),  # [batch,128,5,5]
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),  # [batch,256,3,3]
            nn.ReLU(inplace=True),
        )

        self.classifier5 = nn.Conv2d(in_channels=256, out_channels=4 * class_num, kernel_size=3, stride=1, padding=1)# [batch,4*class_num,3,3]
        self.location5 = nn.Conv2d(in_channels=256, out_channels=4 * 4, kernel_size=3, stride=1, padding=1)# [batch,4*4,3,3]

        self.Conv11_2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1),  # [batch,128,3,3]
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),  # [batch,256,1,1]
            nn.ReLU(inplace=True),
        )

        self.classifier6 = nn.Conv2d(in_channels=256, out_channels=4 * class_num, kernel_size=1) # [batch,4 * class_num,1,1]
        self.location6 = nn.Conv2d(in_channels=256, out_channels=4 * 4, kernel_size=1) # [batch,4*4,1,1]

        if init_weights:
            self._initialize()

    def forward(self,x):
        x=self.VGG1(x)
        #print('VGG1:',x)
        out1=self.l2norm(x)
        #print('l2norm:',out1)
        classifi_out1=self.classifier1(out1).view(x.shape[0],self.class_num,-1)
        location_out1=self.location1(out1).view(x.shape[0],4,-1)
        x=self.VGG2(x)
        #print('VGG2:',x)
        x=self.Conv6(x)
        #print('Conv6:',x)
        x=self.Conv7(x)
        #print('Conv7:',x)
        classifi_out2=self.classifier2(x).view(x.shape[0],self.class_num,-1)
        location_out2 = self.location2(x).view(x.shape[0],4,-1)
        x=self.Conv8_2(x)
        #print('Conv8_2:',x)
        classifi_out3 = self.classifier3(x).view(x.shape[0],self.class_num,-1)
        location_out3 = self.location3(x).view(x.shape[0],4,-1)
        x=self.Conv9_2(x)
        #print('Conv9_2:',x)
        classifi_out4 = self.classifier4(x).view(x.shape[0],self.class_num,-1)
        location_out4 = self.location4(x).view(x.shape[0],4,-1)
        x=self.Conv10_2(x)
        #print('Conv10_2:',x)
        classifi_out5 = self.classifier5(x).view(x.shape[0],self.class_num,-1)
        location_out5 = self.location5(x).view(x.shape[0],4,-1)
        x=self.Conv11_2(x)
        #print('Conv11_2:',x)
        classifi_out6 = self.classifier6(x).view(x.shape[0],self.class_num,-1)
        location_out6 = self.location6(x).view(x.shape[0],4,-1)

        classifi_out=[classifi_out1,classifi_out2,classifi_out3,classifi_out4,classifi_out5,classifi_out6]
        location_out=[location_out1,location_out2,location_out3,location_out4,location_out5,location_out6]

        classifi_out=torch.cat(classifi_out,dim=2).transpose(1,2).contiguous()
        location_out=torch.cat(location_out,dim=2).transpose(1,2).contiguous()
        # classifi_out:[batch_size,8732,class_num+1] 令第0位为背景
        # location_out:[batch_size,8732,4] 4为xywh的偏移量

        return classifi_out,location_out

    def _initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def loadVGG(self,VGG_path):
        our_checkpoint = self.state_dict()
        vgg_checkpoint = torch.load(VGG_path)
        model_name = []
        checkpoint_name = []

        for name, value in our_checkpoint.items():
            name_list = name.split('.')
            if name_list[0] == 'VGG1' or name_list[0] == 'VGG2':
                model_name.append(name)

        for name, value in vgg_checkpoint.items():
            name_list = name.split('.')
            if name_list[0] == 'features':
                checkpoint_name.append(name)
        for i in range(len(model_name)):
            our_checkpoint[model_name[i]] = vgg_checkpoint[checkpoint_name[i]]

        self.load_state_dict(our_checkpoint)

# 损失函数
class LossFunction(nn.Module):
    def __init__(self):
        super(LossFunction, self).__init__()

        # 位置的损失函数使用的是smoothL1
        self.location_loss=nn.SmoothL1Loss(reduction='none')
        # 分类的损失函数使用的是交叉熵
        self.classifi_loss=nn.CrossEntropyLoss(reduction='none')
                                      
    def forward(self,location_output,classifi_output,location_label,classifi_label):
        # 分别输入网络的两个输出，以及对应的gt标签
        # 先计算位置损失，需要计算两个东西：输出与标签是否相等以及g戴帽子
        # location_label:[batch_size,8732，4],location_output:[batch_size,8732，4]

        # 计算输出与标签是否相等,classifi_label维度为[batch_size,8732]
        mask=classifi_label>0
        # 计算每个batch的正样本个数,true为1，false为0，维度为[batch_size]
        pos_num=mask.sum(dim=1)
        #print('pos_num:',pos_num)
        # 定位损失，将四个值的损失加在一起，由[batch_size,8732，4]->[batch_size,8732]
        loc_loss=self.location_loss(location_output,location_label).sum(dim=-1)
        # 只将正样本的定位损失加起来，随后将8732个定位损失加起来，得到每个batch一个的损失
        loc_loss=(loc_loss*mask.float()).sum(dim=-1)
        #print('loc_loss:',loc_loss)
        

        # 置信度的损失，classifi_label为[batch_size,8732],classifi_output为[batch_size,8732，1+class_num]
        # con为[batch_size,8732],这里是把背景当成一个分类进行计算
        con=self.classifi_loss(classifi_output.transpose(1,2),classifi_label.long())

        # 获取负样本
        con_neg=con.clone()
        # 将所有正样本剔除
        con_neg[mask]=0
        # 选出loss大的负样本，首先是按照loss进行排序,con_rank中每个值都是这个loss在所有loss中的排序
        con_rank=con_neg.sort(dim=1,descending=True)[1].sort(dim=1)[1]
        # 负样本的个数,使负样本数量为正样本数量的3倍，但最高不能超过8732减去正样本数，最后再加一个维度，变成[batch_size,1]
        neg_num=torch.clamp(3 * pos_num, max=mask.size(1)).unsqueeze(-1)
        neg_num[neg_num==0]=3
        # 维度为[batch_size, 8732]
        neg_mask=con_rank<neg_num
        #print('pos_con:',con[mask])
        #print('neg_con:',con[neg_mask])
        # con_loss为[batch_size]，sum前是[batch_size,8732]
        con_loss=(con*(mask.float()+neg_mask.float())).sum(dim=1)
        # 损失相加
        alpha=1.0
        total_loss=loc_loss+alpha*con_loss
        pos_num     = torch.where(pos_num != 0, pos_num, torch.ones_like(pos_num))
        total_loss=(total_loss/pos_num).mean(dim=0)
        #print('total_loss:',total_loss)
        return total_loss


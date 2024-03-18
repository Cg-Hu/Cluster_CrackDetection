from datetime import datetime
from functools import partial
import time
import torch
import numpy as np
import os
from colorama import Fore
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from net.siamese import Net
# from net.vgg16 import Vgg16_net
from torch import nn

from torch.utils.tensorboard import SummaryWriter

from utils.tools import squash_packed, WarmupCosineSchedule
from utils.Dataloader import build_dataloader


# from tools import squash_packed, WarmupCosineSchedule
# from Dataloader import build_dataloader

def get_evalutaion_index(TP_FP, gross, TP, TP_TN, TP_FN, prefix=""):
    """
     计算acc, recall, precision, f1score

     @param correct - 预测正确的样本数量.
     @param total - 样本数量
     @param Positive_correct - 正样本中预测对的数量
     @param Positive_label_num - 样本中正样本的数量
     @param Positive_pred_num - 预测为正样本的数量

     @return acc, recall, precision, f1score
    """
    epsilon = 1e-8
    acc = TP_TN / (gross + epsilon)
    recall = TP.item() / (TP_FP.item() + epsilon)
    precision = TP.item() / (TP_FN.item() + epsilon)
    f1score = (1 + 1) * (recall * precision) / (recall + precision + epsilon)

    score = dict(
        acc=acc,
        recall=recall,
        precision=precision,
        f1score=f1score,
    )

    if prefix:
        keys = [key for key in score.keys()]
        for key in keys:
            score[prefix + '_' + key] = score.pop(key)
    return score


class Trainer():

    def __init__(self, device, 
                 weight_decay: float, **kwargs) -> None:
        """用于训练模型

        Args:
            model (_type_): 要训练的模型
            device (_type_): 运行的设备
            criterion (_type_): loss函数
            optimizer (_type_): 优化器
            lr_scheduler (_type_): 用于学习率衰减
            val_loader (_type_): 用于加载验证集
            tqdm (_type_): 用于加载进度条
            tb_writer (_type_): 用于在tensorboard中记录各项指标信息
            log_dir (str): 日志文件目录
        """
        local_params = locals()
        local_params.update(kwargs)
        self.model = Net(**kwargs).to(device)
        # self.model = Vgg16_net().to(device)
        self.device = device
        print("正在加载模型参数")
        para = torch.load("./weights/resnet18/model_acc.pkl", map_location=device)
        del para['fc1.weight']
        del para['fc1.bias']
        del para['fc2.weight']
        del para['fc2.bias']
        # print('删除了部分')
        # checkpoint = torch.load("./weights/resnet18/model_acc.pth_clustere2e.tar", map_location=device)
        # self.model.load_state_dict(checkpoint["state_dict"])
        # 删除后面这几部分，因为没有用到
        
        # for param_tensor in para:
        #打印 key value字典
            # print(param_tensor)
 


        # print('*' * 80)
        # print(self.model)
        
        self.model.load_state_dict(para)
        # print('加载完了参数')/
        # exit('测试结束')
        
        self.model.to(device)
        self.tqdm = partial(tqdm, ncols=95)
        # print("trainer的init也结束了")


    def run(self, loader) -> dict:
        """完成一个epoch的推理工作

        Args:
            epoch (int): 当前epoch ID
            loader (_type_): 用于数据加载
            mode (str): 用于确定当前是训练还是验证

        Returns:
            dict: 返回模型当前各项指标
        """

        device = self.device
        color = Fore.YELLOW
        tqdm_bar = self.tqdm(loader, desc=color + f'cluster epoch ')

        load_time_list = []
        outputs_list = []
        coordinate_list = []
        crack_list = [] # 判断是否是裂缝
        last_time = time.time()

        for patchs, img_index, coordinate, crack in tqdm_bar:
            # print(patchs.shape)
            now_time = time.time()
            patchs = patchs.to(device)
            outputs = self.model(patchs)
            # print(outputs.shape)
            outputs_list.append(outputs)
            coordinate_list.append(coordinate)
            crack_list.append(crack)
            # exit('测试结束')
            end_time = time.time()
            load_time_list.append(now_time - last_time)
            tqdm_bar.set_postfix({
                'Load':
                f'{sum(load_time_list) if len(load_time_list)<10 else sum(load_time_list[-10:]):.2f} s',
                'Cal': f'{end_time-now_time:.2f} s'
            })
            last_time = time.time()

            
            # gross += labels.numel()
            # correct += (preds == labels).sum()

        # feature_tensor = torch.stack(outputs_list) 这个玩意儿不行
        
        feature_tensor = torch.cat(outputs_list)
        coordinate_set = torch.cat(coordinate_list)
        crack_set = np.concatenate(crack_list)
        
        # print(crack_set)

        # print(coordinate_set)
        # print(coordinate_set.shape)
        # exit('测试结束')
        return feature_tensor, coordinate_set, crack_set
        # print(feature_tensor.shape)


        # 接下来的代码就要写 K-means 的算法       
        

    def cluster_epoch(self, loader) -> dict:
        """模型验证

        Args:
            epoch (int): 当前epoch ID

        Returns:
            dict: 当前模型在验证集上的各项指标表现
        """
        self.model.eval()
        with torch.no_grad():
            feature_tensor, coordinate_set, crack_set  = self.run(loader)
        return feature_tensor,coordinate_set, crack_set

    


    def train(self, loader, epoch):
        feature_tensor,coordinate_set, crack_set = self.cluster_epoch(loader)
        return feature_tensor,coordinate_set, crack_set
        

    def Test4lr(self, optim: str, batch_size: int, weight_decay: float,
                test4lr: int, **kargs):
        """用于找到最优学习率
        """
        init_lr = 1e-7
        end_lr = 0.01
        self.optimizer = getattr(torch.optim, optim)(self.model.parameters(),
                                                     lr=init_lr,
                                                     weight_decay=weight_decay)
        loader = self.val_loader if test4lr == 1 else self.train_loader
        if test4lr == 2:
            self.enhance_images(0, test4lr=test4lr)
        step_count = min(len(loader), 40)
        loader.dataset.sequences_list = loader.dataset.sequences_list[:
                                                                      step_count
                                                                      *
                                                                      batch_size]
        lr_scheduler = StepLR(self.optimizer,
                              step_size=1,
                              gamma=(end_lr / init_lr)**(1 / step_count))

        self.model.train()
        device = self.device
        color = Fore.YELLOW
        tqdm_bar = self.tqdm(loader, desc=color + 'Test For lr')

        for index, (seqs, labels) in enumerate(tqdm_bar):
            seqs = seqs.to(device)
            labels = labels.to(device)

            outputs = self.model(seqs)

            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            preds = torch.where(outputs > self.score_thr, 1., 0.)
            gross = labels.numel()
            TP_FP = labels.sum()
            TP_FN = preds.sum()
            TP = torch.sum(torch.logical_and(labels == 1, preds == 1))
            TP_TN = (preds == labels).sum().item()

            result_score = get_evalutaion_index(TP_FP,
                                                gross,
                                                TP,
                                                TP_TN,
                                                TP_FN,
                                                prefix='direction')
            result_score['loss'] = loss.item()

            self.record_tensorboard(index, result_score)
            self.tb_writer.add_scalar('Test4lr/lr',
                                      lr_scheduler.get_last_lr()[0], index)
            lr_scheduler.step()

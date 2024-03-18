from datetime import datetime
import os
from typing import Tuple
import yaml
from config import get_args
import ssl
import random
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from config import get_args
import torch
from utils.train_utils import Trainer
from utils.Dataloader import build_dataloader
from utils.Dataset import Dataset
from colorama import init, Fore
import socket
import numpy as np
import pandas as pd
import torch
from kmeans_pytorch import kmeans, kmeans_predict
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split


ssl._create_default_https_context = ssl._create_unverified_context


def Kmeans_Algo(feature_tensor:torch.tensor, **kwargs):
    patch_num = len(feature_tensor)
    dims = feature_tensor.shape[1]
    num_clusters = 2
    device = kwargs['device']
    cluster_ids_x, cluster_centers = kmeans(
    X=feature_tensor, num_clusters=num_clusters, distance='cosine', device=device
)
    # print(cluster_ids_x.sum()) # 399
    # print(cluster_ids_x.numel() - cluster_ids_x.sum()) # 637
    # 数据集各类别聚类中心
    # print(cluster_centers)
    return cluster_ids_x, cluster_centers


def get_box(patch_size: int, x: int, y: int) -> Tuple[int, int, int, int]:
    
    xmin = x * patch_size
    ymin = y * patch_size
    xmax = (x + 1) * patch_size
    ymax = (y + 1) * patch_size
    
    return xmin, ymin, xmax, ymax

def draw_rectangles(image: np.ndarray,
                        rectangles: list, patch_size: int) -> np.ndarray:
        """
        在原图上标注出对应的切片list

        Parameters
        ----------
        image : np.ndarray
            原图
        rectangles : list
            切片list的box坐标

        Returns
        -------
        np.ndarray
            结果图像
        """
        # 复制图像，以免修改原始图像
        output_image = image.copy()
        patch_real_idx = []

        
        # 遍历矩形列表
        for node in rectangles:
            xmin, ymin, xmax, ymax = get_box(patch_size, node[0],
                                             node[1])
            patch_real_idx.append([xmin, ymin, xmax, ymax])    

            # 在图像上绘制矩形
            cv2.rectangle(output_image, (xmin, ymin), (xmax, ymax),
                          (0, 255, 0), 2)  # 可根据需要修改颜色和线宽

        # 返回绘制了矩形的图像
        return output_image, patch_real_idx


if __name__ == '__main__':
    opt = vars(get_args())
    # log_dir = f'./logs/{datetime.now().strftime("%b%d_%H_%M_%S")}-{opt["desc"]}-{socket.gethostname()}'
    # os.makedirs(log_dir)
    # opt['log_dir'] = log_dir
    dataset_dir_path = opt['dataset_dir_path'] # 单个字符串
    seed = opt['seed']
    gpus = opt['gpus']
    batch_size = opt['batch_size']
    num_workers = opt['num_workers']
    patch_size = opt['patch_size']

    init(autoreset=True)
    
    # 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
    opt.update(device=torch.device(f'cuda:{gpus[1]}' if torch.cuda.is_available() else 'cpu'))
    # print(opt['device'])
    trainer = Trainer(**opt)
    # with open(os.path.join(log_dir, 'config.yml'), 'w+') as file:
    #     # opt.pop('device')
    #     yaml.dump(opt, file)
    
    cluster_class = {}
    acc_list = []
    # 得到所有图像的文件名
    for dir_path in dataset_dir_path:
        if os.path.exists(os.path.join(dir_path, 'all.txt')):
            train_filenames = [i.replace('\n', '') for i in open(os.path.join(dir_path, 'test.txt')).readlines()]
            val_filenames = [i.replace('\n', '') for i in open(os.path.join(dir_path, 'val.txt')).readlines()]
        else:
            filenames = sorted(os.listdir(os.path.join(dir_path, 'image')))
            train_filenames, val_filenames = train_test_split(filenames,
                                                              test_size=0.3,
                                                              random_state=42)
    image_name_list = train_filenames
    print("图片的张数: ",len(image_name_list))
    # image_paths += [
    #             os.path.join(dataset_dir_path, 'image', i)
    #             for i in dataset_filenames[self.mode]
    #         ]
    # label_paths += [
    #     os.path.join(dataset_dir_path, 'label', i)
    #     for i in dataset_filenames[self.mode]
    # ]

    for idx, image_name in enumerate(image_name_list):
        loader = build_dataloader(
                    image_name=image_name,
                    index = idx,
                    **opt
                    # shuffle=True, 在Dataset手动shuffle
                )
        patch_num = loader.dataset.patch_num
        print(f"==>> 第{idx}张图像的patch个数为: {patch_num}")
        feature_tensor, coordinate_set, crack_list = trainer.train(loader, idx)
        cluster_idxs, cluster_centers = Kmeans_Algo(feature_tensor, **opt)
        coordinate_set = coordinate_set.numpy()
        # print(coordinate_set)
        # print(type(coordinate_set))

        cluster_list = []
        patch_real_idx = []

        cluster_1 = cluster_idxs.sum()
        cluster_0 = cluster_idxs.numel() - cluster_1
        # cluster_class[image_name] = [cluster_0, cluster_1]
        # exit('测试结束')
        # 得到聚类类别哪个是裂缝
        crack_symbol = 0
        TP_FP = 0
        if cluster_0 < cluster_1:
            crack_symbol = 0
            TP_FP = cluster_0
            for idx, cluster in enumerate(cluster_idxs):
                if cluster == 0:
                    cluster_list.append(coordinate_set[idx])
        elif cluster_0 > cluster_1:
            crack_symbol = 1
            TP_FP = cluster_1
            for idx, cluster in enumerate(cluster_idxs):
                if cluster == 1:
                    cluster_list.append(coordinate_set[idx])
            
        
        
        # 统计预测的正确数字
        TP = 0
        TN = 0
        TP_TN = 0
        TP_FN = crack_list.sum()
        print(f'聚类情况为:({cluster_0}, {cluster_1})')
       
        for idx, cluster in enumerate(cluster_idxs):
            if cluster == crack_symbol and crack_list[idx] == 1:
                TP += 1
            elif cluster != crack_symbol and crack_list[idx] == 0:
                TN += 1
        TP_TN = TP + TN
        print(f"TP:{TP}")
        print(f"TN:{TN}")
        print(f'TP_FP:{TP_FP}')
        print(f"TP_FN:{TP_FN}")
        epsilon = 1e-8
        acc = float(TP_TN) / (float(patch_num) + epsilon)
        precision = TP / (TP_FP + epsilon)
        recall = TP / (TP_FN + epsilon)
        f1_score = (1 + 1) * (recall * precision) / (recall + precision + epsilon)
        # acc_list.append(acc)
        print(f"{image_name}:此张图片的acc: {acc}")
        print(f"{image_name}:此张图片的precsion: {precision}")
        print(f"{image_name}:此张图片的recall: {recall}")
        print(f"{image_name}:此张图片的f1_score: {f1_score}")
        file_path = 'result/test_patch_size'
        if not os.path.exists(file_path):
            os.mkdir(file_path)

        output_image, patch_real_idx = draw_rectangles(loader.dataset.image, cluster_list, patch_size)
        cv2.imwrite(f'{file_path}/{image_name[:-3]}_cluster.jpg', output_image)
        

        with open(f'{file_path}/acc.txt', 'a') as f: 
            f.writelines(f'{image_name}, acc:{acc}, precsion:{precision}, recall:{recall}, f1_score:{f1_score}, 聚类情况为:({cluster_0}, {cluster_1})\n')
            f.writelines('\n')
        f.close()
        with open(f'{file_path}/coordinate.txt', 'a') as fp:
            fp.writelines(f'{image_name}, acc:{acc}, precsion:{precision}, recall:{recall}, f1_score:{f1_score}, 聚类情况为:({cluster_0}, {cluster_1})\n')
            for pixIdx in patch_real_idx:
                fp.writelines(f'xmin:{str(pixIdx[0])} ')
                fp.writelines(f'ymin:{str(pixIdx[1])} ')
                fp.writelines(f'xmax:{str(pixIdx[2])} ')
                fp.writelines(f'ymax:{str(pixIdx[3])}')
                fp.writelines('\n')
        fp.close()
    
    f.close()
    fp.close()
            # print(key, cluster_class[key])

        


        
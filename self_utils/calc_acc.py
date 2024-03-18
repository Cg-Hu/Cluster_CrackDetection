import torch
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
import socket
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
from config import get_args

def readTxt1(filepath :str, filename :str):
    acc_list = []
    recall_list = []
    precision_list = []
    f1_score_list = []
    with open(os.path.join(filepath, filename), 'r') as f:
        content = f.readlines()

        for idx, row in enumerate(content):
            # print(row)
            if idx % 2 == 0:
                acc_content = row.split(' ')
                acc = float(acc_content[1].split(':')[1][:-1])
                precision = float(acc_content[2].split(':')[1][:-1])
                recall = float(acc_content[3].split(':')[1][:-1])
                f1_score = float(acc_content[4].split(':')[1][:-1])
                acc_list.append(acc)
                precision_list.append(precision)
                recall_list.append(recall)
                f1_score_list.append(f1_score)

    len_image = len(acc_list)
    avg_acc = sum(acc_list) / len_image
    avg_recall =  sum(recall_list) / len_image
    avg_precision = sum(precision_list) / len_image
    avg_f1_score = sum(f1_score_list) / len_image
    print(f'总计图片{len_image}张, avg_acc:{avg_acc}, avg_precision:{avg_precision}, avg_recall:{avg_recall}, avg_f1_score:{avg_f1_score}') # 82, 0.9097
    return acc_list, precision_list, recall_list, f1_score_list

def readTxt2(filepath :str, filename :str):
    acc_list = []
    with open(os.path.join(filepath, filename), 'r') as f:
        content = f.readlines()
        count_acc = 0
        count_acc_num =  0
        for  row in content:
            # print(row)
            acc_content = row.split(' ')
            acc = acc_content[1].split(':')[1][:-1]
            acc = float(acc)
            acc_list.append([acc_content[0][:-1], acc])
            count_acc += acc
            count_acc_num += 1
    
    avg_acc = float(count_acc) / float(count_acc_num)
    print(f'总计图片{count_acc_num}张, 最终的精度为: {avg_acc}') # 82, 0.9097
    print(f'列表中精确度的值有:{len(acc_list)}') # 82
    return acc_list

def drawFig(num_list :list, figname :str):
    

    for idx, acc in enumerate(num_list):
        # print(f'{idx}, {acc}')
        tb_writer.add_scalar(figname, acc, idx)
    


if __name__ == '__main__':
    opt = vars(get_args())
    filepath = '../result/se_s_k_means_mutil_enhanced3'
    filename = 'acc.txt'
    log_dir = f'../logs_tensorboard/{datetime.now().strftime("%b%d_%H:%M:%S")}_se_s_k_means_mutil_enhanced3'
    # os.mkdir(log_dir)
    acc_list, precision_list, recall_list, f1_score_list = readTxt1(filepath, filename)
    tb_writer = SummaryWriter(
            log_dir=os.path.join(log_dir, 'runs'))

    acc_list = sorted(acc_list)
    recall_list = sorted(recall_list)
    precision_list = sorted(precision_list)
    f1_score_list = sorted(f1_score_list)

    print(f'acc_min:{acc_list[0]}, acc_max:{acc_list[len(acc_list)-1]}')
    print(f'precision_min:{precision_list[0]}, precision_max:{precision_list[len(precision_list)-1]}')
    print(f'recall_min:{recall_list[0]}, recall_max:{recall_list[len(recall_list)-1]}')
    print(f'f1_score_min:{f1_score_list[0]}, f1_score_max:{f1_score_list[len(f1_score_list)-1]}')
    
    drawFig(acc_list, 'kmeans/acc')
    drawFig(precision_list, 'kmeans/precision')
    drawFig(recall_list, 'kmeans/recall')
    drawFig(f1_score_list, 'kmeans/f1_score')

    tb_writer.close()



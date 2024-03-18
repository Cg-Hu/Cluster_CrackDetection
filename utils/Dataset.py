from multiprocessing import Manager, Process
from PIL import Image
from colorama import Fore
import albumentations as A
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import cv2
from torch.utils import data
import torch
import numpy as np
import os
from functools import partial
from tqdm import tqdm
import sys
from queue import Queue
from collections import deque
from utils.tools import imread, get_box, get_patch_num, get_patch, expand_image, judge_crack
import random

tqdm = partial(tqdm, ncols=100, file=sys.stdout)

Image.MAX_IMAGE_PIXELS = 150000000

class Dataset(data.Dataset):

    def __init__(self,
                 dataset_dir_path: str,
                 image_name: list[str],
                 index: int,
                 enhanced: int = 3, 
                 patch_size: int = 64,
                 **args) -> None:
        """
        数据集

        Parameters
        ----------
        dataset_dir_paths : List[str]
            数据集的根路径
        patch_size : int, optional
            切片大小, by default 64

        """
        self.dataset_dir_path = dataset_dir_path
        self.patch_size = patch_size
        self.patch_num = 0
        self.pair_list = []
        self.image_name = image_name
        self.index = index
        self.enhanced = enhanced
        self.image = None

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5990, 0.6012, 0.5874),
                                 std=(0.0383, 0.0380, 0.0374)),
        ])

        # 使用不同的变量，尝试解决死锁问题
        self._enhanced_images = []
        self._enhanced_labels = []
        self._pair_list = []

        self.spatial_transform = A.Compose([
            # A.RandomResizedCrop(patch_size * 20, patch_size * 20, scale=(0.5, 1)),
            A.HorizontalFlip(),
            A.ShiftScaleRotate(shift_limit=0.0625,
                               scale_limit=0.2, rotate_limit=45, p=0.2),
            A.OneOf([
                    A.OpticalDistortion(p=0.3),
                    A.GridDistortion(p=.1),
                    # A.PiecewiseAffine(p=0.3),
                    ], p=0.2)
            ],
                additional_targets={
                'label': 'image'
            }
        )
        self.pixel_transform = A.Compose([
            A.GaussNoise(p=0.2),
            A.OneOf([
                    A.MotionBlur(p=.2),
                    A.MedianBlur(blur_limit=3, p=0.1),
                    A.Blur(blur_limit=3, p=0.1),
                    ], p=0.2),
            A.OneOf([
                    A.CLAHE(clip_limit=2),
                    A.Sharpen(),
                    A.Emboss(),
                    A.RandomBrightnessContrast(),
                    ], p=0.3),
            A.HueSaturationValue(p=0.3),
        ])

        self.image_init()


    def image_init(self):
        """初始化图像
        """
        image_name = os.path.join(self.dataset_dir_path[0], 'image', self.image_name)
        label_name = os.path.join(self.dataset_dir_path[0], 'label', self.image_name)

        # 图片都进行了expand扩大
        image, *_ = imread(image_name, -1, self.patch_size, expand=1)
        lable, *_ = imread(label_name, 0, self.patch_size, expand=1)


        self.image = image
        self.label = lable

        if not self.enhanced:   
            
            self.update_patch_pair_from_label(self.index, self.label)
            random.shuffle(self.pair_list)
            # self.pair_list = self._pair_list
        else:
            # print("正在进行图像增强:", self.enhanced)
            self.update_enhanced_images(self.index, self.enhanced)



    def draw_rectangles(self, image: np.ndarray,
                        rectangles: list) -> np.ndarray:
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

        # 遍历矩形列表
        for node in rectangles:
            xmin, ymin, xmax, ymax = get_box(self.patch_size, node['x'],
                                             node['y'])

            # 在图像上绘制矩形
            cv2.rectangle(output_image, (xmin, ymin), (xmax, ymax),
                          (0, 255, 0), 2)  # 可根据需要修改颜色和线宽

        # 返回绘制了矩形的图像
        return output_image

    def enhance_image(self, image: np.ndarray, label: np.ndarray, enhanced: int) -> tuple[np.ndarray, np.ndarray]:
        """图像增强

        Args:
            image (np.ndarray): 原图像
            label (np.ndarray): 标注图像

        Returns:
            tuple[np.ndarray, np.ndarray]: 增强后的图像
        """
        # 由PIL读入图像默认为RGB
        if enhanced == 1 or enhanced == 3:
            augmented_result = self.spatial_transform(image=image, label=label)
            image = augmented_result['image']
            label = augmented_result['label']
        if enhanced == 2 or enhanced == 3:
            image = self.pixel_transform(image=image)['image']
        return image, label
    
    def access_pre_setting(self, epoch: int):
        """
        开始当前epoch的预设置，主要与数据增强相关变量有关

        Parameters
        ----------
        epoch : int
            当前epoch编号
        """ 
        self.enhanced_images = list(self._enhanced_images).copy()
        self.enhanced_labels = list(self._enhanced_labels).copy()
        self.pair_list = list(self._pair_list).copy()
    
    def update_patch_pair_from_label(self, index: int, label_image: np.ndarray):
        """
        根据label生成对应的图像pair对

        Parameters
        ----------
        index : int
            当前label图像对应的index
        label_image : np.ndarray
            label图像
        """
        patch_size = self.patch_size
        h, w = get_patch_num(label_image, patch_size)

        self.patch_num = h * w

        get_pair = lambda index, coordinate, crack: {
                'pic_id': index,
                'patch_coordinate': coordinate,
                'crack': crack
            } 
        
        # 得到像素块的坐标
        crack = 0
        for y in range(h):
            for x in range(w):
                if judge_crack(label_image, x, y, patch_size):
                    crack = 1
                else:
                    crack = 0
                self.pair_list.append(get_pair(index, [x,y], crack))
        

    def update_enhanced_images(self, index: int, enhanced: int):
        """更新数据增强的图像

        Args:
            epoch (int): 数据增广对应的图像id
        """
        import random
        random.seed(index)
        # 因为是单张单张进行的，索性就搞单线程试试看
        
        enhanced_image, enhanced_label = self.enhance_image(
            self.image, self.label, enhanced)
        # cv2.imwrite('test.png', enhanced_image[:, :, ::-1])
        enhanced_image, *_= expand_image(enhanced_image, self.patch_size)
        enhanced_label, *_ = expand_image(enhanced_label, self.patch_size)
        self.image = enhanced_image
        self.label = enhanced_label
        self.update_patch_pair_from_label(index, self.label)
        random.shuffle(self.pair_list)
    
    def start_enhanced_thread(self, epoch: int):
        """
        启动epoch对应的数据增强进程

        Parameters
        ----------
        epoch : int
            epoch的编号
        """        
        print(Fore.LIGHTBLACK_EX + f'Start Loading Enhanced Images of Epoch-{epoch}...')
        self._enhanced_images = Manager().list()
        self._enhanced_labels = Manager().list()
        self._pair_list = Manager().list()
        self.enhanced_thread = Process(
            target=self.update_enhanced_images, args=(epoch, self.enhanced))
        # 设置为守护进程，当父进程结束时，该进程也会自动结束
        self.enhanced_thread.daemon = True
        self.enhanced_thread.start()
    
    def get_enhanced_process(self, epoch: int):
        """获取数据增广当前进度

        Args:
            epoch (int): 数据增广对应epoch id

        Returns:
            str: 表示数据增广进度的字符串.
        """
        return f"{len(self._enhanced_images)} / {len(self.image_paths)}"
    
    def __getitem__(self, index: int):
        
        pair = self.pair_list[index] # 这里的index也没什么实际意义
        # 得到patch的图像
        patch = get_patch(self.image, *pair['patch_coordinate'], self.patch_size)
        return self.transform(patch), torch.tensor(pair['pic_id'], dtype=torch.int), torch.tensor(pair['patch_coordinate'], dtype=torch.int), pair['crack'] # 没有标签

    def __len__(self):
        return len(self.pair_list)



if __name__ == '__main__':#

    dataset = Dataset (dataset_dir_paths='../dataset', image_name='creak_s1.JPG', index=1)
    print(len(dataset))
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, num_workers=0)
    count = 0
    print(f"==>> 第一张图像的patch个数为: {dataset.patch_num}")
    for patch, img_index, coordinate in loader:
        # print(f"==>> label: {patch}")
        print(f"==>> img_index: {img_index}")
        print(f"==>> coordinate: {coordinate}")
        # print(f"==>> patch.shape: {patch.shape}")
        # exit('')
        # count+=1
        # if(count >= dataset.patch_num[0]): break
        
        
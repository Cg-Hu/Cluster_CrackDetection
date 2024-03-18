import ssl
from torch import nn
import torch.nn.functional as F
import torch
import torchvision.models as models
ssl._create_default_https_context = ssl._create_unverified_context

def get_feature_channel(model):
    inputs = torch.rand(1, 3, 64, 64)
    outputs = model(inputs).flatten(1)
    return outputs.shape[1]

import torch
from torch import nn
import torch.nn.functional as F
 
# 224 * 224 * 3
class Vgg16_net(nn.Module):
    def __init__(self):
        super(Vgg16_net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),  # 224 * 224 * 64
            nn.BatchNorm2d(32), # Batch Normalization强行将数据拉回到均值为0，方差为1的正太分布上，一方面使得数据分布一致，另一方面避免梯度消失。
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1), # 224 * 224 * 64
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
 
            nn.MaxPool2d(kernel_size=2, stride=2)  # 112 * 112 * 64
        )
 
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1), # 112 * 112 * 128
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
 
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), # 112 * 112 * 128
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
 
            nn.MaxPool2d(2, 2)  # 56 * 56 * 128
        )
 
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),  # 56 * 56 * 256
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
 
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),  # 56 * 56 * 256
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2)  # 28 * 28 * 256
        )
 
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1), # 28 * 28 * 512
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
 
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1), # 28 * 28 * 512
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
 
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1), # 28 * 28 * 512
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2)  # 14 * 14 * 512
        )
 
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1), # 14 * 14 * 512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
 
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1), # 14 * 14 * 512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
 
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1), # 14 * 14 * 512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
 
            nn.MaxPool2d(2, 2)  # 7 * 7 * 512
        )

        self.layer6 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1), # 14 * 14 * 512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1), # 14 * 14 * 512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1), # 14 * 14 * 512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            nn.MaxPool2d(2, 2)  # 1 * 1 * 512
        )
 
        self.conv = nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.layer5,
            self.layer6
        )



        # 这是直接展平成为512
        # self.fc = nn.Sequential(
        #     nn.Linear(7*7*512, 512),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.5),
        #     nn.Linear(512, 256),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.5),
        # )
 
 
    def forward(self, x):
        x = self.conv(x).flatten(1)
        # _A = self.get_embedding(A)
        # _B = self.get_embedding(B)
        
        # x = torch.cosine_similarity(_A, _B, dim=1).unsqueeze(1)
        return x
    
    def get_embedding(self, x):
        # x = self.conv(x)
        # print(x.shape)
        # exit()
        return 



if __name__ == '__main__':

    model = Vgg16_net()
    A = torch.rand((2, 3, 64, 64))
    # B = torch.rand((2, 3, 64, 64))
    out = model(A)
    print(out)
    print(model)
    print(f"==>> out.shape: {out.shape}")
    
    # torch.save(model.state_dict(), "../logs_tensorboard/Dec02_10:52:34_resnet18_getW_2-VIPA207/weights/exc.pkl")
    # for k, v in model.state_dict().items():
    #     print(k)
    # para = torch.load('../logs_tensorboard/ImagenetPara/weights/vgg16-397923af.pth')
    # print('*' * 50)
    # for k, v in para.items():
    #     print(k)
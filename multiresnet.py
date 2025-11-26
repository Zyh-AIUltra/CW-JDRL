import os
import random
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, \
    confusion_matrix
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.models as models
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from torchvision.models.video import r3d_18
from torch.autograd import Function

class MultiResNetClassifier(nn.Module):
    def __init__(self):
        super(MultiResNetClassifier, self).__init__()

        # 加载五个预训练的 ResNet18 模型
        self.resnet1 = models.resnet18(pretrained=True)
        self.resnet2 = models.resnet18(pretrained=True)
        self.resnet3 = models.resnet18(pretrained=True)
        self.resnet4 = models.resnet18(pretrained=True)
        self.resnet5 = models.resnet18(pretrained=True)
        # self.resnet6 = models.resnet18(pretrained=True)   #共同通道用于解耦

        # 去掉 ResNet 的全连接层，使输出特征维度为 512
        self.resnet1.fc = nn.Identity()
        self.resnet2.fc = nn.Identity()
        self.resnet3.fc = nn.Identity()
        self.resnet4.fc = nn.Identity()
        self.resnet5.fc = nn.Identity()

        # 分类器
        self.fc1 = nn.Linear(5*512, 2)  # 5个 ResNet 的 512 特征拼接

    def forward(self, x):
        # x 形状: (batch_size, 5, 3, 224, 224)

        # 分别送入5个 ResNet18
        x1 = self.resnet1(x[:, 0, :, :, :])  # (batch_size, 512)
        x2 = self.resnet2(x[:, 1, :, :, :])  # (batch_size, 512)
        x3 = self.resnet3(x[:, 2, :, :, :])  # (batch_size, 512)
        x4 = self.resnet4(x[:, 3, :, :, :])  # (batch_size, 512)
        x5 = self.resnet5(x[:, 4, :, :, :])  # (batch_size, 512)


        x = torch.cat([x1,x2, x3, x4, x5], dim=1)
        x = self.fc1(x)
        return  x
class MultiResNetClassifier_LateFusion(nn.Module):
    def __init__(self):
        super(MultiResNetClassifier_LateFusion, self).__init__()

        # 五个独立 ResNet18（决策级融合要求必须独立）
        self.resnet1 = models.resnet18(pretrained=True)
        self.resnet2 = models.resnet18(pretrained=True)
        self.resnet3 = models.resnet18(pretrained=True)
        self.resnet4 = models.resnet18(pretrained=True)
        self.resnet5 = models.resnet18(pretrained=True)

        # 修改 ResNet 的 fc，使其输出为 logits (2 类)
        self.resnet1.fc = nn.Linear(512, 2)
        self.resnet2.fc = nn.Linear(512, 2)
        self.resnet3.fc = nn.Linear(512, 2)
        self.resnet4.fc = nn.Linear(512, 2)
        self.resnet5.fc = nn.Linear(512, 2)

    def forward(self, x):
        # x shape: (B, 5, 3, 224, 224)

        logit1 = self.resnet1(x[:, 0])  # (B,2)
        logit2 = self.resnet2(x[:, 1])  # (B,2)
        logit3 = self.resnet3(x[:, 2])  # (B,2)
        logit4 = self.resnet4(x[:, 3])  # (B,2)
        logit5 = self.resnet5(x[:, 4])  # (B,2)

        # === Late Fusion: 决策级融合（logits 平均）===
        logits = (logit1 + logit2 + logit3 + logit4 + logit5) / 5.0

        return logits   # (B,2)




class CW_JDRL(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        # 主图 ResNet18（独立）
        base1 = models.resnet18(pretrained=True)
        self.resnet_main = nn.Sequential(
            base1.conv1, base1.bn1, base1.relu, base1.maxpool,
            base1.layer1,
            base1.layer2,
            base1.layer3,
            base1.layer4,
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # 后四张图共用的 ResNet18
        base2 = models.resnet18(pretrained=True)
        self.resnet_views = nn.Sequential(
            base2.conv1, base2.bn1, base2.relu, base2.maxpool,
            base2.layer1,
            base2.layer2,
            base2.layer3,
            base2.layer4
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.embed1= nn.Linear(4*256,512)
        self.embed2 = nn.Linear(4 * 256, 512)
        # 分类器（可选，用所有拼接特征）
        self.classifier = nn.Linear(512*5 , num_classes)
        self.classifier1 = nn.Linear(512, num_classes)
        self.classifier2 = nn.Linear(512, num_classes)
        self.classifier3 = nn.Linear(512, num_classes)
        self.classifier4 = nn.Linear(512, num_classes)
        self.feature_maps = []  # 存储 forward 输出
        self.gradients = []  # 存储 backward 梯度
        self.register_hooks()


        self.fusion_embed = nn.Sequential(
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Dropout(0.1)
        )
    def register_hooks(self):
        def hook_fn(module, input, output):
            if not output.requires_grad:
                output = output.clone().detach().requires_grad_()
            output.retain_grad()
            self.feature_maps.append(output)
        self.feature_maps = []
        self.gradients = []
        self.resnet_views[7].register_forward_hook(hook_fn)  # layer4
    def forward(self, x):
        """
        x: (B, 5, H, W)
        返回:
        - logits: 分类预测
        - shared_feats: List of 4 tensors (B, 256)
        - private_feats: List of 4 tensors (B, 256)
        - f_main: 主图特征 (B, 512)
        """

        # 主图
        x_main = x[:, 0, :, :, :]
        f_main = self.resnet_main(x_main).squeeze(-1).squeeze(-1)  # (B, 512)

        shared_feats = []
        private_feats = []

        # 视角图像处理
        x_views = x[:, 1:]  # (B, 4, H, W)
        for i in range(4):
            xi = x_views[:, i, :, :,:]  # (B, 3, H, W)
            feat_map = self.resnet_views(xi)  # (B, 512, 7, 7)

            # 通道划分
            f_shared_map = feat_map[:, :256, :, :]
            f_private_map = feat_map[:, 256:, :, :]

            # GAP
            f_shared = self.pool(f_shared_map).squeeze(-1).squeeze(-1)   # (B, 256)
            f_private = self.pool(f_private_map).squeeze(-1).squeeze(-1) # (B, 256)

            shared_feats.append(f_shared)
            private_feats.append(f_private)
        f1_c, f2_c, f3_c, f4_c = shared_feats
        f1s, f2s, f3s, f4s = private_feats
        f1 = torch.cat((f1_c,f1s), dim=1)
        f2 = torch.cat((f2_c,f2s), dim=1)
        f3 = torch.cat((f3_c,f3s), dim=1)
        f4 = torch.cat((f4_c,f4s), dim=1)
        sr =[f1, f2, f3, f4]

        f_all = torch.cat([f_main,f1,f2,f3,f4 ], dim=1)  # (B, 512 + 4*256 + 4*256)fc ,fs

        logits1 = self.classifier1(f1)
        logits2 = self.classifier2(f2)
        logits3 = self.classifier3(f3)
        logits4 = self.classifier4(f4)
        lg = [logits1, logits2, logits3, logits4]

        logits = self.classifier(f_all)

        return logits, f_main, shared_feats, private_feats,sr,lg

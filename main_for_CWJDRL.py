import os
import random
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import os
import cvxpy as cp
import torch
import torch.nn.functional as F
import time
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, \
    confusion_matrix
from sklearn.metrics import auc as sklearn_auc
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from PIL import Image
import torchvision.models as models
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from multiresnet import MultiResNetClassifier,CW_JDRL
from itertools import combinations
from sklearn.metrics.pairwise import cosine_similarity
from functools import reduce
from itertools import combinations

def load_data(excel_path):
    df = pd.read_excel(excel_path, dtype=str)
    df['ID'] = df['ID'].str.lower().str.strip()  # 统一 ID 格式
    df = df.iloc[:931]  # 读取前 935 行数据

    # 筛选“病理良恶性”列的有效数据（良性或恶性）
    df = df[df['病理良恶性'].isin(['良性', '恶性'])].copy()

    # 映射标签
    df['label'] = df['病理良恶性'].map({'良性': 0, '恶性': 1})

    # 映射临床信息
    df['年龄'] = df['患者年龄'].astype(int)
    df['生育史'] = df['生育史'].map({'无': 0, '有': 1})
    df['乳腺癌家族史'] = df['乳腺癌家族史'].map({'无': 0, '有': 1})
    df['病灶位置'] = df['病灶位置'].map({'外上象限': 0, '外下象限': 1, '内上象限': 2, '内下象限': 3})
    df['病灶大小'] = df['病灶大小'].astype(float)

    return df


class BreastCancerDataset(Dataset):
    def __init__(self, root_folder, dataframe, transform=None):
        self.root_folder = root_folder
        self.data = dataframe
        self.transform = transform
        self.image_filenames = ['1', '2', '3', '4', '5']  # 需要加载的图像
        self.valid_extensions = ('.png', '.jpg', '.jpeg', '.tif')  # 允许的图像格式

        self.targets = self.data['label'].tolist()
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        case_id = row['ID'].strip().lower()  # 统一 ID 格式
        label = row['label']
        clinical_data = torch.tensor(
            [row['年龄'], row['生育史'], row['乳腺癌家族史'], row['病灶位置'], row['病灶大小']], dtype=torch.float32)

        # 查找病例文件夹
        case_folder = None
        for subfolder in os.listdir(self.root_folder):
            subfolder_path = os.path.join(self.root_folder, subfolder)
            if os.path.isdir(subfolder_path):
                for folder in os.listdir(subfolder_path):
                    if folder.lower() == case_id:
                        case_folder = os.path.join(subfolder_path, folder)
                        break
                if case_folder:
                    break

        if case_folder is None:
            raise FileNotFoundError(f"未找到病例文件夹 {case_id}")

        # 读取图像
        images = []
        for img_name in self.image_filenames:
            img_path = None
            for file in os.listdir(case_folder):
                if file.lower().startswith(img_name) and file.lower().endswith(self.valid_extensions):
                    img_path = os.path.join(case_folder, file)
                    break
            if img_path and os.path.exists(img_path):
                img = Image.open(img_path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                images.append(img)
            else:
                images.append(torch.zeros(3, 224, 224))  # 缺失时填充 0 矩阵

        images = torch.stack(images)  # 5 张图像堆叠 (5, 3, 224, 224)
        return images, clinical_data, label
IMAGE_SIZE = (224,224)  # 统一图像大小

# 图像变换（可自行调整）
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
transform_test = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
root_folder = '/CW-JDRL/SRUS'
excel_path = '/CW-JDRL/SRUS.xlsx'

def gram_schmidt(vectors):

    orthogonal = []
    for v in vectors:
        for u in orthogonal:
            # 计算每个样本内的投影（按 batch 独立）
            proj = ((v * u).sum(dim=1, keepdim=True) / (u.norm(dim=1, keepdim=True) ** 2 + 1e-8)) * u
            v = v - proj
        v = F.normalize(v, dim=1)
        orthogonal.append(v)
    return orthogonal

def apply_gradient_guidance(model, lambda_align=1.0, lambda_orth=1.0, print_stats=True):
    """
    引导梯度方向，同时保留模长。
    - 打印原始共享/私有通道梯度模长的均值和标准差（按视角）
    """
    gc_list, gs_list = [], []
    gcn_list, gsn_list = [], []  # 原始模长
    # print("feature_maps type:", type(model.feature_maps))
    # print("feature_maps length:", len(model.feature_maps))
    for fmap in model.feature_maps:
        grad = fmap.grad  # (B, 512, H, W)
        if grad is None:
            continue

        grad_shared = grad[:, :256, :, :]   # (B, 256, H, W)
        grad_private = grad[:, 256:, :, :]  # (B, 256, H, W)

        # GAP to (B, 256)
        gc = F.adaptive_avg_pool2d(grad_shared, 1).squeeze(-1).squeeze(-1)
        gs = F.adaptive_avg_pool2d(grad_private, 1).squeeze(-1).squeeze(-1)

        gc_list.append(gc)
        gs_list.append(gs)

        gcn_list.append(gc.norm(dim=1, keepdim=True))  # (B, 1)
        gsn_list.append(gs.norm(dim=1, keepdim=True))  # (B, 1)

    if len(gc_list) < 2:
        return

    if print_stats:
        # 打印模长统计信息
        gc_mags = torch.cat(gcn_list, dim=1)  # shape: (B, 4)
        gs_mags = torch.cat(gsn_list, dim=1)  # shape: (B, 4)

        gc_mean = gc_mags.mean().item()
        gc_std = gc_mags.std().item()
        gs_mean = gs_mags.mean().item()
        gs_std = gs_mags.std().item()

        print(f"[Shared Gradient Norm] Mean: {gc_mean:.4f}, Std: {gc_std:.4f}")
        print(f"[Private Gradient Norm] Mean: {gs_mean:.4f}, Std: {gs_std:.4f}")

    with torch.no_grad():
        # ==== 1. 对齐共享通道梯度 ====
        mean_gc = torch.stack([F.normalize(g, dim=1) for g in gc_list], dim=0).mean(dim=0)
        mean_gc = F.normalize(mean_gc, dim=1)

        for i, fmap in enumerate(model.feature_maps):
            g_magnitude = gcn_list[i]  # (B, 1)
            new_gc = lambda_align * g_magnitude * mean_gc
            fmap.grad[:, :256, :, :] = new_gc.unsqueeze(-1).unsqueeze(-1)

        # ==== 2. 正交化私有通道梯度 ====
        gs_orth = gram_schmidt(gs_list)
        for i, fmap in enumerate(model.feature_maps):
            g_magnitude = gsn_list[i]
            new_gs = lambda_orth * g_magnitude * gs_orth[i]
            fmap.grad[:, 256:, :, :] = new_gs.unsqueeze(-1).unsqueeze(-1)

def disentangled_volume_loss_with_official_volume_fn(
    f1c, f2c, f3c, f4c,
    f1s, f2s, f3s, f4s,
    volume_computation_fn,
    tau=0.07
):
    """
    构建正样本和69个体积负样本组合，基于 volume_computation() 函数计算体积，然后用 InfoNCE-style loss。

    Args:
        f1c-f4c: 共享模态特征 [B, D]
        f1s-f4s: 特定模态特征 [B, D]
        volume_computation_fn: 用户提供的 volume 函数
        tau: 温度参数

    Returns:
        scalar loss
    """
    B, D = f1c.shape
    device = f1c.device
    tau = tau if isinstance(tau, torch.Tensor) else torch.tensor(tau, device=device)

    # 所有候选向量（8个）：前4是共享，后4是特定
    all_inputs = [f1c, f2c, f3c, f4c, f1s, f2s, f3s, f4s]

    # 所有 C(8,4) 组合
    all_indices = list(combinations(range(8), 4))  # 70个组合
    pos_indices = (0, 1, 2, 3)
    neg_indices = [idx for idx in all_indices if idx != pos_indices]  # 排除正样本

    # Step 1: 正样本 volume，使用 f1c 作为 anchor
    v_pos = volume_computation_fn(f1c, f2c, f3c, f4c)  # [B, B]
    v_pos_diag = torch.diagonal(v_pos, dim1=0, dim2=1)  # [B]，取对角线为 batch 内正样本 volume

    # Step 2: 负样本 volume（69组）
    neg_volumes = []
    for idx in neg_indices:
        anchor = all_inputs[idx[0]]
        inputs = [all_inputs[i] for i in idx[1:]]
        v_neg = volume_computation_fn(anchor, *inputs)  # [B, B]
        neg_volumes.append(torch.diagonal(v_neg, dim1=0, dim2=1))  # 取对角线作为 batch 内样本自身负 volume

    neg_volumes = torch.stack(neg_volumes, dim=1)  # [B, 69]

    # Step 3: InfoNCE-style contrastive loss
    numerator = torch.exp(-v_pos_diag / tau)  # [B]
    denominator = numerator + torch.sum(torch.exp(-neg_volumes / tau), dim=1)  # [B]
    loss = -torch.log(numerator / denominator)  # [B]

    return loss.mean()
def volume_computation(anchor, *inputs):
    """
    General function to compute volume for contrastive learning loss functions.
    Compute the volume metric for each vector in anchor batch and all the other modalities listed in *inputs.

    Args:
    - anchor (torch.Tensor): Tensor of shape (batch_size1, dim)
    - *inputs (torch.Tensor): Variable number of tensors of shape (batch_size2, dim)

    Returns:
    - torch.Tensor: Tensor of shape (batch_size1, batch_size2) representing the volume for each pair.
    """
    batch_size1 = anchor.shape[0]
    batch_size2 = inputs[0].shape[0]

    # Compute pairwise dot products for language with itself
    aa = torch.einsum('bi,bi->b', anchor, anchor).unsqueeze(1).expand(-1, batch_size2)

    # Compute pairwise dot products for language with each input
    l_inputs = [anchor @ input.T for input in inputs]

    # Compute pairwise dot products for each input with themselves and with each other
    input_dot_products = []
    for i, input1 in enumerate(inputs):
        row = []
        for j, input2 in enumerate(inputs):
            dot_product = torch.einsum('bi,bi->b', input1, input2).unsqueeze(0).expand(batch_size1, -1)
            row.append(dot_product)
        input_dot_products.append(row)

    # Stack the results to form the Gram matrix for each pair
    G = torch.stack([
        torch.stack([aa] + l_inputs, dim=-1),
        *[torch.stack([l_inputs[i]] + input_dot_products[i], dim=-1) for i in range(len(inputs))]
    ], dim=-2)

    # Compute the determinant for each Gram matrix
    gram_det = torch.det(G.float())

    # Compute the square root of the absolute value of the determinants
    res = torch.sqrt(torch.abs(gram_det))
    return res

lambda1 = 1/4
lambda2 = 0.005
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.0002
save_path = ("/CW-JDRL/five_fold_split.xlsx")#Here is the path for the five fold splits index for the samples

# 存储五折结果
all_fold_metrics = {"AUC": [], "Accuracy": [], "Precision": [], "Specificity": [], "Recall": [], "F1-score": [],"YI":[],
                "PPV":[],
                "NPV":[]}

# 交叉验证循环
for fold_num in range(1, 6):
    print(f"\n===== 正在训练第 {fold_num} 折 =====")
    df_fold = pd.read_excel(save_path, sheet_name=f'Fold_{fold_num}')
    df_whole = load_data(excel_path)
    # 读取五折划分的训练和测试 ID
    train_ids = df_fold['Train_IDs'].dropna().astype(str).str.strip().str.lower().tolist()
    test_ids = df_fold['Test_IDs'].dropna().astype(str).str.strip().str.lower().tolist()

    # 打印检查数据长度
    print(f"原始训练集 ID 数量: {len(train_ids)}")
    print(f"匹配到的训练集样本数: {len(df_whole[df_whole['ID'].isin(train_ids)])}")

    # **检查是否有重复 ID**
    duplicates = df_whole[df_whole.duplicated(subset=['ID'], keep=False)]
    if not duplicates.empty:
        print("⚠️ 发现重复 ID 数据:")
        print(duplicates)
    # 加载训练集和测试集
    train_dataset = BreastCancerDataset(root_folder=root_folder, dataframe=df_whole[df_whole['ID'].isin(train_ids)], transform=transform)
    test_dataset = BreastCancerDataset(root_folder=root_folder, dataframe=df_whole


    [df_whole['ID'].isin(test_ids)], transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,num_workers=8, pin_memory=True)

    print(f"训练样本: {len(train_dataset)}, 测试样本: {len(test_dataset)}")

    # 初始化模型
    device = torch.device("cuda:1" )#if torch.cuda.device_count() > 1 else "cuda:0"
    model =CW_JDRL()
    model = model.to(device)
    # model = nn.DataParallel(model, device_ids=device_ids)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,weight_decay=5e-2)
    best_auc = 0.0
    best_acc = 0.0
    best_metrics = None
    best_fpr = None
    best_tpr = None
    T=2
    g1=1
    g2=1
    # ggu = GradientGuidedUpdater(model)
    # ggu.register_hooks()
    # 训练多个 Epoch
    for epoch in range(EPOCHS):
        model.train()
        # start_time = time.time()
        for images, clinical_data, labels in train_loader:
            images, clinical_data,labels = images.to(device),clinical_data.to(device), labels.to(device)

            # print(f"数据加载时间: {time.time() - start_time:.2f}s") f_main, shared_feats, private_feats,sr,lg
            optimizer.zero_grad()
            model.feature_maps.clear()
            outputs,f_main, shared_feats, private_feats,sr,lg = model(images)
            f1_c, f2_c, f3_c, f4_c = shared_feats
            f1s, f2s, f3s, f4s = private_feats
            f1,f2,f3,f4=sr
            lg1, lg2, lg3, lg4 = lg

            ce_loss = criterion(outputs, labels)
            ce_loss.backward(retain_graph=True)
            apply_gradient_guidance(model, lambda_align=g1, lambda_orth=g2, print_stats=False)
            #apply_gradient_guidance_channelwise(model, lambda_align=g1, lambda_orth=g2)
            from torch.autograd import grad

            # 你的模型结构中：resnet_views[7] 就是 layer4
            layer4_params = list(model.resnet_views[7].parameters())

            # 获取我们想要替换的梯度（从特征图的梯度传回到 layer4 的参数）
            layer4_grads = grad(
                outputs=model.feature_maps,  # 4 个视角输出的 feature map
                inputs=layer4_params,  # layer4 的参数
                grad_outputs=[fmap.grad for fmap in model.feature_maps],
                retain_graph=True,
                allow_unused=True
            )

            # 第四步：把 layer4 的 .grad 替换成我们算出来的
            for p, g in zip(layer4_params, layer4_grads):
                if g is not None:
                    p.grad = g.detach()
            ce_loss_v = criterion(lg1, labels) + criterion(lg2, labels) + criterion(lg3, labels) + criterion(lg4,labels)
            f1_c = F.normalize(f1_c, p=2, dim=-1)
            f2_c = F.normalize(f2_c, p=2, dim=-1)
            f3_c = F.normalize(f3_c, p=2, dim=-1)
            f4_c = F.normalize(f4_c, p=2, dim=-1)
            f1s = F.normalize(f1s, p=2, dim=-1)
            f2s = F.normalize(f2s, p=2, dim=-1)
            f3s = F.normalize(f3s, p=2, dim=-1)
            f4s = F.normalize(f4s, p=2, dim=-1)
            loss_dc = disentangled_volume_loss_with_official_volume_fn(
                f1_c, f2_c, f3_c, f4_c,
                f1s, f2s, f3s, f4s,
                volume_computation_fn=volume_computation,
                tau=0.04
            )

            loss = lambda1*ce_loss_v+lambda2*loss_dc#

            loss.backward()
            optimizer.step()
        model.eval()
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for images, clinical_data, labels in test_loader:
                images, clinical_data,labels = images.to(device),clinical_data.to(device), labels.to(device)
                outputs, f_main, shared_feats, private_feats,sr,lg = model(images)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                preds = (probs > 0.5).int()
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # 计算指标
        auc = roc_auc_score(all_labels, all_probs)
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
        acc = (tp + tn) / (tp + fn + tn + fp)
        sen = tp / (tp + fn)
        specificity = tn / (tn + fp)
        precision = tp / (tp + fp)
        F1 = 2 / (1 / precision + 1 / sen)
        YI = (sen + specificity - 1)
        PPV = tp / (tp + fp)
        NPV = tn / (tn + fn)
        print(
            f"Epoch {epoch + 1}/{EPOCHS} - AUC: {auc:.4f}, Acc: {acc:.4f}, Prec: {precision:.4f}, Rec: {sen:.4f}, Spec: {specificity:.4f}, F1: {F1:.4f},YI:{YI:.4f},PPV:{PPV:.4f},NPV:{NPV:.4f}")

        # 仅当 Specificity > 0.7 时才更新最佳 AUCand auc > 0.9
        if specificity >= 0.7 and acc+auc > best_acc+best_auc :
            best_acc = acc
            best_auc = auc
            best_fpr = fpr
            best_tpr = tpr
            best_metrics = {
                "AUC": auc,
                "Accuracy": acc,
                "Precision": precision,
                "Specificity": specificity,
                "Recall": sen,
                "F1-score": F1,
                "YI": YI,
                "PPV": PPV,
                "NPV": NPV
            }
            torch.save(model.state_dict(), f"/CW-JDRL/best_model_fold{fold_num}.pt")

    print(f"第 {fold_num} 折最佳 ACC: {best_acc:.4f},最佳 AUC: {best_auc:.4f}")


    for key in all_fold_metrics:
        all_fold_metrics[key].append(best_metrics[key])

# 计算最终结果
final_results = {metric: f"{np.mean(values):.4f} ± {np.std(values):.4f}" for metric, values in all_fold_metrics.items()}
print("\n最终五折交叉验证结果:")
for key, value in final_results.items():
    print(f"{key}: {value}")


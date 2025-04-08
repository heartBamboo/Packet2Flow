import math
import os

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sympy.core.evalf import evalf_rational
from torch.utils.data import Dataset, DataLoader, random_split

class MyDataset(Dataset):
    def __init__(self, data,labels, transform=None):
        self.data = data  # 假设 data 是一个包含所有样本的列表或数组
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample, label


def dataload(data_path):
    print('Loading data...')
    all_data_list = []
    all_labels_list = []
    for root, dirs, files in os.walk(data_path):
        # 创建正向映射字典：类别名称 -> 索引
        label_to_index = {label: idx for idx, label in enumerate(dirs)}
        # 创建反向映射字典：索引 -> 类别名称
        index_to_label = {idx: label for label, idx in label_to_index.items()}
        print("Label to Index Mapping:", label_to_index)

        for dir in dirs:
            if dir == 'a':
                continue
            for root, dirs, files in os.walk(os.path.join(data_path, dir)):
                for file in files:
                    if file.endswith('.npy'):
                        file_path = os.path.join(root, file)
                        # 读取 .ISCX_2016 文件
                        data = np.load(file_path)
                        all_data_list.append(data)
                        # 处理数据
                        print(f"Loaded file: {file_path}")
                        print(f"Data shape: {data.shape}")
                     # 进行其他处理
                        labels=np.full(data.shape[0], label_to_index[dir])
                        all_labels_list.extend(labels)
        for i ,data_np in enumerate(all_data_list):
            print(data_np.shape)
            if len(data_np.shape)==2:
                data_np=data_np.reshape(-1,8,data_np.shape[1])
                all_data_list[i]=data_np
        all_data_np=np.vstack(all_data_list)
        print("Processing complete.")
        print("全部输入的流量数据的维度为:",all_data_np.shape)
        return all_data_np,all_labels_list
        break


def create_dataset(dataset_path, transform=None,mode='to_pretrain'):
    all_data,all_labels = dataload(dataset_path)

    if mode == 'to_pretrain':
        packet_len = all_data.shape[2]
        sqrt = int(math.sqrt(packet_len))
        expand_data_np = all_data.reshape(-1, sqrt, sqrt)
        expanded_labels = np.repeat(all_labels, all_data.shape[1])
        # 使用 random_split 随机划分数据集
        # train_data, test_data, train_labels, test_labels = train_test_split(
        #     expand_data_np, expanded_labels, train_size=0.8, stratify=expanded_labels, random_state=42
        # )
        # train_dataset = MyDataset(train_data, train_labels, transform=transform)
        # test_dataset = MyDataset(test_data, test_labels, transform=transform)
        # print("Train dataset size:", len(train_dataset))
        # print("Test dataset size:", len(test_dataset))
        pretrain_dataset= MyDataset(expand_data_np, expanded_labels, transform=transform)
        return pretrain_dataset

    else:
        print("拆分数据集...")
        packet_len = all_data.shape[2]
        sqrt = int(math.sqrt(packet_len))
        expand_data_np = all_data.reshape(-1,all_data.shape[1], sqrt, sqrt)
        expanded_labels = all_labels

        # 使用 random_split 随机划分数据集
        train_valid_data, test_data, train_valid_labels, test_labels = train_test_split(
            expand_data_np, expanded_labels, test_size=0.1, stratify=expanded_labels, random_state=42
        )
        # 第二次分割：在训练 + 验证集中进一步分割为 75% 训练集和 25% 验证集（相对于原始数据集的60%和20%）
        train_data, valid_data, train_labels, valid_labels = train_test_split(
            train_valid_data, train_valid_labels, test_size=0.11, stratify=train_valid_labels, random_state=42
        )
        train_dataset = MyDataset(train_data, train_labels, transform=transform)
        valid_dataset = MyDataset(valid_data, valid_labels, transform=transform)
        test_dataset = MyDataset(test_data, test_labels, transform=transform)
        print("Train dataset size:", len(train_dataset))
        print("Valid dataset size:", len(valid_dataset))
        print("Test dataset size:", len(test_dataset))
        return train_dataset,valid_dataset, test_dataset

def create_dataloader(dataset, batch_size=32, shuffle=True):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
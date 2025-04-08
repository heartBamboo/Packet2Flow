import math
import os
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import TomekLinks, RandomUnderSampler
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
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
        lable_name =dirs
        label_to_index = {label: idx for idx, label in enumerate(dirs)}
        # 创建反向映射字典：索引 -> 类别名称
        index_to_label = {idx: label for label, idx in label_to_index.items()}
        print("Label to Index Mapping:", label_to_index)

        for dir in dirs:
            # if dir == 'a':
            #     continue
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
        return all_data_np,all_labels_list,lable_name



def pytorch_random_undersample(X: torch.Tensor, y: torch.Tensor):
    """
    PyTorch随机欠采样实现
    参数:
    X: 输入特征张量，形状为 (n_samples, 8, 32, 32)
    y: 标签张量，形状为 (n_samples,)
    返回:
    X_balanced: 欠采样后的特征张量
    y_balanced: 欠采样后的标签张量
    """
    # 获取类别分布
    unique_classes, counts = torch.unique(y, return_counts=True)
    min_samples = 450

    balanced_indices = []
    for cls in unique_classes:
    # 获取当前类别的所有样本索引
        cls_mask = (y == cls)
        cls_indices = torch.where(cls_mask)[0]

        # 如果当前类别样本数超过最小值，则随机采样
        if len(cls_indices) > min_samples:
            selected = torch.randperm(len(cls_indices))[:min_samples]
            balanced_indices.append(cls_indices[selected])
        else:
            balanced_indices.append(cls_indices)

        # 合并并打乱索引
    balanced_indices = torch.cat(balanced_indices)
    balanced_indices = balanced_indices[torch.randperm(len(balanced_indices))]

    return X[balanced_indices], y[balanced_indices]

def create_dataset(args, transform=None,mode='to_pretrain'):
    dataset_path = args.data_path
    all_data,all_labels,lable_name = dataload(dataset_path)
    # all_data = all_data[:2048,:,:]
    # all_labels = all_labels[:2048]

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
        #sampling_strategy = {0: 200,5: 200,2: 200,7:200,12:200,13:200}
        # # 创建一个 Pipeline 来依次应用 TomekLinks 和 RandomUnderSampler
        # pipeline = Pipeline([
        #     ('tomek', TomekLinks()),
        #     ('rus', RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42))
        # ])
        # all_data = all_data.reshape(-1, all_data.shape[1] * all_data.shape[2])
        # all_data, all_labels = pipeline.fit_resample(all_data, all_labels)
        # 使用 random_split 随机划分数据集
        # 计算要采样的样本数量（n的三分之一）
        # sample_count =  6000
        #
        # # 从 n 中随机选择 sample_count 个索引
        # selected_indices = np.random.choice(all_data.shape[0], sample_count, replace=False)
        #
        #
        # # 根据选中的索引从原始张量中抽取数据
        #
        # sampled_tensor = all_data[selected_indices]
        # all_labels = torch.tensor(all_labels)
        # sampled_labels = all_labels[selected_indices]
        # sampled_labels =  sampled_labels.tolist()

        # all_data = all_data.reshape(-1, all_data.shape[1] * all_data.shape[2])
        # sampling_strategy = {0: 400, 1: 400, 3: 400, 4: 400, 6: 400, 7: 400, 8: 400, 9: 400, 10: 400, 11: 800}
        #
        # pipeline = Pipeline([
        #     ('tomek', TomekLinks()),
        #     ('rus', RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42))
        # ])
        #
        # all_data, all_labels = pipeline.fit_resample(all_data, all_labels)

       # all_data, all_labels = pytorch_random_undersample(torch.tensor(all_data), torch.tensor(all_labels))
        #all_data=all_data.numpy()
        #all_labels=all_labels.tolist()
        train_data, valid_test_data, train_labels, valid_test_labels = train_test_split(
            all_data, all_labels, test_size=0.2, stratify=all_labels, random_state=42
        )
        # 第二次分割：在训练 + 验证集中进一步分割为 75% 训练集和 25% 验证集（相对于原始数据集的60%和20%）
        test_data, valid_data, test_labels, valid_labels = train_test_split(
            valid_test_data, valid_test_labels, test_size=0.5, stratify=valid_test_labels, random_state=42
        )
        # 输出分割前的标签分布
        #print("Original dataset labels: ", np.bincount(all_labels))
        print("Training set labels:     ", np.bincount(train_labels))
        print("valid set labels:      ", np.bincount(valid_labels))
        print("Test set labels:       ", np.bincount(test_labels))


        # #进行采样处理
        # train_data = train_data.reshape(-1, train_data.shape[1] * train_data.shape[2])
        # sampling_strategy = {0: 300,1: 300,3:300, 4:300,6:300,7:300,8:300,9:300,10:300,11:300}
        # # # 创建一个 Pipeline 来依次应用 TomekLinks 和 RandomUnderSampler
        # pipeline = Pipeline([
        #     ('tomek', TomekLinks()),
        #     ('rus', RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42))
        # ])
        # # # 执行重采样
        # train_data, train_labels = pipeline.fit_resample(train_data, train_labels)

        # # 处理成32x32的形式

        # 计算类别权重

        #class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)
        # class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
        class_weights_tensor=0
        # print("Class weights:", class_weights_tensor)
        #exit()

        train_labels_counts = np.bincount(train_labels)
        train_labels_weights = 1.0 / train_labels_counts
        sample_weights = train_labels_weights[train_labels]

        valid_labels_counts = np.bincount(valid_labels)
        valid_labels_weights = 1.0 / valid_labels_counts
        valid_sample_weights = valid_labels_weights[valid_labels]

        test_labels_counts = np.bincount(test_labels)
        test_labels_weights = 1.0 / test_labels_counts
        test_sample_weights = test_labels_weights[test_labels]

        datasets = {}
        data_dict = {
            'train_dataset': (train_data, train_labels),
            'valid_dataset': (valid_data, valid_labels),
            'test_dataset': (test_data, test_labels)
        }
        for dataset_name, (data, labels) in data_dict.items():
            data = data.reshape(data.shape[0], args.pack_len, -1)
            packet_len = data.shape[2]
            sqrt = int(math.sqrt(packet_len))
            data = data.reshape(-1, data.shape[1], sqrt, sqrt)
            dataset = MyDataset(data, labels, transform=transform)

            # 将数据集添加到字典中
            datasets[dataset_name] = dataset
            # 打印数据集大小
            print(f"{dataset_name} size:", len(dataset))



        # 返回三个数据集
        sample_weights = [sample_weights,valid_sample_weights,test_sample_weights]
        return datasets['train_dataset'], datasets['valid_dataset'], datasets['test_dataset'], lable_name,sample_weights,class_weights_tensor

def create_dataloader(dataset, sampler=None,batch_size=64,shuffle=True):
    if sampler is not None:
        print("Using sampler:", sampler)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
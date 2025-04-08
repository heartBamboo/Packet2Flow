import argparse
import math
import os
import torch
import numpy as np

from models.WTConv.wtconvnext.wtconvnext import wtconvnext_tiny

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def dataload(args):
    print('Loading data...')
    data_path = args.data_path
    all_data_list = []
    for root, dirs, files in os.walk(data_path):
        for dir in dirs:
            for root, dirs, files in os.walk(os.path.join(data_path, dir)):
                for file in files:
                    if file.endswith('.ISCXVPN'):
                        file_path = os.path.join(root, file)
                        # 读取 .ISCX_2016 文件
                        data = np.load(file_path)
                        all_data_list.append(data)
                        # 处理数据
                        print(f"Loaded file: {file_path}")
                        print(f"Data shape: {data.shape}")
                     # 进行其他处理
        all_data_np=np.vstack(all_data_list)
        print("Processing complete.")
        print("全部输入的流量数据的维度为:",all_data_np.shape)
        return all_data_np
        break

def data_preproces(data_np):
    flow_num,packts_num,packet_len = data_np.shape
    sqrt = int(math.sqrt(packet_len))
    data_np = data_np.reshape(flow_num, packts_num, 1,sqrt, sqrt)
    data_tensor = torch.from_numpy(data_np).to(dtype=torch.float32)
    return data_tensor

if __name__ == '__main__':
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="Process some integers.")

    # 添加参数
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--model', type=str, default='resnet50', help='Model architecture to use')
    parser.add_argument('--data-path', type=str, default='./datasets/ISCX_2016',help='Path to the dataset')

    # 解析参数
    args = parser.parse_args()
    data_np = dataload(args)
    data_tensor=data_preproces(data_np)
    model = wtconvnext_tiny(pretrained=False)
    print(model)
    print(count_parameters(model))
    batch_data = data_tensor[0:64]
    for flow in batch_data:
        print("每条流的维度为:",flow.shape)
        output = model(flow)
        print("输出维度为:",output.shape)
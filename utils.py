import random
from symbol import yield_arg

import numpy as np
import os
import time

from jinja2.nodes import args_as_const
from matplotlib import pyplot as plt
from torch import nn
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score, \
    BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score
import torch
import seaborn as sns
from datetime import datetime
import logging
import torch.nn.functional as F

import random
import numpy as np
import torch
from openTSNE import TSNE
import pandas as pd

def setup_seed(seed):
    """Function: 固定随机种子，使得模型每次结果唯一
    使用说明: 不用改变
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU，确保所有GPU的随机数生成器都使用相同的种子
    np.random.seed(seed)

    # 设置为 True 以保证结果的可重复性
    torch.backends.cudnn.deterministic = True

    # 禁用 benchmark 模式，以保证卷积算法的选择是确定性的
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """
    Computes and stores the average and current value
    Hacked from https://github.com/rwightman/pytorch-image-models/blob/master/timm/utils/metrics.py
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# confidence 1+n用这个
#def compute_metrics(output, target,low_confidence_sample_index, high_confidence_sample_index,arg, device):
def compute_metrics(output, target,arg, device):
    """
    Computes the accuracy, precision, recall, and F1 score for a multiclass classification problem.

    参数:
    - output: 模型输出的 logit 分数，形状为 (batch_size, num_classes)
    - target: 真实标签，形状为 (batch_size,)

    返回:
    - metrics_dict: 包含各个指标的字典
    """
    num_classes = arg.num_classes

    # confidence 1+n时候用用这组
    # 初始化指标对象
    # acc_metric = MulticlassAccuracy(num_classes=num_classes+1).to(device)
    # prec_metric = MulticlassPrecision(num_classes=num_classes+1, average='macro').to(device)
    # recall_metric = MulticlassRecall(num_classes=num_classes+1, average='macro').to(device)
    # f1_metric = MulticlassF1Score(num_classes=num_classes+1, average='macro').to(device)

    acc_metric = MulticlassAccuracy(num_classes=num_classes).to(device)
    prec_metric = MulticlassPrecision(num_classes=num_classes, average='macro').to(device)
    recall_metric = MulticlassRecall(num_classes=num_classes, average='macro').to(device)
    f1_metric = MulticlassF1Score(num_classes=num_classes, average='macro').to(device)


    # 将预测结果传递给指标对象
    output = torch.tensor(output,device=device)
    target = torch.tensor(target,device=device)


    # 计算各项指标
    accuracy = acc_metric(output, target)
    precision = prec_metric(output, target)
    recall = recall_metric(output, target)
    f1_score = f1_metric(output, target)


    # confidence 1+n时候用这下面这些注释的
    # all_pred = torch.full_like(target, 8, dtype=torch.long, device=device)  # 使用-1作为未赋值的标志
    # pred = output.argmax(dim=1)
    # print("low_confidence_sample_index.shape", low_confidence_sample_index.shape)
    # print("high_confidence_sample_index.shape", high_confidence_sample_index.shape)
    # for i in range(len(high_confidence_sample_index)):
    #     print(high_confidence_sample_index[i], end=" ")
    
    # all_pred[high_confidence_sample_index] = pred
    # all_pred[low_confidence_sample_index] = 8 # 未知样本隔离,把未知样本放到最后一个类

    # print("target:")
    # for i in range(len(target)):
    #     print(target[i].item(), end=" ")
    # print("all_pred:")
    # for i in range(len(all_pred)):
    #     print(all_pred[i].item(), end=" ")


    # # 计算各项指标
    # accuracy = acc_metric(all_pred, target)
    # precision = prec_metric(all_pred, target)
    # recall = recall_metric(all_pred, target)
    # f1_score = f1_metric(all_pred, target)


    # 返回包含所有指标的字典
    metrics_dict = {
        'accuracy': accuracy.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'f1_score': f1_score.item()
    }

    return metrics_dict



# confidence 1+n用这个
def compute_metrics_1n(output, target,low_confidence_sample_index, high_confidence_sample_index,arg, device):

    """
    Computes the accuracy, precision, recall, and F1 score for a multiclass classification problem.

    参数:
    - output: 模型输出的 logit 分数，形状为 (batch_size, num_classes)
    - target: 真实标签，形状为 (batch_size,)

    返回:
    - metrics_dict: 包含各个指标的字典
    """
    num_classes = arg.num_classes

    #初始化指标对象
    acc_metric = MulticlassAccuracy(num_classes=num_classes+1).to(device)
    prec_metric = MulticlassPrecision(num_classes=num_classes+1, average='macro').to(device)
    recall_metric = MulticlassRecall(num_classes=num_classes+1, average='macro').to(device)
    f1_metric = MulticlassF1Score(num_classes=num_classes+1, average='macro').to(device)


    # 将预测结果传递给指标对象
    output = torch.tensor(output,device=device)
    target = torch.tensor(target,device=device)

    all_pred = torch.full_like(target, 8, dtype=torch.long, device=device)  # 使用-1作为未赋值的标志
    pred = output.argmax(dim=1)
    print("low_confidence_sample_index.shape", low_confidence_sample_index.shape)
    print("high_confidence_sample_index.shape", high_confidence_sample_index.shape)
    for i in range(len(high_confidence_sample_index)):
        print(high_confidence_sample_index[i], end=" ")
    
    all_pred[high_confidence_sample_index] = pred
    all_pred[low_confidence_sample_index] = 8 # 未知样本隔离,把未知样本放到最后一个类

    print("target:")
    for i in range(len(target)):
        print(target[i].item(), end=" ")
    print("all_pred:")
    for i in range(len(all_pred)):
        print(all_pred[i].item(), end=" ")


    # 计算各项指标
    accuracy = acc_metric(all_pred, target)
    precision = prec_metric(all_pred, target)
    recall = recall_metric(all_pred, target)
    f1_score = f1_metric(all_pred, target)


    # 返回包含所有指标的字典
    metrics_dict = {
        'accuracy': accuracy.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'f1_score': f1_score.item()
    }

    return metrics_dict

def compute_metrics_binary(output, target, device):
    """
    Computes the accuracy, precision, recall, and F1 score for a binary classification problem.

    参数:
    - output: 模型输出的 logit 分数，形状为 (batch_size,)
    - target: 真实标签，形状为 (batch_size,)

    返回:
    - metrics_dict: 包含各个指标的字典
    """
    # 初始化指标对象
    acc_metric = BinaryAccuracy().to(device)
    prec_metric = BinaryPrecision().to(device)
    recall_metric = BinaryRecall().to(device)
    f1_metric = BinaryF1Score().to(device)

    # 将预测结果传递给指标对象
    output = output.to(device)  # 确保 logits 在正确设备上
    target = target.to(device)  # 确保标签在正确设备上

    # 对于二分类任务，通常使用 Sigmoid 函数将 logits 转换为概率，并根据阈值转换为类别预测
    #pred_probs = torch.sigmoid(output).squeeze()
    pred = (output > 0.5).int()  # 根据阈值 0.5 进行分类

    # 计算各项指标
    with torch.no_grad():  # 禁用梯度计算以节省内存
        accuracy = acc_metric(pred, target)
        precision = prec_metric(pred, target)
        recall = recall_metric(pred, target)
        f1_score = f1_metric(pred, target)

    # 返回包含所有指标的字典
    metrics_dict = {
        'accuracy': accuracy.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'f1_score': f1_score.item()
    }

    return metrics_dict

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          save_path=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    If `save_path` is provided, the plot will be saved to the specified file path instead of being shown.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    #else:
        #print('Confusion matrix, without normalization')

    #print(cm)

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap=cmap,
                xticklabels=classes, yticklabels=classes)  # 添加类别标签
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # 旋转 x 轴标签以避免重叠（可选）
    plt.xticks(rotation=90,fontsize=6)
    plt.yticks(rotation=0)

    if save_path:
        save_path=os.path.join(save_path, 'confusion_matrix.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')  # 保存图像并设置 DPI 和边界框
        #print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()  # 如果没有提供 save_path，则显示图像

    plt.close()  # 关闭当前图形以释放内存

def plot_convergence(train_losses, val_accuracies, num_epochs, save_path=None):
    """
    绘制训练损失和验证准确率的收敛图，并可以选择保存为图片。

    参数:
    - train_losses (list): 每个 epoch 的训练损失列表。
    - val_accuracies (list): 每个 epoch 的验证准确率列表。
    - num_epochs (int): 总的 epoch 数量。
    - save_path (str, optional): 保存图表的文件路径。如果不提供，则显示图表。
    """
    # 创建图表
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 绘制训练损失
    color = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss', color=color)
    ax1.plot(range(1, num_epochs + 1), train_losses, label='Training Loss', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # 创建第二个 y 轴用于验证准确率
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Validation Accuracy', color=color)  # 设置第二个 y 轴标签颜色
    ax2.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    # 添加标题和网格
    plt.title('Loss and Accuracy Convergence Over Epochs')
    fig.tight_layout()  # 确保标签不会重叠
    plt.grid(True)

    # 添加图例
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    # 保存或显示图表
    if save_path:
        save_path=os.path.join(save_path, 'convergence_plot.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')  # 保存图像并设置 DPI 和边界框
        print(f"Convergence plot saved to {save_path}")
    else:
        plt.show()

    plt.close()  # 关闭当前图形以释放内存

def plot_one(list, String,num_epochs, save_path=None):
    """
    绘制训练损失的收敛图，并可以选择保存为图片。

    参数:
    - train_losses (list): 每个 epoch 的训练损失列表。
    - num_epochs (int): 总的 epoch 数量。
    - save_path (str, optional): 保存图表的文件路径。如果不提供，则显示图表。
    """
    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制训练损失
    color = 'tab:blue'
    ax.set_xlabel('Epoch')
    ax.set_ylabel(String, color=color)
    ax.plot(range(1, num_epochs + 1), list, label=String, color=color)
    ax.tick_params(axis='y', labelcolor=color)

    # 添加标题和网格
    plt.title(String+'Convergence Over Epochs')
    plt.grid(True)

    # 添加图例
    ax.legend(loc='upper right')

    # 保存或显示图表
    if save_path:
        save_path = os.path.join(save_path, f'{String}_plot.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')  # 保存图像并设置 DPI 和边界框
        print(f'{String} plot saved to {save_path}')
    else:
        plt.show()

    plt.close()  # 关闭当前图形以释放内存


class ModelTrainer(object):
    def  train_one_epoch(train_dataloader, model, loss_f, optimizer, epoch_idx, device,logger,args,wt_mask_ratio=0, bert_mask_ratio=0):
        model.train()
        end = time.time()
        loss_m = AverageMeter()
        batch_time_m = AverageMeter()
        last_idx = len(train_dataloader) - 1
        alpha = 1 # 给 loss1 的权重
        beta = 0.1  # 给 loss2 的权重
        lambda_entropy = 0.1 # 给 用来辅助置信度机制的softmax熵的损失 的权重
        epsilon = 1e-12
        for batch_idx, (batch_data,batch_label) in enumerate(train_dataloader):
            if args.num_classes == 2:
                batch_label = batch_label.float().to(device, non_blocking=True).unsqueeze(1)
            else:
                batch_label = batch_label.to(device, non_blocking=True)
            if args.mode != 'to_pretrain':
                batch_data = batch_data.permute(0, 2, 1, 3).unsqueeze(2)
                batch_data = batch_data.reshape(-1,1, batch_data.size(-1), batch_data.size(-1))
            batch_data = batch_data.to(device, non_blocking=True)
            #print(batch_data.shape)
            # forward & backward
            if wt_mask_ratio > 0: #预训练
                wt_conv_output, bert_input, bert_output,mask= model(batch_data, wt_mask_ratio, bert_mask_ratio)
                #wtconv
                batch_size, hidden, patch, _ = wt_conv_output.shape
                batch_data_reshape = batch_data.reshape(batch_size, hidden, patch, patch)
                loss1 = loss_f(wt_conv_output, batch_data_reshape)
                #bert
                y_true = bert_input[mask]
                cls_mask = torch.zeros(mask.size(0), 1, device=device)
                mask = torch.cat((cls_mask, mask), dim=1).bool()
                y_pred = bert_output[mask]
                loss2 = loss_f(y_pred, y_true)

                if torch.isnan(loss2):
                    print("Loss is nan")
                    print(f"batch_idx: {batch_idx}, epoch_idx: {epoch_idx}")
                    print(f"batch_data contains nan: {torch.isnan(batch_data).any()}")
                    print(f"batch_label contains nan: {torch.isnan(batch_label).any()}")
                    break

                loss = alpha * loss1 + beta * loss2
                logger.info("loss1={0},loss2={1}".format(loss1, loss2))
            else: #微调
                if args.confidence:
                    #print("confidence begin work")
                    bert_output,confidence = model(batch_data, wt_mask_ratio, bert_mask_ratio)
                    mask = confidence < args.confidence
                    # for i in range(len(confidence)):
                    #     if(confidence[i].item() < args.confidence):
                    #         print("confidence:", confidence[i])


                    mask_half = torch.bernoulli(torch.Tensor(confidence.size()).uniform_(0, 1)).bool().cuda()
                    mask = mask * mask_half
                    
                    # 随机掩50%的置信度
                    #mask = torch.bernoulli(torch.Tensor(confidence.size()).uniform_(0, 1)).bool().cuda()
                    Temperature = 10.0
                    probabilities = F.softmax(bert_output / Temperature, dim=1)
                    labels_one_hot = F.one_hot(batch_label, num_classes=args.num_classes).float()
                    # 使用掩码选择需要调整的样本
                    selected_probabilities = probabilities[mask.squeeze(1)]
                    print("未达到阈值的样本数为：",selected_probabilities.size(0))
                    selected_labels_one_hot = labels_one_hot[mask.squeeze(1)]
                    selected_confidence = confidence[mask].unsqueeze(1)

                    # 只对满足条件的样本计算 modified_probabilities
                    if selected_probabilities.numel() > 0:
                        modified_probabilities = selected_confidence * selected_probabilities + (1 - selected_confidence) * selected_labels_one_hot
                        probabilities_clone = probabilities.clone()
                        # 将 modified_probabilities 放回原张量中
                        probabilities_clone[mask.squeeze(1)] = modified_probabilities
                        probabilities = probabilities_clone
                        # Classification Loss
                        classification_loss = -torch.sum(labels_one_hot * torch.log(probabilities + epsilon), dim=1).mean()
                        # Confidence Loss
                        confidence_loss = -torch.log(confidence + epsilon).mean()

                        #计算熵损失,鼓励已知类别的熵值较低
                        entropy = -torch.sum(probabilities * torch.log(probabilities + epsilon), dim=1, keepdim=True)
                        entropy_loss = torch.mean(entropy)

                        print(
                          "classification_loss={0},confidence_loss={1}, entropy_loss={2}".format(classification_loss, confidence_loss, entropy_loss))
                        # Total Loss
                        loss = alpha * classification_loss + beta * confidence_loss + lambda_entropy * entropy_loss


                        if args.confidence > confidence_loss.item():
                                beta = beta / 1.01
                        elif args.confidence <= confidence_loss.item():
                                beta = beta / 0.99
                    else:
                        #print("无需修改概率")
                        loss = loss_f(bert_output, batch_label)



                # print(bert_output.shape)
                # print(bert_output)
                # print(batch_label.shape)
                # print(batch_label)
                # exit()
                else:
                    bert_output = model(batch_data, wt_mask_ratio, bert_mask_ratio)
                    loss = loss_f(bert_output, batch_label)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
            optimizer.step()

            # 记录指标
            loss_m.update(loss.item(), batch_label.size(0))  # 因update里： self.sum += val * n， 因此需要传入batch数量
            # 打印训练信息
            batch_time_m.update(time.time() - end)
            if batch_idx % args.print_freq == args.print_freq - 1:
                logger.info(
                    'Epoch: [{0}]'
                    '{1}: [{2:>4d}/{3}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})'.format(
                        epoch_idx,"train", batch_idx, last_idx, batch_time=batch_time_m,
                        loss=loss_m))
        return loss_m


    def validate(data_loader, model, loss_f,  device,logger,args,wt_mask_ratio=0, bert_mask_ratio=0):
        class_num = args.num_classes
        model.eval()
        #conf_mat = np.zeros((class_num, class_num))
        loss_m_valid = AverageMeter()
        last_idx = len(data_loader) - 1
        true_label_list = []
        pred_label_list = []
        confidence_list = []
        confidence_meter = AverageMeter()
        with torch.no_grad():
            for batch_idx, (batch_data,batch_label) in enumerate(data_loader):
                batch_data = batch_data.permute(0, 2, 1, 3).unsqueeze(2)
                batch_data = batch_data.reshape(-1, 1, batch_data.size(-1), batch_data.size(-1))
                if class_num == 2:
                    batch_label = batch_label.float().to(device, non_blocking=True).unsqueeze(1)
                else:
                    batch_label = batch_label.to(device, non_blocking=True)
                batch_data = batch_data .to(device)
                #print(batch_data.shape)
                if args.confidence:
                    bert_output,confidence = model(batch_data, wt_mask_ratio, bert_mask_ratio)
                    confidence_list.append(confidence)
                    confidence_meter.update(confidence.mean().item(), n=confidence.size(0))
                else:
                    bert_output= model(batch_data, wt_mask_ratio, bert_mask_ratio)
                loss = loss_f(bert_output, batch_label)
                # 记录指标
                loss_m_valid.update(loss.item(), batch_label.size(0))  # 因update里： self.sum += val * n， 因此需要传入batch数量
                true_label_list.append(batch_label)
                pred_label_list.append(bert_output)

                #metrics_dict = compute_metrics(outputs, batch_label, device)

                if batch_idx % args.print_freq == args.print_freq - 1:
                    logger.info(
                        '{0}: [{1:>4d}/{2}]  '
                        'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f}) '.format(
                            "Valid",
                            batch_idx,  # 确保批次索引从1开始显示
                            last_idx,
                            loss=loss_m_valid,
                        )
                    )

        #return loss_m_valid,metrics_dict,conf_mat, all_hidden_features, all_labels
        y_pred = torch.cat(pred_label_list).cpu().numpy()
        y_true = torch.cat(true_label_list).cpu().numpy()
        if args.confidence:
            average_confidence = confidence_meter.avg
            return loss_m_valid,y_pred, y_true,average_confidence
        return loss_m_valid,y_pred, y_true


    def evaluate(data_loader, model, loss_f,  device,logger,args,wt_mask_ratio=0, bert_mask_ratio=0):
        class_num = args.num_classes
        model.eval()
        conf_mat = np.zeros((class_num, class_num))
        loss_m_valid = AverageMeter()
        last_idx = len(data_loader) - 1
        true_label_list = []
        pred_label_list = []
        all_index_list = [] # 记录所有批次置信度高于阈值的标签索引
        all_exclude_index_list = [] # 记录所有批次置信度低于阈值的标签索引
        with torch.no_grad():
            for batch_idx, (batch_data, batch_label) in enumerate(data_loader):
                batch_data = batch_data.permute(0, 2, 1, 3).unsqueeze(2)
                batch_data = batch_data.reshape(-1, 1, batch_data.size(-1), batch_data.size(-1))
                batch_data, batch_label = batch_data.to(device), batch_label.to(device)

                if args.confidence:
                    #confidence_1n
                    bert_output, confidence, index_list, exclude_index_list = model(batch_data, wt_mask_ratio, bert_mask_ratio)
                    
                    #finetune
                    # bert_output, confidence = model(batch_data, wt_mask_ratio, bert_mask_ratio)

                    #batch_label_remain = batch_label[index_list]      #只有置信度高的进了后面

                else:
                    bert_output= model(batch_data, wt_mask_ratio, bert_mask_ratio)

                #loss = loss_f(bert_output, batch_label_remain)      
                # 记录指标
                # loss_m_valid.update(loss.item(), batch_label.size(0))  # 因update里： self.sum += val * n， 因此需要传入batch数量
                #loss_m_valid.update(loss.item(), batch_label_remain.size(0))  # 因update里： self.sum += val * n， 因此需要传入batch数量

                true_label_list.append(batch_label)
                pred_label_list.append(bert_output)
                
                #用这个需要解决最后一个batch不满的情况，这样就会导致乘的数字变小（要把具体的batch_size传进来）
                #all_index_list.append(index_list + batch_idx * batch_data.shape[0] / 8)
                #all_exclude_index_list.append(exclude_index_list + batch_idx * batch_data.shape[0] / 8)
                


                # batch_size = 32
                all_index_list.append(index_list + batch_idx * args.batch_size)
                all_exclude_index_list.append(exclude_index_list + batch_idx * args.batch_size)

                # metrics_dict = compute_metrics(outputs, batch_label, device)

                # if batch_idx % args.print_freq == args.print_freq - 1:
                #     logger.info(
                #         '{0}: [{1:>4d}/{2}]  '
                #         'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f}) '.format(
                #             "Valid",
                #             batch_idx,  # 确保批次索引从1开始显示
                #             last_idx,
                #             loss=loss_m_valid,
                #         )
                #     )

        # return loss_m_valid,metrics_dict,conf_mat, all_hidden_features, all_labels
        y_pred = torch.cat(pred_label_list).cpu().numpy()
        y_true = torch.cat(true_label_list).cpu().numpy()
        high_confidence_sample_index = torch.cat(all_index_list).cpu().numpy()
        low_confidence_sample_index = torch.cat(all_exclude_index_list).cpu().numpy()
        return loss_m_valid, y_pred, y_true, low_confidence_sample_index, high_confidence_sample_index

    def train_one_epoch_only_wtconv(train_dataloader, model, loss_f, optimizer, epoch_idx, device, logger, args,
                                    wt_mask_ratio=0, bert_mask_ratio=0):
        model.train()
        end = time.time()
        loss_m = AverageMeter()
        batch_time_m = AverageMeter()
        last_idx = len(train_dataloader) - 1
        alpha = 1  # 给 loss1 的权重
        beta = 1  # 给 loss2 的权重
        epsilon = 1e-12
        for batch_idx, (batch_data, batch_label) in enumerate(train_dataloader):
            if args.num_classes == 2:
                batch_label = batch_label.float().to(device, non_blocking=True).unsqueeze(1)
            else:
                batch_label = batch_label.to(device, non_blocking=True)
            if args.mode != 'to_pretrain':
                batch_data = batch_data.permute(0, 2, 1, 3).unsqueeze(2)
                batch_data = batch_data.reshape(-1, 1, batch_data.size(-1), batch_data.size(-1))
            batch_data = batch_data.to(device, non_blocking=True)
            # print(batch_data.shape)
            # forward & backward
            if wt_mask_ratio > 0:
                wt_conv_output, bert_input, bert_output, mask = model(batch_data, wt_mask_ratio, bert_mask_ratio)
                # wtconv
                batch_size, hidden, patch, _ = wt_conv_output.shape
                batch_data_reshape = batch_data.reshape(batch_size, hidden, patch, patch)
                loss1 = loss_f(wt_conv_output, batch_data_reshape)
                # bert
                y_true = bert_input[mask]
                cls_mask = torch.zeros(mask.size(0), 1, device=device)
                mask = torch.cat((cls_mask, mask), dim=1).bool()
                y_pred = bert_output[mask]
                loss2 = loss_f(y_pred, y_true)
                loss = alpha * loss1 + beta * loss2
                # logger.info("loss1={0},loss2={1}".format(loss1, loss2))
            else:
                if args.confidence:
                    bert_output, confidence = model(batch_data, wt_mask_ratio, bert_mask_ratio)
                    mask = confidence < args.confidence
                    probabilities = F.softmax(bert_output, dim=1)
                    labels_one_hot = F.one_hot(batch_label, num_classes=args.num_classes).float()
                    # 使用掩码选择需要调整的样本
                    selected_probabilities = probabilities[mask.squeeze(1)]
                    selected_labels_one_hot = labels_one_hot[mask.squeeze(1)]
                    selected_confidence = confidence[mask].unsqueeze(1)

                    # 只对满足条件的样本计算 modified_probabilities
                    if selected_probabilities.numel() > 0:
                        modified_probabilities = selected_confidence * selected_probabilities + (
                                    1 - selected_confidence) * selected_labels_one_hot
                        probabilities_clone = probabilities.clone()
                        # 将 modified_probabilities 放回原张量中
                        probabilities_clone[mask.squeeze(1)] = modified_probabilities
                        probabilities = probabilities_clone

                    # Classification Loss
                    classification_loss = -torch.sum(labels_one_hot * torch.log(probabilities + epsilon), dim=1).mean()
                    # Confidence Loss
                    confidence_loss = -torch.log(confidence + epsilon).mean()

                    # Total Loss
                    loss = classification_loss + beta * confidence_loss
                # print(bert_output.shape)
                # print(bert_output)
                # print(batch_label.shape)
                # print(batch_label)
                # exit()
                else:
                    wtconv_output = model(batch_data, wt_mask_ratio, bert_mask_ratio)
                    loss = loss_f(wtconv_output, batch_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 记录指标
            loss_m.update(loss.item(), batch_label.size(0))  # 因update里： self.sum += val * n， 因此需要传入batch数量
            # 打印训练信息
            batch_time_m.update(time.time() - end)
            if batch_idx % args.print_freq == args.print_freq - 1:
                logger.info(
                    'Epoch: [{0}]'
                    '{1}: [{2:>4d}/{3}]  '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})'.format(
                        epoch_idx, "train", batch_idx, last_idx, batch_time=batch_time_m,
                        loss=loss_m))
        return loss_m

    def validate_only_wtconv(data_loader, model, loss_f,  device,logger,args,wt_mask_ratio=0, bert_mask_ratio=0):
        class_num = args.num_classes
        model.eval()
        #conf_mat = np.zeros((class_num, class_num))
        loss_m_valid = AverageMeter()
        last_idx = len(data_loader) - 1
        true_label_list = []
        pred_label_list = []
        confidence_list = []
        confidence_meter = AverageMeter()
        with torch.no_grad():
            for batch_idx, (batch_data,batch_label) in enumerate(data_loader):
                batch_data = batch_data.permute(0, 2, 1, 3).unsqueeze(2)
                batch_data = batch_data.reshape(-1, 1, batch_data.size(-1), batch_data.size(-1))
                if class_num == 2:
                    batch_label = batch_label.float().to(device, non_blocking=True).unsqueeze(1)
                else:
                    batch_label = batch_label.to(device, non_blocking=True)
                batch_data = batch_data.to(device)
                #print(batch_data.shape)
                if args.confidence:
                    bert_output,confidence = model(batch_data, wt_mask_ratio, bert_mask_ratio)
                    confidence_list.append(confidence)
                    confidence_meter.update(confidence.mean().item(), n=confidence.size(0))
                else:
                    bert_output= model(batch_data, wt_mask_ratio, bert_mask_ratio)
                loss = loss_f(bert_output, batch_label)
                # 记录指标
                loss_m_valid.update(loss.item(), batch_label.size(0))  # 因update里： self.sum += val * n， 因此需要传入batch数量
                true_label_list.append(batch_label)
                pred_label_list.append(bert_output)

                #metrics_dict = compute_metrics(outputs, batch_label, device)

                if batch_idx % args.print_freq == args.print_freq - 1:
                    logger.info(
                        '{0}: [{1:>4d}/{2}]  '
                        'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f}) '.format(
                            "Valid",
                            batch_idx,  # 确保批次索引从1开始显示
                            last_idx,
                            loss=loss_m_valid,
                        )
                    )

        #return loss_m_valid,metrics_dict,conf_mat, all_hidden_features, all_labels
        y_pred = torch.cat(pred_label_list).cpu().numpy()
        y_true = torch.cat(true_label_list).cpu().numpy()
        if args.confidence:
            average_confidence = confidence_meter.avg
            return loss_m_valid,y_pred, y_true,average_confidence
        return loss_m_valid,y_pred, y_true

class Logger(object):
    def __init__(self, path_log):
        log_name = os.path.basename(path_log)
        self.log_name = log_name if log_name else "root"
        self.out_path = path_log

        log_dir = os.path.dirname(self.out_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    def init_logger(self):
        logger = logging.getLogger(self.log_name)
        logger.setLevel(level=logging.INFO)

        # 配置文件Handler
        file_handler = logging.FileHandler(self.out_path, 'w')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # 配置屏幕Handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        # console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

        # 添加handler
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger


def make_logger(out_dir):
    """
    在out_dir文件夹下以当前时间命名，创建日志文件夹，并创建logger用于记录信息
    :param out_dir: str
    :return:
    """
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%Y-%m-%d_%H-%M-%S')
    log_dir = os.path.join(out_dir, time_str)  # 根据config中的创建时间作为文件夹名
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # 创建logger
    path_log = os.path.join(log_dir, "log.log")
    logger = Logger(path_log)
    logger = logger.init_logger()
    return logger, log_dir


def visualize(hidden_features, labels,target_names,save_path,stage):
    # 确保 hidden_features 和 labels 是 NumPy 数组
    hidden_features = np.array(hidden_features)

    # 如果 labels 是 PyTorch 张量，将其转换为 NumPy 数组
    if isinstance(labels, list) and isinstance(labels[0], torch.Tensor):
        labels = np.array([label.detach().cpu().numpy() for label in labels])
    else:
        labels = np.array(labels)

    # 使用 t-SNE 进行降维
    print("正在降维")
    tsne = TSNE(perplexity=10, n_jobs=-1, random_state=42)
    embedding = tsne.fit(hidden_features)

    # 定义类别名称
    #target_names = ['audio', 'video', 'p2p', 'browsing', 'voip', 'email', 'chat', 'file']

    # 创建 DataFrame 以便于绘图
    df = pd.DataFrame({
        'tsne-one': embedding[:, 0],
        'tsne-two': embedding[:, 1],
        'species': [target_names[i] for i in labels]
    })

    # 使用 seaborn 绘制 t-SNE 结果
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        x='tsne-one',
        y='tsne-two',
        hue='species',
        style='species',
        palette='deep',
        data=df,
        s=100,
        alpha=0.6
    )
    plt.title('t-SNE Visualization of Network Traffic Classes')
    plt.legend(title='Traffic Class')

    # 创建目录并保存图像
    print('Saving chart...', end='\r')
    #os.makedirs('./visualize', exist_ok=True)
    plt.savefig(os.path.join(save_path, f'{stage}_t-sne.jpg'), dpi=300, bbox_inches='tight')
    plt.close()
    print('Chart saved successfully!')


class ClassificationHead(nn.Module):
    def  __init__(self, input_dim=256, num_classes=10, dropout_rate=0):
        super(ClassificationHead, self).__init__()

        # 定义一层或几层的全连接层，可以根据需要调整
        self.fc1 = nn.Linear(input_dim, 512)  # 第一层可以映射到更高维度

        # 可选：Batch Normalization 层
        self.bn1 = nn.BatchNorm1d(512)

        # Dropout 层来防止过拟合
        self.dropout = nn.Dropout(dropout_rate)

        # 输出层，映射到类别数
        if num_classes == 2:
            self.fc_out = nn.Linear(512, 1)
            nn.init.xavier_uniform_(self.fc_out.weight, gain=nn.init.calculate_gain('sigmoid'))
        else:
            self.fc_out = nn.Linear(512, num_classes)
            nn.init.xavier_uniform_(self.fc_out.weight, gain=0.1)
        nn.init.constant_(self.fc_out.bias, 0)

        # nn.init.xavier_normal_(self.fc_out.weight)
        # nn.init.constant_(self.fc_out.bias, 0)
        # nn.init.kaiming_uniform_(self.fc_out.weight, mode='fan_in', nonlinearity='relu')
        # nn.init.constant_(self.fc_out.bias, 0)



    def forward(self, x):
        # 假设 x 的形状是 (batch_size, 256)

        # 全连接层 + ReLU 激活 + BatchNorm
        x = self.fc1(x)
        x = F.relu(x)
        x = self.bn1(x)

        # Dropout
        x = self.dropout(x)
        # 输出层
        logits = self.fc_out(x)

        return logits



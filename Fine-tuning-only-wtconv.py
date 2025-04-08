import argparse
import os
import time
from cProfile import label
from pickletools import optimize
from sys import flags

import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torch.nn.functional import dropout
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch
import torchvision.transforms as transforms

from Wtconv_Bert import wtconv_bert
from data_process import create_dataset,create_dataloader
from utils import setup_seed, ModelTrainer, AverageMeter, make_logger, plot_confusion_matrix, visualize, \
    compute_metrics, plot_convergence, plot_one
from models.WTConv.wtconvnext.wtconvnext import wtconvnext_tiny


from models.WTConv.wtconvnext.wtconvnext import wtconvnext_tiny

if __name__ == '__main__':
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="Process some integers.")

    # 添加参数
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=40, help='Number of epochs to train')
    parser.add_argument('--model', type=str, default='wtconv_bert', help='Model architecture to use')
    parser.add_argument('--data_path', type=str, default='./new_datasets/ISCX_VPN_2016_time256_w8',help='Path to the dataset')
    parser.add_argument('--pretrain_valid', type=bool, default=False, help='decide to pretrain')
    parser.add_argument("--print-freq", default=25, type=int, help="step to print frequency")
    parser.add_argument("--output-dir", default="./Finetune_result", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--lr", default=1e-3, type=float, help="initial learning rate")
    parser.add_argument("--save_epoch", default=10, type=int)
    parser.add_argument("--num_classes", default=12, type=int)
    parser.add_argument("--pretrained_model_path", default='./pretrained_model/new_pretrain/wtconvbert_checkpoint_49_batchsize_128_wt_maskratio_0.8_bert_maskratio0.6_dataset_pre_train_w8.pth', type=str)
    parser.add_argument("--finetuned_model_path", default='./pretrained_model/finetune_checkpoint_250_batchsize_64_acc_0.8727587461471558_dataname-vpn_w8_paralel.pth', type=str)
    parser.add_argument("--mode", default='only_wtconv', type=str,choices=['to_pretrain','to_finetune','just_evaluate','only_wtconv'])
    parser.add_argument("--bert_dropout", default=0.01, type=float, help='dropout rate')
    parser.add_argument("--wt_dropout", default=0.01, type=float, help='dropout rate')
    parser.add_argument("--head_dropout", default=0.01, type=float, help='dropout rate')
    parser.add_argument("--wt_drop_path_rate", default=0.01, type=float, help='dropout rate')
    parser.add_argument("--pack_len", default=8, type=int)
    parser.add_argument("--confidence", default=0.0, type=float)

    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=2e-4,#1e-2--5e-4
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )

    # 解析参数
    args = parser.parse_args()
    if args.mode=='to_finetune':
        pretrained_model_path=args.pretrained_model_path
    elif args.mode=='just_evaluate':
        pretrained_model_path=args.finetuned_model_path
    else:
        pretrained_model_path = args.pretrained_model_path
    # 图片预处理参数
    norm_mean = 0.5  # RGB图像的均值
    norm_std = 0.5  # RGB图像的标准差
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    # 基础参数
    setup_seed(3406)
    result_dir = args.output_dir
    save_epoch = args.save_epoch
    mode = args.mode
    dataset_name = args.data_path.split('/')[-1]

    #加载数据集
    train_dataset,valid_dataset,test_dataset,lable_name= create_dataset(args, transform=transform, mode=mode)
    print("lable_name=", lable_name)
    train_dataloader = create_dataloader(train_dataset, batch_size=args.batch_size)
    valid_dataloader = create_dataloader(valid_dataset, batch_size=args.batch_size)
    test_dataloader = create_dataloader(test_dataset, batch_size=args.batch_size)

    #加载已经预训练好了的模型
    model = wtconv_bert(args,mode=mode)
    print(model)
    #exit()
    pretrained_dict = torch.load(pretrained_model_path)['model_state_dict']
    model.load_state_dict(pretrained_dict,strict=False)
    # param_count = sum([m.numel() for m in model.parameters()])
    # print('Model Param Count: ', param_count)
    print("参数加载完成")
    # 冻结所有层
    # for name, param in model.named_parameters():
    #
    #     print(f"Layer: {name}, Shape: {param.shape}")
    #     print(param)
        # if 'head' in name:
        #     param.requires_grad = True
        # else:
        #     param.requires_grad = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 定义优化器和损失函数和调度器
    optimizer = optim.AdamW(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)#\
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=2, mode='max')

    # 打印日志
    # ------------------------------------  log ------------------------------------
    logger, log_dir = make_logger(result_dir)
    best_acc, best_epoch = 0.7, 0
    logger.info(f'args = {args}')
    start_time = time.time()
    epoch_time_m = AverageMeter()
    if mode!='to_pretrain':
        train_losses = []
        val_losses = []
        valid_acc = []
        confidence_avg=[]
        for epoch in range(args.start_epoch, args.epochs):
            loss_m_train = ModelTrainer.train_one_epoch_only_wtconv(train_dataloader, model, criterion, optimizer,
                                                   epoch, device, logger,args)

            if args.confidence:
                loss_m_valid, predict_label, true_label,confidence= ModelTrainer.validate(valid_dataloader, model, criterion,device, logger, args)
                confidence_avg.append(confidence)
            else:
                loss_m_valid, predict_label, true_label = ModelTrainer.validate_only_wtconv(valid_dataloader, model, criterion,
                                                                                device, logger, args)
            # scheduler.step(loss_m_valid)
            train_losses.append(loss_m_train.avg)
            #val_losses.append(loss_m_valid)

            #绘制混淆矩阵
            true_label_tensor = torch.from_numpy(true_label)
            all_preds_tensor = torch.from_numpy(predict_label)
            _, all_preds_tensor = torch.max(all_preds_tensor, 1)
            cm = confusion_matrix(true_label_tensor, all_preds_tensor)
            plot_confusion_matrix(cm, classes=lable_name, normalize=True,save_path=log_dir)
            metric = compute_metrics(predict_label,true_label, args,device)
            valid_acc.append(metric['accuracy'])
            #best_acc, best_epoch = max(best_acc, metric['accuracy']), epoch
            logger.info(
                'Epoch: [{:0>3}/{:0>3}]'
                'Time: {epoch_time.val:.3f} ({epoch_time.avg:.3f})  '
                'Metrics: Accuracy={metrics[accuracy]:>5.4f} Precision={metrics[precision]:>5.4f} '
                        'Recall={metrics[recall]:>5.4f} F1 Score={metrics[f1_score]:>5.4f}'
                'best_acc:{best_acc:.4f}'
                'Train Loss avg: {loss_train.avg:>6.4f} '.format(epoch, args.epochs, epoch_time=epoch_time_m,metrics=metric, best_acc=best_acc, loss_train=loss_m_valid,
                                  ))
            scheduler.step(metric['accuracy'])

            # ------------------------------------ 模型保存 ------------------------------------
            save = False
            if metric['accuracy'] > best_acc:
                best_acc = metric['accuracy']
                best_epoch = epoch
                save=True
            if epoch == args.epochs - 1 or epoch % save_epoch == 0 or save:
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "lr_scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch,
                    "args": args,
                    "best_acc": best_acc}
                pkl_name = "finetune_checkpoint_{}_batchsize_{}_acc_{}_dataname-{}.pth".format(epoch, args.batch_size, metric['accuracy'],dataset_name)
                path_checkpoint = os.path.join(log_dir, pkl_name)
                torch.save(checkpoint, path_checkpoint)

        plot_convergence(train_losses, valid_acc, args.epochs, save_path=log_dir)
        #plot_one(confidence_avg, "Confidence score",args.epochs, save_path=log_dir)
        logger.info('微调完成')
        loss_m_valid, predict_label, true_label = ModelTrainer.evaluate(test_dataloader, model, criterion,device, logger, args)
        visualize(predict_label, true_label, lable_name, log_dir, 'wtconv')
        test_metric = compute_metrics(predict_label, true_label, args, device)
        logger.info(
            'Metrics: Accuracy={metrics[accuracy]:>5.4f} Precision={metrics[precision]:>5.4f} '
            'Recall={metrics[recall]:>5.4f} F1 Score={metrics[f1_score]:>5.4f}'
            .format(metrics=test_metric))

    # else:
    #     loss_m_valid, predict_label,true_label , conf_mat = ModelTrainer.validate(valid_dataloader, model,criterion,device, logger, args)
    #     metric = compute_metrics(predict_label, true_label, args, device)
    #     logger.info(
    #         'Metrics: Accuracy={metrics[accuracy]:>5.4f} Precision={metrics[precision]:>5.4f} '
    #         'Recall={metrics[recall]:>5.4f} F1 Score={metrics[f1_score]:>5.4f}'.format(poch_time=epoch_time_m, metrics=metric))
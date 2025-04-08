import argparse
import os
import time


import torch.optim as optim

import torch
import torchvision.transforms as transforms

from data_process import create_dataset,create_dataloader

from utils import setup_seed, ModelTrainer, AverageMeter, make_logger, plot_confusion_matrix, visualize, \
    plot_convergence#, plot_train_loss

# from Wtconv_Bert import wtconv_bert
from Wtconv_mamba import wtconv_mamba


def count_parameters(models):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('模型的参数量是model', total_params,trainable_params)
    return total_params, trainable_params


if __name__ == '__main__':
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="Process some integers.")

    # 添加参数
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')#128.256
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')#30
    parser.add_argument('--model', type=str, default='wtconv_mamba', help='Model architecture to use')
    parser.add_argument('--data_path', type=str, default='./new_datasets/pre_train',help='Path to the dataset')
    #parser.add_argument('--pretrain_valid', type=bool, default=True, help='decide to pretrain')
    parser.add_argument("--print-freq", default=100, type=int, help="step to print frequency")
    parser.add_argument("--output-dir", default="./Pretrain_result", type=str, help="path to save outputs")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")\
    # 学习率不能给高，不然损失会崩
    parser.add_argument("--lr", default=5e-4, type=float, help="initial learning rate")
    parser.add_argument("--save_epoch", default=2, type=int)
    parser.add_argument("--wt_mask_ratio", default=0.5, type=float)
    parser.add_argument("--mamba_mask_ratio", default=0.9, type=float)
    parser.add_argument("--mode", default='to_pretrain', type=str,choices=['to_pretrain','to_finetune','just_evaluate'])
    parser.add_argument("--num_classes", default=20, type=int)
    parser.add_argument("--confidence", default=0, type=float)
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=3e-5,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    #parser.add_argument("--bert_dropout", default=0.01, type=float, help='dropout rate')
    parser.add_argument("--wt_dropout", default=0.1, type=float, help='dropout rate')
    parser.add_argument("--head_dropout", default=0.1, type=float, help='dropout rate')
    parser.add_argument("--wt_drop_path_rate", default=0.1, type=float, help='dropout rate')
    parser.add_argument("--pack_len", default=8, type=int)
    # 解析参数
    args = parser.parse_args()

    # 图片预处理参数
    norm_mean = 0.5  # RGB图像的均值
    norm_std = 0.5  # RGB图像的标准差
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    #基础参数
    setup_seed(33203)
    result_dir = args.output_dir
    save_epoch = args.save_epoch
    mode = args.mode
    wt_mask_ratio=args.wt_mask_ratio
    mamba_mask_ratio=args.mamba_mask_ratio
    dataset_name = args.data_path.split('/')[-1]

    pretrain_dataset= create_dataset(args, transform=transform, mode=mode)
    train_dataloader = create_dataloader(pretrain_dataset, batch_size=args.batch_size)

    model = wtconv_mamba(args,mode)

    print(model)
    print("总共的参数为",count_parameters(model))
    #print(torch.cuda.is_available())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 定义优化器和损失函数和调度器
    optimizer = optim.AdamW(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.weight_decay)
    criterion = torch.nn.MSELoss()
    criterion_valid = torch.nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    #打印日志
    # ------------------------------------  log ------------------------------------
    logger, log_dir = make_logger(result_dir)
    #writer = SummaryWriter(log_dir=log_dir)
    logger.info(f'args = {args}')
    start_time = time.time()
    epoch_time_m = AverageMeter()
    end = time.time()
    loss_m_train_list=[]
    for epoch in range(args.start_epoch, args.epochs):
        loss_m_train = ModelTrainer.train_one_epoch(train_dataloader, model, criterion, optimizer,
                                               epoch, device, logger,args,wt_mask_ratio=wt_mask_ratio,bert_mask_ratio=mamba_mask_ratio)
        loss_m_train_list.append(loss_m_train.avg)
        epoch_time_m.update(time.time() - end)
        logger.info(
            'Epoch: [{:0>3}/{:0>3}]  '
            'Time: {epoch_time.val:.3f} ({epoch_time.avg:.3f})  '
            'Train Loss avg: {loss_train.avg:>6.4f}  '
            'LR: {lr}'.format(epoch, args.epochs, epoch_time=epoch_time_m, loss_train=loss_m_train,lr=scheduler.get_last_lr()[0]))
        scheduler.step()


        # ------------------------------------ 模型保存 ------------------------------------
        if epoch == args.epochs - 1 or epoch % save_epoch == 0:
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch,
                "args": args}
            pkl_name = "wtconvmamba_checkpoint_{}_batchsize_{}_wt_maskratio_{}_bert_maskratio{}_dataset_{}.pth".format(epoch, args.batch_size, wt_mask_ratio,mamba_mask_ratio,dataset_name)
            path_checkpoint = os.path.join(log_dir, pkl_name)
            torch.save(checkpoint, path_checkpoint)
    logger.info('预训练完成')
    plot_convergence(loss_m_train_list, loss_m_train_list, args.epochs, save_path=log_dir)
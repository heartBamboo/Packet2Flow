import argparse
import os
import time
from functools import partial


import optuna

from sklearn.metrics import confusion_matrix


import torch.optim as optim

import torch
import torchvision.transforms as transforms

from Wtconv_Bert import wtconv_bert
from Wtconv_mamba import wtconv_mamba
from data_process import create_dataset,create_dataloader
from utils import setup_seed, ModelTrainer, AverageMeter, make_logger, plot_confusion_matrix, visualize, \
    compute_metrics, plot_convergence, plot_one


def optuna_objective(trial):
    parser = argparse.ArgumentParser(description="Process some integers.")
    # 定义参数空间
    config = {
        'batch_size': trial.suggest_categorical('batch_size', [32, 256]),
        #'confidence': trial.suggest_categorical('confidence', [0.85, 0.8]),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 5e-3),
        'weight_decay': trial.suggest_loguniform('weight_decay', 5e-5, 1e-3),
        'factor': trial.suggest_loguniform('factor', 0.1, 0.9),
        #'dropout_rate': trial.suggest_uniform('dropout_rate', 0.0, 0.1),
        #'bert_dropout': trial.suggest_loguniform('bert_dropout', 0.1, 0.5),
        #'wt_dropout': trial.suggest_loguniform('wt_drop_path_rate', 0.1, 0.5),
        #'head_dropout': trial.suggest_loguniform('head_dropout', 0.1, 0.5),
        #'wt_drop_path_rate': trial.suggest_loguniform('wt_drop_path_rate', 0.1, 0.3)
        # 模型微调所需用到的固定参数
    }
    # 添加参数
    parser.add_argument("--bert_dropout", default=0.001, type=float, help='dropout rate')
    parser.add_argument("--wt_dropout", default=0.001, type=float, help='dropout rate')
    parser.add_argument("--head_dropout", default=0.001, type=float, help='dropout rate')
    parser.add_argument("--wt_drop_path_rate", default=0.001, type=float, help='dropout rate')
    parser.add_argument("--pack_len", default=8, type=int)
    parser.add_argument("--learning_rate", default=config['learning_rate'], type=int)
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--model', type=str, default='wtconv_mamba', help='Model architecture to use')
    parser.add_argument('--data_path', type=str, default='./new_datasets/CSTNET_TLS_1.3_time_256_w8', help='Path to the dataset')
    parser.add_argument('--pretrain_valid', type=bool, default=False, help='decide to pretrain')
    parser.add_argument("--print-freq", default=25, type=int, help="step to print frequency")
    parser.add_argument("--output-dir", default="./optFinetune/vpn_mamba", type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--save_epoch", default=5, type=int)
    parser.add_argument("--num_classes", default=120, type=int)
    parser.add_argument("--pretrained_model_path",
                        default='./Pretrain_result/2025-03-08_01-09-23/wtconvmamba_checkpoint_20_batchsize_256_wt_maskratio_0.1_bert_maskratio0.9_dataset_pre_train.pth',
                        type=str)
    parser.add_argument("--finetuned_model_path",
                        default='./Pretrain_result/2025-03-08_01-09-23/wtconvmamba_checkpoint_20_batchsize_256_wt_maskratio_0.1_bert_maskratio0.9_dataset_pre_train.pth',
                        type=str)
    parser.add_argument("--mode", default='to_finetune', type=str,
                        choices=['to_pretrain', 'to_finetune', 'just_evaluate'])
    parser.add_argument("--confidence",default=0.9, type=float)
    parser.add_argument("--factor", default=config['factor'], type=float)


    # 解析参数
    args = parser.parse_args()
    if args.mode == 'to_finetune':
        pretrained_model_path = args.pretrained_model_path
    elif args.mode == 'just_evaluate':
        pretrained_model_path = args.finetuned_model_path


    # 基础参数
    setup_seed(33203)
    result_dir = args.output_dir
    save_epoch = args.save_epoch
    mode = args.mode
    dataset_name = args.data_path.split('/')[-1]


    # 图片预处理参数
    norm_mean = 0.5  # RGB图像的均值
    norm_std = 0.5  # RGB图像的标准差
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])
    train_dataset, valid_dataset, test_dataset, lable_name = create_dataset(args, transform=transform, mode=args.mode)[:4]
    train_dataloader = create_dataloader(train_dataset, batch_size=config['batch_size'])
    valid_dataloader = create_dataloader(valid_dataset, batch_size=config['batch_size'])
    test_dataloader = create_dataloader(test_dataset, batch_size=config['batch_size'])

    # 加载已经预训练好了的模型
    model = wtconv_mamba(args,mode=mode)
    print(model)
    pretrained_dict = torch.load(pretrained_model_path)['model_state_dict']
    model.load_state_dict(pretrained_dict, strict=False)
    print("learning_rate", args.learning_rate)
    print("参数加载完成")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # 打印日志
    # ------------------------------------  log ------------------------------------
    logger, log_dir = make_logger(result_dir)
    best_acc, best_epoch = 0.7, 0
    logger.info(f'args = {args}')
    start_time = time.time()
    epoch_time_m = AverageMeter()
    end = time.time()

    # 定义优化器和损失函数和调度器
    optimizer = optim.AdamW(model.parameters(),
                            lr=config['learning_rate'],
                            weight_decay=config['weight_decay'])
    criterion = torch.nn.CrossEntropyLoss()
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)#\
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], mode='max')

    if mode == 'to_finetune':
        train_losses = []
        val_losses = []
        valid_acc = []
        confidence_avg = []
        for epoch in range(args.start_epoch, args.epochs):
            loss_m_train = ModelTrainer.train_one_epoch(train_dataloader, model, criterion, optimizer,
                                                        epoch, device, logger, args)
            if args.confidence:
                loss_m_valid, predict_label, true_label, confidence = ModelTrainer.validate(valid_dataloader, model,
                                                                                            criterion, device, logger,
                                                                                            args)
                confidence_avg.append(confidence)
            else:
                loss_m_valid, predict_label, true_label = ModelTrainer.validate(valid_dataloader, model, criterion,
                                                                                device, logger, args)
            train_losses.append(loss_m_train.avg)
            # val_losses.append(loss_m_valid)

            # 绘制混淆矩阵
            true_label_tensor = torch.from_numpy(true_label)
            all_preds_tensor = torch.from_numpy(predict_label)
            _, all_preds_tensor = torch.max(all_preds_tensor, 1)
            cm = confusion_matrix(true_label_tensor, all_preds_tensor)
            plot_confusion_matrix(cm, classes=lable_name, normalize=True, save_path=log_dir)

            metric = compute_metrics(predict_label, true_label, args, device)
            valid_acc.append(metric['accuracy'])
            # best_acc, best_epoch = max(best_acc, metric['accuracy']), epoch
            logger.info(
                'Epoch: [{:0>3}/{:0>3}]  '
                'Time: {epoch_time.val:.3f} ({epoch_time.avg:.3f})  '
                'Metrics: Accuracy={metrics[accuracy]:>5.4f} Precision={metrics[precision]:>5.4f} '
                'Recall={metrics[recall]:>5.4f} F1 Score={metrics[f1_score]:>5.4f}'
                'best_acc:{best_acc:.4f}'
                'Train Loss avg: {loss_train.avg:>6.4f} '.format(epoch, args.epochs, epoch_time=epoch_time_m,
                                                                 metrics=metric, best_acc=best_acc,
                                                                 loss_train=loss_m_valid,
                                                                 ))
            scheduler.step(metric['accuracy'])
            # ------------------------------------ 模型保存 ------------------------------------
            save = False
            if metric['accuracy'] > best_acc:
                best_acc = metric['accuracy']
                best_epoch = epoch
                save = True
            if epoch == args.epochs - 1 or epoch % save_epoch == 0 or save:
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "lr_scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch,
                    "args": args,
                    "best_acc": best_acc}
                pkl_name = "finetune_checkpoint_{}_batchsize_{}_acc_{}_dataname-{}.pth".format(epoch,
                                                                                               config['batch_size'],
                                                                                               metric['accuracy'],
                                                                                               dataset_name)
                path_checkpoint = os.path.join(log_dir, pkl_name)
                torch.save(checkpoint, path_checkpoint)

        plot_convergence(train_losses, valid_acc, args.epochs, save_path=log_dir)
        plot_one(confidence_avg, "Confidence score", args.epochs, save_path=log_dir)
        logger.info('微调完成')
        loss_m_valid, predict_label, true_label = ModelTrainer.evaluate(test_dataloader, model, criterion, device,
                                                                        logger, args)
        test_metric = compute_metrics(predict_label, true_label, args, device)
        logger.info(
            'Metrics: Accuracy={metrics[accuracy]:>5.4f} Precision={metrics[precision]:>5.4f} '
            'Recall={metrics[recall]:>5.4f} F1 Score={metrics[f1_score]:>5.4f}'
            .format(metrics=test_metric))
        return test_metric['accuracy']


def optimizer_optuna(n_trials, algo):
    # 定义使用TPE或者GP
    if algo == "TPE":
        algo = optuna.samplers.TPESampler(n_startup_trials=15, n_ei_candidates=20)
    elif algo == "GP":
        from optuna.integration import SkoptSampler
        import skopt
        algo = SkoptSampler(skopt_kwargs={'base_estimator': 'GP',  # 选择高斯过程
                                          'n_initial_points': 30,  # 初始观测点10个
                                          'acq_func': 'EI'}  # 选择的采集函数为EI，期望增量
                            )

    # 实际优化过程，首先实例化优化器
    study = optuna.create_study(sampler=algo  # 要使用的具体算法
                                , direction="maximize"  # 优化的方向，可以填写minimize或maximize
                                )
    # 开始优化，n_trials为允许的最大迭代次数
    # 由于参数空间已经在目标函数中定义好，因此不需要输入参数空间
    partial_objective = partial(optuna_objective)
    study.optimize(partial_objective  # 目标函数
                   , n_trials=n_trials  # 最大迭代次数（包括最初的观测值的）
                   , show_progress_bar=True  # 要不要展示进度条呀？
                   )

    # 可直接从优化好的对象study中调用优化的结果
    # 打印最佳参数与最佳损失值
    print("\n", "\n", "best params: ", study.best_trial.params,
          "\n", "\n", "best score: ", study.best_trial.values,
          "\n")

    return study.best_trial.params, study.best_trial.values

def optimized_optuna_search_and_report(n_trials, algo):
    start_time = time.time()

    # 进行贝叶斯优化
    best_params, best_score = optimizer_optuna(n_trials, algo)

    # 打印最佳参数和分数
    print("\n","\n","best params: ", best_params,
          "\n","\n","best score: ", best_score,
          "\n")

    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60  # 转换为分钟
    print(f"Optimization completed in {elapsed_time:.2f}minutes.")

    return best_params, best_score

if __name__=="__main__":


    best_params, best_score=optimized_optuna_search_and_report(100, 'TPE')
    print(best_params,best_score)
